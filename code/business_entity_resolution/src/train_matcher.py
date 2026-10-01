"""
Trains the LightGBM matcher.

Two things this file is careful about, because getting either wrong
quietly inflates validation scores without helping the leaderboard:

1. Train/validation split is done by SOURCE1 ENTITY, not by pair. If we
   split by pair, the same S1 entity's other positive pairs could leak
   into both train and val, making validation look better than reality.

2. Negatives are sampled from the *blocked* candidate set (pairs that
   survived blocking but aren't in ground truth) — not randomly from the
   whole dataset — because random pairs are almost all trivially
   dissimilar and teach the model nothing about the actually-hard
   near-miss cases (e.g. "Apollo Hospitals" vs "Apollo Pharmacy") it will
   face at inference time.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config


def load_ground_truth(path) -> dict:
    """Returns {source1_entity_id: set(matched_entity_ids)}. Rows with an
    empty matched_entity_ids field are singletons — kept as an empty set,
    not dropped, since correctly predicting "no match" for them is exactly
    what the abstention logic (infer.py) needs to be evaluated against."""
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = {}
    for row in gt.itertuples():
        ids = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        result[row.source1_entity_id] = ids
    return result


def label_candidates(feature_df: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["label"] = feature_df.apply(
        lambda r: int(r.candidate_entity_id in ground_truth.get(r.source1_entity_id, set())),
        axis=1,
    )
    return feature_df


def split_by_entity(entity_ids: list, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    unique_entities = np.array(sorted(set(entity_ids)))
    rng.shuffle(unique_entities)
    cutoff = int(len(unique_entities) * (1 - val_fraction))
    return set(unique_entities[:cutoff]), set(unique_entities[cutoff:])


def sample_negatives(labeled_df: pd.DataFrame, ratio: int, seed: int) -> pd.DataFrame:
    """Keeps all positives; downsamples negatives to `ratio`x the positive
    count, biased toward the highest-scoring (hardest) negatives by
    name_token_sort_ratio so the model sees real near-misses, not just
    easy ones it would get right anyway."""
    pos = labeled_df[labeled_df["label"] == 1]
    neg = labeled_df[labeled_df["label"] == 0]

    n_keep = min(len(neg), len(pos) * ratio)
    if n_keep <= 0 or len(neg) == 0:
        return pos

    hard_half = neg.sort_values("name_token_sort_ratio", ascending=False).head(n_keep // 2)
    remaining = neg.drop(hard_half.index)
    random_half = remaining.sample(
        n=min(n_keep - len(hard_half), len(remaining)), random_state=seed
    ) if len(remaining) else remaining

    return pd.concat([pos, hard_half, random_half], ignore_index=True)


def train(labeled_feature_df: pd.DataFrame):
    """labeled_feature_df must have config.FEATURE_COLUMNS + 'label' +
    'source1_entity_id'. Returns (booster, val_df_with_predictions)."""
    train_ids, val_ids = split_by_entity(
        labeled_feature_df["source1_entity_id"].tolist(),
        config.VALIDATION_FRACTION,
        config.RANDOM_SEED,
    )

    train_df = labeled_feature_df[labeled_feature_df["source1_entity_id"].isin(train_ids)]
    val_df = labeled_feature_df[labeled_feature_df["source1_entity_id"].isin(val_ids)]

    train_df = sample_negatives(train_df, config.NEGATIVE_PER_POSITIVE, config.RANDOM_SEED)

    n_pos, n_neg = (train_df["label"] == 1).sum(), (train_df["label"] == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    params = dict(config.LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    train_set = lgb.Dataset(train_df[config.FEATURE_COLUMNS], label=train_df["label"])
    val_set = lgb.Dataset(val_df[config.FEATURE_COLUMNS], label=val_df["label"], reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    val_df = val_df.copy()
    val_df["score"] = booster.predict(val_df[config.FEATURE_COLUMNS])
    return booster, val_df


def save_model(booster, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def save_threshold(threshold: float, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"threshold": threshold}, f)
