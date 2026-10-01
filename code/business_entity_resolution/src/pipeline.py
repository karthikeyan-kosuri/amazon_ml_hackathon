"""
End-to-end CLI. Run stages individually while developing (each person on
the team can run just their stage), or `--stage all` for the full thing.

Usage:
    python -m src.pipeline --stage block
    python -m src.pipeline --stage train
    python -m src.pipeline --stage infer
    python -m src.pipeline --stage all
"""

# ============================================================
# CHANGE LOG
# ============================================================
#
# Previous approach:
# `stage_block()` built a combined Source2/Source3 dataframe, then discarded
# it; train and inference concatenated the same large sources again.
#
# Problem:
# Repeated concatenation temporarily duplicated multi-million-row dataframes.
#
# New approach:
# Internal train/inference callers can request the already-built combined
# dataframe through `stage_block(include_other=True)`.
#
# Reason:
# Reuse the existing dataframe without changing the default stage contract.
#
# Expected impact:
# Lower peak memory during train and inference setup.
#
# Validation performed:
# Default and opt-in stage return contracts were verified with a focused smoke
# test, DuckDB raw blocking integration was added, and the module was compiled
# successfully. Training Source2/Source3 loads now support the requested
# bounded 300,000-row run.
#
# ============================================================

import argparse
import sys
import time

import pandas as pd

from . import blocking, config, evaluate, features, infer, train_matcher


def load_source(path, nrows=None) -> pd.DataFrame:
    started = time.perf_counter()
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": str},
        keep_default_na=True,
        nrows=nrows,
    )
    print(
        f"[load] {path}: {len(df)} rows in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return df


def load_training_target_source(path, required_ids, row_cap=300000) -> pd.DataFrame:
    source_df = load_source(path)
    required = source_df[source_df["entity_id"].isin(required_ids)]
    remaining = source_df[~source_df["entity_id"].isin(required_ids)]
    extra_count = max(0, row_cap - len(required))
    if extra_count and len(remaining):
        extra = remaining.sample(
            n=min(extra_count, len(remaining)),
            random_state=config.RANDOM_SEED,
        )
        return pd.concat([required, extra], ignore_index=True)
    return required.reset_index(drop=True)


def stage_block(train_or_test: str, include_other: bool = False):
    if train_or_test == "train":
        s1, s2, s3 = config.TRAIN_SOURCE1, config.TRAIN_SOURCE2, config.TRAIN_SOURCE3
    else:
        s1, s2, s3 = config.TEST_SOURCE1, config.TEST_SOURCE2, config.TEST_SOURCE3

    s1_df = load_source(s1, nrows=None)
    if train_or_test == "train":
        ground_truth = train_matcher.load_ground_truth(config.TRAIN_GROUND_TRUTH)
        required_target_ids = set()
        for source1_id in s1_df["entity_id"]:
            required_target_ids.update(ground_truth.get(source1_id, set()))
        s2_df = load_training_target_source(s2, required_target_ids)
        s3_df = load_training_target_source(s3, required_target_ids)
        if config.DUCKDB_RAW_BLOCKING:
            raw_candidates = blocking.generate_token_candidates_from_paths(
                s1_df, s2_df, s3_df
            )
            blocking.add_normalized_columns(s1_df)
            blocking.add_normalized_columns(s2_df)
            blocking.add_normalized_columns(s3_df)
        else:
            raw_candidates = blocking.generate_candidates(s1_df, s2_df, s3_df)
    else:
        use_raw_duckdb = (
            config.DUCKDB_RAW_BLOCKING
            and (blocking.emb_module is None or not blocking.emb_module.EMBEDDINGS_AVAILABLE)
        )
        if use_raw_duckdb:
            raw_candidates = blocking.generate_token_candidates_from_paths(s1, s2, s3)
            s2_df, s3_df = blocking.load_candidate_source_rows(s2, s3, raw_candidates)
            blocking.add_normalized_columns(s1_df)
            blocking.add_normalized_columns(s2_df)
            blocking.add_normalized_columns(s3_df)
        else:
            s2_df, s3_df = load_source(s2), load_source(s3)
            raw_candidates = blocking.generate_candidates(s1_df, s2_df, s3_df)

    other_df = pd.concat([s2_df, s3_df], ignore_index=True)
    ranking_started = time.perf_counter()
    capped = blocking.rank_and_cap_candidates(raw_candidates, s1_df, other_df)
    print(
        f"[ranking] {len(capped)} candidates in "
        f"{time.perf_counter() - ranking_started:.2f}s",
        flush=True,
    )

    print(f"[{train_or_test}] raw candidates: {len(raw_candidates)}, capped: {len(capped)}")
    print(f"[{train_or_test}] S1 entities with >=1 candidate: {capped['source1_entity_id'].nunique()} / {len(s1_df)}")
    if include_other:
        return s1_df, s2_df, s3_df, capped, other_df
    return s1_df, s2_df, s3_df, capped


def stage_features(s1_df, other_df, candidates_df, use_embeddings: bool = True):
    started = time.perf_counter()
    embedding_cache = {}
    if use_embeddings:
        embedding_cache.update(features.build_embedding_cache(s1_df))
        embedding_cache.update(features.build_embedding_cache(other_df))
    feature_df = features.build_feature_table(
        candidates_df, s1_df, other_df, embedding_cache or None
    )
    print(
        f"[features] {len(feature_df)} rows in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return feature_df


def stage_train():
    s1_df, _s2_df, _s3_df, candidates_df, other_df = stage_block(
        "train", include_other=True
    )
    del _s2_df, _s3_df
    feature_df = stage_features(
        s1_df, other_df, candidates_df, use_embeddings=config.USE_EMBEDDINGS
    )

    ground_truth = train_matcher.load_ground_truth(config.TRAIN_GROUND_TRUTH)
    labeled_df = train_matcher.label_candidates(feature_df, ground_truth)

    print(f"Labeled candidates: {len(labeled_df)} "
          f"({labeled_df['label'].sum()} positive, {(labeled_df['label']==0).sum()} negative)")

    booster, val_df = train_matcher.train(labeled_df)
    train_matcher.save_model(booster, config.LIGHTGBM_MODEL_PATH)

    val_s1_ids = val_df["source1_entity_id"].unique().tolist()
    val_ground_truth = {sid: ids for sid, ids in ground_truth.items() if sid in val_s1_ids}

    from . import assign
    assigned_val = assign.enforce_one_s1_per_candidate(val_df.rename(columns={"score": "score"}))
    best_threshold, best_f05 = evaluate.search_best_threshold(assigned_val, val_ground_truth, val_s1_ids)
    train_matcher.save_threshold(best_threshold, config.THRESHOLD_PATH)

    print(f"Validation macro F0.5: {best_f05:.4f} at threshold {best_threshold:.2f}")
    return booster, best_threshold


def stage_infer():
    import json
    with open(config.THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    s1_df, _s2_df, _s3_df, candidates_df, other_df = stage_block(
        "test", include_other=True
    )
    del _s2_df, _s3_df
    feature_df = stage_features(
        s1_df, other_df, candidates_df, use_embeddings=config.USE_EMBEDDINGS
    )

    candidate_preds, final_preds = infer.run_inference(
        feature_df, s1_df["entity_id"].tolist(), config.LIGHTGBM_MODEL_PATH, threshold
    )
    infer.write_submission(candidate_preds, final_preds, s1_df["entity_id"].tolist())
    print(f"Wrote {config.CANDIDATE_PAIRS_OUT} and {config.MATCHING_RESULTS_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["block", "train", "infer", "all"], required=True)
    args = parser.parse_args()

    if args.stage == "block":
        stage_block("train")
    elif args.stage == "train":
        stage_train()
    elif args.stage == "infer":
        stage_infer()
    elif args.stage == "all":
        stage_train()
        stage_infer()


if __name__ == "__main__":
    sys.exit(main())
