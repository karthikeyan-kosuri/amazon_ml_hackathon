"""
Runs the trained matcher over the TEST candidate set and writes both
required output files.

Output-format rules this deliberately enforces (see problem statement):
  - Every Source1 test entity gets exactly one row, even with zero
    candidates/matches (empty matched_entity_ids field, not a missing row).
  - No duplicate entity IDs within an ID list, no duplicate
    source1_entity_id rows.
  - matching_results.tsv predictions are a strict subset of
    candidate_pairs.tsv for the same S1 entity (guaranteed here because
    matching_results is literally filtered from the same scored/assigned
    table candidate_pairs is written from).
"""

import lightgbm as lgb
import pandas as pd

from . import assign, config


def score_candidates(feature_df: pd.DataFrame, model_path) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    scored = feature_df.copy()
    scored["score"] = booster.predict(scored[config.FEATURE_COLUMNS])
    return scored


def write_id_list_tsv(predictions: dict, all_s1_ids: list, out_path, id_col_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1_id in all_s1_ids:
            ids = sorted(predictions.get(s1_id, set()))  # sorted -> deterministic, easy to diff
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def run_inference(
    test_feature_df: pd.DataFrame,
    all_test_s1_ids: list,
    model_path,
    threshold: float,
):
    """test_feature_df: the FULL candidate set (post blocking + ranking,
    pre-threshold) with features attached, for the test S1 entities.
    Returns (candidate_predictions, final_predictions) as
    {source1_entity_id: set(ids)} dicts — write these with
    write_id_list_tsv using config.CANDIDATE_PAIRS_OUT /
    config.MATCHING_RESULTS_OUT respectively.
    """
    scored = score_candidates(test_feature_df, model_path)

    # Candidate file = the exact set fed to the model for inference, i.e.
    # scored (pre-assignment, pre-threshold) — NOT an earlier raw blocking
    # pass. See problem statement: "the final candidate list just before
    # the ML model scores them."
    candidate_predictions = {
        s1_id: set(group["candidate_entity_id"])
        for s1_id, group in scored.groupby("source1_entity_id")
    }

    assigned = assign.enforce_one_s1_per_candidate(scored)
    kept = assigned[assigned["score"] >= threshold]
    final_predictions = {
        s1_id: set(group["candidate_entity_id"])
        for s1_id, group in kept.groupby("source1_entity_id")
    }

    return candidate_predictions, final_predictions


def write_submission(candidate_predictions: dict, final_predictions: dict, all_test_s1_ids: list):
    write_id_list_tsv(
        candidate_predictions, all_test_s1_ids, config.CANDIDATE_PAIRS_OUT, "candidate_entity_ids"
    )
    write_id_list_tsv(
        final_predictions, all_test_s1_ids, config.MATCHING_RESULTS_OUT, "matched_entity_ids"
    )
