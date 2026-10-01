"""
Global assignment step.

The training ground truth showed: every S2/S3 entity_id belongs to AT MOST
ONE Source1 entity (verified: 0 cases of a S2/S3 id matching multiple S1
rows). That's a hard structural fact about how this dataset was
constructed, not a heuristic — so before thresholding, for every candidate
S2/S3 record that the matcher scored against more than one S1 entity, we
keep only its single highest-scoring S1 assignment and drop the rest.

This runs BEFORE threshold cutoff (see evaluate.search_best_threshold),
because it's a structural constraint independent of the precision/recall
trade-off the threshold controls — dropping the losing duplicates first is
strictly correct regardless of what threshold ends up chosen.

IMPORTANT: this assumption should be re-verified on the actual training
ground truth file before trusting it in the submitted pipeline — it was
observed on one summary run, not proven from the problem statement itself.
If it turns out to be an approximation rather than a hard rule for some
edge case, this step should be relaxed (e.g. only dedup when the margin
between the top-two S1 scores for that candidate is large).
"""

import pandas as pd


def enforce_one_s1_per_candidate(scored_df: pd.DataFrame) -> pd.DataFrame:
    """scored_df needs columns: source1_entity_id, candidate_entity_id, score.
    Returns a filtered copy where each candidate_entity_id appears under at
    most one source1_entity_id (the highest-scoring one)."""
    idx_of_best = scored_df.groupby("candidate_entity_id")["score"].idxmax()
    return scored_df.loc[idx_of_best].reset_index(drop=True)
