"""
Macro-averaged per-Source1-entity F0.5 — matches the challenge's stated
scoring exactly: computed per S1 entity (precision/recall over that
entity's predicted vs. true match set), then averaged across ALL S1
entities, singletons included. A correctly-predicted empty list scores
1.0; a wrongly-predicted non-empty list on a true singleton scores 0.0.

This must be used for every threshold decision in this pipeline — the
pipeline's optimum under this metric can differ meaningfully from what a
pooled/micro precision-recall calculation would suggest, precisely because
singletons and small-match entities count equally to large-match ones here.
"""

import numpy as np


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def per_entity_f05(predicted: set, truth: set) -> float:
    """The exact per-entity rule from the problem statement: an entity
    with an empty truth set scores 1.0 if predicted is also empty, else 0.0
    — this is NOT the same as precision=0/recall=undefined falling through
    to 0.0 by accident, it's an explicit rule worth keeping explicit here."""
    if len(truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0
    if len(predicted) == 0:
        return 0.0  # recall = 0 -> F0.5 = 0 regardless of precision formula edge cases
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_beta(precision, recall, beta=0.5)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """predictions, ground_truth: {source1_entity_id: set(matched_ids)}.
    Every S1 entity in ground_truth must have an entry in predictions
    (missing keys are treated as an empty prediction, matching how the
    real evaluator would score a missing/blank row)."""
    scores = [
        per_entity_f05(predictions.get(s1_id, set()), truth)
        for s1_id, truth in ground_truth.items()
    ]
    return float(np.mean(scores)) if scores else 0.0


def predictions_from_scored_pairs(scored_df, threshold: float) -> dict:
    """scored_df needs columns: source1_entity_id, candidate_entity_id,
    score (post-assignment, i.e. after enforcing the one-S1-per-S2/S3-record
    constraint in assign.py — do NOT call this on raw unassigned scores)."""
    predictions = {}
    kept = scored_df[scored_df["score"] >= threshold]
    for s1_id, group in kept.groupby("source1_entity_id"):
        predictions[s1_id] = set(group["candidate_entity_id"])
    return predictions


def search_best_threshold(scored_df, ground_truth: dict, all_s1_ids, thresholds=None) -> tuple:
    """Sweeps thresholds and returns (best_threshold, best_macro_f05).
    `all_s1_ids` must include S1 entities with NO surviving candidates at
    all, so singleton-correctness is scored properly at every threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    empty_gt = {sid: set() for sid in all_s1_ids}
    empty_gt.update(ground_truth)

    best_threshold, best_score = 0.5, -1.0
    for t in thresholds:
        preds = predictions_from_scored_pairs(scored_df, t)
        score = macro_f05(preds, empty_gt)
        if score > best_score:
            best_threshold, best_score = t, score
    return best_threshold, best_score
