"""
Progressive scale benchmark for the entity-resolution pipeline.

Runs blocking on reproducible sampled subsets and reports wall-clock time,
Python peak allocation, candidate volume, and token-frequency diagnostics.
This is a diagnostic tool; it does not write submission files or alter the
training and inference pipeline.
"""

# ============================================================
# CHANGE LOG
# ============================================================
#
# Previous approach:
# The full block stage was started without staged measurements and stopped
# before runtime, memory, or candidate-volume data was collected.
#
# Problem:
# A full-scale run could not identify which stage caused slowdowns or memory
# pressure.
#
# New approach:
# Reproducible 1%, 5%, and 20% sampled blocking benchmarks with timing,
# Python allocation tracking, candidate-volume metrics, and DuckDB token counts.
#
# Reason:
# Progressive measurements reveal nonlinear scaling before a full run.
#
# Expected impact:
# Provides evidence for threshold tuning and identifies the next bottleneck.
#
# Validation performed:
# Module compilation and a small sampled benchmark are required before using
# results to change production configuration.
#
# ============================================================

import argparse
import time
import tracemalloc
from pathlib import Path

import duckdb
import pandas as pd

from . import blocking, config


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def sample_source(path: Path, fraction: float, seed: int) -> pd.DataFrame:
    """Read a deterministic sample through DuckDB, returning only a sample."""
    connection = duckdb.connect()
    try:
        connection.execute("SET threads TO 1")
        return connection.execute(
            """
            SELECT *
            FROM read_csv(?, sep='\\t', header=true, all_varchar=true)
            WHERE MOD(ABS(HASH(entity_id) + ?), 10000) < ?
            """,
            [str(path), seed, max(1, min(10000, int(fraction * 10000)))],
        ).fetchdf()[SOURCE_COLUMNS]
    finally:
        connection.close()


def token_frequency_diagnostics(paths: dict[str, Path], fraction: float, seed: int) -> pd.DataFrame:
    """Report the most frequent normalized blocking tokens on a sample."""
    frames = []
    for source, path in paths.items():
        frame = sample_source(path, fraction, seed)
        blocking.add_normalized_columns(frame)
        rows = []
        for country, tokens in zip(frame["country"], frame["name_tokens"]):
            for token in blocking.normalize.blocking_tokens(
                tokens, config.MIN_TOKEN_LENGTH_FOR_BLOCKING
            ):
                rows.append((country, token))
        frames.append(pd.DataFrame(rows, columns=["country", "token"]))

    if not frames:
        return pd.DataFrame(columns=["country", "token", "frequency"])
    token_frame = pd.concat(frames, ignore_index=True)
    return (
        token_frame.groupby(["country", "token"], as_index=False)
        .size()
        .rename(columns={"size": "frequency"})
        .sort_values("frequency", ascending=False)
        .head(20)
    )


def sampled_blocking_recall(
    candidates: pd.DataFrame,
    sampled_s1: pd.DataFrame,
    ground_truth_path: Path,
) -> tuple[float, int, int]:
    """Return recall over positive links for the sampled Source1 entities."""
    connection = duckdb.connect()
    try:
        connection.register("sampled_s1", sampled_s1[["entity_id"]])
        truth = connection.execute(
            """
            SELECT gt.source1_entity_id, matched_entity_id
            FROM read_csv(?, sep='\\t', header=true, all_varchar=true) AS gt
            JOIN sampled_s1 AS s1
              ON gt.source1_entity_id = s1.entity_id
            CROSS JOIN UNNEST(string_split(gt.matched_entity_ids, ',')) AS ids(matched_entity_id)
            WHERE gt.matched_entity_ids IS NOT NULL
              AND gt.matched_entity_ids <> ''
            """,
            [str(ground_truth_path)],
        ).fetchdf()
    finally:
        connection.close()

    if truth.empty:
        return 1.0, 0, 0
    expected = set(zip(truth["source1_entity_id"], truth["matched_entity_id"]))
    actual = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    survived = len(expected & actual)
    return survived / len(expected), survived, len(expected)


def run_benchmark(fraction: float, seed: int = 42) -> dict:
    """Run one sampled blocking benchmark and return scalar metrics."""
    paths = {
        "S1": config.TRAIN_SOURCE1,
        "S2": config.TRAIN_SOURCE2,
        "S3": config.TRAIN_SOURCE3,
    }
    s1 = sample_source(paths["S1"], fraction, seed)
    s2 = sample_source(paths["S2"], fraction, seed)
    s3 = sample_source(paths["S3"], fraction, seed)

    tracemalloc.start()
    started = time.perf_counter()
    candidates = blocking.generate_candidates(s1, s2, s3)
    recall, survived_links, total_links = sampled_blocking_recall(
        candidates, s1, config.TRAIN_GROUND_TRUTH
    )
    elapsed = time.perf_counter() - started
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    per_entity = candidates.groupby("source1_entity_id").size()
    return {
        "fraction": fraction,
        "s1_rows": len(s1),
        "s2_rows": len(s2),
        "s3_rows": len(s3),
        "raw_candidates": len(candidates),
        "entities_with_candidates": len(per_entity),
        "average_candidates": float(per_entity.mean()) if len(per_entity) else 0.0,
        "maximum_candidates": int(per_entity.max()) if len(per_entity) else 0,
        "blocking_recall": recall,
        "survived_true_links": survived_links,
        "sampled_true_links": total_links,
        "runtime_seconds": elapsed,
        "python_peak_megabytes": peak_bytes / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.20],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-fraction", type=float, default=0.01)
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Temporarily override TOKEN_FREQUENCY_THRESHOLD for this benchmark.",
    )
    args = parser.parse_args()

    if args.threshold is not None:
        config.TOKEN_FREQUENCY_THRESHOLD = args.threshold

    for fraction in args.fractions:
        metrics = run_benchmark(fraction, args.seed)
        print(metrics)

    diagnostics = token_frequency_diagnostics(
        {"S1": config.TRAIN_SOURCE1, "S2": config.TRAIN_SOURCE2, "S3": config.TRAIN_SOURCE3},
        args.token_fraction,
        args.seed,
    )
    print("top sampled blocking tokens:")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
