# Business Entity Resolution — Pipeline Scaffold

This is a working, smoke-tested scaffold (verified end-to-end on synthetic
data — see `tests/make_synthetic_data.py`), not a finished submission. It
runs on your real data with zero code changes if your files live under
`dataset/train/` and `dataset/test/` as named in the problem statement.

## Change history

- Token blocking now uses DuckDB relations and SQL joins instead of a Python
  `defaultdict(set)` inverted index.
- When embeddings are unavailable, raw S1/S2/S3 TSV files are scanned through
  DuckDB in bounded batches; only S2/S3 rows referenced by candidates are
  loaded back into pandas.
- Ranking and name-similarity features use per-S1 `rapidfuzz.process.cdist`
  batches rather than one Python scoring call per candidate pair.
- `src/benchmark.py` measures progressive blocking scale, token frequencies,
  candidate volume, peak Python allocation, and sampled blocking recall.

## What's already built and verified

- `src/normalize.py` — Unicode-generic normalization (NFKD diacritic
  stripping, works on French accents / any script without per-language
  code) + hand-written legal-suffix abbreviation map.
- `src/blocking.py` — DuckDB token blocking (fast, catches same-script noise)
  - optional multilingual-embedding ANN blocking (catches cross-script and
    the zero-shot France case).
- `src/embeddings.py` — wraps `sentence-transformers` +
  `faiss`; degrades gracefully (skips itself) if either isn't installed.
- `src/features.py` — string similarity, embedding cosine, address
  overlap, **country encoded only as a same/different boolean** (never a
  fixed category list — required so France works with zero code changes).
- `src/train_matcher.py` — LightGBM, split **by S1 entity** (not by pair,
  to avoid leakage), hard-negative-mined training set, class-imbalance
  weighting.
- `src/assign.py` — enforces the "each S2/S3 record belongs to at most one
  S1 entity" structural rule found in the training ground truth.
- `src/evaluate.py` — the exact macro-averaged per-entity F0.5 from the
  problem statement (singletons scored explicitly), used to pick the
  decision threshold.
- `src/infer.py` — writes both output files in the exact required format.
- `src/pipeline.py` — CLI: `python -m src.pipeline --stage {block,train,infer,all}`

## Setup

```bash
pip install -r requirements.txt
# Optional but recommended — needs one-time internet access to download
# weights, then runs offline. Skip if you're offline / time-constrained;
# the pipeline runs without it, just with weaker cross-script recall.
pip install sentence-transformers faiss-cpu
```

Point `src/config.py`'s `DATA_DIR` at your real dataset location (defaults
to `dataset/` matching the problem statement's layout).

## Run

```bash
python -m src.pipeline --stage train   # blocks + trains + picks threshold
python -m src.pipeline --stage infer   # scores test set, writes output/
python3 tests/validate_local.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-source1 dataset/test/test_source1.tsv \
    --test-source2 dataset/test/test_source2.tsv \
    --test-source3 dataset/test/test_source3.tsv
# THEN also run the official validator before every real submission:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Known gaps to close before this is submission-ready

- **Scale.** The raw token path is DuckDB-first and uses bounded source
  batches, but the candidate relation is still materialized as a pandas
  dataframe before ranking. Complete Arrow/streaming candidate ranking and
  feature storage are still required for the full 12.5M-record workload.
- **Measured scale.** At 1% with threshold 100, blocking produced 374K raw
  candidates, about 254MB Python peak allocation, and 13.4 seconds. At 5%,
  it produced 1.24M candidates, about 1.17GB peak allocation, and 139.8
  seconds. A 20% run did not complete before resource limits were reached.
- **Recall gate.** Token-only sampled recall was 2.19% at 5% with threshold 100. The optional embedding dependencies are not installed in the current
  environment, so the >=99% blocking-recall target has not been demonstrated.
- **Embedding index at scale.** `embeddings.py` uses a flat FAISS index
  (`IndexFlatIP`) — fine for tens of thousands of rows, too slow for
  millions. Swap to `IndexIVFFlat` (trained on a sample) once real data
  size is confirmed — noted in the code comment.
- **The one-S1-per-candidate assumption** in `assign.py` was observed on
  the training ground truth in one summary pass, not proven from the
  problem statement text itself. Re-verify directly against the real
  `train_ground_truth.tsv` before trusting it (a one-line groupby check —
  see the docstring in `assign.py` for exactly what to check).
- **No hyperparameter tuning yet** — `config.py`'s `LGBM_PARAMS` are
  reasonable defaults, not tuned on your real data.

Run the progressive benchmark before changing the production threshold:

```bash
python -m src.benchmark --fractions 0.01 0.05 0.20 --threshold 100
```

## Team division (4 people)

Suggested split so everyone can work in parallel from day 1 without
blocking on each other — hand off through `output/` files and the
`config.FEATURE_COLUMNS` contract, not through waiting on someone else's
code:

**Person A — Data + normalization + scale**

- Owns `src/normalize.py`, ingestion, and re-verifying every dataset fact
  we've assumed (missingness rates, the one-S1-per-candidate rule, script
  distribution) against the REAL files, not last session's numbers.
- Converts the blocking/feature loops from pandas `.iterrows()` to
  vectorized/DuckDB operations once real data is loaded — this is the
  #1 thing that will silently make Day 1 unusable if skipped.

**Person B — Blocking + embeddings**

- Owns `src/blocking.py` and `src/embeddings.py`.
- Gets the embedding+FAISS channel actually installed and running (needs
  the one-time internet access) — this is the piece that makes France
  work at all, so it's the highest-priority non-obvious task.
- Runs the blocking-recall gate check first thing: what % of true
  ground-truth matches survive blocking? Target ≥99% before anyone
  spends time on the matcher.

**Person C — Features + matcher**

- Owns `src/features.py` and `src/train_matcher.py`.
- Tunes `NEGATIVE_PER_POSITIVE`, `LGBM_PARAMS`, and hard-negative sampling
  once real blocked candidates exist.
- Adds any extra features that come up once real messy data is visible
  (the synthetic data can't surface every real noise pattern).

**Person D — Assignment, evaluation, submission, methodology doc**

- Owns `src/assign.py`, `src/evaluate.py`, `src/infer.py`.
- Owns threshold search and the abstention/singleton behavior
  specifically — this is worth disproportionate points under
  macro-averaging (123k singleton-type entities each count as much as any
  multi-match entity).
- Runs `tests/validate_local.py` AND the official validator before every
  leaderboard upload, and writes `Documentation_template.md` incrementally
  as the pipeline solidifies, not all at once on the last day.

All four: re-verify the country/script assumptions in this README against
the real data on Day 1 — everything here was designed from an earlier
summary of the dataset, and a design decision (like the country-agreement
boolean or the France-handling embedding channel) is only as good as the
fact it was based on.
