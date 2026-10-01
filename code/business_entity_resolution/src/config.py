"""
Central configuration for the pipeline. Nothing here should hard-code a
specific country value — `country` must stay an open string set so the
pipeline works unmodified on the France-only-in-test slice.
"""

# ============================================================
# CHANGE LOG
# ============================================================
#
# Previous approach:
# Token blocking used an in-memory Python inverted index.
#
# Problem:
# High-cardinality token indexes create excessive Python object overhead.
#
# New approach:
# Token frequency filtering is configurable for DuckDB-backed blocking.
#
# Reason:
# Extremely common tokens can create candidate explosions at full scale.
#
# Expected impact:
# Fewer SQL join rows and lower memory pressure during candidate generation.
#
# Validation performed:
# Threshold behavior was verified with a focused blocking smoke test.
#
# ============================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust these three if your data lives somewhere else. Everything
# else in the pipeline reads from here, so this is the only file most people
# need to touch to point it at their own copy of the dataset.
# ---------------------------------------------------------------------------
DATA_DIR = Path("../../dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

CANDIDATE_PAIRS_OUT = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_OUT = OUTPUT_DIR / "matching_results.tsv"

LIGHTGBM_MODEL_PATH = MODEL_DIR / "matcher_lgbm.txt"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------
# Max candidates kept per S1 entity after blocking+ranking, before features
# are computed. Observed max true matches per S1 in the labeled training
# data was 11 — this gives generous headroom without blowing up compute.
MAX_CANDIDATES_PER_ENTITY = 40

# Token-blocking: names are split into normalized tokens; a S2/S3 record is
# a token-blocking candidate for a S1 entity if they share >=1 non-stopword
# token. Legal-suffix tokens (see normalize.py LEGAL_SUFFIX_TOKENS) are
# excluded from the token index because they're too common to be useful
# blocking keys and would blow up candidate lists.
MIN_TOKEN_LENGTH_FOR_BLOCKING = 3

# Token keys with a higher per-country frequency are excluded from blocking.
# This is intentionally configurable because the useful cutoff depends on the
# source distribution and available memory at the target dataset scale.
TOKEN_FREQUENCY_THRESHOLD = 150

# DuckDB-first token blocking is used when the optional embedding channel is
# unavailable. The embedding path keeps the existing pandas behavior until a
# disk-backed embedding store is benchmarked.
DUCKDB_RAW_BLOCKING = True
BLOCKING_BATCH_SIZE = 100_000

# Bound the SQL candidate relation before it crosses into pandas. Candidates
# with more shared blocking tokens are retained first; the existing RapidFuzz
# ranker still performs the final MAX_CANDIDATES_PER_ENTITY selection.
SQL_PRE_RANK_CANDIDATES_PER_ENTITY = 100

# Embedding-based blocking (optional channel — see embeddings.py). Skipped
# automatically if sentence-transformers/faiss are not installed.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_TOP_K = 20
USE_EMBEDDINGS = False

# ---------------------------------------------------------------------------
# Matcher / training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.15  # fraction of S1 entities (not pairs!) held out
NEGATIVE_PER_POSITIVE = 8   # ratio of sampled negatives to positives for training

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "seed": RANDOM_SEED,
    "verbose": -1,
}
LGBM_NUM_BOOST_ROUND = 500
LGBM_EARLY_STOPPING_ROUNDS = 30

# ---------------------------------------------------------------------------
# Feature columns — single source of truth so train_matcher.py and infer.py
# can never silently drift apart on what the model expects.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_char_trigram_jaccard",
    "name_embedding_cosine",       # 0.0 if embeddings unavailable/skipped
    "has_embedding_feature",       # 1/0 flag so the model can learn to trust it or not
    "name_len_diff",
    "name_token_count_diff",
    "address_token_jaccard",
    "address_present_both",
    "country_match",
]
