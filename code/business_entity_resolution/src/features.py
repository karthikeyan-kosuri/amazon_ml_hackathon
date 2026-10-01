"""
Feature engineering for (S1, candidate) pairs.

Notable design choice: `country` is encoded ONLY as a same/different
boolean (`country_match`), never as a one-hot/categorical value. A
categorical encoding would have no representation for "France" at train
time and the model would have undefined behaviour on it; a boolean
same/different signal needs no vocabulary and works identically on any
country string, seen or unseen.
"""

# ============================================================
# CHANGE LOG
# ============================================================
#
# Previous approach:
# Each candidate pair used pandas `.loc` against two indexed dataframes.
#
# Problem:
# Repeated pandas indexing and Series construction added overhead for large
# candidate tables.
#
# New approach:
# Build row dictionaries once and use direct dictionary lookups per candidate.
#
# Reason:
# Preserve the existing feature calculations while reducing hot-loop overhead.
#
# Expected impact:
# Faster feature extraction and lower temporary pandas-object allocation.
#
# Validation performed:
# Batched feature values were compared with the scalar helper, empty-output
# schema was checked, the chunk iterator was added, and the module was compiled
# successfully.
#
# ============================================================

import numpy as np
import pandas as pd
import time
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import normalize
from . import config

try:
    from . import embeddings as emb_module
except ImportError:
    emb_module = None


def _char_trigram_jaccard(a: str, b: str) -> float:
    set_a, set_b = normalize.char_ngrams(a, 3), normalize.char_ngrams(b, 3)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _token_jaccard(tokens_a: list, tokens_b: list) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_pair_features(
    s1_row: pd.Series,
    other_row: pd.Series,
    embedding_cache: dict | None = None,
    name_scores: tuple[float, float, float] | None = None,
) -> dict:
    """embedding_cache, if provided, maps entity_id -> precomputed
    embedding vector (see build_embedding_cache). Avoids re-embedding the
    same name once per candidate pair."""
    name_a, name_b = s1_row["name_expanded"], other_row["name_expanded"]

    if name_scores is None:
        jw = JaroWinkler.normalized_similarity(name_a, name_b) if name_a and name_b else 0.0
        tsr = fuzz.token_sort_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0
        pr = fuzz.partial_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0
    else:
        jw, tsr, pr = name_scores
    trigram_jac = _char_trigram_jaccard(name_a, name_b)

    emb_cos, has_emb = 0.0, 0
    if embedding_cache is not None:
        vec_a = embedding_cache.get(s1_row["entity_id"])
        vec_b = embedding_cache.get(other_row["entity_id"])
        if vec_a is not None and vec_b is not None:
            emb_cos = float(np.dot(vec_a, vec_b))  # vectors are L2-normalized
            has_emb = 1

    addr_a_tokens = s1_row.get("address_tokens", [])
    addr_b_tokens = other_row.get("address_tokens", [])
    addr_a_present = bool(s1_row.get("address_present", False))
    addr_b_present = bool(other_row.get("address_present", False))
    addr_both_present = addr_a_present and addr_b_present
    addr_jac = _token_jaccard(addr_a_tokens, addr_b_tokens) if addr_both_present else 0.0

    country_match = int(s1_row.get("country") == other_row.get("country"))

    return {
        "name_jaro_winkler": jw,
        "name_token_sort_ratio": tsr,
        "name_partial_ratio": pr,
        "name_char_trigram_jaccard": trigram_jac,
        "name_embedding_cosine": emb_cos,
        "has_embedding_feature": has_emb,
        "name_len_diff": abs(len(name_a) - len(name_b)),
        "name_token_count_diff": abs(len(s1_row.get("name_tokens", [])) - len(other_row.get("name_tokens", []))),
        "address_token_jaccard": addr_jac,
        "address_present_both": int(addr_both_present),
        "country_match": country_match,
    }


def build_embedding_cache(df: pd.DataFrame) -> dict:
    """entity_id -> embedding vector, for every row in df. Returns {} if
    the embedding stack isn't installed (features.py then just leaves
    name_embedding_cosine at 0.0 with has_embedding_feature=0, and the
    model learns to weight that feature down for those rows)."""
    if emb_module is None or not emb_module.EMBEDDINGS_AVAILABLE:
        return {}
    texts = df["name_expanded"].tolist()
    vectors = emb_module.embed_texts(texts)
    if vectors is None:
        return {}
    return dict(zip(df["entity_id"], vectors))


def build_feature_table(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    embedding_cache: dict | None = None,
) -> pd.DataFrame:
    """candidates_df needs columns: source1_entity_id, candidate_entity_id.
    Returns candidates_df with config.FEATURE_COLUMNS appended."""
    required_s1_ids = candidates_df["source1_entity_id"].unique()
    required_other_ids = candidates_df["candidate_entity_id"].unique()
    s1_lookup = (
        s1_df[s1_df["entity_id"].isin(required_s1_ids)]
        .set_index("entity_id")
        .to_dict(orient="index")
    )
    other_lookup = (
        other_df[other_df["entity_id"].isin(required_other_ids)]
        .set_index("entity_id")
        .to_dict(orient="index")
    )

    def build_chunk(groups):
        feature_rows = []
        for s1_id, group in groups:
            s1_row = s1_lookup[s1_id]
            candidate_ids = group["candidate_entity_id"].tolist()
            other_rows = [other_lookup[candidate_id] for candidate_id in candidate_ids]
            other_names = [row["name_expanded"] for row in other_rows]
            s1_name = s1_row["name_expanded"]

            if s1_name:
                jw_scores = process.cdist(
                    [s1_name], other_names, scorer=JaroWinkler.normalized_similarity
                )[0]
                tsr_scores = process.cdist(
                    [s1_name], other_names, scorer=fuzz.token_sort_ratio
                )[0] / 100.0
                partial_scores = process.cdist(
                    [s1_name], other_names, scorer=fuzz.partial_ratio
                )[0] / 100.0
            else:
                jw_scores = [0.0] * len(other_rows)
                tsr_scores = [0.0] * len(other_rows)
                partial_scores = [0.0] * len(other_rows)

            for position, other_row in enumerate(other_rows):
                feature_rows.append(
                    compute_pair_features(
                        s1_row,
                        other_row,
                        embedding_cache,
                        name_scores=(
                            jw_scores[position],
                            tsr_scores[position],
                            partial_scores[position],
                        ),
                    )
                )

        feature_df = pd.DataFrame(feature_rows, columns=config.FEATURE_COLUMNS)
        return pd.concat([pd.concat([group for _, group in groups], ignore_index=True), feature_df], axis=1)

    output_chunks = []
    group_chunk = []
    processed_groups = 0
    next_progress = 100_000
    started = time.perf_counter()
    for group in candidates_df.groupby("source1_entity_id", sort=False):
        group_chunk.append(group)
        if len(group_chunk) >= 150_000:
            output_chunks.append(build_chunk(group_chunk))
            processed_groups += len(group_chunk)
            group_chunk = []
            if processed_groups >= next_progress:
                print(
                    f"[features progress] {processed_groups} groups in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                next_progress += 100_000

    if group_chunk:
        output_chunks.append(build_chunk(group_chunk))
        processed_groups += len(group_chunk)
        if processed_groups >= next_progress:
            print(
                f"[features progress] {processed_groups} groups in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )

    if not output_chunks:
        return pd.concat(
            [candidates_df.reset_index(drop=True), pd.DataFrame(columns=config.FEATURE_COLUMNS)],
            axis=1,
        )
    return pd.concat(output_chunks, ignore_index=True)


def iter_feature_tables(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    embedding_cache: dict | None = None,
    chunk_size: int = 100_000,
):
    """Yield feature tables in bounded candidate chunks.

    The existing ``build_feature_table`` API remains available for callers
    that need one dataframe. This iterator lets training, diagnostics, or
    inference integrations process candidate batches without constructing a
    feature dataframe for every candidate at once.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(candidates_df), chunk_size):
        yield build_feature_table(
            candidates_df.iloc[start:start + chunk_size],
            s1_df,
            other_df,
            embedding_cache,
        )
