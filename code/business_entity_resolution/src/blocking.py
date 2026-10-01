"""
Candidate generation (blocking).

Two independent channels, merged:
  1. Token-inverted-index blocking — fast, catches same-script/abbreviation
     noise (Corp/Corporation, word-order swaps, typos that still share a
     token).
  2. Embedding ANN blocking (optional, skipped if sentence-transformers /
     faiss aren't installed) — catches cross-script and zero-shot-country
     (France) cases that share no tokens or n-grams at all.

`country` is treated as an open string set throughout: grouping is done by
whatever value appears in the data (`df.groupby("country")`), never by a
fixed {US, India} list, so a country with zero training rows (France) is
handled by exactly the same code path.

Output: a DataFrame with columns
    source1_entity_id, candidate_entity_id, source ('S2' or 'S3'),
    block_channel ('token', 'embedding', or 'both')
This is the *pre-ranking* candidate set. `rank_and_cap_candidates` below
narrows it to config.MAX_CANDIDATES_PER_ENTITY per S1 entity — the result
of that step is what should be written out as candidate_pairs.tsv, since
the spec wants "the last filtering stage before the matching model runs
inference," not this raw union.
"""

# ============================================================
# CHANGE LOG
# ============================================================
#
# Previous approach:
# Python `defaultdict(set)` inverted index, rebuilt for each country.
#
# Problem:
# Millions of Python dictionary and set objects caused excessive RAM usage
# and candidate explosions on the full challenge dataset.
#
# New approach:
# DuckDB token relations, per-country token-frequency filtering, and SQL joins.
#
# Reason:
# Let the database execute token expansion, aggregation, filtering, and
# deduplication without materializing the inverted index in Python.
#
# Expected impact:
# Lower Python object overhead and fewer candidates reaching pandas and the
# downstream ranking and feature stages.
#
# Validation performed:
# DuckDB blocking smoke tests, frequency-filter tests, batched ranker
# equivalence tests, raw-source sampling benchmarks, and Python compilation.
#
# ============================================================

import duckdb
import pandas as pd
import time
from rapidfuzz import fuzz, process

from . import config, normalize

try:
    from . import embeddings as emb_module
except ImportError:
    emb_module = None


def token_channel_for_country(
    s1_group: pd.DataFrame,
    other_group: pd.DataFrame,
    source_label: str,
) -> pd.DataFrame:
    """Generate token candidates with a DuckDB join.

    ``name_tokens`` is a normalized list column. DuckDB expands that list in
    SQL, filters legal/short tokens, removes high-frequency country/token
    keys, and only then materializes the filtered candidate relation as pandas.
    """
    del source_label
    columns = ["source1_entity_id", "candidate_entity_id", "source"]
    if s1_group.empty or other_group.empty:
        return pd.DataFrame(columns=columns)

    connection = duckdb.connect()
    try:
        connection.register("s1_rows", s1_group[["entity_id", "country", "name_tokens"]])
        connection.register(
            "other_rows",
            other_group[["entity_id", "country", "name_tokens", "source"]],
        )

        legal_suffixes = sorted(normalize.LEGAL_SUFFIX_TOKENS)
        placeholders = ", ".join("?" for _ in legal_suffixes)
        query = f"""
            WITH tokens AS (
                SELECT
                    CAST(entity_id AS VARCHAR) AS entity_id,
                    'S1' AS source,
                    country,
                    token
                FROM s1_rows
                CROSS JOIN UNNEST(name_tokens) AS expanded(token)
                WHERE length(token) >= ?
                  AND token NOT IN ({placeholders})

                UNION ALL

                SELECT
                    CAST(entity_id AS VARCHAR) AS entity_id,
                    source,
                    country,
                    token
                FROM other_rows
                CROSS JOIN UNNEST(name_tokens) AS expanded(token)
                WHERE length(token) >= ?
                  AND token NOT IN ({placeholders})
            ),
            usable_tokens AS (
                SELECT country, token
                FROM (
                    SELECT DISTINCT entity_id, source, country, token
                    FROM tokens
                ) AS distinct_tokens
                GROUP BY country, token
                HAVING COUNT(*) <= ?
            )
            SELECT DISTINCT
                s1.entity_id AS source1_entity_id,
                other.entity_id AS candidate_entity_id,
                other.source
            FROM tokens AS s1
            JOIN tokens AS other
                ON s1.token = other.token
               AND s1.country = other.country
            JOIN usable_tokens
                ON usable_tokens.country = s1.country
               AND usable_tokens.token = s1.token
            WHERE s1.source = 'S1'
              AND other.source IN ('S2', 'S3')
        """
        parameters = (
            [config.MIN_TOKEN_LENGTH_FOR_BLOCKING]
            + legal_suffixes
            + [config.MIN_TOKEN_LENGTH_FOR_BLOCKING]
            + legal_suffixes
            + [config.TOKEN_FREQUENCY_THRESHOLD]
        )
        return connection.execute(query, parameters).fetchdf()
    finally:
        connection.close()


def generate_token_candidates_from_paths(
    s1_path,
    s2_path,
    s3_path,
    batch_size: int = config.BLOCKING_BATCH_SIZE,
) -> pd.DataFrame:
    """Generate token candidates from raw TSV files without full S2/S3 pandas loads.

    Raw records are scanned by DuckDB. Each bounded Arrow batch is normalized
    with the existing normalization functions and inserted into a temporary
    relational token table; the Python token-list column exists only for that
    batch and is released before the next batch is read.
    """
    connection = duckdb.connect()
    try:
        table_started = time.perf_counter()
        connection.execute(
            """
            CREATE TEMPORARY TABLE tokens (
                entity_id VARCHAR,
                source VARCHAR,
                country VARCHAR,
                token VARCHAR
            )
            """
        )
        print(
            f"[duckdb] token table created in "
            f"{time.perf_counter() - table_started:.2f}s",
            flush=True,
        )
        for source, source_data in (("S1", s1_path), ("S2", s2_path), ("S3", s3_path)):
            reader = connection.cursor()
            if isinstance(source_data, pd.DataFrame):
                relation = f"{source.lower()}_capped"
                connection.register(relation, source_data)
                reader.register(relation, source_data)
                reader.execute(
                    f"SELECT entity_id, business_name, country FROM {relation}"
                )
            else:
                reader.execute(
                    """
                    SELECT entity_id, business_name, country
                    FROM read_csv(?, sep='\\t', header=true, all_varchar=true)
                    """,
                    [str(source_data)],
                )
            while True:
                batch = reader.fetch_df_chunk(batch_size)
                if batch.empty:
                    break
                name_norm = batch["business_name"].fillna("").map(normalize.normalize_name)
                batch["name_tokens"] = name_norm.map(lambda value: value["tokens"])
                connection.register("normalized_batch", batch)
                connection.execute(
                    """
                    INSERT INTO tokens
                    SELECT CAST(entity_id AS VARCHAR), ?, country, token
                    FROM normalized_batch
                    CROSS JOIN UNNEST(name_tokens) AS expanded(token)
                    WHERE length(token) >= ?
                      AND token NOT IN (
                          SELECT * FROM UNNEST(?::VARCHAR[])
                      )
                    """,
                    [source, config.MIN_TOKEN_LENGTH_FOR_BLOCKING, sorted(normalize.LEGAL_SUFFIX_TOKENS)],
                )
                connection.unregister("normalized_batch")

        join_started = time.perf_counter()
        candidates = connection.execute(
            """
            WITH distinct_tokens AS (
                SELECT DISTINCT entity_id, source, country, token
                FROM tokens
            ),
            usable_tokens AS (
                SELECT country, token
                FROM distinct_tokens
                GROUP BY country, token
                HAVING COUNT(*) <= ?
            ),
            candidate_scores AS (
                SELECT
                    s1.entity_id AS source1_entity_id,
                    other.entity_id AS candidate_entity_id,
                    other.source,
                    COUNT(DISTINCT s1.token) AS shared_token_count
                FROM distinct_tokens AS s1
                JOIN distinct_tokens AS other
                    ON s1.token = other.token
                   AND s1.country = other.country
                JOIN usable_tokens
                    ON usable_tokens.country = s1.country
                   AND usable_tokens.token = s1.token
                WHERE s1.source = 'S1'
                  AND other.source IN ('S2', 'S3')
                GROUP BY s1.entity_id, other.entity_id, other.source
            ),
            ranked_candidates AS (
                SELECT
                    source1_entity_id,
                    candidate_entity_id,
                    source,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id
                        ORDER BY shared_token_count DESC, candidate_entity_id
                    ) AS candidate_rank
                FROM candidate_scores
            )
            SELECT
                source1_entity_id,
                candidate_entity_id,
                source
            FROM ranked_candidates
            WHERE candidate_rank <= ?
            """,
            [
                config.TOKEN_FREQUENCY_THRESHOLD,
                config.SQL_PRE_RANK_CANDIDATES_PER_ENTITY,
            ],
        ).fetchdf()
        print(
            f"[candidate join] {len(candidates)} rows in "
            f"{time.perf_counter() - join_started:.2f}s",
            flush=True,
        )
        candidates["block_channel"] = "token"
        return candidates[
            ["source1_entity_id", "candidate_entity_id", "source", "block_channel"]
        ]
    finally:
        connection.close()


def load_candidate_source_rows(
    s2_path,
    s3_path,
    candidates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only S2/S3 rows referenced by a blocked candidate dataframe."""
    if candidates_df.empty:
        empty = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        return empty.copy(), empty.copy()

    ids = candidates_df[["candidate_entity_id", "source"]].drop_duplicates()
    connection = duckdb.connect()
    try:
        connection.register("candidate_ids", ids)
        frames = []
        for source, path in (("S2", s2_path), ("S3", s3_path)):
            frames.append(
                connection.execute(
                    """
                    SELECT r.*
                    FROM read_csv(?, sep='\\t', header=true, all_varchar=true) AS r
                    JOIN candidate_ids AS ids
                      ON r.entity_id = ids.candidate_entity_id
                    WHERE ids.source = ?
                    """,
                    [str(path), source],
                ).fetchdf()
            )
        return frames[0], frames[1]
    finally:
        connection.close()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds name_clean / name_tokens / address_clean / address_tokens /
    address_present columns in place (returns the same df for chaining)."""
    name_norm = df["business_name"].fillna("").map(normalize.normalize_name)
    df["name_clean"] = name_norm.map(lambda d: d["clean"])
    df["name_expanded"] = name_norm.map(lambda d: d["expanded"])
    df["name_tokens"] = name_norm.map(lambda d: d["tokens"])

    addr_norm = df["business_address"].map(normalize.normalize_address)
    df["address_clean"] = addr_norm.map(lambda d: d["clean"])
    df["address_expanded"] = addr_norm.map(lambda d: d["expanded"])
    df["address_tokens"] = addr_norm.map(lambda d: d["tokens"])
    df["address_present"] = addr_norm.map(lambda d: d["present"])
    return df


def embedding_channel_for_country(s1_group: pd.DataFrame, other_group: pd.DataFrame):
    """Returns dict: source1_entity_id -> set of candidate entity_ids via
    ANN search over multilingual embeddings. Returns {} if the embedding
    stack isn't installed — callers should treat that as "channel skipped",
    not an error."""
    if emb_module is None or not emb_module.EMBEDDINGS_AVAILABLE:
        return {}
    if len(other_group) == 0 or len(s1_group) == 0:
        return {}

    other_texts = other_group["name_expanded"].tolist()
    s1_texts = s1_group["name_expanded"].tolist()

    other_emb = emb_module.embed_texts(other_texts)
    s1_emb = emb_module.embed_texts(s1_texts)
    if other_emb is None or s1_emb is None:
        return {}

    index = emb_module.build_ann_index(other_emb)
    top_k = min(config.EMBEDDING_TOP_K, len(other_group))
    _scores, idxs = emb_module.query_ann_index(index, s1_emb, top_k)

    other_ids = other_group["entity_id"].to_numpy()
    result = {}
    for row_pos, s1_entity_id in enumerate(s1_group["entity_id"].tolist()):
        candidate_ids = {other_ids[i] for i in idxs[row_pos] if i != -1}
        result[s1_entity_id] = candidate_ids
    return result


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Main entry point. Runs both channels, per country group (grouped
    generically — works for any country string, including one never seen
    in training, e.g. France). Returns a long-format candidate table:
    source1_entity_id, candidate_entity_id, source, block_channel.
    """
    for df in (s1_df, s2_df, s3_df):
        add_normalized_columns(df)

    s2_df = s2_df.copy()
    s3_df = s3_df.copy()
    s2_df["source"] = "S2"
    s3_df["source"] = "S3"
    other_df = pd.concat([s2_df, s3_df], ignore_index=True)

    token_candidates = token_channel_for_country(s1_df, other_df, "token")
    candidate_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "source",
        "block_channel",
    ]
    if emb_module is None or not emb_module.EMBEDDINGS_AVAILABLE:
        token_candidates["block_channel"] = "token"
        return token_candidates[candidate_columns]

    token_map = {}
    for row in token_candidates.itertuples(index=False):
        token_map.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)

    rows = []
    for country_value, s1_group in s1_df.groupby("country"):
        other_group = other_df[other_df["country"] == country_value]
        if len(other_group) == 0:
            # No S2/S3 records at all for this country in this split.
            # Every S1 entity here will correctly end up with an empty
            # candidate list downstream — this is expected, not a bug.
            continue

        emb_map = embedding_channel_for_country(s1_group, other_group)

        other_lookup = other_group.set_index("entity_id")["source"].to_dict()

        for s1_entity_id in s1_group["entity_id"]:
            tok_c = token_map.get(s1_entity_id, set())
            emb_c = emb_map.get(s1_entity_id, set())
            for cid in tok_c | emb_c:
                channel = (
                    "both" if (cid in tok_c and cid in emb_c)
                    else "token" if cid in tok_c
                    else "embedding"
                )
                rows.append((s1_entity_id, cid, other_lookup[cid], channel))

    return pd.DataFrame(
        rows, columns=["source1_entity_id", "candidate_entity_id", "source", "block_channel"]
    )


def rank_and_cap_candidates(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_per_entity: int = config.MAX_CANDIDATES_PER_ENTITY,
) -> pd.DataFrame:
    """Cheap rapidfuzz-based ranking to cut each S1's candidate list down
    to `max_per_entity` before the (more expensive) full feature pipeline
    runs. This is the candidate set that should be written to
    candidate_pairs.tsv, since it's the last filtering stage before the
    matching model scores anything.
    """
    required_s1_ids = candidates_df["source1_entity_id"].unique()
    required_other_ids = candidates_df["candidate_entity_id"].unique()
    s1_names = (
        s1_df[s1_df["entity_id"].isin(required_s1_ids)]
        .set_index("entity_id")["name_expanded"]
        .to_dict()
    )
    other_names = (
        other_df[other_df["entity_id"].isin(required_other_ids)]
        .set_index("entity_id")["name_expanded"]
        .to_dict()
    )

    output_chunks = []
    group_chunk = []
    processed_groups = 0
    next_progress = 100_000
    started = time.perf_counter()

    def rank_chunk(groups):
        kept_rows = []
        for s1_id, group in groups:
            s1_name = s1_names.get(s1_id, "")
            candidate_ids = group["candidate_entity_id"].tolist()
            candidate_names = [other_names.get(candidate_id, "") for candidate_id in candidate_ids]
            scores = process.cdist(
                [s1_name],
                candidate_names,
                scorer=fuzz.token_sort_ratio,
            )[0]
            ranked_positions = sorted(
                range(len(candidate_ids)),
                key=lambda position: scores[position],
                reverse=True,
            )
            top_positions = ranked_positions[:max_per_entity]
            top_ids = {candidate_ids[position] for position in top_positions}
            kept_rows.append(group[group["candidate_entity_id"].isin(top_ids)])
        return pd.concat(kept_rows, ignore_index=True) if kept_rows else candidates_df.iloc[0:0]

    for group in candidates_df.groupby("source1_entity_id"):
        group_chunk.append(group)
        if len(group_chunk) >= 150_000:
            output_chunks.append(rank_chunk(group_chunk))
            processed_groups += len(group_chunk)
            group_chunk = []
            if processed_groups >= next_progress:
                print(
                    f"[ranking progress] {processed_groups} groups in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                next_progress += 100_000

    if group_chunk:
        output_chunks.append(rank_chunk(group_chunk))
        processed_groups += len(group_chunk)
        if processed_groups >= next_progress:
            print(
                f"[ranking progress] {processed_groups} groups in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )

    if not output_chunks:
        return candidates_df.iloc[0:0]
    return pd.concat(output_chunks, ignore_index=True)
