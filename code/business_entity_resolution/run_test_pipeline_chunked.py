
import argparse
import gc
import sys
import time
from pathlib import Path

import duckdb
import joblib
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "dataset" / "test"

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

MODEL_PATH = ROOT / "code" / "business_entity_resolution" / "models" / "entity_match_model.joblib"
THRESHOLD_PATH = ROOT / "code" / "business_entity_resolution" / "models" / "threshold.txt"
OUTPUT_DIR = ROOT / "code" / "business_entity_resolution" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from blocking import make_block_keys
from features import build_feature_frame

# ============================================================
# SETTINGS
# ============================================================

DEFAULT_S1_CHUNK_SIZE = 5_000
DEFAULT_TARGET_CHUNK_SIZE = 100_000
MAX_CANDIDATES_PER_KEY = 20

KEY_COLUMNS = [
    "block_name_exact",
    "block_name_first",
    "block_name_prefix",
    "block_address_first",
    "block_address_tokens",
]

TARGET_COLUMNS = [
    "row_id",
    "entity_id",
    "name_norm",
    "address_norm",
    "country_norm",
    "block_name_exact",
    "block_name_first",
    "block_name_prefix",
    "block_address_first",
    "block_address_tokens",
]


def qident(name):
    """Validate simple SQL identifiers used by this script."""
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def load_model():
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    print("Model:", type(model).__name__)
    print("Features:", len(feature_columns))
    print("Threshold:", threshold)

    return model, feature_columns, threshold


def connect_db(db_path):
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    return con


def build_target_store(con, filepath, table_name, target_chunk_size, target_rows=None):
    """
    Stream the target TSV through pandas chunks.

    Only normalized fields and blocking fields are persisted in DuckDB.
    The full target is never loaded into pandas.
    """
    table_name = qident(table_name)

    print()
    print("=" * 70)
    print(f"BUILDING TARGET STORE: {filepath.name}")
    print("=" * 70)

    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    reader_kwargs = {
        "sep": "\t",
        "dtype": str,
        "chunksize": target_chunk_size,
    }
    if target_rows is not None:
        reader_kwargs["nrows"] = target_rows

    reader = pd.read_csv(filepath, **reader_kwargs)

    first = True
    offset = 0
    total = 0
    started = time.time()

    for chunk_no, raw in enumerate(reader, start=1):
        block = make_block_keys(raw)

        block["row_id"] = range(offset, offset + len(block))

        store = block[TARGET_COLUMNS].copy()

        con.register("incoming_target", store)

        if first:
            con.execute(
                f"CREATE TABLE {table_name} AS "
                "SELECT * FROM incoming_target"
            )
            first = False
        else:
            con.execute(
                f"INSERT INTO {table_name} "
                "SELECT * FROM incoming_target"
            )

        con.unregister("incoming_target")

        total += len(store)
        offset += len(store)

        print(
            f"Target chunk {chunk_no}: "
            f"{total:,} rows stored"
        )

        del raw, block, store
        gc.collect()

    if first:
        raise RuntimeError(f"No rows were read from {filepath}")

    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_row_id ON {table_name}(row_id)")
    con.execute(f"ANALYZE {table_name}")

    print(f"Target store ready: {total:,} rows")
    print(f"Build time: {(time.time() - started) / 60:.2f} minutes")

    return total


def build_lookup_tables(con, table_name):
    """
    Reproduce Blocker(max_candidates_per_key=20) on disk.

    Each lookup stores only key + target row_id.
    """
    table_name = qident(table_name)

    print()
    print("=" * 70)
    print("BUILDING BLOCKING LOOKUPS")
    print("=" * 70)

    lookup_names = []

    for key in KEY_COLUMNS:
        key = qident(key)
        lookup = qident(f"lookup_{key}")
        lookup_names.append(lookup)

        con.execute(f"DROP TABLE IF EXISTS {lookup}")

        print(f"Building {lookup}...")

        con.execute(
            f"""
            CREATE TABLE {lookup} AS
            SELECT {key}, row_id
            FROM (
                SELECT
                    {key},
                    row_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY {key}
                        ORDER BY row_id
                    ) AS rn
                FROM {table_name}
                WHERE {key} <> ''
            ) ranked
            WHERE rn <= {MAX_CANDIDATES_PER_KEY}
            """
        )

        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{lookup}_key ON {lookup}({key})")
        con.execute(f"ANALYZE {lookup}")

        count = con.execute(f"SELECT COUNT(*) FROM {lookup}").fetchone()[0]
        print(f"  {count:,} lookup rows")

    return lookup_names


def generate_candidates(con, s1_chunk, lookup_names):
    """
    Generate candidate pairs for one S1 chunk.
    """
    con.register("s1_chunk", s1_chunk)

    parts = []

    for key, lookup in zip(KEY_COLUMNS, lookup_names):
        parts.append(
            f"""
            SELECT
                s.entity_id AS source1_entity_id,
                l.row_id AS target_row_id
            FROM s1_chunk s
            INNER JOIN {qident(lookup)} l
                ON s.{qident(key)} = l.{qident(key)}
            """
        )

    sql = """
        SELECT DISTINCT source1_entity_id, target_row_id
        FROM (
            """ + "\nUNION ALL\n".join(parts) + """
        ) candidate_union
    """

    candidates = con.execute(sql).fetch_df()
    con.unregister("s1_chunk")

    if len(candidates) == 0:
        return candidates

    target_rows = con.execute(
        """
        SELECT
            c.source1_entity_id,
            t.entity_id AS entity_id,
            t.name_norm,
            t.address_norm,
            t.country_norm
        FROM candidates c
        INNER JOIN target_store t
            ON c.target_row_id = t.row_id
        """
    ).fetch_df()

    # Candidate pairs used by the feature builder.
    # target_rows contains the target ID as `entity_id`;
    # build_feature_frame expects the candidate-pair column
    # to be named `candidate_entity_id`.
    pair_ids = (
        target_rows[
            ["source1_entity_id", "entity_id"]
        ]
        .rename(columns={"entity_id": "candidate_entity_id"})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # The feature builder expects the target/source-2 dataframe
    # itself to contain `entity_id`.
    target_candidates = (
        target_rows[
            ["entity_id", "name_norm", "address_norm", "country_norm"]
        ]
        .drop_duplicates("entity_id")
        .reset_index(drop=True)
    )

    return pair_ids, target_candidates


def process_source(
    con,
    source_name,
    target_path,
    model,
    feature_columns,
    threshold,
    s1_chunk_size,
    target_chunk_size,
    target_rows=None,
    s1_rows=None,
):
    table_name = "target_store"

    output_path = OUTPUT_DIR / f"test_matches_{source_name.lower()}.csv"

    if output_path.exists():
        output_path.unlink()

    build_target_store(
        con,
        target_path,
        table_name,
        target_chunk_size,
        target_rows=target_rows,
    )

    lookup_names = build_lookup_tables(con, table_name)

    print()
    print("=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    reader_kwargs = {
        "sep": "\t",
        "dtype": str,
        "chunksize": s1_chunk_size,
    }
    if s1_rows is not None:
        reader_kwargs["nrows"] = s1_rows

    reader = pd.read_csv(S1_PATH, **reader_kwargs)

    first_write = True
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    started = time.time()

    for chunk_no, raw_s1 in enumerate(reader, start=1):
        chunk_start = time.time()

        s1 = make_block_keys(raw_s1)
        total_s1 += len(s1)

        result = generate_candidates(con, s1, lookup_names)

        if isinstance(result, tuple):
            candidates, target_candidates = result
        else:
            candidates = result
            target_candidates = pd.DataFrame(
                columns=[
                    "entity_id",
                    "name_norm",
                    "address_norm",
                    "country_norm",
                ]
            )

        candidate_count = len(candidates)
        total_candidates += candidate_count

        print(
            f"{source_name} | chunk {chunk_no} | "
            f"S1={len(s1):,} | candidates={candidate_count:,}"
        )

        if candidate_count:
            X, _ = build_feature_frame(
                s1,
                target_candidates,
                candidates,
            )

            X = X[feature_columns]

            probabilities = model.predict_proba(X)[:, 1]
            mask = probabilities >= threshold

            match_count = int(mask.sum())
            total_matches += match_count

            if match_count:
                matched = candidates.loc[mask].copy()
                matched["source"] = source_name
                matched["score"] = probabilities[mask]

                matched.to_csv(
                    output_path,
                    mode="a",
                    header=first_write,
                    index=False,
                )
                first_write = False

            print(f"  matches={match_count:,}")

            del X, probabilities, mask

        del raw_s1, s1, candidates, target_candidates
        gc.collect()

        print(f"  chunk time={(time.time() - chunk_start):.2f}s")

    print()
    print("=" * 70)
    print(f"{source_name} COMPLETE")
    print("=" * 70)
    print(f"S1 rows processed: {total_s1:,}")
    print(f"Total candidates:   {total_candidates:,}")
    print(f"Total matches:      {total_matches:,}")
    print(f"Total time:         {(time.time() - started) / 60:.2f} minutes")
    print(f"Output:             {output_path}")

    # Clean target tables before the next source.
    con.execute("DROP TABLE IF EXISTS target_store")
    for lookup in lookup_names:
        con.execute(f"DROP TABLE IF EXISTS {qident(lookup)}")

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["S2", "S3", "both"],
        default="S2",
    )
    parser.add_argument(
        "--s1-rows",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--s1-chunk-size",
        type=int,
        default=DEFAULT_S1_CHUNK_SIZE,
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=DEFAULT_TARGET_CHUNK_SIZE,
    )

    args = parser.parse_args()

    model, feature_columns, threshold = load_model()

    print()
    print("=" * 70)
    print("OPENING DUCKDB")
    print("=" * 70)

    DB_DIR = Path("C:/ByteForgeTemp")

    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DB_PATH = (
    DB_DIR
    / "entity_resolution.duckdb"
)

    if DB_PATH.exists():
        DB_PATH.unlink()

    con = connect_db(DB_PATH)

    try:
        if args.source in ("S2", "both"):
            process_source(
                con,
                "S2",
                S2_PATH,
                model,
                feature_columns,
                threshold,
                args.s1_chunk_size,
                args.target_chunk_size,
                target_rows=args.target_rows,
                s1_rows=args.s1_rows,
            )

        if args.source in ("S3", "both"):
            process_source(
                con,
                "S3",
                S3_PATH,
                model,
                feature_columns,
                threshold,
                args.s1_chunk_size,
                args.target_chunk_size,
                target_rows=args.target_rows,
                s1_rows=args.s1_rows,
            )

    finally:
        con.close()

    print()
    print("=" * 70)
    print("RUN FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
