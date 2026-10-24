import gc
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "dataset" / "test"

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

SRC_DIR = ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from blocking import make_block_keys


S1_CHUNK_SIZE = 5_000
TARGET_CHUNK_SIZE = 100_000
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

TEMP_DIR = Path("C:/ByteForgeTemp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = TEMP_DIR / "candidate_export.duckdb"

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S2_TEMP = TEMP_DIR / "candidate_pairs_s2.tsv"
S3_TEMP = TEMP_DIR / "candidate_pairs_s3.tsv"
FINAL_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"


def qident(name):
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def connect_db():
    if DB_PATH.exists():
        DB_PATH.unlink()

    return duckdb.connect(str(DB_PATH))


def build_target_store(con, target_path, table_name):
    print("=" * 70)
    print(f"BUILDING TARGET STORE: {target_path.name}")
    print("=" * 70)

    con.execute(f"DROP TABLE IF EXISTS {qident(table_name)}")

    reader = pd.read_csv(
        target_path,
        sep="\t",
        dtype=str,
        chunksize=TARGET_CHUNK_SIZE,
    )

    first = True
    total = 0

    for chunk_no, raw in enumerate(reader, start=1):
        chunk = make_block_keys(raw)

        chunk.insert(
            0,
            "row_id",
            range(total, total + len(chunk)),
        )

        chunk = chunk[TARGET_COLUMNS]

        con.register("target_chunk", chunk)

        if first:
            con.execute(
                f"""
                CREATE TABLE {qident(table_name)} AS
                SELECT * FROM target_chunk
                """
            )
            first = False
        else:
            con.execute(
                f"""
                INSERT INTO {qident(table_name)}
                SELECT * FROM target_chunk
                """
            )

        con.unregister("target_chunk")

        total += len(chunk)

        if chunk_no % 10 == 0:
            print(f"  loaded {total:,} target rows")

        del raw
        del chunk
        gc.collect()

    print(f"Target rows: {total:,}")


def build_lookup_tables(con, table_name):
    print("BUILDING BLOCK LOOKUPS")

    lookup_names = []

    for key in KEY_COLUMNS:
        lookup = "lookup_" + key

        con.execute(
            f"DROP TABLE IF EXISTS {qident(lookup)}"
        )

        con.execute(
            f"""
            CREATE TABLE {qident(lookup)} AS
            SELECT
                {qident(key)},
                row_id
            FROM (
                SELECT
                    {qident(key)},
                    row_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY {qident(key)}
                        ORDER BY row_id
                    ) AS rn
                FROM {qident(table_name)}
                WHERE {qident(key)} IS NOT NULL
                  AND {qident(key)} <> ''
            )
            WHERE rn <= {MAX_CANDIDATES_PER_KEY}
            """
        )

        count = con.execute(
            f"SELECT COUNT(*) FROM {qident(lookup)}"
        ).fetchone()[0]

        print(f"  {key}: {count:,} lookup rows")

        lookup_names.append(lookup)

    return lookup_names


def generate_candidates(con, s1_chunk, lookup_names):
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
        SELECT DISTINCT
            source1_entity_id,
            target_row_id
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
            t.entity_id
        FROM candidates c
        INNER JOIN target_store t
            ON c.target_row_id = t.row_id
        """
    ).fetch_df()

    return (
        target_rows[
            ["source1_entity_id", "entity_id"]
        ]
        .rename(
            columns={
                "entity_id": "candidate_entity_id"
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def process_source(
    con,
    source_name,
    target_path,
    output_path,
):
    print()
    print("=" * 70)
    print(f"PROCESSING {source_name} CANDIDATES")
    print("=" * 70)

    build_target_store(
        con,
        target_path,
        "target_store",
    )

    lookup_names = build_lookup_tables(
        con,
        "target_store",
    )

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        chunksize=S1_CHUNK_SIZE,
    )

    first_write = True
    total_candidates = 0
    total_s1 = 0

    for chunk_no, raw_s1 in enumerate(reader, start=1):

        s1 = make_block_keys(raw_s1)

        candidate_pairs = generate_candidates(
            con,
            s1,
            lookup_names,
        )

        if len(candidate_pairs):
            grouped = (
                candidate_pairs
                .groupby("source1_entity_id")[
                    "candidate_entity_id"
                ]
                .agg(
                    lambda x: ",".join(
                        sorted(set(x))
                    )
                )
                .rename("candidate_entity_ids")
                .reset_index()
            )
        else:
            grouped = pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_ids",
                ]
            )

        output_chunk = pd.DataFrame(
            {
                "source1_entity_id":
                    raw_s1["entity_id"].astype(str)
            }
        )

        output_chunk = output_chunk.merge(
            grouped,
            on="source1_entity_id",
            how="left",
        )

        output_chunk["candidate_entity_ids"] = (
            output_chunk["candidate_entity_ids"]
            .fillna("")
        )

        output_chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

        total_s1 += len(s1)
        total_candidates += len(candidate_pairs)

        print(
            f"{source_name} | chunk {chunk_no} | "
            f"S1={len(s1):,} | "
            f"candidates={len(candidate_pairs):,}"
        )

        del raw_s1
        del s1
        del candidate_pairs
        del grouped
        del output_chunk

        gc.collect()

    print()
    print(f"{source_name} S1 rows:    {total_s1:,}")
    print(f"{source_name} candidates: {total_candidates:,}")

    con.execute(
        "DROP TABLE IF EXISTS target_store"
    )

    for lookup in lookup_names:
        con.execute(
            f"DROP TABLE IF EXISTS {qident(lookup)}"
        )


def combine_sources():
    print()
    print("=" * 70)
    print("COMBINING S2 + S3 CANDIDATES")
    print("=" * 70)

    s2_path = S2_TEMP.as_posix()
    s3_path = S3_TEMP.as_posix()
    s1_path = S1_PATH.as_posix()
    final_path = FINAL_OUTPUT.as_posix()

    con = duckdb.connect()

    con.execute(
        f"""
        CREATE TABLE candidates AS
        SELECT
            source1_entity_id,
            candidate_entity_ids
        FROM read_csv(
            '{s2_path}',
            delim='\\t',
            header=true
        )
        """
    )

    con.execute(
        f"""
        INSERT INTO candidates
        SELECT
            source1_entity_id,
            candidate_entity_ids
        FROM read_csv(
            '{s3_path}',
            delim='\\t',
            header=true
        )
        """
    )

    con.execute(
        """
        CREATE TABLE exploded AS
        SELECT
            source1_entity_id,
            TRIM(candidate_id) AS candidate_entity_id
        FROM candidates,
        UNNEST(
            STRING_SPLIT(
                candidate_entity_ids,
                ','
            )
        ) AS t(candidate_id)
        WHERE candidate_entity_ids <> ''
        """
    )

    con.execute(
        f"""
        COPY (
            WITH all_s1 AS (
                SELECT
                    entity_id AS source1_entity_id
                FROM read_csv(
                    '{s1_path}',
                    delim='\\t',
                    header=true
                )
            ),
            grouped AS (
                SELECT
                    source1_entity_id,
                    STRING_AGG(
                        candidate_entity_id,
                        ','
                        ORDER BY candidate_entity_id
                    ) AS candidate_entity_ids
                FROM (
                    SELECT DISTINCT
                        source1_entity_id,
                        candidate_entity_id
                    FROM exploded
                    WHERE candidate_entity_id <> ''
                )
                GROUP BY source1_entity_id
            )
            SELECT
                a.source1_entity_id,
                COALESCE(
                    g.candidate_entity_ids,
                    ''
                ) AS candidate_entity_ids
            FROM all_s1 a
            LEFT JOIN grouped g
                ON a.source1_entity_id =
                   g.source1_entity_id
            ORDER BY a.source1_entity_id
        )
        TO '{final_path}'
        (
            DELIMITER '\\t',
            HEADER
        )
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{final_path}',
            delim='\\t',
            header=true
        )
        """
    ).fetchone()[0]

    print(f"Final candidate rows: {count:,}")

    con.close()


def main():
    for path in [
        S2_TEMP,
        S3_TEMP,
        FINAL_OUTPUT,
    ]:
        if path.exists():
            path.unlink()

    con = connect_db()

    try:
        process_source(
            con,
            "S2",
            S2_PATH,
            S2_TEMP,
        )

        process_source(
            con,
            "S3",
            S3_PATH,
            S3_TEMP,
        )

    finally:
        con.close()

    combine_sources()

    print()
    print("=" * 70)
    print("CANDIDATE EXPORT COMPLETE")
    print("=" * 70)
    print(f"Output: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()