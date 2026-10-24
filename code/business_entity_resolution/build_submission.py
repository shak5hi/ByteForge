from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "code" / "business_entity_resolution" / "outputs"

S1_PATH = TEST_DIR / "test_source1.tsv"
COMBINED_PATH = OUTPUT_DIR / "combined_matches.csv"

SUBMISSION_PATH = OUTPUT_DIR / "submission_all_s1.csv"


print("=" * 70)
print("LOADING SOURCE 1 TEST IDS")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id"],
)

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)

print(
    f"S1 test entities: {len(s1):,}"
)

print()
print("=" * 70)
print("LOADING COMBINED MATCHES")
print("=" * 70)

combined_parts = []

for chunk in pd.read_csv(
    COMBINED_PATH,
    dtype={
        "source1_entity_id": "string",
        "candidate_entity_id": "string",
        "source": "string",
        "score": "float32",
    },
    chunksize=100_000,
):
    combined_parts.append(
        chunk[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "source",
                "score",
            ]
        ]
    )

matches = pd.concat(
    combined_parts,
    ignore_index=True,
)

print(
    f"Combined match rows: {len(matches):,}"
)

print()
print("=" * 70)
print("CREATING MATCHED-ID LISTS")
print("=" * 70)

# Sort by score so the strongest model predictions appear first.
matches = matches.sort_values(
    [
        "source1_entity_id",
        "score",
    ],
    ascending=[
        True,
        False,
    ],
)

# Remove any duplicate candidate IDs for an S1 entity.
matches = matches.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

grouped = (
    matches
    .groupby(
        "source1_entity_id",
        sort=False,
    )["candidate_entity_id"]
    .apply(
        lambda x: ",".join(x.astype(str))
    )
    .reset_index()
)

grouped = grouped.rename(
    columns={
        "candidate_entity_id": "matched_entity_ids"
    }
)

print(
    f"S1 entities with matches: "
    f"{len(grouped):,}"
)

print()
print("=" * 70)
print("MERGING WITH ALL SOURCE 1 ENTITIES")
print("=" * 70)

submission = s1.merge(
    grouped,
    on="source1_entity_id",
    how="left",
)

submission["matched_entity_ids"] = (
    submission["matched_entity_ids"]
    .fillna("")
)

print(
    f"Final rows: {len(submission):,}"
)

print(
    "Rows with at least one match:",
    int(
        (
            submission["matched_entity_ids"]
            != ""
        ).sum()
    ),
)

print(
    "Rows with no predicted match:",
    int(
        (
            submission["matched_entity_ids"]
            == ""
        ).sum()
    ),
)

submission.to_csv(
    SUBMISSION_PATH,
    sep="\t",
    index=False,
)

print()
print("=" * 70)
print("SUBMISSION FILE CREATED")
print("=" * 70)

print(
    f"Output: {SUBMISSION_PATH}"
)