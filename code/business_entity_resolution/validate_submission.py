from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SUBMISSION_PATH = (
    ROOT
    / "code"
    / "business_entity_resolution"
    / "outputs"
    / "submission_all_s1.csv"
)

S1_PATH = (
    ROOT
    / "dataset"
    / "test"
    / "test_source1.tsv"
)


EXPECTED_S1_ROWS = 1_732_544

S1_PATTERN = re.compile(
    r"^S1-\d+$"
)

MATCH_PATTERN = re.compile(
    r"^S[23]-\d+$"
)


print("=" * 70)
print("FINAL SUBMISSION VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# Check header
# ------------------------------------------------------------

header = pd.read_csv(
    SUBMISSION_PATH,
    sep="\t",
    nrows=0,
)

expected_columns = [
    "source1_entity_id",
    "matched_entity_ids",
]

print()
print("Columns:")
print(list(header.columns))

if list(header.columns) != expected_columns:
    raise ValueError(
        f"Unexpected columns: {list(header.columns)}"
    )

print("Header: PASS")


# ------------------------------------------------------------
# Check rows and IDs
# ------------------------------------------------------------

total_rows = 0
unique_s1 = set()

bad_s1 = 0
bad_matches = 0
duplicate_matches = 0
empty_matches = 0
matched_s1 = 0


for chunk in pd.read_csv(
    SUBMISSION_PATH,
    sep="\t",
    dtype=str,
    chunksize=100_000,
):

    total_rows += len(chunk)

    # S1 IDs
    for value in chunk["source1_entity_id"]:
        if not S1_PATTERN.fullmatch(str(value)):
            bad_s1 += 1

        unique_s1.add(value)

    # Matched IDs
    for value in chunk["matched_entity_ids"].fillna(""):

        if not value:
            empty_matches += 1
            continue

        matched_s1 += 1

        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        # Check duplicate candidate IDs inside one row.
        if len(ids) != len(set(ids)):
            duplicate_matches += 1

        # Check every ID format.
        for candidate_id in ids:
            if not MATCH_PATTERN.fullmatch(
                candidate_id
            ):
                bad_matches += 1


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Rows:                 {total_rows:,}"
)

print(
    f"Unique S1 IDs:        {len(unique_s1):,}"
)

print(
    f"Rows with matches:    {matched_s1:,}"
)

print(
    f"Rows without matches: {empty_matches:,}"
)

print(
    f"Bad S1 IDs:           {bad_s1:,}"
)

print(
    f"Bad match IDs:        {bad_matches:,}"
)

print(
    f"Rows with duplicate matches: "
    f"{duplicate_matches:,}"
)


# ------------------------------------------------------------
# Expected counts
# ------------------------------------------------------------

print()
print("=" * 70)
print("CHECKS")
print("=" * 70)

assert total_rows == EXPECTED_S1_ROWS, (
    f"Expected {EXPECTED_S1_ROWS:,} rows, "
    f"got {total_rows:,}"
)

assert len(unique_s1) == EXPECTED_S1_ROWS, (
    f"Expected {EXPECTED_S1_ROWS:,} unique S1 IDs, "
    f"got {len(unique_s1):,}"
)

assert bad_s1 == 0, (
    f"Found {bad_s1:,} malformed S1 IDs"
)

assert bad_matches == 0, (
    f"Found {bad_matches:,} malformed match IDs"
)

assert duplicate_matches == 0, (
    f"Found {duplicate_matches:,} rows "
    f"with duplicate candidate IDs"
)


print()
print("ALL VALIDATION CHECKS PASSED")
print()
print(
    f"Validated submission: {SUBMISSION_PATH}"
)