from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "code"
    / "business_entity_resolution"
    / "outputs"
)

S2_PATH = OUTPUT_DIR / "test_matches_s2.csv"
S3_PATH = OUTPUT_DIR / "test_matches_s3.csv"

COMBINED_PATH = OUTPUT_DIR / "combined_matches.csv"


def process_file(path):
    parts = []

    for chunk in pd.read_csv(
        path,
        dtype={
            "source1_entity_id": "string",
            "candidate_entity_id": "string",
            "source": "string",
            "score": "float32",
        },
        chunksize=100_000,
    ):
        parts.append(
            chunk[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "source",
                    "score",
                ]
            ]
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


print("=" * 70)
print("LOADING S2 MATCHES")
print("=" * 70)

s2 = process_file(S2_PATH)

print(
    f"S2 rows: {len(s2):,}"
)

print()
print("=" * 70)
print("LOADING S3 MATCHES")
print("=" * 70)

s3 = process_file(S3_PATH)

print(
    f"S3 rows: {len(s3):,}"
)

print()
print("=" * 70)
print("COMBINING")
print("=" * 70)

combined = pd.concat(
    [s2, s3],
    ignore_index=True,
)

# Remove exact duplicate predictions.
combined = combined.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

# Sort for reproducibility.
combined = combined.sort_values(
    [
        "source1_entity_id",
        "source",
        "score",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)

combined.to_csv(
    COMBINED_PATH,
    index=False,
)

print(
    f"Combined rows: {len(combined):,}"
)

print(
    f"Unique S1 IDs: "
    f"{combined['source1_entity_id'].nunique():,}"
)

print(
    f"Output: {COMBINED_PATH}"
)