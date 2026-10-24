import sys

sys.path.insert(0, "code/business_entity_resolution/src")

import pandas as pd

from features import build_feature_frame
from blocking import make_block_keys


# =========================================================
# 1. Load training pairs
# =========================================================

pairs = pd.read_csv(
    "code/business_entity_resolution/training_pairs_sample.csv"
)

print("Training pairs:", len(pairs))


# =========================================================
# 2. Load required Source 1 records
# =========================================================

s1_ids = set(
    pairs["source1_entity_id"]
)

s1_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[
        chunk["entity_id"].isin(s1_ids)
    ]

    if not part.empty:
        s1_parts.append(part)


s1 = pd.concat(
    s1_parts,
    ignore_index=True,
)


# =========================================================
# 3. Process S2 pairs
# =========================================================

pairs_s2 = pairs[
    pairs["source"] == "S2"
].copy()

s2_ids = set(
    pairs_s2["candidate_entity_id"]
)

s2_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[
        chunk["entity_id"].isin(s2_ids)
    ]

    if not part.empty:
        s2_parts.append(part)


s2 = pd.concat(
    s2_parts,
    ignore_index=True,
)


# =========================================================
# 4. Process S3 pairs
# =========================================================

pairs_s3 = pairs[
    pairs["source"] == "S3"
].copy()

s3_ids = set(
    pairs_s3["candidate_entity_id"]
)

s3_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[
        chunk["entity_id"].isin(s3_ids)
    ]

    if not part.empty:
        s3_parts.append(part)


s3 = pd.concat(
    s3_parts,
    ignore_index=True,
)


# =========================================================
# 5. Normalize blocking fields
# =========================================================

s1 = make_block_keys(s1)
s2 = make_block_keys(s2)
s3 = make_block_keys(s3)


# =========================================================
# 6. Build S2 features
# =========================================================

print("Building S2 features...")

X2, keys2 = build_feature_frame(
    s1,
    s2,
    pairs_s2,
)


X2["source"] = "S2"

X2["source1_entity_id"] = [
    key[0] for key in keys2
]

X2["candidate_entity_id"] = [
    key[1] for key in keys2
]

X2["label"] = pairs_s2["label"].values


# =========================================================
# 7. Build S3 features
# =========================================================

print("Building S3 features...")

X3, keys3 = build_feature_frame(
    s1,
    s3,
    pairs_s3,
)


X3["source"] = "S3"

X3["source1_entity_id"] = [
    key[0] for key in keys3
]

X3["candidate_entity_id"] = [
    key[1] for key in keys3
]

X3["label"] = pairs_s3["label"].values


# =========================================================
# 8. Combine
# =========================================================

features = pd.concat(
    [X2, X3],
    ignore_index=True,
)


# =========================================================
# 9. Save
# =========================================================

output_path = (
    "code/business_entity_resolution/"
    "training_features_sample.csv"
)

features.to_csv(
    output_path,
    index=False,
)


# =========================================================
# 10. Summary
# =========================================================

print()
print("==============================")
print("FEATURE DATASET")
print("==============================")

print("Rows:", len(features))

print(
    "Features:",
    len(
        [
            c
            for c in features.columns
            if c
            not in [
                "source",
                "source1_entity_id",
                "candidate_entity_id",
                "label",
            ]
        ]
    ),
)

print(
    "Positive:",
    int(features["label"].sum()),
)

print(
    "Negative:",
    int((features["label"] == 0).sum()),
)

print()
print("Saved:", output_path)