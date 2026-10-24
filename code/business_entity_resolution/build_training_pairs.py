import sys

sys.path.insert(0, "code/business_entity_resolution/src")

import pandas as pd
from blocking import Blocker


# =========================================================
# 1. Load 1,000 labelled Source-1 records
# =========================================================

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=1000,
)

s1_ids = set(gt["source1_entity_id"])


# =========================================================
# 2. Load Source 1 records
# =========================================================

s1_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s1_ids)]

    if not part.empty:
        s1_parts.append(part)

s1 = pd.concat(
    s1_parts,
    ignore_index=True,
)


# =========================================================
# 3. Get ground-truth target IDs
# =========================================================

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


# =========================================================
# 4. Load the known positive target records
# =========================================================

s2_parts = []
s3_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s2_ids)]

    if not part.empty:
        s2_parts.append(part)


for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s3_ids)]

    if not part.empty:
        s3_parts.append(part)


s2 = pd.concat(
    s2_parts,
    ignore_index=True,
)

s3 = pd.concat(
    s3_parts,
    ignore_index=True,
)


# =========================================================
# 5. Generate candidate pairs
# =========================================================

print("Generating S1 -> S2 candidates...")

blocker2 = Blocker()
blocker2.fit(s2)

candidates2 = blocker2.generate(s1)


print("Generating S1 -> S3 candidates...")

blocker3 = Blocker()
blocker3.fit(s3)

candidates3 = blocker3.generate(s1)


# =========================================================
# 6. Create ground-truth lookup
# =========================================================

positive_pairs = set()

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    for target_id in str(
        row["matched_entity_ids"]
    ).split(","):

        target_id = target_id.strip()

        positive_pairs.add(
            (source1_id, target_id)
        )


# =========================================================
# 7. Convert candidates into labelled pairs
# =========================================================

pairs = []


for _, row in candidates2.iterrows():

    pair = (
        row["source1_entity_id"],
        row["candidate_entity_id"],
    )

    pairs.append({
        "source1_entity_id": pair[0],
        "candidate_entity_id": pair[1],
        "source": "S2",
        "label": int(pair in positive_pairs),
    })


for _, row in candidates3.iterrows():

    pair = (
        row["source1_entity_id"],
        row["candidate_entity_id"],
    )

    pairs.append({
        "source1_entity_id": pair[0],
        "candidate_entity_id": pair[1],
        "source": "S3",
        "label": int(pair in positive_pairs),
    })


pairs_df = pd.DataFrame(pairs)


# =========================================================
# 8. Remove duplicate pairs
# =========================================================

pairs_df = pairs_df.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
).reset_index(drop=True)


# =========================================================
# 9. Print statistics
# =========================================================

print()
print("==============================")
print("TRAINING PAIR SUMMARY")
print("==============================")

print("Total pairs:", len(pairs_df))

print(
    "Positive pairs:",
    int(pairs_df["label"].sum()),
)

print(
    "Negative pairs:",
    int((pairs_df["label"] == 0).sum()),
)

print()
print(
    pairs_df.groupby(
        ["source", "label"]
    ).size()
)


# =========================================================
# 10. Save
# =========================================================

output_path = (
    "code/business_entity_resolution/"
    "training_pairs_sample.csv"
)

pairs_df.to_csv(
    output_path,
    index=False,
)

print()
print("Saved:", output_path)

sys.path.insert(0, "code/business_entity_resolution/src")

import pandas as pd
from blocking import Blocker


# =========================================================
# 1. Load 1,000 labelled Source-1 records
# =========================================================

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=1000,
)

s1_ids = set(gt["source1_entity_id"])


# =========================================================
# 2. Load Source 1 records
# =========================================================

s1_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s1_ids)]

    if not part.empty:
        s1_parts.append(part)

s1 = pd.concat(
    s1_parts,
    ignore_index=True,
)


# =========================================================
# 3. Get ground-truth target IDs
# =========================================================

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


# =========================================================
# 4. Load the known positive target records
# =========================================================

s2_parts = []
s3_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s2_ids)]

    if not part.empty:
        s2_parts.append(part)


for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s3_ids)]

    if not part.empty:
        s3_parts.append(part)


s2 = pd.concat(
    s2_parts,
    ignore_index=True,
)

s3 = pd.concat(
    s3_parts,
    ignore_index=True,
)


# =========================================================
# 5. Generate candidate pairs
# =========================================================

print("Generating S1 -> S2 candidates...")

blocker2 = Blocker()
blocker2.fit(s2)

candidates2 = blocker2.generate(s1)


print("Generating S1 -> S3 candidates...")

blocker3 = Blocker()
blocker3.fit(s3)

candidates3 = blocker3.generate(s1)


# =========================================================
# 6. Create ground-truth lookup
# =========================================================

positive_pairs = set()

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    for target_id in str(
        row["matched_entity_ids"]
    ).split(","):

        target_id = target_id.strip()

        positive_pairs.add(
            (source1_id, target_id)
        )


# =========================================================
# 7. Convert candidates into labelled pairs
# =========================================================

pairs = []


for _, row in candidates2.iterrows():

    pair = (
        row["source1_entity_id"],
        row["candidate_entity_id"],
    )

    pairs.append({
        "source1_entity_id": pair[0],
        "candidate_entity_id": pair[1],
        "source": "S2",
        "label": int(pair in positive_pairs),
    })


for _, row in candidates3.iterrows():

    pair = (
        row["source1_entity_id"],
        row["candidate_entity_id"],
    )

    pairs.append({
        "source1_entity_id": pair[0],
        "candidate_entity_id": pair[1],
        "source": "S3",
        "label": int(pair in positive_pairs),
    })


pairs_df = pd.DataFrame(pairs)


# =========================================================
# 8. Remove duplicate pairs
# =========================================================

pairs_df = pairs_df.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
).reset_index(drop=True)


# =========================================================
# 9. Print statistics
# =========================================================

print()
print("==============================")
print("TRAINING PAIR SUMMARY")
print("==============================")

print("Total pairs:", len(pairs_df))

print(
    "Positive pairs:",
    int(pairs_df["label"].sum()),
)

print(
    "Negative pairs:",
    int((pairs_df["label"] == 0).sum()),
)

print()
print(
    pairs_df.groupby(
        ["source", "label"]
    ).size()
)


# =========================================================
# 10. Save
# =========================================================

output_path = (
    "code/business_entity_resolution/"
    "training_pairs_sample.csv"
)

pairs_df.to_csv(
    output_path,
    index=False,
)

print()
print("Saved:", output_path)