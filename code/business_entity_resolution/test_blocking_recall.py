import sys
sys.path.insert(0, "code/business_entity_resolution/src")

import pandas as pd
from blocking import Blocker


# ---------------------------------------------------------
# 1. Load 1,000 ground-truth rows
# ---------------------------------------------------------

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=1000,
)

print("Ground-truth rows:", len(gt))


# ---------------------------------------------------------
# 2. Get the corresponding Source 1 records
# ---------------------------------------------------------

s1_ids = set(gt["source1_entity_id"])

s1_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    chunksize=200000,
):
    part = chunk[chunk["entity_id"].isin(s1_ids)]

    if not part.empty:
        s1_parts.append(part)

s1 = pd.concat(s1_parts, ignore_index=True)

print("Source 1 records found:", len(s1))


# ---------------------------------------------------------
# 3. Get every ground-truth target ID
# ---------------------------------------------------------

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:
    for entity_id in str(value).split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)

print("Ground-truth S2 IDs:", len(s2_ids))
print("Ground-truth S3 IDs:", len(s3_ids))


# ---------------------------------------------------------
# 4. Load those target records
# ---------------------------------------------------------

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


s2 = pd.concat(s2_parts, ignore_index=True)
s3 = pd.concat(s3_parts, ignore_index=True)

print("S2 records found:", len(s2))
print("S3 records found:", len(s3))


# ---------------------------------------------------------
# 5. Generate candidates
# ---------------------------------------------------------

print("\nGenerating S1 -> S2 candidates...")

blocker2 = Blocker()
blocker2.fit(s2)
candidates2 = blocker2.generate(s1)

print("S2 candidates:", len(candidates2))


print("\nGenerating S1 -> S3 candidates...")

blocker3 = Blocker()
blocker3.fit(s3)
candidates3 = blocker3.generate(s1)

print("S3 candidates:", len(candidates3))


# ---------------------------------------------------------
# 6. Build lookup sets
# ---------------------------------------------------------

candidate_set2 = set(
    zip(
        candidates2["source1_entity_id"],
        candidates2["candidate_entity_id"],
    )
)

candidate_set3 = set(
    zip(
        candidates3["source1_entity_id"],
        candidates3["candidate_entity_id"],
    )
)


# ---------------------------------------------------------
# 7. Calculate recall
# ---------------------------------------------------------

s2_total = 0
s2_found = 0

s3_total = 0
s3_found = 0

missed_s2 = []
missed_s3 = []


for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for target_id in matched_ids:

        target_id = target_id.strip()

        pair = (
            source1_id,
            target_id,
        )

        if target_id.startswith("S2-"):

            s2_total += 1

            if pair in candidate_set2:
                s2_found += 1
            else:
                missed_s2.append(pair)

        elif target_id.startswith("S3-"):

            s3_total += 1

            if pair in candidate_set3:
                s3_found += 1
            else:
                missed_s3.append(pair)


# ---------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------

print("\n==============================")
print("BLOCKING RECALL RESULTS")
print("==============================")

print("S2 total:", s2_total)
print("S2 found:", s2_found)
print("S2 recall:", round(s2_found / s2_total, 4))

print()

print("S3 total:", s3_total)
print("S3 found:", s3_found)
print("S3 recall:", round(s3_found / s3_total, 4))

overall_total = s2_total + s3_total
overall_found = s2_found + s3_found

print()

print("Overall total:", overall_total)
print("Overall found:", overall_found)
print(
    "Overall recall:",
    round(overall_found / overall_total, 4),
)

print()

print("Missed S2 examples:")
print(missed_s2[:10])

print()

print("Missed S3 examples:")
print(missed_s3[:10])

# ---------------------------------------------------------
# 9. Inspect missed matches
# ---------------------------------------------------------

from rapidfuzz import fuzz


print("\n==============================")
print("MISSED S2 EXAMPLES")
print("==============================")


s1_lookup = s1.set_index("entity_id")
s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")


for source1_id, target_id in missed_s2[:10]:

    s1_row = s1_lookup.loc[source1_id]
    target_row = s2_lookup.loc[target_id]

    name_score = fuzz.ratio(
        str(s1_row["business_name"]),
        str(target_row["business_name"]),
    )

    address_score = fuzz.ratio(
        str(s1_row["business_address"]),
        str(target_row["business_address"]),
    )

    print("\nS1:", source1_id)
    print("S2:", target_id)
    print("S1 name:", s1_row["business_name"])
    print("S2 name:", target_row["business_name"])
    print("S1 address:", s1_row["business_address"])
    print("S2 address:", target_row["business_address"])
    print("Name similarity:", round(name_score, 2))
    print("Address similarity:", round(address_score, 2))


print("\n==============================")
print("MISSED S3 EXAMPLES")
print("==============================")


for source1_id, target_id in missed_s3[:10]:

    s1_row = s1_lookup.loc[source1_id]
    target_row = s3_lookup.loc[target_id]

    name_score = fuzz.ratio(
        str(s1_row["business_name"]),
        str(target_row["business_name"]),
    )

    address_score = fuzz.ratio(
        str(s1_row["business_address"]),
        str(target_row["business_address"]),
    )

    print("\nS1:", source1_id)
    print("S3:", target_id)
    print("S1 name:", s1_row["business_name"])
    print("S3 name:", target_row["business_name"])
    print("S1 address:", s1_row["business_address"])
    print("S3 address:", target_row["business_address"])
    print("Name similarity:", round(name_score, 2))
    print("Address similarity:", round(address_score, 2))