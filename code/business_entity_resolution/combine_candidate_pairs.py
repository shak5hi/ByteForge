import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path(r"C:\ByteForgeTemp")

S2 = TEMP / "candidate_pairs_s2.tsv"
S3 = TEMP / "candidate_pairs_s3.tsv"

OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "candidate_pairs.tsv"


def split_ids(value):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


with open(S2, "r", encoding="utf-8", newline="") as f2, \
     open(S3, "r", encoding="utf-8", newline="") as f3, \
     open(OUT, "w", encoding="utf-8", newline="") as fo:

    r2 = csv.DictReader(f2, delimiter="\t")
    r3 = csv.DictReader(f3, delimiter="\t")

    writer = csv.writer(
        fo,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    count = 0

    while True:
        a = next(r2, None)
        b = next(r3, None)

        if a is None and b is None:
            break

        if a is None or b is None:
            raise RuntimeError("S2 and S3 files have different row counts.")

        s1_a = a["source1_entity_id"]
        s1_b = b["source1_entity_id"]

        if s1_a != s1_b:
            raise RuntimeError(
                f"S1 order mismatch: {s1_a} != {s1_b}"
            )

        ids = set(split_ids(a["candidate_entity_ids"]))
        ids.update(split_ids(b["candidate_entity_ids"]))

        writer.writerow([
            s1_a,
            ",".join(sorted(ids))
        ])

        count += 1

        if count % 100000 == 0:
            print(f"Processed {count:,} S1 rows")

print()
print("COMBINATION COMPLETE")
print(f"Rows written: {count:,}")
print(f"Output: {OUT}")
