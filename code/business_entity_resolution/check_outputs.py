from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("code/business_entity_resolution/outputs")

for source in ["s2", "s3"]:
    path = OUTPUT_DIR / f"test_matches_{source}.csv"

    print("=" * 60)
    print(f"CHECKING {path.name}")
    print("=" * 60)

    total_rows = 0
    unique_s1 = set()

    for chunk in pd.read_csv(
        path,
        dtype=str,
        chunksize=100_000,
    ):
        total_rows += len(chunk)
        unique_s1.update(chunk["source1_entity_id"])

    print(f"Match rows:       {total_rows:,}")
    print(f"Unique S1 IDs:    {len(unique_s1):,}")
    print(f"File size:        {path.stat().st_size:,} bytes")
    print()