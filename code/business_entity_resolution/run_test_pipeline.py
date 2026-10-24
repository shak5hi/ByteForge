import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)

from blocking import Blocker
from features import build_feature_frame


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

MODEL_PATH = os.path.join(MODEL_DIR, "entity_match_model.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 25_000


def load_source1():
    return pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
    )


def load_target_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )


def main():
    print("=" * 60)
    print("FULL TEST ENTITY RESOLUTION")
    print("=" * 60)

    print("\nLoading Source 1...")
    source1 = load_source1()

    print("Source 1 rows:", len(source1))

    print("\nLoading Source 2...")
    source2 = load_target_source(S2_PATH)
    print("Source 2 rows:", len(source2))

    print("\nLoading Source 3...")
    source3 = load_target_source(S3_PATH)
    print("Source 3 rows:", len(source3))

    print("\nLoading trained model...")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    print("Threshold:", threshold)
    print("Model type:", type(model).__name__)
    print("Feature columns:", len(feature_columns))

    print("\nPreparing blocking indices...")

    blocker_s2 = Blocker(use_fuzzy=False)
    blocker_s2.fit(source2)

    blocker_s3 = Blocker(use_fuzzy=False)
    blocker_s3.fit(source3)

    print("Blocking indices ready.")

    all_predictions = []

    total_chunks = (len(source1) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_number, start in enumerate(
        range(0, len(source1), CHUNK_SIZE),
        start=1,
    ):
        end = min(start + CHUNK_SIZE, len(source1))

        print(
            f"\nProcessing chunk {chunk_number}/{total_chunks} "
            f"({start:,} - {end:,})"
        )

        source1_chunk = source1.iloc[start:end].copy()

        print("  Generating S2 candidates...")
        candidates_s2 = blocker_s2.generate(source1_chunk)

        print("  S2 candidates:", len(candidates_s2))

        print("  Generating S3 candidates...")
        candidates_s3 = blocker_s3.generate(source1_chunk)

        print("  S3 candidates:", len(candidates_s3))

        predictions = []

        if len(candidates_s2) > 0:
            print("  Building S2 features...")

            X_s2, keys_s2 = build_feature_frame(
                source1_chunk,
                source2,
                candidates_s2,
            )
            X_s2 = X_s2[feature_columns]
            probabilities_s2 = model.predict_proba(X_s2)

            for (source1_id, candidate_id), probability in zip(
                keys_s2,
                probabilities_s2,
            ):
                if probability >= threshold:
                    predictions.append(
                        {
                            "source1_entity_id": source1_id,
                            "candidate_entity_id": candidate_id,
                            "source": "S2",
                            "match_probability": float(probability),
                        }
                    )

        if len(candidates_s3) > 0:
            print("  Building S3 features...")

            X_s3, keys_s3 = build_feature_frame(
                source1_chunk,
                source3,
                candidates_s3,
            )
            X_s3 = X_s3[feature_columns]
            probabilities_s3 = model.predict_proba(X_s3)

            for (source1_id, candidate_id), probability in zip(
                keys_s3,
                probabilities_s3,
            ):
                if probability >= threshold:
                    predictions.append(
                        {
                            "source1_entity_id": source1_id,
                            "candidate_entity_id": candidate_id,
                            "source": "S3",
                            "match_probability": float(probability),
                        }
                    )

        chunk_output = os.path.join(
            OUTPUT_DIR,
            f"predictions_chunk_{chunk_number:04d}.csv",
        )

        pd.DataFrame(predictions).to_csv(
            chunk_output,
            index=False,
        )

        print("  Matches found:", len(predictions))
        print("  Saved:", chunk_output)

    print("\nCombining chunk predictions...")

    files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("predictions_chunk_")
        and f.endswith(".csv")
    ]

    frames = []

    for path in sorted(files):
        df = pd.read_csv(path, dtype=str)

        if len(df) > 0:
            frames.append(df)

    if frames:
        predictions = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        predictions = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "source",
                "match_probability",
            ]
        )

    predictions = predictions.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv",
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("TEST PREDICTION SUMMARY")
    print("=" * 60)

    print("Total predicted matches:", len(predictions))

    if len(predictions):
        print("\nMatches by source:")
        print(predictions["source"].value_counts())

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()