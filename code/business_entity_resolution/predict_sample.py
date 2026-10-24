import joblib
import pandas as pd


# =========================================================
# 1. Load feature dataset
# =========================================================

features_path = (
    "code/business_entity_resolution/"
    "training_features_sample.csv"
)

df = pd.read_csv(features_path)

print("Loaded feature rows:", len(df))


# =========================================================
# 2. Load trained model
# =========================================================

model_path = (
    "code/business_entity_resolution/"
    "models/entity_match_model.joblib"
)

threshold_path = (
    "code/business_entity_resolution/"
    "models/threshold.txt"
)

saved = joblib.load(model_path)

model = saved["model"]
feature_columns = saved["feature_columns"]


with open(threshold_path, "r") as f:
    threshold = float(f.read().strip())


print("Loaded threshold:", threshold)


# =========================================================
# 3. Build prediction matrix
# =========================================================

X = df[feature_columns]


# =========================================================
# 4. Predict match probabilities
# =========================================================

probabilities = model.predict_proba(X)[:, 1]

df["match_probability"] = probabilities

df["predicted_match"] = (
    probabilities >= threshold
).astype(int)


# =========================================================
# 5. Save predictions
# =========================================================

output_path = (
    "code/business_entity_resolution/"
    "sample_predictions.csv"
)

df[
    [
        "source",
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
        "predicted_match",
        "label",
    ]
].to_csv(
    output_path,
    index=False,
)


# =========================================================
# 6. Summary
# =========================================================

print()
print("==============================")
print("PREDICTION SUMMARY")
print("==============================")

print("Total candidates:", len(df))

print(
    "Predicted matches:",
    int(df["predicted_match"].sum()),
)

print(
    "Predicted non-matches:",
    int(
        (df["predicted_match"] == 0).sum()
    ),
)

print()
print("Predictions by source:")

print(
    df.groupby("source")[
        "predicted_match"
    ].sum()
)

print()
print("Saved:", output_path)


# =========================================================
# 7. Show strongest predictions
# =========================================================

print()
print("Top predicted matches:")

print(
    df[
        df["predicted_match"] == 1
    ]
    .sort_values(
        "match_probability",
        ascending=False,
    )
    [
        [
            "source",
            "source1_entity_id",
            "candidate_entity_id",
            "match_probability",
            "label",
        ]
    ]
    .head(10)
    .to_string(index=False)
)