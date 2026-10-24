import os
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier


# =========================================================
# 1. Load feature dataset
# =========================================================

input_path = (
    "code/business_entity_resolution/"
    "training_features_sample.csv"
)

df = pd.read_csv(input_path)

print("Loaded rows:", len(df))


# =========================================================
# 2. Separate features and labels
# =========================================================

metadata_columns = [
    "source",
    "source1_entity_id",
    "candidate_entity_id",
    "label",
]

feature_columns = [
    c for c in df.columns
    if c not in metadata_columns
]

X = df[feature_columns]
y = df["label"]


# =========================================================
# 3. Train/validation split by Source 1 entity
# =========================================================
# Important:
# We split by source1_entity_id rather than individual pairs.
# This prevents pairs belonging to the same Source 1 entity
# from appearing in both training and validation.

groups = df["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, valid_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]


print()
print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Training positives:", int(y_train.sum()))
print("Validation positives:", int(y_valid.sum()))


# =========================================================
# 4. Train XGBoost model
# =========================================================

positive = int(y_train.sum())
negative = int((y_train == 0).sum())

scale_pos_weight = negative / positive

print()
print("Scale positive weight:", scale_pos_weight)


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


print()
print("Training model...")

model.fit(
    X_train,
    y_train,
)


# =========================================================
# 5. Validation predictions
# =========================================================

valid_prob = model.predict_proba(X_valid)[:, 1]


# =========================================================
# 6. Find threshold for F0.5
# =========================================================

best_threshold = 0.50
best_f05 = 0.0

for threshold in [
    i / 100
    for i in range(5, 96)
]:

    predictions = (
        valid_prob >= threshold
    ).astype(int)

    score = fbeta_score(
        y_valid,
        predictions,
        beta=0.5,
        zero_division=0,
    )

    if score > best_f05:
        best_f05 = score
        best_threshold = threshold


print()
print("==============================")
print("VALIDATION RESULTS")
print("==============================")

print("Best threshold:", best_threshold)
print("Best F0.5:", best_f05)


# =========================================================
# 7. Save model
# =========================================================

model_dir = (
    "code/business_entity_resolution/"
    "models"
)

os.makedirs(
    model_dir,
    exist_ok=True,
)

model_path = (
    model_dir +
    "/entity_match_model.joblib"
)

threshold_path = (
    model_dir +
    "/threshold.txt"
)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
    },
    model_path,
)

with open(
    threshold_path,
    "w",
) as f:
    f.write(str(best_threshold))


# =========================================================
# 8. Final summary
# =========================================================

print()
print("Model saved:", model_path)
print("Threshold saved:", threshold_path)