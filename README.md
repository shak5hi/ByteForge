# ByteForge — Business Entity Resolution

Team **ByteForge**'s solution to the *ML Challenge 2026: Business Entity Resolution* problem.

Given business records from three independent, noisy sources, find every record in
**Source 2** and **Source 3** that refers to the same real-world business as each
**Source 1** (reference) record. The records have no shared IDs, so matches have to be
found from noisy names and addresses alone: abbreviations, typos, transliterations,
missing address parts and landmark-based addresses ("Near SBI ATM").

Full problem statement: [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md)

---

## Approach

```
 Source 1 ─┐
           ├─► Normalize ─► Blocking ─► Pair features ─► XGBoost ─► Threshold ─► Matches
 Source 2/3┘   (unidecode,   (multi-key   (fuzzy name /    classifier   (tuned for
               lowercase,    candidate    address sim.)                 macro F0.5)
               abbrevs)      generation)
```

1. **Normalization.** Unicode transliteration (`unidecode`) so non-Latin names can be
   compared, lowercasing, punctuation stripping, and canonical forms for legal suffixes
   and address abbreviations (`Road → rd`, `Pvt / Private`, `& → and`).
2. **Blocking (candidate generation).** Comparing every pair is impossible at this scale
   (1.7M+ Source 1 test records). Each record gets several blocking keys: exact
   normalized name, first name token, name prefix, first address token and address
   tokens. Only records that share a key are compared, with at most 20 candidates per
   key. On the test set the key lookups run in **DuckDB**, so memory use stays bounded.
3. **Pair features.** 16 similarity features per candidate pair, from `rapidfuzz`: name
   and address ratio, token-set / token-sort ratio, Jaro-Winkler, token Jaccard,
   overlap of numeric address tokens, name length ratio, country match, and
   name × address interactions.
4. **Classifier.** An XGBoost binary classifier, trained on labelled candidate pairs with
   a train/validation split grouped by Source 1 entity so no entity appears in both.
5. **Threshold.** The decision threshold is tuned for the competition metric,
   **macro F0.5**, which weights precision over recall. The current model uses
   `0.63` (`models/threshold.txt`).

`country` is treated as an open set of labels. Nothing is hard-coded to `US` / `India`,
so the unseen `France` records in the test set go through the same pipeline.

---

## Repository layout

```
ByteForge/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── preprocess.py           # text normalization
│   │   ├── blocking.py             # blocking keys + Blocker
│   │   ├── features.py             # pair similarity features
│   │   ├── model.py                # XGBoost wrapper
│   │   ├── metrics.py              # macro F0.5
│   │   ├── config.py               # paths and hyperparameters
│   │   └── pipeline.py             # original end-to-end S1→S2 pipeline
│   ├── build_training_pairs.py     # labelled candidate pairs from ground truth
│   ├── build_features.py           # features for training pairs
│   ├── train_model.py              # train XGBoost + tune threshold
│   ├── test_blocking_recall.py     # measure blocking recall on labelled data
│   ├── predict_sample.py           # sanity-check predictions on the sample
│   ├── run_test_pipeline_chunked.py# full test-set inference (DuckDB, chunked)
│   ├── run_test_pipeline.py        # simpler in-memory inference (small runs)
│   ├── build_candidate_pairs.py    # export output/candidate_pairs.tsv
│   ├── combine_candidate_pairs.py  # merge per-source candidate files
│   ├── build_combined_matches.py   # merge S2 + S3 predictions
│   ├── build_submission.py         # one row per Source 1 test entity
│   ├── validate_submission.py      # format / ID / row-count checks
│   ├── check_outputs.py            # quick stats on match files
│   ├── models/                     # trained model + threshold
│   └── requirements.txt
├── docs/PROBLEM_STATEMENT.md
├── Documentation_template.md       # solution write-up
├── utils/validate_submission.py    # official submission validator
└── validate_submission.py
```

---

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/shak5hi/ByteForge.git
cd ByteForge
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

### Dataset

The dataset (~2.4 GB) is **not** in this repo. Put the challenge files here:

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

All files are **tab-separated**. Always read them with `sep="\t"`.

---

## Usage

Run every command from the **repository root**.

### 1. Train

```bash
python code/business_entity_resolution/test_blocking_recall.py   # optional: check blocking recall
python code/business_entity_resolution/build_training_pairs.py
python code/business_entity_resolution/build_features.py
python code/business_entity_resolution/train_model.py
```

This writes `models/entity_match_model.joblib` and `models/threshold.txt`. A trained
model is already committed, so you can skip straight to inference.

### 2. Run inference on the test set

```bash
python code/business_entity_resolution/run_test_pipeline_chunked.py --source both
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `S2` | `S2`, `S3`, or `both` |
| `--s1-rows` | all | limit Source 1 rows (quick test runs) |
| `--target-rows` | all | limit Source 2/3 rows |
| `--s1-chunk-size` | 5000 | Source 1 records per batch |
| `--target-chunk-size` | 100000 | rows per load into DuckDB |

The output goes to `code/business_entity_resolution/outputs/test_matches_s{2,3}.csv`.

### 3. Build and validate the submission

```bash
python code/business_entity_resolution/build_combined_matches.py
python code/business_entity_resolution/build_submission.py
python code/business_entity_resolution/validate_submission.py
python code/business_entity_resolution/build_candidate_pairs.py   # output/candidate_pairs.tsv
```

> **Note:** the DuckDB scripts write temporary databases to `C:/ByteForgeTemp`. On
> macOS/Linux, change `DB_DIR` / `TEMP_DIR` at the top of those scripts. The temporary
> files can take tens of GB on the full test set.

---

## Evaluation

The score is **macro F0.5**: F0.5 is computed for each Source 1 entity, then averaged
(see `src/metrics.py`). Because F0.5 favours precision, a wrong merge costs more than a
missed match, and the classification threshold is tuned with that in mind.

---

## Place-search ML layer

Beyond the challenge solution, `ml/` adds a natural-language place-search layer for an app
(query parser, entity matcher, ranker) behind a three-function API in `ml/api.py`.
See [`ml/README.md`](ml/README.md) for the design and results, and
[`docs/ml_interface.md`](docs/ml_interface.md) for the backend contract.

```bash
pip install -r ml/requirements.txt
pytest                              # 19 tests
python -m ml.evaluation.run_eval    # parser and ranking evaluation
```

---

## Team

| Member | GitHub |
|---|---|
| Manaswita Shrivastav | [@Manz203](https://github.com/Manz203) |
| Aadya Gupta | [@aadya197](https://github.com/aadya197) |
| Shakshi Sharma | [@shak5hi](https://github.com/shak5hi) |
