# ml/ — AI/ML layer for the ByteForge place-search app

Natural-language place search ("Raja Hotel near Bennett University") built on top of the
entity-resolution model from the ML Challenge 2026 submission in `code/business_entity_resolution/`.

```
ml/
├── api.py                    # the ONLY module the backend imports
├── schemas.py                # Pydantic contracts (ParsedQuery, Place, MatchResult, RankedResult)
├── _legacy.py                # loads the competition normalizer/features/model (single source of truth)
├── entity_resolution/matcher.py   # single-pair scoring with the trained XGBoost model
├── query_parser/             # rule-based parser + category gazetteer
└── ranking/                  # transparent weighted scorer
```

Setup: `pip install -r ml/requirements.txt`, then `pytest` from the repo root.

How the pieces connect:
- `match_entities` reuses the competition's normalization, 16 similarity features and the
  committed model (threshold 0.63). It is how duplicate listings from different data sources get merged.
- `parse_location_query` turns free text into category / name / landmark / relation / radius.
- `rank_results` orders PostGIS candidates by name match, distance decay, category and match confidence.

Interface details: [`docs/ml_interface.md`](../docs/ml_interface.md).

## Results

Everything below is reproducible with `pytest` and `python -m ml.evaluation.run_eval`
(full per-query breakdown and failures in [`docs/evaluation_report.md`](../docs/evaluation_report.md)).

### Query parser

Rule-based: regex, a category gazetteer, and edit-distance matching for typos. Scored on
81 hand-labelled queries, a query counting as correct only if every field matches.

| Query set | Queries | Exact match | How it was built |
|---|---|---|---|
| Core | 44 | 100% | typical queries; the rules were written against them |
| Hard | 10 | 100% | weak spots found by the first evaluation (0% before the fixes), so also tuned |
| Held-out | 27 | **92.6%** (25/27) | written after the fixes, scored once, never tuned on |

The held-out row is the honest estimate for unseen queries. It is a small sample written by the
team, not a public benchmark. Known misses: superlatives like "nearest ATM" (implies "near me")
and brand names containing a category word, such as "Cafe Coffee Day".

What it handles: category synonyms and plurals, typos ("restarant"), landmark-first order
("Bennett University nearby restaurants"), "within 2 km / 1 mile / 500 m", "near me", price and
opening-hours phrases (stripped and reported in `notes`), and phrasing like "where can I eat near X".

### Ranking

Weighted score of name match, distance decay, category match and entity-match confidence, with a
hard filter for "within X km". On 10 hand-built cases (same name at different distances, a named
place versus a closer wrong name, category versus closer wrong category, typo in name, confidence
tie-break) the expected place ranks first in all 10 (MRR 1.0). The candidates are synthetic, so this
shows the scoring behaves sensibly, not how it performs on real listings.

### Entity matching

Uses the XGBoost model from the ML Challenge 2026 submission: 16 name and address similarity
features (RapidFuzz ratios, Jaro-Winkler, token Jaccard, numeric address overlap, country match)
and a decision threshold of 0.63 tuned for F0.5 on a validation split grouped by Source 1 entity.
The validation score is printed by `code/business_entity_resolution/train_model.py` but is not
recorded in this repository; see `docs/resume_notes.md`.

Two things to know when reading this model's behaviour in the app:
- One of its 16 features, `block_name_score`, comes from the competition's blocker. For a single
  pair it is approximated by name-token overlap, so scores can differ slightly from the batch pipeline.
- When both records have coordinates and are more than 500 m apart, the score is halved. This is a
  hand-chosen rule, not something the model learned.
