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
