# Resume notes (ML layer)

Draft bullets using only numbers that can be reproduced from this repo. Edit freely, but keep each
claim checkable: an interviewer may ask how it was measured.

## Draft bullets

- Built the ML layer of a natural-language place-search app in Python: a query parser, an entity
  matcher and a ranker, exposed through a three-function API with Pydantic schemas for a FastAPI backend.
- Wrote a rule-based query parser (regex, category gazetteer, edit-distance typo matching) that turns
  queries like "Raja Hotel near Bennett University" into name, category, landmark, relation and radius;
  92.6% exact match (25/27) on held-out queries and 100% on 54 development queries.
- Built an evaluation harness (81 labelled parser queries, 10 ranking cases, 19 regression tests) that
  surfaced a plural-handling bug and took the hard tier from 0/10 to 10/10; reproducible with one command.
- Reused an XGBoost entity-resolution model (16 RapidFuzz name/address features, F0.5-tuned threshold)
  to merge duplicate business listings across sources. [ADD: validation F0.5 = ____ , see below]
- Designed a transparent ranking score (name match, distance decay, category, match confidence) with
  radius filtering; verified on 10 hand-built cases.

## Numbers still missing

The competition's validation F0.5 is **not recorded anywhere in this repository**, so it is not
quoted above. To get it:

1. Download the challenge dataset into `dataset/` (see the main README).
2. Run `python code/business_entity_resolution/train_model.py` and copy the "Best F0.5" line.
3. If the team kept a competition score or leaderboard result from an earlier run, use that instead.

Be precise about which metric you quote:
- `train_model.py` prints a **pair-level** F0.5 over candidate pairs.
- The competition scored **macro F0.5**, averaged per Source 1 entity (see `src/metrics.py`).
These are different numbers; do not present one as the other.

## Say honestly in interviews

- The parser's held-out set is 27 queries written by the team, not a public benchmark.
- Ranking was checked on synthetic candidates, not real listings.
- The app's backend and map were built by a teammate; say which parts are yours.
