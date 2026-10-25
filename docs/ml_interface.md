# ML interface (Person 1 → Person 2)

Import only from `ml.api`. All inputs and outputs are plain dicts. Schemas: `ml/schemas.py`.

## `parse_location_query(query: str) -> dict`

```python
parse_location_query("Raja Hotel near Bennett University")
# {"raw": "...", "name": "Raja", "category": "hotel", "category_text": "Hotel",
#  "reference": "Bennett University", "relation": "near", "radius_km": None,
#  "use_user_location": False, "confidence": 1.0, "ambiguous": False, "notes": []}
```

- `category` is canonical and singular; `category_text` is what the user typed.
- `relation`: `near | in | at | within | none`; `radius_km` is set only for `within`.
- `use_user_location=True` for "near me" / "nearby": use the browser location.
- Never raises. Check `ambiguous`, `confidence`, `notes`.

## `match_entities(a: dict, b: dict) -> dict`

Records: `name` (required), optional `address`, `country`, `lat`, `lon`.
Returns `{"score", "is_match", "threshold", "source"}`; `source` is `"model"` (trained XGBoost)
or `"heuristic"` if the model artifact can't be loaded. If both records have coordinates and are
more than 500 m apart, the score is halved.

## `rank_results(parsed_query, candidates, ref_lat=None, ref_lon=None, limit=20) -> list[dict]`

- `candidates`: PostGIS results with `id, name, lat, lon` (+ optional `category, address, phone, website, match_confidence`).
- `ref_lat/ref_lon`: resolved landmark or user location. Without them distance is neutral.
- Returns `[{"id", "score", "distance_km", "components"}]`, best first. `within` drops candidates beyond `radius_km`.

## Expected backend flow

1. `parsed = parse_location_query(q)`
2. Resolve `parsed["reference"]` to coordinates (PostGIS landmark lookup) or use the user location.
3. Query PostGIS for candidates near it (filter by `parsed["category"]` if set).
4. `rank_results(parsed, candidates, lat, lon)` → return to React.
