# Connecting the ML layer to `/search`

For the backend. About an hour of work. Only `backend/app/search.py`, the `/search` route and
the response model change. `geo.py`, `models.py` and the database are untouched.

## What you get

One free-text box. `GET /search?q=hotels near Bennett university` (typos fine) is parsed into
category, landmark, radius and name; the landmark is resolved with your `geo.resolve_landmark`;
candidates come from your `geo` queries; and the ML ranker orders them.

```
q ──► parse_location_query ──► resolve landmark (geo.resolve_landmark)
                                       │
            name given? ── yes ──► geo.search_by_name  ┐
                        └─ no ───► geo.within_radius   ┴─► rank_results ──► response
```

The orchestration is already written and tested in [`ml/flow.py`](../ml/flow.py) with fake
data, so you do not touch any ML code. You write three small adapters that wrap your `geo`
functions.

## Step 1: dependencies

`ml.api` needs these extra packages in the backend environment (same pins as the competition code):

```
pandas==2.2.3
numpy==2.1.3
rapidfuzz==3.10.1
joblib==1.4.2
```

`xgboost` and `scikit-learn` are **not** needed for search. They are only used by
`match_entities` (duplicate detection). If they are missing, `match_entities` quietly falls back
to a simple heuristic and reports `"source": "heuristic"`.

## Step 2: make `ml` importable

`ml/` lives at the repository root, `app/` lives in `backend/`.

- Local: from `backend/`, run with `PYTHONPATH=..` , for example
  `PYTHONPATH=.. python -m uvicorn app.main:app --reload --port 8000`.
- Docker: the image has to contain `ml/` **and** `code/business_entity_resolution/` (its `src/`
  and `models/`), because the ML layer reuses the competition normalizer from there. The folder
  layout inside the image matters: `ml/` and `code/` must sit side by side. Your build context is
  already the repo root and `.dockerignore` does not exclude them, so only `backend/Dockerfile`
  changes:

```dockerfile
# backend/Dockerfile: add PYTHONPATH to the existing ENV block ...
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# ... and, right after the existing `COPY backend/ .` line:
COPY ml/ /app/ml/
COPY code/business_entity_resolution/src/ /app/code/business_entity_resolution/src/
COPY code/business_entity_resolution/models/ /app/code/business_entity_resolution/models/
```

  `docker-compose.yml` needs no change. Add the four packages from Step 1 to
  `backend/requirements.txt`. Expect the image to grow by roughly the size of pandas and numpy.
  I ran the ML flow in exactly this layout (`/app/backend`, `/app/ml`,
  `/app/code/business_entity_resolution`) with `PYTHONPATH=/app` and without xgboost or
  scikit-learn, and it worked; I could not build the image itself, so check the build once.

## Step 3: replace `backend/app/search.py`

```python
# backend/app/search.py  (replace the whole file)
"""Search service: free-text query -> ranked places, using the ML layer (ml/flow.py)."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app import geo
from app.schemas import PlaceOut
from ml.flow import DEFAULT_RADIUS_KM, FlowResult, run_search

DEFAULT_NEAR_RADIUS_KM = DEFAULT_RADIUS_KM  # name the router already uses


def _as_dict(hit: geo.PlaceHit) -> dict:
    return PlaceOut.from_hit(hit).model_dump()


def search(
    session: Session,
    q: str,
    user_location: Optional[tuple[float, float]] = None,
    near: Optional[str] = None,
    radius_km: Optional[float] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> FlowResult:
    def resolve_landmark(text, near_point):
        hits = geo.resolve_landmark(session, text, near=near_point, limit=1)
        return _as_dict(hits[0]) if hits else None

    def find_nearby(lat, lon, radius, cat, n):
        hits = geo.within_radius(session, lat, lon, radius * 1000, limit=n, category=cat, landmarks=False)
        return [_as_dict(h) for h in hits]

    def find_by_name(text, center, radius, cat, n):
        hits = geo.search_by_name(
            session, text, near=center, radius_m=radius * 1000 if center and radius else None,
            category=cat, limit=n,
        )
        return [_as_dict(h) for h in hits]

    return run_search(
        q,
        resolve_landmark=resolve_landmark,
        find_nearby=find_nearby,
        find_by_name=find_by_name,
        user_location=user_location,
        limit=limit,
        # explicit query parameters still win over what the parser guessed
        overrides={"reference": near, "category": category, "radius_km": radius_km},
    )
```

## Step 4: update the `/search` route and `SearchResponse`

In `schemas.py`, add three fields to `SearchResponse`:

```python
class SearchResponse(BaseModel):
    query: str
    status: str = "ok"                       # "ok" | "needs_user_location"
    parsed: Optional[dict] = None            # what the parser understood (useful for the UI)
    notes: list[str] = []                    # e.g. "ignored unsupported filter: 'under 2000'"
    reference: Optional[PlaceOut] = None
    center: Optional[Point] = None
    radius_km: Optional[float] = None
    count: int
    results: list[PlaceOut]
```

In `routers/search.py`, replace the body of `search(...)` after the lat/lon check, and update the
`q` description (`'Free text, e.g. "hotels near Bennett University"'`):

```python
    user_location = (lat, lon) if lat is not None else None
    result = search_service.search(session, q, user_location, near, radius_km, category, limit)

    if result.status == "landmark_not_found":
        raise HTTPException(status_code=404, detail=f"No landmark matching {result.parsed['reference']!r}")

    return SearchResponse(
        query=q,
        status=result.status,
        parsed=result.parsed,
        notes=result.notes,
        reference=PlaceOut(**result.reference) if result.reference else None,
        center=Point(lat=result.center[0], lon=result.center[1]) if result.center else None,
        radius_km=result.radius_km,
        count=len(result.results),
        results=[PlaceOut(**r) for r in result.results],
    )
```

Remove `LandmarkNotFound` (no longer used).

## Behaviour to know about

| Situation | Response |
|---|---|
| "hotels near Bennett University" | `status: "ok"`, results ranked by name match, distance, category, confidence |
| Landmark not found | 404, as before |
| "pharmacy near me" and no `lat`/`lon` | **200** with `status: "needs_user_location"` and no results. The frontend should ask the browser for its location and repeat the request with `lat`/`lon` |
| "hotels" alone (no landmark, no location) | same `needs_user_location` status |
| "Raja Hotel" but stored category is missing | retried without the category filter; `notes` says so |
| "hotels under 2000 near X" | price is ignored and listed in `notes` |
| Explicit `near=`, `category=`, `radius_km=` | override whatever the parser found |

`score` in each result is now the overall relevance score (0 to 1), and `score_components`
(name, distance, category, confidence) is in the raw dict if you want to expose it.

## How to check it works

With the database loaded and the API running (`docker compose up -d --build`, then
`docker compose run --rm api python -m scripts.import_osm`). First confirm the ML layer is inside
the image: `docker compose run --rm api python -c "import ml.flow; print('ml ok')"`. Then:

```bash
curl "localhost:8000/search?q=hotels%20near%20bennett%20univ"
curl "localhost:8000/search?q=cafes%20within%202%20km%20of%20pari%20chowk"
curl "localhost:8000/search?q=pharmacy%20near%20me&lat=28.4506&lon=77.5838"
curl "localhost:8000/search?q=pharmacy%20near%20me"      # expect status needs_user_location
curl "localhost:8000/search?q=restarant%20near%20pari%20chowk"   # typo
```

Backend tests: monkeypatch `geo.resolve_landmark`, `geo.within_radius` and `geo.search_by_name`
with fakes and call `search_service.search`. `tests/test_flow.py` in the repository root shows
the same idea one level down and runs without PostGIS.

## Questions to ask the ML side

- A query the parser gets wrong: send the exact text; it is added to `ml/evaluation/parser_queries.jsonl`.
- A different ranking order: weights are in `ml/ranking/scorer.py`.
