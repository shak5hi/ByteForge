# ByteForge backend: place search API

FastAPI + PostgreSQL/PostGIS service behind the place-search app. It stores businesses
and landmarks with their coordinates and answers "what's near here", "find X near
landmark Y" and "tell me about this place".

```
React app ──► FastAPI (app/routers) ──► search service (app/search.py) ──► geo queries (app/geo.py) ──► PostGIS
                                              │
                                              └─ later: ml.api parse_location_query / rank_results
```

## Run it locally

Needs Docker and Python 3.11 (the same environment as the ML code, so the ML layer can
be imported later).

```bash
# from the repo root
docker compose up -d db                              # PostGIS 16 on localhost:5432

cd backend
cp .env.example .env
pip install -r requirements.txt
python -m alembic upgrade head                       # create extensions, tables, indexes
python -m scripts.import_osm                         # load Greater Noida from OpenStreetMap
python -m uvicorn app.main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

To load a different area, pass a bounding box (south,west,north,east):
`python -m scripts.import_osm --bbox 28.50,77.30,28.65,77.45`. Re-running updates
existing places instead of duplicating them.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/search?q=&near=&lat=&lon=&radius_km=&category=&limit=` | Businesses by fuzzy name, optionally around a landmark (`near`, typos OK, default 5 km) or the user's location. 404 if `near` matches nothing |
| GET | `/nearby?lat=&lon=&radius_km=&category=&include_landmarks=&limit=` | Closest places to a point; everything within `radius_km` if given (max 50 km) |
| GET | `/places/{id}` | Full details, with aliases and source records |
| GET | `/landmarks/resolve?q=` | Best-matching landmarks for free text (autocomplete) |
| GET | `/health` | Liveness, place count and PostGIS version |

```bash
curl "localhost:8000/search?q=stanza&near=pari%20chok&radius_km=3"
curl "localhost:8000/nearby?lat=28.4506&lon=77.5838&category=hotel&radius_km=5"
```

Each result has `id, name, lat, lon, category, address, phone, website,
match_confidence`, which are the fields of `ml.schemas.Place`, plus `distance_km` from
the reference point and a 0–1 name-match `score` for text searches.

Categories use the ML query parser's canonical names: `restaurant`, `hotel`, `cafe`,
`hospital`, `pharmacy`, `atm`, `bank`, `school`, `gym`, `supermarket`, `petrol pump`.
Landmarks use `university`, `college`, `mall`, `station`, `park`, `area`, `junction`, …

## Database

| Table | Holds |
|---|---|
| `places` | One business or landmark: name, category, address, contact, `geom geography(Point,4326)`, `match_confidence`, `is_landmark` |
| `place_aliases` | Other names for a place ("BU", Hindi names, old names), used by landmark lookup |
| `place_sources` | The original records a place came from (`osm` + `node/123`, or `source2` + `S2-…`) with the entity-resolution `match_score` that linked them |

Indexes: GiST on `geom` (radius and nearest-neighbour queries), trigram GIN on
`name_norm` / `alias_norm` (fuzzy name search), plus `category` and a partial index on
landmarks. Distances are metres on the real Earth (geography), not degrees.

Schema changes go through Alembic: edit `app/models.py`, then
`python -m alembic revision --autogenerate -m "..."` and review the generated file.

## Layout

```
backend/
├── app/
│   ├── main.py          FastAPI app, CORS, /health
│   ├── routers/         HTTP layer: places.py (/places, /nearby), search.py (/search, /landmarks)
│   ├── search.py        Search service (the ML integration point)
│   ├── geo.py           PostGIS queries: nearest, within_radius, resolve_landmark, search_by_name
│   ├── models.py        SQLAlchemy tables
│   ├── schemas.py       API response models
│   ├── text.py          Name normalisation for fuzzy matching
│   ├── config.py, db.py Settings (backend/.env) and DB session
├── alembic/             Migrations
└── scripts/import_osm.py
```

## Data

Places come from **OpenStreetMap** (© OpenStreetMap contributors, ODbL), because the
competition dataset has names and addresses but no coordinates. OSM coverage of Greater
Noida is thin (about 240 named places), so expect gaps.

## Not done yet

- ML integration: `parse_location_query` in `/search` so a single `q` like
  "hotels near Bennett University" works, and `rank_results` for ordering
- Automated tests, Docker image for the API, deployment
