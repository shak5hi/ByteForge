"""Load businesses and landmarks for an area from OpenStreetMap (Overpass API).

    cd backend
    python -m scripts.import_osm                       # default: Greater Noida
    python -m scripts.import_osm --bbox 28.40,77.40,28.56,77.62

Re-running is safe: places are upserted by their OSM id (place_sources.source_id).
Categories use the same canonical names as the ML query parser (ml/query_parser).
Map data (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import argparse
import sys

import httpx
from geoalchemy2.elements import WKTElement
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Place, PlaceAlias, PlaceSource
from app.text import normalize

# Public Overpass mirrors, tried in order (any one can be down or blocked on a network).
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
GREATER_NOIDA_BBOX = (28.40, 77.40, 28.56, 77.62)  # south, west, north, east

# (OSM key, OSM value) -> canonical category
BUSINESS_TAGS: dict[tuple[str, str], str] = {
    ("amenity", "restaurant"): "restaurant",
    ("amenity", "fast_food"): "restaurant",
    ("amenity", "food_court"): "restaurant",
    ("tourism", "hotel"): "hotel",
    ("tourism", "guest_house"): "hotel",
    ("tourism", "hostel"): "hotel",
    ("tourism", "motel"): "hotel",
    ("amenity", "cafe"): "cafe",
    ("amenity", "hospital"): "hospital",
    ("amenity", "clinic"): "hospital",
    ("amenity", "doctors"): "hospital",
    ("amenity", "pharmacy"): "pharmacy",
    ("shop", "chemist"): "pharmacy",
    ("amenity", "atm"): "atm",
    ("amenity", "bank"): "bank",
    ("amenity", "school"): "school",
    ("leisure", "fitness_centre"): "gym",
    ("shop", "supermarket"): "supermarket",
    ("shop", "convenience"): "supermarket",
    ("shop", "grocery"): "supermarket",
    ("amenity", "fuel"): "petrol pump",
}

# Places people search *relative to* ("near Bennett University", "in Pari Chowk").
LANDMARK_TAGS: dict[tuple[str, str], str] = {
    ("amenity", "university"): "university",
    ("amenity", "college"): "college",
    ("shop", "mall"): "mall",
    ("railway", "station"): "station",
    ("station", "subway"): "station",
    ("tourism", "attraction"): "attraction",
    ("leisure", "park"): "park",
    ("leisure", "stadium"): "stadium",
    ("place", "suburb"): "area",
    ("place", "neighbourhood"): "area",
    ("place", "quarter"): "area",
    ("place", "village"): "area",
    ("place", "town"): "area",
    ("place", "city"): "area",
    ("junction", "yes"): "junction",
    ("highway", "traffic_signals"): "junction",
}

ALIAS_KEYS = {
    "alt_name": "alt_name",
    "short_name": "short_name",
    "old_name": "old_name",
    "official_name": "official_name",
    "name:en": "name_en",
    "name:hi": "name_hi",
}


def build_query(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    selectors = "".join(
        f'nwr["{k}"="{v}"]["name"]({s},{w},{n},{e});'
        for k, v in [*BUSINESS_TAGS, *LANDMARK_TAGS]
    )
    return f"[out:json][timeout:180];({selectors});out center tags;"


def fetch(bbox: tuple[float, float, float, float], urls: list[str]) -> list[dict]:
    query = build_query(bbox)
    for url in urls:
        try:
            resp = httpx.post(url, data={"data": query}, timeout=httpx.Timeout(240, connect=20),
                              headers={"User-Agent": "ByteForge-place-search/0.1"})
            resp.raise_for_status()
            return resp.json()["elements"]
        except (httpx.HTTPError, ValueError) as exc:
            print(f"  {url} failed: {exc!r}", file=sys.stderr)
    raise SystemExit("All Overpass mirrors failed; try again later or pass --overpass-url")


def classify(tags: dict) -> tuple[str, bool] | None:
    """(category, is_landmark) for an OSM element, or None if we don't keep it."""
    for (k, v), cat in BUSINESS_TAGS.items():
        if tags.get(k) == v:
            return cat, False
    for (k, v), cat in LANDMARK_TAGS.items():
        if tags.get(k) == v:
            return cat, True
    return None


def coords(el: dict) -> tuple[float, float] | None:
    if "lat" in el:
        return el["lat"], el["lon"]
    if "center" in el:
        return el["center"]["lat"], el["center"]["lon"]
    return None


def address(tags: dict) -> str | None:
    if tags.get("addr:full"):
        return tags["addr:full"]
    parts = [
        " ".join(p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:postcode"),
    ]
    return ", ".join(p for p in parts if p) or None


def upsert(session: Session, el: dict) -> str | None:
    tags = el.get("tags", {})
    kind = classify(tags)
    point = coords(el)
    if not kind or not point or not tags.get("name"):
        return None

    category, is_landmark = kind
    lat, lon = point
    source_id = f"{el['type']}/{el['id']}"

    existing = session.scalar(
        select(PlaceSource).where(PlaceSource.source == "osm", PlaceSource.source_id == source_id)
    )
    place = existing.place if existing else Place()

    place.name = tags["name"]
    place.name_norm = normalize(tags["name"])
    place.category = category
    place.is_landmark = is_landmark
    place.address = address(tags)
    place.city = tags.get("addr:city")
    place.country = "India" if tags.get("addr:country", "IN") == "IN" else tags.get("addr:country")
    place.phone = tags.get("phone") or tags.get("contact:phone")
    place.website = tags.get("website") or tags.get("contact:website")
    place.geom = WKTElement(f"POINT({lon} {lat})", srid=4326)

    seen = {a.alias_norm for a in place.aliases} | {place.name_norm}
    for key, alias_kind in ALIAS_KEYS.items():
        for alias in filter(None, (a.strip() for a in tags.get(key, "").split(";"))):
            norm = normalize(alias)
            if norm and norm not in seen:
                place.aliases.append(PlaceAlias(alias=alias, alias_norm=norm, kind=alias_kind))
                seen.add(norm)

    if existing:
        existing.raw = tags
        return "updated"
    place.sources.append(PlaceSource(source="osm", source_id=source_id, raw=tags))
    session.add(place)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", default=",".join(map(str, GREATER_NOIDA_BBOX)),
                        help="south,west,north,east (default: Greater Noida)")
    parser.add_argument("--overpass-url", help="use this Overpass endpoint instead of the built-in mirrors")
    args = parser.parse_args()
    bbox = tuple(float(x) for x in args.bbox.split(","))
    if len(bbox) != 4:
        parser.error("--bbox needs 4 numbers: south,west,north,east")

    print(f"Fetching OSM data for bbox {bbox} ...")
    elements = fetch(bbox, [args.overpass_url] if args.overpass_url else OVERPASS_URLS)
    print(f"Received {len(elements):,} elements")

    counts = {"created": 0, "updated": 0, "skipped": 0}
    with SessionLocal() as session:
        for i, el in enumerate(elements, 1):
            counts[upsert(session, el) or "skipped"] += 1
            if i % 500 == 0:
                session.commit()
        session.commit()

    print(f"Done: {counts['created']:,} created, {counts['updated']:,} updated, {counts['skipped']:,} skipped")


if __name__ == "__main__":
    main()
