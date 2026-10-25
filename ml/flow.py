"""Natural-language search flow: parse -> resolve landmark -> fetch candidates -> rank.

The database is injected as three small callables, so this module has no PostGIS or
SQLAlchemy dependency and is fully testable with fake data. The backend only has to write
the three adapters (see docs/ml_integration_guide.md).

Place dicts exchanged with the backend use the backend's own fields
(id, name, lat, lon, category, address, match_confidence, ...). `id` may be an int.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ml.api import parse_location_query, rank_results

DEFAULT_RADIUS_KM = 5.0
FETCH_MULTIPLIER = 3  # fetch more than `limit` so the ranker has something to reorder

Place = dict
LatLon = tuple[float, float]

# resolve_landmark(text, near) -> best matching landmark, or None
ResolveLandmark = Callable[[str, Optional[LatLon]], Optional[Place]]
# find_nearby(lat, lon, radius_km, category, limit) -> places around a point
FindNearby = Callable[[float, float, float, Optional[str], int], list[Place]]
# find_by_name(text, center, radius_km, category, limit) -> places whose name fuzzily matches
FindByName = Callable[[str, Optional[LatLon], Optional[float], Optional[str], int], list[Place]]


@dataclass
class FlowResult:
    status: str  # "ok" | "landmark_not_found" | "needs_user_location"
    parsed: dict
    reference: Optional[Place] = None  # the landmark the query resolved to
    center: Optional[LatLon] = None
    radius_km: Optional[float] = None
    results: list[Place] = field(default_factory=list)  # input fields plus score, distance_km, score_components
    notes: list[str] = field(default_factory=list)


def run_search(
    query: str,
    *,
    resolve_landmark: ResolveLandmark,
    find_nearby: FindNearby,
    find_by_name: FindByName,
    user_location: Optional[LatLon] = None,
    limit: int = 20,
    default_radius_km: float = DEFAULT_RADIUS_KM,
    overrides: Optional[dict] = None,
) -> FlowResult:
    """Answer a free-text place query.

    `overrides` lets explicit API parameters win over what the parser extracted:
    keys `reference`, `category`, `radius_km` (None values are ignored).
    """
    parsed = parse_location_query(query)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        parsed[key] = value
        if key == "reference":
            parsed["use_user_location"] = False
        if key == "radius_km" and parsed["relation"] == "none":
            parsed["relation"] = "within"

    out = FlowResult(status="ok", parsed=parsed, notes=list(parsed["notes"]))

    # 1. Where to search around.
    center: Optional[LatLon] = None
    if parsed["reference"]:
        ref = resolve_landmark(parsed["reference"], user_location)
        if ref is None:
            out.status = "landmark_not_found"
            return out
        out.reference = ref
        center = (ref["lat"], ref["lon"])
    elif parsed["use_user_location"]:
        if user_location is None:
            out.status = "needs_user_location"
            out.notes.append("query is relative to the user's location, which was not provided")
            return out
        center = user_location
    out.center = center
    out.radius_km = (parsed["radius_km"] or default_radius_km) if center else None

    # 2. Candidates.
    name, category = parsed["name"], parsed["category"]
    fetch_n = limit * FETCH_MULTIPLIER
    rank_parsed = parsed
    if name:
        candidates = find_by_name(name, center, out.radius_km, category, fetch_n)
        if not candidates and category:  # the stored category may be missing or different
            candidates = find_by_name(name, center, out.radius_km, None, fetch_n)
            if candidates:
                out.notes.append(f"no '{category}' match; category filter relaxed")
    elif category:
        if center is None:
            out.status = "needs_user_location"
            out.notes.append(f"'{category}' search needs a landmark or the user's location")
            return out
        candidates = find_nearby(center[0], center[1], out.radius_km, category, fetch_n)
    else:
        # Nothing recognised: fall back to a plain name search on the whole query.
        candidates = find_by_name(query, center, out.radius_km, None, fetch_n)
        rank_parsed = {**parsed, "name": query}
        out.notes.append("no category or name recognised; searched the whole query as a name")

    # 3. Rank. The ML schema uses string ids; keep the backend's originals for the response.
    by_id = {str(c["id"]): c for c in candidates}
    ranker_input = [{**c, "id": str(c["id"])} for c in candidates]
    ref_lat, ref_lon = center if center else (None, None)
    ranked = rank_results(rank_parsed, ranker_input, ref_lat, ref_lon, limit)

    for r in ranked:
        item = dict(by_id[r["id"]])
        item["score"] = r["score"]
        if r["distance_km"] is not None:
            item["distance_km"] = r["distance_km"]
        item["score_components"] = r["components"]
        out.results.append(item)
    return out
