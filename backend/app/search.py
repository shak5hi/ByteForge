"""Search service: turns a search request into ranked places.

Current flow (no ML yet):
    1. reference point = resolved `near` landmark, else the user's lat/lon, else none
    2. fuzzy name search, limited to the radius around the reference point if there is one

When Person 1's ML layer is integrated, this is the only module that changes:
    parse_location_query(q) supplies name/category/reference/radius, and
    rank_results() orders the candidates (see docs/ml_interface.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app import geo

DEFAULT_NEAR_RADIUS_KM = 5.0


class LandmarkNotFound(Exception):
    def __init__(self, text: str):
        super().__init__(f"No landmark matching {text!r}")
        self.text = text


@dataclass
class SearchResult:
    hits: list[geo.PlaceHit]
    reference: Optional[geo.PlaceHit]  # the landmark `near` resolved to, if any
    center: Optional[tuple[float, float]]
    radius_km: Optional[float]


def search(
    session: Session,
    q: str,
    near: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: Optional[float] = None,
    category: Optional[str] = None,
    limit: int = 20,
) -> SearchResult:
    reference = None
    center = (lat, lon) if lat is not None and lon is not None else None

    if near:
        matches = geo.resolve_landmark(session, near, near=center, limit=1)
        if not matches:
            raise LandmarkNotFound(near)
        reference = matches[0]
        center = (reference.lat, reference.lon)
        radius_km = radius_km or DEFAULT_NEAR_RADIUS_KM

    hits = geo.search_by_name(
        session,
        q,
        near=center,
        radius_m=radius_km * 1000 if radius_km and center else None,
        category=category,
        limit=limit,
    )
    return SearchResult(hits=hits, reference=reference, center=center, radius_km=radius_km if center else None)
