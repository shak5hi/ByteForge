"""Public interface of the ML layer. Backend code should import ONLY from here.

All functions take and return plain dicts / lists (JSON-friendly); schemas are in ml/schemas.py.
See docs/ml_interface.md for examples.
"""
from __future__ import annotations

from typing import Optional

from ml.entity_resolution.matcher import match_pair
from ml.query_parser import parse_query
from ml.ranking import rank
from ml.schemas import ParsedQuery, Place


def parse_location_query(query: str) -> dict:
    """"Raja Hotel near Bennett University" ->
    {"name": "Raja", "category": "hotel", "reference": "Bennett University", "relation": "near", ...}
    Never raises on odd input; check `ambiguous`, `confidence`, `notes`.
    """
    return parse_query(query).model_dump()


def match_entities(a: dict, b: dict) -> dict:
    """Do two business records describe the same real-world place?
    Each record: name, optional address / country / lat / lon.
    Returns {"score", "is_match", "threshold", "source"}.
    """
    return match_pair(a, b).model_dump()


def rank_results(
    parsed_query: dict,
    candidates: list[dict],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    limit: int = 20,
) -> list[dict]:
    """Order candidate places for a parsed query. ref_lat/lon = resolved landmark or user location."""
    parsed = ParsedQuery(**parsed_query)
    places = [Place(**c) for c in candidates]
    return [r.model_dump() for r in rank(parsed, places, ref_lat, ref_lon, limit)]


__all__ = ["parse_location_query", "match_entities", "rank_results"]
