"""Pydantic schemas: the contract between the ML layer (Person 1) and the backend (Person 2)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Relation = Literal["near", "in", "at", "within", "none"]


class ParsedQuery(BaseModel):
    raw: str
    name: Optional[str] = None            # business name, e.g. "Raja"
    category: Optional[str] = None        # canonical singular, e.g. "hotel"
    category_text: Optional[str] = None   # category as typed, e.g. "hotels"
    reference: Optional[str] = None       # landmark to resolve, e.g. "Bennett University"
    relation: Relation = "none"
    radius_km: Optional[float] = None     # set for "within 2 km of ..."
    use_user_location: bool = False       # "near me" / "nearby" with no landmark
    confidence: float = 0.0
    ambiguous: bool = False
    notes: list[str] = Field(default_factory=list)


class Place(BaseModel):
    """A business record. Used for matching and as a ranking candidate."""
    id: str
    name: str
    lat: float
    lon: float
    category: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    match_confidence: float = 1.0         # entity-resolution confidence, if known


class MatchResult(BaseModel):
    score: float
    is_match: bool
    threshold: float
    source: Literal["model", "heuristic"]


class RankedResult(BaseModel):
    id: str
    score: float
    distance_km: Optional[float] = None
    components: dict[str, float] = Field(default_factory=dict)
