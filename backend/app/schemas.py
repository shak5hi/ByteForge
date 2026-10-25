"""API response models.

PlaceOut's core fields (id, name, lat, lon, category, address, phone, website,
match_confidence) match ml.schemas.Place, so search candidates can be passed to the
ML ranker as-is.
"""
from typing import Optional

from pydantic import BaseModel

from app.geo import PlaceHit


class PlaceOut(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    category: Optional[str] = None
    is_landmark: bool = False
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    match_confidence: float = 1.0
    distance_km: Optional[float] = None  # from the search/nearby reference point, if any
    score: Optional[float] = None  # name-match score (0-1), if the result came from a text search

    @classmethod
    def from_hit(cls, hit: PlaceHit) -> "PlaceOut":
        p = hit.place
        return cls(
            id=p.id,
            name=p.name,
            lat=hit.lat,
            lon=hit.lon,
            category=p.category,
            is_landmark=p.is_landmark,
            address=p.address,
            city=p.city,
            country=p.country,
            phone=p.phone,
            website=p.website,
            match_confidence=p.match_confidence,
            distance_km=None if hit.distance_m is None else round(hit.distance_m / 1000, 3),
            score=None if hit.similarity is None else round(hit.similarity, 3),
        )


class AliasOut(BaseModel):
    alias: str
    kind: str


class SourceOut(BaseModel):
    source: str
    source_id: str
    match_score: Optional[float] = None


class PlaceDetail(PlaceOut):
    aliases: list[AliasOut] = []
    sources: list[SourceOut] = []

    @classmethod
    def from_hit(cls, hit: PlaceHit) -> "PlaceDetail":
        base = PlaceOut.from_hit(hit).model_dump()
        return cls(
            **base,
            aliases=[AliasOut(alias=a.alias, kind=a.kind) for a in hit.place.aliases],
            sources=[SourceOut(source=s.source, source_id=s.source_id, match_score=s.match_score)
                     for s in hit.place.sources],
        )


class Point(BaseModel):
    lat: float
    lon: float


class NearbyResponse(BaseModel):
    center: Point
    radius_km: Optional[float] = None
    count: int
    results: list[PlaceOut]
