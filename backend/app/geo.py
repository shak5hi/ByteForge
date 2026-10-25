"""Geographic queries over places (PostGIS).

All distances are metres on the WGS84 spheroid (geography type), and every query is
served by the GiST index on places.geom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session, selectinload

from app.models import Place, PlaceAlias
from app.text import normalize

MAX_RADIUS_M = 50_000


@dataclass
class PlaceHit:
    place: Place
    lat: float
    lon: float
    distance_m: Optional[float] = None
    similarity: Optional[float] = None


def point(lat: float, lon: float):
    """A geography point. Note PostGIS order is (lon, lat)."""
    return func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326).cast(Place.geom.type)


def _with_coords(stmt: Select) -> Select:
    g = func.geometry(Place.geom)
    return stmt.add_columns(func.ST_Y(g).label("lat"), func.ST_X(g).label("lon"))


def _filters(stmt: Select, category: Optional[str], landmarks: Optional[bool]) -> Select:
    if category:
        stmt = stmt.where(Place.category == category)
    if landmarks is not None:
        stmt = stmt.where(Place.is_landmark.is_(landmarks))
    return stmt


def get_place(session: Session, place_id: int) -> Optional[PlaceHit]:
    stmt = _with_coords(
        select(Place).options(selectinload(Place.aliases), selectinload(Place.sources)).where(Place.id == place_id)
    )
    row = session.execute(stmt).first()
    return PlaceHit(row.Place, row.lat, row.lon) if row else None


def nearest(
    session: Session,
    lat: float,
    lon: float,
    limit: int = 20,
    category: Optional[str] = None,
    landmarks: Optional[bool] = False,
) -> list[PlaceHit]:
    """The `limit` closest places to a point, however far away (index-assisted KNN)."""
    here = point(lat, lon)
    dist = func.ST_Distance(Place.geom, here).label("distance_m")
    stmt = _filters(_with_coords(select(Place, dist)), category, landmarks)
    stmt = stmt.order_by(Place.geom.op("<->")(here)).limit(limit)
    return [PlaceHit(r.Place, r.lat, r.lon, distance_m=r.distance_m) for r in session.execute(stmt)]


def within_radius(
    session: Session,
    lat: float,
    lon: float,
    radius_m: float,
    limit: int = 50,
    category: Optional[str] = None,
    landmarks: Optional[bool] = False,
) -> list[PlaceHit]:
    """Places within `radius_m` of a point, closest first."""
    radius_m = min(radius_m, MAX_RADIUS_M)
    here = point(lat, lon)
    dist = func.ST_Distance(Place.geom, here).label("distance_m")
    stmt = _filters(_with_coords(select(Place, dist)), category, landmarks)
    stmt = stmt.where(func.ST_DWithin(Place.geom, here, radius_m)).order_by(dist).limit(limit)
    return [PlaceHit(r.Place, r.lat, r.lon, distance_m=r.distance_m) for r in session.execute(stmt)]


LANDMARK_MIN_SCORE = 0.45
LANDMARK_BONUS = 0.1  # prefer a real landmark over a shop that happens to share its name


def resolve_landmark(
    session: Session,
    text: str,
    near: Optional[tuple[float, float]] = None,
    limit: int = 5,
) -> list[PlaceHit]:
    """Places whose name or alias fuzzily matches `text` ("benett univ" -> Bennett University).

    Score = best trigram match over the name and all aliases, using word_similarity so a
    partial name ("Bennett") still matches the full one. Landmarks get a small bonus; with
    `near`, ties are broken by distance. Best first; empty if nothing scores above 0.45.
    """
    q = normalize(text)
    if not q:
        return []

    names = select(Place.id.label("place_id"), Place.name_norm.label("term"))
    aliases = select(PlaceAlias.place_id, PlaceAlias.alias_norm.label("term"))
    terms = names.union_all(aliases).subquery()

    term_score = func.greatest(func.similarity(terms.c.term, q), func.word_similarity(q, terms.c.term))
    best = (
        select(
            terms.c.place_id,
            func.max(term_score).label("score"),
            func.max(func.similarity(terms.c.term, q)).label("exactness"),  # tie-break: "Pari Chowk" over "Pari Chowk Park"
        )
        .where(terms.c.term.op("%>")(q) | terms.c.term.op("%")(q))  # trigram-indexable prefilter
        .group_by(terms.c.place_id)
        .subquery()
    )
    score = (best.c.score + case((Place.is_landmark, LANDMARK_BONUS), else_=0.0)).label("similarity")

    stmt = _with_coords(select(Place, score).join(best, best.c.place_id == Place.id))
    stmt = stmt.where(best.c.score >= LANDMARK_MIN_SCORE)
    order = [score.desc(), best.c.exactness.desc()]
    if near:
        dist = func.ST_Distance(Place.geom, point(*near)).label("distance_m")
        stmt = stmt.add_columns(dist)
        order.append(dist)
    stmt = stmt.order_by(*order).limit(limit)

    return [
        PlaceHit(r.Place, r.lat, r.lon, distance_m=getattr(r, "distance_m", None), similarity=r.similarity)
        for r in session.execute(stmt)
    ]
