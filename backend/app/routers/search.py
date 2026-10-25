from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import geo, search as search_service
from app.db import get_session
from app.schemas import PlaceOut, Point, SearchResponse

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=200, description='Business name, e.g. "Radisson" or "stanza living"'),
    near: Optional[str] = Query(None, max_length=200, description='Landmark to search around, e.g. "Bennett University"'),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="User latitude (used when `near` is not given)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="User longitude"),
    radius_km: Optional[float] = Query(None, gt=0, le=geo.MAX_RADIUS_M / 1000,
                                       description=f"Search radius around the reference point "
                                                   f"(default {search_service.DEFAULT_NEAR_RADIUS_KM:g} km with `near`)"),
    category: Optional[str] = Query(None, description='Canonical category, e.g. "hotel"'),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> SearchResponse:
    """Find businesses by name, optionally around a landmark (`near`) or the user's location.

    Results are ordered by name match, then distance. 404 if `near` matches no known landmark.
    """
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=422, detail="Give both lat and lon, or neither")
    try:
        result = search_service.search(session, q, near, lat, lon, radius_km, category, limit)
    except search_service.LandmarkNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    results = [PlaceOut.from_hit(h) for h in result.hits]
    return SearchResponse(
        query=q,
        reference=PlaceOut.from_hit(result.reference) if result.reference else None,
        center=Point(lat=result.center[0], lon=result.center[1]) if result.center else None,
        radius_km=result.radius_km,
        count=len(results),
        results=results,
    )


@router.get("/landmarks/resolve", response_model=list[PlaceOut])
def resolve_landmark(
    q: str = Query(..., min_length=1, max_length=200, description='Landmark name, typos OK, e.g. "benett univ"'),
    limit: int = Query(5, ge=1, le=20),
    session: Session = Depends(get_session),
) -> list[PlaceOut]:
    """Best-matching landmarks for free text, best first (for autocomplete and 'near' pickers)."""
    return [PlaceOut.from_hit(h) for h in geo.resolve_landmark(session, q, limit=limit)]
