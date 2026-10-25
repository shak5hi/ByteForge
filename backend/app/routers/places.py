from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import geo
from app.db import get_session
from app.schemas import NearbyResponse, PlaceDetail, PlaceOut, Point

router = APIRouter(tags=["places"])

Lat = Query(..., ge=-90, le=90, description="Latitude of the reference point")
Lon = Query(..., ge=-180, le=180, description="Longitude of the reference point")


@router.get("/places/{place_id}", response_model=PlaceDetail)
def get_place(place_id: int, session: Session = Depends(get_session)) -> PlaceDetail:
    """Everything we know about one place, including its aliases and source records."""
    hit = geo.get_place(session, place_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"Place {place_id} not found")
    return PlaceDetail.from_hit(hit)


@router.get("/nearby", response_model=NearbyResponse)
def nearby(
    lat: float = Lat,
    lon: float = Lon,
    radius_km: Optional[float] = Query(None, gt=0, le=geo.MAX_RADIUS_M / 1000,
                                       description="Only places within this distance. Omit for the closest N."),
    category: Optional[str] = Query(None, description='Canonical category, e.g. "hotel", "cafe", "petrol pump"'),
    include_landmarks: bool = Query(False, description="Also return landmarks (universities, stations, ...)"),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> NearbyResponse:
    """Places around a point, closest first.

    With `radius_km`: everything inside that radius. Without it: the `limit` nearest places.
    """
    landmarks = None if include_landmarks else False
    if radius_km:
        hits = geo.within_radius(session, lat, lon, radius_km * 1000, limit, category, landmarks)
    else:
        hits = geo.nearest(session, lat, lon, limit, category, landmarks)
    results = [PlaceOut.from_hit(h) for h in hits]
    return NearbyResponse(center=Point(lat=lat, lon=lon), radius_km=radius_km, count=len(results), results=results)
