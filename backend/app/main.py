from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.routers import places, search

app = FastAPI(
    title="ByteForge Place Search API",
    version="0.1.0",
    description="Search businesses and landmarks, find what's nearby, and look up place details.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(places.router)
app.include_router(search.router)


@app.get("/health", tags=["meta"])
def health(session: Session = Depends(get_session)) -> dict:
    """Liveness plus a database round-trip, so a broken DB connection shows up here."""
    place_count = session.execute(text("SELECT count(*) FROM places")).scalar_one()
    postgis = session.execute(text("SELECT postgis_lib_version()")).scalar_one()
    return {"status": "ok", "places": place_count, "postgis": postgis}
