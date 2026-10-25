"""Database tables.

places          one real-world business or landmark, with its coordinates
place_aliases   other names it is known by ("BU" for "Bennett University")
place_sources   the raw record(s) each place was built from (OSM id, S1/S2/S3 id, ...)

Coordinates are a PostGIS geography(Point, 4326), so distances come back in metres
and radius filters use real-world distance instead of degrees.
"""
from datetime import datetime
from typing import Any, Optional

from geoalchemy2 import Geography
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Place(Base):
    __tablename__ = "places"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    name_norm: Mapped[str] = mapped_column(Text)  # lowercase, ASCII, no punctuation
    category: Mapped[Optional[str]] = mapped_column(String(64))  # canonical, e.g. "hotel"
    is_landmark: Mapped[bool] = mapped_column(Boolean, default=False)

    address: Mapped[Optional[str]] = mapped_column(Text)
    city: Mapped[Optional[str]] = mapped_column(Text)
    country: Mapped[Optional[str]] = mapped_column(String(64))
    phone: Mapped[Optional[str]] = mapped_column(Text)
    website: Mapped[Optional[str]] = mapped_column(Text)

    geom: Mapped[Any] = mapped_column(Geography(geometry_type="POINT", srid=4326, spatial_index=False))
    match_confidence: Mapped[float] = mapped_column(Float, default=1.0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    aliases: Mapped[list["PlaceAlias"]] = relationship(back_populates="place", cascade="all, delete-orphan")
    sources: Mapped[list["PlaceSource"]] = relationship(back_populates="place", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_places_geom", "geom", postgresql_using="gist"),
        Index("ix_places_name_norm_trgm", "name_norm", postgresql_using="gin",
              postgresql_ops={"name_norm": "gin_trgm_ops"}),
        Index("ix_places_category", "category"),
        Index("ix_places_landmark", "is_landmark", postgresql_where="is_landmark"),
    )


class PlaceAlias(Base):
    __tablename__ = "place_aliases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    place_id: Mapped[int] = mapped_column(ForeignKey("places.id", ondelete="CASCADE"))
    alias: Mapped[str] = mapped_column(Text)
    alias_norm: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="alt_name")  # alt_name | short_name | old_name | ...

    place: Mapped[Place] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("place_id", "alias_norm", name="uq_place_aliases_place_alias"),
        Index("ix_place_aliases_alias_norm_trgm", "alias_norm", postgresql_using="gin",
              postgresql_ops={"alias_norm": "gin_trgm_ops"}),
    )


class PlaceSource(Base):
    __tablename__ = "place_sources"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    place_id: Mapped[int] = mapped_column(ForeignKey("places.id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(32))  # "osm", "source1", "source2", ...
    source_id: Mapped[str] = mapped_column(Text)  # e.g. "node/123456", "S2-192345572"
    raw: Mapped[Optional[dict]] = mapped_column(JSONB)
    match_score: Mapped[Optional[float]] = mapped_column(Float)  # entity-resolution score that linked it

    place: Mapped[Place] = relationship(back_populates="sources")

    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_place_sources_source_id"),
        Index("ix_place_sources_place_id", "place_id"),
    )
