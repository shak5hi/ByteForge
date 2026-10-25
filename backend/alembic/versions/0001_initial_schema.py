"""Initial schema: PostGIS + pg_trgm, places, place_aliases, place_sources

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "places",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_norm", sa.Text(), nullable=False),
        sa.Column("category", sa.String(64)),
        sa.Column("is_landmark", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("address", sa.Text()),
        sa.Column("city", sa.Text()),
        sa.Column("country", sa.String(64)),
        sa.Column("phone", sa.Text()),
        sa.Column("website", sa.Text()),
        sa.Column(
            "geom",
            geoalchemy2.Geography(geometry_type="POINT", srid=4326, spatial_index=False),
            nullable=False,
        ),
        sa.Column("match_confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("match_confidence BETWEEN 0 AND 1", name="ck_places_match_confidence"),
    )
    op.create_index("ix_places_geom", "places", ["geom"], postgresql_using="gist")
    op.create_index("ix_places_name_norm_trgm", "places", ["name_norm"], postgresql_using="gin",
                    postgresql_ops={"name_norm": "gin_trgm_ops"})
    op.create_index("ix_places_category", "places", ["category"])
    op.create_index("ix_places_landmark", "places", ["is_landmark"], postgresql_where=sa.text("is_landmark"))

    op.create_table(
        "place_aliases",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("place_id", sa.BigInteger(), sa.ForeignKey("places.id", ondelete="CASCADE"), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("alias_norm", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False, server_default="alt_name"),
        sa.UniqueConstraint("place_id", "alias_norm", name="uq_place_aliases_place_alias"),
    )
    op.create_index("ix_place_aliases_alias_norm_trgm", "place_aliases", ["alias_norm"], postgresql_using="gin",
                    postgresql_ops={"alias_norm": "gin_trgm_ops"})

    op.create_table(
        "place_sources",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("place_id", sa.BigInteger(), sa.ForeignKey("places.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("raw", postgresql.JSONB()),
        sa.Column("match_score", sa.Float()),
        sa.UniqueConstraint("source", "source_id", name="uq_place_sources_source_id"),
    )
    op.create_index("ix_place_sources_place_id", "place_sources", ["place_id"])


def downgrade() -> None:
    op.drop_table("place_sources")
    op.drop_table("place_aliases")
    op.drop_table("places")
