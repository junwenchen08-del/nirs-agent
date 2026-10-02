"""Add owner-scoped NIR source assets and interpretation profiles.

Revision ID: 0003_nir_datasets_profiles
Revises: 0002_runs_token_usage

Only metadata enters SQL. Original MAT/CSV bytes remain in the user-owned
persistent directory and are not created or removed by this migration.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_nir_datasets_profiles"
down_revision: str | Sequence[str] | None = "0002_runs_token_usage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_dataset_table() -> None:
    op.create_table(
        "nir_datasets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_user_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("original_filename", sa.String(length=256), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=True),
        sa.Column("storage_relpath", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "owner_user_id", name="uq_nir_dataset_id_owner"),
    )
    op.create_index("ix_nir_datasets_owner_status_created", "nir_datasets", ["owner_user_id", "status", "created_at"])
    op.create_index(
        "uq_nir_datasets_active_owner_sha256",
        "nir_datasets",
        ["owner_user_id", "sha256"],
        unique=True,
        sqlite_where=sa.text("status != 'deleted' AND sha256 IS NOT NULL"),
        postgresql_where=sa.text("status != 'deleted' AND sha256 IS NOT NULL"),
    )


def _create_profile_table() -> None:
    op.create_table(
        "nir_dataset_profiles",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_user_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("profile_status", sa.String(length=32), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=True),
        sa.Column("schema_status", sa.String(length=32), nullable=False),
        sa.Column("mapping_json", sa.JSON(), nullable=False),
        sa.Column("mapping_sha256", sa.String(length=64), nullable=True),
        sa.Column("confirmed_by", sa.String(length=64), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_profile_dataset_owner",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id", "profile_version", name="uq_nir_profile_dataset_version"),
    )
    op.create_index("ix_nir_profiles_owner_dataset", "nir_dataset_profiles", ["owner_user_id", "dataset_id"])


def _validate_existing_table(
    table_name: str,
    *,
    columns: set[str],
    unique_constraint: str,
    indexes: set[str],
) -> bool:
    """Accept a matching pre-created table; reject an incompatible collision."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(table_name):
        return False
    actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
    actual_uniques = {constraint["name"] for constraint in inspector.get_unique_constraints(table_name)}
    actual_indexes = {index["name"] for index in inspector.get_indexes(table_name)}
    if actual_columns != columns or unique_constraint not in actual_uniques or not indexes.issubset(actual_indexes):
        raise RuntimeError(f"Existing {table_name} table does not match the NIR library migration contract")
    if table_name == "nir_dataset_profiles":
        expected_foreign_key = (["dataset_id", "owner_user_id"], "nir_datasets", ["id", "owner_user_id"])
        actual_foreign_keys = {(tuple(key["constrained_columns"]), key["referred_table"], tuple(key["referred_columns"])) for key in inspector.get_foreign_keys(table_name)}
        if (tuple(expected_foreign_key[0]), expected_foreign_key[1], tuple(expected_foreign_key[2])) not in actual_foreign_keys:
            raise RuntimeError(f"Existing {table_name} table lacks the owner-scoped dataset foreign key")
    return True


def upgrade() -> None:
    if not _validate_existing_table(
        "nir_datasets",
        columns={"id", "owner_user_id", "name", "original_filename", "sha256", "size_bytes", "media_type", "storage_relpath", "status", "metadata_json", "created_at", "updated_at", "last_used_at"},
        unique_constraint="uq_nir_dataset_id_owner",
        indexes={"ix_nir_datasets_owner_status_created", "uq_nir_datasets_active_owner_sha256"},
    ):
        _create_dataset_table()
    if not _validate_existing_table(
        "nir_dataset_profiles",
        columns={"id", "owner_user_id", "dataset_id", "profile_version", "profile_status", "task_type", "schema_status", "mapping_json", "mapping_sha256", "confirmed_by", "confirmed_at", "created_at"},
        unique_constraint="uq_nir_profile_dataset_version",
        indexes={"ix_nir_profiles_owner_dataset"},
    ):
        _create_profile_table()


def downgrade() -> None:
    # Downgrade drops the SQL index/metadata, so operators must first take a
    # consistent database + asset-volume backup. Normal rollout uses only
    # upgrade/feature-flag rollback, never an automatic destructive downgrade.
    op.drop_index("ix_nir_profiles_owner_dataset", table_name="nir_dataset_profiles")
    op.drop_table("nir_dataset_profiles")
    op.drop_index("uq_nir_datasets_active_owner_sha256", table_name="nir_datasets")
    op.drop_index("ix_nir_datasets_owner_status_created", table_name="nir_datasets")
    op.drop_table("nir_datasets")
