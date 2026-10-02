"""Add persistent owner-scoped NIR model versions.

Revision ID: 0005_nir_model_versions
Revises: 0004_nir_dataset_uses
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_nir_model_versions"
down_revision: str | Sequence[str] | None = "0004_nir_dataset_uses"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("nir_model_versions"):
        expected_columns = {
            "id",
            "owner_user_id",
            "model_id",
            "version",
            "artifact_relpath",
            "artifact_size_bytes",
            "artifact_sha256",
            "metrics_sha256",
            "training_data_sha256",
            "validation_scope",
            "method",
            "preprocessing_json",
            "source_dataset_id",
            "source_profile_id",
            "source_thread_id",
            "source_run_id",
            "source_attempt",
            "promotion_key_sha256",
            "status",
            "metadata_json",
            "created_at",
            "updated_at",
            "last_used_at",
        }
        actual_columns = {column["name"] for column in inspector.get_columns("nir_model_versions")}
        actual_uniques = {constraint["name"] for constraint in inspector.get_unique_constraints("nir_model_versions")}
        actual_indexes = {index["name"] for index in inspector.get_indexes("nir_model_versions")}
        actual_foreign_keys = {(tuple(key["constrained_columns"]), key["referred_table"], tuple(key["referred_columns"])) for key in inspector.get_foreign_keys("nir_model_versions")}
        if (
            actual_columns != expected_columns
            or not {
                "uq_nir_model_owner_id_version",
                "uq_nir_model_owner_source_artifact",
                "uq_nir_model_owner_promotion_key",
            }.issubset(actual_uniques)
            or not {"ix_nir_models_owner_status_created", "ix_nir_models_owner_model_created"}.issubset(actual_indexes)
            or (("source_dataset_id", "owner_user_id"), "nir_datasets", ("id", "owner_user_id")) not in actual_foreign_keys
            or (("source_profile_id",), "nir_dataset_profiles", ("id",)) not in actual_foreign_keys
        ):
            raise RuntimeError("Existing nir_model_versions table does not match the migration contract")
        return

    op.create_table(
        "nir_model_versions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_user_id", sa.String(length=64), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("artifact_relpath", sa.Text(), nullable=False),
        sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("metrics_sha256", sa.String(length=64), nullable=False),
        sa.Column("training_data_sha256", sa.String(length=64), nullable=False),
        sa.Column("validation_scope", sa.String(length=64), nullable=False),
        sa.Column("method", sa.String(length=64), nullable=True),
        sa.Column("preprocessing_json", sa.JSON(), nullable=False),
        sa.Column("source_dataset_id", sa.String(length=64), nullable=True),
        sa.Column("source_profile_id", sa.String(length=64), nullable=True),
        sa.Column("source_thread_id", sa.String(length=64), nullable=False),
        sa.Column("source_run_id", sa.String(length=64), nullable=True),
        sa.Column("source_attempt", sa.Integer(), nullable=False),
        sa.Column("promotion_key_sha256", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_model_dataset_owner",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["source_profile_id"],
            ["nir_dataset_profiles.id"],
            name="fk_nir_model_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_user_id", "model_id", "version", name="uq_nir_model_owner_id_version"),
        sa.UniqueConstraint(
            "owner_user_id",
            "source_thread_id",
            "artifact_sha256",
            name="uq_nir_model_owner_source_artifact",
        ),
        sa.UniqueConstraint(
            "owner_user_id",
            "promotion_key_sha256",
            name="uq_nir_model_owner_promotion_key",
        ),
    )
    op.create_index(
        "ix_nir_models_owner_status_created",
        "nir_model_versions",
        ["owner_user_id", "status", "created_at"],
    )
    op.create_index(
        "ix_nir_models_owner_model_created",
        "nir_model_versions",
        ["owner_user_id", "model_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_nir_models_owner_model_created", table_name="nir_model_versions")
    op.drop_index("ix_nir_models_owner_status_created", table_name="nir_model_versions")
    op.drop_table("nir_model_versions")
