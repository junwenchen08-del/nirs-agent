"""Add idempotent NIR dataset attachment/use records.

Revision ID: 0004_nir_dataset_uses
Revises: 0003_nir_datasets_profiles
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_nir_dataset_uses"
down_revision: str | Sequence[str] | None = "0003_nir_datasets_profiles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("nir_dataset_uses"):
        expected_columns = {
            "id",
            "owner_user_id",
            "dataset_id",
            "profile_id",
            "thread_id",
            "run_id",
            "workflow_project_id",
            "use_kind",
            "idempotency_key_sha256",
            "attachment_filename",
            "source_sha256",
            "status",
            "created_at",
            "updated_at",
        }
        actual_columns = {column["name"] for column in inspector.get_columns("nir_dataset_uses")}
        actual_uniques = {constraint["name"] for constraint in inspector.get_unique_constraints("nir_dataset_uses")}
        actual_indexes = {index["name"] for index in inspector.get_indexes("nir_dataset_uses")}
        actual_foreign_keys = {(tuple(key["constrained_columns"]), key["referred_table"], tuple(key["referred_columns"])) for key in inspector.get_foreign_keys("nir_dataset_uses")}
        if (
            actual_columns != expected_columns
            or "uq_nir_use_owner_thread_idempotency" not in actual_uniques
            or not {"ix_nir_uses_owner_dataset_created", "ix_nir_uses_owner_thread_created"}.issubset(actual_indexes)
            or (("dataset_id", "owner_user_id"), "nir_datasets", ("id", "owner_user_id")) not in actual_foreign_keys
            or (("profile_id",), "nir_dataset_profiles", ("id",)) not in actual_foreign_keys
        ):
            raise RuntimeError("Existing nir_dataset_uses table does not match the migration contract")
        return
    op.create_table(
        "nir_dataset_uses",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_user_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("profile_id", sa.String(length=64), nullable=True),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("workflow_project_id", sa.String(length=128), nullable=True),
        sa.Column("use_kind", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("attachment_filename", sa.String(length=256), nullable=True),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["nir_dataset_profiles.id"], name="fk_nir_use_profile", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_use_dataset_owner",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_user_id", "thread_id", "idempotency_key_sha256", name="uq_nir_use_owner_thread_idempotency"),
    )
    op.create_index("ix_nir_uses_owner_dataset_created", "nir_dataset_uses", ["owner_user_id", "dataset_id", "created_at"])
    op.create_index("ix_nir_uses_owner_thread_created", "nir_dataset_uses", ["owner_user_id", "thread_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_nir_uses_owner_thread_created", table_name="nir_dataset_uses")
    op.drop_index("ix_nir_uses_owner_dataset_created", table_name="nir_dataset_uses")
    op.drop_table("nir_dataset_uses")
