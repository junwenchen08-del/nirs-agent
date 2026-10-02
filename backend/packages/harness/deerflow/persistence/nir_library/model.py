"""ORM rows for owner-scoped NIR datasets, uses, and promoted models."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


def _utc_now() -> datetime:
    return datetime.now(UTC)


class NIRDatasetRow(Base):
    """One owner-scoped copy of an original, byte-identical spectral file."""

    __tablename__ = "nir_datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    media_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="preparing")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The redundant composite key lets the profile FK enforce ownership in
        # both SQLite and PostgreSQL, not just in a future service-layer check.
        UniqueConstraint("id", "owner_user_id", name="uq_nir_dataset_id_owner"),
        Index("ix_nir_datasets_owner_status_created", "owner_user_id", "status", "created_at"),
        # Deleted tombstones can release their fingerprint so the same bytes
        # can be deliberately saved again without reusing an old asset ID.
        Index(
            "uq_nir_datasets_active_owner_sha256",
            "owner_user_id",
            "sha256",
            unique=True,
            sqlite_where=text("status != 'deleted' AND sha256 IS NOT NULL"),
            postgresql_where=text("status != 'deleted' AND sha256 IS NOT NULL"),
        ),
    )


class NIRDatasetProfileRow(Base):
    """Immutable interpretation version for a source asset.

    ``schema_status`` describes structural inspection; ``profile_status`` is
    the separate user-confirmation lifecycle. Neither status makes raw bytes
    scientifically valid without the existing training-time checks.
    """

    __tablename__ = "nir_dataset_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, nullable=False)
    profile_status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_status: Mapped[str] = mapped_column(String(32), nullable=False, default="needs_user_mapping")
    mapping_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    mapping_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_profile_dataset_owner",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("dataset_id", "profile_version", name="uq_nir_profile_dataset_version"),
        Index("ix_nir_profiles_owner_dataset", "owner_user_id", "dataset_id"),
    )


class NIRDatasetUseRow(Base):
    """One idempotent dataset attachment or later workflow-use event."""

    __tablename__ = "nir_dataset_uses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("nir_dataset_profiles.id", ondelete="RESTRICT"), nullable=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workflow_project_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    use_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="attach")
    idempotency_key_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    attachment_filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="preparing")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_use_dataset_owner",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("owner_user_id", "thread_id", "idempotency_key_sha256", name="uq_nir_use_owner_thread_idempotency"),
        Index("ix_nir_uses_owner_dataset_created", "owner_user_id", "dataset_id", "created_at"),
        Index("ix_nir_uses_owner_thread_created", "owner_user_id", "thread_id", "created_at"),
    )


class NIRModelVersionRow(Base):
    """One immutable, verified model package promoted for cross-session reuse."""

    __tablename__ = "nir_model_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_relpath: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    training_data_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_scope: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preprocessing_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source_dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_profile_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("nir_dataset_profiles.id", ondelete="RESTRICT"),
        nullable=True,
    )
    source_thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    promotion_key_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="preparing")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_dataset_id", "owner_user_id"],
            ["nir_datasets.id", "nir_datasets.owner_user_id"],
            name="fk_nir_model_dataset_owner",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("owner_user_id", "model_id", "version", name="uq_nir_model_owner_id_version"),
        UniqueConstraint(
            "owner_user_id",
            "source_thread_id",
            "artifact_sha256",
            name="uq_nir_model_owner_source_artifact",
        ),
        UniqueConstraint(
            "owner_user_id",
            "promotion_key_sha256",
            name="uq_nir_model_owner_promotion_key",
        ),
        Index("ix_nir_models_owner_status_created", "owner_user_id", "status", "created_at"),
        Index("ix_nir_models_owner_model_created", "owner_user_id", "model_id", "created_at"),
    )
