"""Batch B persistence contracts for user-scoped NIR assets and profiles."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command as alembic_command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import Session

from deerflow.persistence.base import Base
from deerflow.persistence.bootstrap import _get_alembic_config, _upgrade
from deerflow.persistence.nir_library.model import NIRDatasetProfileRow, NIRDatasetRow, NIRDatasetUseRow, NIRModelVersionRow


def _dataset(*, row_id: str, owner: str, sha256: str) -> NIRDatasetRow:
    return NIRDatasetRow(
        id=row_id,
        owner_user_id=owner,
        name="Example spectra",
        original_filename="example.csv",
        sha256=sha256,
        size_bytes=128,
        media_type="text/csv",
        storage_relpath=f"{row_id}/source.csv",
        status="ready",
        metadata_json={},
    )


def test_nir_library_models_register_dataset_and_model_tables() -> None:
    assert {"nir_datasets", "nir_dataset_profiles", "nir_dataset_uses", "nir_model_versions"}.issubset(Base.metadata.tables)


def test_dataset_hash_is_unique_per_owner_but_not_across_owners() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    digest = "a" * 64

    with Session(engine) as session:
        session.add(_dataset(row_id="ds_one", owner="alice", sha256=digest))
        session.commit()

        session.add(_dataset(row_id="ds_duplicate", owner="alice", sha256=digest))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        session.add(_dataset(row_id="ds_other_owner", owner="bob", sha256=digest))
        session.commit()

        deleted = session.get(NIRDatasetRow, "ds_one")
        assert deleted is not None
        deleted.status = "deleted"
        session.commit()

        session.add(_dataset(row_id="ds_reimported", owner="alice", sha256=digest))
        session.commit()


def test_profile_versions_are_immutable_per_dataset() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(_dataset(row_id="ds_one", owner="alice", sha256="b" * 64))
        session.flush()
        common = {
            "dataset_id": "ds_one",
            "owner_user_id": "alice",
            "profile_version": 1,
            "task_type": "calibration",
            "schema_status": "needs_user_mapping",
            "profile_status": "draft",
            "mapping_json": {"x_cols": "1:", "y_col": 0},
            "mapping_sha256": "c" * 64,
        }
        profile = NIRDatasetProfileRow(id="dsp_one", **common)
        session.add(profile)
        session.commit()
        assert profile.profile_status == "draft"
        assert profile.schema_status == "needs_user_mapping"

        session.add(NIRDatasetProfileRow(id="dsp_duplicate", **common))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_profile_owner_must_match_dataset_owner() -> None:
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_dataset(row_id="ds_one", owner="alice", sha256="d" * 64))
        session.commit()
        session.add(
            NIRDatasetProfileRow(
                id="dsp_wrong_owner",
                owner_user_id="bob",
                dataset_id="ds_one",
                profile_version=1,
                profile_status="draft",
                task_type="calibration",
                schema_status="needs_user_mapping",
                mapping_json={},
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


@pytest.mark.asyncio
async def test_existing_0002_database_upgrades_dataset_tables(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'nir-library.db').as_posix()}")
    try:
        cfg = _get_alembic_config(engine)
        await asyncio.to_thread(_upgrade, cfg, "0002_runs_token_usage")
        async with engine.connect() as conn:
            before = await conn.run_sync(lambda connection: set(sa.inspect(connection).get_table_names()))
        assert "nir_datasets" not in before

        await asyncio.to_thread(_upgrade, cfg, "head")
        async with engine.connect() as conn:
            after = await conn.run_sync(lambda connection: set(sa.inspect(connection).get_table_names()))
            revision = (await conn.execute(sa.text("SELECT version_num FROM alembic_version"))).scalar_one()
        assert {"nir_datasets", "nir_dataset_profiles", "nir_dataset_uses", "nir_model_versions"}.issubset(after)
        assert revision == "0005_nir_model_versions"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_dataset_migration_downgrade_and_reupgrade_on_temporary_database(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'nir-library-roundtrip.db').as_posix()}")
    try:
        cfg = _get_alembic_config(engine)
        await asyncio.to_thread(_upgrade, cfg, "head")
        await asyncio.to_thread(alembic_command.downgrade, cfg, "0002_runs_token_usage")
        async with engine.connect() as conn:
            downgraded = await conn.run_sync(lambda connection: set(sa.inspect(connection).get_table_names()))
            revision = (await conn.execute(sa.text("SELECT version_num FROM alembic_version"))).scalar_one()
        assert "nir_datasets" not in downgraded
        assert "nir_dataset_profiles" not in downgraded
        assert revision == "0002_runs_token_usage"

        await asyncio.to_thread(_upgrade, cfg, "head")
        async with engine.connect() as conn:
            upgraded = await conn.run_sync(lambda connection: set(sa.inspect(connection).get_table_names()))
        assert {"nir_datasets", "nir_dataset_profiles", "nir_dataset_uses", "nir_model_versions"}.issubset(upgraded)
    finally:
        await engine.dispose()


def test_dataset_use_owner_and_idempotency_are_database_enforced() -> None:
    engine = sa.create_engine("sqlite:///:memory:")

    @sa.event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_dataset(row_id="ds_one", owner="alice", sha256="e" * 64))
        session.commit()
        use = dict(
            owner_user_id="alice",
            dataset_id="ds_one",
            thread_id="thread1",
            use_kind="attach",
            idempotency_key_sha256="f" * 64,
            source_sha256="e" * 64,
            status="attached",
        )
        session.add(NIRDatasetUseRow(id="use_one", **use))
        session.commit()
        session.add(NIRDatasetUseRow(id="use_duplicate", **use))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.add(NIRDatasetUseRow(id="use_wrong_owner", **{**use, "owner_user_id": "bob", "thread_id": "thread2"}))
        with pytest.raises(IntegrityError):
            session.commit()


def test_model_version_identity_and_promotion_idempotency_are_database_enforced() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    common = {
        "owner_user_id": "alice",
        "model_id": "tablet_assay",
        "version": "tablet_assay-v1",
        "artifact_relpath": "tablet_assay/tablet_assay-v1/model.pkl",
        "artifact_size_bytes": 1024,
        "artifact_sha256": "a" * 64,
        "metrics_sha256": "b" * 64,
        "training_data_sha256": "c" * 64,
        "validation_scope": "independent_holdout_not_external",
        "source_thread_id": "thread1",
        "source_attempt": 1,
        "status": "ready",
    }
    with Session(engine) as session:
        session.add(NIRModelVersionRow(id="nmv_one", **common))
        session.commit()

        session.add(NIRModelVersionRow(id="nmv_duplicate_version", **common))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        session.add(
            NIRModelVersionRow(
                id="nmv_duplicate_promotion",
                **{**common, "version": "tablet_assay-v2"},
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
