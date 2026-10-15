"""Read-only recovery of a model's exact recorded input, never an estimator or Test cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import PurePosixPath

from deerflow.community.nir._common import _read_regular_bytes
from deerflow.community.nir.datasets.service import _hash_regular_file, _source_path
from deerflow.community.nir.process import _digest, _id


def verified_library_source(paths, owner, dataset):
    relative = PurePosixPath(dataset.storage_relpath or "")
    if len(relative.parts) != 2 or relative.parts[0] != dataset.id:
        return False
    path = paths.user_nir_dataset_dir(owner, dataset.id) / relative.name
    manifest = path.parent / "manifest.json"
    if any(p.is_symlink() for p in (path, manifest, *path.parents)) or not path.is_file() or path.stat().st_size != dataset.size_bytes:
        return False
    record = json.loads(_read_regular_bytes(manifest, label="Dataset manifest", limit=4096))
    return record == {"schema_version": 1, "dataset_id": dataset.id, "source_filename": relative.name, "sha256": dataset.sha256, "size_bytes": dataset.size_bytes} and _hash_regular_file(path) == dataset.sha256


def recorded_source(metrics, paths, owner, thread):
    """Require the exact finalized session and frozen model before trusting its source."""
    audit = metrics.get("autonomous_decision_audit") or {}
    if not audit.get("session_id"):
        return None
    root = paths.thread_dir(thread, user_id=owner) / "nir-process"
    index = root / "index.sqlite3"
    if any(p.is_symlink() for p in (index, *index.parents)) or not index.is_file():
        return None
    with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM search_sessions WHERE id=?", (_id(audit["session_id"]),)).fetchone()
    if row is None or len(row[0]) > 8 * 1024**2:
        return None
    session = json.loads(row[0])
    if session.get("stage") != "finalized" or session.get("search_hash") != metrics.get("training_data_hash") or session.get("freeze") != audit.get("freeze"):
        return None
    task = session.get("task_snapshot") or {}
    source = task.get("source_file") or {}
    if source and source != (audit.get("task") or {}).get("source_file"):
        return None  # The saved, integrity-checked model metrics must agree.
    virtual = source.get("virtual_path") or (task.get("_workflow_snapshot") or {}).get("data_path")
    path = _source_path(paths, owner, thread, virtual)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        return None
    sha = _hash_regular_file(path)
    if not sha or source and (source.get("sha256") != sha or source.get("size_bytes") != path.stat().st_size):
        return None
    return {"path": path, "virtual_path": virtual, "sha256": sha, "session": session, "recorded_hash": bool(source)}


def verify_historical_source(source):
    """Old sessions lack a file hash: require the entire original partition binding.

    Only bounded original uploads are parsed. Private search/holdout arrays and
    fitted artifacts are never opened; no metric is recomputed or model fitted.
    Ambiguous MAT mappings or changed input fail without a filename-based guess.
    """
    if source["recorded_hash"]:
        return True
    import numpy as np
    from nir_core.io.loaders import load_csv
    from nir_core.model.evaluation import split_dataset
    from nir_core.utils.scientific_validation import _array_digest

    from deerflow.community.nir.datasets.preview import _check_mat_size
    from deerflow.community.nir.datasets.workspace import fixed_regression_partitions, load_workspace_export
    from deerflow.community.nir.search_tools import _load_search_mat

    path, session = source["path"], source["session"]
    if path.suffix.lower() == ".mat":
        _check_mat_size(path)
        data = _load_search_mat(str(path))
    else:
        # Apply the same bounded table limit as the library workbench before
        # the historical CSV loader materializes any full matrix.
        from deerflow.community.nir.datasets.workspace import read_workbook

        read_workbook(path)
        workspace = load_workspace_export(str(path))
        if workspace is not None:
            data = fixed_regression_partitions(workspace)
        else:
            loaded = load_csv(str(path), y_col=0)
            data = {"X": loaded.X, "y": loaded.y, **({"wv": loaded.wv} if loaded.wv is not None else {})}
    if "X_cal" not in data:
        cal, val, test = split_dataset(data["X"], data["y"], test_ratio=0.2, val_ratio=0.2, random_state=42)
        data = {"X_cal": cal[0], "y_cal": cal[1], "X_val": val[0], "y_val": val[1], "X_test": test[0], "y_test": test[1], **({"wv": data["wv"]} if "wv" in data else {})}
    keys = ("X_cal", "y_cal", "X_val", "y_val", "X_test", "y_test")
    arrays = {key: np.asarray(data[key], dtype=float) for key in keys}
    for name in ("cal", "val", "test"):
        arrays["y_" + name] = arrays["y_" + name].reshape(-1)
    if "wv" in data:
        arrays["wv"] = np.asarray(data["wv"], dtype=float)
    if sum(a.size for a in arrays.values()) > 5_000_000 or any(not np.isfinite(a).all() for a in arrays.values()):
        return False
    # Version-1 ingestion binding, exactly as written by SearchSessionStore.
    values = [_array_digest(*(arrays[key] for key in sorted(arrays))), session["cv_folds"], session["max_experiments"], session["max_fits"], session["domain"]]
    if session.get("spectral_axis_unit"):
        values.append({"wv_unit": session["spectral_axis_unit"]})
    binding = _digest(values + ([session["decision_strategy"]] if session.get("decision_strategy") != "legacy" else []))
    if session.get("autonomous_rounds"):
        binding = _digest([binding, "autonomous_rounds", session["max_decisions"]])
    if session.get("delivery_policy"):
        binding = _digest([binding, "autonomous_delivery", session["delivery_policy"], session.get("task_snapshot") or {}])
    return binding == session.get("binding") and _hash_regular_file(path) == source["sha256"]
