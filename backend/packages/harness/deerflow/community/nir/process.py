"""Thread-local, durable process evidence; charts never enter agent checkpoints.

The index is a separate SQLite database in the owner thread directory. Its
versioned schema is transactionally bootstrapped, and thread deletion removes
it with the thread. It is intentionally outside sandbox-visible user-data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
CHART_LIMIT = 1024 * 1024
STEP_KEYS = ("audit", "split", "preprocessing", "wavelength", "model", "validation", "reflection", "registration")
MODEL_TOOLS = frozenset({"nir_analyze", "nir_train_model", "nir_train_auto_split_model", "nir_train_partitioned_model"})
TOOL_STEPS = {"nir_inspect": "audit", "nir_load_data": "audit", "nir_preprocess": "preprocessing", "nir_reflect": "reflection", "nir_register_model": "registration", "nir_workflow": "workflow"}
_current: ContextVar[tuple | None] = ContextVar("nir_process_execution", default=None)


def _now():
    return datetime.now(UTC).isoformat()


def _json(value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=default)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value):
        raise ValueError("Invalid process identifier")
    return value


def _safe(value):
    """Keep explanatory facts, exclude arrays/paths/secrets from method details."""
    if isinstance(value, dict):
        return {str(key)[:100]: _safe(item) for key, item in value.items() if not any(word in str(key).lower() for word in ("path", "secret", "token", "password", "indices", "prediction", "_model", "_predict"))}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value[:500]]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        if re.fullmatch(r"https://chemotools\.org/methods/[A-Za-z0-9_./#-]{1,450}", value):
            return value
        return None if re.search(r"/mnt/|[A-Za-z]:[\\/]|\\\\", value) else value[:2000]
    return value if value is None or isinstance(value, (int, bool)) else None


def comparison_key(context):
    required = ("dataset_hash", "interpretation_hash", "target", "unit", "protocol", "split_hash", "evaluation_kind", "evaluation_hash", "metric_definition_version")
    if context.get("evaluation_kind") not in {"tuning", "cross_validation"} or any(not context.get(key) for key in required):
        return None
    if context["evaluation_kind"] == "cross_validation" and not context.get("cv_folds_hash"):
        return None
    return _digest({key: context.get(key) for key in (*required, "cv_folds_hash", "validation_scope", "aggregation", "target_transform")})


def validate_chart(chart):
    """Reject inconsistent vectors before exposing them as available evidence."""
    if chart.get("type") not in {"spectra", "distribution", "prediction", "residual", "wavelength", "cv"}:
        raise ValueError("Unsupported chart type")
    for key in ("n_total", "n_returned"):
        if not isinstance(chart.get(key), int) or chart[key] < 0:
            raise ValueError("Invalid chart counts")
    if not chart.get("sampling"):
        raise ValueError("Missing chart sampling description")
    x = chart.get("x", [])
    vectors = [chart["y"]] if "y" in chart else [item["values"] for item in chart.get("series", [])]
    if any(len(vector) != len(x) for vector in vectors):
        raise ValueError("Chart vector length mismatch")
    if any(not np.all(np.isfinite(np.asarray(vector, dtype=float))) for vector in [x, *vectors]):
        raise ValueError("Non-finite chart values")
    if chart["type"] in {"prediction", "residual"}:
        if len(chart.get("sample_ids", [])) != len(x) or len(x) != chart["n_returned"] or "y" not in chart:
            raise ValueError("Chart sample identity length mismatch")
    if chart["type"] == "spectra" and (len(chart.get("series", [])) != chart["n_returned"] or len(x) != chart.get("axis_returned")):
        raise ValueError("Chart spectra length mismatch")
    if chart["type"] == "wavelength":
        ranges = chart.get("ranges", [])
        if any(item["start"] > item["end"] or item["count"] < 1 for item in ranges) or sum(item["count"] for item in ranges) != chart["n_returned"]:
            raise ValueError("Chart wavelength interval length mismatch")


class ProcessStore:
    """Atomic index and immutable, bounded, hash-verified chart blobs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = self.root / "index.sqlite3"

    def _validate_root(self):
        if any(parent.is_symlink() for parent in (self.root, *self.root.parents)):
            raise ValueError("Unsafe process storage")

    @contextmanager
    def _connection(self, *, write=False):
        self._validate_root()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.db.is_symlink():
            raise ValueError("Unsafe process storage")
        connection = sqlite3.connect(self.db, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if not write:
                    connection.execute("BEGIN IMMEDIATE")
                for ddl in (
                    "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, stage TEXT, updated TEXT)",
                    "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, call_id TEXT NOT NULL, "
                    "number INTEGER NOT NULL, tool TEXT NOT NULL, status TEXT NOT NULL, started TEXT, ended TEXT, "
                    "UNIQUE(run_id, call_id), UNIQUE(run_id, number))",
                    "CREATE TABLE IF NOT EXISTS steps (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, attempt_id TEXT, step_key TEXT NOT NULL, payload TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS charts (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step_id TEXT NOT NULL, hash TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, execution_id TEXT NOT NULL, kind TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(execution_id, kind))",
                    "CREATE INDEX IF NOT EXISTS process_steps_run ON steps(run_id, attempt_id)",
                ):
                    connection.execute(ddl)
                connection.execute("PRAGMA user_version=1")
                if not write:
                    connection.commit()
            if version in {0, 1}:
                if not write:
                    connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE IF NOT EXISTS agent_runs (attempt_id TEXT PRIMARY KEY, agent_run_id TEXT NOT NULL)")
                connection.execute("CREATE INDEX IF NOT EXISTS process_agent_run ON agent_runs(agent_run_id)")
                connection.execute("PRAGMA user_version=2")
                if not write:
                    connection.commit()
            elif version != 2:
                raise ValueError("Unsupported process index schema")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin(self, project, call_id, tool, *, agent_run_id=None):
        run_id = "run_" + _digest(str(project))[:32]
        attempt_id = "attempt_" + _digest([run_id, call_id])[:32]
        with self._connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?,0,'execution',?)", (run_id, _now()))
            existing = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if existing is None:
                number = db.execute("SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE run_id=?", (run_id,)).fetchone()[0]
                db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,'running',?,NULL)", (attempt_id, run_id, str(call_id), number, tool, _now()))
                db.execute("INSERT INTO events(run_id,execution_id,kind,created) VALUES (?,?,'started',?)", (run_id, attempt_id, _now()))
                db.execute("UPDATE runs SET revision=revision+1,stage='execution',updated=? WHERE id=?", (_now(), run_id))
            else:
                number = existing["number"]
            if agent_run_id:
                db.execute("INSERT OR IGNORE INTO agent_runs VALUES (?,?)", (attempt_id, str(agent_run_id)))
        return {"run_id": run_id, "attempt_id": attempt_id, "number": number, "tool": tool}

    def recover(self, agent_run_id):
        """Only recover attempts tied to a runtime run confirmed orphaned."""
        if not self.db.exists():
            return
        with self._connection() as db:
            attempts = [dict(row) for row in db.execute("SELECT id AS attempt_id,run_id FROM attempts WHERE status='running' AND id IN (SELECT attempt_id FROM agent_runs WHERE agent_run_id=?)", (str(agent_run_id),))]
        for attempt in attempts:
            self.finish(attempt, "interrupted", {"stage": "execution"})

    def record_action(self, project, call_id, step_key, status, payload):
        run_id = "run_" + _digest(str(project))[:32]
        execution_id = "step_" + _digest([run_id, call_id, step_key])[:32]
        detail = {"step_key": step_key, "step_execution_id": execution_id, "attempt_id": None, "execution_status": status, "evidence_status": "available", "facts": _safe(payload), "charts": [], "candidates": []}
        with self._connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?,0,'data_audit',?)", (run_id, _now()))
            inserted = db.execute("INSERT OR IGNORE INTO steps VALUES (?,?,NULL,?,?)", (execution_id, run_id, step_key, _json(detail))).rowcount
            if inserted:
                db.execute("INSERT INTO events(run_id,execution_id,kind,created) VALUES (?,?,?,?)", (run_id, execution_id, status, _now()))
                db.execute("UPDATE runs SET revision=revision+1,stage=?,updated=? WHERE id=?", (payload.get("stage", "data_audit"), _now(), run_id))

    def publish(self, attempt, steps):
        prepared = []
        for source in steps:
            detail = dict(source)
            key = str(detail["step_key"])
            sid = "step_" + _digest([attempt["attempt_id"], key])[:32]
            refs = []
            for index, source_chart in enumerate(detail.pop("charts", [])):
                chart = {"chart_schema_version": 1, "evidence_status": "available", "scope": "unknown", **source_chart}
                chart["step_execution_id"] = sid
                chart["attempt_id"] = attempt["attempt_id"]
                chart["run_id"] = attempt["run_id"]
                chart_id = "chart_" + _digest([sid, index])[:32]
                chart["chart_id"] = chart_id
                raw = _json(chart).encode()
                if len(raw) > CHART_LIMIT:
                    raise ValueError("Chart size exceeds limit")
                validate_chart(chart)
                digest = hashlib.sha256(raw).hexdigest()
                charts_dir = self.root / "charts"
                charts_dir.mkdir(parents=True, exist_ok=True)
                if charts_dir.is_symlink():
                    raise ValueError("Unsafe chart storage")
                destination = charts_dir / f"{digest}.json"
                if not destination.exists():
                    temporary = charts_dir / f"{uuid.uuid4().hex}.tmp"
                    try:
                        with temporary.open("xb") as stream:
                            stream.write(raw)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                refs.append({"chart_id": chart_id, "version": digest, "sha256": digest, "type": chart["type"], "size_bytes": len(raw), "evidence_status": "available"})
            detail.update(step_execution_id=sid, attempt_id=attempt["attempt_id"], charts=refs)
            detail.setdefault("execution_status", "succeeded")
            detail.setdefault("evidence_status", "available" if refs or detail.get("facts") or detail.get("candidates") else "not_recorded")
            prepared.append((sid, key, detail))
        with self._connection(write=True) as db:
            row = db.execute("SELECT status FROM attempts WHERE id=? AND run_id=?", (attempt["attempt_id"], attempt["run_id"])).fetchone()
            if row is None:
                raise KeyError("Attempt not found")
            if row["status"] != "running":
                return
            for sid, key, detail in prepared:
                inserted = db.execute("INSERT OR IGNORE INTO steps VALUES (?,?,?,?,?)", (sid, attempt["run_id"], attempt["attempt_id"], key, _json(detail))).rowcount
                if inserted:
                    for chart in detail["charts"]:
                        db.execute("INSERT INTO charts VALUES (?,?,?,?)", (chart["chart_id"], attempt["run_id"], sid, chart["sha256"]))
                    db.execute("INSERT INTO events(run_id,execution_id,kind,created) VALUES (?,?,?,?)", (attempt["run_id"], sid, detail["execution_status"], _now()))
            db.execute("UPDATE runs SET revision=revision+1,updated=? WHERE id=?", (_now(), attempt["run_id"]))

    def finish(self, attempt, status, workflow):
        with self._connection(write=True) as db:
            changed = db.execute("UPDATE attempts SET status=?,ended=? WHERE id=? AND status='running'", (status, _now(), attempt["attempt_id"])).rowcount
            if changed:
                db.execute("INSERT INTO events(run_id,execution_id,kind,created) VALUES (?,?,?,?)", (attempt["run_id"], attempt["attempt_id"], status, _now()))
                db.execute("UPDATE runs SET revision=revision+1,stage=?,updated=? WHERE id=?", (workflow.get("stage", "execution"), _now(), attempt["run_id"]))

    def runs(self, offset=0):
        if not self.db.exists():
            return {"schema_version": 1, "data": [], "next_offset": None}
        with self._connection() as db:
            rows = [dict(row) for row in db.execute("SELECT id AS run_id,revision,stage,updated FROM runs ORDER BY updated DESC LIMIT 51 OFFSET ?", (offset,))]
        return {"schema_version": 1, "data": rows[:50], "next_offset": offset + 50 if len(rows) > 50 else None}

    def summary(self, run_id, offset=0, attempt_id=None):
        _id(run_id)
        with self._connection() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("Run not found")
            if attempt_id:
                _id(attempt_id)
                selected = db.execute("SELECT number FROM attempts WHERE id=? AND run_id=?", (attempt_id, run_id)).fetchone()
                if selected is None:
                    raise KeyError("Attempt not found")
                ahead = db.execute("SELECT COUNT(*) FROM attempts WHERE run_id=? AND number>?", (run_id, selected[0])).fetchone()[0]
                offset = (ahead // 20) * 20
            attempts = []
            for row in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY number DESC LIMIT 21 OFFSET ?", (run_id, offset)):
                steps = []
                for item in db.execute("SELECT payload FROM steps WHERE run_id=? AND attempt_id=?", (run_id, row["id"])):
                    data = json.loads(item[0])
                    steps.append({key: data.get(key) for key in ("step_execution_id", "step_key", "execution_status", "evidence_status", "comparison_group_id", "comparison_reason", "score", "facts") if key != "facts"})
                attempts.append(
                    {
                        "attempt_id": row["id"],
                        "number": row["number"],
                        "source_tool": row["tool"],
                        "execution_status": row["status"],
                        "started_at": row["started"],
                        "ended_at": row["ended"],
                        "steps": sorted(steps, key=lambda item: STEP_KEYS.index(item["step_key"]) if item["step_key"] in STEP_KEYS else 99),
                    }
                )
            shared = [json.loads(item[0]) for item in db.execute("SELECT payload FROM steps WHERE run_id=? AND attempt_id IS NULL ORDER BY rowid DESC LIMIT 50", (run_id,))]
        return {
            "schema_version": 1,
            "run_id": run_id,
            "revision": run["revision"],
            "stage": run["stage"],
            "updated_at": run["updated"],
            "capabilities": {"can_view_raw_spectra": True, "can_view_sample_points": True},
            "attempts": sorted(attempts[:20], key=lambda item: item["number"]),
            "offset": offset,
            "next_offset": offset + 20 if len(attempts) > 20 else None,
            "shared_steps": [{key: item[key] for key in ("step_execution_id", "step_key", "execution_status", "evidence_status")} for item in shared],
        }

    def step(self, run_id, step_id):
        _id(run_id)
        _id(step_id)
        with self._connection() as db:
            row = db.execute("SELECT payload FROM steps WHERE id=? AND run_id=?", (step_id, run_id)).fetchone()
        if row is None:
            raise KeyError("Step not found")
        result = json.loads(row[0])
        result["candidate_count"] = len(result.pop("candidates", []))
        return result

    def candidates(self, run_id, step_id, offset=0):
        _id(run_id)
        _id(step_id)
        with self._connection() as db:
            row = db.execute("SELECT payload FROM steps WHERE id=? AND run_id=?", (step_id, run_id)).fetchone()
        if row is None:
            raise KeyError("Step not found")
        candidates = json.loads(row[0]).get("candidates", [])
        return {"data": candidates[offset : offset + 50], "total": len(candidates), "next_offset": offset + 50 if offset + 50 < len(candidates) else None}

    def chart(self, run_id, chart_id, version):
        _id(run_id)
        _id(chart_id)
        with self._connection() as db:
            row = db.execute("SELECT hash FROM charts WHERE id=? AND run_id=?", (chart_id, run_id)).fetchone()
        if row is None:
            raise KeyError("Chart not found")
        if row["hash"] != version:
            raise ValueError("Chart version mismatch")
        from ._common import _read_regular_bytes

        try:
            parent = self.root / "charts"
            if parent.is_symlink():
                raise ValueError("Unsafe chart directory")
            raw = _read_regular_bytes(parent / f"{row['hash']}.json", label="Process chart", limit=CHART_LIMIT)
            if hashlib.sha256(raw).hexdigest() != row["hash"]:
                raise ValueError("Chart integrity mismatch")
            chart = json.loads(raw)
            if chart.get("chart_schema_version") != 1 or chart.get("chart_id") != chart_id or chart.get("run_id") != run_id:
                raise ValueError("Unsupported chart schema")
            return chart
        except (ValueError, OSError):
            return {"chart_schema_version": 1, "chart_id": chart_id, "evidence_status": "unavailable", "reason": "missing_or_corrupt_evidence"}


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        normalized = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(normalized.shape).encode())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _spectra_chart(X, wv, *, scope, namespace, max_curves=12, max_axis=700):
    rows = np.linspace(0, len(X) - 1, min(max_curves, len(X)), dtype=int)
    representatives = X[rows]
    if X.shape[1] <= max_axis:
        columns = np.arange(X.shape[1])
    else:
        columns_list = [0, X.shape[1] - 1]
        for bucket in np.array_split(np.arange(X.shape[1]), max_axis // 2 - 1):
            subset = representatives[:, bucket]
            columns_list.extend([int(bucket[np.argmin(np.min(subset, axis=0))]), int(bucket[np.argmax(np.max(subset, axis=0))])])
        columns = np.asarray(sorted(set(columns_list)), dtype=int)
    axis = wv if wv is not None else np.arange(X.shape[1])
    return {
        "type": "spectra",
        "scope": scope,
        "x_label": "wavelength" if wv is not None else "variable",
        "x_unit": None,
        "y_label": "spectral_value",
        "y_unit": None,
        "x": np.asarray(axis)[columns].tolist(),
        "series": [{"id": _digest([namespace, "calibration" if scope.startswith("calibration_") else scope, int(row)])[:12], "values": np.asarray(X[row, columns]).tolist()} for row in rows],
        "n_total": len(X),
        "n_returned": len(rows),
        "axis_total": X.shape[1],
        "axis_returned": len(columns),
        "sampling": "even_rows_representative_envelope_minmax_axis_v1",
    }


def _distribution(values, scope):
    counts, edges = np.histogram(np.asarray(values).ravel(), bins=min(20, max(1, int(np.sqrt(len(values))))))
    return {
        "type": "distribution",
        "scope": scope,
        "x_label": "reference",
        "y_label": "count",
        "x": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "y": counts.tolist(),
        "bin_edges": edges.tolist(),
        "n_total": len(values),
        "n_returned": len(counts),
        "sampling": "full_population_histogram_v1",
    }


def regression_evidence(*, X, y, wv, partitions, processed_cal, predictions, metrics, pipeline=None, unit=None, target=None, namespace="", point_limit=2000, model_candidates=None, fitted_model=None):
    """Export existing deterministic results, without refitting or resplitting."""
    X, y = np.asarray(X), np.asarray(y)
    partition_hashes = {name: _array_hash(*pair) for name, pair in partitions.items()}
    context = {
        "dataset_hash": metrics.get("training_data_hash"),
        "interpretation_hash": _digest({"target": target, "axis": _array_hash(wv) if wv is not None else None, "shape": X.shape}),
        "target": target,
        "unit": unit,
        "protocol": metrics.get("protocol"),
        "split_hash": _digest(partition_hashes),
        "evaluation_kind": "tuning",
        "evaluation_hash": partition_hashes.get("tuning"),
        "metric_definition_version": "rmse-v1",
        "validation_scope": metrics.get("validation_scope"),
        "aggregation": "single_target",
        "target_transform": "identity",
    }
    group = comparison_key(context)
    facts = {"target": target, "unit": unit, "protocol": metrics.get("protocol"), "validation_scope": metrics.get("validation_scope")}
    audit = {
        "step_key": "audit",
        "facts": {**facts, "n_samples": len(X), "n_wavelengths": X.shape[1], "finite_values": bool(np.all(np.isfinite(X)))},
        "charts": [_spectra_chart(X, wv, scope="all", namespace=namespace), {**_distribution(y, "all"), "x_unit": unit}],
    }
    split = {
        "step_key": "split",
        "facts": {**facts, "partitions": {name: len(pair[1]) for name, pair in partitions.items()}, "split_hash": context["split_hash"], "split": _safe(metrics.get("split", {}))},
        "charts": [{**_distribution(pair[1], name), "x_unit": unit} for name, pair in partitions.items()],
    }
    raw_cal = partitions["calibration"][0]
    pre_selection = metrics.get("preprocessing_selection") or {}
    pre_candidates = pre_selection.get("candidates", [])
    if "evaluation" in pre_selection:
        evaluation = pre_selection["evaluation"]
        raw_candidates = evaluation.get("candidates", {})
        pre_candidates = list(raw_candidates.values()) if isinstance(raw_candidates, dict) else raw_candidates
    pre_candidates = [
        {
            **_safe(item),
            "candidate_id": item.get("candidate_id", str(index)),
            "score": _safe(item.get("cv_rmse")),
            "evaluation_kind": "cross_validation",
            "selected": item.get("selected", item.get("candidate_id") == pre_selection.get("evaluation", {}).get("best_candidate_id", pre_selection.get("selected_candidate_id"))),
        }
        for index, item in enumerate(pre_candidates)
    ]
    preprocessing = {
        "step_key": "preprocessing",
        "facts": {**facts, "pipeline": _safe(pipeline or []), "description": metrics.get("preprocessing"), "selection": _safe(pre_selection)},
        "candidates": pre_candidates,
        "charts": [_spectra_chart(raw_cal, wv, scope="calibration_before", namespace=namespace), _spectra_chart(processed_cal, wv, scope="calibration_after", namespace=namespace)],
    }
    from ._preprocessing_plan import method_knowledge_summary

    method_knowledge = pre_selection.get("recommendation", {}).get("method_knowledge")
    if method_knowledge:
        preprocessing["facts"]["method_knowledge"] = _safe(method_knowledge_summary(method_knowledge))
    selection = metrics.get("wavelength_selection") or {}
    selected = selection.get("selected_indices")
    if selected is None:
        selected = list(range(X.shape[1]))
    wave_candidates = metrics.get("wavelength_selection_candidates", metrics.get("candidate_results", []))
    wave_candidates = [
        {**_safe(item), "candidate_id": item.get("method", str(index)), "score": item.get("RMSE_tuning"), "evaluation_kind": "tuning", "selected": item.get("method") == selection.get("method")} for index, item in enumerate(wave_candidates)
    ]
    wavelength = {
        "step_key": "wavelength",
        "execution_status": "skipped" if selection.get("method") == "none" else "succeeded",
        "facts": {**facts, "selection": _safe(selection), "decision": _safe(metrics.get("wavelength_selection_decision", {})), "reason": "full_spectrum_retained" if selection.get("method") == "none" else None},
        "candidates": wave_candidates,
        "charts": [
            {
                "type": "wavelength",
                "scope": "selected_variables",
                "axis_bounds": [float(np.min(wv)), float(np.max(wv))] if wv is not None else [0, X.shape[1] - 1],
                "x_label": "wavelength" if wv is not None else "variable",
                "x_unit": None,
                "n_total": X.shape[1],
                "n_returned": len(selected),
                "ranges": _selected_ranges(selected, wv),
                "sampling": "exact_selected_intervals_v1",
            }
        ],
    }
    candidate_rows = []
    for index, item in enumerate(model_candidates if model_candidates is not None else metrics.get("model_candidates", [])):
        estimator = item.get("_model")
        parameters = _safe(estimator.get_params(deep=False)) if hasattr(estimator, "get_params") else None
        candidate_rows.append({**_safe(item), "candidate_id": item.get("method", str(index)), "score": item.get("RMSE_tuning"), "parameters": parameters, "evaluation_kind": "tuning", "selected": item.get("method") == metrics.get("method")})
    model = {
        "step_key": "model",
        "comparison_group_id": group,
        "comparison_reason": None if group else "missing_evaluation_context",
        "score": metrics.get("val", {}).get("RMSE"),
        "evaluation_context": context,
        "facts": {
            **facts,
            "method": metrics.get("method"),
            "n_components": metrics.get("n_components"),
            "parameters": _safe(fitted_model.get_params(deep=False)) if hasattr(fitted_model, "get_params") else None,
            "decision": _safe(metrics.get("model_selection_decision", {})),
        },
        "candidates": candidate_rows,
        "charts": [],
    }
    if not candidate_rows:
        model["candidates"] = [
            {
                "candidate_id": str(metrics.get("method") or "recorded_model"),
                "method": metrics.get("method"),
                "n_components": metrics.get("n_components"),
                "score": metrics.get("val", {}).get("RMSE"),
                "selected": True,
                "evaluation_kind": "tuning",
                "reason": "single_requested_model",
            }
        ]
    refitted = metrics.get("protocol") in {"deterministic_auto_split_holdout", "named_partition_external_validation", "named_partition_independent_holdout"}
    cv = metrics.get("cv_results") or {}
    if isinstance(cv, dict) and len(cv.get("n_components", [])) and len(cv.get("mean_rmse_cv", [])):
        model["charts"].append(
            {
                "type": "cv",
                "scope": "final_fit_cv" if refitted else "calibration_cv",
                "x_label": "components",
                "y_label": "RMSE",
                "y_unit": unit,
                "evaluation_kind": "cross_validation",
                "x": _safe(cv["n_components"]),
                "y": _safe(cv["mean_rmse_cv"]),
                "n_total": len(cv["n_components"]),
                "n_returned": len(cv["n_components"]),
                "sampling": "all_recorded_candidates",
            }
        )
    diagnosis = {
        "step_key": "validation",
        "facts": {
            **facts,
            "residual_definition": "prediction_minus_reference",
            "quality": _safe(metrics.get("quality", {})),
            "training_scope": "calibration_and_tuning_refit" if refitted else "calibration",
            "adaptive_holdout_warning": "Repeated holdout inspection is not independent final validation.",
        },
        "metrics": {name: _safe(metrics.get(name, {})) for name in ("train", "val", "test")},
        "charts": [],
    }
    for scope, prediction in predictions.items():
        reference = np.asarray(partitions[scope][1]).ravel()
        prediction = np.asarray(prediction).ravel()
        indices = np.unique(np.linspace(0, len(reference) - 1, min(point_limit, len(reference)), dtype=int))
        ids = [_digest([namespace, scope, int(index)])[:16] for index in indices]
        common = {
            "scope": scope,
            "evaluation_kind": "external" if scope == "holdout" and facts["validation_scope"] == "independent_external_validation" else scope,
            "x_label": "reference",
            "x_unit": unit,
            "y_unit": unit,
            "n_total": len(reference),
            "n_returned": len(indices),
            "sampling": "deterministic_even_rows_v1",
            "sample_ids": ids,
            "x": reference[indices].tolist(),
        }
        diagnosis["charts"].extend(
            [{**common, "type": "prediction", "y_label": "prediction", "y": prediction[indices].tolist()}, {**common, "type": "residual", "y_label": "residual", "y": (prediction[indices] - reference[indices]).tolist()}]
        )
    return [audit, split, preprocessing, wavelength, model, diagnosis]


def _selected_ranges(indices, wv):
    """Encode exact contiguous selections without returning huge index arrays."""
    indices = sorted(set(int(value) for value in indices))
    groups = []
    for index in indices:
        if groups and index == groups[-1][1] + 1:
            groups[-1][1] = index
        else:
            groups.append([index, index])
    return [{"start": min(float(wv[start]), float(wv[end])) if wv is not None else start, "end": max(float(wv[start]), float(wv[end])) if wv is not None else end, "count": end - start + 1} for start, end in groups]


def runtime_store(runtime):
    from deerflow.config.paths import get_paths
    from deerflow.runtime.user_context import resolve_runtime_user_id

    config = getattr(runtime, "config", None)
    thread_id = config.get("configurable", {}).get("thread_id") if isinstance(config, dict) else None
    if not thread_id:
        return None
    return ProcessStore(get_paths().thread_dir(str(thread_id), user_id=resolve_runtime_user_id(runtime)) / "nir-process")


def start_execution(runtime, tool, call_id):
    if os.getenv("NIR_PROCESS_ENABLED", "true").lower() == "false":
        return None
    store = runtime_store(runtime)
    if store is None or tool not in MODEL_TOOLS:
        return None
    workflow = getattr(runtime, "state", {}).get("nir_workflow") or {}
    if not workflow.get("project_id"):
        return None
    context = getattr(runtime, "context", None)
    attempt = store.begin(workflow["project_id"], call_id, tool, agent_run_id=context.get("run_id") if isinstance(context, dict) else None)
    return store, attempt


def bind_execution(execution):
    return _current.set(execution)


def reset_execution(token):
    _current.reset(token)


def publish_regression(runtime, *, tool, call_id, **kwargs):
    """Chart failures degrade independently of core model/scientific evidence."""
    try:
        execution = _current.get() or start_execution(runtime, tool, call_id)
        if execution is None:
            return {"evidence_status": "not_recorded"}
        store, attempt = execution
        workflow = getattr(runtime, "state", {}).get("nir_workflow") or {}
        steps = regression_evidence(**kwargs, unit=workflow.get("unit"), target=kwargs["metrics"].get("target") or workflow.get("analyte"), namespace=attempt["attempt_id"])
        store.publish(attempt, steps)
        if _current.get() is None:
            store.finish(attempt, "succeeded", {"stage": "review"})
        return {"run_id": attempt["run_id"], "attempt_id": attempt["attempt_id"], "evidence_status": "available"}
    except Exception:
        logger.exception("Unable to publish NIR chart evidence")
        return {"evidence_status": "failed", "reason": "chart_persistence_failed"}


def finish_execution(runtime, execution, tool, call_id, payload, workflow, status):
    """Record only structured public facts; visualization cannot weaken guards."""
    try:
        if execution is None and os.getenv("NIR_PROCESS_ENABLED", "true").lower() == "false":
            return
        if execution:
            store, attempt = execution
            summary = store.summary(attempt["run_id"], attempt_id=attempt["attempt_id"])
            current = next((item for item in summary["attempts"] if item["attempt_id"] == attempt["attempt_id"]), None)
            if current is not None and not current["steps"]:
                store.publish(
                    attempt,
                    [
                        {
                            "step_key": "model",
                            "execution_status": status,
                            "evidence_status": "failed" if (payload or {}).get("process_evidence", {}).get("evidence_status") == "failed" else "not_recorded",
                            "facts": {"source_tool": tool, "reason": "chart_evidence_not_recorded"},
                        }
                    ],
                )
            store.finish(attempt, status, workflow)
        elif tool in TOOL_STEPS and workflow.get("project_id"):
            store = runtime_store(runtime)
            if store is not None:
                step_key = "registration" if tool == "nir_workflow" and workflow.get("stage") in {"review", "approved", "registered", "completed"} else TOOL_STEPS[tool]
                if tool == "nir_workflow":
                    facts = {key: workflow.get(key) for key in ("approval_status", "stage", "next_action", "task_type", "unit", "analyte")}
                    facts["registered"] = any(item.get("action") == "registered" for item in workflow.get("history", []))
                    facts["model_library"] = None
                else:
                    facts = _safe(payload or {})
                store.record_action(workflow["project_id"], call_id, step_key, status, {"stage": workflow.get("stage"), "result": facts, "source_tool": tool})
    except Exception:
        logger.exception("Unable to record NIR process outcome")
