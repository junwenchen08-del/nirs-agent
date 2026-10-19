"""Bounded web projection of existing decisions, sessions and feedback.

SQLite is opened read-only. No array/model loader, training or finalization
service is called. Old records without chart evidence remain unavailable.
"""

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager

from .process import CHART_LIMIT, _digest, _id, _safe


def scoped_metrics(value, *, final=False):
    result = {}
    for key, scope in (("cal", "calibration"), ("cal_cv", "calibration_cv"), ("val", "validation"), ("test", "final_test")):
        if key == "test" and not final:
            continue
        item = (value or {}).get(key, (value or {}).get({"cal": "train", "cal_cv": "train_cv"}.get(key, key)))
        if isinstance(item, dict) and item.get("scope") == scope:
            result[key] = {"scope": scope, **{name: item[name] for name in ("R2", "RMSE", "MAE", "RPD", "bias", "n") if type(item.get(name)) in {int, float} and math.isfinite(item[name])}}
    return result


class AutonomousView:
    def __init__(self, process):
        self.process = process

    @contextmanager
    def _read(self):
        self.process._validate_root()
        if not self.process.db.exists():
            yield None
            return
        if self.process.db.is_symlink():
            raise ValueError("Unsafe process index")
        db = sqlite3.connect(self.process.db.as_uri() + "?mode=ro", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _session(db, sid):
        if not db or not db.execute("SELECT 1 FROM sqlite_master WHERE name='search_sessions'").fetchone():
            raise KeyError(sid)
        row = db.execute("SELECT payload FROM search_sessions WHERE id=?", (_id(sid),)).fetchone() if db else None
        if row is None:
            raise KeyError(sid)
        session = json.loads(row[0])
        if session.get("decision_strategy") != "llm_autonomous":
            raise KeyError(sid)
        return session

    def sessions(self, offset=0):
        with self._read() as db:
            if not db or not db.execute("SELECT 1 FROM sqlite_master WHERE name='search_sessions'").fetchone():
                return {"data": [], "next_offset": None}
            rows = db.execute("SELECT project,payload FROM search_sessions WHERE json_extract(payload, '$.decision_strategy')='llm_autonomous' ORDER BY rowid DESC LIMIT 21 OFFSET ?", (offset,)).fetchall()
        data = []
        for row in rows:
            item = json.loads(row["payload"])
            if item.get("decision_strategy") == "llm_autonomous":
                data.append({"session_id": item["id"], "stage": item["stage"], "project_id": row["project"]})
        return {"data": data[:20], "next_offset": offset + 20 if len(data) > 20 else None}

    @staticmethod
    def _record(row, *, details=False):
        value = json.loads(row["payload"])
        proposal = value["proposal"]
        execution = json.loads(row["execution"]) if row["execution"] else None
        selected = next((item for item in proposal["alternatives"] if item["id"] == proposal["selected_alternative_id"]), None)
        result = {key: value.get(key) for key in ("decision_id", "context_id", "experiment_id", "round_index", "action", "decision_time", "base_experiment_id", "retained_experiment_id", "reserved_fits")}
        # STOP/RETAIN keep these identities in their immutable proposal rather
        # than the outer training record. Project the original recorded links.
        for key in ("base_experiment_id", "retained_experiment_id"):
            if key not in value:
                result[key] = _safe(proposal.get(key))
        result.update(status=row["status"], selected=_safe(selected), rationale=_safe(proposal["rationale"]), expected_outcome=_safe(proposal["expected_outcome"]), alternative_count=len(proposal["alternatives"]))
        origin = value["origin"]
        result["origin"] = {key: origin.get(key) for key in ("source", "model_name", "model_class", "verification_kind", "issued_at", "agent_run_id")}
        if execution:
            keys = (
                "feedback_id",
                "actual_configuration",
                "actual_parameters",
                "fits",
                "elapsed_seconds",
                "error_type",
                "failure_code",
                "failure_reason",
                "comparisons",
                "expected_improvement",
                "retained_experiment_id",
                "stop_source",
                "cv_settings",
            )
            result["feedback"] = _safe({key: execution[key] for key in keys if key in execution})
            result["feedback"]["metrics"] = scoped_metrics(execution.get("metrics"))
            result["diagnostics_status"] = "available" if execution.get("diagnostics_sha256") else "failed" if execution.get("diagnostics_error_type") else "not_recorded"
            if details:
                result["feedback"]["trials"] = _safe(execution.get("trials", []))
        if details:
            result["proposal"] = _safe(proposal)
            result["accepted_spec"] = _safe(value.get("accepted_spec"))
        return result

    def decisions(self, sid, offset=0, limit=10):
        with self._read() as db:
            self._session(db, sid)
            total = db.execute("SELECT count(*) FROM initial_decisions WHERE session_id=?", (sid,)).fetchone()[0]
            rows = db.execute("SELECT * FROM initial_decisions WHERE session_id=? ORDER BY rowid LIMIT ? OFFSET ?", (sid, limit, offset)).fetchall()
        return {"session_id": sid, "data": [self._record(row) for row in rows], "total": total, "offset": offset, "next_offset": offset + limit if total > offset + limit else None}

    def decision(self, sid, did):
        with self._read() as db:
            self._session(db, sid)
            row = db.execute("SELECT * FROM initial_decisions WHERE session_id=? AND id=?", (sid, _id(did))).fetchone()
            following = db.execute("SELECT * FROM initial_decisions WHERE session_id=? AND rowid > (SELECT rowid FROM initial_decisions WHERE session_id=? AND id=?) ORDER BY rowid LIMIT 1", (sid, sid, did)).fetchone()
        if not row:
            raise KeyError(did)
        result = self._record(row, details=True)
        result["next_decision"] = self._record(following) if following else None
        return result

    def rejections(self, sid, offset=0):
        with self._read() as db:
            self._session(db, sid)
            # Each row is a bounded official process action, never a receipt
            # inferred as a successful submission or an experimental result.
            rows = db.execute("SELECT payload FROM steps WHERE json_extract(payload,'$.facts.session_id')=? AND json_extract(payload,'$.execution_status')='rejected' ORDER BY rowid LIMIT 11 OFFSET ?", (sid, offset)).fetchall()
        data = [json.loads(row[0])["facts"]["decision_rejection"] for row in rows]
        return {"session_id": sid, "data": data[:10], "next_offset": offset + 10 if len(data) > 10 else None}

    def summary(self, sid):
        with self._read() as db:
            session = self._session(db, sid)
            rows = db.execute("SELECT * FROM initial_decisions WHERE session_id=? ORDER BY rowid", (sid,)).fetchall()
            plotted = {row["id"] for row in db.execute("SELECT id FROM search_experiments WHERE session_id=? AND status='succeeded' AND json_extract(result,'$.spectra_sha256') IS NOT NULL", (sid,)).fetchall()}
        records = [self._record(row) for row in rows]
        successes = [item for item in records if item["status"] == "succeeded"]
        trends = [
            {
                "experiment_id": item["experiment_id"],
                "decision_id": item["decision_id"],
                "round_index": item["round_index"],
                "metrics": item["feedback"]["metrics"],
                "method": (item["feedback"].get("actual_configuration") or {}).get("method"),
            }
            for item in successes
        ]
        val_scored = [item for item in trends if "RMSE" in item["metrics"].get("val", {})]
        # Informative marker only. Retention is read independently from the LLM
        # decision and is never inferred from this minimum.
        best = min(val_scored, key=lambda item: item["metrics"]["val"]["RMSE"])["experiment_id"] if val_scored else None
        final = session.get("final") if session["stage"] == "finalized" else None
        export = session.get("export") or {}
        policy = session.get("delivery_policy") or {}
        budget = {
            "fits_used": session["fits"],
            "experiments_used": session["experiments"],
            "actions_used": len(records),
            "fits_reserved": session.get("reserved_fits", 0),
            "limits": {key: policy.get(key) for key in ("max_fits", "max_experiments", "max_decisions", "max_wall_seconds")},
            "llm": session.get("llm_budget"),
            "execution_seconds": session.get("execution_seconds", 0),
            "provider_cost": session.get("provider_cost_snapshot"),
        }
        downloads = []
        if export and session.get("export_state", {}).get("status") == "succeeded":
            for key in ("model", "report", "report_html", "bundle", "metrics_path", "decisions"):
                path = export.get(key, "")
                if path.startswith("/mnt/user-data/outputs/") and ".." not in path.split("/"):
                    downloads.append({"name": path.rsplit("/", 1)[-1], "path": path.lstrip("/")})
        return {
            "schema_version": 1,
            "session_id": sid,
            "revision": _digest([session, [dict(row) for row in rows]])[:24],
            "stage": session["stage"],
            "test_status": "consumed" if session["stage"] in {"testing", "finalized", "final_test_failed"} else "sealed",
            "task": _safe({key: value for key, value in session.get("task_snapshot", {}).items() if key != "_workflow_snapshot"}),
            "budget": budget,
            "stop_source": session.get("stop_source"),
            "stop_reason": _safe(session.get("stop_reason")),
            "terminal_failure_source": session.get("terminal_failure_source"),
            "terminal_failure_reason": _safe(session.get("terminal_failure_reason")),
            "provider_failure": _safe(session.get("provider_failure")),
            "retained_experiment_id": session.get("retained_experiment_id"),
            "retained_selection_source": session.get("retained_selection_source"),
            "best_val_experiment_id": best,
            "active_decision_id": session.get("active_decision_id"),
            "decision_count": len(records),
            "comparison_group_id": _digest([sid, session["binding"], session["cv_manifest"]]),
            "data_context": _safe(
                {
                    "partition_counts": session["counts"],
                    "cv_folds": session["cv_folds"],
                    "calibration_samples": session["profile"].get("n_samples"),
                    "n_features": session["profile"].get("n_wavelengths"),
                    "unit": session.get("task_snapshot", {}).get("unit"),
                }
            ),
            "trends": trends,
            "metrics": scoped_metrics((final or {}).get("metrics"), final=bool(final)),
            "quality": _safe((final or {}).get("metrics", {}).get("quality")),
            "freeze": _safe({key: session.get("freeze", {}).get(key) for key in ("experiment_id", "decision_id", "stop_decision_id", "parameters", "configuration", "frozen_at", "fit_scope")}) if session.get("freeze") else None,
            "lifecycle_at_delivery": export.get("lifecycle"),
            "lifecycle": export.get("lifecycle"),
            "export_status": session.get("export_state", {}).get("status"),
            "automatic_delivery": _safe({key: session.get("automatic_delivery", {}).get(key) for key in ("status", "events", "error_type", "terminal_decision_id", "retained_experiment_id")}),
            "autonomous_delivery": bool(session.get("delivery_policy")),
            "downloads": downloads,
            "chart_scopes": list((final or {}).get("charts", {})),
            "spectra_status": "available" if plotted else "not_recorded",
            "spectra_experiments": [{key: item[key] for key in ("experiment_id", "decision_id", "round_index", "method")} for item in trends if item["experiment_id"] in plotted],
        }

    def spectra(self, sid, eid):
        """Owner/thread/session-bound cached Cal evidence; no numeric IO."""
        from ._common import _read_regular_bytes

        with self._read() as db:
            self._session(db, sid)
            row = db.execute("SELECT result,artifact_hash FROM search_experiments WHERE session_id=? AND id=? AND status='succeeded'", (sid, _id(eid))).fetchone()
        if not row:
            raise KeyError(eid)
        result = json.loads(row["result"])
        missing = {"scope": "calibration", "experiment_id": eid, "evidence_status": "not_recorded"}
        if not result.get("spectra_sha256"):
            return missing
        try:
            path = self.process.root / "sessions" / _id(sid) / (_id(eid) + ".spectra.json")
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("Unsafe Cal chart cache")
            raw = _read_regular_bytes(path, label="Cal chart", limit=CHART_LIMIT)
            payload = json.loads(raw)
            if (
                hashlib.sha256(raw).hexdigest() != result["spectra_sha256"]
                or payload.get("scope") != "calibration"
                or payload.get("session_id") != sid
                or payload.get("experiment_id") != eid
                or payload.get("model_hash") != row["artifact_hash"]
            ):
                raise ValueError("Cal chart integrity mismatch")
            return payload
        except (OSError, ValueError):
            return {**missing, "evidence_status": "unavailable", "reason": "missing_or_corrupt_evidence"}

    def diagnostics(self, sid, eid):
        """Serve only the immutable display cache for this exact experiment."""
        from ._common import _read_regular_bytes
        from .round_diagnostics import validate_round_diagnostics

        with self._read() as db:
            session = self._session(db, sid)
            row = db.execute("SELECT result,artifact_hash FROM search_experiments WHERE session_id=? AND id=? AND status='succeeded'", (sid, _id(eid))).fetchone()
        if not row:
            raise KeyError(eid)
        result = json.loads(row["result"])
        missing = {"schema_version": 1, "session_id": sid, "experiment_id": eid, "evidence_status": "failed" if result.get("diagnostics_error_type") else "not_recorded"}
        if not result.get("diagnostics_sha256"):
            return missing
        try:
            path = self.process.root / "sessions" / _id(sid) / (_id(eid) + ".diagnostics.json")
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("Unsafe round chart cache")
            raw = _read_regular_bytes(path, label="Round diagnostics", limit=CHART_LIMIT)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or (
                hashlib.sha256(raw).hexdigest() != result["diagnostics_sha256"]
                or payload.get("session_id") != sid
                or payload.get("experiment_id") != eid
                or payload.get("decision_id") != result.get("decision_id")
                or payload.get("model_hash") != row["artifact_hash"]
                or payload.get("comparison_group_id") != _digest([sid, session["binding"], session["cv_manifest"]])
            ):
                raise ValueError("Round chart integrity mismatch")
            validate_round_diagnostics(payload)
            for scope, chart in payload["charts"].items():
                if chart["n_total"] != session["counts"]["validation" if scope == "validation" else "calibration"]:
                    raise ValueError("Round chart population differs from the locked split")
            return payload
        except (OSError, ValueError, TypeError, KeyError):
            return {**missing, "evidence_status": "unavailable", "reason": "missing_or_corrupt_evidence"}

    def chart(self, sid, scope):
        if scope != "final_test":
            raise KeyError(scope)
        with self._read() as db:
            session = self._session(db, sid)
        if session["stage"] != "finalized":
            raise KeyError("Test remains sealed")
        return session.get("final", {}).get("charts", {}).get(scope, {"scope": scope, "evidence_status": "not_recorded", "n_total": 0, "n_returned": 0})
