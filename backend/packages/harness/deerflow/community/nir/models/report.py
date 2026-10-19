"""Model cards and reports from verified saved results, without fitting or loading models."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping

from deerflow.community.nir.autonomous_view import AutonomousView
from deerflow.community.nir.process import ProcessStore, _digest, _safe


def _object(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    return value if type(value) in {int, float} and math.isfinite(value) else None


def _parameters(value):
    value = _safe(_object(value))
    return {key: item for key, item in list(value.items())[:40] if item is None or type(item) in {str, int, float, bool}}


def _text(value):
    value = _safe(value)
    return value[:200] if isinstance(value, str) else None


def _indices(length, limit):
    return sorted({round(i * (length - 1) / max(1, min(length, limit) - 1)) for i in range(min(length, limit))})


def bounded_chart(source):
    """Retain finite, aligned recorded points and their actual evaluation scope."""
    if not isinstance(source, dict) or source.get("evidence_status", "available") != "available":
        return None
    kind, scope = source.get("type"), source.get("scope")
    if kind not in {"prediction", "spectra"} or scope not in {"calibration", "calibration_before", "calibration_after", "tuning", "validation", "holdout", "final_test"}:
        return None
    x = source.get("x")
    if not isinstance(x, list) or not 0 < len(x) <= 4000 or any(_number(item) is None for item in x):
        return None
    total = source.get("n_total")
    if type(total) is not int or total < 1:
        return None
    indices = _indices(len(x), 256 if kind == "spectra" else 500)
    result = {"type": kind, "scope": scope, "x": [x[i] for i in indices], "n_total": total}
    result.update({key: _safe(source.get(key)) for key in ("x_label", "x_unit", "y_unit")})
    if kind == "prediction":
        y = source.get("y")
        if not isinstance(y, list) or len(y) != len(x) or total < len(x) or any(_number(item) is None for item in y):
            return None
        result.update(y=[y[i] for i in indices], n_returned=len(indices))
    else:
        series = source.get("series")
        if not isinstance(series, list) or not 0 < len(series) <= 24 or total < len(series):
            return None
        if len(x) > 1 and not (all(a < b for a, b in zip(x, x[1:])) or all(a > b for a, b in zip(x, x[1:]))):
            return None
        curves = []
        for item in series[:12]:
            values = item.get("values") if isinstance(item, dict) else None
            if not isinstance(values, list) or len(values) != len(x) or any(_number(v) is None for v in values):
                return None
            curves.append({"id": str(item.get("id", len(curves)))[:80], "values": [values[i] for i in indices]})
        result.update(series=curves, n_returned=len(curves), axis_total=source.get("axis_total", len(x)))
    return result


def metrics_for_report(metrics):
    rows = []
    for scope, keys in (("calibration", ("cal", "train")), ("validation", ("val",)), ("final_test", ("test",))):
        source = next((_object(metrics[key]) for key in keys if isinstance(metrics.get(key), dict)), {})
        if source.get("scope") not in {None, scope}:
            continue
        values = {key: _number(source.get(key)) for key in ("R2", "RMSE", "MAE", "n")}
        # These historical summaries refer to different partitions: never pair them.
        if scope == "validation" and values["R2"] is None:
            values["R2"] = _number(metrics.get("R2_val"))
        if scope == "final_test":
            if values["R2"] is None:
                values["R2"] = _number(metrics.get("R2_test"))
            if values["RMSE"] is None:
                values["RMSE"] = _number(metrics.get("RMSEP"))
        if any(value is not None for value in values.values()):
            rows.append({"scope": scope, **values})
    return rows


def _saved_charts(metrics, source, process):
    """Read only an exact recorded call, or the finalized session bound in metrics."""
    charts = []
    audit = _object(metrics.get("autonomous_decision_audit"))
    if audit:
        view = AutonomousView(process)
        sid, eid = audit.get("session_id"), audit.get("retained_experiment_id")
        with view._read() as db:
            session = view._session(db, sid)
        freeze = _object(audit.get("freeze"))
        if session.get("stage") != "finalized" or session.get("search_hash") != metrics.get("training_data_hash") or session.get("freeze") != freeze or freeze.get("experiment_id") != eid:
            return []
        final = view.chart(sid, "final_test")
        if final.get("evidence_status") == "available":
            charts.append({**final, "type": "prediction", "x": final.get("observed"), "y": final.get("predicted")})
        spectra = view.spectra(sid, eid)
        if spectra.get("evidence_status") == "available" and spectra.get("model_hash") == freeze.get("model_hash"):
            charts.extend([spectra["before"], spectra["after"]])
    elif source.get("tool_call_id") and source.get("workflow_project_id"):
        # Derive IDs using the original process writer's exact identity function.
        run_id = "run_" + _digest(str(source["workflow_project_id"]))[:32]
        attempt_id = "attempt_" + _digest([run_id, source["tool_call_id"]])[:32]
        view = AutonomousView(process)
        with view._read() as db:
            if db is None:
                return []
            attempt = db.execute("SELECT status FROM attempts WHERE id=? AND run_id=?", (attempt_id, run_id)).fetchone()
            if attempt is None or attempt[0] != "succeeded":
                return []
            steps = [json.loads(row[0]) for row in db.execute("SELECT payload FROM steps WHERE attempt_id=? AND run_id=?", (attempt_id, run_id))]
            for step in steps:
                if step.get("step_key") not in {"preprocessing", "validation"}:
                    continue
                for ref in step.get("charts", [])[:8]:
                    charts.append(process.chart(run_id, ref["chart_id"], ref["sha256"]))
    return [chart for item in charts if (chart := bounded_chart(item)) is not None][:6]


def build_report(metrics: Mapping, source: dict, process_root=None):
    """Small portable snapshot, also compatible with old packages lacking charts."""
    reproducibility = _object(metrics.get("reproducibility"))
    config = _object(reproducibility.get("actual_configuration"))
    audit = _object(metrics.get("autonomous_decision_audit"))
    task = _object(audit.get("task"))
    report_context = _object(metrics.get("report_context"))
    parameters = _parameters(reproducibility.get("actual_parameters") or metrics.get("best_params") or metrics.get("model_parameters"))
    for key in ("n_components",):
        if _number(metrics.get(key)) is not None:
            parameters.setdefault(key, metrics[key])
    steps = metrics.get("preprocessing_steps", config.get("steps"))
    preprocessing = None
    if isinstance(steps, list):
        preprocessing = [{"method": str(step["method"])[:100], "params": _parameters(step.get("params"))} for step in steps[:21] if isinstance(step, dict) and isinstance(step.get("method"), str)]
    report = {
        "schema_version": 1,
        "target": _text(metrics.get("target") or task.get("analyte") or task.get("target") or report_context.get("analyte")),
        "unit": _text(metrics.get("unit") or task.get("unit") or report_context.get("unit")),
        "method": _text(metrics.get("method") or config.get("method")),
        "sample_count": _number(metrics.get("n_samples")),
        "feature_count": _number(metrics.get("n_wavelengths_original")),
        "model_feature_count": _number(metrics.get("n_wavelengths_model")),
        "parameters": parameters,
        "preprocessing": preprocessing,
        "metrics": metrics_for_report(metrics),
        "charts": [],
    }
    if process_root is not None:
        try:
            report["charts"] = _saved_charts(metrics, source, ProcessStore(process_root))
        except (KeyError, ValueError, OSError, TypeError, sqlite3.Error):
            pass  # Missing historical evidence cannot invalidate the saved model.
    return complete_dimensions(report, metrics)


def complete_dimensions(report, metrics):
    """Fill missing display counts from saved input/curve metadata, without refitting."""
    report = dict(report)
    summary = _object(_object(_object(metrics.get("scientific_validation")).get("dataset")).get("summary"))
    before = next((c.get("axis_total") for c in report.get("charts", []) if c.get("scope") == "calibration_before"), None)
    after = next((c.get("axis_total") for c in report.get("charts", []) if c.get("scope") == "calibration_after"), None)
    for key, value in (("feature_count", before or summary.get("n_wavelengths")), ("model_feature_count", after)):
        if report.get(key) is None and type(value) is int and 0 < value <= 20000:
            report[key] = value
    return report
