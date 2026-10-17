"""Private, immutable split sessions sharing the existing ProcessStore index.

Only trusted ingestion sees all three partitions. Search reads search.npz;
the holdout file can be opened only after a durable, single-use final lease.
"""

from __future__ import annotations

import json
import time
import uuid

import joblib
import numpy as np

from ._common import _sha256_file
from .process import ProcessStore, _digest, _id, _json, _now


class SearchSessionStore:
    def __init__(self, process: ProcessStore):
        self.process = process

    def _dir(self, session_id):
        path = self.process.root / "sessions" / _id(session_id)
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Unsafe private session path")
        return path

    def _session(self, db, session_id):
        row = db.execute("SELECT payload FROM search_sessions WHERE id=?", (_id(session_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown owner/thread search session")
        return json.loads(row[0])

    @staticmethod
    def _save(db, session):
        db.execute("UPDATE search_sessions SET payload=? WHERE id=?", (_json(session), session["id"]))

    @staticmethod
    def _view(session):
        return {
            "calibration_profile": session["profile"],
            "partition_counts": session["counts"],
            "session_id": session["id"],
            "stage": session["stage"],
            "cv_folds": session["cv_folds"],
            "test_status": "consumed" if session["stage"] in {"testing", "finalized", "final_test_failed"} else "sealed",
            "supported_models": ["pls", "ridge", "svr", "et"],
        }

    def prepare(self, project: str, partitions: dict, *, cv_folds=5, max_experiments=12, max_fits=200, domain="default", decision_strategy="legacy", autonomous_rounds=False, max_decisions=20, delivery_policy=None, task_snapshot=None):
        """Once-only trusted ingestion. Returned metadata has no Test statistics."""
        from nir_core.utils.scientific_validation import _array_digest, validate_calibration_dataset, validate_partition_separation

        required = ("X_cal", "y_cal", "X_val", "y_val", "X_test", "y_test")
        arrays = {key: np.asarray(partitions[key], dtype=float) for key in required}
        wv = partitions.get("wv")
        axis_unit = str(np.asarray(partitions["wv_unit"]).item()) if "wv_unit" in partitions else None
        if axis_unit is not None and (axis_unit not in {"nm", "cm^-1"} or wv is None):
            raise ValueError("wv_unit must declare nm or cm^-1 alongside an explicit wavelength axis")
        if wv is not None:
            arrays["wv"] = np.asarray(wv, dtype=float)
        for name in ("cal", "val", "test"):
            X, y = arrays["X_" + name], arrays["y_" + name]
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
                arrays["y_" + name] = y
            if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
                raise ValueError("Invalid single-target partition shape or numeric integrity")
        if len({arrays[key].shape[1] for key in ("X_cal", "X_val", "X_test")}) != 1:
            raise ValueError("Partition wavelength dimensions differ")
        for name, minimum in (("cal", 6), ("val", 2)):
            gate = validate_calibration_dataset(arrays["X_" + name], arrays["y_" + name], wv, minimum_samples=minimum)
            if not gate["passed"]:
                raise ValueError("Scientific Cal/Val integrity checks failed")
        if len(arrays["y_test"]) < 2:
            raise ValueError("At least two final-test samples are required")
        separation = validate_partition_separation({name: arrays["X_" + name] for name in ("cal", "val", "test")})
        if not separation["passed"]:
            raise ValueError("Scientific partition separation failed")
        group_keys = ("groups_cal", "groups_val", "groups_test")
        if any(key in partitions for key in group_keys):
            if not all(key in partitions for key in group_keys):
                raise ValueError("Groups must be specified for all partitions")
            group_sets = []
            for name, key in zip(("cal", "val", "test"), group_keys, strict=True):
                group = np.asarray(partitions[key], dtype=str)
                if group.ndim != 1 or len(group) != len(arrays["y_" + name]):
                    raise ValueError("Misaligned partition groups")
                arrays[key] = group
                group_sets.append(set(group))
            if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError("Groups cross partition boundaries")
            if len(group_sets[0]) < cv_folds:
                raise ValueError("Insufficient calibration groups for frozen CV")
        if delivery_policy is not None:
            from .decision_budget import SERVER_DEFAULTS, resolve_policy

            checked = resolve_policy(delivery_policy["profile"], {key: delivery_policy[key] for key in SERVER_DEFAULTS})
            if checked != delivery_policy or decision_strategy != "llm_autonomous" or not autonomous_rounds:
                raise ValueError("Delivery policy requires a server-validated multi-round autonomous session")
            max_experiments, max_fits, max_decisions = (checked[key] for key in ("max_experiments", "max_fits", "max_decisions"))
        if not 2 <= cv_folds <= min(10, len(arrays["y_cal"])) or (delivery_policy is None and (not 1 <= max_experiments <= 12 or not 1 <= max_fits <= 200)):
            raise ValueError("Invalid session CV or resource bounds")
        source_digest = _array_digest(*(arrays[key] for key in sorted(arrays)))
        from nir_core.preprocess.recommendation import profile_preprocessing_data

        profile = {**profile_preprocessing_data(arrays["X_cal"], wv).as_dict(), "scope": "calibration"}
        if axis_unit:
            profile["wavelength_unit"] = {"value": axis_unit, "source": "declared_input_metadata"}
        if decision_strategy not in {"legacy", "llm_autonomous"}:
            raise ValueError("Unsupported decision strategy")
        if autonomous_rounds and decision_strategy != "llm_autonomous" or (delivery_policy is None and not 1 <= max_decisions <= 24):
            raise ValueError("Invalid multi-round decision policy")
        binding_values = [source_digest, cv_folds, max_experiments, max_fits, domain]
        if axis_unit:
            binding_values.append({"wv_unit": axis_unit})
        binding = _digest(binding_values + ([decision_strategy] if decision_strategy != "legacy" else []))
        if autonomous_rounds:
            binding = _digest([binding, "autonomous_rounds", max_decisions])
        if delivery_policy:
            binding = _digest([binding, "autonomous_delivery", delivery_policy, task_snapshot or {}])
        from nir_core.model.safe_search import make_cv_manifest

        cv_manifest = make_cv_manifest(arrays["X_cal"], cv_folds=cv_folds, groups=arrays.get("groups_cal"))
        with self.process._connection(write=True) as db:
            previous = db.execute("SELECT payload FROM search_sessions WHERE project=?", (project,)).fetchone()
            if previous:
                session = json.loads(previous[0])
                if session["binding"] != binding:
                    raise ValueError("Dataset and split are already locked for this project")
                return self._view(session)
            session_id = "search_" + uuid.uuid4().hex
            path = self._dir(session_id)
            path.mkdir(parents=True, exist_ok=False)
            search = {key: value for key, value in arrays.items() if key not in {"X_test", "y_test", "groups_test"}}
            with (path / "search.npz").open("xb") as file:
                np.savez(file, **search)
            with (path / "holdout.npz").open("xb") as file:
                np.savez(file, X_test=arrays["X_test"], y_test=arrays["y_test"])
            session = {
                "id": session_id,
                "stage": "searching",
                "binding": binding,
                "search_hash": _sha256_file(str(path / "search.npz")),
                "holdout_hash": _sha256_file(str(path / "holdout.npz")),
                "cv_folds": cv_folds,
                "max_experiments": max_experiments,
                "max_fits": max_fits,
                "fits": 0,
                "experiments": 0,
                "freeze": None,
                "final": None,
                "domain": domain,
                "decision_strategy": decision_strategy,
                "autonomous_rounds": autonomous_rounds,
                "max_decisions": max_decisions,
                "cv_manifest": cv_manifest,
                "profile": profile,
                "counts": {"calibration": len(arrays["y_cal"]), "validation": len(arrays["y_val"])},
                "spectral_axis_unit": axis_unit,
            }
            if delivery_policy:
                from .decision_budget import INTAKE_ID, initial_ledger

                session.update(
                    delivery_policy=delivery_policy,
                    budget_started_at=time.time(),
                    llm_budget=initial_ledger(),
                    invalid_decisions=0,
                    task_snapshot=task_snapshot or {},
                    scientific_validation={"schema_version": 1, "passed": True, "dataset": validate_calibration_dataset(arrays["X_cal"], arrays["y_cal"], wv, minimum_samples=6), "partition_separation": separation},
                )
                starter = (task_snapshot or {}).get("workflow_start_call_id")
                intake_row = db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (INTAKE_ID,)).fetchone()
                if intake_row:
                    intake = json.loads(intake_row[0])
                    if intake["budget"]["inflight"] or intake["stage"] != "intake":
                        raise ValueError("intake: closed or uncertain provider request cannot prepare search")
                    session.update(llm_budget=intake["budget"], budget_started_at=intake["started_at"], intake_model_calls=intake["budget"]["search"]["calls"])
                    intake.update(stage="bound", session_id=session_id)
                    db.execute("UPDATE model_issued_calls SET payload=? WHERE id=?", (_json(intake), INTAKE_ID))
                if starter and not (intake_row and delivery_policy.get("profile") == "unified"):
                    intake = [json.loads(row[0]) for row in db.execute("SELECT payload FROM model_issued_calls WHERE tool='__model_response__'")]
                    intake = [call for call in intake if not call.get("session_id") and call.get("workflow_start_call_id") == starter and not call.get("intake_lease_id")]
                    bucket = session["llm_budget"]["search"]
                    for call in intake:
                        usage = call.get("token_usage") or {}
                        bucket["calls"] += 1
                        bucket["actual_input_tokens"] += usage.get("input_tokens", 0)
                        bucket["actual_output_tokens"] += usage.get("output_tokens", 0)
                        bucket["charged_tokens"] += max(call.get("admission_tokens_estimated") or 0, usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
                        bucket["unknown_usage_calls"] += int(not bool(usage))
                        bucket["seconds"] += call.get("elapsed_seconds") or 0.0
                    session["intake_model_calls"] = session.get("intake_model_calls", 0) + len(intake)
                    session["budget_started_at"] -= sum(call.get("elapsed_seconds") or 0.0 for call in intake)
            db.execute("INSERT INTO search_sessions VALUES (?,?,?)", (session_id, project, _json(session)))
            return self._view(session)

    def search(self, session_id, call_id, *, method, steps, max_components=10, parameter_plan=None, decision_id=None, checkpoint=lambda _: None):
        from nir_core.model.safe_search import fit_search_model
        from nir_core.models import PreprocessingStep
        from nir_core.utils.metrics import evaluate_quality

        parsed = [step if isinstance(step, PreprocessingStep) else PreprocessingStep(**step) for step in steps]
        spec = {"method": method, "steps": [step.model_dump() for step in parsed], "max_components": max_components}
        if parameter_plan is not None:
            spec["parameter_plan"] = parameter_plan
        experiment_id = "exp_" + _digest([session_id, call_id])[:32]
        with self.process._connection(write=True) as db:
            session = self._session(db, session_id)
            if session["stage"] != "searching":
                raise ValueError("Search is closed after freeze or final-test consumption")
            if session.get("decision_strategy") == "llm_autonomous":
                row = db.execute("SELECT payload,status FROM initial_decisions WHERE id=? AND session_id=?", (decision_id, session_id)).fetchone()
                if row is None or row["status"] != "executing" or call_id != decision_id or json.loads(row["payload"])["accepted_spec"] != spec:
                    raise ValueError("Autonomous search requires an accepted decision execution with exactly matching configuration")
            prior = db.execute("SELECT * FROM search_experiments WHERE id=?", (experiment_id,)).fetchone()
            if prior:
                if json.loads(prior["spec"]) != spec:
                    raise ValueError("Conflicting experiment replay")
                if prior["status"] == "succeeded":
                    return json.loads(prior["result"])
                raise ValueError("Experiment already consumed; failed/running replay cannot refit")
            if session["experiments"] >= session["max_experiments"]:
                raise ValueError("Search experiment budget exhausted")
            session["experiments"] += 1
            self._save(db, session)
            db.execute("INSERT INTO search_experiments VALUES (?,?,'running',?,NULL,NULL)", (experiment_id, session_id, _json(spec)))
        path = self._dir(session_id)

        def before_fit(stage):
            checkpoint(stage)
            with self.process._connection(write=True) as db:
                current = self._session(db, session_id)
                if current["stage"] != "searching":
                    raise ValueError("Search is closed")
                if current["fits"] >= current["max_fits"]:
                    raise ValueError("Search fit budget exhausted")
                if current.get("delivery_policy"):
                    from .decision_budget import wall_exhausted

                    if wall_exhausted(current):
                        raise ValueError("budget: cumulative cooperative wall deadline exceeded")
                current["fits"] += 1
                self._save(db, current)

        try:
            if _sha256_file(str(path / "search.npz")) != session["search_hash"]:
                raise ValueError("Search data integrity failure")
            with np.load(path / "search.npz", allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            fitted = fit_search_model(
                data["X_cal"],
                data["y_cal"],
                data["X_val"],
                data["y_val"],
                method=method,
                steps=parsed,
                wv=data.get("wv"),
                groups=data.get("groups_cal"),
                cv_folds=session["cv_folds"],
                max_components=max_components,
                parameter_plan=parameter_plan,
                cv_manifest=session.get("cv_manifest"),
                checkpoint=before_fit,
            )
            fitted.metrics["domain"] = session["domain"]
            fitted.metrics["quality"] = evaluate_quality(fitted.metrics, domain=session["domain"], evaluation_scope="validation")
            artifact = path / (experiment_id + ".joblib")
            with artifact.open("xb") as file:
                joblib.dump(fitted.model, file)
            digest = _sha256_file(str(artifact))
            result = {"experiment_id": experiment_id, "metrics": fitted.metrics, "parameters": fitted.parameters, "trials": fitted.trials, "fits": fitted.fits, "configuration": spec, "cv_manifest": fitted.cv_manifest}
            if session.get("decision_strategy") == "llm_autonomous":
                from .cal_spectra import capture_cal_spectra
                from .process import CHART_LIMIT

                try:
                    curves = capture_cal_spectra(
                        data["X_cal"], data.get("wv"), fitted.model, session_id=session_id, experiment_id=experiment_id, decision_id=decision_id, configuration=spec, model_hash=digest, axis_unit=session.get("spectral_axis_unit")
                    )
                    raw = _json(curves).encode("utf-8")
                    if len(raw) > CHART_LIMIT:
                        raise ValueError("Cal plot cache exceeds bounded chart size")
                    with (path / (experiment_id + ".spectra.json")).open("xb") as file:
                        file.write(raw)
                    result["spectra_sha256"] = _sha256_file(str(path / (experiment_id + ".spectra.json")))
                except Exception as exc:
                    # A display failure never changes the chosen model or runs
                    # a fallback experiment. Preserve the numeric success.
                    result["spectra_error_type"] = type(exc).__name__
                from .round_diagnostics import capture_round_diagnostics

                try:
                    diagnostics = capture_round_diagnostics(
                        data["y_cal"],
                        data["y_val"],
                        fitted.predictions,
                        session_id=session_id,
                        experiment_id=experiment_id,
                        decision_id=decision_id,
                        model_hash=digest,
                        comparison_group_id=_digest([session_id, session["binding"], session["cv_manifest"]]),
                        unit=session.get("task_snapshot", {}).get("unit"),
                    )
                    raw = _json(diagnostics).encode("utf-8")
                    if len(raw) > CHART_LIMIT:
                        raise ValueError("Round diagnostics exceed bounded chart size")
                    with (path / (experiment_id + ".diagnostics.json")).open("xb") as file:
                        file.write(raw)
                    result["diagnostics_sha256"] = _sha256_file(str(path / (experiment_id + ".diagnostics.json")))
                except Exception as exc:
                    result["diagnostics_error_type"] = type(exc).__name__
            if decision_id is not None:
                result["decision_id"] = decision_id
            with self.process._connection(write=True) as db:
                db.execute("UPDATE search_experiments SET status='succeeded',result=?,artifact_hash=? WHERE id=?", (_json(result), digest, experiment_id))
            return json.loads(_json(result))
        except BaseException:
            with self.process._connection(write=True) as db:
                db.execute("UPDATE search_experiments SET status='failed' WHERE id=?", (experiment_id,))
            raise

    def freeze(self, session_id, experiment_id):
        """Freeze the caller's selected successful experiment; never auto-select."""
        with self.process._connection(write=True) as db:
            session = self._session(db, session_id)
            if session["freeze"] and session["freeze"]["experiment_id"] == experiment_id:
                path = self._dir(session_id) / (experiment_id + ".joblib")
                if _sha256_file(str(path)) != session["freeze"]["model_hash"]:
                    raise ValueError("Frozen artifact integrity failure")
                return session["freeze"]
            delivery = bool(session.get("delivery_policy"))
            if delivery and (session["stage"] != "stopped" or session.get("retained_experiment_id") != experiment_id):
                raise ValueError("An accepted LLM STOP or budget retention must bind this successful experiment")
            if not delivery and session["stage"] != "searching":
                raise ValueError("Search is closed")
            running = db.execute("SELECT 1 FROM search_experiments WHERE session_id=? AND status='running'", (session_id,)).fetchone()
            row = db.execute("SELECT * FROM search_experiments WHERE id=? AND session_id=? AND status='succeeded'", (_id(experiment_id), session_id)).fetchone()
            if running or not row:
                raise ValueError("Cannot freeze an unfinished or unknown experiment")
            extra = {}
            if delivery:
                extra = self._validate_delivery_freeze(db, session, row)
            session["stage"] = "frozen"
            session["freeze"] = {"experiment_id": experiment_id, "model_hash": row["artifact_hash"], **extra}
            self._save(db, session)
            return session["freeze"]

    def _validate_delivery_freeze(self, db, session, row):
        from nir_core.model.parameter_control import authorized_candidates
        from nir_core.models import PreprocessingStep
        from sklearn.pipeline import Pipeline

        from .decision_context import _runtime_snapshot

        sid = session["id"]
        stop = db.execute("SELECT * FROM initial_decisions WHERE id=? AND session_id=?", (session.get("stop_decision_id"), sid)).fetchone()
        active = db.execute("SELECT 1 FROM initial_decisions WHERE session_id=? AND status IN ('accepted','executing')", (sid,)).fetchone()
        if not stop or stop["status"] != "stopped" or active or session["llm_budget"]["inflight"]:
            raise ValueError("Freeze requires a durable terminal choice and no unfinished execution")
        terminal = json.loads(stop["payload"])
        if terminal["proposal"]["retained_experiment_id"] != row["id"] or terminal["origin"]["source"] != "lead_agent_model_call":
            raise ValueError("Terminal choice identity/provenance mismatch")
        selected = json.loads(row["result"])
        decision = db.execute("SELECT * FROM initial_decisions WHERE id=? AND session_id=? AND status='succeeded'", (selected.get("decision_id"), sid)).fetchone()
        if not decision:
            raise ValueError("Selected experiment has no successful accepted decision")
        record = json.loads(decision["payload"])
        feedback = json.loads(decision["execution"])
        spec = json.loads(row["spec"])
        if record["accepted_spec"] != spec or selected["configuration"] != spec or feedback["actual_configuration"] != spec or selected["parameters"] != feedback["actual_parameters"]:
            raise ValueError("Actual configuration differs from accepted decision")
        if selected["cv_manifest"] != session["cv_manifest"] or record["cv_manifest_id"] != session["cv_manifest"]["manifest_id"]:
            raise ValueError("Frozen CV manifest identity mismatch")
        context = json.loads(db.execute("SELECT payload FROM decision_contexts WHERE id=?", (record["context_id"],)).fetchone()[0])
        if context["evaluation"]["search_data_sha256"] != session["search_hash"]:
            raise ValueError("Frozen split identity mismatch")
        completed = [json.loads(item[0]) for item in db.execute("SELECT execution FROM initial_decisions WHERE session_id=? AND status IN ('succeeded','failed')", (sid,))]
        count = db.execute("SELECT count(*) FROM search_experiments WHERE session_id=?", (sid,)).fetchone()[0]
        if sum(item["fits"] for item in completed) != session["fits"] or count != session["experiments"]:
            raise ValueError("Budget ledger differs from persisted executions")
        artifact = self._dir(sid) / (row["id"] + ".joblib")
        if _sha256_file(str(artifact)) != row["artifact_hash"] or _sha256_file(str(self._dir(sid) / "search.npz")) != session["search_hash"]:
            raise ValueError("Model/search artifact integrity failure")
        model = joblib.load(artifact)
        estimator = model.estimator.steps[-1][1] if isinstance(model.estimator, Pipeline) else model.estimator
        parameters = estimator.get_params(deep=False)
        steps = [step.model_dump() for step in model.preprocessor.steps]
        if _json(parameters) != _json(selected["parameters"]) or steps != spec["steps"]:
            raise ValueError("Persisted model parameters/preprocessing do not match execution evidence")
        if record["runtime_snapshot"] != _runtime_snapshot([PreprocessingStep(**step) for step in steps]):
            raise ValueError("Runtime adapter changed; no replacement model may be frozen")
        upper = min(20, session["profile"]["n_wavelengths"], min(len(fold["train"]) - 1 for fold in session["cv_manifest"]["folds"]))
        candidates = authorized_candidates(spec["method"], spec["parameter_plan"], component_upper=upper, cv_folds=session["cv_folds"])
        if not any(all(parameters[key] == value for key, value in candidate.items()) for candidate in candidates):
            raise ValueError("Actual model lies outside accepted parameter region")
        task = session["task_snapshot"]
        if task.get("model_constraint") and spec["method"] != task["model_constraint"] or task.get("pipeline_constraint") is not None and steps != json.loads(task["pipeline_constraint"]):
            raise ValueError("Retained model violates user hard constraints")
        return {
            "stop_decision_id": terminal["decision_id"],
            "decision_id": record["decision_id"],
            "configuration": spec,
            "parameters": parameters,
            "cv_manifest_id": record["cv_manifest_id"],
            "search_data_sha256": session["search_hash"],
            "preprocessing_state_sha256": row["artifact_hash"],
            "preprocessing_state_hash_scope": "serialized_full_pipeline_artifact",
            "fit_scope": "calibration",
            "frozen_at": _now(),
            "stop_source": session["stop_source"],
        }

    def finalize(self, session_id, *, checkpoint=lambda _: None):
        """Acquire the only Test lease before IO; uncertain failure stays closed."""
        from nir_core.utils.metrics import compute_metrics, evaluate_quality

        with self.process._connection(write=True) as db:
            session = self._session(db, session_id)
            if session["stage"] == "finalized":
                return session["final"]
            if session["stage"] in {"testing", "final_test_failed"}:
                raise ValueError("Final test lease already consumed")
            if session["stage"] != "frozen":
                raise ValueError("A selected model must be frozen before final test")
            checkpoint("before_final_test_lease")
            session["stage"] = "testing"
            self._save(db, session)
            row = db.execute("SELECT result FROM search_experiments WHERE id=?", (session["freeze"]["experiment_id"],)).fetchone()
            selected = json.loads(row[0])
        path = self._dir(session_id)
        try:
            artifact = path / (session["freeze"]["experiment_id"] + ".joblib")
            if _sha256_file(str(artifact)) != session["freeze"]["model_hash"] or _sha256_file(str(path / "holdout.npz")) != session["holdout_hash"]:
                raise ValueError("Frozen model/holdout integrity failure")
            model = joblib.load(artifact)
            with np.load(path / "holdout.npz", allow_pickle=False) as archive:
                checkpoint("final_test_prediction")
                observed = archive["y_test"].ravel()
                predicted = model.predict(archive["X_test"]).ravel()
                test = compute_metrics(observed, predicted)
                # Capture plot evidence during the original single-use Test
                # lease. Read-only web requests never reopen the holdout/model.
                indices = np.linspace(0, len(observed) - 1, min(len(observed), 2000), dtype=int)
                chart = {
                    "scope": "final_test",
                    "evidence_status": "available",
                    "type": "prediction_residual",
                    "observed": observed[indices].tolist(),
                    "predicted": predicted[indices].tolist(),
                    "residual": (predicted[indices] - observed[indices]).tolist(),
                    "sample_ids": [f"test_{index}" for index in indices],
                    "n_total": len(observed),
                    "n_returned": len(indices),
                    "sampling": "full_population" if len(indices) == len(observed) else "deterministic_evenly_spaced_max_2000",
                }
            metrics = {**selected["metrics"], "evaluation_scope": "final_test", "test_status": "consumed", "test": {**test, "scope": "final_test"}, "R2_val": selected["metrics"]["val"]["R2"], "RPD": test["RPD"], "RMSEP": test["RMSE"]}
            metrics["quality"] = evaluate_quality(metrics, domain=session["domain"], evaluation_scope="final_test")
            final = {"session_id": session_id, "freeze": session["freeze"], "parameters": selected["parameters"], "configuration": selected["configuration"], "metrics": metrics, "charts": {"final_test": chart}}
            with self.process._connection(write=True) as db:
                session = self._session(db, session_id)
                session.update(stage="finalized", final=final)
                self._save(db, session)
            return json.loads(_json(final))
        except BaseException:
            with self.process._connection(write=True) as db:
                session = self._session(db, session_id)
                session["stage"] = "final_test_failed"
                self._save(db, session)
            raise
