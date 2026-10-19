"""Versioned decisions and feedback sharing the existing sealed search store."""

import json
import time

import numpy as np

from ._common import _sha256_file
from .decision_contracts import DecisionContext, DecisionProposal, execution_spec
from .decision_origin import issued_origin
from .decision_schema import proposal_schema, submission_guide
from .process import _digest, _id, _json, _now


def _runtime_snapshot(steps):
    from nir_core.model.parameter_control import model_capabilities
    from nir_core.preprocess.pipeline import PreprocessingPipeline
    from nir_core.preprocess.registry import catalog_hash, get_provider_implementation

    pipeline = PreprocessingPipeline(steps)
    return {
        "models_sha256": _digest(model_capabilities()),
        "preprocessing_sha256": catalog_hash(),
        "provider_policy": pipeline._provider_policy,
        "preprocessing_providers": [get_provider_implementation(step.method, provider).as_dict(is_default=False) for step, provider in zip(steps, pipeline._provider_bindings, strict=True)],
    }


def _compact_schema(value):
    if isinstance(value, dict):
        return {key: _compact_schema(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value


def _proposal_schema(delivery=False):
    schema = DecisionProposal.model_json_schema()
    if not delivery:
        schema["properties"].pop("stop_evidence")
        for name in ("StopEvidence", "UnverifiedDirection"):
            schema["$defs"].pop(name, None)
        schema["properties"]["phase"]["enum"].remove("final_selection")
        schema["properties"]["action"]["enum"].remove("RETAIN")
        schema["$defs"]["Alternative"]["properties"]["action"]["enum"].remove("RETAIN")
    return _compact_schema(schema)


def _cv_settings(manifest):
    return {
        "splitter": manifest["splitter"],
        "n_splits": manifest["n_splits"],
        "shuffle": manifest["splitter"] == "KFold",
        "random_state": manifest.get("random_state"),
        "manifest_id": manifest["manifest_id"],
        "scope": "calibration_cv",
        "source": "frozen_manifest_and_safe_search_splitter_implementation",
    }


def _decision_task(session):
    # Dataset/source hashes that include Test stay private until final delivery.
    return {key: session["task_snapshot"].get(key) for key in ("task_type", "analyte", "unit", "domain", "validation_goal", "model_constraint", "pipeline_constraint")}


def _candidate_keys(method, candidates):
    from nir_core.model.parameter_control import SCHEMAS

    # Alpha/C/etc have the same numeric meaning for 2 and 2.0. Preserve
    # integer-vs-fraction semantics of Extra Trees max_features.
    return [_json({name: float(value) if SCHEMAS[method][name]["type"] == "number" and name != "max_features" and type(value) in {int, float} else value for name, value in candidate.items()}) for candidate in candidates]


def _region_signature(spec, candidates):
    return _digest({"method": spec["method"], "steps": spec["steps"], "candidates": sorted(set(_candidate_keys(spec["method"], candidates)))})


class InitialDecisionService:
    def __init__(self, search):
        self.search = search

    def _records(self, db, session_id):
        return [
            {**json.loads(row["payload"]), "status": row["status"], "execution": json.loads(row["execution"]) if row["execution"] else None}
            for row in db.execute("SELECT * FROM initial_decisions WHERE session_id=? ORDER BY rowid", (session_id,))
        ]

    @staticmethod
    def _summary(record, *, full=False):
        result = {key: record.get(key) for key in ("decision_id", "experiment_id", "round_index", "action", "decision_time", "accepted_spec")}
        result["status"] = record["status"]
        proposal = record["proposal"]
        result.update(rationale=proposal["rationale"], expected_outcome=proposal["expected_outcome"])
        feedback = record.get("execution") or {}
        result.update({key: feedback.get(key) for key in ("feedback_id", "actual_configuration", "actual_parameters", "metrics", "fits", "elapsed_seconds", "failure_reason", "expected_improvement", "comparisons")})
        if not full and result["actual_configuration"] == result["accepted_spec"]:
            result.pop("actual_configuration")
            result["execution_matches_accepted_spec"] = True
        # Quality is descriptive evidence, never a next-action gate.
        if result.get("metrics"):
            result["metrics"] = {key: result["metrics"].get(key) for key in ("train", "train_cv", "val", "RMSECV", "quality", "evaluation_scope", "test_status")}
        if full:
            result.update(proposal=proposal, runtime_snapshot=record.get("runtime_snapshot"), cv_manifest_id=record.get("cv_manifest_id"))
        result["source"] = "private_persisted_decisions_and_experiments"
        return result

    @staticmethod
    def _recent(record):
        feedback = record.get("execution") or {}
        return {
            "decision_id": record["decision_id"],
            "experiment_id": record["experiment_id"],
            "action": record.get("action", "EXPLORE"),
            "status": record["status"],
            "val": feedback.get("metrics", {}).get("val"),
            "train_cv": feedback.get("metrics", {}).get("train_cv"),
            "fits": feedback.get("fits"),
            "failure_reason": feedback.get("failure_reason"),
            "rationale": record["proposal"]["rationale"],
            "expected_outcome": record["proposal"]["expected_outcome"],
            "expected_improvement": feedback.get("expected_improvement"),
        }

    def history(self, session_id, *, offset=0, limit=4, experiment_id=None):
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 4:
            raise ValueError("history: offset >= 0 and limit 1..4 required")
        with self.search.process._connection() as db:
            self.search._session(db, session_id)
            records = self._records(db, session_id)
        if experiment_id is not None:
            records = [record for record in records if record.get("experiment_id") == experiment_id]
            if not records:
                raise ValueError("history: unknown owner/thread/session experiment")
        return {
            "records": [self._summary(record, full=True) for record in records[offset : offset + limit]],
            "total": len(records),
            "offset": offset,
            "has_more": offset + limit < len(records),
            "source": "private_persisted_decisions_and_experiments",
            "test_status": "sealed",
        }

    @staticmethod
    def _budget_stop(session, records):
        if not session.get("autonomous_rounds") or session["stage"] != "searching":
            return
        exhausted = []
        if session["max_experiments"] <= session["experiments"]:
            exhausted.append("experiments")
        if session["max_fits"] - session["fits"] < session["cv_folds"] + 1:
            exhausted.append("fits")
        if len(records) >= session["max_decisions"]:
            exhausted.append("accepted_actions")
        if session.get("delivery_policy"):
            from .decision_budget import wall_exhausted

            if wall_exhausted(session):
                exhausted.append("wall_seconds")
        if exhausted and not any(record["status"] in {"accepted", "executing"} for record in records):
            session.update(stage="budget_exhausted", stop_source="budget" if session.get("delivery_policy") else "hard_budget", stop_reason=exhausted, stopped_at=_now())
            if session.get("delivery_policy") and not any(record["status"] == "succeeded" for record in records):
                session["stage"] = "no_deliverable_model"

    def _final_context(self, session, records, user_request):
        successful = [record for record in records if record["status"] == "succeeded"]
        if not successful:
            with self.search.process._connection(write=True) as db:
                current = self.search._session(db, session["id"])
                current["stage"] = "no_deliverable_model"
                self.search._save(db, current)
            raise ValueError("final_selection: no successful model exists; Test remains sealed")
        content = {
            "schema_version": 3,
            "session_id": session["id"],
            "phase": "final_selection",
            "history": [self._recent(record) for record in records[-4:]],
            "history_state": {
                "source": "private_persisted_decisions_and_experiments",
                "successful_experiments": [
                    {
                        "experiment_id": record["experiment_id"],
                        "decision_id": record["decision_id"],
                        "model": record["accepted_spec"]["method"],
                        "preprocessing": record["accepted_spec"]["steps"],
                        "val": record["execution"]["metrics"]["val"],
                        "train_cv": record["execution"]["metrics"]["train_cv"],
                        "fits": record["execution"]["fits"],
                        "source": "persisted_execution",
                    }
                    for record in successful
                ],
                "current": self._summary(records[-1]),
                "best_is_information_only": True,
                "detail_tool": "nir_decision_history",
            },
            "task": _decision_task(session),
            "user_request": {"text": str(user_request)[:4000], "source": "original_human_message"},
            "dataset_profile": {"source": "persisted_calibration_profile", **session["profile"]},
            "knowledge": {"source": "accepted_decision_evidence_references", "new_retrieval": False},
            "capabilities": {"allowed_actions": ["RETAIN"], "new_fits_allowed": False},
            "evaluation": {"search_protocol": "sealed_holdout", "test_access": "after_controlled_freeze", "cv_settings": _cv_settings(session["cv_manifest"])},
            "budget": {
                "stop_source": session["stop_source"],
                "stop_reason": session.get("stop_reason"),
                "fits_used": session["fits"],
                "fits_remaining": session["max_fits"] - session["fits"],
                "experiments_used": session["experiments"],
                "new_fits_allowed": False,
                "reserved_finalization": session["delivery_policy"],
                "llm_usage": session["llm_budget"],
            },
            "evidence_ids": ["task:user", "profile:calibration", "history:session", "budget:session", "protocol:sealed_holdout", *["experiment:" + record["experiment_id"] for record in records]],
            "proposal_schema": proposal_schema("final_selection"),
            "submission_guide": submission_guide("final_selection", session["cv_folds"]),
        }
        digest = _digest(content)
        with self.search.process._connection(write=True) as db:
            current = self.search._session(db, session["id"])
            if current["stage"] != "budget_exhausted":
                raise ValueError("final_selection: session is no longer awaiting retention")
            identity = "ctx_" + _digest([session["id"], digest])[:32]
            prior = db.execute("SELECT payload FROM decision_contexts WHERE id=?", (identity,)).fetchone()
            if prior:
                payload = json.loads(prior[0])
            else:
                revision = db.execute("SELECT MAX(revision) FROM decision_contexts WHERE session_id=?", (session["id"],)).fetchone()[0] or 0
                payload = DecisionContext.model_validate({**content, "context_id": identity, "context_sha256": digest, "revision": revision + 1}).model_dump()
                db.execute("INSERT INTO decision_contexts VALUES (?,?,?,?)", (identity, session["id"], revision + 1, _json(payload)))
            current["decision_context_id"] = identity
            current["context_basis"] = self._context_basis(current, records, None)
            self.search._save(db, current)
            return payload

    @staticmethod
    def _context_basis(session, records, query):
        from nir_core.model.parameter_control import model_capabilities
        from nir_core.preprocess.registry import catalog_hash

        from .method_catalog import get_method_store

        return _digest([session["stage"], [(record["decision_id"], record["status"]) for record in records], query, _digest(model_capabilities()), catalog_hash(), _digest(get_method_store().snapshot())])

    def context(self, session_id, workflow, user_request, *, knowledge_query=None):
        from nir_core.knowledge.methods import MethodKnowledgeBase
        from nir_core.model.parameter_control import model_capabilities
        from nir_core.preprocess.registry import catalog_hash, catalog_snapshot

        from .method_catalog import get_method_store

        if workflow.get("decision_strategy") != "llm_autonomous" or workflow.get("search_protocol") != "sealed_holdout":
            raise ValueError("Initial LLM decisions require llm_autonomous with sealed_holdout")
        with self.search.process._connection() as db:
            session = self.search._session(db, session_id)
            records = self._records(db, session_id)
        multi = bool(session.get("autonomous_rounds"))
        if session.get("decision_strategy") != "llm_autonomous":
            raise ValueError("Initial decisions cannot reuse a legacy search session")
        if session.get("delivery_policy") and session.get("decision_context_id") and session.get("context_basis") == self._context_basis(session, records, knowledge_query):
            with self.search.process._connection() as db:
                row = db.execute("SELECT payload FROM decision_contexts WHERE id=? AND session_id=?", (session["decision_context_id"], session_id)).fetchone()
            if row:
                return json.loads(row[0])
        if session.get("delivery_policy") and session["stage"] == "budget_exhausted":
            return self._final_context(session, records, user_request)
        if session["stage"] != "searching" or (not multi and (session["experiments"] or session.get("initial_decision_id"))):
            raise ValueError("Initial context requires an unused searching session; iteration is not enabled")
        if any(record["status"] in {"accepted", "executing"} for record in records):
            raise ValueError("context: accepted decision must execute before a new round")
        path = self.search._dir(session_id) / "search.npz"
        if _sha256_file(str(path)) != session["search_hash"]:
            raise ValueError("Calibration/validation integrity failure")
        with np.load(path, allow_pickle=False) as archive:
            y = archive["y_cal"]
            target_profile = {"n": len(y), "mean": float(np.mean(y)), "std": float(np.std(y)), "minimum": float(np.min(y)), "maximum": float(np.max(y)), "scope": "calibration"}
        snapshot = get_method_store().snapshot()
        kb = MethodKnowledgeBase(payload=snapshot["payload"])
        queries = [" ".join(session["profile"].get("tags", [])) or "spectral preprocessing", "model_pls", "model_ridge", "model_svr", "model_extra_trees"]
        cards = {}
        if knowledge_query:
            if len(knowledge_query) > 200:
                raise ValueError("knowledge_query: maximum 200 characters")
            queries.append(knowledge_query)
        for query in queries:
            for card in kb.search(query, top_k=1 if multi else 8, profile=None)["results"]:
                cards[card["evidence_id"]] = card
        preprocessing = catalog_snapshot()
        for method in preprocessing["methods"]:
            method["autonomous_supported"] = not method["requires_y"] and not method["changes_axis"]
        if multi:
            from nir_core.preprocess.registry import get_method_spec

            preprocessing = {
                "methods": [
                    {
                        "method": method.method_id,
                        "summary": method.summary_zh,
                        "parameters": {name: parameter.as_dict() for name, parameter in method.parameter_schema.items()},
                        "requires_y": method.requires_y,
                        "changes_axis": method.changes_axis,
                        "requires_regular_axis": method.requires_regular_axis,
                        "autonomous_supported": not method.requires_y and not method.changes_axis,
                    }
                    for method in [get_method_spec(item["method_id"]) for item in catalog_snapshot()["methods"]]
                ]
            }
            knowledge_fields = {
                "evidence_id",
                "method_id",
                "title",
                "source_url",
                "source_section",
                "content_sha256",
                "summary_zh",
                "applicability_zh",
                "limitations_zh",
                "parameter_guidance_zh",
                "regular_axis_required",
                "review_status",
                "untrusted_evidence",
            }
            cards = {key: {field: value for field, value in card.items() if field in knowledge_fields} for key, card in cards.items()}
        executed = [record for record in records if record.get("execution") and record["status"] in {"succeeded", "failed"}]
        successful = [record for record in executed if record["status"] == "succeeded"]
        best = min(successful, key=lambda record: record["execution"]["metrics"]["val"]["RMSE"]) if successful else None
        retained = next((record for record in successful if record["experiment_id"] == session.get("retained_experiment_id")), None)
        history_state = {
            "source": "private_persisted_decisions_and_experiments",
            "current": self._summary(executed[-1]) if executed else None,
            "best_validation": ({"experiment_id": best["experiment_id"], "same_as": "current"} if executed and best == executed[-1] else self._summary(best)) if best else None,
            "best_is_information_only": True,
            "retained": self._summary(retained) if retained else None,
            "successful_count": len(successful),
            "failed_count": len(executed) - len(successful),
            "tried_regions": [
                {
                    "decision_id": record["decision_id"],
                    "experiment_id": record["experiment_id"],
                    "status": record["status"],
                    "model": record["accepted_spec"]["method"],
                    "preprocessing": record["accepted_spec"]["steps"],
                    "parameter_region": {key: record["accepted_spec"]["parameter_plan"][key] for key in ("strategy", "fixed_params", "search_space")},
                }
                for record in executed
            ],
            "detail_tool": "nir_decision_history",
        }
        content = {
            "schema_version": 2 if multi else 1,
            "session_id": session_id,
            "phase": "feedback" if executed else "initial",
            "history": [self._recent(record) for record in executed[-4:]] if multi else [],
            "history_state": history_state if multi else {},
            "task": {key: workflow.get(key) for key in ("task_type", "analyte", "unit", "domain", "validation_goal", "model_constraint", "pipeline_constraint")},
            "user_request": {"text": str(user_request)[:4000], "omitted_characters": max(0, len(str(user_request)) - 4000), "source": "original_human_message", "trust": "user_instruction"},
            "dataset_profile": {
                **session["profile"],
                "target": target_profile,
                "source": "sealed_session:calibration_profile",
                "evidence_id": "profile:calibration",
                "uncertainty": "Scores are descriptive proxies; scatter/noise/baseline causes are not established and do not mandate preprocessing.",
            },
            "knowledge": {
                "results": list(cards.values()),
                "queries": queries,
                "source": "managed_published_method_catalog",
                "snapshot_revision": snapshot["revision"],
                "snapshot_sha256": _digest(snapshot),
                "ranking_is_execution_order": False,
                "trust": "untrusted_evidence",
            },
            "capabilities": {
                "models": model_capabilities(),
                "models_sha256": _digest(model_capabilities()),
                "preprocessing": preprocessing,
                "preprocessing_sha256": catalog_hash(),
                "source": "actual_execution_registry_and_adapters",
                "raw_allowed": True,
                "wavelength_selection": ["none"],
                "parameter_defaults_policy": "Displayed adapter defaults apply only to omitted model parameters; no family or preprocessing fallback.",
            },
            "evaluation": {
                "search_protocol": "sealed_holdout",
                "selection_scope": "validation",
                "parameter_objective": "train_cv_rmse",
                "cv_manifest": session["cv_manifest"],
                "cv_settings": _cv_settings(session["cv_manifest"]),
                "test_access": "after_explicit_freeze_only",
                "partition_counts": session["counts"],
                "search_data_sha256": session["search_hash"],
            },
            "budget": {
                "experiments_remaining": session["max_experiments"] - session["experiments"],
                "fits_remaining": session["max_fits"] - session["fits"] - session.get("reserved_fits", 0),
                "fixed_required_fits": session["cv_folds"] + 1,
                "max_grid_candidates": min(40, (session["max_fits"] - session["fits"] - 1) // session["cv_folds"]),
                "fit_cost_formula": "Cartesian candidate count * frozen CV n_splits + 1 full-Cal refit; fixed has one candidate and still evaluates CV.",
                "grid_semantics": "Explicit discrete lists form a Cartesian product; fixed_params and search_space keys must be disjoint.",
                "source": "persisted_session_budget",
            },
            "evidence_ids": ["task:user", "profile:calibration", "capabilities:runtime", "protocol:sealed_holdout", "budget:session", *cards],
            "proposal_schema": _proposal_schema(bool(session.get("delivery_policy"))) if multi else DecisionProposal.model_json_schema(),
        }
        if multi:
            # The exact immutable fold manifest is persisted; initial context carries it
            # once. Subsequent rounds reference its digest/count rather than repeat rows.
            if executed:
                content["evaluation"]["cv_manifest"] = {key: session["cv_manifest"][key] for key in ("manifest_id", "sha256", "scope", "n_splits", "splitter", "random_state") if key in session["cv_manifest"]}
            with self.search.process._connection() as db:
                calls = [json.loads(row[0]) for row in db.execute("SELECT payload FROM model_issued_calls WHERE tool='__model_response__'")]
            calls = [call for call in calls if call.get("session_id") == session_id]
            usage = [call.get("token_usage") for call in calls]
            content["budget"].update(
                accepted_actions_used=len(records),
                actions_remaining=session["max_decisions"] - len(records),
                experiments_used=session["experiments"],
                fits_used=session["fits"],
                execution_seconds=session.get("execution_seconds", 0.0),
                llm_calls=len(calls),
                llm_seconds=sum(call.get("elapsed_seconds") or 0 for call in calls),
                input_tokens=sum((item or {}).get("input_tokens", 0) for item in usage),
                output_tokens=sum((item or {}).get("output_tokens", 0) for item in usage),
                calls_with_unknown_token_usage=sum(item is None for item in usage),
                monetary_cost="not supplied by provider",
            )
            content["evidence_ids"] += ["history:session", *["experiment:" + record["experiment_id"] for record in executed]]
        if session.get("delivery_policy"):
            content["schema_version"] = 3
            content["proposal_schema"] = proposal_schema(content["phase"])
            content["submission_guide"] = submission_guide(content["phase"], session["cv_folds"])
            content["task"] = _decision_task(session)
            content["budget"].update(delivery_policy=session["delivery_policy"], llm_usage=session["llm_budget"], wall_seconds_used=time.time() - session["budget_started_at"], stop_source=session.get("stop_source"))
            ledger = session["llm_budget"]
            content["budget"].update(
                llm_calls=sum(ledger[name]["calls"] for name in ("search", "finalization")),
                llm_seconds=sum(ledger[name]["seconds"] for name in ("search", "finalization")),
                input_tokens=sum(ledger[name]["actual_input_tokens"] for name in ("search", "finalization")),
                output_tokens=sum(ledger[name]["actual_output_tokens"] for name in ("search", "finalization")),
                calls_with_unknown_token_usage=sum(ledger[name]["unknown_usage_calls"] for name in ("search", "finalization")),
            )
            content["capabilities"]["parameter_authority"] = "Only each model's parameters keys are writable. operational_parameters are read-only adapter settings, never valid fixed_params/search_space keys."
            content["history_state"]["unverified_hypotheses"] = [
                {"decision_id": record["decision_id"], "hypotheses": record["proposal"].get("hypotheses", []), "expected_improvement": (record.get("execution") or {}).get("expected_improvement")} for record in executed[-4:]
            ]
        digest = _digest(content)
        with self.search.process._connection(write=True) as db:
            current = self.search._session(db, session_id)
            current_records = self._records(db, session_id)
            if (not multi and (current.get("initial_decision_id") or current["experiments"])) or any(record["status"] in {"accepted", "executing"} for record in current_records):
                raise ValueError("Initial decision already accepted or executed")
            if [record["decision_id"] for record in current_records] != [record["decision_id"] for record in records] or current["stage"] != "searching":
                raise ValueError("context: session changed during snapshot; retry")
            prior = db.execute("SELECT revision FROM decision_contexts WHERE session_id=? ORDER BY revision DESC LIMIT 1", (session_id,)).fetchone()
            context_id = "ctx_" + _digest([session_id, digest])[:32]
            existing = db.execute("SELECT payload FROM decision_contexts WHERE id=?", (context_id,)).fetchone()
            if existing:
                current["decision_context_id"] = context_id
                self.search._save(db, current)
                return json.loads(existing[0])
            payload = DecisionContext.model_validate({**content, "context_id": context_id, "revision": (prior[0] + 1) if prior else 1, "context_sha256": digest}).model_dump()
            db.execute("INSERT INTO decision_contexts VALUES (?,?,?,?)", (context_id, session_id, payload["revision"], _json(payload)))
            current["decision_context_id"] = context_id
            if session.get("delivery_policy"):
                current["context_basis"] = self._context_basis(current, current_records, knowledge_query)
            self.search._save(db, current)
            return payload

    def submit(self, session_id, proposal, call_id):
        from nir_core.model.parameter_control import SCHEMAS, authorized_candidates, model_capabilities
        from nir_core.models import PreprocessingStep
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.preprocess.registry import catalog_hash

        from .method_catalog import get_method_store

        origin = issued_origin(self.search.process, call_id, "nir_submit_decision", {"proposal": proposal})
        serialized = _json(proposal) if isinstance(proposal, dict) else proposal
        if len(serialized) > 40000:
            raise ValueError("proposal: exceeds size limit")
        parsed = DecisionProposal.model_validate(proposal) if isinstance(proposal, dict) else DecisionProposal.model_validate_json(proposal)
        with self.search.process._connection(write=True) as db:
            session = self.search._session(db, session_id)
            decision_id = "decision_" + _digest([session_id, call_id])[:32]
            replay = db.execute("SELECT payload FROM initial_decisions WHERE id=? AND session_id=?", (decision_id, session_id)).fetchone()
            if replay and DecisionProposal.model_validate(json.loads(replay[0])["proposal"]).model_dump() == parsed.model_dump():
                return json.loads(replay[0])
            row = db.execute("SELECT payload FROM decision_contexts WHERE id=? AND session_id=?", (parsed.context_id, session_id)).fetchone()
            if not row:
                raise ValueError("context: unknown or not bound to session")
            context = json.loads(row[0])
            if parsed.expected_revision != context["revision"] or parsed.context_sha256 != context["context_sha256"] or session.get("decision_context_id") != parsed.context_id:
                raise ValueError("context: stale revision/hash")
            if context["phase"] != "final_selection" and (
                context["capabilities"]["preprocessing_sha256"] != catalog_hash()
                or context["capabilities"]["models_sha256"] != _digest(model_capabilities())
                or context["knowledge"]["snapshot_sha256"] != _digest(get_method_store().snapshot())
            ):
                raise ValueError("context: runtime/knowledge snapshot changed; request a fresh context")
            records = self._records(db, session_id)
            multi = bool(session.get("autonomous_rounds"))
            if records and not multi:
                raise ValueError("Only one initial decision is enabled; further optimization is a later phase")
            terminal_selection = bool(session.get("delivery_policy") and session["stage"] == "budget_exhausted")
            if session["stage"] != "searching" and not terminal_selection or (not multi and session["experiments"]):
                raise ValueError("Initial search already used/closed")
            if any(record["status"] in {"accepted", "executing"} for record in records):
                raise ValueError("decision: a decision is already awaiting execution")
            executed = [record for record in records if record["status"] in {"succeeded", "failed"}]
            base = executed[-1] if executed else None
            if terminal_selection and (parsed.phase != "final_selection" or parsed.action != "RETAIN"):
                raise ValueError("final_selection: only RETAIN of existing successful evidence is allowed; no training")
            if parsed.phase != ("final_selection" if terminal_selection else "feedback" if base else "initial") or context["phase"] != parsed.phase:
                raise ValueError("phase: context does not match executed history")
            if not base and parsed.action != "EXPLORE":
                raise ValueError("action: initial selection must be EXPLORE")
            if base:
                if parsed.base_experiment_id != base["experiment_id"]:
                    raise ValueError("base_experiment_id: must equal current persisted experiment " + base["experiment_id"])
                for key in ("uncertainties", "hypotheses", "not_selected_reasons"):
                    if not getattr(parsed, key):
                        raise ValueError(key + ": required and must be nonempty in feedback/final_selection")
                if parsed.expected_fits is None:
                    raise ValueError("expected_fits: required integer; candidates * CV folds + 1 for training, zero for STOP/RETAIN")
            if multi and len(records) >= session["max_decisions"] and not terminal_selection:
                raise ValueError("budget: accepted action cap reached")
            refs = set(parsed.evidence_refs) | {ref for claim in parsed.observations for ref in claim.evidence_refs}
            if not refs <= set(context["evidence_ids"]):
                raise ValueError("evidence_refs: unknown evidence; use context evidence IDs")
            ids = [alt.id for alt in parsed.alternatives]
            if len(set(ids)) != len(ids) or parsed.selected_alternative_id not in ids:
                raise ValueError("selected_alternative_id: missing or duplicate candidate identity")
            selected = next(alt for alt in parsed.alternatives if alt.id == parsed.selected_alternative_id)
            if selected.action != parsed.action:
                raise ValueError(f"action: must equal alternatives.{ids.index(selected.id)}.action ({selected.action})")
            rejected_ids = set(ids) - {parsed.selected_alternative_id}
            if base and set(parsed.not_selected_reasons) != rejected_ids:
                raise ValueError("not_selected_reasons: explain every unselected candidate")
            retained_id = parsed.retained_experiment_id or session.get("retained_experiment_id")
            retained = next((record for record in executed if record["experiment_id"] == retained_id and record["status"] == "succeeded"), None)
            if retained_id and not retained:
                raise ValueError("retained_experiment_id: must name a successful experiment in this session")
            if parsed.action in {"STOP", "RETAIN"}:
                if parsed.action == "RETAIN" and not terminal_selection:
                    raise ValueError("RETAIN: only available for a budget-terminal final selection")
                if not multi or not base or not parsed.retained_experiment_id or not retained or selected.pipeline is not None or selected.optimization is not None or parsed.expected_fits != 0:
                    raise ValueError("STOP: require a successful retained experiment, null execution plan and zero fits")
                if session.get("delivery_policy"):
                    if parsed.stop_evidence is None or "experiment:" + retained_id not in refs:
                        raise ValueError("STOP: require retained advantages/limitations, coverage, unverified directions/reasons and retained experiment evidence")
                    task = session["task_snapshot"]
                    if task.get("model_constraint") and retained["accepted_spec"]["method"] != task["model_constraint"]:
                        raise ValueError("model_constraint: retained experiment violates user requirement")
                    if task.get("pipeline_constraint") is not None and retained["accepted_spec"]["steps"] != json.loads(task["pipeline_constraint"]):
                        raise ValueError("pipeline_constraint: retained experiment violates user requirement")
                record = {
                    "decision_id": decision_id,
                    "context_id": parsed.context_id,
                    "context_sha256": parsed.context_sha256,
                    "session_id": session_id,
                    "experiment_id": None,
                    "origin": origin,
                    "proposal": parsed.model_dump(),
                    "proposal_sha256": _digest(parsed.model_dump()),
                    "accepted_spec": None,
                    "reserved_fits": 0,
                    "decision_time": _now(),
                    "selection_origin": "llm",
                    "round_index": len(records) + 1,
                    "action": parsed.action,
                }
                stop_source = "budget" if terminal_selection else "llm"
                if session.get("delivery_policy"):
                    record.update(stop_budget_snapshot=context["budget"], choice_kind="budget_final_retention" if terminal_selection else "llm_stop")
                feedback = {
                    "decision_id": decision_id,
                    "experiment_id": None,
                    "status": "stopped",
                    "model_executed": False,
                    "retained_experiment_id": retained_id,
                    "test_status": "sealed",
                    "stop_source": stop_source,
                    "pending_final_confirmation": not bool(session.get("delivery_policy")),
                }
                db.execute("INSERT INTO initial_decisions VALUES (?,?,?,'stopped',?)", (decision_id, session_id, _json(record), _json(feedback)))
                session.update(
                    stage="stopped" if session.get("delivery_policy") else "stopped_pending_confirmation",
                    stop_source=stop_source,
                    stop_decision_id=decision_id,
                    retained_experiment_id=retained_id,
                    retained_selection_source="llm",
                    stopped_at=_now(),
                )
                if (session.get("delivery_policy") or {}).get("automatic_delivery"):
                    session["automatic_delivery"] = {
                        "status": "queued",
                        "terminal_decision_id": decision_id,
                        "retained_experiment_id": retained_id,
                        "lease": None,
                        "trigger_run_id": origin.get("agent_run_id"),
                        "events": [{"stage": "queued", "at": _now(), "source": "validated_persisted_terminal_decision"}],
                    }
                self.search._save(db, session)
                return record
            if selected.pipeline is None or selected.optimization is None:
                raise ValueError("pipeline: execution action requires pipeline and optimization")
            spec = execution_spec(selected)
            constraint = context["task"].get("model_constraint")
            if constraint and spec["method"] != constraint:
                raise ValueError("model_constraint: proposal violates user hard requirement")
            pipeline_constraint = context["task"].get("pipeline_constraint")
            if pipeline_constraint is not None and spec["steps"] != json.loads(pipeline_constraint):
                raise ValueError("pipeline_constraint: proposal violates user hard requirement")
            steps = [PreprocessingStep(**step) for step in spec["steps"]]
            valid, errors = validate_pipeline(steps)
            if not valid:
                raise ValueError("pipeline.preprocessing: " + str(errors))
            for step in steps:
                descriptor = context["dataset_profile"]
                from nir_core.preprocess.registry import get_method_spec

                method = get_method_spec(step.method)
                if method.requires_y or method.changes_axis:
                    raise ValueError(f"pipeline.{step.method}: supervised/axis-changing preprocessing is not adapted to this protocol")
                if method.requires_regular_axis and not descriptor["regular_wavelength_axis"]:
                    raise ValueError(f"pipeline.{step.method}: a regular wavelength axis is required")
                step.params = {name: parameter.default for name, parameter in method.parameter_schema.items() if parameter.has_default} | step.params
                if step.params.get("window", 0) > descriptor["n_wavelengths"]:
                    raise ValueError(f"pipeline.{step.method}.window: exceeds spectral width")
            PreprocessingPipeline(steps)  # Resolve actual provider binding, no fit.
            spec["steps"] = [step.model_dump() for step in steps]
            cv = session["cv_manifest"]
            upper = min(20, context["dataset_profile"]["n_wavelengths"], min(len(fold["train"]) - 1 for fold in cv["folds"]))
            candidates = authorized_candidates(spec["method"], spec["parameter_plan"], component_upper=upper, cv_folds=session["cv_folds"])
            defaults = {name: parameter["default"] for name, parameter in SCHEMAS[spec["method"]].items() if name not in spec["parameter_plan"]["search_space"]}
            spec["parameter_plan"]["fixed_params"] = defaults | spec["parameter_plan"]["fixed_params"]
            cost = len(candidates) * session["cv_folds"] + 1
            if parsed.expected_fits is not None and parsed.expected_fits != cost:
                raise ValueError(f"expected_fits: declared budget must equal actual {cost} fits")
            candidate_keys = _candidate_keys(spec["method"], candidates)
            if multi and len(set(candidate_keys)) != len(candidate_keys):
                raise ValueError("duplicate: parameter grid contains repeated identical candidates")
            signature = _region_signature(spec, candidates)
            # Recompute from immutable accepted plans to remain compatible with
            # earlier signature encodings without rewriting historical payloads.
            repeated = any(
                prior.get("accepted_spec")
                and _region_signature(prior["accepted_spec"], authorized_candidates(prior["accepted_spec"]["method"], prior["accepted_spec"]["parameter_plan"], component_upper=upper, cv_folds=session["cv_folds"])) == signature
                for prior in records
            )
            if repeated and not parsed.replication_reason:
                raise ValueError("duplicate: identical pipeline/parameter candidates were already accepted; provide explicit replication_reason to repeat")
            if base:
                same_model = spec["method"] == base["accepted_spec"]["method"]
                same_pipe = spec["steps"] == base["accepted_spec"]["steps"]
                if parsed.action == "TUNE" and not (same_model and same_pipe):
                    raise ValueError("TUNE: model family and preprocessing must remain fixed")
                if parsed.action == "CHANGE_PREPROCESS" and not (same_model and not same_pipe):
                    raise ValueError("CHANGE_PREPROCESS: retain model family and change preprocessing")
                if parsed.action == "SWITCH_MODEL" and not (not same_model and same_pipe):
                    raise ValueError("SWITCH_MODEL: change model family and retain preprocessing")
                if parsed.action == "EXPLORE" and (same_model and same_pipe or not parsed.difference):
                    raise ValueError("EXPLORE: a distinct pipeline direction and difference rationale are required")
            if cost > session["max_fits"] - session["fits"] or session["experiments"] >= session["max_experiments"]:
                from .decision_budget import DecisionBudgetError

                remaining = session["max_fits"] - session["fits"]
                raise DecisionBudgetError(
                    "insufficient remaining session fits/experiments",
                    fits_remaining=remaining,
                    cv_folds=session["cv_folds"],
                    candidates=len(candidates),
                    required_fits=cost,
                    max_executable_candidates=max(0, min(40, (remaining - 1) // session["cv_folds"])),
                    experiments_remaining=session["max_experiments"] - session["experiments"],
                )
            record = {
                "decision_id": decision_id,
                "context_id": parsed.context_id,
                "context_sha256": parsed.context_sha256,
                "session_id": session_id,
                "experiment_id": "exp_" + _digest([session_id, decision_id])[:32],
                "origin": origin,
                "proposal": parsed.model_dump(),
                "proposal_sha256": _digest(parsed.model_dump()),
                "accepted_spec": spec,
                "runtime_snapshot": _runtime_snapshot(steps),
                "cv_manifest_id": cv["manifest_id"],
                "reserved_fits": cost,
                "decision_time": _now(),
                "selection_origin": "llm",
                "round_index": len(records) + 1,
                "action": parsed.action,
                "base_experiment_id": base["experiment_id"] if base else None,
                "retained_experiment_id": retained_id,
                "execution_signature": signature,
            }
            db.execute("INSERT INTO initial_decisions VALUES (?,?,?,'accepted',NULL)", (decision_id, session_id, _json(record)))
            if not session.get("initial_decision_id"):
                session["initial_decision_id"] = decision_id
            session.update(active_decision_id=decision_id, reserved_fits=cost)
            if session.get("delivery_policy"):
                session["invalid_decisions"] = 0
            if retained_id:
                session["retained_experiment_id"] = retained_id
            self.search._save(db, session)
            return record

    def get(self, decision_id):
        with self.search.process._connection() as db:
            row = db.execute("SELECT * FROM initial_decisions WHERE id=?", (_id(decision_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown owner/thread decision")
        return {**json.loads(row["payload"]), "status": row["status"], "execution": json.loads(row["execution"]) if row["execution"] else None}

    def execute(self, decision_id, call_id, *, checkpoint=lambda _: None):
        origin = issued_origin(self.search.process, call_id, "nir_execute_decision", {"decision_id": decision_id})
        with self.search.process._connection(write=True) as db:
            row = db.execute("SELECT * FROM initial_decisions WHERE id=?", (_id(decision_id),)).fetchone()
            if row is None:
                raise ValueError("Unknown accepted decision")
            if row["status"] in {"succeeded", "failed"}:
                return json.loads(row["execution"])
            if row["status"] != "accepted":
                raise ValueError("Decision execution lease already consumed")
            record = json.loads(row["payload"])
            session = self.search._session(db, record["session_id"])
            if session["stage"] != "searching":
                raise ValueError("Search is closed")
            db.execute("UPDATE initial_decisions SET status='executing' WHERE id=?", (decision_id,))
            before = session["fits"]
        started = time.monotonic()
        try:
            from nir_core.models import PreprocessingStep

            if record["runtime_snapshot"] != _runtime_snapshot([PreprocessingStep(**step) for step in record["accepted_spec"]["steps"]]):
                raise ValueError("runtime: model/preprocessing adapter changed after acceptance; no replacement can execute")
            result = self.search.search(record["session_id"], decision_id, **record["accepted_spec"], decision_id=decision_id, checkpoint=checkpoint)
            feedback = {
                **result,
                "decision_id": decision_id,
                "context_id": record["context_id"],
                "status": "succeeded",
                "actual_parameters": result["parameters"],
                "actual_configuration": result["configuration"],
                "runtime_snapshot": record["runtime_snapshot"],
                "elapsed_seconds": time.monotonic() - started,
                "execution_origin": origin,
            }
        except BaseException as exc:
            with self.search.process._connection() as db:
                fits = self.search._session(db, record["session_id"])["fits"] - before
            feedback = {
                "decision_id": decision_id,
                "context_id": record["context_id"],
                "experiment_id": record["experiment_id"],
                "cv_manifest_id": record["cv_manifest_id"],
                "status": "failed",
                "actual_configuration": record["accepted_spec"],
                "fits": fits,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "failure_code": getattr(exc, "code", None),
                "failure_reason": str(exc).replace(str(self.search.process.root), "<private-storage>")[:1000] or type(exc).__name__,
                "execution_origin": origin,
            }
            if not isinstance(exc, Exception):
                self._finish(record, feedback)
                raise
        self._finish(record, feedback)
        return feedback

    def _finish(self, record, feedback):
        with self.search.process._connection(write=True) as db:
            records = self._records(db, record["session_id"])
            comparisons = {}
            for name, identity in (("previous", record.get("base_experiment_id")), ("retained", record.get("retained_experiment_id"))):
                reference = next((prior for prior in records if identity and prior.get("experiment_id") == identity), None)
                prior_val = ((reference or {}).get("execution") or {}).get("metrics", {}).get("val")
                current_val = feedback.get("metrics", {}).get("val")
                comparisons[name] = {"experiment_id": identity, "available": bool(prior_val and current_val)}
                if prior_val and current_val:
                    comparisons[name].update(val_rmse_change=current_val["RMSE"] - prior_val["RMSE"], val_r2_change=current_val["R2"] - prior_val["R2"])
            goal = record["proposal"].get("expected_improvement")
            assessment = {"goal": goal, "status": "not_evaluable", "source": "persisted_validation_metrics", "text_expectation": record["proposal"]["expected_outcome"]}
            if goal and comparisons[goal["comparison"]]["available"]:
                delta = comparisons[goal["comparison"]][goal["metric"] + "_change"]
                improvement = -delta if goal["metric"] == "val_rmse" else delta
                assessment.update(status="achieved" if improvement >= goal["minimum_change"] else "not_achieved", actual_improvement=improvement)
            feedback.update(
                feedback_id="feedback_" + _digest([record["decision_id"], record["experiment_id"]])[:32],
                previous_experiment_id=record.get("base_experiment_id"),
                retained_experiment_id=record.get("retained_experiment_id"),
                comparisons=comparisons,
                expected_improvement=assessment,
            )
            session = self.search._session(db, record["session_id"])
            feedback["cv_settings"] = _cv_settings(session["cv_manifest"])
            db.execute("UPDATE initial_decisions SET status=?,execution=? WHERE id=?", (feedback["status"], _json(feedback), record["decision_id"]))
            session["reserved_fits"] = 0
            session["execution_seconds"] = session.get("execution_seconds", 0.0) + feedback["elapsed_seconds"]
            session["last_decision_id"] = record["decision_id"]
            session["active_decision_id"] = None
            if session.get("autonomous_rounds"):
                session["decision_context_id"] = None
            self._budget_stop(session, self._records(db, record["session_id"]))
            if session.get("delivery_policy") and feedback.get("failure_code") == "resource_cancelled":
                session.update(stage="cancelled", stop_source="user", stop_reason=["runtime_cancelled"], stopped_at=_now())
            if session.get("delivery_policy") and feedback.get("failure_code") == "resource_deadline_exceeded":
                session.update(stage="budget_exhausted", stop_source="budget", stop_reason=["operation_deadline"], stopped_at=_now())
            feedback["budget"] = {
                "fits_used": session["fits"],
                "fits_remaining": session["max_fits"] - session["fits"],
                "experiments_used": session["experiments"],
                "experiments_remaining": session["max_experiments"] - session["experiments"],
                "accepted_actions_used": len(records),
                "actions_remaining": (session.get("max_decisions", 20) if session.get("autonomous_rounds") else 1) - len(records),
                "execution_seconds": session["execution_seconds"],
                "stage": session["stage"],
                "stop_source": session.get("stop_source"),
                "stop_reason": session.get("stop_reason"),
                "source": "persisted_session_budget",
            }
            db.execute("UPDATE initial_decisions SET execution=? WHERE id=?", (_json(feedback), record["decision_id"]))
            self.search._save(db, session)
