"""Provider-only compression preserves durable evidence and legal decisions."""

import copy
import json
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from test_nir_delivery_contract_reliability import _request
from test_nir_delivery_decisions import setup_delivery

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware


def test_provider_context_schema_is_single_copy_and_cv_identity_survives(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    wf.update(stage="execution", search_session_id=ctx["session_id"], decision_context_id=ctx["context_id"])
    before = copy.deepcopy(ctx)
    request = _request(wf, ctx)
    bound = NIRWorkflowMiddleware._with_delivery_contract(request)
    view = json.loads(next(message.content for message in bound.messages if isinstance(message, ToolMessage)))
    assert "proposal_schema" not in view
    assert view["proposal_schema_source"] == "nir_submit_decision.current_tool_schema"
    for name in ("context_id", "context_sha256", "revision", "task", "user_request", "capabilities", "budget", "history", "history_state", "knowledge"):
        assert view[name] == ctx[name]
    assert view["evaluation"]["cv_manifest"]["manifest_id"] == ctx["evaluation"]["cv_manifest"]["manifest_id"]
    assert "folds" not in view["evaluation"]["cv_manifest"]
    assert ctx == before
    profile = NIRWorkflowMiddleware._request_cost_profile(bound)
    assert profile["input_utf8_bytes"] < NIRWorkflowMiddleware._admission_estimate(request)
    assert profile["measurement"] == "utf8_bytes_proxy_not_tokenizer"
    assert sum(profile["components_bytes"].values()) == profile["input_utf8_bytes"]
    rebound = NIRWorkflowMiddleware._with_delivery_contract(bound)
    assert [tool.name for tool in rebound.tools] == [tool.name for tool in bound.tools]
    assert rebound.messages[-1].content == bound.messages[-1].content


def test_after_delivery_only_legitimate_approval_tools_are_exposed(tmp_path):
    _, ctx, wf = setup_delivery(tmp_path)
    request = _request({**wf, "stage": "review", "search_session_id": ctx["session_id"]}, ctx)
    bound = NIRWorkflowMiddleware._with_delivery_contract(request)
    assert {tool.name for tool in bound.tools} <= {"nir_workflow"}
    assert NIRWorkflowMiddleware._admission_estimate(bound) < NIRWorkflowMiddleware._admission_estimate(request)


def test_legacy_request_and_persisted_receipts_keep_true_usage(tmp_path):
    from langchain_core.messages import AIMessage

    from deerflow.community.nir.decision_origin import record_model_response
    from deerflow.community.nir.process import ProcessStore

    request = SimpleNamespace(state={}, messages=[], tools=[], runtime=None)
    assert NIRWorkflowMiddleware._with_delivery_contract(request) is request
    process = ProcessStore(tmp_path)
    profile = {"measurement": "utf8_bytes_proxy_not_tokenizer", "input_utf8_bytes": 200}
    message = AIMessage(content="Result", id="actual", usage_metadata={"input_tokens": 80, "output_tokens": 5, "total_tokens": 85})
    record_model_response(process, SimpleNamespace(_llm_type="fake"), [message], session_id="session", estimated_tokens=200, request_profile=profile)
    with process._connection() as db:
        receipt = json.loads(db.execute("SELECT payload FROM model_issued_calls WHERE tool='__model_response__'").fetchone()[0])
    assert receipt["token_usage"]["total_tokens"] == 85
    assert receipt["request_profile"] == profile
    assert "Result" not in json.dumps(receipt)


def test_current_schema_binds_only_real_context_evidence_ids(tmp_path):
    from deerflow.community.nir.decision_schema import bound_tool_schema

    _, context, _ = setup_delivery(tmp_path)
    bound = bound_tool_schema(context)
    proposal = bound["properties"]["proposal"]
    assert proposal["properties"]["evidence_refs"]["items"]["enum"] == context["evidence_ids"]
    assert bound["$defs"]["EvidenceClaim"]["properties"]["evidence_refs"]["items"]["enum"] == context["evidence_ids"]


def test_autonomous_skill_projection_preserves_science_user_constraints_and_source():
    import hashlib
    from pathlib import Path

    from langchain_core.messages import HumanMessage

    from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware

    root = Path(__file__).resolve().parents[2]
    source = (root / "skills/custom/nir-coordinator/SKILL.md").read_text(encoding="utf-8")
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    activation = SimpleNamespace(
        mode="auto",
        skill_name="nir-coordinator",
        category="custom",
        container_file_path="/mnt/skills/custom/nir-coordinator/SKILL.md",
        content_hash=original_hash,
        skill_content=source,
        remaining_text="User hard requirement: Ridge only; final Test must stay sealed",
    )
    message = SkillActivationMiddleware._make_activation_message(HumanMessage(content=activation.remaining_text, id="user"), SkillActivationMiddleware._build_activation_reminder(activation))
    request = _request({"autonomous_delivery": True, "decision_strategy": "llm_autonomous"})
    request.messages = [message, HumanMessage(content=activation.remaining_text)]
    before = copy.deepcopy(message.model_dump())
    projected = NIRWorkflowMiddleware._with_delivery_contract(request)
    skill = projected.messages[0]
    assert len(skill.content.encode()) < len(message.content.encode())
    assert "autonomous_skill_projection" in skill.content and original_hash in skill.content
    assert activation.remaining_text in skill.content
    for heading in ("Phase-1", "Phase-2", "Phase-3", "Phase-4", "Runtime validation evidence"):
        assert heading in skill.content
    assert "## 模式 Q" not in skill.content
    assert projected.messages[1] is request.messages[1]
    assert message.model_dump() == before
    legacy = _request({})
    legacy.messages = request.messages
    assert NIRWorkflowMiddleware._with_delivery_contract(legacy) is legacy
