"""Real-provider failure regressions; model choices remain explicit and audited."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import SystemMessage
from test_nir_delivery_decisions import setup_delivery, stop_plan
from test_nir_initial_decisions import issue, proposal, submit
from test_nir_multiround_decisions import execute, next_plan

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir import decision_tools
from deerflow.community.nir.decision_budget import reserve_call


def test_schema_matches_initial_feedback_and_read_only_retention(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path, max_experiments=2)
    assert ctx["proposal_schema"]["properties"]["action"]["enum"] == ["EXPLORE"]
    initial = submit(service, ctx, proposal(ctx))
    execute(service, initial)
    ctx = service.context(ctx["session_id"], wf, "Feedback")
    schema = ctx["proposal_schema"]
    assert set(schema["properties"]["action"]["enum"]) == {"TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "EXPLORE", "STOP"}
    assert {"base_experiment_id", "uncertainties", "hypotheses", "not_selected_reasons", "expected_fits"} <= set(schema["required"])
    execute(service, submit(service, ctx, next_plan(ctx), "second"), "second-execute")
    ctx = service.context(ctx["session_id"], wf, "Retain")
    schema = ctx["proposal_schema"]
    assert schema["properties"]["action"]["enum"] == ["RETAIN"]
    assert schema["properties"]["expected_fits"]["const"] == 0
    assert {"retained_experiment_id", "stop_evidence"} <= set(schema["required"])
    assert "PipelineSpec" not in schema.get("$defs", {})


@pytest.mark.parametrize("terminal", [False, True])
def test_unchanged_context_is_cached_without_io_or_revision_churn(tmp_path, monkeypatch, terminal):
    service, ctx, wf = setup_delivery(tmp_path, max_experiments=1 if terminal else 5)
    if terminal:
        execute(service, submit(service, ctx, proposal(ctx)))
        ctx = service.context(ctx["session_id"], wf, "Task")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Repeated unchanged context reread data"))
    with service.search.process._connection(write=True) as db:
        session = service.search._session(db, ctx["session_id"])
        session["llm_budget"]["search"]["calls"] += 1
        service.search._save(db, session)
    assert service.context(ctx["session_id"], wf, "Repeated read") == ctx


def test_object_proposal_preserves_exact_agent_origin_and_old_string_contract(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    plan = proposal(ctx)
    issue(service, "nir_submit_decision", {"proposal": plan}, "object")
    accepted = service.submit(ctx["session_id"], plan, "object")
    assert accepted["proposal"]["rationale"] == plan["rationale"]
    assert accepted["origin"]["verification_kind"] == "fake_model"
    changed = copy.deepcopy(plan)
    changed["rationale"] = "Forged"
    with pytest.raises(ValueError, match="model-issued"):
        service.submit(ctx["session_id"], changed, "object")
    execute(service, accepted)
    ctx = service.context(ctx["session_id"], wf, "Feedback")
    second = submit(service, ctx, next_plan(ctx), "string")
    assert service.get(second["decision_id"])["status"] == "accepted"


def _request(wf, context=None, tools=None):
    from langchain_core.messages import AIMessage, ToolMessage

    messages = []
    if context:
        messages = [AIMessage(content="", tool_calls=[{"name": "nir_decision_context", "id": "ctx-call", "args": {}}]), ToolMessage(content=json.dumps(context), tool_call_id="ctx-call", name="nir_decision_context")]
    request = SimpleNamespace(
        state={"nir_workflow": wf},
        messages=messages,
        system_message=SystemMessage(content="Prompt"),
        tools=tools or [decision_tools.nir_submit_decision_tool, decision_tools.nir_decision_context_tool, decision_tools.nir_execute_decision_tool],
    )
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    return request


def test_model_tool_schema_is_object_scoped_and_original_tool_unchanged(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    wf.update(stage="execution", search_session_id=ctx["session_id"], decision_context_id=ctx["context_id"])
    original = copy.deepcopy(decision_tools.nir_submit_decision_tool.args)
    request = _request(wf, ctx)
    updated = NIRWorkflowMiddleware._with_delivery_contract(request)
    assert {tool.name for tool in updated.tools} == {"nir_submit_decision", "nir_decision_history"} & {tool.name for tool in request.tools}
    schema = updated.tools[0].args["proposal"]
    assert schema["type"] == "object" and schema["properties"]["phase"]["const"] == "initial"
    assert schema["properties"]["context_id"]["const"] == ctx["context_id"]
    assert decision_tools.nir_submit_decision_tool.args == original


def test_legacy_model_request_and_tools_are_unchanged():
    request = _request({"autonomous_delivery": False})
    assert NIRWorkflowMiddleware._with_delivery_contract(request) is request


def test_search_transition_defers_admission_until_final_context_request(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path, max_llm_calls=1)
    execute(service, submit(service, ctx, proposal(ctx)))
    with service.search.process._connection(write=True) as db:
        session = service.search._session(db, ctx["session_id"])
        session["llm_budget"]["search"]["calls"] = 1
        service.search._save(db, session)
    lease = reserve_call(service.search, ctx["session_id"], 1000, defer_finalization=True)
    assert lease is None
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "budget_exhausted" and session["stop_source"] == "budget"
    assert session["llm_budget"]["finalization"]["calls"] == 0


def test_semantic_errors_name_the_exact_missing_field(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    execute(service, submit(service, ctx, proposal(ctx)))
    ctx = service.context(ctx["session_id"], wf, "Feedback")
    bad = next_plan(ctx)
    bad["uncertainties"] = []
    with pytest.raises(ValueError, match=r"^uncertainties:"):
        submit(service, ctx, bad, "missing-uncertainty")


def test_final_schema_accepts_a_real_retention_contract(tmp_path):
    import jsonschema

    service, ctx, wf = setup_delivery(tmp_path, max_experiments=1)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    ctx = service.context(ctx["session_id"], wf, "Retain")
    plan = stop_plan(ctx, result["experiment_id"], "RETAIN")
    plan["alternatives"][1].update(action="RETAIN", pipeline=None, optimization=None)
    jsonschema.validate(plan, ctx["proposal_schema"])
    assert submit(service, ctx, plan, "retain")["action"] == "RETAIN"


def test_admission_estimate_includes_complete_tool_definitions(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    wf.update(stage="execution", search_session_id=ctx["session_id"], decision_context_id=ctx["context_id"])
    request = NIRWorkflowMiddleware._with_delivery_contract(_request(wf, ctx))
    first = NIRWorkflowMiddleware._admission_estimate(request)
    schema = copy.deepcopy(request.tools[0].args_schema)
    schema["$defs"]["Alternative"]["description"] = "x" * 1000
    request.tools[0] = request.tools[0].model_copy(update={"args_schema": schema})
    assert NIRWorkflowMiddleware._admission_estimate(request) >= first + 1000


def test_reserved_provider_failure_preserves_source_and_error_type(tmp_path):
    from deerflow.community.nir.decision_budget import finish_call

    service, ctx, wf = setup_delivery(tmp_path, max_experiments=1)
    execute(service, submit(service, ctx, proposal(ctx)))
    lease = reserve_call(service.search, ctx["session_id"], 100)
    finish_call(service.search, ctx["session_id"], lease, None, 0.1, failed=True, failure_reason={"exception_type": "TimeoutError", "http_status": None})
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stop_source"] == "budget"
    assert session["provider_failure"]["exception_type"] == "TimeoutError"
    assert session["llm_budget"]["finalization"]["calls"] == 1


def test_converted_provider_schema_has_resolvable_references(tmp_path):
    import jsonschema
    from langchain_core.utils.function_calling import convert_to_openai_tool

    service, ctx, wf = setup_delivery(tmp_path)
    wf.update(stage="execution", search_session_id=ctx["session_id"], decision_context_id=ctx["context_id"])
    request = NIRWorkflowMiddleware._with_delivery_contract(_request(wf, ctx))
    schema = convert_to_openai_tool(request.tools[0])["function"]["parameters"]
    plan = proposal(ctx)
    for alt in plan["alternatives"]:
        alt["action"] = "EXPLORE"
    jsonschema.validate({"proposal": plan}, schema)


def test_writable_parameter_schema_rejects_adapter_fields_and_accepts_svr_values(tmp_path):
    import jsonschema

    service, ctx, wf = setup_delivery(tmp_path)
    plan = proposal(ctx, family="svr", params={"C": 10.0, "gamma": "scale", "epsilon": 0.05})
    for alt in plan["alternatives"]:
        alt["action"] = "EXPLORE"
    jsonschema.validate(plan, ctx["proposal_schema"])
    plan["alternatives"][0]["pipeline"]["fixed_params"]["random_state"] = 42
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(plan, ctx["proposal_schema"])


def test_repair_receipt_reports_remaining_correction_attempts(tmp_path, monkeypatch):
    service, ctx, wf = setup_delivery(tmp_path)
    wf.update(stage="execution", revision=1, search_session_id=ctx["session_id"], decision_context_id=ctx["context_id"])
    monkeypatch.setattr(decision_tools, "_service", lambda runtime: service.search)
    for index, remaining in enumerate([2, 1, 0]):
        plan = proposal(ctx, params={"alpha": -1.0})
        call = "invalid-repair-" + str(index)
        issue(service, "nir_submit_decision", {"proposal": plan}, call)
        runtime = SimpleNamespace(state={"nir_workflow": wf}, tool_call_id=call, context={}, config={})
        command = decision_tools.nir_submit_decision_tool.func(runtime=runtime, proposal=plan)
        payload = json.loads(command.update["messages"][-1].content)
        assert payload["repair_contract"]["remaining_repairs"] == remaining
        assert payload["can_correct_proposal"] == (remaining > 0)


@pytest.mark.parametrize("finish_with", ["llm", "budget", "call_budget"])
def test_scoped_object_decisions_repair_and_deliver_through_real_lead(tmp_path, monkeypatch, finish_with):
    import jsonschema
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import PrivateAttr
    from test_nir_delivery_agent import DeliveryFakeLLM, make_delivery_lead

    from deerflow.community.nir import search_tools
    from deerflow.community.nir.process import runtime_store
    from deerflow.config.paths import get_paths

    class StructuredLLM(DeliveryFakeLLM):
        _schemas: dict = PrivateAttr(default_factory=dict)
        _bad_sent: bool = PrivateAttr(default=False)

        def bind_tools(self, tools, **kwargs):
            self._schemas = {tool.name: convert_to_openai_tool(tool)["function"]["parameters"] for tool in tools}
            return self

        def _generate(self, messages, **kwargs):
            result = super()._generate(messages, **kwargs)
            response = result.generations[0].message
            for call in response.tool_calls:
                args = call["args"]
                if call["name"] == "nir_workflow" and finish_with == "call_budget":
                    args["task_budget"] = json.dumps({"max_llm_calls": 6})
                if call["name"] == "nir_submit_decision":
                    plan = json.loads(args["proposal"])
                    for alt in plan["alternatives"]:
                        alt.setdefault("action", "EXPLORE")
                        if plan["phase"] == "final_selection":
                            alt.update(action="RETAIN", pipeline=None, optimization=None)
                    if not self._bad_sent and finish_with != "call_budget":
                        self._bad_sent = True
                        plan["alternatives"][0]["pipeline"]["optimization"] = plan["alternatives"][0].pop("optimization")
                        args["proposal"] = plan
                        return result  # Deliberate malformed model output must be rejected.
                    args["proposal"] = plan
                assert call["name"] in self._schemas, "Wrong process-stage tool was exposed/selected"
                jsonschema.validate(args, self._schemas[call["name"]])
            return result

    thread = "scoped-delivery-" + finish_with
    get_paths().ensure_thread_dirs(thread)
    source = get_paths().sandbox_uploads_dir(thread) / "source.npz"
    output = get_paths().sandbox_outputs_dir(thread)
    X = np.random.default_rng(621).normal(size=(60, 8))
    np.savez(source, X=X, y=X[:, 0] - X[:, 1], wv=np.arange(8, dtype=float))
    monkeypatch.setattr(search_tools, "_resolve", lambda runtime, path, **k: str(source) if k.get("read_only") else str(output))
    model = StructuredLLM(finish_with="budget" if finish_with == "budget" else "llm")
    result = make_delivery_lead(monkeypatch, model).invoke({"messages": [HumanMessage(content="使用 autonomous_delivery=true 自主建模并交付模型报告")]}, {"configurable": {"thread_id": thread}, "recursion_limit": 50})
    wf = result["nir_workflow"]
    assert wf["stage"] == "review", wf
    store = runtime_store(SimpleNamespace(config={"configurable": {"thread_id": thread}}, context={}))
    with store._connection() as db:
        session = search_tools.SearchSessionStore(store)._session(db, wf["search_session_id"])
    assert session["stage"] == "finalized" and session["export"]
    assert session["stop_source"] == ("llm" if finish_with == "llm" else "budget")
    assert session["llm_budget"]["finalization"]["calls"] <= 6
    assert session["fits"] == (4 if finish_with == "call_budget" else 8)
    errors = [json.loads(message.content) for message in result["messages"] if isinstance(message, ToolMessage) and message.name == "nir_submit_decision" and json.loads(message.content).get("status") == "error"]
    if finish_with != "call_budget":
        assert errors and errors[0]["errors"][0]["field"].startswith("alternatives.0")
