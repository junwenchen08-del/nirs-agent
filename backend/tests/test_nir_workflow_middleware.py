"""Tests for runtime enforcement of the NIR agent workflow."""

from __future__ import annotations

import json

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.workflow import start_workflow, transition_workflow


def _request(
    tool_name: str,
    state: dict,
    *,
    args: dict | None = None,
    call_id: str = "call-1",
    context: dict | None = None,
) -> ToolCallRequest:
    runtime = ToolRuntime(
        state=state,
        context=context or {},
        config={"configurable": {}},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={"id": call_id, "name": tool_name, "args": args or {}},
        tool=None,
        state=state,
        runtime=runtime,
    )


def _execution_state(
    *,
    max_attempts: int = 3,
    task_type: str = "calibration",
    validation_goal: str = "internal_holdout",
) -> dict:
    state = start_workflow(
        task_type=task_type,
        data_path="/mnt/user-data/uploads/data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal=validation_goal,
        instrument="nir-instrument-1" if validation_goal in {"external_validation", "production"} else None,
        grouping_column="batch" if validation_goal in {"external_validation", "production"} else None,
        reference_method="laboratory reference" if validation_goal in {"external_validation", "production"} else None,
        max_attempts=max_attempts,
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    return transition_workflow(state, action="plan_ready")


def _result(tool_name: str, payload: dict, *, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call_id,
        name=tool_name,
    )


def test_domain_tool_requires_started_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_analyze", {"status": "ok"})

    result = middleware.wrap_tool_call(_request("nir_analyze", {}), handler)

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["code"] == "nir_workflow_required"
    assert payload["next_action"] == "start_workflow"


def test_domain_tool_is_denied_outside_authorized_stage() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: _result("nir_train_model", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_workflow_stage_denied"
    assert payload["stage"] == "data_audit"
    assert payload["allowed_stages"] == ["execution"]
    assert payload["next_action"] == "inspect_data"


def test_domain_tool_denial_preserves_targeted_clarification_for_the_agent() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(task_type="calibration", data_path="data.npz")
    workflow = transition_workflow(
        workflow,
        action="record_audit",
        audit_passed=True,
        domain="food_protein",
        analyte="protein",
        unit="%",
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: _result("nir_train_model", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["stage"] == "clarification"
    assert payload["next_action"] == "ask_targeted_clarification"
    assert payload["missing_inputs"] == ["validation_goal"]
    assert payload["clarification_questions"][0]["fields"] == ["validation_goal"]


def test_tool_is_denied_for_incompatible_task_type() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="calibration")

    result = middleware.wrap_tool_call(
        _request("nir_predict", {"nir_workflow": workflow}),
        lambda _: _result("nir_predict", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    assert json.loads(result.update["messages"][0].content)["code"] == "nir_workflow_task_denied"


@pytest.mark.parametrize(
    "tool_name",
    [
        "nir_train_auto_split_model",
        "nir_train_model",
        "nir_train_partitioned_model",
        "nir_analyze",
        "nir_analyze_collection",
        "nir_compare",
        "nir_register_model",
    ],
)
def test_exploratory_goal_hard_blocks_modeling_and_registration(tool_name: str) -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    workflow["validation_goal"] = "exploratory"

    result = middleware.wrap_tool_call(
        _request(tool_name, {"nir_workflow": workflow}),
        lambda _: _result(tool_name, {"status": "ok", "passed": True}),
    )

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert message.status == "error"
    payload = json.loads(message.content)
    assert payload["code"] == "nir_validation_goal_conflict"
    assert payload["details"]["validation_goal"] == "exploratory"
    assert payload["details"]["action_required"] == ("report_exploratory_findings_or_request_validation_goal_change")
    assert result.update["nir_workflow"]["tool_observations"][-1]["code"] == ("nir_validation_goal_conflict")


def test_internal_holdout_goal_still_allows_auto_split_modeling() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    original = _result(
        "nir_train_auto_split_model",
        {"status": "ok", "passed": False, "grade": "D"},
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_auto_split_model", {"nir_workflow": workflow}),
        lambda _: original,
    )

    assert isinstance(result, Command)
    assert result.update["messages"] == [original]
    assert result.update["nir_workflow"]["attempt"] == 1


def test_internal_holdout_goal_rejects_external_partition_protocol() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_train_partitioned_model", {"status": "ok", "passed": True})

    result = middleware.wrap_tool_call(
        _request("nir_train_partitioned_model", {"nir_workflow": workflow}),
        handler,
    )

    assert called is False
    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_validation_goal_conflict"
    assert payload["details"]["required_validation_scope"] == "independent_holdout_not_external"


@pytest.mark.parametrize("validation_goal", ["external_validation", "production"])
def test_high_assurance_goals_require_external_partition_protocol(validation_goal: str) -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis", validation_goal=validation_goal)
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_train_auto_split_model", {"status": "ok", "passed": True})

    result = middleware.wrap_tool_call(
        _request("nir_train_auto_split_model", {"nir_workflow": workflow}),
        handler,
    )

    assert called is False
    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_validation_goal_conflict"
    assert payload["details"]["allowed_modeling_tools"] == ["nir_train_partitioned_model"]
    assert payload["details"]["required_validation_scope"] == "independent_external_validation"


def test_unrelated_tool_passes_through_without_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    original = ToolMessage(content="ok", tool_call_id="call-1", name="read_file")

    result = middleware.wrap_tool_call(_request("read_file", {}), lambda _: original)

    assert result is original


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        (
            "write_file",
            {
                "path": "/mnt/user-data/workspace/inspect_mat.py",
                "content": "import scipy.io\nprint('inspect')\n",
            },
        ),
        ("bash", {"command": "python /mnt/user-data/workspace/inspect_mat.py"}),
        # Regression: python3.10 was a false negative (\\d* matched only '3',
        # then '.' failed the trailing \\s/$ boundary). Now covered by \\d*(?:\\.\\d+)*.
        ("bash", {"command": "python3.10 /mnt/user-data/workspace/inspect_mat.py"}),
        # python after a shell separator must still be blocked.
        ("bash", {"command": "echo prepare && python /mnt/user-data/workspace/inspect_mat.py"}),
        # sudo/time/env prefixes are command-start positions, must block.
        ("bash", {"command": "sudo python /mnt/user-data/workspace/inspect_mat.py"}),
        # Direct script execution (shebang) at command-start position.
        ("bash", {"command": "/mnt/user-data/workspace/inspect_mat.py"}),
        ("bash", {"command": "./run_analysis.py --flag"}),
        # R/Julia/MATLAB interpreters (previously caught only by broad suffix
        # check; now covered by _INTERPRETER_COMMAND_RE).
        ("bash", {"command": "Rscript /mnt/user-data/workspace/cv.R"}),
        ("bash", {"command": "julia /mnt/user-data/workspace/sim.jl"}),
        ("bash", {"command": "matlab -batch /mnt/user-data/workspace/run.m"}),
    ],
)
def test_active_nir_workflow_blocks_python_analysis_fallbacks(tool_name: str, args: dict) -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/tablets.MAT",
        analyte="active",
        unit="%w/w",
        domain="pharma",
    )
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result(tool_name, {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request(tool_name, {"nir_workflow": workflow}, args=args),
        handler,
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["code"] == "nir_code_execution_forbidden"
    assert payload["next_action"] == "inspect_data"


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/user-data/uploads/data.csv",
        "/mnt/user-data/uploads/DATA.MAT",
        "/mnt/user-data/uploads/spectra.npz",
        "/mnt/user-data/uploads/export.xlsx",
    ],
)
def test_active_nir_workflow_blocks_raw_data_from_model_context(path: str) -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path=path,
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="exploratory",
    )
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("read_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request("read_file", {"nir_workflow": workflow}, args={"path": path}),
        handler,
    )

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["code"] == "nir_raw_data_read_forbidden"
    assert payload["details"]["action_required"] == "use_nir_inspect_summary"


def test_completed_exploratory_workflow_still_blocks_raw_data_read() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path="/mnt/user-data/uploads/data.csv",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="exploratory",
    )
    workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
    workflow = transition_workflow(workflow, action="plan_ready")

    result = middleware.wrap_tool_call(
        _request(
            "read_file",
            {"nir_workflow": workflow},
            args={"path": "/mnt/user-data/uploads/data.csv"},
        ),
        lambda _: _result("read_file", {"status": "ok"}),
    )

    assert isinstance(result, ToolMessage)
    assert json.loads(result.content)["code"] == "nir_raw_data_read_forbidden"
    assert json.loads(result.content)["stage"] == "completed"


def test_active_nir_workflow_still_allows_small_metrics_report_read() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    original = _result("read_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request(
            "read_file",
            {"nir_workflow": workflow},
            args={"path": "/mnt/user-data/outputs/metrics.json"},
        ),
        lambda _: original,
    )

    assert result is original


@pytest.mark.parametrize(
    "command",
    [
        # Regression: ``python`` as a search term / argument must NOT be blocked.
        "grep python requirements.txt",
        "echo python",
        "echo use python for this",
        "cat notes.txt | grep python",
        "find . -name python",
        "ls /usr/lib/python3.10/site-packages",
        "cd python-projects",
        "grep -r pythonic .",
    ],
)
def test_active_nir_workflow_allows_bash_mentioning_python_as_argument(command: str) -> None:
    """Regression: the previous regex treated any whitespace before ``python``
    as a command-start signal, blocking legitimate bash commands that mention
    ``python`` as a search term or argument (e.g. ``grep python file``)."""
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path="/mnt/user-data/uploads/data.mat",
        analyte="protein",
        unit="%",
        domain="default",
    )
    original = _result("bash", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request("bash", {"nir_workflow": workflow}, args={"command": command}),
        lambda _: original,
    )

    assert result is original


@pytest.mark.parametrize(
    "command",
    [
        # Regression: script suffix as an argument must NOT be blocked. The old
        # _SCRIPT_SUFFIX_RE.search(command) matched .py at end of string,
        # blocking non-executing commands like cat/grep/ls/rm on script files.
        "cat script.py",
        "grep pattern file.py",
        "ls script.py",
        "rm script.py",
        "echo script.py",
        "mv a.py b.txt",
        "cp config.py config.py.bak",
        "chmod +x script.py",
        "cat notes.txt | grep foo file.py",
    ],
)
def test_active_nir_workflow_allows_bash_operating_on_script_files(command: str) -> None:
    """Regression: a script extension appearing as a file argument (not at a
    command-start position) must NOT be blocked. Only direct execution of a
    script file (e.g. ``./script.py``) or interpreter invocation should be
    blocked."""
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path="/mnt/user-data/uploads/data.mat",
        analyte="protein",
        unit="%",
        domain="default",
    )
    original = _result("bash", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request("bash", {"nir_workflow": workflow}, args={"command": command}),
        lambda _: original,
    )

    assert result is original


def test_active_nir_workflow_allows_non_script_artifact_write() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path="/mnt/user-data/uploads/data.mat",
        analyte="protein",
        unit="%",
        domain="default",
    )
    original = _result("write_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request(
            "write_file",
            {"nir_workflow": workflow},
            args={"path": "/mnt/user-data/outputs/notes.md", "content": "# Notes"},
        ),
        lambda _: original,
    )

    assert result is original


@pytest.mark.parametrize(
    ("path", "content"),
    [
        # Empty __init__.py: script suffix but no analysis imports
        ("/mnt/user-data/workspace/pkg/__init__.py", ""),
        # config.py without analysis imports
        ("/mnt/user-data/workspace/config.py", "DEBUG = True\n"),
        # Non-analysis Python module (no numpy/pandas/scipy/sklearn/nir_core)
        ("/mnt/user-data/workspace/utils.py", "def helper():\n    return 42\n"),
    ],
)
def test_active_nir_workflow_allows_legitimate_python_module_files(path: str, content: str) -> None:
    """Regression: script-suffix alone must NOT block legitimate Python module
    files. Only script suffix AND analysis-style imports should be blocked."""
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="analysis",
        data_path="/mnt/user-data/uploads/data.mat",
        analyte="protein",
        unit="%",
        domain="default",
    )
    original = _result("write_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request(
            "write_file",
            {"nir_workflow": workflow},
            args={"path": path, "content": content},
        ),
        lambda _: original,
    )

    assert result is original


def test_successful_model_result_automatically_enters_review() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = _result(
        "nir_train_model",
        {
            "status": "ok",
            "passed": True,
            "grade": "A",
            "model_path": "/mnt/user-data/outputs/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics.json",
        },
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    assert result.update["messages"] == [message]
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "review"
    assert updated["attempt"] == 1
    assert updated["approval_status"] == "pending"
    assert updated["model_path"] == "/mnt/user-data/outputs/model.pkl"
    assert updated["metrics_path"] == "/mnt/user-data/outputs/metrics.json"


def test_successful_auto_split_model_result_automatically_enters_review() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    message = _result(
        "nir_train_auto_split_model",
        {
            "status": "ok",
            "passed": True,
            "grade": "B",
            "model_path": "/mnt/user-data/outputs/auto-split.pkl",
            "metrics_path": "/mnt/user-data/outputs/auto-split.json",
            "evidence": {
                "schema_version": 1,
                "protocol": "deterministic_auto_split_holdout",
                "validation_scope": "independent_holdout_not_external",
                "model_sha256": "a" * 64,
                "metrics_sha256": "b" * 64,
                "training_data_sha256": "c" * 64,
            },
        },
    )

    result = middleware.wrap_tool_call(
        _request(
            "nir_train_auto_split_model",
            {"nir_workflow": workflow},
            call_id="model-call-7",
            context={"run_id": "run-7", "deerflow_trace_id": "trace-7"},
        ),
        lambda _: message,
    )

    assert isinstance(result, Command)
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "review"
    assert updated["attempt"] == 1
    assert updated["model_path"].endswith("auto-split.pkl")
    assert updated["attempt_evidence"] == {
        "schema_version": 1,
        "tool_name": "nir_train_auto_split_model",
        "tool_call_id": "model-call-7",
        "run_id": "run-7",
        "trace_id": "trace-7",
        "protocol": "deterministic_auto_split_holdout",
        "validation_scope": "independent_holdout_not_external",
        "model_path": "/mnt/user-data/outputs/auto-split.pkl",
        "metrics_path": "/mnt/user-data/outputs/auto-split.json",
        "model_sha256": "a" * 64,
        "metrics_sha256": "b" * 64,
        "training_data_sha256": "c" * 64,
        "method": "auto",
        "pipeline_steps": [{"method": "tool_default", "params": {}}],
        "model_args": {},
        "execution_signature": updated["attempt_evidence"]["execution_signature"],
        "metrics_summary": {"grade": "B", "passed": True},
    }


def test_successful_mat_collection_result_automatically_enters_review() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="analysis")
    message = _result(
        "nir_analyze_collection",
        {
            "status": "ok",
            "passed": True,
            "grade": "excellent",
            "primary_subset": "R562",
            "model_path": "/mnt/user-data/outputs/nir_collection/R562/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/nir_collection/R562/metrics.json",
        },
    )

    result = middleware.wrap_tool_call(
        _request("nir_analyze_collection", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "review"
    assert updated["attempt"] == 1
    assert updated["model_path"].endswith("/R562/model.pkl")


def test_successful_multi_model_result_automatically_enters_review() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="multi_modeling")
    message = _result(
        "nir_train_multi_model",
        {
            "status": "ok",
            "passed": True,
            "grade": "good",
            "model_path": "/mnt/user-data/outputs/multi-model.pkl",
            "metrics_path": "/mnt/user-data/outputs/multi-metrics.json",
        },
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_multi_model", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "review"
    assert updated["attempt"] == 1
    assert updated["model_path"].endswith("multi-model.pkl")


def test_successful_tool_observation_is_checkpointed_with_run_context() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()

    result = middleware.wrap_tool_call(
        _request(
            "nir_train_model",
            {"nir_workflow": workflow},
            context={"run_id": "run-001", "deerflow_trace_id": "trace-001"},
        ),
        lambda _: _result(
            "nir_train_model",
            {
                "status": "ok",
                "passed": True,
                "grade": "A",
                "model_path": "model.pkl",
                "metrics_path": "metrics.json",
            },
        ),
    )

    assert isinstance(result, Command)
    updated = result.update["nir_workflow"]
    assert updated["run_ids"] == ["run-001"]
    assert updated["trace_ids"] == ["trace-001"]
    assert updated["tool_observations"][-1] == {
        "name": "nir_train_model",
        "status": "success",
        "stage_before": "execution",
        "stage_after": "review",
        "code": None,
        "requested_method": "pls",
        "call_id": "call-1",
        "run_id": "run-001",
        "trace_id": "trace-001",
    }


def test_unrelated_tool_is_not_recorded_in_active_nir_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = _result("read_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request("read_file", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert result is message


def test_denied_tool_observation_is_checkpointed_for_agent_evaluation() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )

    result = middleware.wrap_tool_call(
        _request(
            "nir_train_model",
            {"nir_workflow": workflow},
            context={"run_id": "run-denied"},
        ),
        lambda _: _result("nir_train_model", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert json.loads(message.content)["code"] == "nir_workflow_stage_denied"
    observation = result.update["nir_workflow"]["tool_observations"][-1]
    assert observation["status"] == "error"
    assert observation["code"] == "nir_workflow_stage_denied"
    assert observation["stage_before"] == "data_audit"
    assert observation["stage_after"] == "data_audit"


def test_failed_model_requires_bound_reflection_then_search_returns_to_retry_planning() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    model_result = middleware.wrap_tool_call(
        _request("nir_analyze", {"nir_workflow": workflow}),
        lambda _: _result(
            "nir_analyze",
            {
                "status": "ok",
                "passed": False,
                "grade": "C",
                "model": "/mnt/user-data/outputs/model.pkl",
                "metrics": "/mnt/user-data/outputs/metrics.json",
            },
        ),
    )
    assert isinstance(model_result, Command)
    failed_state = model_result.update["nir_workflow"]
    assert failed_state["stage"] == "evaluation"
    assert failed_state["attempt"] == 1

    captured_args: dict = {}

    def reflect_handler(request: ToolCallRequest) -> ToolMessage:
        captured_args.update(request.tool_call["args"])
        return _result(
            "nir_reflect",
            {
                "status": "ok",
                "attempt": 1,
                "should_retry": True,
                "reason": "Residual curvature remains.",
                "diagnostics": {"residual_trend": "curved"},
                "knowledge_hint": {"query": "NIR curved residual derivative"},
            },
            call_id="call-reflect",
        )

    reflect_result = middleware.wrap_tool_call(
        _request(
            "nir_reflect",
            {"nir_workflow": failed_state},
            args={"metrics_path": "/wrong.json", "attempt": 99, "history": '[{"fake": true}]'},
            call_id="call-reflect",
        ),
        reflect_handler,
    )
    assert isinstance(reflect_result, Command)
    reflected_state = reflect_result.update["nir_workflow"]
    assert reflected_state["stage"] == "knowledge"
    assert captured_args["metrics_path"] == "/mnt/user-data/outputs/metrics.json"
    assert captured_args["attempt"] == 1
    assert captured_args["history"] == "[]"

    search_result = middleware.wrap_tool_call(
        _request("nir_search_knowledge", {"nir_workflow": reflected_state}, call_id="call-2"),
        lambda _: _result(
            "nir_search_knowledge",
            {"results": [{"id": "paper-001"}], "count": 1},
            call_id="call-2",
        ),
    )

    assert isinstance(search_result, Command)
    retried_state = search_result.update["nir_workflow"]
    assert retried_state["stage"] == "planning"
    assert retried_state["next_action"] == "prepare_retry_plan"


def test_retry_modeling_call_must_match_recorded_plan() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    first = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": workflow},
            args={"method": "auto", "pipeline_steps": '["snv"]'},
        ),
        lambda _: _result(
            "nir_analyze",
            {
                "status": "ok",
                "passed": False,
                "grade": "C",
                "metrics": "/mnt/user-data/outputs/metrics.json",
            },
        ),
    )
    failed = first.update["nir_workflow"]
    reflected = transition_workflow(
        failed,
        action="record_reflection",
        reflection_result={
            "attempt": 1,
            "should_retry": True,
            "diagnostics": {"residual_trend": "curved"},
        },
    )
    planned = transition_workflow(
        reflected,
        action="record_retry_plan",
        retry_tool="nir_analyze",
        retry_method="auto",
        retry_pipeline_steps='["snv", {"method": "derivative1", "params": {"window": 11}}]',
        retry_rationale="Use a derivative to address residual curvature.",
        expected_improvement="Lower RMSEP without increasing bias.",
    )
    execution = transition_workflow(planned, action="plan_ready")
    called = False

    def wrong_handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_analyze", {"status": "ok"})

    denied = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": execution},
            args={"method": "auto", "pipeline_steps": '["snv", "autoscale"]'},
        ),
        wrong_handler,
    )
    assert called is False
    assert json.loads(denied.update["messages"][0].content)["code"] == "nir_retry_plan_mismatch"

    changed_hyperparameter = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": execution},
            args={
                "method": "auto",
                "pipeline_steps": '["snv", {"method": "derivative1", "params": {"window": 11}}]',
                "max_components": 8,
            },
        ),
        wrong_handler,
    )
    assert called is False
    assert json.loads(changed_hyperparameter.update["messages"][0].content)["code"] == "nir_retry_plan_mismatch"

    allowed = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": execution},
            args={
                "method": "auto",
                "pipeline_steps": '["snv", {"method": "derivative1", "params": {"window": 11}}]',
            },
        ),
        lambda _: _result(
            "nir_analyze",
            {
                "status": "ok",
                "passed": True,
                "grade": "B",
                "model": "/mnt/user-data/outputs/retry.pkl",
                "metrics": "/mnt/user-data/outputs/retry.json",
            },
        ),
    )
    assert allowed.update["nir_workflow"]["stage"] == "review"
    assert allowed.update["nir_workflow"]["retry_plan"]["status"] == "executed"


def test_tool_error_does_not_advance_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = ToolMessage(
        content=json.dumps({"status": "error", "error": "training failed"}),
        tool_call_id="call-1",
        name="nir_train_model",
        status="error",
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    assert result.update["messages"] == [message]
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "execution"
    assert updated["attempt"] == 0
    assert updated["tool_observations"][-1]["status"] == "error"


def test_failed_explicit_model_cannot_be_silently_replaced() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    failed = ToolMessage(
        content=json.dumps(
            {
                "status": "error",
                "code": "nir_model_runtime_unavailable",
                "error": "PyTorch is unavailable",
            }
        ),
        tool_call_id="call-cnn",
        name="nir_analyze",
        status="error",
    )
    first_result = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": workflow, "messages": [HumanMessage("使用 1D-CNN 建模")]},
            args={"method": "cnn"},
            call_id="call-cnn",
        ),
        lambda _: failed,
    )
    assert isinstance(first_result, Command)

    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_analyze", {"status": "ok"}, call_id="call-mlp")

    denied = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {
                "nir_workflow": first_result.update["nir_workflow"],
                "messages": [HumanMessage("使用 1D-CNN 建模")],
            },
            args={"method": "mlp"},
            call_id="call-mlp",
        ),
        handler,
    )

    assert called is False
    assert isinstance(denied, Command)
    payload = json.loads(denied.update["messages"][0].content)
    assert payload["code"] == "nir_model_substitution_requires_approval"
    assert payload["details"]["failed_method"] == "cnn"
    assert payload["details"]["attempted_method"] == "mlp"


def test_latest_user_can_explicitly_approve_model_substitution() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    failed = ToolMessage(
        content=json.dumps({"status": "error", "error": "training failed"}),
        tool_call_id="call-cnn",
        name="nir_analyze",
        status="error",
    )
    first_result = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {"nir_workflow": workflow, "messages": [HumanMessage("使用 CNN")]},
            args={"method": "cnn"},
            call_id="call-cnn",
        ),
        lambda _: failed,
    )
    assert isinstance(first_result, Command)
    replacement = _result(
        "nir_analyze",
        {"status": "ok", "passed": True, "grade": "A"},
        call_id="call-mlp",
    )

    allowed = middleware.wrap_tool_call(
        _request(
            "nir_analyze",
            {
                "nir_workflow": first_result.update["nir_workflow"],
                "messages": [HumanMessage("CNN 失败的话，改用 MLP 吧")],
            },
            args={"method": "mlp"},
            call_id="call-mlp",
        ),
        lambda _: replacement,
    )

    assert isinstance(allowed, Command)
    assert allowed.update["messages"] == [replacement]
    assert allowed.update["nir_workflow"]["tool_observations"][-1]["requested_method"] == "mlp"


@pytest.mark.parametrize(
    "message",
    [
        "不要改用 MLP",
        "我不想使用 MLP",
        "Do not use MLP",
    ],
)
def test_model_substitution_denial_is_not_mistaken_for_approval(message: str) -> None:
    from deerflow.agents.middlewares.nir_workflow_middleware import (
        _latest_user_approved_substitution,
    )

    request = _request(
        "nir_analyze",
        {"messages": [HumanMessage(message)]},
        args={"method": "mlp"},
    )

    assert _latest_user_approved_substitution(request, "mlp") is False


def test_compare_uses_best_result_and_preserves_existing_command_fields() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="compare")
    message = _result(
        "nir_compare",
        {
            "status": "ok",
            "best": {"passed": True, "grade": "B"},
            "all_metrics": "/mnt/user-data/outputs/all_metrics.json",
        },
    )
    command = Command(update={"messages": [message], "artifact": "gallery"}, goto="next")

    result = middleware.wrap_tool_call(
        _request("nir_compare", {"nir_workflow": workflow}),
        lambda _: command,
    )

    assert isinstance(result, Command)
    assert result.goto == "next"
    assert result.update["artifact"] == "gallery"
    assert result.update["nir_workflow"]["stage"] == "review"
    assert result.update["nir_workflow"]["metrics_path"].endswith("all_metrics.json")


def test_knowledge_task_completes_after_successful_search() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(task_type="knowledge")

    result = middleware.wrap_tool_call(
        _request("nir_search_knowledge", {"nir_workflow": workflow}),
        lambda _: _result("nir_search_knowledge", {"results": [], "count": 0}),
    )

    assert isinstance(result, Command)
    assert result.update["nir_workflow"]["stage"] == "completed"
    assert result.update["nir_workflow"]["next_action"] == "none"


def test_knowledge_evidence_prefers_stable_evidence_id() -> None:
    from deerflow.agents.middlewares.nir_workflow_middleware import (
        _knowledge_evidence_ids,
    )

    assert _knowledge_evidence_ids(
        {
            "results": [
                {
                    "evidence_id": "doi:10.1000/paper#cjk-section-v2-0001",
                    "source": "renamable-file.pdf",
                }
            ]
        }
    ) == ["doi:10.1000/paper#cjk-section-v2-0001"]


def test_successful_registration_automatically_enters_registered_stage() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    workflow = transition_workflow(
        workflow,
        action="record_attempt",
        attempt_passed=True,
        model_path="/mnt/user-data/outputs/model.pkl",
        metrics_path="/mnt/user-data/outputs/metrics.json",
        attempt_evidence={
            "schema_version": 1,
            "tool_name": "nir_train_model",
            "protocol": "random_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "model_path": "/mnt/user-data/outputs/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics.json",
        },
    )
    workflow = transition_workflow(workflow, action="approve")

    result = middleware.wrap_tool_call(
        _request(
            "nir_register_model",
            {"nir_workflow": workflow},
            args={
                "model_path": "/mnt/user-data/outputs/model.pkl",
                "metrics_path": "/mnt/user-data/outputs/metrics.json",
            },
        ),
        lambda _: _result(
            "nir_register_model",
            {"status": "registered", "model_id": "corn-protein", "version": 1},
        ),
    )

    assert isinstance(result, Command)
    assert result.update["nir_workflow"]["stage"] == "registered"
    assert result.update["nir_workflow"]["next_action"] == "complete_workflow"


def test_registration_rejects_artifacts_not_bound_to_approved_attempt() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    workflow = transition_workflow(
        workflow,
        action="record_attempt",
        attempt_passed=True,
        model_path="/mnt/user-data/outputs/approved.pkl",
        metrics_path="/mnt/user-data/outputs/approved.json",
        attempt_evidence={
            "schema_version": 1,
            "tool_name": "nir_train_model",
            "protocol": "random_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "model_path": "/mnt/user-data/outputs/approved.pkl",
            "metrics_path": "/mnt/user-data/outputs/approved.json",
        },
    )
    workflow = transition_workflow(workflow, action="approve")
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_register_model", {"status": "registered"})

    result = middleware.wrap_tool_call(
        _request(
            "nir_register_model",
            {"nir_workflow": workflow},
            args={
                "model_path": "/mnt/user-data/outputs/different.pkl",
                "metrics_path": "/mnt/user-data/outputs/approved.json",
            },
        ),
        handler,
    )

    assert called is False
    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_registration_evidence_mismatch"


@pytest.mark.anyio
async def test_async_wrapper_enforces_the_same_policy() -> None:
    middleware = NIRWorkflowMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_predict", {"status": "ok"})

    result = await middleware.awrap_tool_call(_request("nir_predict", {}), handler)

    assert called is False
    assert isinstance(result, ToolMessage)
    assert json.loads(result.content)["code"] == "nir_workflow_required"
