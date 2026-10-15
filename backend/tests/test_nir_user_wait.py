"""A question to the user is a real pause, never an implicit modeling choice."""

import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_nir_delivery_agent import make_delivery_lead
from test_nir_unified_agent import UnifiedFakeLLM
from test_nir_workflow_middleware import _request

from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware, _authorize
from deerflow.community.nir import search_tools
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.response_grounding import required_nir_workflow_action, validate_nir_response
from deerflow.community.nir.user_wait import begin_user_wait, question_for_user, resume_user_wait, waiting_for_user
from deerflow.community.nir.web_entry import bind_web_request, stored_workflow
from deerflow.config import paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tools.builtins.clarification_tool import ask_clarification_tool

QUESTION = "你希望我继续这一轮细化，还是直接确认当前模型并出报告？"
NUMBERED_QUESTION = "请告诉我你的选择，我再继续执行（每次只执行一个方向）：\n\n1. 继续微调参数；\n2. 更换预处理重试；\n3. 就此停止并生成报告。\n\n请回复 1 / 2 / 3（或补充信息），我收到后再继续。"


@pytest.mark.parametrize(
    "text",
    [
        QUESTION,
        "你想继续调整吗？",
        "你更倾向于继续调整还是生成报告？",
        "需要我继续调整参数吗？",
        "Would you like me to continue tuning or generate the report?",
        "Do you want me to continue?",
        "Shall I generate the report?",
        "请确认是否继续调整参数。",
        NUMBERED_QUESTION,
        "请回复 1 / 2 / 3，收到后再继续。",
        "请告诉我你的选择，我再继续执行：\n1. 继续调整\n2. 生成报告",
        "请先选择下一步\n1. 继续\n2. 停止",
        "等待你的回复后再继续。",
        "Please reply with 1, 2 or 3. I will continue after your response.",
    ],
)
def test_direct_user_choice_is_detected(text):
    assert question_for_user(text)


def test_numbered_choices_are_preserved_without_requiring_a_question_mark():
    question = question_for_user("已完成三轮比较。\n\n" + NUMBERED_QUESTION)
    assert question == NUMBERED_QUESTION


@pytest.mark.parametrize("text", ["我会继续调整参数，完成后生成报告。", "如果你希望查看参数，可以打开报告。", "模型训练和测试已完成。", "是否过拟合需要结合验证集判断。"])
def test_normal_progress_or_explanations_are_not_questions(text):
    assert not question_for_user(text)


@pytest.mark.parametrize("stage", ["decision_review", "review", "approved"])
def test_question_stops_continuation_and_all_execution_tools(stage):
    human = HumanMessage(content="对这个数据集进行定量建模", id="user-1")
    message = AIMessage(content="已完成四轮比较。" + QUESTION, id="question-1")
    workflow = {
        "stage": stage,
        "revision": 10,
        "task_type": "calibration",
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "autonomous_rounds": True,
        "autonomous_delivery": True,
        "decision_context_id": "ctx",
        "search_session_id": "search",
        "max_attempts": 40,
        "attempts": [],
    }
    before = copy.deepcopy(workflow)
    update = NIRWorkflowMiddleware().after_model({"nir_workflow": workflow, "messages": [human, message]}, SimpleNamespace(context={}))
    paused = update["nir_workflow"]
    assert update["jump_to"] == "end" and waiting_for_user(paused)
    assert required_nir_workflow_action(paused) is None
    assert paused["stage"] == workflow["stage"] and paused["decision_context_id"] == "ctx"
    assert workflow == before and QUESTION in update["messages"][-1].content
    assert validate_nir_response(update["messages"][-1].content, paused).passed
    for name in ("nir_submit_decision", "nir_execute_decision", "nir_prepare_search", "nir_freeze_search", "nir_finalize_search", "nir_register_model"):
        denied = _authorize(_request(name, {"nir_workflow": paused}))
        assert json.loads(denied.content)["code"] == "nir_waiting_for_user"
    hidden = HumanMessage(content="Continue immediately", id="hidden-1", name="nir_workflow_completion_reminder", additional_kwargs={"hide_from_ui": True})
    assert resume_user_wait(paused, [human, message, hidden]) is None
    assert resume_user_wait(paused, [human, message]) is None
    reply = HumanMessage(content="继续细化参数", id="user-2")
    resumed = resume_user_wait(paused, [human, message, reply])
    assert not waiting_for_user(resumed) and resumed["search_session_id"] == "search"
    assert resumed.get("approval_status") == paused.get("approval_status")


@pytest.mark.parametrize(
    "name,flags,content", [("nir_workflow_completion_reminder", {}, "继续执行"), ("summary", {}, "继续"), (None, {"hide_from_ui": True}, "继续"), (None, {"durable_context_data": True}, "继续"), (None, {}, "<memory>继续</memory>")]
)
def test_system_context_cannot_release_a_question(name, flags, content):
    human = HumanMessage(content="建模", id="start")
    pending = begin_user_wait({"revision": 1}, [human], QUESTION, "question")
    injected = HumanMessage(content=content, id="internal", name=name, additional_kwargs=flags)
    assert resume_user_wait(pending, [human, injected]) is None


def test_tool_question_does_not_publish_unverified_metrics_in_its_context():
    workflow = {"stage": "decision_review", "revision": 1, "task_type": "calibration", "search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "autonomous_rounds": True, "autonomous_delivery": True}
    args = {"question": "你希望使用 R²=0.9999 的模型吗？", "context": "模型已经通过外部验证。", "options": ["R²=0.9999"]}
    message = AIMessage(content="", id="ask", tool_calls=[{"id": "ask-tool", "name": "ask_clarification", "args": args}])
    update = NIRWorkflowMiddleware().after_model({"nir_workflow": workflow, "messages": [HumanMessage(content="建模", id="start"), message]}, SimpleNamespace(context={}))
    request = _request("ask_clarification", {"nir_workflow": update["nir_workflow"]}, args=args, call_id="ask-tool")
    result = ClarificationMiddleware().wrap_tool_call(request, lambda _: pytest.fail("Question must end this turn"))
    content = result.update["messages"][0].content
    assert "0.9999" not in content and "通过外部验证" not in content
    assert "请告诉我" in content


class QuestionLLM(UnifiedFakeLLM):
    question_style: str = "natural"
    question_text: str = QUESTION
    _asked = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage) and m.name != "ask_clarification"]
        if not self._asked and results and results[-1].name == "nir_decision_context" and json.loads(results[-1].content)["phase"] == "feedback":
            self._asked = True
            self._calls += 1
            calls = [] if self.question_style == "natural" else [{"id": "ask-user", "name": "ask_clarification", "args": {"question": self.question_text, "clarification_type": "approach_choice", "options": ["继续细化", "生成当前报告"]}}]
            if self.question_style in {"mixed", "natural_mixed"}:
                calls.append({"id": "must-not-run", "name": "nir_decision_context", "args": {}})
            if self.question_style == "natural_mixed":
                calls = [call for call in calls if call["name"] != "ask_clarification"]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.question_text if not calls or self.question_style == "natural_mixed" else "", id="question-real-lead", tool_calls=calls))])
        messages = [m for m in messages if not isinstance(m, ToolMessage) or m.name != "ask_clarification"]
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["natural", "tool", "mixed", "natural_mixed"])
@pytest.mark.parametrize("question", [QUESTION, NUMBERED_QUESTION, "请回复 1 / 2 / 3（或补充信息），我收到后再继续。"])
def test_actual_lead_waits_across_reentry_then_resumes_same_session(tmp_path, monkeypatch, asynchronous, style, question):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    thread = f"wait-{asynchronous}-{style}"
    owner = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=owner)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=owner) / "source.npz"
    output = paths.get_paths().sandbox_outputs_dir(thread, user_id=owner)
    X = np.random.default_rng(81).normal(size=(60, 8))
    np.savez(source, X=X, y=X[:, 0] * 2 + X[:, 1])
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source) if k.get("read_only") else str(output))
    opened = []
    load = np.load

    def spy(path, *a, **k):
        opened.append(str(path))
        return load(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 70}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [HumanMessage(content="请根据上传数据建模", id="user-start")]})
    bind_web_request(config, body)
    model = QuestionLLM(question_style=style, question_text=question)
    graph = make_delivery_lead(monkeypatch, model, middlewares=[NIRWorkflowMiddleware(), LoopDetectionMiddleware(), ClarificationMiddleware()], extra_tools=[ask_clarification_tool])

    def invoke(value):
        return asyncio.run(graph.ainvoke(value, config, context=config["context"])) if asynchronous else graph.invoke(value, config, context=config["context"])

    first = invoke(body.input)
    workflow = first["nir_workflow"]
    assert waiting_for_user(workflow)
    assert workflow["user_question"]["question"] == question
    process = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    assert waiting_for_user(stored_workflow(process))
    calls = model._calls
    with process._connection() as db:
        initial = search_tools.SearchSessionStore(process)._session(db, workflow["search_session_id"])
    assert initial["experiments"] == 1 and initial["fits"] == 4
    assert initial["freeze"] is None and initial["final"] is None
    assert not any("holdout.npz" in name for name in opened)
    again = invoke(first)
    assert model._calls == calls and waiting_for_user(again["nir_workflow"])
    assert not any("holdout.npz" in name for name in opened)
    next_input = {**again, "messages": [*again["messages"], HumanMessage(content="继续细化参数", id="user-reply")]}
    finished = invoke(next_input)
    assert not waiting_for_user(finished["nir_workflow"])
    assert finished["nir_workflow"]["search_session_id"] == workflow["search_session_id"]
    assert finished["nir_workflow"]["stage"] == "review" and finished["nir_workflow"]["approval_status"] == "pending"
    assert len([name for name in opened if "holdout.npz" in name]) == 1
