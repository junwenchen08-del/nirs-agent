"""Tests for deterministic routing of natural-language NIR requests."""

from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares import skill_activation_middleware as middleware_module
from deerflow.agents.middlewares.skill_activation_middleware import (
    SkillActivationMiddleware,
    SkillAutoRoute,
    is_slash_skill_activation_reminder,
)
from deerflow.skills.types import Skill, SkillCategory


def _make_skill(tmp_path: Path, name: str) -> Skill:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"# {name}\nFollow the domain workflow.", encoding="utf-8")
    return Skill(
        name=name,
        description="NIR workflow",
        license="MIT",
        skill_dir=skill_dir,
        skill_file=skill_file,
        relative_path=Path(name),
        category=SkillCategory.CUSTOM,
        enabled=True,
    )


def _storage(tmp_path: Path, skills: list[Skill]):
    return SimpleNamespace(
        load_skills=lambda *, enabled_only: skills,
        get_container_root=lambda: "/mnt/skills",
        get_skills_root_path=lambda: tmp_path,
    )


def _request(message: HumanMessage, *, state: dict | None = None) -> ModelRequest:
    request_state = {"messages": [message], **(state or {})}
    return ModelRequest(
        model=object(),
        messages=[message],
        state=request_state,
    )


def test_natural_language_nir_request_automatically_loads_coordinator(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path, "nir-coordinator")
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _storage(tmp_path, [skill]),
    )
    middleware = SkillActivationMiddleware(auto_routes=(SkillAutoRoute("nir-coordinator", (r"近红外(?:光谱)?",)),))
    captured = {}

    def handler(request: ModelRequest):
        captured["messages"] = request.messages
        return AIMessage(content="ok")

    middleware.wrap_model_call(
        _request(HumanMessage(content="分析这批近红外光谱并建立水分模型", id="msg-1")),
        handler,
    )

    activation = captured["messages"][0]
    assert is_slash_skill_activation_reminder(activation)
    assert 'mode="automatic"' in activation.content
    assert "runtime deterministically routed" in activation.content
    assert "Follow the domain workflow." in activation.content


def test_unrelated_request_does_not_activate_nir_skill(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path, "nir-coordinator")
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _storage(tmp_path, [skill]),
    )
    middleware = SkillActivationMiddleware(auto_routes=(SkillAutoRoute("nir-coordinator", (r"近红外",)),))
    original = HumanMessage(content="总结这个普通文本文件")
    captured = {}

    def handler(request: ModelRequest):
        captured["messages"] = request.messages
        return AIMessage(content="ok")

    middleware.wrap_model_call(_request(original), handler)

    assert captured["messages"] == [original]


def test_active_nir_workflow_keeps_coordinator_loaded_for_follow_up(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path, "nir-coordinator")
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _storage(tmp_path, [skill]),
    )
    middleware = SkillActivationMiddleware(
        auto_routes=(
            SkillAutoRoute(
                "nir-coordinator",
                (r"近红外",),
                state_key="nir_workflow",
            ),
        )
    )
    captured = {}

    def handler(request: ModelRequest):
        captured["messages"] = request.messages
        return AIMessage(content="ok")

    middleware.wrap_model_call(
        _request(
            HumanMessage(content="继续", id="msg-2"),
            state={"nir_workflow": {"stage": "planning"}},
        ),
        handler,
    )

    activation = captured["messages"][0]
    assert is_slash_skill_activation_reminder(activation)
    assert 'mode="automatic"' in activation.content


def test_completed_nir_workflow_does_not_route_unrelated_follow_up(monkeypatch, tmp_path):
    skill = _make_skill(tmp_path, "nir-coordinator")
    monkeypatch.setattr(
        middleware_module,
        "get_or_new_skill_storage",
        lambda **kwargs: _storage(tmp_path, [skill]),
    )
    middleware = SkillActivationMiddleware(
        auto_routes=(
            SkillAutoRoute(
                "nir-coordinator",
                (r"近红外",),
                state_key="nir_workflow",
                terminal_statuses=("completed", "blocked"),
            ),
        )
    )
    original = HumanMessage(content="继续处理普通文档", id="msg-3")
    captured = {}

    def handler(request: ModelRequest):
        captured["messages"] = request.messages
        return AIMessage(content="ok")

    middleware.wrap_model_call(
        _request(
            original,
            state={"nir_workflow": {"stage": "completed"}},
        ),
        handler,
    )

    assert captured["messages"] == [original]
