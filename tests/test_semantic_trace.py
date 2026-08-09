from __future__ import annotations

from types import SimpleNamespace

import openai

from ai_language.agent import CodingAgent, Workspace
from ai_language.semantic_trace import build_semantic_trace, extract_requirements


def test_extract_requirements_builds_stable_ids_and_kinds() -> None:
    requirements = extract_requirements(
        "Добавь rate limiting, но не ломай public API, и добавь тесты."
    )

    assert [requirement.id for requirement in requirements] == ["REQ-1", "REQ-2", "REQ-3"]
    assert requirements[0].kind == "action"
    assert requirements[1].kind == "constraint"
    assert requirements[2].kind == "validation"


def test_semantic_trace_contains_intent_and_repository_graphs() -> None:
    trace = build_semantic_trace(
        "Refactor parser and run tests",
        ["src/parser.py", "tests/test_parser.py"],
    )

    assert trace.intent_graph.nodes[0].kind == "task"
    assert {node.value for node in trace.repository_graph.nodes if node.kind == "file"} == {
        "src/parser.py",
        "tests/test_parser.py",
    }
    assert "REQ-1" in trace.instruction_context()
    assert "src/parser.py" in trace.instruction_context()


def test_agent_records_requirement_to_change_trace(monkeypatch, tmp_path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("value = 1\n", encoding="utf-8")
    observed: list[dict] = []

    class FakeResponses:
        def create(self, **kwargs):
            observed.append(kwargs)
            if len(observed) == 1:
                call = SimpleNamespace(
                    type="function_call",
                    name="replace_in_file",
                    arguments=(
                        '{"path":"demo.py","old_text":"value = 1",'
                        '"new_text":"value = 2","requirement_ids":["REQ-1"]}'
                    ),
                    call_id="call-1",
                )
                return SimpleNamespace(id="response-1", output=[call], output_text="")
            return SimpleNamespace(id="response-2", output=[], output_text="done")

    fake_client = SimpleNamespace(responses=FakeResponses())
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: fake_client)

    agent = CodingAgent(
        workspace=Workspace(tmp_path, auto_approve=True),
        api_key="test-key",
        model="gpt-5.6",
        max_steps=1,
    )

    assert agent.run("Update demo value") == "done"
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert agent.last_trace is not None
    assert len(agent.last_trace.events) == 1
    event = agent.last_trace.events[0]
    assert event.tool == "replace_in_file"
    assert event.target == "demo.py"
    assert event.requirement_ids == ("REQ-1",)
    assert event.status == "ok"
    assert "value = 2" not in agent.trace_json()
    assert "requirement_ids" in observed[0]["instructions"]
