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


def test_semantic_trace_contains_python_symbols_imports_and_calls(tmp_path) -> None:
    src = tmp_path / "src"
    tests = tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    (src / "parser.py").write_text(
        "import json\n\n"
        "class Parser:\n"
        "    def parse(self, text):\n"
        "        return normalize(text)\n\n"
        "def normalize(text):\n"
        "    return text.strip()\n",
        encoding="utf-8",
    )
    (tests / "test_parser.py").write_text(
        "from src.parser import normalize\n\n"
        "def test_normalize():\n"
        "    assert normalize(' x ') == 'x'\n",
        encoding="utf-8",
    )

    trace = build_semantic_trace(
        "Refactor parser and run tests",
        ["src/parser.py", "tests/test_parser.py"],
        repository_root=tmp_path,
    )

    kinds = {node.kind for node in trace.repository_graph.nodes}
    assert {"file", "class", "method", "function", "test", "module"} <= kinds
    values = {node.value for node in trace.repository_graph.nodes}
    assert "src/parser.py::Parser.parse" in values
    assert "src/parser.py::normalize" in values
    assert "tests/test_parser.py::test_normalize" in values

    normalize_id = next(
        node.id for node in trace.repository_graph.nodes if node.value == "src/parser.py::normalize"
    )
    test_id = next(
        node.id
        for node in trace.repository_graph.nodes
        if node.value == "tests/test_parser.py::test_normalize"
    )
    assert any(
        edge.source == test_id and edge.relation == "calls" and edge.target == normalize_id
        for edge in trace.repository_graph.edges
    )
    assert "REQ-1" in trace.instruction_context()
    assert "Parser.parse" in trace.instruction_context()


def test_coverage_change_graph_and_impact_graph(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "core.py").write_text(
        "def calculate(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_core.py").write_text(
        "from src.core import calculate\n\n"
        "def test_calculate():\n    assert calculate(1) == 2\n",
        encoding="utf-8",
    )
    trace = build_semantic_trace(
        "Update calculate. Run tests.",
        ["src/core.py", "tests/test_core.py"],
        repository_root=tmp_path,
    )

    trace.record(
        "replace_in_file",
        {"path": "src/core.py", "requirement_ids": ["REQ-1"]},
        "Updated src/core.py; replaced 1 occurrence(s)",
    )
    assert trace.coverage()[0].status == "implemented"
    assert trace.unresolved_requirement_ids() == ["REQ-2"]

    trace.record(
        "run_command",
        {"command": "pytest", "requirement_ids": ["REQ-1", "REQ-2"]},
        "exit_code=0",
    )
    coverage = {item.requirement_id: item for item in trace.coverage()}
    assert coverage["REQ-1"].status == "verified"
    assert coverage["REQ-2"].status == "verified"
    assert trace.unresolved_requirement_ids() == []

    change_relations = {edge.relation for edge in trace.change_graph().edges}
    assert {"implemented_by", "validated_by", "targets"} <= change_relations
    impact_values = {node.value for node in trace.impact_graph().nodes}
    assert "src/core.py" in impact_values
    assert "src/core.py::calculate" in impact_values
    assert "tests/test_core.py::test_calculate" in impact_values
    payload = trace.to_dict()
    assert payload["schema"] == "ai-language.semantic-trace/v2"
    assert payload["unresolved_requirement_ids"] == []


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
    assert agent.last_trace.coverage()[0].status == "implemented"
    assert "value = 2" not in agent.trace_json()
    assert "requirement_ids" in observed[0]["instructions"]
