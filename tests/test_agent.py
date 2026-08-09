from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from ai_language.agent import CodingAgent, Workspace, WorkspaceSecurityError


def test_workspace_tools_are_confined_and_hide_secrets(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "demo.py"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=never-expose-this", encoding="utf-8")

    workspace = Workspace(tmp_path, auto_approve=True)

    assert "src/demo.py" in workspace.list_files()
    assert ".env" not in workspace.list_files()
    assert "1: alpha" in workspace.read_file("src/demo.py")
    assert "src/demo.py:2: beta" in workspace.search_text("BETA")

    workspace.replace_in_file("src/demo.py", "beta", "gamma")
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"

    with pytest.raises(WorkspaceSecurityError):
        workspace.read_file(".env")
    with pytest.raises(WorkspaceSecurityError):
        workspace.read_file("../outside.txt")
    with pytest.raises(WorkspaceSecurityError):
        workspace.run_command("rm -rf .")


def test_read_only_mode_blocks_mutations(tmp_path) -> None:
    workspace = Workspace(tmp_path, read_only=True)

    with pytest.raises(WorkspaceSecurityError):
        workspace.write_file("new.txt", "data")
    with pytest.raises(WorkspaceSecurityError):
        workspace.run_command("python -m compileall .")


def test_agent_executes_function_call_loop(monkeypatch, tmp_path) -> None:
    (tmp_path / "hello.txt").write_text("hello agent\n", encoding="utf-8")
    observed: list[dict] = []

    class FakeResponses:
        def create(self, **kwargs):
            observed.append(kwargs)
            if len(observed) == 1:
                call = SimpleNamespace(
                    type="function_call",
                    name="read_file",
                    arguments='{"path":"hello.txt"}',
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
    )

    assert agent.run("Read hello.txt") == "done"
    assert observed[1]["previous_response_id"] == "response-1"
    tool_output = observed[1]["input"][0]
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call-1"
    assert "hello agent" in tool_output["output"]
