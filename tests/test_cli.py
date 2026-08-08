from __future__ import annotations

import json
from pathlib import Path

from ai_language import cli


class DummyClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, *, temperature: float) -> str:
        return f"model={self.model};temp={temperature};prompt={prompt}"

    def plan(self, prompt: str, *, temperature: float) -> str:
        return "generate planned_service | retries; logging\n"


def test_cli_ask_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "AILanguageClient", DummyClient)

    rc = cli.main(["ask", "hello", "--model", "x", "--temperature", "0.5"])

    assert rc == 0
    assert "model=x;temp=0.5;prompt=hello" in capsys.readouterr().out


def test_cli_generate_writes_all_artifacts(tmp_path: Path, capsys) -> None:
    source = tmp_path / "prog.ailang"
    source.write_text("generate api | auth", encoding="utf-8")
    out = tmp_path / "out.py"
    graph = tmp_path / "graph.json"
    ast = tmp_path / "ast.json"
    ir = tmp_path / "ir.json"

    rc = cli.main(
        [
            "generate",
            str(source),
            "--target",
            "python",
            "--out",
            str(out),
            "--emit-graph",
            str(graph),
            "--emit-ast",
            str(ast),
            "--emit-ir",
            str(ir),
        ]
    )

    assert rc == 0
    assert out.exists() and graph.exists() and ast.exists() and ir.exists()
    assert json.loads(graph.read_text(encoding="utf-8"))["nodes"][0]["id"] == "program"
    assert json.loads(ast.read_text(encoding="utf-8"))["instructions"][0]["action"] == "generate"
    assert json.loads(ir.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "fingerprint" in capsys.readouterr().out


def test_cli_inspect_prints_json(tmp_path: Path, capsys) -> None:
    source = tmp_path / "prog.ailang"
    source.write_text("generate api", encoding="utf-8")

    rc = cli.main(["inspect", str(source)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "python"
    assert "code" not in payload


def test_cli_plan_writes_source_and_code(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "AILanguageClient", DummyClient)
    out = tmp_path / "planned.py"
    source = tmp_path / "planned.ailang"

    rc = cli.main(
        [
            "plan",
            "build service",
            "--out",
            str(out),
            "--emit-source",
            str(source),
        ]
    )

    assert rc == 0
    assert "planned_service" in out.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == (
        "generate planned_service | retries; logging\n"
    )
    assert "Planned and generated" in capsys.readouterr().out


def test_cli_run_executes_generated_python(tmp_path: Path, capsys) -> None:
    generated = tmp_path / "generated.py"
    generated.write_text("print('from-machine')\n", encoding="utf-8")

    rc = cli.main(["run", str(generated), "--timeout", "1"])

    assert rc == 0
    assert "from-machine" in capsys.readouterr().out


def test_cli_returns_2_for_bad_source(tmp_path: Path, capsys) -> None:
    source = tmp_path / "bad.ailang"
    source.write_text("badline", encoding="utf-8")

    rc = cli.main(["inspect", str(source)])

    assert rc == 2
    assert "Line 1" in capsys.readouterr().err
