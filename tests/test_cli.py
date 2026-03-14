from __future__ import annotations

from pathlib import Path

from ai_language import cli


class DummyClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, *, temperature: float) -> str:
        return f"model={self.model};temp={temperature};prompt={prompt}"


def test_cli_ask_success(monkeypatch, capsys):
    monkeypatch.setattr(cli, "AILanguageClient", DummyClient)

    rc = cli.main(["ask", "hello", "--model", "x", "--temperature", "0.5"])

    assert rc == 0
    assert "model=x;temp=0.5;prompt=hello" in capsys.readouterr().out


def test_cli_generate_writes_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "prog.ailang"
    source.write_text("generate api", encoding="utf-8")
    out = tmp_path / "out.py"

    rc = cli.main(["generate", str(source), "--target", "python", "--out", str(out)])

    assert rc == 0
    assert out.exists()
    assert "Generated python code" in capsys.readouterr().out
