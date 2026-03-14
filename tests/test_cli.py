from __future__ import annotations

from ai_language import cli


class DummyClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, *, temperature: float) -> str:
        return f"model={self.model};temp={temperature};prompt={prompt}"


def test_cli_success(monkeypatch, capsys):
    monkeypatch.setattr(cli, "AILanguageClient", DummyClient)

    rc = cli.main(["hello", "--model", "x", "--temperature", "0.5"])

    assert rc == 0
    assert "model=x;temp=0.5;prompt=hello" in capsys.readouterr().out
