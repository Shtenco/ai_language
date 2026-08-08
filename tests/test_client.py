from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ai_language.client import AILanguageClient


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAI:
    next_output = "hello"
    last_instance: FakeOpenAI | None = None

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.responses = FakeResponses(self.next_output)
        type(self).last_instance = self


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    FakeOpenAI.next_output = output
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))


def test_generate_uses_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch, " model output ")
    client = AILanguageClient(api_key="test-key", model="test-model")

    output = client.generate("hello", instructions="be concise", temperature=0.4)

    assert output == "model output"
    assert FakeOpenAI.last_instance is not None
    call = FakeOpenAI.last_instance.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["input"] == "hello"
    assert call["instructions"] == "be concise"
    assert call["temperature"] == 0.4


def test_plan_strips_fence_validates_and_canonicalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch, "```text\nGENERATE   api | auth ; retries\n```")
    client = AILanguageClient(api_key="test-key")

    assert client.plan("build api") == "generate api | auth; retries\n"


def test_generate_validates_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch, "output")
    client = AILanguageClient(api_key="test-key")

    with pytest.raises(ValueError, match="between 0 and 2"):
        client.generate("hello", temperature=3)
