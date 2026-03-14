from __future__ import annotations

import pytest

from ai_language.config import MissingAPIKeyError, get_api_key


def test_get_api_key_from_explicit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_api_key(" sk-test ") == "sk-test"


def test_get_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert get_api_key() == "env-key"


def test_get_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        get_api_key()
