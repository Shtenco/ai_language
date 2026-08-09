"""Configuration utilities for API key management."""

from __future__ import annotations

import os

DEFAULT_MODEL = "gpt-5.6"


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key can be found."""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(override=False)


def get_api_key(explicit_key: str | None = None) -> str:
    """Return API key from an explicit argument or OPENAI_API_KEY."""
    if explicit_key:
        return explicit_key.strip()

    _load_dotenv_if_available()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise MissingAPIKeyError(
        "OPENAI_API_KEY is not set. Pass --api-key or set OPENAI_API_KEY in your environment."
    )
