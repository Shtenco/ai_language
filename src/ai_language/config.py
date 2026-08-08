"""Configuration utilities for model and API-key management."""

from __future__ import annotations

import os

DEFAULT_MODEL = "gpt-4o-mini"


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key can be found."""


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(override=False)


def get_api_key(explicit_key: str | None = None) -> str:
    """Return an API key from an explicit argument or ``OPENAI_API_KEY``."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    _load_dotenv_if_available()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise MissingAPIKeyError(
        "OPENAI_API_KEY is not set. Pass --api-key or define it in the environment/.env."
    )
