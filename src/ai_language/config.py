"""Configuration utilities for API key management."""

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
    """Return API key from argument or environment.

    Lookup order:
    1. Explicit argument
    2. OPENAI_API_KEY from environment / .env
    """
    if explicit_key:
        return explicit_key.strip()

    _load_dotenv_if_available()
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise MissingAPIKeyError(
        "OPENAI_API_KEY is not set. Pass --api-key or create a .env file with OPENAI_API_KEY=..."
    )
