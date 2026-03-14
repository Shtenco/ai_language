"""Thin wrapper around the OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_MODEL, get_api_key


@dataclass
class AILanguageClient:
    """AI language client with simple text generation interface."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        from openai import OpenAI  # Lazy import for better offline/dev ergonomics.

        resolved_api_key = get_api_key(self.api_key)
        self._client = OpenAI(api_key=resolved_api_key)

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        """Generate text from the provided prompt."""
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
        )
        return response.output_text.strip()
