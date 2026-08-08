"""OpenAI Responses API adapter used by optional natural-language planning commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_MODEL, get_api_key
from .pipeline import canonical_source, parse_instructions

PLANNER_INSTRUCTIONS = """\
Translate the user's software intent into AI Language DSL only.
Return one instruction per line using exactly this grammar:
ACTION TARGET | constraint1; constraint2
The | constraints section is optional.
Use concise lowercase English action verbs. Do not use Markdown, code fences, prose, bullets,
or explanations. Preserve important requirements as constraints. Produce at least one line.
"""


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


@dataclass(slots=True)
class AILanguageClient:
    """Small injectable client around the OpenAI Responses API."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI  # Lazy import keeps deterministic core usable offline.

        self._client = OpenAI(api_key=get_api_key(self.api_key))

    def generate(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        """Generate text with the configured Responses API model."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens

        response = self._client.responses.create(**kwargs)
        output = response.output_text.strip()
        if not output:
            raise RuntimeError("Model returned an empty response.")
        return output

    def plan(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Translate natural language into validated, canonical .ailang source."""
        raw = self.generate(
            prompt,
            instructions=PLANNER_INSTRUCTIONS,
            temperature=temperature,
            max_output_tokens=1200,
        )
        source = _strip_code_fence(raw)
        return canonical_source(parse_instructions(source))
