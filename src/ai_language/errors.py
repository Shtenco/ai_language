"""Domain-specific exceptions for AI Language Pro."""

from __future__ import annotations


class AILanguageSyntaxError(ValueError):
    """Raised when an .ailang source line does not match the language grammar."""

    def __init__(self, line_number: int, line: str, reason: str) -> None:
        self.line_number = line_number
        self.line = line
        self.reason = reason
        super().__init__(f"Line {line_number}: {reason}: {line!r}")
