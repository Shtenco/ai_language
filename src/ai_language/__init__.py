"""Public SDK surface for AI Language Pro."""

from .errors import AILanguageSyntaxError
from .pipeline import (
    IR_SCHEMA_VERSION,
    SUPPORTED_TARGETS,
    PipelineResult,
    canonical_source,
    compile_source,
    parse_instructions,
)

__all__ = [
    "AILanguageSyntaxError",
    "IR_SCHEMA_VERSION",
    "PipelineResult",
    "SUPPORTED_TARGETS",
    "canonical_source",
    "compile_source",
    "parse_instructions",
    "__version__",
]

__version__ = "0.4.0"
