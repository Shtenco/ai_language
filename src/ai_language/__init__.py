"""AI Language Pro package."""

from .agent import CodingAgent, Workspace
from .pipeline import PipelineResult, compile_source

__all__ = [
    "CodingAgent",
    "PipelineResult",
    "Workspace",
    "compile_source",
    "__version__",
]

__version__ = "0.4.0"
