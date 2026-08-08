"""Intermediate representations used by AI Language Pro."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Instruction:
    """A normalized high-level instruction parsed from source text."""

    action: str
    target: str
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticNode:
    """Node in the semantic graph."""

    id: str
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    """Directed semantic relation between two graph nodes."""

    source: str
    relation: str
    target: str


@dataclass(slots=True)
class SemanticGraph:
    """Deterministic directed graph describing program semantics."""

    nodes: list[SemanticNode] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProgramAST:
    """Root AST for a normalized AI Language program."""

    instructions: tuple[Instruction, ...]
