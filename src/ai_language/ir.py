"""Intermediate representations for AI Language Pro."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Instruction:
    """High-level instruction parsed from source text."""

    action: str
    target: str
    constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticNode:
    """Node in semantic graph representation."""

    id: str
    kind: str
    value: str


@dataclass(slots=True)
class SemanticEdge:
    """Edge in semantic graph representation."""

    source: str
    relation: str
    target: str


@dataclass(slots=True)
class SemanticGraph:
    """Simple directed graph for program semantics."""

    nodes: list[SemanticNode] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)


@dataclass(slots=True)
class ProgramAST:
    """Root AST for AI Language Pro."""

    instructions: list[Instruction]
