"""Semantic requirement graph and execution trace for the coding agent."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .ir import SemanticEdge, SemanticGraph, SemanticNode

_MAX_REPOSITORY_FILES = 240
_MAX_EVENT_SUMMARY = 500


@dataclass(slots=True, frozen=True)
class Requirement:
    """One normalized requirement extracted from a natural-language task."""

    id: str
    text: str
    kind: str


@dataclass(slots=True, frozen=True)
class TraceEvent:
    """One repository action linked back to one or more requirements."""

    index: int
    tool: str
    target: str
    requirement_ids: tuple[str, ...]
    status: str
    summary: str


@dataclass(slots=True)
class SemanticTrace:
    """Task intent, repository snapshot, and requirement-to-action provenance."""

    prompt: str
    requirements: list[Requirement]
    intent_graph: SemanticGraph
    repository_graph: SemanticGraph
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def requirement_ids(self) -> set[str]:
        return {requirement.id for requirement in self.requirements}

    def record(self, tool: str, arguments: dict[str, Any], result: str) -> None:
        """Record a tool action without persisting file contents or raw credentials."""
        raw_ids = arguments.get("requirement_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = []
        requirement_ids = tuple(
            value
            for value in raw_ids
            if isinstance(value, str) and value in self.requirement_ids
        )
        target = _event_target(tool, arguments)
        first_line = result.strip().splitlines()[0] if result.strip() else "[empty]"
        status = "error" if first_line.startswith("ERROR:") else "ok"
        self.events.append(
            TraceEvent(
                index=len(self.events) + 1,
                tool=tool,
                target=target,
                requirement_ids=requirement_ids,
                status=status,
                summary=first_line[:_MAX_EVENT_SUMMARY],
            )
        )

    def instruction_context(self) -> str:
        """Compact semantic context injected into the agent instructions."""
        requirement_lines = "\n".join(
            f"- {req.id} [{req.kind}]: {req.text}" for req in self.requirements
        )
        files = [
            node.value
            for node in self.repository_graph.nodes
            if node.kind == "file"
        ]
        repository_lines = "\n".join(f"- {path}" for path in files[:120])
        if len(files) > 120:
            repository_lines += f"\n- ... {len(files) - 120} more files in snapshot"
        return (
            "SEMANTIC TASK MODEL\n"
            "Requirements:\n"
            f"{requirement_lines or '- REQ-1 [action]: ' + self.prompt.strip()}\n\n"
            "Repository snapshot:\n"
            f"{repository_lines or '- [empty workspace]'}\n\n"
            "Traceability rule: every write_file, replace_in_file, delete_file, and "
            "run_command call MUST include requirement_ids containing the REQ-* items "
            "that justify the change or validation. Use only IDs listed above."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai-language.semantic-trace/v1",
            "prompt": self.prompt,
            "requirements": [asdict(requirement) for requirement in self.requirements],
            "intent_graph": _graph_to_dict(self.intent_graph),
            "repository_graph": _graph_to_dict(self.repository_graph),
            "events": [
                {
                    **asdict(event),
                    "requirement_ids": list(event.requirement_ids),
                }
                for event in self.events
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def render_text(self) -> str:
        """Human-readable trace grouped by requirement."""
        lines = ["Semantic trace"]
        for requirement in self.requirements:
            lines.append(f"{requirement.id} [{requirement.kind}] {requirement.text}")
            linked = [
                event for event in self.events if requirement.id in event.requirement_ids
            ]
            if not linked:
                lines.append("  - no mapped actions")
                continue
            for event in linked:
                target = f" {event.target}" if event.target else ""
                lines.append(
                    f"  - {event.tool}{target} [{event.status}] — {event.summary}"
                )

        unmapped = [event for event in self.events if not event.requirement_ids]
        if unmapped:
            lines.append("Unmapped tool activity")
            for event in unmapped:
                target = f" {event.target}" if event.target else ""
                lines.append(f"  - {event.tool}{target} [{event.status}]")
        return "\n".join(lines)


def build_semantic_trace(prompt: str, repository_files: list[str]) -> SemanticTrace:
    """Create a deterministic intent graph plus a bounded repository graph snapshot."""
    requirements = extract_requirements(prompt)
    return SemanticTrace(
        prompt=prompt,
        requirements=requirements,
        intent_graph=build_intent_graph(requirements),
        repository_graph=build_repository_graph(repository_files),
    )


def extract_requirements(prompt: str) -> list[Requirement]:
    """Split a task into stable requirement IDs without requiring an extra model call."""
    clauses = _split_prompt(prompt)
    if not clauses:
        clauses = [prompt.strip() or "Complete the requested task"]
    return [
        Requirement(id=f"REQ-{index}", text=clause, kind=_classify_requirement(clause))
        for index, clause in enumerate(clauses, start=1)
    ]


def build_intent_graph(requirements: list[Requirement]) -> SemanticGraph:
    graph = SemanticGraph(
        nodes=[SemanticNode(id="task", kind="task", value="coding_task")]
    )
    for requirement in requirements:
        node_id = requirement.id.lower()
        graph.nodes.append(
            SemanticNode(id=node_id, kind=requirement.kind, value=requirement.text)
        )
        graph.edges.append(
            SemanticEdge(source="task", relation="contains", target=node_id)
        )
    return graph


def build_repository_graph(repository_files: list[str]) -> SemanticGraph:
    graph = SemanticGraph(
        nodes=[SemanticNode(id="repository", kind="repository", value="workspace")]
    )
    seen: set[str] = set()
    for raw_path in repository_files:
        path = raw_path.strip()
        if not path or path.startswith("...") or path.startswith("[") or path in seen:
            continue
        seen.add(path)
        if len(seen) > _MAX_REPOSITORY_FILES:
            break
        node_id = f"file:{len(seen)}"
        graph.nodes.append(SemanticNode(id=node_id, kind="file", value=path))
        graph.edges.append(
            SemanticEdge(source="repository", relation="contains", target=node_id)
        )
    return graph


def _split_prompt(prompt: str) -> list[str]:
    normalized = re.sub(r"^[\s>*-]+", "", prompt.strip(), flags=re.MULTILINE)
    chunks = re.split(r"\n+|(?<=[.!?;])\s+", normalized)
    clauses: list[str] = []
    conjunction = re.compile(
        r",\s+(?=(?:но\b|but\b|и\s+(?:добав|сохран|пров|запуст|обнов)|"
        r"and\s+(?:add|preserve|keep|check|run|update)\b))",
        flags=re.IGNORECASE,
    )
    for chunk in chunks:
        for part in conjunction.split(chunk):
            text = re.sub(r"\s+", " ", part).strip(" -\t,.;")
            if text and text not in clauses:
                clauses.append(text)
    return clauses[:24]


def _classify_requirement(text: str) -> str:
    lowered = text.casefold()
    validation_markers = (
        "test",
        "pytest",
        "проверь",
        "провер",
        "тест",
        "lint",
        "ruff",
        "validate",
        "validation",
    )
    constraint_markers = (
        "не лом",
        "не меня",
        "не удал",
        "без ",
        "сохрани",
        "совместим",
        "preserve",
        "keep ",
        "without ",
        "must not",
        "backward",
        "compatible",
    )
    if any(marker in lowered for marker in validation_markers):
        return "validation"
    if any(marker in lowered for marker in constraint_markers):
        return "constraint"
    return "action"


def _event_target(tool: str, arguments: dict[str, Any]) -> str:
    if tool in {"write_file", "replace_in_file", "delete_file", "read_file"}:
        value = arguments.get("path", "")
    elif tool == "run_command":
        value = arguments.get("command", "")
    elif tool == "search_text":
        value = arguments.get("query", "")
    else:
        value = ""
    return str(value)[:300]


def _graph_to_dict(graph: SemanticGraph) -> dict[str, list[dict[str, str]]]:
    return {
        "nodes": [asdict(node) for node in graph.nodes],
        "edges": [asdict(edge) for edge in graph.edges],
    }
