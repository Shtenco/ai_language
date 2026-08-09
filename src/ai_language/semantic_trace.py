"""Semantic requirement, repository, impact, and execution graphs for the coding agent."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ir import SemanticEdge, SemanticGraph, SemanticNode

_MAX_REPOSITORY_FILES = 240
_MAX_EVENT_SUMMARY = 500
_MAX_SOURCE_BYTES = 1_000_000
_MUTATION_TOOLS = {"write_file", "replace_in_file", "delete_file"}
_VALIDATION_TOOLS = {"run_command"}


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


@dataclass(slots=True, frozen=True)
class RequirementCoverage:
    """Derived implementation/verification state for one requirement."""

    requirement_id: str
    kind: str
    status: str
    mutation_events: int
    validation_events: int
    targets: tuple[str, ...]


@dataclass(slots=True)
class SemanticTrace:
    """Task intent, repository structure, impact, and requirement-to-action provenance."""

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

    def coverage(self) -> list[RequirementCoverage]:
        """Derive requirement coverage from successful mutation and validation events."""
        result: list[RequirementCoverage] = []
        for requirement in self.requirements:
            linked = [
                event
                for event in self.events
                if requirement.id in event.requirement_ids and event.status == "ok"
            ]
            mutations = [event for event in linked if event.tool in _MUTATION_TOOLS]
            validations = [event for event in linked if event.tool in _VALIDATION_TOOLS]

            if requirement.kind == "validation":
                status = "verified" if validations else "unresolved"
            elif requirement.kind == "constraint":
                status = "verified" if validations else "addressed" if mutations else "unresolved"
            else:
                if mutations and validations:
                    status = "verified"
                elif mutations:
                    status = "implemented"
                else:
                    status = "unresolved"

            targets = tuple(dict.fromkeys(event.target for event in linked if event.target))
            result.append(
                RequirementCoverage(
                    requirement_id=requirement.id,
                    kind=requirement.kind,
                    status=status,
                    mutation_events=len(mutations),
                    validation_events=len(validations),
                    targets=targets,
                )
            )
        return result

    def unresolved_requirement_ids(self) -> list[str]:
        return [item.requirement_id for item in self.coverage() if item.status == "unresolved"]

    def instruction_context(self) -> str:
        """Compact semantic context injected into the agent instructions."""
        requirement_lines = "\n".join(
            f"- {req.id} [{req.kind}]: {req.text}" for req in self.requirements
        )
        structure_nodes = [
            node
            for node in self.repository_graph.nodes
            if node.kind in {"file", "class", "function", "method", "test"}
        ]
        repository_lines = "\n".join(
            f"- [{node.kind}] {node.value}" for node in structure_nodes[:160]
        )
        if len(structure_nodes) > 160:
            repository_lines += f"\n- ... {len(structure_nodes) - 160} more structural nodes"
        return (
            "SEMANTIC TASK MODEL\n"
            "Requirements:\n"
            f"{requirement_lines or '- REQ-1 [action]: ' + self.prompt.strip()}\n\n"
            "Repository structure snapshot:\n"
            f"{repository_lines or '- [empty workspace]'}\n\n"
            "Traceability rule: every write_file, replace_in_file, delete_file, and "
            "run_command call MUST include requirement_ids containing the REQ-* items "
            "that justify the change or validation. Use only IDs listed above. Before "
            "finishing, ensure action requirements have an implementation event and explicit "
            "validation requirements have a successful validation command whenever possible."
        )

    def change_graph(self) -> SemanticGraph:
        """Build requirement -> execution-event -> target provenance graph."""
        graph = SemanticGraph(nodes=[SemanticNode(id="changes", kind="change_graph", value="changes")])
        for requirement in self.requirements:
            req_node = f"requirement:{requirement.id}"
            graph.nodes.append(SemanticNode(id=req_node, kind=requirement.kind, value=requirement.text))
            graph.edges.append(SemanticEdge(source="changes", relation="tracks", target=req_node))

        target_nodes: dict[str, str] = {}
        for event in self.events:
            event_id = f"event:{event.index}"
            graph.nodes.append(
                SemanticNode(id=event_id, kind="event", value=f"{event.tool}:{event.status}")
            )
            for requirement_id in event.requirement_ids:
                graph.edges.append(
                    SemanticEdge(
                        source=f"requirement:{requirement_id}",
                        relation="validated_by" if event.tool in _VALIDATION_TOOLS else "implemented_by",
                        target=event_id,
                    )
                )
            if event.target:
                target_id = target_nodes.setdefault(event.target, f"target:{len(target_nodes) + 1}")
                if not any(node.id == target_id for node in graph.nodes):
                    graph.nodes.append(SemanticNode(id=target_id, kind="target", value=event.target))
                graph.edges.append(SemanticEdge(source=event_id, relation="targets", target=target_id))
        return graph

    def impact_graph(self) -> SemanticGraph:
        """Project changed files onto defined symbols and tests that call them."""
        graph = SemanticGraph(nodes=[SemanticNode(id="impact", kind="impact_graph", value="impact")])
        changed_paths = {
            event.target
            for event in self.events
            if event.status == "ok" and event.tool in _MUTATION_TOOLS and event.target
        }
        if not changed_paths:
            return graph

        repo_nodes = {node.id: node for node in self.repository_graph.nodes}
        file_ids = {
            node.id
            for node in self.repository_graph.nodes
            if node.kind == "file" and node.value in changed_paths
        }
        selected_ids = set(file_ids)
        for edge in self.repository_graph.edges:
            if edge.source in file_ids and edge.relation == "defines":
                selected_ids.add(edge.target)

        changed_symbols = {
            node_id
            for node_id in selected_ids
            if repo_nodes.get(node_id) and repo_nodes[node_id].kind in {"class", "function", "method"}
        }
        for edge in self.repository_graph.edges:
            source = repo_nodes.get(edge.source)
            if edge.relation == "calls" and edge.target in changed_symbols and source and source.kind == "test":
                selected_ids.add(edge.source)

        for node_id in selected_ids:
            node = repo_nodes.get(node_id)
            if node:
                graph.nodes.append(node)
                graph.edges.append(SemanticEdge(source="impact", relation="includes", target=node.id))
        for edge in self.repository_graph.edges:
            if edge.source in selected_ids and edge.target in selected_ids:
                graph.edges.append(edge)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai-language.semantic-trace/v2",
            "prompt": self.prompt,
            "requirements": [asdict(requirement) for requirement in self.requirements],
            "coverage": [
                {**asdict(item), "targets": list(item.targets)} for item in self.coverage()
            ],
            "unresolved_requirement_ids": self.unresolved_requirement_ids(),
            "intent_graph": _graph_to_dict(self.intent_graph),
            "repository_graph": _graph_to_dict(self.repository_graph),
            "change_graph": _graph_to_dict(self.change_graph()),
            "impact_graph": _graph_to_dict(self.impact_graph()),
            "events": [
                {**asdict(event), "requirement_ids": list(event.requirement_ids)}
                for event in self.events
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def render_text(self) -> str:
        """Human-readable trace grouped by requirement."""
        lines = ["Semantic trace"]
        coverage_by_id = {item.requirement_id: item for item in self.coverage()}
        for requirement in self.requirements:
            coverage = coverage_by_id[requirement.id]
            lines.append(
                f"{requirement.id} [{requirement.kind}/{coverage.status}] {requirement.text}"
            )
            linked = [event for event in self.events if requirement.id in event.requirement_ids]
            if not linked:
                lines.append("  - no mapped actions")
                continue
            for event in linked:
                target = f" {event.target}" if event.target else ""
                lines.append(f"  - {event.tool}{target} [{event.status}] — {event.summary}")

        unmapped = [event for event in self.events if not event.requirement_ids]
        if unmapped:
            lines.append("Unmapped tool activity")
            for event in unmapped:
                target = f" {event.target}" if event.target else ""
                lines.append(f"  - {event.tool}{target} [{event.status}]")
        unresolved = self.unresolved_requirement_ids()
        lines.append("Unresolved: " + (", ".join(unresolved) if unresolved else "none"))
        return "\n".join(lines)

    def render_coverage_text(self) -> str:
        """Compact requirement coverage report."""
        coverage = self.coverage()
        resolved = sum(item.status != "unresolved" for item in coverage)
        verified = sum(item.status == "verified" for item in coverage)
        lines = [
            f"Requirement coverage: {resolved}/{len(coverage)} resolved; "
            f"{verified}/{len(coverage)} verified"
        ]
        requirement_map = {requirement.id: requirement for requirement in self.requirements}
        for item in coverage:
            requirement = requirement_map[item.requirement_id]
            targets = f" -> {', '.join(item.targets)}" if item.targets else ""
            lines.append(
                f"- {item.requirement_id} [{item.status}] {requirement.text}{targets}"
            )
        return "\n".join(lines)


def build_semantic_trace(
    prompt: str,
    repository_files: list[str],
    repository_root: Path | str | None = None,
) -> SemanticTrace:
    """Create deterministic intent and repository graphs plus an empty execution trace."""
    requirements = extract_requirements(prompt)
    return SemanticTrace(
        prompt=prompt,
        requirements=requirements,
        intent_graph=build_intent_graph(requirements),
        repository_graph=build_repository_graph(repository_files, repository_root=repository_root),
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
    graph = SemanticGraph(nodes=[SemanticNode(id="task", kind="task", value="coding_task")])
    for requirement in requirements:
        node_id = requirement.id.lower()
        graph.nodes.append(SemanticNode(id=node_id, kind=requirement.kind, value=requirement.text))
        graph.edges.append(SemanticEdge(source="task", relation="contains", target=node_id))
    return graph


def build_repository_graph(
    repository_files: list[str],
    repository_root: Path | str | None = None,
) -> SemanticGraph:
    """Build a bounded file/symbol/import/call graph; Python parsing is best-effort."""
    graph = SemanticGraph(
        nodes=[SemanticNode(id="repository", kind="repository", value="workspace")]
    )
    root = Path(repository_root).resolve() if repository_root is not None else None
    seen: set[str] = set()
    file_nodes: dict[str, str] = {}
    parsed: dict[str, ast.Module] = {}
    symbol_by_name: dict[str, list[str]] = {}

    for raw_path in repository_files:
        path = raw_path.strip()
        if not path or path.startswith("...") or path.startswith("[") or path in seen:
            continue
        seen.add(path)
        if len(seen) > _MAX_REPOSITORY_FILES:
            break
        node_id = f"file:{len(seen)}"
        file_nodes[path] = node_id
        graph.nodes.append(SemanticNode(id=node_id, kind="file", value=path))
        graph.edges.append(SemanticEdge(source="repository", relation="contains", target=node_id))

        if root is None or not path.endswith(".py"):
            continue
        source_path = (root / path).resolve()
        try:
            source_path.relative_to(root)
            if not source_path.is_file() or source_path.stat().st_size > _MAX_SOURCE_BYTES:
                continue
            tree = ast.parse(source_path.read_text(encoding="utf-8", errors="strict"), filename=path)
        except (OSError, UnicodeError, SyntaxError, ValueError):
            continue
        parsed[path] = tree

    module_nodes: dict[str, str] = {}
    symbol_nodes_by_path: dict[str, list[str]] = {}

    for path, tree in parsed.items():
        file_id = file_nodes[path]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for module_name in names:
                module_id = module_nodes.setdefault(module_name, f"module:{module_name}")
                if not any(existing.id == module_id for existing in graph.nodes):
                    graph.nodes.append(SemanticNode(id=module_id, kind="module", value=module_name))
                graph.edges.append(SemanticEdge(source=file_id, relation="imports", target=module_id))

        for qualname, node, parent_kind in _iter_symbols(tree):
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif qualname.split(".")[-1].startswith("test_") or path.startswith("tests/"):
                kind = "test"
            elif parent_kind == "class":
                kind = "method"
            else:
                kind = "function"
            symbol_id = f"symbol:{path}:{qualname}"
            graph.nodes.append(SemanticNode(id=symbol_id, kind=kind, value=f"{path}::{qualname}"))
            graph.edges.append(SemanticEdge(source=file_id, relation="defines", target=symbol_id))
            symbol_nodes_by_path.setdefault(path, []).append(symbol_id)
            symbol_by_name.setdefault(qualname.split(".")[-1], []).append(symbol_id)

    external_calls: dict[str, str] = {}
    for path, tree in parsed.items():
        file_id = file_nodes[path]
        for qualname, node, _parent_kind in _iter_symbols(tree):
            if isinstance(node, ast.ClassDef):
                continue
            source_id = f"symbol:{path}:{qualname}"
            for call_name in _call_names(node):
                candidates = symbol_by_name.get(call_name, [])
                if len(candidates) == 1:
                    target_id = candidates[0]
                else:
                    target_id = external_calls.setdefault(call_name, f"call:{call_name}")
                    if not any(existing.id == target_id for existing in graph.nodes):
                        graph.nodes.append(SemanticNode(id=target_id, kind="call", value=call_name))
                graph.edges.append(SemanticEdge(source=source_id, relation="calls", target=target_id))

        module_level_calls = _module_level_call_names(tree)
        for call_name in module_level_calls:
            candidates = symbol_by_name.get(call_name, [])
            if len(candidates) == 1:
                target_id = candidates[0]
            else:
                target_id = external_calls.setdefault(call_name, f"call:{call_name}")
                if not any(existing.id == target_id for existing in graph.nodes):
                    graph.nodes.append(SemanticNode(id=target_id, kind="call", value=call_name))
            graph.edges.append(SemanticEdge(source=file_id, relation="calls", target=target_id))

    return _dedupe_graph(graph)


def _iter_symbols(tree: ast.Module) -> list[tuple[str, ast.AST, str | None]]:
    result: list[tuple[str, ast.AST, str | None]] = []

    def visit(body: list[ast.stmt], prefix: str = "", parent_kind: str | None = None) -> None:
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualname = f"{prefix}.{node.name}" if prefix else node.name
            result.append((qualname, node, parent_kind))
            child_kind = "class" if isinstance(node, ast.ClassDef) else "function"
            visit(node.body, qualname, child_kind)

    visit(tree.body)
    return result


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _module_level_call_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names.update(_call_names(node))
    return names


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


def _dedupe_graph(graph: SemanticGraph) -> SemanticGraph:
    nodes: list[SemanticNode] = []
    node_ids: set[str] = set()
    for node in graph.nodes:
        if node.id not in node_ids:
            node_ids.add(node.id)
            nodes.append(node)
    edges: list[SemanticEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        key = (edge.source, edge.relation, edge.target)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(edge)
    return SemanticGraph(nodes=nodes, edges=edges)


def _graph_to_dict(graph: SemanticGraph) -> dict[str, list[dict[str, str]]]:
    return {
        "nodes": [asdict(node) for node in graph.nodes],
        "edges": [asdict(edge) for edge in graph.edges],
    }
