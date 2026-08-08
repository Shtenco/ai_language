"""Deterministic language pipeline: DSL -> semantic graph -> AST -> target scaffold."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .errors import AILanguageSyntaxError
from .ir import Instruction, ProgramAST, SemanticEdge, SemanticGraph, SemanticNode

IR_SCHEMA_VERSION = 1
SUPPORTED_TARGETS = frozenset({"python", "c", "rust", "solidity", "kotlin"})


@dataclass(slots=True)
class PipelineResult:
    """Immutable-by-convention artifacts produced by one compilation."""

    target: str
    instructions: tuple[Instruction, ...]
    semantic_graph: SemanticGraph
    ast: ProgramAST
    fingerprint: str
    code: str

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable, versioned representation of the compilation."""
        payload: dict[str, Any] = {
            "schema_version": IR_SCHEMA_VERSION,
            "target": self.target,
            "fingerprint": self.fingerprint,
            "instructions": [asdict(item) for item in self.instructions],
            "semantic_graph": {
                "nodes": [asdict(node) for node in self.semantic_graph.nodes],
                "edges": [asdict(edge) for edge in self.semantic_graph.edges],
            },
            "ast": {"instructions": [asdict(item) for item in self.ast.instructions]},
        }
        if include_code:
            payload["code"] = self.code
        return payload


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _validate_line_characters(line: str, line_number: int) -> None:
    bad = [char for char in line if ord(char) < 32 and char != "\t"]
    if bad:
        raise AILanguageSyntaxError(
            line_number,
            line,
            "control characters are not allowed",
        )


def parse_instructions(source: str) -> tuple[Instruction, ...]:
    """Parse .ailang source into normalized instructions.

    Grammar::

        ACTION TARGET [| constraint1; constraint2; ...]

    Blank lines and lines beginning with ``#`` are ignored. The parser deliberately
    stays small and deterministic: no hidden LLM call happens during parsing.
    """
    instructions: list[Instruction] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        _validate_line_characters(line, line_number)
        left, separator, right = line.partition("|")
        parts = left.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise AILanguageSyntaxError(
                line_number,
                line,
                "expected ACTION TARGET",
            )

        action, target = parts
        action = action.strip().lower()
        target = _normalize_space(target)
        if not action:
            raise AILanguageSyntaxError(line_number, line, "action cannot be empty")
        if not target:
            raise AILanguageSyntaxError(line_number, line, "target cannot be empty")

        constraints: tuple[str, ...] = ()
        if separator:
            normalized = tuple(
                _normalize_space(item) for item in right.split(";") if _normalize_space(item)
            )
            constraints = normalized

        instructions.append(Instruction(action=action, target=target, constraints=constraints))

    if not instructions:
        raise ValueError("No valid instructions found in source.")
    return tuple(instructions)


def render_instruction(instruction: Instruction) -> str:
    """Render one instruction into canonical source form."""
    base = f"{instruction.action} {instruction.target}"
    if instruction.constraints:
        return f"{base} | {'; '.join(instruction.constraints)}"
    return base


def canonical_source(instructions: tuple[Instruction, ...]) -> str:
    """Return canonical source used for reproducible fingerprints."""
    return "\n".join(render_instruction(item) for item in instructions) + "\n"


def fingerprint_instructions(instructions: tuple[Instruction, ...]) -> str:
    """Return SHA-256 of the normalized semantic source."""
    normalized = canonical_source(instructions).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_semantic_graph(instructions: tuple[Instruction, ...]) -> SemanticGraph:
    """Create a stable semantic graph from normalized instructions."""
    graph = SemanticGraph()
    graph.nodes.append(SemanticNode(id="program", kind="root", value="AIProgram"))

    for idx, instr in enumerate(instructions, start=1):
        node_id = f"instr_{idx}"
        graph.nodes.append(
            SemanticNode(
                id=node_id,
                kind="instruction",
                value=f"{instr.action}:{instr.target}",
            )
        )
        graph.edges.append(SemanticEdge(source="program", relation="contains", target=node_id))

        for constraint_idx, constraint in enumerate(instr.constraints, start=1):
            constraint_id = f"{node_id}_constraint_{constraint_idx}"
            graph.nodes.append(SemanticNode(id=constraint_id, kind="constraint", value=constraint))
            graph.edges.append(
                SemanticEdge(
                    source=node_id,
                    relation="constrained_by",
                    target=constraint_id,
                )
            )

    return graph


def build_ast(instructions: tuple[Instruction, ...]) -> ProgramAST:
    """Build the deterministic AST."""
    return ProgramAST(instructions=instructions)


def _escape_c_like(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _escape_kotlin(value: str) -> str:
    return _escape_c_like(value).replace("$", "\\$")


def _generated_lines(ast: ProgramAST) -> list[str]:
    return [render_instruction(item) for item in ast.instructions]


def _generate_python(ast: ProgramAST, fingerprint: str) -> str:
    lines = [
        '"""Generated by AI Language Pro.',
        f"Semantic fingerprint: {fingerprint}",
        '"""',
        "",
        "def run_program() -> list[str]:",
        "    return [",
    ]
    for text in _generated_lines(ast):
        lines.append(f"        {text!r},")
    lines.extend(
        [
            "    ]",
            "",
            "",
            "if __name__ == '__main__':",
            "    for step in run_program():",
            "        print(step)",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_c(ast: ProgramAST, fingerprint: str) -> str:
    lines = [
        "/* Generated by AI Language Pro.",
        f" * Semantic fingerprint: {fingerprint}",
        " */",
        "#include <stdio.h>",
        "",
        "int main(void) {",
    ]
    for text in _generated_lines(ast):
        lines.append(f'    puts("{_escape_c_like(text)}");')
    lines.extend(["    return 0;", "}"])
    return "\n".join(lines) + "\n"


def _generate_rust(ast: ProgramAST, fingerprint: str) -> str:
    lines = [
        "// Generated by AI Language Pro.",
        f"// Semantic fingerprint: {fingerprint}",
        "fn main() {",
    ]
    for text in _generated_lines(ast):
        lines.append(f'    println!("{{}}", "{_escape_c_like(text)}");')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _generate_solidity(ast: ProgramAST, fingerprint: str) -> str:
    lines = [
        "// SPDX-License-Identifier: MIT",
        "// Generated by AI Language Pro.",
        f"// Semantic fingerprint: {fingerprint}",
        "pragma solidity ^0.8.20;",
        "",
        "contract AIProgram {",
        "    function steps() external pure returns (string[] memory out) {",
        f"        out = new string[]({len(ast.instructions)});",
    ]
    for idx, text in enumerate(_generated_lines(ast)):
        lines.append(f'        out[{idx}] = unicode"{_escape_c_like(text)}";')
    lines.extend(["    }", "}"])
    return "\n".join(lines) + "\n"


def _generate_kotlin(ast: ProgramAST, fingerprint: str) -> str:
    lines = [
        "// Generated by AI Language Pro.",
        f"// Semantic fingerprint: {fingerprint}",
        "fun main() {",
    ]
    for text in _generated_lines(ast):
        lines.append(f'    println("{_escape_kotlin(text)}")')
    lines.append("}")
    return "\n".join(lines) + "\n"


Generator = Callable[[ProgramAST, str], str]
_GENERATORS: dict[str, Generator] = {
    "python": _generate_python,
    "c": _generate_c,
    "rust": _generate_rust,
    "solidity": _generate_solidity,
    "kotlin": _generate_kotlin,
}


def generate_code(ast: ProgramAST, target: str, *, fingerprint: str | None = None) -> str:
    """Generate a deterministic executable scaffold for a target language."""
    target_normalized = target.lower().strip()
    if target_normalized not in SUPPORTED_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TARGETS))
        raise ValueError(f"Unsupported target language: {target!r}. Supported: {supported}")

    semantic_fingerprint = fingerprint or fingerprint_instructions(ast.instructions)
    return _GENERATORS[target_normalized](ast, semantic_fingerprint)


def compile_source(source: str, target: str = "python") -> PipelineResult:
    """Run the complete deterministic compilation pipeline."""
    target_normalized = target.lower().strip()
    instructions = parse_instructions(source)
    semantic_graph = build_semantic_graph(instructions)
    ast = build_ast(instructions)
    fingerprint = fingerprint_instructions(instructions)
    code = generate_code(ast, target=target_normalized, fingerprint=fingerprint)
    return PipelineResult(
        target=target_normalized,
        instructions=instructions,
        semantic_graph=semantic_graph,
        ast=ast,
        fingerprint=fingerprint,
        code=code,
    )
