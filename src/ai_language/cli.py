"""Command-line interface for AI Language Pro."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .client import AILanguageClient
from .compiler import ExecutionError, compile_python_file, execute_python_file
from .config import DEFAULT_MODEL, MissingAPIKeyError
from .errors import AILanguageSyntaxError
from .pipeline import IR_SCHEMA_VERSION, SUPPORTED_TARGETS, PipelineResult, compile_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-language",
        description=("Deterministic AI Language DSL compiler with optional LLM planning runtime."),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Compile an .ailang source file.")
    gen.add_argument("source", help="Path to .ailang source file.")
    gen.add_argument(
        "--target",
        default="python",
        choices=sorted(SUPPORTED_TARGETS),
        help="Target scaffold language.",
    )
    gen.add_argument("--out", required=True, help="Output path for generated code.")
    gen.add_argument("--emit-graph", help="Optional JSON path for semantic graph.")
    gen.add_argument("--emit-ast", help="Optional JSON path for AST.")
    gen.add_argument("--emit-ir", help="Optional JSON path for complete versioned IR.")

    inspect_cmd = subparsers.add_parser(
        "inspect", help="Print versioned compilation IR as JSON without writing code."
    )
    inspect_cmd.add_argument("source", help="Path to .ailang source file.")
    inspect_cmd.add_argument(
        "--target",
        default="python",
        choices=sorted(SUPPORTED_TARGETS),
    )
    inspect_cmd.add_argument(
        "--include-code",
        action="store_true",
        help="Include generated target code in JSON output.",
    )

    plan = subparsers.add_parser(
        "plan",
        help="Translate natural-language intent into validated .ailang and compile it.",
    )
    plan.add_argument("prompt", help="Natural-language software intent.")
    plan.add_argument("--api-key", dest="api_key", help="User-provided OpenAI API key.")
    plan.add_argument("--model", default=DEFAULT_MODEL, help="Responses API model name.")
    plan.add_argument("--temperature", type=float, default=0.0)
    plan.add_argument(
        "--target",
        default="python",
        choices=sorted(SUPPORTED_TARGETS),
    )
    plan.add_argument("--out", required=True, help="Output path for generated code.")
    plan.add_argument("--emit-source", help="Optional path for canonical generated .ailang.")
    plan.add_argument("--emit-ir", help="Optional path for complete versioned IR JSON.")

    check = subparsers.add_parser("check", help="Compile/validate a generated Python file.")
    check.add_argument("file", help="Path to generated Python file.")

    run = subparsers.add_parser("run", help="Validate and execute a generated Python file.")
    run.add_argument("file", help="Path to generated Python file.")
    run.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Wall-clock execution timeout in seconds (default: 5).",
    )

    ask = subparsers.add_parser("ask", help="Send a free-form prompt to the model runtime.")
    ask.add_argument("prompt", help="Prompt text to send to the model.")
    ask.add_argument("--api-key", dest="api_key", help="User-provided OpenAI API key.")
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Responses API model name.")
    ask.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")

    return parser


def _write_text(path: str | Path, content: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    return _write_text(path, content)


def _graph_payload(result: PipelineResult) -> dict[str, Any]:
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "fingerprint": result.fingerprint,
        "nodes": [asdict(node) for node in result.semantic_graph.nodes],
        "edges": [asdict(edge) for edge in result.semantic_graph.edges],
    }


def _ast_payload(result: PipelineResult) -> dict[str, Any]:
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "fingerprint": result.fingerprint,
        "instructions": [asdict(item) for item in result.ast.instructions],
    }


def _write_compilation(
    result: PipelineResult,
    *,
    out: str,
    emit_graph: str | None = None,
    emit_ast: str | None = None,
    emit_ir: str | None = None,
) -> Path:
    output = _write_text(out, result.code)
    if emit_graph:
        _write_json(emit_graph, _graph_payload(result))
    if emit_ast:
        _write_json(emit_ast, _ast_payload(result))
    if emit_ir:
        _write_json(emit_ir, result.to_dict(include_code=True))
    return output


def _run_generate(args: argparse.Namespace) -> int:
    source_text = Path(args.source).read_text(encoding="utf-8")
    result = compile_source(source_text, target=args.target)
    output = _write_compilation(
        result,
        out=args.out,
        emit_graph=args.emit_graph,
        emit_ast=args.emit_ast,
        emit_ir=args.emit_ir,
    )
    print(f"Generated {result.target} -> {output} (fingerprint {result.fingerprint[:12]})")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    source_text = Path(args.source).read_text(encoding="utf-8")
    result = compile_source(source_text, target=args.target)
    print(
        json.dumps(
            result.to_dict(include_code=args.include_code),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    client = AILanguageClient(api_key=args.api_key, model=args.model)
    source = client.plan(args.prompt, temperature=args.temperature)
    result = compile_source(source, target=args.target)
    output = _write_compilation(result, out=args.out, emit_ir=args.emit_ir)
    if args.emit_source:
        _write_text(args.emit_source, source)
    print(
        f"Planned and generated {result.target} -> {output} (fingerprint {result.fingerprint[:12]})"
    )
    return 0


def _run_check(args: argparse.Namespace) -> int:
    compiled_path = compile_python_file(args.file)
    print(f"Python bytecode compiled -> {compiled_path}")
    return 0


def _run_machine(args: argparse.Namespace) -> int:
    compile_python_file(args.file)
    output = execute_python_file(args.file, timeout=args.timeout)
    if output:
        print(output)
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    client = AILanguageClient(api_key=args.api_key, model=args.model)
    output = client.generate(args.prompt, temperature=args.temperature)
    print(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        handlers = {
            "generate": _run_generate,
            "inspect": _run_inspect,
            "plan": _run_plan,
            "check": _run_check,
            "run": _run_machine,
            "ask": _run_ask,
        }
        return handlers[args.command](args)
    except (
        MissingAPIKeyError,
        ExecutionError,
        AILanguageSyntaxError,
        FileNotFoundError,
        IsADirectoryError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - last-resort CLI boundary
        print(f"Unexpected runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
