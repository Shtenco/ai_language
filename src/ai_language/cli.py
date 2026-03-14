"""Command-line interface for ai-language."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import AILanguageClient
from .compiler import compile_python_file
from .config import DEFAULT_MODEL, MissingAPIKeyError
from .pipeline import SUPPORTED_TARGETS, compile_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-language",
        description="AI programming language toolkit: compile instructions into code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate code from instruction source file.")
    gen.add_argument("source", help="Path to .ailang instruction file.")
    gen.add_argument(
        "--target",
        default="python",
        choices=sorted(SUPPORTED_TARGETS),
        help="Target language.",
    )
    gen.add_argument("--out", required=True, help="Output file path for generated code.")
    gen.add_argument(
        "--emit-graph",
        help="Optional JSON output path for semantic graph artifact.",
    )

    check = subparsers.add_parser("check", help="Compile/validate generated Python file.")
    check.add_argument("file", help="Path to generated Python file.")

    ask = subparsers.add_parser("ask", help="Send natural-language prompt to model runtime.")
    ask.add_argument("prompt", help="Prompt text to send to the model.")
    ask.add_argument("--api-key", dest="api_key", help="User-provided API key.")
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    ask.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")

    return parser


def _run_generate(args: argparse.Namespace) -> int:
    source_text = Path(args.source).read_text(encoding="utf-8")
    result = compile_source(source_text, target=args.target)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.code, encoding="utf-8")

    if args.emit_graph:
        graph_payload = {
            "nodes": [node.__dict__ for node in result.semantic_graph.nodes],
            "edges": [edge.__dict__ for edge in result.semantic_graph.edges],
        }
        graph_path = Path(args.emit_graph)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(graph_payload, indent=2, ensure_ascii=False)
        graph_path.write_text(payload, encoding="utf-8")

    print(f"Generated {args.target} code -> {out_path}")
    return 0


def _run_check(args: argparse.Namespace) -> int:
    compiled_path = compile_python_file(args.file)
    print(f"Python bytecode compiled -> {compiled_path}")
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    client = AILanguageClient(api_key=args.api_key, model=args.model)
    output = client.generate(args.prompt, temperature=args.temperature)
    print(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "check":
            return _run_check(args)
        if args.command == "ask":
            return _run_ask(args)
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    except MissingAPIKeyError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
