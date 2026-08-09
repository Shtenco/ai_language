"""Command-line interface for AI Language Pro."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentError, AgentLimitError, CodingAgent, Workspace, terminal_approval
from .client import AILanguageClient
from .compiler import ExecutionError, compile_python_file, execute_python_file
from .config import DEFAULT_MODEL, MissingAPIKeyError
from .pipeline import SUPPORTED_TARGETS, compile_source

_REASONING_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-language",
        description="AI Language toolkit and local repository-aware coding agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    agent = subparsers.add_parser(
        "agent",
        help="Start the local AI coding agent in the current repository.",
    )
    agent.add_argument("prompt", nargs="?", help="One-shot task. Omit for interactive mode.")
    agent.add_argument("--cwd", default=".", help="Workspace root exposed to the agent.")
    agent.add_argument("--api-key", dest="api_key", help="API key; environment is recommended.")
    agent.add_argument("--model", default=DEFAULT_MODEL, help="Responses API model.")
    agent.add_argument(
        "--reasoning",
        choices=_REASONING_LEVELS,
        default="high",
        help="Reasoning effort for GPT-5.6 models.",
    )
    agent.add_argument("--max-steps", type=int, default=30, help="Maximum model/tool rounds per task.")
    agent.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Approve local file edits and safe command execution automatically.",
    )
    agent.add_argument(
        "--read-only",
        action="store_true",
        help="Disable file mutations and command execution.",
    )
    agent.add_argument(
        "--quiet-tools",
        action="store_true",
        help="Hide tool-call progress from stderr.",
    )

    gen = subparsers.add_parser("generate", help="Generate code from instruction source file.")
    gen.add_argument("source", help="Path to .ailang instruction file.")
    gen.add_argument(
        "--target",
        default="python",
        choices=sorted(SUPPORTED_TARGETS),
        help="Target language.",
    )
    gen.add_argument("--out", required=True, help="Output file path for generated code.")
    gen.add_argument("--emit-graph", help="Optional JSON output path for semantic graph artifact.")

    check = subparsers.add_parser("check", help="Compile/validate generated Python file.")
    check.add_argument("file", help="Path to generated Python file.")

    run = subparsers.add_parser("run", help="Compile and run generated Python on machine.")
    run.add_argument("file", help="Path to generated Python file.")

    ask = subparsers.add_parser("ask", help="Send natural-language prompt to model runtime.")
    ask.add_argument("prompt", help="Prompt text to send to the model.")
    ask.add_argument("--api-key", dest="api_key", help="User-provided API key.")
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    ask.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for models that support it.",
    )

    return parser


def _tool_logger(message: str) -> None:
    print(message, file=sys.stderr)


def _make_agent(args: argparse.Namespace) -> CodingAgent:
    workspace = Workspace(
        Path(args.cwd),
        read_only=args.read_only,
        auto_approve=args.yes,
        approval_callback=terminal_approval,
    )
    return CodingAgent(
        workspace=workspace,
        api_key=args.api_key,
        model=args.model,
        reasoning_effort=args.reasoning,
        max_steps=args.max_steps,
        tool_logger=None if args.quiet_tools else _tool_logger,
    )


def _run_agent(args: argparse.Namespace) -> int:
    agent = _make_agent(args)
    if args.prompt:
        print(agent.run(args.prompt))
        return 0

    print(
        f"AI Language Agent | model={args.model} | workspace={agent.workspace.root}\n"
        "Commands: /help, /clear, /diff, /trace, /exit"
    )
    while True:
        try:
            prompt = input("ail> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/help":
            print(
                "Describe a coding task in natural language.\n"
                "/clear  reset model conversation context\n"
                "/diff   show local git status/diff\n"
                "/trace  show semantic requirement-to-action trace\n"
                "/exit   leave the agent"
            )
            continue
        if prompt == "/clear":
            agent.reset()
            print("Context cleared.")
            continue
        if prompt == "/diff":
            print(agent.workspace.git_diff())
            continue
        if prompt == "/trace":
            print(agent.trace_text())
            continue
        try:
            print(agent.run(prompt))
        except AgentLimitError as exc:
            print(f"Agent limit: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
    return 0


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


def _run_machine(args: argparse.Namespace) -> int:
    compile_python_file(args.file)
    output = execute_python_file(args.file)
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
        if args.command == "agent":
            return _run_agent(args)
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "check":
            return _run_check(args)
        if args.command == "run":
            return _run_machine(args)
        if args.command == "ask":
            return _run_ask(args)
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    except (MissingAPIKeyError, ExecutionError, AgentError) as exc:
        print(f"Runtime configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 1


def agent_main(argv: list[str] | None = None) -> int:
    """Entry point for the short ``ail`` coding-agent command."""
    tail = sys.argv[1:] if argv is None else argv
    return main(["agent", *tail])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
