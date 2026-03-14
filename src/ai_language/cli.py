"""Command-line interface for ai-language."""

from __future__ import annotations

import argparse
import sys

from .client import AILanguageClient
from .config import DEFAULT_MODEL, MissingAPIKeyError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-language",
        description="Professional AI language CLI with user-managed API key configuration.",
    )
    parser.add_argument("prompt", help="Prompt text to send to the model.")
    parser.add_argument("--api-key", dest="api_key", help="API key passed explicitly by user.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature between 0 and 2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        client = AILanguageClient(api_key=args.api_key, model=args.model)
        output = client.generate(args.prompt, temperature=args.temperature)
        print(output)
        return 0
    except MissingAPIKeyError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
