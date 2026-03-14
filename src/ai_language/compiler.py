"""Compiler and runtime utilities for generated code."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


class ExecutionError(RuntimeError):
    """Raised when generated program execution fails."""


def compile_python_file(path: str | Path) -> Path:
    """Compile a Python file to bytecode to validate syntax."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    compiled = py_compile.compile(str(file_path), doraise=True)
    return Path(compiled)


def execute_python_file(path: str | Path) -> str:
    """Execute a generated Python file and return stdout.

    This models the final step in the chain:
    instructions -> semantic graph -> AST -> code -> compiler -> machine.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    completed = subprocess.run(
        [sys.executable, str(file_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ExecutionError(completed.stderr.strip() or "Execution failed.")
    return completed.stdout.strip()
