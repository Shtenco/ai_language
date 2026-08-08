"""Compiler validation and local execution utilities."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


class ExecutionError(RuntimeError):
    """Raised when generated program execution fails or exceeds its time budget."""


def _require_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if not file_path.is_file():
        raise IsADirectoryError(file_path)
    return file_path


def compile_python_file(path: str | Path) -> Path:
    """Compile a Python file to bytecode to validate syntax."""
    file_path = _require_file(path)
    compiled = py_compile.compile(str(file_path), doraise=True)
    return Path(compiled)


def execute_python_file(path: str | Path, *, timeout: float = 5.0) -> str:
    """Execute a Python file in an isolated interpreter and return stdout.

    ``-I`` isolates Python import/environment settings, but this is *not* an OS sandbox.
    Only execute code you trust. ``timeout`` limits wall-clock execution time.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    file_path = _require_file(path).resolve()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(file_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(f"Execution timed out after {timeout:g} seconds.") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise ExecutionError(stderr or f"Execution failed with exit code {completed.returncode}.")
    return completed.stdout.strip()
