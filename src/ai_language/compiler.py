"""Compiler and runtime utilities for generated code."""

from __future__ import annotations

import py_compile
from pathlib import Path


def compile_python_file(path: str | Path) -> Path:
    """Compile a Python file to bytecode to validate syntax."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    compiled = py_compile.compile(str(file_path), doraise=True)
    return Path(compiled)
