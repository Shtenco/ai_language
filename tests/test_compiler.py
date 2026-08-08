from __future__ import annotations

from pathlib import Path

import pytest

from ai_language.compiler import ExecutionError, compile_python_file, execute_python_file


def test_compile_python_file(tmp_path: Path) -> None:
    path = tmp_path / "gen.py"
    path.write_text("x=1\nprint(x)\n", encoding="utf-8")

    compiled = compile_python_file(path)

    assert compiled.exists()


def test_execute_python_file(tmp_path: Path) -> None:
    path = tmp_path / "prog.py"
    path.write_text("print('machine-ok')\n", encoding="utf-8")

    assert execute_python_file(path) == "machine-ok"


def test_execute_python_file_reports_failure(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(ExecutionError, match="boom"):
        execute_python_file(path)


def test_execute_python_file_times_out(tmp_path: Path) -> None:
    path = tmp_path / "slow.py"
    path.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")

    with pytest.raises(ExecutionError, match="timed out"):
        execute_python_file(path, timeout=0.05)


def test_execute_requires_positive_timeout(tmp_path: Path) -> None:
    path = tmp_path / "prog.py"
    path.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="greater than zero"):
        execute_python_file(path, timeout=0)
