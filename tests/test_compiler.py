from __future__ import annotations

from pathlib import Path

from ai_language.compiler import compile_python_file


def test_compile_python_file(tmp_path: Path) -> None:
    path = tmp_path / "gen.py"
    path.write_text("x=1\nprint(x)\n", encoding="utf-8")
    compiled = compile_python_file(path)
    assert compiled.exists()
