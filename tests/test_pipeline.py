from __future__ import annotations

from ai_language.pipeline import compile_source, parse_instructions


def test_parse_and_compile_python() -> None:
    src = """
    generate rest_api | auth; pagination
    test smoke_suite
    """
    result = compile_source(src, target="python")

    assert len(result.instructions) == 2
    assert result.instructions[0].action == "generate"
    assert result.instructions[0].constraints == ["auth", "pagination"]
    assert "def run_program()" in result.code


def test_compile_solidity() -> None:
    src = "deploy token_contract"
    result = compile_source(src, target="solidity")
    assert "contract AIProgram" in result.code


def test_invalid_line_raises() -> None:
    bad = "justoneword"
    try:
        parse_instructions(bad)
    except ValueError as exc:
        assert "Invalid instruction" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
