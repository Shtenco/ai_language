from __future__ import annotations

import pytest

from ai_language.errors import AILanguageSyntaxError
from ai_language.pipeline import (
    SUPPORTED_TARGETS,
    canonical_source,
    compile_source,
    parse_instructions,
)


def test_parse_normalizes_and_compiles_python() -> None:
    src = """
    # comment
    GENERATE   rest_api   | auth ;  pagination
    test smoke_suite
    """
    result = compile_source(src, target="python")

    assert len(result.instructions) == 2
    assert result.instructions[0].action == "generate"
    assert result.instructions[0].target == "rest_api"
    assert result.instructions[0].constraints == ("auth", "pagination")
    assert "def run_program()" in result.code
    assert "generate rest_api | auth; pagination" in result.code


def test_semantically_equivalent_source_has_same_fingerprint() -> None:
    one = compile_source("GENERATE   api | auth ; retries\n")
    two = compile_source("generate api | auth; retries\n")

    assert one.fingerprint == two.fingerprint
    assert canonical_source(one.instructions) == "generate api | auth; retries\n"


@pytest.mark.parametrize("target", sorted(SUPPORTED_TARGETS))
def test_all_targets_preserve_constraints_and_fingerprint(target: str) -> None:
    result = compile_source('generate service "v2" | retries; idempotency', target=target)

    assert "retries" in result.code
    assert "idempotency" in result.code
    assert result.fingerprint in result.code


def test_c_like_targets_escape_quotes_and_backslashes() -> None:
    source = r'generate path\\to\\"service" | quote "safe"'

    for target in ("c", "rust", "solidity", "kotlin"):
        code = compile_source(source, target=target).code
        assert r"\\" in code
        assert r"\"" in code


def test_to_dict_is_json_ready_and_versioned() -> None:
    result = compile_source("deploy token_contract", target="solidity")
    payload = result.to_dict(include_code=False)

    assert payload["schema_version"] == 1
    assert payload["target"] == "solidity"
    assert payload["fingerprint"] == result.fingerprint
    assert payload["semantic_graph"]["nodes"][0]["id"] == "program"
    assert "code" not in payload


def test_invalid_line_reports_line_number() -> None:
    with pytest.raises(AILanguageSyntaxError) as caught:
        parse_instructions("generate api\njustoneword\n")

    assert caught.value.line_number == 2
    assert "expected ACTION TARGET" in str(caught.value)


def test_empty_source_raises() -> None:
    with pytest.raises(ValueError, match="No valid instructions"):
        parse_instructions("# comments only\n\n")


def test_unsupported_target_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported target"):
        compile_source("generate api", target="brainfuck")
