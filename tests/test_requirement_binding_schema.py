import json
from pathlib import Path

from sinergy_requirement_binding import (
    bind_change_set_requirements,
    canonical_registry_digest,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/requirement_binding_v1.schema.json"


def _registry() -> dict:
    return {
        "schema": "sinergy.requirement-registry/v1",
        "as_of": "2026-08-18",
        "default_verification_status": "PENDING_RUNNER",
        "requirements": [
            {
                "id": "REQ-AI-001",
                "statement": "AI proposal authority is not execution authority.",
                "owner": "Shtenco/agi_olga",
                "implementation": ["integration/policy_proposal_v2.schema.json"],
                "tests": ["integration/check_policy_proposal_v2_schema.py"],
                "implementation_status": "ADDRESSED",
                "verification_status": "PENDING_RUNNER",
            }
        ],
    }


def _payload() -> dict:
    registry = _registry()
    result = bind_change_set_requirements(
        {
            "schema": "engineering.change-set/v1",
            "requirement_refs": ["REQ-AI-001"],
        },
        registry,
        input_requirement_registry_digest=canonical_registry_digest(registry),
    )
    return result.to_dict()


def test_runtime_emits_exact_declared_shape() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = _payload()
    assert set(payload) == set(schema["required"])
    assert set(payload) <= set(schema["properties"])
    assert payload["schema"] == "requirement.binding/v1"


def test_authority_is_locked_to_semantic_binding_only() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["authority"] == {"const": "SEMANTIC_BINDING_ONLY"}
    assert _payload()["authority"] == "SEMANTIC_BINDING_ONLY"
