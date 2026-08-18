from copy import deepcopy

import pytest

from sinergy_requirement_binding import (
    BindingInvariantError,
    bind_change_set_requirements,
    canonical_registry_digest,
    validate_requirement_registry,
)


def _registry() -> dict:
    return {
        "schema": "sinergy.requirement-registry/v1",
        "as_of": "2026-08-18",
        "default_verification_status": "PENDING_RUNNER",
        "requirements": [
            {
                "id": "REQ-FIN-004",
                "statement": "Principal repayment must not be distributable profit.",
                "owner": "Shtenco/synergy_financial_os",
                "implementation": ["sinergy_financial_os/reporting_metrics.py"],
                "tests": ["tests/test_reporting_metrics.py"],
                "implementation_status": "ADDRESSED",
                "verification_status": "PENDING_RUNNER",
            },
            {
                "id": "REQ-AI-001",
                "statement": "AI may emit proposals but may not mutate the ledger.",
                "owner": "Shtenco/agi_olga",
                "implementation": ["integration/policy_proposal_v2.schema.json"],
                "tests": ["integration/check_policy_proposal_v2_schema.py"],
                "implementation_status": "ADDRESSED",
                "verification_status": "PENDING_RUNNER",
            },
        ],
    }


def _change_set(refs=None) -> dict:
    return {
        "schema": "engineering.change-set/v1",
        "change_set_id": "chg-001",
        "repositories": ["Shtenco/synergy_financial_os"],
        "requirement_refs": ["REQ-FIN-004"] if refs is None else refs,
        "hardening_refs": [],
        "risk_domains": ["FINANCIAL"],
        "evidence": [],
        "summary": "Example",
        "automation_authority": "NONE",
    }


def test_exact_registry_and_known_requirement_bind_successfully() -> None:
    registry = _registry()
    digest = canonical_registry_digest(registry)
    result = bind_change_set_requirements(
        _change_set(), registry, input_requirement_registry_digest=digest
    )
    assert result.status == "BOUND"
    assert result.registry_digest == digest
    assert result.requirement_refs == ("REQ-FIN-004",)
    assert result.blockers == ()


def test_same_ids_but_changed_statement_make_old_binding_stale() -> None:
    original = _registry()
    old_digest = canonical_registry_digest(original)
    changed = deepcopy(original)
    changed["requirements"][0]["statement"] += " Exact external evidence is required."
    assert canonical_registry_digest(changed) != old_digest
    result = bind_change_set_requirements(
        _change_set(), changed, input_requirement_registry_digest=old_digest
    )
    assert result.status == "STALE_REGISTRY"
    assert result.blockers == ("REQUIREMENT_REGISTRY_DIGEST_MISMATCH",)


def test_unknown_requirement_is_blocked_even_when_registry_digest_matches() -> None:
    registry = _registry()
    result = bind_change_set_requirements(
        _change_set(["REQ-FAKE-999"]),
        registry,
        input_requirement_registry_digest=canonical_registry_digest(registry),
    )
    assert result.status == "DRAFT_BLOCKED"
    assert result.blockers == ("UNKNOWN_REQUIREMENT:REQ-FAKE-999",)


def test_empty_requirement_refs_are_valid_semantic_binding() -> None:
    registry = _registry()
    result = bind_change_set_requirements(
        _change_set([]),
        registry,
        input_requirement_registry_digest=canonical_registry_digest(registry),
    )
    assert result.status == "BOUND"
    assert result.requirement_refs == ()


def test_duplicate_requirement_ids_in_registry_fail_closed() -> None:
    registry = _registry()
    registry["requirements"].append(deepcopy(registry["requirements"][0]))
    with pytest.raises(BindingInvariantError):
        validate_requirement_registry(registry)


def test_duplicate_requirement_refs_in_change_set_fail_closed() -> None:
    registry = _registry()
    with pytest.raises(BindingInvariantError):
        bind_change_set_requirements(
            _change_set(["REQ-FIN-004", "REQ-FIN-004"]),
            registry,
            input_requirement_registry_digest=canonical_registry_digest(registry),
        )


def test_wrong_change_set_schema_fails_closed() -> None:
    registry = _registry()
    change = _change_set()
    change["schema"] = "other/v1"
    with pytest.raises(BindingInvariantError):
        bind_change_set_requirements(
            change,
            registry,
            input_requirement_registry_digest=canonical_registry_digest(registry),
        )


def test_requirement_owner_must_be_repository_form() -> None:
    registry = _registry()
    registry["requirements"][0]["owner"] = "not-a-repository"
    with pytest.raises(BindingInvariantError):
        validate_requirement_registry(registry)
