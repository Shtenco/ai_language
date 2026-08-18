from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


REGISTRY_SCHEMA = "sinergy.requirement-registry/v1"
CHANGE_SET_SCHEMA = "engineering.change-set/v1"
BINDING_SCHEMA = "requirement.binding/v1"
HEX = frozenset("0123456789abcdef")


class BindingInvariantError(ValueError):
    """Raised when requirement governance input is malformed or ambiguous."""


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise BindingInvariantError(f"{field_name} must be text")
    result = value.strip()
    if not result:
        raise BindingInvariantError(f"{field_name} is required")
    return result


def _hex64(value: str, field_name: str) -> str:
    result = _text(value, field_name).lower()
    if len(result) != 64 or any(char not in HEX for char in result):
        raise BindingInvariantError(f"{field_name} must be a 64-character hex digest")
    return result


def _canonical_json(payload: Any) -> bytes:
    """Canonical registry bytes: no float semantics are needed in this metadata contract."""

    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BindingInvariantError("payload must be JSON-compatible") from exc


def canonical_registry_digest(registry: Mapping[str, Any]) -> str:
    validated = validate_requirement_registry(registry)
    return hashlib.sha256(_canonical_json(validated)).hexdigest()


def validate_requirement_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise BindingInvariantError("requirement registry must be an object")
    payload = dict(registry)
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise BindingInvariantError("unsupported requirement registry schema")
    as_of = _text(payload.get("as_of"), "registry.as_of")
    default_status = _text(
        payload.get("default_verification_status"),
        "registry.default_verification_status",
    )
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise BindingInvariantError("requirement registry must contain requirements")

    normalized_requirements: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_requirements):
        if not isinstance(raw, Mapping):
            raise BindingInvariantError(f"requirements[{index}] must be an object")
        requirement = dict(raw)
        requirement_id = _text(requirement.get("id"), f"requirements[{index}].id")
        if not requirement_id.startswith("REQ-"):
            raise BindingInvariantError("requirement IDs must use the REQ-* namespace")
        if requirement_id in ids:
            raise BindingInvariantError(f"duplicate requirement id: {requirement_id}")
        ids.add(requirement_id)
        statement = _text(
            requirement.get("statement"), f"requirements[{index}].statement"
        )
        owner = _text(requirement.get("owner"), f"requirements[{index}].owner")
        if owner.count("/") != 1:
            raise BindingInvariantError("requirement owner must use owner/repository form")
        implementation = requirement.get("implementation")
        tests = requirement.get("tests")
        if not isinstance(implementation, list) or any(
            not isinstance(item, str) or not item.strip() for item in implementation
        ):
            raise BindingInvariantError("requirement implementation must be a string array")
        if not isinstance(tests, list) or any(
            not isinstance(item, str) or not item.strip() for item in tests
        ):
            raise BindingInvariantError("requirement tests must be a string array")
        implementation_status = _text(
            requirement.get("implementation_status"),
            f"requirements[{index}].implementation_status",
        )
        verification_status = _text(
            requirement.get("verification_status"),
            f"requirements[{index}].verification_status",
        )
        normalized_requirements.append(
            {
                "id": requirement_id,
                "statement": statement,
                "owner": owner,
                "implementation": [item.strip() for item in implementation],
                "tests": [item.strip() for item in tests],
                "implementation_status": implementation_status,
                "verification_status": verification_status,
            }
        )

    return {
        "schema": REGISTRY_SCHEMA,
        "as_of": as_of,
        "default_verification_status": default_status,
        "requirements": normalized_requirements,
    }


def _change_set_requirement_refs(change_set: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(change_set, Mapping):
        raise BindingInvariantError("change set must be an object")
    if change_set.get("schema") != CHANGE_SET_SCHEMA:
        raise BindingInvariantError("unsupported change-set schema")
    refs = change_set.get("requirement_refs")
    if not isinstance(refs, list):
        raise BindingInvariantError("change-set requirement_refs must be an array")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        ref = _text(raw, "requirement_ref")
        if not ref.startswith("REQ-"):
            raise BindingInvariantError("change-set requirement_refs must use REQ-* IDs")
        if ref in seen:
            raise BindingInvariantError("change-set requirement_refs must be unique")
        seen.add(ref)
        normalized.append(ref)
    return tuple(normalized)


@dataclass(frozen=True)
class RequirementBindingResult:
    status: str
    registry_digest: str
    requirement_refs: tuple[str, ...]
    blockers: tuple[str, ...]
    authority: str = "SEMANTIC_BINDING_ONLY"

    def __post_init__(self) -> None:
        if self.status not in {"BOUND", "STALE_REGISTRY", "DRAFT_BLOCKED"}:
            raise BindingInvariantError("unsupported requirement binding status")
        _hex64(self.registry_digest, "registry_digest")
        if self.authority != "SEMANTIC_BINDING_ONLY":
            raise BindingInvariantError("requirement binding cannot claim merge/deploy authority")
        if self.status == "BOUND" and self.blockers:
            raise BindingInvariantError("BOUND result cannot contain blockers")
        if self.status != "BOUND" and not self.blockers:
            raise BindingInvariantError("non-BOUND result requires blockers")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": BINDING_SCHEMA,
            "status": self.status,
            "registry_digest": self.registry_digest,
            "requirement_refs": list(self.requirement_refs),
            "blockers": list(self.blockers),
            "authority": self.authority,
        }


def bind_change_set_requirements(
    change_set: Mapping[str, Any],
    requirement_registry: Mapping[str, Any],
    *,
    input_requirement_registry_digest: str,
) -> RequirementBindingResult:
    """Bind REQ references to one exact registry version; never review code itself."""

    registry = validate_requirement_registry(requirement_registry)
    current_digest = hashlib.sha256(_canonical_json(registry)).hexdigest()
    supplied_digest = _hex64(
        input_requirement_registry_digest, "input_requirement_registry_digest"
    )
    refs = _change_set_requirement_refs(change_set)

    if supplied_digest != current_digest:
        return RequirementBindingResult(
            status="STALE_REGISTRY",
            registry_digest=current_digest,
            requirement_refs=refs,
            blockers=("REQUIREMENT_REGISTRY_DIGEST_MISMATCH",),
        )

    known = {item["id"] for item in registry["requirements"]}
    missing = tuple(sorted(set(refs) - known))
    if missing:
        return RequirementBindingResult(
            status="DRAFT_BLOCKED",
            registry_digest=current_digest,
            requirement_refs=refs,
            blockers=tuple(f"UNKNOWN_REQUIREMENT:{item}" for item in missing),
        )

    return RequirementBindingResult(
        status="BOUND",
        registry_digest=current_digest,
        requirement_refs=refs,
        blockers=(),
    )
