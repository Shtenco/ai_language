from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .binding import (
    REGISTRY_SCHEMA,
    BindingInvariantError,
    canonical_registry_digest,
    validate_requirement_registry,
)


OVERLAY_SCHEMA = "sinergy.requirement-overlay/v3"
BASE_OVERLAY_SCHEMA = "sinergy.requirement-overlay/v2"
OVERLAY_AUTHORITY = "SEMANTIC_REQUIREMENT_OVERLAY_ONLY"
HEX = frozenset("0123456789abcdef")


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingInvariantError(f"{field_name} is required")
    return value.strip()


def _hex64(value: Any, field_name: str) -> str:
    result = _text(value, field_name).lower()
    if len(result) != 64 or any(c not in HEX for c in result):
        raise BindingInvariantError(f"{field_name} must be hex64")
    return result


def _canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BindingInvariantError("overlay must be JSON-compatible") from exc


def validate_requirement_overlay_v3(overlay: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(overlay, Mapping):
        raise BindingInvariantError("requirement overlay must be an object")
    payload = dict(overlay)
    if payload.get("schema") != OVERLAY_SCHEMA:
        raise BindingInvariantError("unsupported requirement overlay schema")
    if payload.get("base_registry_schema") != REGISTRY_SCHEMA:
        raise BindingInvariantError("overlay must target requirement-registry/v1")
    if payload.get("base_overlay_schema") != BASE_OVERLAY_SCHEMA:
        raise BindingInvariantError("v3 must explicitly extend requirement-overlay/v2")
    if payload.get("authority") != OVERLAY_AUTHORITY:
        raise BindingInvariantError("overlay cannot claim code/deploy/financial authority")
    as_of = _text(payload.get("as_of"), "overlay.as_of")
    requirements = payload.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise BindingInvariantError("overlay must contain requirements")
    normalized = validate_requirement_registry(
        {
            "schema": REGISTRY_SCHEMA,
            "as_of": as_of,
            "default_verification_status": "PENDING_RUNNER",
            "requirements": requirements,
        }
    )
    return {
        "schema": OVERLAY_SCHEMA,
        "as_of": as_of,
        "base_registry_schema": REGISTRY_SCHEMA,
        "base_overlay_schema": BASE_OVERLAY_SCHEMA,
        "authority": OVERLAY_AUTHORITY,
        "requirements": normalized["requirements"],
    }


def requirement_overlay_v3_digest(overlay: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(validate_requirement_overlay_v3(overlay))).hexdigest()


@dataclass(frozen=True)
class MaterializedRequirementRegistryV3:
    base_materialized_registry_digest: str
    overlay_digest: str
    registry: dict[str, Any]
    registry_digest: str
    added_requirement_ids: tuple[str, ...]
    authority: str = "SEMANTIC_MATERIALIZATION_ONLY"

    def __post_init__(self) -> None:
        _hex64(self.base_materialized_registry_digest, "base_materialized_registry_digest")
        _hex64(self.overlay_digest, "overlay_digest")
        _hex64(self.registry_digest, "registry_digest")
        if self.authority != "SEMANTIC_MATERIALIZATION_ONLY":
            raise BindingInvariantError("materialization cannot claim code/financial authority")


def materialize_requirement_overlay_v3(
    base_materialized_registry: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    expected_base_materialized_registry_digest: str,
) -> MaterializedRequirementRegistryV3:
    base = validate_requirement_registry(base_materialized_registry)
    current_digest = canonical_registry_digest(base)
    expected = _hex64(
        expected_base_materialized_registry_digest,
        "expected_base_materialized_registry_digest",
    )
    if current_digest != expected:
        raise BindingInvariantError("v3 overlay base materialized registry digest mismatch")
    addition = validate_requirement_overlay_v3(overlay)
    base_ids = {item["id"] for item in base["requirements"]}
    new_ids = [item["id"] for item in addition["requirements"]]
    collisions = tuple(sorted(base_ids.intersection(new_ids)))
    if collisions:
        raise BindingInvariantError(
            "requirement overlay is append-only and cannot replace existing IDs: "
            + ",".join(collisions)
        )
    materialized = validate_requirement_registry(
        {
            "schema": REGISTRY_SCHEMA,
            "as_of": max(base["as_of"], addition["as_of"]),
            "default_verification_status": base["default_verification_status"],
            "requirements": base["requirements"] + addition["requirements"],
        }
    )
    return MaterializedRequirementRegistryV3(
        base_materialized_registry_digest=current_digest,
        overlay_digest=requirement_overlay_v3_digest(addition),
        registry=materialized,
        registry_digest=canonical_registry_digest(materialized),
        added_requirement_ids=tuple(new_ids),
    )
