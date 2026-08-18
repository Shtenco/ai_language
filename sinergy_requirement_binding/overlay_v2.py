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

OVERLAY_SCHEMA = "sinergy.requirement-overlay/v2"
OVERLAY_AUTHORITY = "SEMANTIC_REQUIREMENT_OVERLAY_ONLY"


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingInvariantError(f"{field_name} is required")
    return value.strip()


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


def validate_requirement_overlay_v2(overlay: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(overlay, Mapping):
        raise BindingInvariantError("requirement overlay must be an object")
    payload = dict(overlay)
    if payload.get("schema") != OVERLAY_SCHEMA:
        raise BindingInvariantError("unsupported requirement overlay schema")
    if payload.get("base_registry_schema") != REGISTRY_SCHEMA:
        raise BindingInvariantError("overlay must target sinergy.requirement-registry/v1")
    if payload.get("authority") != OVERLAY_AUTHORITY:
        raise BindingInvariantError("overlay cannot claim merge/deploy/financial authority")
    as_of = _text(payload.get("as_of"), "overlay.as_of")
    requirements = payload.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise BindingInvariantError("overlay must contain requirements")

    # Reuse the exact base-registry requirement validator by wrapping overlay rows
    # into a temporary v1-shaped registry. This keeps semantic field rules identical.
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
        "authority": OVERLAY_AUTHORITY,
        "requirements": normalized["requirements"],
    }


def requirement_overlay_v2_digest(overlay: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(validate_requirement_overlay_v2(overlay))).hexdigest()


@dataclass(frozen=True)
class MaterializedRequirementRegistryV2:
    base_registry_digest: str
    overlay_digest: str
    registry: dict[str, Any]
    registry_digest: str
    added_requirement_ids: tuple[str, ...]
    authority: str = "SEMANTIC_MATERIALIZATION_ONLY"

    def __post_init__(self) -> None:
        if self.authority != "SEMANTIC_MATERIALIZATION_ONLY":
            raise BindingInvariantError("materialization cannot claim code or financial authority")


def materialize_requirement_overlay_v2(
    base_registry: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> MaterializedRequirementRegistryV2:
    base = validate_requirement_registry(base_registry)
    addition = validate_requirement_overlay_v2(overlay)
    base_ids = {item["id"] for item in base["requirements"]}
    new_ids = [item["id"] for item in addition["requirements"]]
    collisions = tuple(sorted(base_ids.intersection(new_ids)))
    if collisions:
        raise BindingInvariantError(
            "requirement overlay is append-only and cannot replace existing IDs: "
            + ",".join(collisions)
        )
    if len(new_ids) != len(set(new_ids)):
        raise BindingInvariantError("overlay requirement IDs must be unique")

    materialized = {
        "schema": REGISTRY_SCHEMA,
        "as_of": max(base["as_of"], addition["as_of"]),
        "default_verification_status": base["default_verification_status"],
        "requirements": base["requirements"] + addition["requirements"],
    }
    # Revalidate the merged v1-compatible registry before exposing it to the
    # existing requirement.binding/v1 contract.
    materialized = validate_requirement_registry(materialized)
    return MaterializedRequirementRegistryV2(
        base_registry_digest=canonical_registry_digest(base),
        overlay_digest=requirement_overlay_v2_digest(addition),
        registry=materialized,
        registry_digest=canonical_registry_digest(materialized),
        added_requirement_ids=tuple(new_ids),
    )
