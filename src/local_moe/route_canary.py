from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from filelock import FileLock, Timeout

from .assistant_bridge_attestation import (
    ed25519_public_key_sha256,
    load_ed25519_public_key_pem,
)
from .assistant_bridge_integrity import canonical_json_bytes, canonical_sha256
from .route_policy import ShadowRouteDecision
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    DIFFICULTIES,
    ROUTE_PLANS,
    VerifiedRoutingError,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


AUTHORIZATION_PAYLOAD_TYPE = (
    "application/vnd.mymoe.verified-routing-canary-authorization+json"
)
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_CANARY_BASIS_POINTS = 500
_MIN_ASSIGNMENT_SECRET_BYTES = 32
_MAX_ASSIGNMENT_SECRET_BYTES = 1024
_ROUTE_RANK = {"local": 0, "local_then_verify": 1, "premium": 2}
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")

_RUNTIME_FIELDS = {
    "schema_version",
    "mode",
    "route_policy_path",
    "scorecard_path",
    "manifest_path",
    "authorization_path",
    "operator_key_id",
    "operator_public_key_path",
    "operator_public_key_sha256",
    "assignment_secret_env",
    "chronology_path",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "contract",
    "current_mode",
    "target_mode",
    "authority",
    "producer_authenticity",
    "applied",
    "not_before",
    "expires_at",
    "evidence_valid_until",
    "canary_basis_points",
    "assignment_salt_sha256",
    "lineage",
    "enabled_cells",
    "invariants",
    "manifest_sha256",
}
_LINEAGE_FIELDS = {
    "plan_sha256",
    "report_sha256",
    "gate_policy_digest",
    "route_policy_digest",
    "scorecard_digest",
    "training_source_digest",
    "evaluator_sha256",
}
_CELL_FIELDS = {
    "profile",
    "capabilities",
    "difficulty",
    "baseline_route",
    "candidate_route",
    "config_sha256",
    "signal_provider_config_sha256",
    "runtime_plan_sha256",
    "paired_tasks",
    "candidate_success_rate",
    "candidate_success_ci_lower",
}
_INVARIANT_FIELDS = {
    "monotone_less_premium_only",
    "privacy_budget_and_capability_guards_preserved",
    "runtime_integration_required_before_application",
    "trusted_signature_required_before_runtime_consumption",
}
_AUTHORIZATION_FIELDS = {
    "schema_version",
    "contract",
    "activation_id",
    "operator_key_id",
    "manifest_sha256",
    "bridge_config_sha256",
    "route_policy_digest",
    "scorecard_digest",
    "issued_at",
    "not_before",
    "expires_at",
    "maximum_canary_basis_points",
    "authorization_sha256",
}
_DECISION_FIELDS = {
    "schema_version",
    "contract",
    "mode",
    "task_fingerprint",
    "profile",
    "capabilities",
    "difficulty",
    "baseline_route",
    "effective_route",
    "shadow_recommended_route",
    "applied",
    "abstained",
    "reason_codes",
    "route_receipt_id",
    "route_receipt_sha256",
    "runtime_plan_sha256",
    "signal_provider_config_sha256",
    "shadow_decision_sha256",
    "policy_digest",
    "scorecard_digest",
    "bridge_config_sha256",
    "manifest_sha256",
    "authorization_sha256",
    "operator_key_id",
    "assignment_bucket",
    "canary_basis_points",
    "decision_sha256",
}

_VERIFIED_AUTHORITY_PROOF = object()


class RouteCanaryError(VerifiedRoutingError):
    """Raised when canary authority or evidence fails closed."""


@dataclass(frozen=True)
class VerifiedRoutingRuntimeConfig:
    route_policy_path: str
    scorecard_path: str
    manifest_path: str
    authorization_path: str
    operator_key_id: str
    operator_public_key_path: str
    operator_public_key_sha256: str
    assignment_secret_env: str
    chronology_path: str
    source_sha256: str
    schema_version: str = CONTRACT_VERSION
    mode: str = "canary"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.mode != "canary":
            raise RouteCanaryError("Verified routing runtime config is unsupported.")
        for name in (
            "route_policy_path",
            "scorecard_path",
            "manifest_path",
            "authorization_path",
            "operator_public_key_path",
            "chronology_path",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise RouteCanaryError(f"{name} is required.")
        require_safe_id(self.operator_key_id, "operator_key_id")
        require_sha256(
            self.operator_public_key_sha256,
            "operator_public_key_sha256",
        )
        require_sha256(self.source_sha256, "runtime config source_sha256")
        if _ENV_NAME.fullmatch(self.assignment_secret_env) is None:
            raise RouteCanaryError("assignment_secret_env is invalid.")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "route_policy_path": self.route_policy_path,
            "scorecard_path": self.scorecard_path,
            "manifest_path": self.manifest_path,
            "authorization_path": self.authorization_path,
            "operator_key_id": self.operator_key_id,
            "operator_public_key_path": self.operator_public_key_path,
            "operator_public_key_sha256": self.operator_public_key_sha256,
            "assignment_secret_env": self.assignment_secret_env,
            "chronology_path": self.chronology_path,
        }


@dataclass(frozen=True)
class CanaryCell:
    profile: str
    capabilities: tuple[str, ...]
    difficulty: str
    baseline_route: str
    candidate_route: str
    config_sha256: str
    signal_provider_config_sha256: str
    runtime_plan_sha256: str
    paired_tasks: int
    candidate_success_rate: float
    candidate_success_ci_lower: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", require_safe_id(self.profile, "profile"))
        capabilities = tuple(
            sorted(require_identifier_tuple(self.capabilities, "capabilities"))
        )
        object.__setattr__(self, "capabilities", capabilities)
        if self.difficulty not in DIFFICULTIES:
            raise RouteCanaryError("Canary cell difficulty is unsupported.")
        if self.baseline_route not in ROUTE_PLANS or self.candidate_route not in ROUTE_PLANS:
            raise RouteCanaryError("Canary cell route is unsupported.")
        if _ROUTE_RANK[self.candidate_route] >= _ROUTE_RANK[self.baseline_route]:
            raise RouteCanaryError("Canary cells must move monotonically toward less premium use.")
        for name in (
            "config_sha256",
            "signal_provider_config_sha256",
            "runtime_plan_sha256",
        ):
            require_sha256(getattr(self, name), name)
        paired = require_non_negative_int(self.paired_tasks, "paired_tasks")
        if paired == 0:
            raise RouteCanaryError("Canary cells require paired evidence.")
        object.__setattr__(self, "paired_tasks", paired)
        success = require_finite_number(
            self.candidate_success_rate,
            "candidate_success_rate",
            minimum=0.0,
            maximum=1.0,
        )
        lower = require_finite_number(
            self.candidate_success_ci_lower,
            "candidate_success_ci_lower",
            minimum=0.0,
            maximum=1.0,
        )
        if lower > success:
            raise RouteCanaryError("Canary confidence lower bound exceeds the observed rate.")
        object.__setattr__(self, "candidate_success_rate", success)
        object.__setattr__(self, "candidate_success_ci_lower", lower)

    @property
    def match_key(self) -> tuple[object, ...]:
        return (
            self.profile,
            self.capabilities,
            self.difficulty,
            self.baseline_route,
            self.candidate_route,
            self.config_sha256,
            self.signal_provider_config_sha256,
            self.runtime_plan_sha256,
        )

    def payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "capabilities": list(self.capabilities),
            "difficulty": self.difficulty,
            "baseline_route": self.baseline_route,
            "candidate_route": self.candidate_route,
            "config_sha256": self.config_sha256,
            "signal_provider_config_sha256": self.signal_provider_config_sha256,
            "runtime_plan_sha256": self.runtime_plan_sha256,
            "paired_tasks": self.paired_tasks,
            "candidate_success_rate": self.candidate_success_rate,
            "candidate_success_ci_lower": self.candidate_success_ci_lower,
        }


@dataclass(frozen=True)
class VerifiedRoutingCanaryManifest:
    not_before: str
    expires_at: str
    evidence_valid_until: str
    canary_basis_points: int
    assignment_salt_sha256: str
    lineage: Mapping[str, str]
    enabled_cells: tuple[CanaryCell, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("not_before", "expires_at", "evidence_valid_until"):
            object.__setattr__(
                self,
                name,
                require_utc_timestamp(getattr(self, name), name),
            )
        if not _timestamp(self.not_before) < _timestamp(self.expires_at):
            raise RouteCanaryError("Canary manifest time window is invalid.")
        if _timestamp(self.evidence_valid_until) < _timestamp(self.expires_at):
            raise RouteCanaryError("Canary manifest outlives its evidence.")
        basis_points = require_non_negative_int(
            self.canary_basis_points,
            "canary_basis_points",
        )
        if not 0 < basis_points <= _MAX_CANARY_BASIS_POINTS:
            raise RouteCanaryError("Canary size is outside the runtime safety cap.")
        object.__setattr__(self, "canary_basis_points", basis_points)
        require_sha256(self.assignment_salt_sha256, "assignment_salt_sha256")
        lineage = dict(self.lineage)
        _reject_unknown(lineage, _LINEAGE_FIELDS, "canary lineage")
        if set(lineage) != _LINEAGE_FIELDS:
            raise RouteCanaryError("Canary lineage is incomplete.")
        for name, value in lineage.items():
            require_sha256(value, f"lineage {name}")
        object.__setattr__(self, "lineage", MappingProxyType(lineage))
        cells = tuple(self.enabled_cells)
        if not cells or any(not isinstance(item, CanaryCell) for item in cells):
            raise RouteCanaryError("Canary manifest enabled_cells are invalid.")
        if len({item.match_key for item in cells}) != len(cells):
            raise RouteCanaryError("Canary manifest repeats an enabled cell.")
        object.__setattr__(self, "enabled_cells", cells)
        require_sha256(self.manifest_sha256, "manifest_sha256")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedRoutingCanaryManifest",
            "current_mode": "shadow",
            "target_mode": "canary",
            "authority": "structural_eligibility_only",
            "producer_authenticity": "not_attested",
            "applied": False,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
            "evidence_valid_until": self.evidence_valid_until,
            "canary_basis_points": self.canary_basis_points,
            "assignment_salt_sha256": self.assignment_salt_sha256,
            "lineage": dict(self.lineage),
            "enabled_cells": [item.payload() for item in self.enabled_cells],
            "invariants": {name: True for name in sorted(_INVARIANT_FIELDS)},
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "VerifiedRoutingCanaryManifest":
        payload = _mapping(raw, "canary manifest")
        _reject_unknown(payload, _MANIFEST_FIELDS, "canary manifest")
        if set(payload) != _MANIFEST_FIELDS:
            raise RouteCanaryError("Canary manifest is incomplete.")
        fixed = {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedRoutingCanaryManifest",
            "current_mode": "shadow",
            "target_mode": "canary",
            "authority": "structural_eligibility_only",
            "producer_authenticity": "not_attested",
            "applied": False,
        }
        if any(payload.get(name) != value for name, value in fixed.items()):
            raise RouteCanaryError("Canary manifest authority contract is invalid.")
        invariants = _mapping(payload["invariants"], "canary invariants")
        _reject_unknown(invariants, _INVARIANT_FIELDS, "canary invariants")
        if set(invariants) != _INVARIANT_FIELDS or any(
            invariants[name] is not True for name in _INVARIANT_FIELDS
        ):
            raise RouteCanaryError("Canary manifest invariants are not preserved.")
        cells_raw = payload["enabled_cells"]
        if not isinstance(cells_raw, list):
            raise RouteCanaryError("Canary enabled_cells must be a list.")
        cells: list[CanaryCell] = []
        for item in cells_raw:
            cell = _mapping(item, "canary cell")
            _reject_unknown(cell, _CELL_FIELDS, "canary cell")
            if set(cell) != _CELL_FIELDS:
                raise RouteCanaryError("Canary cell is incomplete.")
            cells.append(CanaryCell(**cell))  # type: ignore[arg-type]
        manifest = cls(
            not_before=payload["not_before"],  # type: ignore[arg-type]
            expires_at=payload["expires_at"],  # type: ignore[arg-type]
            evidence_valid_until=payload["evidence_valid_until"],  # type: ignore[arg-type]
            canary_basis_points=payload["canary_basis_points"],  # type: ignore[arg-type]
            assignment_salt_sha256=payload["assignment_salt_sha256"],  # type: ignore[arg-type]
            lineage=_mapping(payload["lineage"], "canary lineage"),
            enabled_cells=tuple(cells),
            manifest_sha256=payload["manifest_sha256"],  # type: ignore[arg-type]
        )
        if sha256_json(manifest.content_payload()) != manifest.manifest_sha256:
            raise RouteCanaryError("Canary manifest digest is invalid.")
        return manifest


@dataclass(frozen=True)
class VerifiedRoutingCanaryAuthorization:
    activation_id: str
    operator_key_id: str
    manifest_sha256: str
    bridge_config_sha256: str
    route_policy_digest: str
    scorecard_digest: str
    issued_at: str
    not_before: str
    expires_at: str
    maximum_canary_basis_points: int
    authorization_sha256: str

    def __post_init__(self) -> None:
        for name in ("activation_id", "operator_key_id"):
            object.__setattr__(self, name, require_safe_id(getattr(self, name), name))
        for name in (
            "manifest_sha256",
            "bridge_config_sha256",
            "route_policy_digest",
            "scorecard_digest",
            "authorization_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("issued_at", "not_before", "expires_at"):
            object.__setattr__(
                self,
                name,
                require_utc_timestamp(getattr(self, name), name),
            )
        if not (
            _timestamp(self.issued_at)
            <= _timestamp(self.not_before)
            < _timestamp(self.expires_at)
        ):
            raise RouteCanaryError("Canary authorization time window is invalid.")
        maximum = require_non_negative_int(
            self.maximum_canary_basis_points,
            "maximum_canary_basis_points",
        )
        if not 0 < maximum <= _MAX_CANARY_BASIS_POINTS:
            raise RouteCanaryError("Canary authorization cap is unsafe.")
        object.__setattr__(self, "maximum_canary_basis_points", maximum)
        if canonical_sha256(self.content_payload()) != self.authorization_sha256:
            raise RouteCanaryError("Canary authorization digest is invalid.")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedRoutingCanaryAuthorization",
            "activation_id": self.activation_id,
            "operator_key_id": self.operator_key_id,
            "manifest_sha256": self.manifest_sha256,
            "bridge_config_sha256": self.bridge_config_sha256,
            "route_policy_digest": self.route_policy_digest,
            "scorecard_digest": self.scorecard_digest,
            "issued_at": self.issued_at,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
            "maximum_canary_basis_points": self.maximum_canary_basis_points,
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["authorization_sha256"] = self.authorization_sha256
        return payload

    @classmethod
    def from_payload(
        cls,
        raw: Mapping[str, object],
    ) -> "VerifiedRoutingCanaryAuthorization":
        payload = _mapping(raw, "canary authorization")
        _reject_unknown(payload, _AUTHORIZATION_FIELDS, "canary authorization")
        if set(payload) != _AUTHORIZATION_FIELDS:
            raise RouteCanaryError("Canary authorization is incomplete.")
        if (
            payload["schema_version"] != CONTRACT_VERSION
            or payload["contract"] != "VerifiedRoutingCanaryAuthorization"
        ):
            raise RouteCanaryError("Canary authorization contract is invalid.")
        data = dict(payload)
        data.pop("schema_version")
        data.pop("contract")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class VerifiedCanaryAuthority:
    """Opaque proof that one authorization passed trust and chronology checks."""

    authorization: VerifiedRoutingCanaryAuthorization
    verified_at: str
    runtime_config_sha256: str
    _proof: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._proof is not _VERIFIED_AUTHORITY_PROOF:
            raise RouteCanaryError(
                "Verified canary authority must come from the runtime verifier."
            )
        if not isinstance(self.authorization, VerifiedRoutingCanaryAuthorization):
            raise RouteCanaryError("Verified canary authority is invalid.")
        object.__setattr__(
            self,
            "verified_at",
            require_utc_timestamp(self.verified_at, "verified_at"),
        )
        require_sha256(self.runtime_config_sha256, "runtime_config_sha256")

    @property
    def manifest_sha256(self) -> str:
        return self.authorization.manifest_sha256

    @property
    def authorization_sha256(self) -> str:
        return self.authorization.authorization_sha256

    @property
    def operator_key_id(self) -> str:
        return self.authorization.operator_key_id


@dataclass(frozen=True)
class CanaryRouteDecision:
    task_fingerprint: str
    profile: str
    capabilities: tuple[str, ...]
    difficulty: str
    baseline_route: str
    effective_route: str
    shadow_recommended_route: str
    applied: bool
    abstained: bool
    reason_codes: tuple[str, ...]
    route_receipt_id: str
    route_receipt_sha256: str
    runtime_plan_sha256: str
    signal_provider_config_sha256: str
    shadow_decision_sha256: str
    policy_digest: str
    scorecard_digest: str
    bridge_config_sha256: str
    manifest_sha256: str
    authorization_sha256: str
    operator_key_id: str
    assignment_bucket: int
    canary_basis_points: int
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "task_fingerprint",
            "route_receipt_sha256",
            "runtime_plan_sha256",
            "signal_provider_config_sha256",
            "shadow_decision_sha256",
            "policy_digest",
            "scorecard_digest",
            "bridge_config_sha256",
            "manifest_sha256",
            "authorization_sha256",
        ):
            require_sha256(getattr(self, name), name)
        object.__setattr__(self, "profile", require_safe_id(self.profile, "profile"))
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(require_identifier_tuple(self.capabilities, "capabilities"))),
        )
        if self.difficulty not in DIFFICULTIES:
            raise RouteCanaryError("Canary decision difficulty is unsupported.")
        if self.baseline_route not in ROUTE_PLANS or self.effective_route not in ROUTE_PLANS:
            raise RouteCanaryError("Canary decision route is unsupported.")
        if self.shadow_recommended_route not in ROUTE_PLANS:
            raise RouteCanaryError("Canary shadow recommendation is unsupported.")
        if not isinstance(self.applied, bool) or not isinstance(self.abstained, bool):
            raise RouteCanaryError("Canary decision flags must be boolean.")
        reasons = tuple(require_safe_id(item, "reason_codes") for item in self.reason_codes)
        if not reasons or len(set(reasons)) != len(reasons):
            raise RouteCanaryError("Canary decision reason codes are invalid.")
        object.__setattr__(self, "reason_codes", reasons)
        require_safe_id(self.route_receipt_id, "route_receipt_id")
        require_safe_id(self.operator_key_id, "operator_key_id")
        bucket = require_non_negative_int(self.assignment_bucket, "assignment_bucket")
        basis_points = require_non_negative_int(
            self.canary_basis_points,
            "canary_basis_points",
        )
        if bucket >= 10_000 or not 0 < basis_points <= _MAX_CANARY_BASIS_POINTS:
            raise RouteCanaryError("Canary assignment is invalid.")
        object.__setattr__(self, "assignment_bucket", bucket)
        object.__setattr__(self, "canary_basis_points", basis_points)
        if self.applied:
            if self.abstained or self.effective_route != self.shadow_recommended_route:
                raise RouteCanaryError("Applied canary decision is inconsistent.")
            if _ROUTE_RANK[self.effective_route] >= _ROUTE_RANK[self.baseline_route]:
                raise RouteCanaryError("Applied canary decision widens provider use.")
            if bucket >= basis_points:
                raise RouteCanaryError("Applied canary decision is outside its cohort.")
        elif self.effective_route != self.baseline_route or not self.abstained:
            raise RouteCanaryError("Abstained canary decision must retain baseline.")
        object.__setattr__(self, "decision_sha256", canonical_sha256(self.content_payload()))

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "contract": "CanaryRouteDecision",
            "mode": "canary",
            "task_fingerprint": self.task_fingerprint,
            "profile": self.profile,
            "capabilities": list(self.capabilities),
            "difficulty": self.difficulty,
            "baseline_route": self.baseline_route,
            "effective_route": self.effective_route,
            "shadow_recommended_route": self.shadow_recommended_route,
            "applied": self.applied,
            "abstained": self.abstained,
            "reason_codes": list(self.reason_codes),
            "route_receipt_id": self.route_receipt_id,
            "route_receipt_sha256": self.route_receipt_sha256,
            "runtime_plan_sha256": self.runtime_plan_sha256,
            "signal_provider_config_sha256": self.signal_provider_config_sha256,
            "shadow_decision_sha256": self.shadow_decision_sha256,
            "policy_digest": self.policy_digest,
            "scorecard_digest": self.scorecard_digest,
            "bridge_config_sha256": self.bridge_config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "authorization_sha256": self.authorization_sha256,
            "operator_key_id": self.operator_key_id,
            "assignment_bucket": self.assignment_bucket,
            "canary_basis_points": self.canary_basis_points,
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["decision_sha256"] = self.decision_sha256
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "CanaryRouteDecision":
        payload = _mapping(raw, "canary decision")
        _reject_unknown(payload, _DECISION_FIELDS, "canary decision")
        if set(payload) != _DECISION_FIELDS:
            raise RouteCanaryError("Canary decision is incomplete.")
        if (
            payload["schema_version"] != CONTRACT_VERSION
            or payload["contract"] != "CanaryRouteDecision"
            or payload["mode"] != "canary"
        ):
            raise RouteCanaryError("Canary decision contract is invalid.")
        claimed = require_sha256(payload["decision_sha256"], "decision_sha256")
        data = dict(payload)
        for name in ("schema_version", "contract", "mode", "decision_sha256"):
            data.pop(name)
        decision = cls(**data)  # type: ignore[arg-type]
        if decision.decision_sha256 != claimed:
            raise RouteCanaryError("Canary decision digest is invalid.")
        return decision


def load_verified_routing_runtime_config(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> VerifiedRoutingRuntimeConfig:
    source = _absolute_path(path)
    raw = _load_json_object(source, "verified routing runtime config")
    _reject_unknown(raw, _RUNTIME_FIELDS, "verified routing runtime config")
    if set(raw) != _RUNTIME_FIELDS:
        raise RouteCanaryError("Verified routing runtime config is incomplete.")
    digest = sha256_json(raw)
    if expected_source_sha256 is not None and digest != require_sha256(
        expected_source_sha256,
        "expected runtime config digest",
    ):
        raise RouteCanaryError(
            "Verified routing runtime config changed after bridge load."
        )
    if raw["schema_version"] != CONTRACT_VERSION or raw["mode"] != "canary":
        raise RouteCanaryError("Verified routing runtime config is unsupported.")
    root = source.parent.parent
    resolved = dict(raw)
    for name in (
        "route_policy_path",
        "scorecard_path",
        "manifest_path",
        "authorization_path",
        "operator_public_key_path",
        "chronology_path",
    ):
        resolved[name] = str(_resolve_path(resolved[name], root, name))
    resolved.pop("schema_version")
    resolved.pop("mode")
    return VerifiedRoutingRuntimeConfig(
        **resolved,  # type: ignore[arg-type]
        source_sha256=digest,
    )


def load_verified_routing_canary_manifest(
    path: str | Path,
) -> VerifiedRoutingCanaryManifest:
    return VerifiedRoutingCanaryManifest.from_payload(
        _load_json_object(Path(path), "canary manifest")
    )


def load_and_verify_canary_authorization(
    path: str | Path,
    *,
    manifest: VerifiedRoutingCanaryManifest,
    runtime: VerifiedRoutingRuntimeConfig,
    bridge_config_sha256: str,
    now: str | datetime | None = None,
) -> VerifiedCanaryAuthority:
    envelope = _strict_json_loads(
        _read_bounded_regular_file(Path(path), "canary authorization")
    )
    if not isinstance(envelope, dict) or set(envelope) != {
        "payloadType",
        "payload",
        "signatures",
    }:
        raise RouteCanaryError("Canary authorization envelope is invalid.")
    if envelope["payloadType"] != AUTHORIZATION_PAYLOAD_TYPE:
        raise RouteCanaryError("Canary authorization payload type is invalid.")
    signatures = envelope["signatures"]
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise RouteCanaryError("Canary authorization requires one signature.")
    signature = _mapping(signatures[0], "canary authorization signature")
    if (
        set(signature) != {"keyid", "sig"}
        or signature.get("keyid") != runtime.operator_key_id
        or not isinstance(signature.get("sig"), str)
    ):
        raise RouteCanaryError("Canary authorization signature descriptor is invalid.")
    if not isinstance(envelope["payload"], str):
        raise RouteCanaryError("Canary authorization payload encoding is invalid.")
    payload_bytes = _decode_base64(envelope["payload"], "authorization payload")
    payload_raw = _strict_json_loads(payload_bytes)
    if not isinstance(payload_raw, dict) or canonical_json_bytes(payload_raw) != payload_bytes:
        raise RouteCanaryError("Canary authorization payload is not canonical JSON.")
    authorization = VerifiedRoutingCanaryAuthorization.from_payload(payload_raw)
    if authorization.operator_key_id != runtime.operator_key_id:
        raise RouteCanaryError("Canary authorization operator key is not trusted.")
    public_key = _load_trusted_public_key(runtime)
    try:
        public_key.verify(
            _decode_base64(signature["sig"], "authorization signature"),
            _dsse_pae(AUTHORIZATION_PAYLOAD_TYPE, payload_bytes),
        )
    except InvalidSignature as exc:
        raise RouteCanaryError("Canary authorization signature is invalid.") from exc
    current = _timestamp(now or datetime.now(timezone.utc))
    if not _timestamp(authorization.not_before) <= current < _timestamp(
        authorization.expires_at
    ):
        raise RouteCanaryError("Canary authorization is not currently active.")
    if not _timestamp(manifest.not_before) <= current < _timestamp(manifest.expires_at):
        raise RouteCanaryError("Canary manifest is not currently active.")
    if (
        authorization.manifest_sha256 != manifest.manifest_sha256
        or authorization.bridge_config_sha256
        != require_sha256(bridge_config_sha256, "bridge_config_sha256")
        or authorization.route_policy_digest
        != manifest.lineage["route_policy_digest"]
        or authorization.scorecard_digest != manifest.lineage["scorecard_digest"]
        or authorization.maximum_canary_basis_points < manifest.canary_basis_points
        or _timestamp(authorization.not_before) < _timestamp(manifest.not_before)
        or _timestamp(authorization.expires_at) > _timestamp(manifest.expires_at)
    ):
        raise RouteCanaryError("Canary authorization bindings are invalid.")
    _record_trusted_chronology(
        runtime,
        authorization=authorization,
        manifest=manifest,
        observed_at=current,
    )
    return VerifiedCanaryAuthority(
        authorization=authorization,
        verified_at=current.isoformat(),
        runtime_config_sha256=runtime.source_sha256,
        _proof=_VERIFIED_AUTHORITY_PROOF,
    )


def assignment_secret_from_environment(
    runtime: VerifiedRoutingRuntimeConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    source = os.environ if environment is None else environment
    value = source.get(runtime.assignment_secret_env)
    if not isinstance(value, str):
        raise RouteCanaryError("Canary assignment secret is unavailable.")
    secret = value.encode("utf-8")
    if not _MIN_ASSIGNMENT_SECRET_BYTES <= len(secret) <= _MAX_ASSIGNMENT_SECRET_BYTES:
        raise RouteCanaryError("Canary assignment secret size is invalid.")
    if b"\x00" in secret or b"\n" in secret or b"\r" in secret:
        raise RouteCanaryError("Canary assignment secret format is invalid.")
    return secret


def canary_assignment_bucket(secret: bytes, task_fingerprint: str) -> int:
    if not isinstance(secret, bytes) or not (
        _MIN_ASSIGNMENT_SECRET_BYTES <= len(secret) <= _MAX_ASSIGNMENT_SECRET_BYTES
    ):
        raise RouteCanaryError("Canary assignment secret size is invalid.")
    fingerprint = require_sha256(task_fingerprint, "task_fingerprint")
    digest = hmac.new(secret, fingerprint.encode("ascii"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def decide_route_canary(
    shadow: ShadowRouteDecision,
    *,
    manifest: VerifiedRoutingCanaryManifest,
    authorization: VerifiedCanaryAuthority,
    bridge_config_sha256: str,
    assignment_secret: bytes,
    evaluated_at: str | datetime,
) -> CanaryRouteDecision:
    if not isinstance(shadow, ShadowRouteDecision):
        raise RouteCanaryError("Canary evaluation requires a shadow decision.")
    if not isinstance(authorization, VerifiedCanaryAuthority):
        raise RouteCanaryError(
            "Canary evaluation requires verified operator authority."
        )
    current = _timestamp(evaluated_at)
    if current != _timestamp(authorization.verified_at):
        raise RouteCanaryError(
            "Canary authority must be consumed at its verified evaluation time."
        )
    signed = authorization.authorization
    if not (
        _timestamp(signed.not_before) <= current < _timestamp(signed.expires_at)
        and _timestamp(manifest.not_before) <= current < _timestamp(manifest.expires_at)
    ):
        raise RouteCanaryError("Canary authority is no longer active.")
    if signed.maximum_canary_basis_points < manifest.canary_basis_points:
        raise RouteCanaryError("Canary authority cap does not permit this manifest.")
    config_sha256 = require_sha256(bridge_config_sha256, "bridge_config_sha256")
    if hashlib.sha256(assignment_secret).hexdigest() != manifest.assignment_salt_sha256:
        raise RouteCanaryError("Canary assignment secret does not match the manifest.")
    if (
        shadow.policy_digest != manifest.lineage["route_policy_digest"]
        or shadow.scorecard_digest != manifest.lineage["scorecard_digest"]
        or signed.manifest_sha256 != manifest.manifest_sha256
        or signed.bridge_config_sha256 != config_sha256
    ):
        raise RouteCanaryError("Canary decision lineage is invalid.")
    bucket = canary_assignment_bucket(assignment_secret, shadow.task_fingerprint)
    reasons: list[str] = []
    candidate = next(
        (
            item
            for item in shadow.candidates
            if item.route == shadow.recommended_route
        ),
        None,
    )
    desired_key = (
        shadow.profile,
        tuple(shadow.task_signals.capabilities),
        shadow.task_signals.difficulty,
        shadow.baseline_route,
        shadow.recommended_route,
        config_sha256,
        shadow.task_signals.provider_config_sha256,
        shadow.runtime_plan_sha256,
    )
    cell = next(
        (item for item in manifest.enabled_cells if item.match_key == desired_key),
        None,
    )
    if shadow.abstained:
        reasons.append("shadow_abstained")
    if shadow.baseline_route not in ROUTE_PLANS:
        reasons.append("baseline_not_executable")
    if shadow.recommended_route not in ROUTE_PLANS or (
        shadow.baseline_route in ROUTE_PLANS
        and _ROUTE_RANK[shadow.recommended_route]
        >= _ROUTE_RANK[shadow.baseline_route]
    ):
        reasons.append("not_monotone_less_premium")
    if candidate is None or not candidate.hard_eligible or not candidate.pareto_eligible:
        reasons.append("shadow_candidate_ineligible")
    if cell is None:
        reasons.append("cell_not_enabled")
    if bucket >= manifest.canary_basis_points:
        reasons.append("outside_canary_cohort")
    if shadow.profile == "offline" and shadow.recommended_route != "local":
        reasons.append("offline_remote_forbidden")
    applied = not reasons
    return CanaryRouteDecision(
        task_fingerprint=shadow.task_fingerprint,
        profile=shadow.profile,
        capabilities=tuple(shadow.task_signals.capabilities),
        difficulty=shadow.task_signals.difficulty,
        baseline_route=shadow.baseline_route,
        effective_route=(shadow.recommended_route if applied else shadow.baseline_route),
        shadow_recommended_route=shadow.recommended_route,
        applied=applied,
        abstained=not applied,
        reason_codes=tuple(reasons or ("authorized_canary_applied",)),
        route_receipt_id=shadow.route_receipt_id,
        route_receipt_sha256=shadow.route_receipt_sha256,
        runtime_plan_sha256=shadow.runtime_plan_sha256,
        signal_provider_config_sha256=(
            shadow.task_signals.provider_config_sha256
        ),
        shadow_decision_sha256=shadow.decision_sha256,
        policy_digest=shadow.policy_digest,
        scorecard_digest=shadow.scorecard_digest,
        bridge_config_sha256=config_sha256,
        manifest_sha256=manifest.manifest_sha256,
        authorization_sha256=signed.authorization_sha256,
        operator_key_id=signed.operator_key_id,
        assignment_bucket=bucket,
        canary_basis_points=manifest.canary_basis_points,
    )


def validate_canary_receipt_binding(
    decision_or_payload: CanaryRouteDecision | Mapping[str, object],
    receipt_payload: Mapping[str, object],
) -> CanaryRouteDecision:
    """Validate a canary decision against both current and baseline receipts."""

    decision = (
        decision_or_payload
        if isinstance(decision_or_payload, CanaryRouteDecision)
        else CanaryRouteDecision.from_payload(decision_or_payload)
    )
    receipt = _mapping(receipt_payload, "route receipt")
    if receipt.get("route_canary") is None:
        raise RouteCanaryError("Route receipt is missing its canary decision.")
    embedded = CanaryRouteDecision.from_payload(
        _mapping(receipt["route_canary"], "route canary")
    )
    if embedded != decision:
        raise RouteCanaryError("Route receipt carries a different canary decision.")
    if (
        receipt.get("route") != decision.effective_route
        or receipt.get("config_sha256") != decision.bridge_config_sha256
    ):
        raise RouteCanaryError("Route canary receipt binding is invalid.")

    task = _mapping(receipt.get("task"), "route task")
    demand = _mapping(task.get("capability_demand"), "capability demand")
    task_fingerprint = require_sha256(
        task.get("task_fingerprint"),
        "task_fingerprint",
    )
    profile = require_safe_id(task.get("profile"), "profile")
    capabilities = tuple(
        sorted(
            require_identifier_tuple(
                demand.get("required"),
                "required capabilities",
            )
        )
    )
    from .route_outcomes import runtime_plan_sha256
    from .route_signals import MetadataTaskSignalProvider

    signals = MetadataTaskSignalProvider().signals_from_metadata(task)
    runtime_digest = runtime_plan_sha256(receipt)
    if (
        decision.task_fingerprint != task_fingerprint
        or decision.profile != profile
        or decision.capabilities != capabilities
        or decision.difficulty != signals.difficulty
        or decision.signal_provider_config_sha256
        != signals.provider_config_sha256
        or decision.runtime_plan_sha256 != runtime_digest
    ):
        raise RouteCanaryError("Route canary task or runtime lineage is invalid.")

    current = dict(receipt)
    claimed_current_id = require_safe_id(
        current.pop("receipt_id", None),
        "route receipt_id",
    )
    expected_current_id = f"route-{canonical_sha256(current)[:32]}"
    if claimed_current_id != expected_current_id:
        raise RouteCanaryError("Route receipt content digest is invalid.")

    expected_marker = (
        "verified_route_canary_applied"
        if decision.applied
        else "verified_route_canary_baseline_retained"
    )
    rationale = receipt.get("rationale_codes")
    if (
        not isinstance(rationale, list)
        or not rationale
        or rationale[-1] != expected_marker
    ):
        raise RouteCanaryError("Route canary rationale binding is invalid.")
    flows = {
        "local": ["local", "verify", "stop"],
        "local_then_verify": [
            "local",
            "verify",
            "stop_or_capsule",
            "premium",
            "verify",
        ],
        "premium": ["capsule", "premium", "verify"],
    }
    if receipt.get("expected_flow") != flows[decision.effective_route]:
        raise RouteCanaryError("Route canary effective flow is invalid.")

    baseline = dict(receipt)
    baseline.pop("route_canary", None)
    baseline["receipt_id"] = decision.route_receipt_id
    baseline["route"] = decision.baseline_route
    baseline["rationale_codes"] = rationale[:-1]
    baseline["expected_flow"] = flows[decision.baseline_route]
    if decision.baseline_route in {"local_then_verify", "premium"}:
        premium_runtime = _mapping(
            baseline.get("premium_runtime"),
            "premium runtime",
        )
        baseline["premium_provider"] = require_safe_id(
            premium_runtime.get("provider_id"),
            "premium provider",
        )
    else:
        baseline["premium_provider"] = None
    unsigned_baseline = dict(baseline)
    unsigned_baseline.pop("receipt_id")
    expected_baseline_id = f"route-{canonical_sha256(unsigned_baseline)[:32]}"
    if (
        decision.route_receipt_id != expected_baseline_id
        or decision.route_receipt_sha256 != canonical_sha256(baseline)
    ):
        raise RouteCanaryError("Route canary baseline receipt lineage is invalid.")
    return decision


def _record_trusted_chronology(
    runtime: VerifiedRoutingRuntimeConfig,
    *,
    authorization: VerifiedRoutingCanaryAuthorization,
    manifest: VerifiedRoutingCanaryManifest,
    observed_at: datetime,
) -> None:
    path = Path(runtime.chronology_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = FileLock(f"{path}.lock", timeout=5.0)
    try:
        with lock:
            previous: dict[str, Any] | None = None
            if path.exists():
                previous = _load_json_object(path, "canary chronology")
                expected = {
                    "schema_version",
                    "contract",
                    "operator_key_id",
                    "latest_issued_at",
                    "latest_activation_id",
                    "latest_manifest_sha256",
                    "latest_authorization_sha256",
                    "last_observed_at",
                    "state_sha256",
                }
                _reject_unknown(previous, expected, "canary chronology")
                if set(previous) != expected:
                    raise RouteCanaryError("Canary chronology is incomplete.")
                content = dict(previous)
                claimed = require_sha256(
                    content.pop("state_sha256"),
                    "chronology state_sha256",
                )
                if canonical_sha256(content) != claimed:
                    raise RouteCanaryError("Canary chronology digest is invalid.")
                if (
                    previous["schema_version"] != CONTRACT_VERSION
                    or previous["contract"] != "VerifiedRoutingCanaryChronology"
                    or previous["operator_key_id"] != runtime.operator_key_id
                ):
                    raise RouteCanaryError("Canary chronology trust binding is invalid.")
                last_observed = _timestamp(str(previous["last_observed_at"]))
                latest_issued = _timestamp(str(previous["latest_issued_at"]))
                if observed_at < last_observed:
                    raise RouteCanaryError("Canary chronology detected clock rollback.")
                issued = _timestamp(authorization.issued_at)
                if issued < latest_issued:
                    raise RouteCanaryError("Canary chronology rejected activation rollback.")
                if issued == latest_issued and (
                    previous["latest_activation_id"] != authorization.activation_id
                    or previous["latest_manifest_sha256"] != manifest.manifest_sha256
                    or previous["latest_authorization_sha256"]
                    != authorization.authorization_sha256
                ):
                    raise RouteCanaryError("Canary chronology detected signed equivocation.")
            content = {
                "schema_version": CONTRACT_VERSION,
                "contract": "VerifiedRoutingCanaryChronology",
                "operator_key_id": runtime.operator_key_id,
                "latest_issued_at": authorization.issued_at,
                "latest_activation_id": authorization.activation_id,
                "latest_manifest_sha256": manifest.manifest_sha256,
                "latest_authorization_sha256": authorization.authorization_sha256,
                "last_observed_at": observed_at.replace(microsecond=0).isoformat(),
            }
            payload = {**content, "state_sha256": canonical_sha256(content)}
            _atomic_private_json_write(path, payload)
    except Timeout as exc:
        raise RouteCanaryError("Canary chronology lock is unavailable.") from exc


def _load_trusted_public_key(
    runtime: VerifiedRoutingRuntimeConfig,
) -> Ed25519PublicKey:
    value = _read_bounded_regular_file(
        Path(runtime.operator_public_key_path),
        "operator public key",
    )
    key = load_ed25519_public_key_pem(value)
    if ed25519_public_key_sha256(key) != runtime.operator_public_key_sha256:
        raise RouteCanaryError("Operator public key does not match the trust policy.")
    return key


def _atomic_private_json_write(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload) + b"\n"
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(encoded)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("chronology write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as exc:
        raise RouteCanaryError("Canary chronology cannot be persisted safely.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve_path(value: object, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RouteCanaryError(f"{label} is required.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return _absolute_path(path)


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    raw = _strict_json_loads(_read_bounded_regular_file(path, label))
    return _mapping(raw, label)


def _read_bounded_regular_file(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RouteCanaryError(f"{label} is unavailable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RouteCanaryError(f"{label} must be a regular non-link file.")
    if not 0 < before.st_size <= _MAX_ARTIFACT_BYTES:
        raise RouteCanaryError(f"{label} size is invalid.")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_ARTIFACT_BYTES:
                    raise RouteCanaryError(f"{label} exceeds its size limit.")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RouteCanaryError(f"{label} cannot be read safely.") from exc
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise RouteCanaryError(f"{label} changed while it was read.")
    return b"".join(chunks)


def _strict_json_loads(value: bytes) -> object:
    def reject_constant(token: str) -> object:
        raise RouteCanaryError(f"Non-finite JSON number {token!r} is forbidden.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise RouteCanaryError(f"Duplicate JSON key {key!r} is forbidden.")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteCanaryError("Canary artifact is not valid JSON.") from exc


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RouteCanaryError(f"{label} must be an object with string keys.")
    return dict(value)


def _reject_unknown(raw: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise RouteCanaryError(f"Unknown {label} fields: {', '.join(unknown)}.")


def _decode_base64(value: str, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RouteCanaryError(f"{label} is not valid base64.") from exc
    if not decoded or len(decoded) > _MAX_ARTIFACT_BYTES:
        raise RouteCanaryError(f"{label} size is invalid.")
    return decoded


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


def _timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(
            require_utc_timestamp(value, "timestamp").replace("Z", "+00:00")
        )
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RouteCanaryError("Timestamp must use UTC.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


__all__ = [
    "AUTHORIZATION_PAYLOAD_TYPE",
    "CanaryRouteDecision",
    "RouteCanaryError",
    "VerifiedRoutingCanaryAuthorization",
    "VerifiedCanaryAuthority",
    "VerifiedRoutingCanaryManifest",
    "VerifiedRoutingRuntimeConfig",
    "assignment_secret_from_environment",
    "canary_assignment_bucket",
    "decide_route_canary",
    "load_and_verify_canary_authorization",
    "load_verified_routing_canary_manifest",
    "load_verified_routing_runtime_config",
    "validate_canary_receipt_binding",
]
