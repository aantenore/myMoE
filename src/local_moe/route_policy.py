from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .route_signals import (
    MetadataTaskSignalProvider,
    TaskSignalProvider,
    TaskSignals,
)
from .route_outcomes import runtime_plan_sha256
from .route_scorecard import RouteScorecard, RouteScorecardEntry
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    ROUTE_PLANS,
    VerifiedRoutingError,
    reject_unknown,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    sha256_json,
)

if TYPE_CHECKING:
    from .assistant_bridge import RouteDecisionReceipt


_PROFILE_NAMES = frozenset({"economy", "balanced", "quality", "privacy", "offline"})
_ROOT_FIELDS = {"schema_version", "mode", "normalization", "profiles"}
_PROFILE_FIELDS = {
    "weights",
    "min_success_rate",
    "min_samples",
    "min_confidence",
}
_WEIGHT_FIELDS = {"quality", "cost", "latency", "egress", "premium"}
_NORMALIZATION_FIELDS = {"cost_usd", "latency_ms", "egress_chars", "premium_calls"}
_ROUTE_ORDER = {"local": 0, "local_then_verify": 1, "premium": 2}
_DECISION_ROUTES = frozenset((*ROUTE_PLANS, "blocked"))
_CANDIDATE_FIELDS = {
    "route",
    "hard_eligible",
    "pareto_eligible",
    "utility",
    "verified_samples",
    "success_rate",
    "p95_latency_ms",
    "mean_tokens",
    "cost_sample_count",
    "mean_cost_usd",
    "mean_premium_calls",
    "mean_egress_chars",
    "rejection_codes",
}
_DECISION_FIELDS = {
    "schema_version",
    "contract",
    "mode",
    "profile",
    "route_receipt_id",
    "route_receipt_sha256",
    "runtime_plan_sha256",
    "task_fingerprint",
    "task_signals",
    "baseline_route",
    "recommended_route",
    "applied",
    "abstained",
    "reason_codes",
    "candidates",
    "policy_digest",
    "scorecard_digest",
    "decision_sha256",
}
_ROUTE_RECEIPT_FIELDS = {
    "schema_version",
    "contract",
    "receipt_id",
    "task",
    "route",
    "local_provider",
    "premium_provider",
    "local_gaps",
    "premium_gaps",
    "remote_allowed",
    "premium_call_budget",
    "rationale_codes",
    "expected_flow",
    "config_sha256",
    "workspace",
    "local_runtime",
    "premium_runtime",
}
_ROUTE_TASK_FIELDS = {
    "allow_remote",
    "allow_remote_workspace",
    "capability_demand",
    "constraint_count",
    "max_premium_calls",
    "no_change_expected",
    "objective_chars",
    "objective_sha256",
    "profile",
    "required_verifier_ids",
    "task_fingerprint",
    "task_id",
}
_ROUTE_DEMAND_FIELDS = {"required", "risk_class", "tools"}
_BRIDGE_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class RoutePolicyWeights:
    quality: float
    cost: float
    latency: float
    egress: float
    premium: float

    def __post_init__(self) -> None:
        for name in _WEIGHT_FIELDS:
            object.__setattr__(
                self,
                name,
                _weight(getattr(self, name), f"policy weight {name}"),
            )
        if self.total <= 0.0:
            raise VerifiedRoutingError("Policy weights must have a positive sum.")

    @property
    def total(self) -> float:
        return self.quality + self.cost + self.latency + self.egress + self.premium

    def payload(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "cost": self.cost,
            "latency": self.latency,
            "egress": self.egress,
            "premium": self.premium,
        }


@dataclass(frozen=True)
class RouteProfilePolicy:
    weights: RoutePolicyWeights
    min_success_rate: float
    min_samples: int
    min_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.weights, RoutePolicyWeights):
            raise VerifiedRoutingError("Profile weights are invalid.")
        object.__setattr__(
            self,
            "weights",
            RoutePolicyWeights(**self.weights.payload()),
        )
        object.__setattr__(
            self,
            "min_success_rate",
            require_finite_number(
                self.min_success_rate,
                "profile min_success_rate",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        min_samples = require_non_negative_int(
            self.min_samples,
            "profile min_samples",
        )
        if min_samples == 0:
            raise VerifiedRoutingError("Profile min_samples must be positive.")
        object.__setattr__(self, "min_samples", min_samples)
        object.__setattr__(
            self,
            "min_confidence",
            require_finite_number(
                self.min_confidence,
                "profile min_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "weights": self.weights.payload(),
            "min_success_rate": self.min_success_rate,
            "min_samples": self.min_samples,
            "min_confidence": self.min_confidence,
        }


@dataclass(frozen=True)
class RoutePolicyNormalization:
    cost_usd: float
    latency_ms: float
    egress_chars: float
    premium_calls: float

    def __post_init__(self) -> None:
        for name in _NORMALIZATION_FIELDS:
            object.__setattr__(
                self,
                name,
                _positive_number(
                    getattr(self, name),
                    f"policy normalization {name}",
                ),
            )

    def payload(self) -> dict[str, float]:
        return {
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "egress_chars": self.egress_chars,
            "premium_calls": self.premium_calls,
        }


@dataclass(frozen=True)
class VerifiedRoutePolicy:
    profiles: Mapping[str, RouteProfilePolicy]
    normalization: RoutePolicyNormalization
    digest: str
    mode: str = "shadow"
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported route policy schema_version.")
        if self.mode != "shadow":
            raise VerifiedRoutingError("Route policy mode must be shadow in schema 1.0.")
        if not isinstance(self.profiles, Mapping):
            raise VerifiedRoutingError("profiles must be a mapping.")
        if set(self.profiles) != _PROFILE_NAMES:
            raise VerifiedRoutingError(
                "profiles must define economy, balanced, quality, privacy, and offline."
            )
        profiles: dict[str, RouteProfilePolicy] = {}
        for name in sorted(self.profiles):
            profile = self.profiles[name]
            if not isinstance(profile, RouteProfilePolicy):
                raise VerifiedRoutingError(f"profiles.{name} is invalid.")
            profiles[name] = RouteProfilePolicy(
                weights=RoutePolicyWeights(**profile.weights.payload()),
                min_success_rate=profile.min_success_rate,
                min_samples=profile.min_samples,
                min_confidence=profile.min_confidence,
            )
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        if not isinstance(self.normalization, RoutePolicyNormalization):
            raise VerifiedRoutingError("normalization is invalid.")
        object.__setattr__(
            self,
            "normalization",
            RoutePolicyNormalization(**self.normalization.payload()),
        )
        digest = require_sha256(self.digest, "route policy digest")
        object.__setattr__(self, "digest", digest)
        if digest != sha256_json(self.payload()):
            raise VerifiedRoutingError("Route policy digest does not match its content.")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "normalization": self.normalization.payload(),
            "profiles": {
                name: self.profiles[name].payload() for name in sorted(self.profiles)
            },
        }


@dataclass(frozen=True)
class CandidateRouteScore:
    route: str
    hard_eligible: bool
    pareto_eligible: bool
    utility: float | None
    verified_samples: int | None
    success_rate: float | None
    p95_latency_ms: float | None
    mean_tokens: float | None
    cost_sample_count: int | None
    mean_cost_usd: float | None
    mean_premium_calls: float | None
    mean_egress_chars: float | None
    rejection_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Candidate route is not supported.")
        for name in ("hard_eligible", "pareto_eligible"):
            if not isinstance(getattr(self, name), bool):
                raise VerifiedRoutingError(f"Candidate {name} must be boolean.")
        object.__setattr__(
            self,
            "utility",
            _optional_metric(self.utility, "candidate utility"),
        )
        for name in ("verified_samples", "cost_sample_count"):
            value = getattr(self, name)
            if value is not None:
                value = require_non_negative_int(value, f"candidate {name}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "success_rate",
            _optional_metric(
                self.success_rate,
                "candidate success_rate",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for name in (
            "p95_latency_ms",
            "mean_tokens",
            "mean_cost_usd",
            "mean_premium_calls",
            "mean_egress_chars",
        ):
            object.__setattr__(
                self,
                name,
                _optional_metric(getattr(self, name), f"candidate {name}", minimum=0.0),
            )
        rejections = require_identifier_tuple(
            self.rejection_codes,
            "candidate rejection_codes",
        )
        object.__setattr__(self, "rejection_codes", rejections)
        if self.verified_samples is None:
            if any(
                getattr(self, name) is not None
                for name in (
                    "success_rate",
                    "p95_latency_ms",
                    "mean_tokens",
                    "cost_sample_count",
                    "mean_cost_usd",
                    "mean_premium_calls",
                    "mean_egress_chars",
                )
            ):
                raise VerifiedRoutingError(
                    "Candidate metrics require verified sample evidence."
                )
        else:
            if self.verified_samples == 0:
                raise VerifiedRoutingError(
                    "Candidate verified_samples must be positive when present."
                )
            required_metrics = (
                self.success_rate,
                self.p95_latency_ms,
                self.mean_tokens,
                self.cost_sample_count,
                self.mean_premium_calls,
                self.mean_egress_chars,
            )
            if any(value is None for value in required_metrics):
                raise VerifiedRoutingError(
                    "Candidate sample evidence requires complete metrics."
                )
            if self.cost_sample_count > self.verified_samples:
                raise VerifiedRoutingError(
                    "Candidate cost samples cannot exceed verified samples."
                )
            if (self.cost_sample_count == 0) != (self.mean_cost_usd is None):
                raise VerifiedRoutingError(
                    "Candidate mean cost must be present exactly when cost samples exist."
                )
        if self.pareto_eligible and (
            not self.hard_eligible
            or self.utility is None
            or self.rejection_codes
        ):
            raise VerifiedRoutingError(
                "Pareto candidates must be hard-eligible, scored, and unrejected."
            )
        if not self.hard_eligible and (
            self.pareto_eligible or self.utility is not None or not self.rejection_codes
        ):
            raise VerifiedRoutingError(
                "Hard-ineligible candidates must be rejected and unscored."
            )

    def payload(self) -> dict[str, object]:
        return {
            "route": self.route,
            "hard_eligible": self.hard_eligible,
            "pareto_eligible": self.pareto_eligible,
            "utility": self.utility,
            "verified_samples": self.verified_samples,
            "success_rate": self.success_rate,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_tokens": self.mean_tokens,
            "cost_sample_count": self.cost_sample_count,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_premium_calls": self.mean_premium_calls,
            "mean_egress_chars": self.mean_egress_chars,
            "rejection_codes": list(self.rejection_codes),
        }


@dataclass(frozen=True)
class ShadowRouteDecision:
    profile: str
    route_receipt_id: str
    route_receipt_sha256: str
    runtime_plan_sha256: str
    task_fingerprint: str
    task_signals: TaskSignals
    baseline_route: str
    recommended_route: str
    applied: bool
    abstained: bool
    reason_codes: tuple[str, ...]
    candidates: tuple[CandidateRouteScore, ...]
    policy_digest: str
    scorecard_digest: str
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_NAMES:
            raise VerifiedRoutingError("Shadow decision profile is unsupported.")
        require_safe_id(self.route_receipt_id, "route_receipt_id")
        for name in (
            "route_receipt_sha256",
            "runtime_plan_sha256",
            "task_fingerprint",
            "policy_digest",
            "scorecard_digest",
        ):
            require_sha256(getattr(self, name), name)
        if not isinstance(self.task_signals, TaskSignals):
            raise VerifiedRoutingError("task_signals must be a TaskSignals contract.")
        if self.task_signals.request_fingerprint != self.task_fingerprint:
            raise VerifiedRoutingError("Shadow decision task signals are not task-bound.")
        if self.baseline_route not in _DECISION_ROUTES:
            raise VerifiedRoutingError("Shadow decision baseline route is unsupported.")
        if self.recommended_route not in _DECISION_ROUTES:
            raise VerifiedRoutingError("Shadow decision recommendation is unsupported.")
        if self.applied is not False:
            raise VerifiedRoutingError("Shadow decisions can never apply a route.")
        if not isinstance(self.abstained, bool):
            raise VerifiedRoutingError("Shadow decision abstained must be boolean.")
        reasons = require_identifier_tuple(self.reason_codes, "reason_codes")
        object.__setattr__(self, "reason_codes", reasons)
        candidates = tuple(self.candidates)
        if any(not isinstance(item, CandidateRouteScore) for item in candidates):
            raise VerifiedRoutingError("Shadow decision candidates are invalid.")
        routes = tuple(item.route for item in candidates)
        if len(routes) != len(set(routes)):
            raise VerifiedRoutingError("Shadow decision candidate routes must be unique.")
        if routes != tuple(sorted(routes, key=_ROUTE_ORDER.__getitem__)):
            raise VerifiedRoutingError("Shadow decision candidates must be canonical.")
        if self.baseline_route == "blocked":
            if (
                self.recommended_route != "blocked"
                or not self.abstained
                or candidates
            ):
                raise VerifiedRoutingError(
                    "Blocked decisions must abstain, remain blocked, and contain no candidates."
                )
        else:
            if self.recommended_route == "blocked":
                raise VerifiedRoutingError(
                    "Executable receipt decisions cannot recommend blocked."
                )
            if routes != ROUTE_PLANS:
                raise VerifiedRoutingError(
                    "Executable receipt decisions must contain every route candidate."
                )
            if self.abstained and self.recommended_route != self.baseline_route:
                raise VerifiedRoutingError(
                    "Abstained decisions must retain the baseline route."
                )
            if not self.abstained:
                selected = next(
                    item
                    for item in candidates
                    if item.route == self.recommended_route
                )
                if (
                    not selected.hard_eligible
                    or not selected.pareto_eligible
                    or selected.utility is None
                    or selected.rejection_codes
                ):
                    raise VerifiedRoutingError(
                        "A non-abstained recommendation must select an eligible Pareto candidate."
                    )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "decision_sha256",
            sha256_json(self.content_payload()),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "contract": "ShadowRouteDecision",
            "mode": "shadow",
            "profile": self.profile,
            "route_receipt_id": self.route_receipt_id,
            "route_receipt_sha256": self.route_receipt_sha256,
            "runtime_plan_sha256": self.runtime_plan_sha256,
            "task_fingerprint": self.task_fingerprint,
            "task_signals": self.task_signals.payload(),
            "baseline_route": self.baseline_route,
            "recommended_route": self.recommended_route,
            "applied": self.applied,
            "abstained": self.abstained,
            "reason_codes": list(self.reason_codes),
            "candidates": [candidate.payload() for candidate in self.candidates],
            "policy_digest": self.policy_digest,
            "scorecard_digest": self.scorecard_digest,
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["decision_sha256"] = self.decision_sha256
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "ShadowRouteDecision":
        payload = _strict_mapping(raw, "ShadowRouteDecision")
        reject_unknown(payload, _DECISION_FIELDS, "ShadowRouteDecision")
        missing = sorted(_DECISION_FIELDS.difference(payload))
        if missing:
            raise VerifiedRoutingError(
                f"Missing ShadowRouteDecision fields: {', '.join(missing)}."
            )
        if payload["schema_version"] != CONTRACT_VERSION:
            raise VerifiedRoutingError(
                "ShadowRouteDecision schema_version is unsupported."
            )
        if payload["contract"] != "ShadowRouteDecision" or payload["mode"] != "shadow":
            raise VerifiedRoutingError("ShadowRouteDecision contract is invalid.")
        signals = TaskSignals.from_payload(
            _strict_mapping(payload["task_signals"], "task_signals")
        )
        candidates_raw = payload["candidates"]
        if not isinstance(candidates_raw, list):
            raise VerifiedRoutingError("ShadowRouteDecision candidates must be a list.")
        reasons_raw = payload["reason_codes"]
        if not isinstance(reasons_raw, list):
            raise VerifiedRoutingError("ShadowRouteDecision reason_codes must be a list.")
        decision = cls(
            profile=payload["profile"],  # type: ignore[arg-type]
            route_receipt_id=payload["route_receipt_id"],  # type: ignore[arg-type]
            route_receipt_sha256=payload["route_receipt_sha256"],  # type: ignore[arg-type]
            runtime_plan_sha256=payload["runtime_plan_sha256"],  # type: ignore[arg-type]
            task_fingerprint=payload["task_fingerprint"],  # type: ignore[arg-type]
            task_signals=signals,
            baseline_route=payload["baseline_route"],  # type: ignore[arg-type]
            recommended_route=payload["recommended_route"],  # type: ignore[arg-type]
            applied=payload["applied"],  # type: ignore[arg-type]
            abstained=payload["abstained"],  # type: ignore[arg-type]
            reason_codes=tuple(reasons_raw),  # type: ignore[arg-type]
            candidates=tuple(_candidate_from_payload(item) for item in candidates_raw),
            policy_digest=payload["policy_digest"],  # type: ignore[arg-type]
            scorecard_digest=payload["scorecard_digest"],  # type: ignore[arg-type]
        )
        claimed = require_sha256(payload["decision_sha256"], "decision_sha256")
        if decision.decision_sha256 != claimed:
            raise VerifiedRoutingError(
                "ShadowRouteDecision digest does not match its content."
            )
        return decision


@dataclass(frozen=True)
class _DecisionLineage:
    route_receipt_id: str
    route_receipt_sha256: str
    runtime_plan_sha256: str
    task_fingerprint: str
    task_signals: TaskSignals


def load_route_policy(path: str | Path) -> VerifiedRoutePolicy:
    raw = _load_json(Path(path))
    if not isinstance(raw, dict):
        raise VerifiedRoutingError("Route policy must be a JSON object.")
    return route_policy_from_payload(raw)


def route_policy_from_payload(raw: Mapping[str, object]) -> VerifiedRoutePolicy:
    data = dict(raw)
    reject_unknown(data, _ROOT_FIELDS, "route policy")
    if data.get("schema_version") != CONTRACT_VERSION:
        raise VerifiedRoutingError("Unsupported route policy schema_version.")
    if data.get("mode") != "shadow":
        raise VerifiedRoutingError("Route policy mode must be shadow in schema 1.0.")

    normalization_raw = data.get("normalization")
    if not isinstance(normalization_raw, dict):
        raise VerifiedRoutingError("normalization must be an object.")
    reject_unknown(normalization_raw, _NORMALIZATION_FIELDS, "normalization")
    if set(normalization_raw) != _NORMALIZATION_FIELDS:
        raise VerifiedRoutingError("normalization fields are incomplete.")
    normalization = RoutePolicyNormalization(
        cost_usd=normalization_raw["cost_usd"],  # type: ignore[arg-type]
        latency_ms=normalization_raw["latency_ms"],  # type: ignore[arg-type]
        egress_chars=normalization_raw["egress_chars"],  # type: ignore[arg-type]
        premium_calls=normalization_raw["premium_calls"],  # type: ignore[arg-type]
    )

    profiles_raw = data.get("profiles")
    if not isinstance(profiles_raw, dict):
        raise VerifiedRoutingError("profiles must be an object.")
    if set(profiles_raw) != _PROFILE_NAMES:
        raise VerifiedRoutingError(
            "profiles must define economy, balanced, quality, privacy, and offline."
        )
    profiles: dict[str, RouteProfilePolicy] = {}
    for name in sorted(profiles_raw):
        profile_raw = profiles_raw[name]
        if not isinstance(profile_raw, dict):
            raise VerifiedRoutingError(f"profiles.{name} must be an object.")
        reject_unknown(profile_raw, _PROFILE_FIELDS, f"profiles.{name}")
        if set(profile_raw) != _PROFILE_FIELDS:
            raise VerifiedRoutingError(f"profiles.{name} fields are incomplete.")
        weights_raw = profile_raw["weights"]
        if not isinstance(weights_raw, dict):
            raise VerifiedRoutingError(f"profiles.{name}.weights must be an object.")
        reject_unknown(weights_raw, _WEIGHT_FIELDS, f"profiles.{name}.weights")
        if set(weights_raw) != _WEIGHT_FIELDS:
            raise VerifiedRoutingError(f"profiles.{name}.weights fields are incomplete.")
        weights = RoutePolicyWeights(
            quality=weights_raw["quality"],  # type: ignore[arg-type]
            cost=weights_raw["cost"],  # type: ignore[arg-type]
            latency=weights_raw["latency"],  # type: ignore[arg-type]
            egress=weights_raw["egress"],  # type: ignore[arg-type]
            premium=weights_raw["premium"],  # type: ignore[arg-type]
        )
        profiles[name] = RouteProfilePolicy(
            weights=weights,
            min_success_rate=profile_raw["min_success_rate"],  # type: ignore[arg-type]
            min_samples=profile_raw["min_samples"],  # type: ignore[arg-type]
            min_confidence=profile_raw["min_confidence"],  # type: ignore[arg-type]
        )

    normalized = {
        "schema_version": CONTRACT_VERSION,
        "mode": "shadow",
        "normalization": normalization.payload(),
        "profiles": {name: profiles[name].payload() for name in sorted(profiles)},
    }
    return VerifiedRoutePolicy(
        profiles=profiles,
        normalization=normalization,
        digest=sha256_json(normalized),
    )


def recommend_shadow_route(
    receipt: "RouteDecisionReceipt",
    signals: "TaskSignals",
    scorecard: RouteScorecard,
    policy: VerifiedRoutePolicy,
    *,
    profile: str,
    now: str | datetime | None = None,
    signal_provider: TaskSignalProvider | None = None,
) -> ShadowRouteDecision:
    if profile not in policy.profiles:
        raise VerifiedRoutingError("Unknown route policy profile.")
    receipt_payload = _route_receipt_payload(receipt)
    task = _strict_mapping(receipt_payload["task"], "route receipt task")
    _validate_receipt_view(receipt, receipt_payload)
    if not isinstance(signals, TaskSignals):
        raise VerifiedRoutingError("signals must be a TaskSignals contract.")
    task_fingerprint = require_sha256(
        task.get("task_fingerprint"),
        "route receipt task_fingerprint",
    )
    demand = _strict_mapping(
        task.get("capability_demand"),
        "route receipt capability_demand",
    )
    required_capabilities = tuple(
        sorted(
            require_identifier_tuple(
                demand.get("required"),
                "route receipt capability_demand.required",
            )
        )
    )
    if required_capabilities != signals.capabilities:
        raise VerifiedRoutingError(
            "Task signal capabilities do not exactly match the route receipt."
        )
    selected_provider = signal_provider or MetadataTaskSignalProvider()
    expected_signals = selected_provider.signals_from_metadata(task)
    if expected_signals.payload() != signals.payload():
        raise VerifiedRoutingError(
            "Task signals do not match the configured signal provider."
        )
    lineage = _DecisionLineage(
        route_receipt_id=require_safe_id(
            receipt_payload["receipt_id"],
            "route receipt_id",
        ),
        route_receipt_sha256=sha256_json(receipt_payload),
        runtime_plan_sha256=runtime_plan_sha256(receipt_payload),
        task_fingerprint=task_fingerprint,
        task_signals=signals,
    )
    baseline = str(receipt.route)
    if str(task.get("profile", "")) != profile:
        raise VerifiedRoutingError("Route policy profile does not match the receipt.")
    if task_fingerprint != signals.request_fingerprint:
        raise VerifiedRoutingError("Task signals do not match the receipt.")
    if baseline == "blocked":
        return _decision(
            profile=profile,
            baseline=baseline,
            recommended=baseline,
            abstained=True,
            reasons=["shadow_mode", "receipt_blocked", "baseline_retained"],
            candidates=[],
            policy=policy,
            scorecard=scorecard,
            lineage=lineage,
        )
    if baseline not in ROUTE_PLANS:
        raise VerifiedRoutingError("Receipt route is not supported.")
    profile_policy = policy.profiles[profile]
    if profile == "offline" and baseline in {"local_then_verify", "premium"}:
        raise VerifiedRoutingError("Offline receipts cannot contain a remote route.")
    hard_rejections = _hard_rejections(receipt, profile=profile)

    stale = _is_stale(scorecard, now)
    global_reasons: list[str] = ["shadow_mode"]
    evidence_abstain = False
    if bool(signals.abstained):
        evidence_abstain = True
        global_reasons.append("signal_abstained")
    if float(signals.confidence) < profile_policy.min_confidence:
        evidence_abstain = True
        global_reasons.append("signal_confidence_below_profile")
    if stale:
        evidence_abstain = True
        global_reasons.append("scorecard_stale")

    scored: list[CandidateRouteScore] = []
    for route in ROUTE_PLANS:
        rejections = list(hard_rejections[route])
        entry: RouteScorecardEntry | None = None
        if not rejections and not stale:
            entry = scorecard.conservative_entry(
                config_sha256=str(receipt.config_sha256),
                signal_provider_config_sha256=signals.provider_config_sha256,
                runtime_plan_sha256=lineage.runtime_plan_sha256,
                route=route,
                capabilities=tuple(signals.capabilities),
                difficulty=str(signals.difficulty),
            )
            if entry is None:
                rejections.append("scorecard_out_of_distribution")
                evidence_abstain = True
            elif entry.verified_samples < profile_policy.min_samples:
                rejections.append("insufficient_verified_samples")
                evidence_abstain = True
            elif (
                profile_policy.weights.cost > 0.0
                and (
                    entry.mean_cost_usd is None
                    or entry.cost_sample_count < entry.verified_samples
                )
            ):
                rejections.append("incomplete_cost_evidence")
                evidence_abstain = True
            elif entry.success_rate < profile_policy.min_success_rate:
                rejections.append("success_below_profile_minimum")
        scored.append(
            _candidate_score(
                route=route,
                entry=entry,
                hard_eligible=not hard_rejections[route],
                rejections=tuple(rejections),
                profile=profile_policy,
                normalization=policy.normalization,
            )
        )

    if evidence_abstain:
        if any(
            "scorecard_out_of_distribution" in candidate.rejection_codes
            for candidate in scored
        ):
            global_reasons.append("scorecard_out_of_distribution")
        if any(
            "insufficient_verified_samples" in candidate.rejection_codes
            for candidate in scored
        ):
            global_reasons.append("insufficient_verified_samples")
        if any(
            "incomplete_cost_evidence" in candidate.rejection_codes
            for candidate in scored
        ):
            global_reasons.append("incomplete_cost_evidence")
        global_reasons.append("baseline_retained")
        return _decision(
            profile=profile,
            baseline=baseline,
            recommended=baseline,
            abstained=True,
            reasons=global_reasons,
            candidates=scored,
            policy=policy,
            scorecard=scorecard,
            lineage=lineage,
        )

    acceptable = [candidate for candidate in scored if not candidate.rejection_codes]
    if not acceptable:
        global_reasons.extend(("no_candidate_meets_success_floor", "baseline_retained"))
        return _decision(
            profile=profile,
            baseline=baseline,
            recommended=baseline,
            abstained=True,
            reasons=global_reasons,
            candidates=scored,
            policy=policy,
            scorecard=scorecard,
            lineage=lineage,
        )

    pareto_routes = _pareto_routes(acceptable, profile_policy.weights)
    rescored = [
        _with_pareto(candidate, candidate.route in pareto_routes) for candidate in scored
    ]
    finalists = [
        candidate
        for candidate in rescored
        if candidate.route in pareto_routes and candidate.utility is not None
    ]
    winner = sorted(
        finalists,
        key=lambda candidate: (
            -float(candidate.utility),
            0 if candidate.route == baseline else 1,
            _ROUTE_ORDER[candidate.route],
        ),
    )[0]
    global_reasons.append(
        "baseline_retained" if winner.route == baseline else "shadow_recommendation_available"
    )
    return _decision(
        profile=profile,
        baseline=baseline,
        recommended=winner.route,
        abstained=False,
        reasons=global_reasons,
        candidates=rescored,
        policy=policy,
        scorecard=scorecard,
        lineage=lineage,
    )


def _hard_rejections(
    receipt: "RouteDecisionReceipt",
    *,
    profile: str,
) -> dict[str, tuple[str, ...]]:
    local_gaps = tuple(receipt.local_gaps)
    premium_gaps = tuple(receipt.premium_gaps)
    remote_allowed = bool(receipt.remote_allowed)
    premium_budget = int(receipt.premium_call_budget)
    result: dict[str, tuple[str, ...]] = {}
    for route in ROUTE_PLANS:
        reasons: list[str] = []
        if profile == "offline" and route in {"local_then_verify", "premium"}:
            reasons.append("offline_remote_forbidden")
        if route in {"local", "local_then_verify"} and local_gaps:
            reasons.append("local_capability_gap")
        if route in {"local_then_verify", "premium"}:
            if premium_gaps:
                reasons.append("premium_capability_gap")
            if not remote_allowed:
                reasons.append("remote_not_allowed")
            if premium_budget <= 0:
                reasons.append("premium_budget_unavailable")
        result[route] = tuple(reasons)
    return result


def _candidate_score(
    *,
    route: str,
    entry: RouteScorecardEntry | None,
    hard_eligible: bool,
    rejections: tuple[str, ...],
    profile: RouteProfilePolicy,
    normalization: RoutePolicyNormalization,
) -> CandidateRouteScore:
    utility = None
    if entry is not None and not rejections:
        utility = _utility(entry, profile, normalization)
    return CandidateRouteScore(
        route=route,
        hard_eligible=hard_eligible,
        pareto_eligible=False,
        utility=utility,
        verified_samples=None if entry is None else entry.verified_samples,
        success_rate=None if entry is None else entry.success_rate,
        p95_latency_ms=None if entry is None else entry.p95_latency_ms,
        mean_tokens=None if entry is None else entry.mean_tokens,
        cost_sample_count=None if entry is None else entry.cost_sample_count,
        mean_cost_usd=None if entry is None else entry.mean_cost_usd,
        mean_premium_calls=None if entry is None else entry.mean_premium_calls,
        mean_egress_chars=None if entry is None else entry.mean_egress_chars,
        rejection_codes=rejections,
    )


def _utility(
    entry: RouteScorecardEntry,
    profile: RouteProfilePolicy,
    normalization: RoutePolicyNormalization,
) -> float:
    weights = profile.weights
    if weights.cost > 0.0 and entry.mean_cost_usd is None:
        raise VerifiedRoutingError("Cost-weighted utility requires complete cost evidence.")
    cost_usd = 0.0 if entry.mean_cost_usd is None else entry.mean_cost_usd
    reward = weights.quality * entry.success_rate
    penalties = (
        weights.cost * _unit(cost_usd / normalization.cost_usd)
        + weights.latency * _unit(entry.p95_latency_ms / normalization.latency_ms)
        + weights.egress * _unit(entry.mean_egress_chars / normalization.egress_chars)
        + weights.premium
        * _unit(entry.mean_premium_calls / normalization.premium_calls)
    )
    return round((reward - penalties) / weights.total, 12)


def _pareto_routes(
    candidates: list[CandidateRouteScore],
    weights: RoutePolicyWeights,
) -> frozenset[str]:
    nondominated: set[str] = set()
    for candidate in candidates:
        dominated = any(
            other.route != candidate.route
            and _dominates(other, candidate, weights)
            for other in candidates
        )
        if not dominated:
            nondominated.add(candidate.route)
    return frozenset(nondominated)


def _dominates(
    left: CandidateRouteScore,
    right: CandidateRouteScore,
    weights: RoutePolicyWeights,
) -> bool:
    left_values = _metric_vector(left, weights)
    right_values = _metric_vector(right, weights)
    return all(
        left_value >= right_value
        for left_value, right_value in zip(left_values, right_values)
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(left_values, right_values)
    )


def _metric_vector(
    candidate: CandidateRouteScore,
    weights: RoutePolicyWeights,
) -> tuple[float, ...]:
    values: list[float] = []
    if weights.quality > 0.0:
        values.append(_required_metric(candidate.success_rate, "success_rate"))
    if weights.cost > 0.0:
        values.append(-_required_metric(candidate.mean_cost_usd, "mean_cost_usd"))
    if weights.latency > 0.0:
        values.append(-_required_metric(candidate.p95_latency_ms, "p95_latency_ms"))
    if weights.egress > 0.0:
        values.append(-_required_metric(candidate.mean_egress_chars, "mean_egress_chars"))
    if weights.premium > 0.0:
        values.append(
            -_required_metric(candidate.mean_premium_calls, "mean_premium_calls")
        )
    return tuple(values)


def _required_metric(value: float | None, label: str) -> float:
    if value is None:
        raise VerifiedRoutingError(f"Pareto filtering requires {label} evidence.")
    return value


def _with_pareto(
    candidate: CandidateRouteScore,
    pareto_eligible: bool,
) -> CandidateRouteScore:
    rejections = candidate.rejection_codes
    if not pareto_eligible and not rejections and candidate.utility is not None:
        rejections = ("pareto_dominated",)
    return CandidateRouteScore(
        route=candidate.route,
        hard_eligible=candidate.hard_eligible,
        pareto_eligible=pareto_eligible,
        utility=candidate.utility,
        verified_samples=candidate.verified_samples,
        success_rate=candidate.success_rate,
        p95_latency_ms=candidate.p95_latency_ms,
        mean_tokens=candidate.mean_tokens,
        cost_sample_count=candidate.cost_sample_count,
        mean_cost_usd=candidate.mean_cost_usd,
        mean_premium_calls=candidate.mean_premium_calls,
        mean_egress_chars=candidate.mean_egress_chars,
        rejection_codes=rejections,
    )


def _decision(
    *,
    profile: str,
    baseline: str,
    recommended: str,
    abstained: bool,
    reasons: list[str],
    candidates: list[CandidateRouteScore],
    policy: VerifiedRoutePolicy,
    scorecard: RouteScorecard,
    lineage: _DecisionLineage,
) -> ShadowRouteDecision:
    return ShadowRouteDecision(
        profile=profile,
        route_receipt_id=lineage.route_receipt_id,
        route_receipt_sha256=lineage.route_receipt_sha256,
        runtime_plan_sha256=lineage.runtime_plan_sha256,
        task_fingerprint=lineage.task_fingerprint,
        task_signals=lineage.task_signals,
        baseline_route=baseline,
        recommended_route=recommended,
        applied=False,
        abstained=abstained,
        reason_codes=tuple(dict.fromkeys(reasons)),
        candidates=tuple(sorted(candidates, key=lambda item: _ROUTE_ORDER[item.route])),
        policy_digest=policy.digest,
        scorecard_digest=scorecard.digest,
    )


def _route_receipt_payload(receipt: object) -> dict[str, object]:
    payload_method = getattr(receipt, "payload", None)
    if callable(payload_method):
        raw = payload_method()
    else:
        raw = getattr(receipt, "raw_payload", None)
    payload = _strict_mapping(raw, "route receipt")
    reject_unknown(payload, _ROUTE_RECEIPT_FIELDS, "route receipt")
    missing = sorted(_ROUTE_RECEIPT_FIELDS.difference(payload))
    if missing:
        raise VerifiedRoutingError(
            f"Missing route receipt fields: {', '.join(missing)}."
        )
    if payload["schema_version"] != _BRIDGE_SCHEMA_VERSION:
        raise VerifiedRoutingError("Route receipt schema_version is unsupported.")
    if payload["contract"] != "RouteDecisionReceipt":
        raise VerifiedRoutingError("Route receipt contract is invalid.")
    require_safe_id(payload["receipt_id"], "route receipt_id")
    require_sha256(payload["config_sha256"], "route receipt config_sha256")
    payload["task"] = _validated_route_task(payload["task"])
    _strict_mapping(payload["workspace"], "route receipt workspace")
    runtime_plan_sha256(payload)
    return payload


def _validate_receipt_view(
    receipt: object,
    payload: Mapping[str, object],
) -> None:
    expected = {
        "receipt_id": payload["receipt_id"],
        "route": payload["route"],
        "config_sha256": payload["config_sha256"],
        "remote_allowed": payload["remote_allowed"],
        "premium_call_budget": payload["premium_call_budget"],
    }
    for name, value in expected.items():
        if getattr(receipt, name, None) != value:
            raise VerifiedRoutingError(
                f"Route receipt {name} view does not match its payload."
            )
    for name in ("local_gaps", "premium_gaps"):
        raw = payload[name]
        if not isinstance(raw, list):
            raise VerifiedRoutingError(f"Route receipt {name} must be a list.")
        values = require_identifier_tuple(raw, f"route receipt {name}")
        if tuple(getattr(receipt, name, ())) != values:
            raise VerifiedRoutingError(
                f"Route receipt {name} view does not match its payload."
            )
    if not isinstance(payload["remote_allowed"], bool):
        raise VerifiedRoutingError("Route receipt remote_allowed must be boolean.")
    require_non_negative_int(
        payload["premium_call_budget"],
        "route receipt premium_call_budget",
    )


def _strict_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise VerifiedRoutingError(f"{label} keys must be strings.")
    return dict(value)


def _validated_route_task(value: object) -> dict[str, object]:
    task = _strict_mapping(value, "route receipt task")
    reject_unknown(task, _ROUTE_TASK_FIELDS, "route receipt task")
    missing = sorted(_ROUTE_TASK_FIELDS.difference(task))
    if missing:
        raise VerifiedRoutingError(
            f"Missing route receipt task fields: {', '.join(missing)}."
        )
    require_safe_id(task["task_id"], "route receipt task_id")
    require_sha256(task["objective_sha256"], "route receipt objective_sha256")
    require_sha256(task["task_fingerprint"], "route receipt task_fingerprint")
    require_non_negative_int(
        task["objective_chars"],
        "route receipt objective_chars",
    )
    require_non_negative_int(
        task["constraint_count"],
        "route receipt constraint_count",
    )
    if task["profile"] not in _PROFILE_NAMES:
        raise VerifiedRoutingError("Route receipt task profile is unsupported.")
    verifiers = task["required_verifier_ids"]
    if not isinstance(verifiers, list) or any(
        not isinstance(item, str) for item in verifiers
    ):
        raise VerifiedRoutingError(
            "Route receipt required_verifier_ids must be a string list."
        )
    require_identifier_tuple(verifiers, "route receipt required_verifier_ids")
    for name in ("no_change_expected", "allow_remote_workspace"):
        if not isinstance(task[name], bool):
            raise VerifiedRoutingError(f"Route receipt {name} must be boolean.")
    if task["allow_remote"] is not None and not isinstance(
        task["allow_remote"], bool
    ):
        raise VerifiedRoutingError(
            "Route receipt allow_remote must be boolean or null."
        )
    if task["allow_remote_workspace"] and task["allow_remote"] is not True:
        raise VerifiedRoutingError(
            "Route receipt allow_remote_workspace requires allow_remote=true."
        )
    max_premium_calls = task["max_premium_calls"]
    if max_premium_calls is not None:
        maximum = require_non_negative_int(
            max_premium_calls,
            "route receipt max_premium_calls",
        )
        if maximum > 1:
            raise VerifiedRoutingError(
                "Route receipt max_premium_calls cannot exceed one."
            )
    demand = _strict_mapping(
        task["capability_demand"],
        "route receipt capability_demand",
    )
    reject_unknown(demand, _ROUTE_DEMAND_FIELDS, "route receipt capability_demand")
    demand_missing = sorted(_ROUTE_DEMAND_FIELDS.difference(demand))
    if demand_missing:
        raise VerifiedRoutingError(
            "Missing route receipt capability_demand fields: "
            + ", ".join(demand_missing)
            + "."
        )
    for name in ("required", "tools"):
        identifiers = demand[name]
        if not isinstance(identifiers, list) or any(
            not isinstance(item, str) for item in identifiers
        ):
            raise VerifiedRoutingError(
                f"Route receipt capability_demand.{name} must be a string list."
            )
        require_identifier_tuple(
            identifiers,
            f"route receipt capability_demand.{name}",
        )
    require_safe_id(
        demand["risk_class"],
        "route receipt capability_demand.risk_class",
    )
    return task


def _candidate_from_payload(raw: object) -> CandidateRouteScore:
    payload = _strict_mapping(raw, "ShadowRouteDecision candidate")
    reject_unknown(payload, _CANDIDATE_FIELDS, "ShadowRouteDecision candidate")
    missing = sorted(_CANDIDATE_FIELDS.difference(payload))
    if missing:
        raise VerifiedRoutingError(
            f"Missing ShadowRouteDecision candidate fields: {', '.join(missing)}."
        )
    rejection_codes = payload["rejection_codes"]
    if not isinstance(rejection_codes, list):
        raise VerifiedRoutingError("Candidate rejection_codes must be a list.")
    return CandidateRouteScore(
        route=payload["route"],  # type: ignore[arg-type]
        hard_eligible=payload["hard_eligible"],  # type: ignore[arg-type]
        pareto_eligible=payload["pareto_eligible"],  # type: ignore[arg-type]
        utility=payload["utility"],  # type: ignore[arg-type]
        verified_samples=payload["verified_samples"],  # type: ignore[arg-type]
        success_rate=payload["success_rate"],  # type: ignore[arg-type]
        p95_latency_ms=payload["p95_latency_ms"],  # type: ignore[arg-type]
        mean_tokens=payload["mean_tokens"],  # type: ignore[arg-type]
        cost_sample_count=payload["cost_sample_count"],  # type: ignore[arg-type]
        mean_cost_usd=payload["mean_cost_usd"],  # type: ignore[arg-type]
        mean_premium_calls=payload["mean_premium_calls"],  # type: ignore[arg-type]
        mean_egress_chars=payload["mean_egress_chars"],  # type: ignore[arg-type]
        rejection_codes=tuple(rejection_codes),  # type: ignore[arg-type]
    )


def _optional_metric(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return require_finite_number(
        value,
        label,
        minimum=minimum,
        maximum=maximum,
    )


def _is_stale(scorecard: RouteScorecard, now: str | datetime | None) -> bool:
    if now is None:
        current = datetime.now(timezone.utc)
    elif isinstance(now, datetime):
        if now.tzinfo is None:
            raise VerifiedRoutingError("now must be timezone-aware.")
        current = now.astimezone(timezone.utc)
    else:
        try:
            current = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except ValueError as exc:
            raise VerifiedRoutingError("now must be an ISO-8601 timestamp.") from exc
        if current.tzinfo is None:
            raise VerifiedRoutingError("now must be timezone-aware.")
        current = current.astimezone(timezone.utc)
    generated = datetime.fromisoformat(scorecard.generated_at.replace("Z", "+00:00"))
    expires = datetime.fromisoformat(scorecard.expires_at.replace("Z", "+00:00"))
    return generated > current or current >= expires


def _positive_number(value: object, label: str) -> float:
    rendered = require_finite_number(value, label, minimum=0.0)
    if rendered == 0.0:
        raise VerifiedRoutingError(f"{label} must be positive.")
    return rendered


def _weight(value: object, label: str) -> float:
    return require_finite_number(value, label, minimum=0.0)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: _reject_non_finite(value),
        )
    except json.JSONDecodeError as exc:
        raise VerifiedRoutingError("Invalid route policy JSON.") from exc


def _reject_non_finite(value: str) -> object:
    raise VerifiedRoutingError(f"Non-finite number {value} is not allowed.")


__all__ = [
    "CandidateRouteScore",
    "RoutePolicyNormalization",
    "RoutePolicyWeights",
    "RouteProfilePolicy",
    "ShadowRouteDecision",
    "VerifiedRoutePolicy",
    "load_route_policy",
    "recommend_shadow_route",
    "route_policy_from_payload",
]
