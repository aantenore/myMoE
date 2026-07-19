from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .route_scorecard import RouteScorecard, RouteScorecardEntry
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    ROUTE_PLANS,
    VerifiedRoutingError,
    reject_unknown,
    require_finite_number,
    require_non_negative_int,
    sha256_json,
)

if TYPE_CHECKING:
    from .assistant_bridge import RouteDecisionReceipt
    from .route_signals import TaskSignals


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


@dataclass(frozen=True)
class RoutePolicyWeights:
    quality: float
    cost: float
    latency: float
    egress: float
    premium: float

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
        if self.digest != sha256_json(self.payload()):
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
    baseline_route: str
    recommended_route: str
    applied: bool
    abstained: bool
    reason_codes: tuple[str, ...]
    candidates: tuple[CandidateRouteScore, ...]
    policy_digest: str
    scorecard_digest: str

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "mode": "shadow",
            "profile": self.profile,
            "baseline_route": self.baseline_route,
            "recommended_route": self.recommended_route,
            "applied": self.applied,
            "abstained": self.abstained,
            "reason_codes": list(self.reason_codes),
            "candidates": [candidate.payload() for candidate in self.candidates],
            "policy_digest": self.policy_digest,
            "scorecard_digest": self.scorecard_digest,
        }


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
        cost_usd=_positive_number(normalization_raw["cost_usd"], "cost_usd"),
        latency_ms=_positive_number(normalization_raw["latency_ms"], "latency_ms"),
        egress_chars=_positive_number(
            normalization_raw["egress_chars"], "egress_chars"
        ),
        premium_calls=_positive_number(
            normalization_raw["premium_calls"], "premium_calls"
        ),
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
            quality=_weight(weights_raw["quality"], f"profiles.{name}.quality"),
            cost=_weight(weights_raw["cost"], f"profiles.{name}.cost"),
            latency=_weight(weights_raw["latency"], f"profiles.{name}.latency"),
            egress=_weight(weights_raw["egress"], f"profiles.{name}.egress"),
            premium=_weight(weights_raw["premium"], f"profiles.{name}.premium"),
        )
        if weights.total <= 0.0:
            raise VerifiedRoutingError(f"profiles.{name}.weights must have positive sum.")
        min_samples = require_non_negative_int(
            profile_raw["min_samples"], f"profiles.{name}.min_samples"
        )
        if min_samples == 0:
            raise VerifiedRoutingError(f"profiles.{name}.min_samples must be positive.")
        profiles[name] = RouteProfilePolicy(
            weights=weights,
            min_success_rate=require_finite_number(
                profile_raw["min_success_rate"],
                f"profiles.{name}.min_success_rate",
                minimum=0.0,
                maximum=1.0,
            ),
            min_samples=min_samples,
            min_confidence=require_finite_number(
                profile_raw["min_confidence"],
                f"profiles.{name}.min_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
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
) -> ShadowRouteDecision:
    if profile not in policy.profiles:
        raise VerifiedRoutingError("Unknown route policy profile.")
    baseline = str(receipt.route)
    task = receipt.task
    if not isinstance(task, Mapping):
        raise VerifiedRoutingError("Receipt task metadata must be a mapping.")
    if str(task.get("profile", "")) != profile:
        raise VerifiedRoutingError("Route policy profile does not match the receipt.")
    if str(task.get("task_fingerprint", "")) != str(signals.request_fingerprint):
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
) -> ShadowRouteDecision:
    return ShadowRouteDecision(
        profile=profile,
        baseline_route=baseline,
        recommended_route=recommended,
        applied=False,
        abstained=abstained,
        reason_codes=tuple(dict.fromkeys(reasons)),
        candidates=tuple(sorted(candidates, key=lambda item: _ROUTE_ORDER[item.route])),
        policy_digest=policy.digest,
        scorecard_digest=scorecard.digest,
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
