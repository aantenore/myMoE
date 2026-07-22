"""Strict contracts for evidence-bound speculative decoding qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_finite_number,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    sha256_json,
)


SPECULATIVE_RUNTIME = "llama.cpp"
SPECULATION_MODES = frozenset(
    {
        "none",
        "draft-simple",
        "draft-eagle3",
        "draft-dflash",
        "draft-mtp",
        "ngram-cache",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
    }
)
STATEFUL_SPECULATION_MODES = frozenset({"ngram-cache", "ngram-mod"})
ALPHA_SPECULATION_MODES = SPECULATION_MODES - STATEFUL_SPECULATION_MODES
DRAFT_MODEL_MODES = frozenset({"draft-simple", "draft-eagle3", "draft-dflash"})
REGIMES = ("cold", "warm")
ORDERS = frozenset({"AB", "BA"})
DECISIONS = frozenset({"qualified", "rejected", "abstained"})
REASON_CODES = frozenset(
    {
        "candidate_acceptance_below_threshold",
        "candidate_acceptance_missing",
        "candidate_memory_budget_exceeded",
        "evidence_execution_failed",
        "evidence_incomplete",
        "evidence_not_counterbalanced",
        "evidence_schedule_mismatch",
        "exact_output_mismatch",
        "median_speedup_below_threshold",
        "p95_latency_regression",
        "p95_ttft_regression",
    }
)
ABSTENTION_REASON_CODES = frozenset(
    {
        "candidate_acceptance_missing",
        "evidence_execution_failed",
        "evidence_incomplete",
        "evidence_not_counterbalanced",
        "evidence_schedule_mismatch",
    }
)
REJECTION_REASON_CODES = REASON_CODES - ABSTENTION_REASON_CODES
MAX_CASES = 256
MAX_TRIALS_PER_CASE = 64

SPECULATIVE_QUALIFIER_CONTRACT = {
    "schema_version": CONTRACT_VERSION,
    "qualifier_id": "mymoe.speculative_cell_qualifier.v1",
    "comparison": "paired_exact_cell_ab_ba",
    "required_regimes": list(REGIMES),
    "regime_gate": "every_regime_must_pass",
    "schedule": "globally_preregistered",
    "stateful_modes": "excluded_from_alpha",
    "output_equivalence": "canonical_text_envelope_sha256",
    "authority": "host_attested_unsigned_advisory",
    "activation_authorized": False,
}
SPECULATIVE_QUALIFIER_CONTRACT_SHA256 = sha256_json(SPECULATIVE_QUALIFIER_CONTRACT)


class SpeculativeCellContractError(VerifiedRoutingError):
    """Raised when speculative-cell evidence is malformed or unbound."""


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except SpeculativeCellContractError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise SpeculativeCellContractError(str(exc)) from exc


def _safe(value: object, label: str) -> str:
    return _call(require_safe_id, value, label)


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _positive_int(value: object, label: str) -> int:
    rendered = _call(require_non_negative_int, value, label)
    if rendered < 1:
        raise SpeculativeCellContractError(f"{label} must be positive.")
    return rendered


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _call(
        require_finite_number,
        value,
        label,
        minimum=minimum,
        maximum=maximum,
    )


def _optional_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, label, minimum=minimum, maximum=maximum)


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _call(require_non_negative_int, value, label)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise SpeculativeCellContractError(f"{label} must be a boolean.")
    return value


def _enum(value: object, allowed: Iterable[str], label: str) -> str:
    rendered = str(value or "")
    if rendered not in allowed:
        raise SpeculativeCellContractError(f"{label} is not supported.")
    return rendered


def _schema(value: object, label: str) -> str:
    if value != CONTRACT_VERSION:
        raise SpeculativeCellContractError(f"Unsupported {label} schema version.")
    return CONTRACT_VERSION


def _digest(value: object, content: dict[str, object], label: str) -> str:
    expected = _call(sha256_json, content)
    if value not in (None, "") and _sha(value, label) != expected:
        raise SpeculativeCellContractError(f"{label} does not match its content.")
    return expected


@dataclass(frozen=True)
class SpeculativeExecutionBinding:
    runtime_revision_sha256: str
    runtime_binary_sha256: str
    runtime_binding_manifest_sha256: str
    hardware_sha256: str
    target_model_sha256: str
    shared_runtime_config_sha256: str
    request_policy_sha256: str
    regime_protocol_sha256: str
    harness_sha256: str
    collector_sha256: str
    adapter_contract_sha256: str
    qualifier_contract_sha256: str = SPECULATIVE_QUALIFIER_CONTRACT_SHA256
    runtime: str = SPECULATIVE_RUNTIME
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "execution binding")
        if self.runtime != SPECULATIVE_RUNTIME:
            raise SpeculativeCellContractError(
                "Only the llama.cpp adapter is supported."
            )
        for name in (
            "runtime_revision_sha256",
            "runtime_binary_sha256",
            "runtime_binding_manifest_sha256",
            "hardware_sha256",
            "target_model_sha256",
            "shared_runtime_config_sha256",
            "request_policy_sha256",
            "regime_protocol_sha256",
            "harness_sha256",
            "collector_sha256",
            "adapter_contract_sha256",
            "qualifier_contract_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.qualifier_contract_sha256 != SPECULATIVE_QUALIFIER_CONTRACT_SHA256:
            raise SpeculativeCellContractError(
                "Execution binding targets another qualifier contract."
            )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "execution binding digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime": self.runtime,
            "runtime_revision_sha256": self.runtime_revision_sha256,
            "runtime_binary_sha256": self.runtime_binary_sha256,
            "runtime_binding_manifest_sha256": (self.runtime_binding_manifest_sha256),
            "hardware_sha256": self.hardware_sha256,
            "target_model_sha256": self.target_model_sha256,
            "shared_runtime_config_sha256": self.shared_runtime_config_sha256,
            "request_policy_sha256": self.request_policy_sha256,
            "regime_protocol_sha256": self.regime_protocol_sha256,
            "harness_sha256": self.harness_sha256,
            "collector_sha256": self.collector_sha256,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "qualifier_contract_sha256": self.qualifier_contract_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeCellBinding:
    cell_id: str
    speculation_config_sha256: str
    speculation_mode: str
    draft_model_sha256: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell binding")
        object.__setattr__(self, "cell_id", _safe(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "speculation_config_sha256",
            _sha(self.speculation_config_sha256, "speculation_config_sha256"),
        )
        mode = _enum(self.speculation_mode, ALPHA_SPECULATION_MODES, "speculation_mode")
        draft = (
            None
            if self.draft_model_sha256 is None
            else _sha(self.draft_model_sha256, "draft_model_sha256")
        )
        if (mode in DRAFT_MODEL_MODES) != (draft is not None):
            raise SpeculativeCellContractError(
                "Draft-model speculation must bind exactly one draft model."
            )
        object.__setattr__(self, "speculation_mode", mode)
        object.__setattr__(self, "draft_model_sha256", draft)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "cell binding digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "draft_model_sha256": self.draft_model_sha256,
            "speculation_config_sha256": self.speculation_config_sha256,
            "speculation_mode": self.speculation_mode,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeQualificationPolicy:
    trials_per_case: int = 4
    minimum_median_speedup_ratio: float = 1.10
    maximum_p95_latency_ratio: float = 1.00
    maximum_p95_ttft_ratio: float = 1.05
    minimum_acceptance_rate: float = 0.05
    maximum_candidate_peak_memory_bytes: int = 24 * 1024**3
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "qualification policy")
        trials = _positive_int(self.trials_per_case, "trials_per_case")
        if trials > MAX_TRIALS_PER_CASE:
            raise SpeculativeCellContractError("trials_per_case exceeds its bound.")
        object.__setattr__(self, "trials_per_case", trials)
        for name, minimum, maximum in (
            ("minimum_median_speedup_ratio", 1.0, 10.0),
            ("maximum_p95_latency_ratio", 0.1, 10.0),
            ("maximum_p95_ttft_ratio", 0.1, 10.0),
            ("minimum_acceptance_rate", 0.0, 1.0),
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name, minimum=minimum, maximum=maximum),
            )
        object.__setattr__(
            self,
            "maximum_candidate_peak_memory_bytes",
            _positive_int(
                self.maximum_candidate_peak_memory_bytes,
                "maximum_candidate_peak_memory_bytes",
            ),
        )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "policy digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trials_per_case": self.trials_per_case,
            "minimum_median_speedup_ratio": self.minimum_median_speedup_ratio,
            "maximum_p95_latency_ratio": self.maximum_p95_latency_ratio,
            "maximum_p95_ttft_ratio": self.maximum_p95_ttft_ratio,
            "minimum_acceptance_rate": self.minimum_acceptance_rate,
            "maximum_candidate_peak_memory_bytes": (
                self.maximum_candidate_peak_memory_bytes
            ),
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeQualificationPlan:
    plan_id: str
    execution: SpeculativeExecutionBinding
    baseline: SpeculativeCellBinding
    candidate: SpeculativeCellBinding
    workload_sha256: str
    case_sha256s: tuple[str, ...]
    order_seed_sha256: str
    policy: SpeculativeQualificationPolicy
    required_regimes: tuple[str, ...] = REGIMES
    authority: str = "advisory_only"
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "qualification plan")
        object.__setattr__(self, "plan_id", _safe(self.plan_id, "plan_id"))
        if not isinstance(self.execution, SpeculativeExecutionBinding):
            raise SpeculativeCellContractError(
                "Plan execution must be an exact bound contract."
            )
        if not isinstance(self.baseline, SpeculativeCellBinding) or not isinstance(
            self.candidate, SpeculativeCellBinding
        ):
            raise SpeculativeCellContractError("Plan cells must be bound contracts.")
        if self.baseline.speculation_mode != "none":
            raise SpeculativeCellContractError("Baseline speculation must be disabled.")
        if self.candidate.speculation_mode == "none":
            raise SpeculativeCellContractError("Candidate speculation must be enabled.")
        if self.baseline.cell_id == self.candidate.cell_id:
            raise SpeculativeCellContractError(
                "Baseline and candidate ids must differ."
            )
        if (
            self.baseline.speculation_config_sha256
            == self.candidate.speculation_config_sha256
        ):
            raise SpeculativeCellContractError(
                "Baseline and candidate speculation configs must differ."
            )
        if self.baseline.digest == self.candidate.digest:
            raise SpeculativeCellContractError("Baseline and candidate must differ.")
        object.__setattr__(
            self, "workload_sha256", _sha(self.workload_sha256, "workload_sha256")
        )
        object.__setattr__(
            self,
            "order_seed_sha256",
            _sha(self.order_seed_sha256, "order_seed_sha256"),
        )
        if not isinstance(self.case_sha256s, (tuple, list)):
            raise SpeculativeCellContractError("case_sha256s must be a list.")
        cases = tuple(_sha(item, "case_sha256s") for item in self.case_sha256s)
        if (
            not cases
            or len(cases) > MAX_CASES
            or len(set(cases)) != len(cases)
            or cases != tuple(sorted(cases))
        ):
            raise SpeculativeCellContractError(
                "case_sha256s must be non-empty, unique, bounded, and sorted."
            )
        object.__setattr__(self, "case_sha256s", cases)
        if not isinstance(self.required_regimes, (tuple, list)):
            raise SpeculativeCellContractError("required_regimes must be a list.")
        regimes = tuple(str(item) for item in self.required_regimes)
        if regimes != REGIMES:
            raise SpeculativeCellContractError("Cold and warm regimes are required.")
        object.__setattr__(self, "required_regimes", regimes)
        if self.authority != "advisory_only":
            raise SpeculativeCellContractError("Qualification is advisory only.")
        if not isinstance(self.policy, SpeculativeQualificationPolicy):
            raise SpeculativeCellContractError("Plan policy is invalid.")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "plan digest"),
        )

    @property
    def expected_trial_count(self) -> int:
        return (
            len(self.case_sha256s)
            * len(self.required_regimes)
            * self.policy.trials_per_case
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "execution": self.execution.payload(),
            "baseline": self.baseline.payload(),
            "candidate": self.candidate.payload(),
            "workload_sha256": self.workload_sha256,
            "case_sha256s": list(self.case_sha256s),
            "order_seed_sha256": self.order_seed_sha256,
            "policy": self.policy.payload(),
            "required_regimes": list(self.required_regimes),
            "authority": self.authority,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeArmMeasurement:
    cell_sha256: str
    success: bool
    output_sha256: str | None = None
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    predicted_tokens: int | None = None
    predicted_ms: float | None = None
    peak_memory_bytes: int | None = None
    draft_generated_tokens: int | None = None
    draft_accepted_tokens: int | None = None
    error_code: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "arm measurement")
        object.__setattr__(self, "cell_sha256", _sha(self.cell_sha256, "cell_sha256"))
        success = _bool(self.success, "success")
        object.__setattr__(self, "success", success)
        if success:
            if self.error_code is not None:
                raise SpeculativeCellContractError(
                    "Successful measurements cannot carry an error code."
                )
            output = _sha(self.output_sha256, "output_sha256")
            ttft = _number(self.ttft_ms, "ttft_ms", minimum=0.0)
            total = _number(self.total_latency_ms, "total_latency_ms", minimum=0.0)
            predicted_tokens = _positive_int(self.predicted_tokens, "predicted_tokens")
            predicted_ms = _number(self.predicted_ms, "predicted_ms", minimum=0.000001)
            peak = _positive_int(self.peak_memory_bytes, "peak_memory_bytes")
            if ttft > total:
                raise SpeculativeCellContractError(
                    "ttft_ms cannot exceed total latency."
                )
            draft_generated = _optional_int(
                self.draft_generated_tokens, "draft_generated_tokens"
            )
            draft_accepted = _optional_int(
                self.draft_accepted_tokens, "draft_accepted_tokens"
            )
            if (draft_generated is None) != (draft_accepted is None):
                raise SpeculativeCellContractError(
                    "Draft counters must be supplied together."
                )
            if (
                draft_generated is not None
                and draft_accepted is not None
                and draft_accepted > draft_generated
            ):
                raise SpeculativeCellContractError(
                    "Accepted draft tokens cannot exceed generated draft tokens."
                )
            values = {
                "output_sha256": output,
                "ttft_ms": ttft,
                "total_latency_ms": total,
                "predicted_tokens": predicted_tokens,
                "predicted_ms": predicted_ms,
                "peak_memory_bytes": peak,
                "draft_generated_tokens": draft_generated,
                "draft_accepted_tokens": draft_accepted,
                "error_code": None,
            }
        else:
            if self.error_code is None:
                raise SpeculativeCellContractError(
                    "Failed measurements require a stable error code."
                )
            error_code = _safe(self.error_code, "error_code")
            fields = (
                self.output_sha256,
                self.ttft_ms,
                self.total_latency_ms,
                self.predicted_tokens,
                self.predicted_ms,
                self.peak_memory_bytes,
                self.draft_generated_tokens,
                self.draft_accepted_tokens,
            )
            if any(item is not None for item in fields):
                raise SpeculativeCellContractError(
                    "Failed measurements cannot claim performance or output evidence."
                )
            values = {
                "output_sha256": None,
                "ttft_ms": None,
                "total_latency_ms": None,
                "predicted_tokens": None,
                "predicted_ms": None,
                "peak_memory_bytes": None,
                "draft_generated_tokens": None,
                "draft_accepted_tokens": None,
                "error_code": error_code,
            }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "arm measurement digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cell_sha256": self.cell_sha256,
            "success": self.success,
            "output_sha256": self.output_sha256,
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_latency_ms,
            "predicted_tokens": self.predicted_tokens,
            "predicted_ms": self.predicted_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "draft_generated_tokens": self.draft_generated_tokens,
            "draft_accepted_tokens": self.draft_accepted_tokens,
            "error_code": self.error_code,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeTrial:
    plan_sha256: str
    sequence_index: int
    case_sha256: str
    repetition: int
    regime: str
    order: str
    baseline: SpeculativeArmMeasurement
    candidate: SpeculativeArmMeasurement
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "trial")
        object.__setattr__(self, "plan_sha256", _sha(self.plan_sha256, "plan_sha256"))
        sequence_index = _call(
            require_non_negative_int, self.sequence_index, "sequence_index"
        )
        if sequence_index >= MAX_CASES * len(REGIMES) * MAX_TRIALS_PER_CASE:
            raise SpeculativeCellContractError("sequence_index exceeds its bound.")
        object.__setattr__(self, "sequence_index", sequence_index)
        object.__setattr__(self, "case_sha256", _sha(self.case_sha256, "case_sha256"))
        repetition = _call(require_non_negative_int, self.repetition, "repetition")
        if repetition >= MAX_TRIALS_PER_CASE:
            raise SpeculativeCellContractError("repetition exceeds its bound.")
        object.__setattr__(self, "repetition", repetition)
        object.__setattr__(self, "regime", _enum(self.regime, REGIMES, "regime"))
        object.__setattr__(self, "order", _enum(self.order, ORDERS, "order"))
        if not isinstance(self.baseline, SpeculativeArmMeasurement) or not isinstance(
            self.candidate, SpeculativeArmMeasurement
        ):
            raise SpeculativeCellContractError("Trial arm measurements are invalid.")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "trial digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "sequence_index": self.sequence_index,
            "case_sha256": self.case_sha256,
            "repetition": self.repetition,
            "regime": self.regime,
            "order": self.order,
            "baseline": self.baseline.payload(),
            "candidate": self.candidate.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class SpeculativeQualificationReceipt:
    plan_sha256: str
    evidence_sha256: str
    decision: str
    reason_codes: tuple[str, ...]
    expected_trials: int
    expected_cases: int
    trials_per_case: int
    observed_trials: int
    unique_cases: int
    cold_trials: int
    warm_trials: int
    failed_arms: int
    output_mismatches: int
    cold_median_speedup_ratio: float | None
    cold_p95_latency_ratio: float | None
    cold_p95_ttft_ratio: float | None
    cold_candidate_acceptance_rate: float | None
    warm_median_speedup_ratio: float | None
    warm_p95_latency_ratio: float | None
    warm_p95_ttft_ratio: float | None
    warm_candidate_acceptance_rate: float | None
    candidate_peak_memory_bytes: int | None
    activation_authorized: bool = False
    authority: str = "host_attested_unsigned_advisory"
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "qualification receipt")
        for name in ("plan_sha256", "evidence_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        decision = _enum(self.decision, DECISIONS, "decision")
        object.__setattr__(self, "decision", decision)
        if not isinstance(self.reason_codes, (tuple, list)):
            raise SpeculativeCellContractError("reason_codes must be a list.")
        reasons = tuple(str(item) for item in self.reason_codes)
        if (
            len(set(reasons)) != len(reasons)
            or reasons != tuple(sorted(reasons))
            or any(reason not in REASON_CODES for reason in reasons)
            or (decision == "qualified") == bool(reasons)
        ):
            raise SpeculativeCellContractError("Receipt reason codes are invalid.")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "expected_trials",
            _positive_int(self.expected_trials, "expected_trials"),
        )
        object.__setattr__(
            self,
            "expected_cases",
            _positive_int(self.expected_cases, "expected_cases"),
        )
        trials_per_case = _positive_int(self.trials_per_case, "trials_per_case")
        if trials_per_case > MAX_TRIALS_PER_CASE:
            raise SpeculativeCellContractError("trials_per_case exceeds its bound.")
        object.__setattr__(self, "trials_per_case", trials_per_case)
        if self.expected_trials != (
            self.expected_cases * len(REGIMES) * self.trials_per_case
        ):
            raise SpeculativeCellContractError(
                "Receipt expected counts are inconsistent."
            )
        for name in (
            "observed_trials",
            "unique_cases",
            "cold_trials",
            "warm_trials",
            "failed_arms",
            "output_mismatches",
        ):
            object.__setattr__(
                self, name, _call(require_non_negative_int, getattr(self, name), name)
            )
        if self.observed_trials > self.expected_trials:
            raise SpeculativeCellContractError(
                "Observed trials cannot exceed expected trials."
            )
        if self.cold_trials + self.warm_trials != self.observed_trials:
            raise SpeculativeCellContractError(
                "Cold and warm counts must equal observed trials."
            )
        if self.observed_trials == self.expected_trials and (
            self.expected_trials % len(REGIMES) != 0
            or self.cold_trials != self.expected_trials // len(REGIMES)
            or self.warm_trials != self.expected_trials // len(REGIMES)
        ):
            raise SpeculativeCellContractError(
                "Complete receipt regime counts are inconsistent."
            )
        if (
            self.unique_cases > self.expected_cases
            or self.unique_cases > self.observed_trials
            or ((self.observed_trials == 0) != (self.unique_cases == 0))
        ):
            raise SpeculativeCellContractError("Receipt case counts are inconsistent.")
        if (
            self.observed_trials == self.expected_trials
            and self.unique_cases != self.expected_cases
        ):
            raise SpeculativeCellContractError(
                "Complete receipt case counts are inconsistent."
            )
        if self.failed_arms > 2 * self.observed_trials:
            raise SpeculativeCellContractError("Receipt failed-arm count is invalid.")
        if self.output_mismatches > self.observed_trials:
            raise SpeculativeCellContractError("Receipt mismatch count is invalid.")

        performance_names = (
            "cold_median_speedup_ratio",
            "cold_p95_latency_ratio",
            "cold_p95_ttft_ratio",
            "warm_median_speedup_ratio",
            "warm_p95_latency_ratio",
            "warm_p95_ttft_ratio",
        )
        for name in performance_names:
            value = _optional_number(getattr(self, name), name, minimum=0.0)
            if value is not None and value <= 0:
                raise SpeculativeCellContractError(
                    "Receipt regime performance metrics must be positive."
                )
            object.__setattr__(
                self,
                name,
                value,
            )
        acceptance_names = (
            "cold_candidate_acceptance_rate",
            "warm_candidate_acceptance_rate",
        )
        for name in acceptance_names:
            object.__setattr__(
                self,
                name,
                _optional_number(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        performance_present = [
            getattr(self, name) is not None for name in performance_names
        ]
        if any(performance_present) and not all(performance_present):
            raise SpeculativeCellContractError(
                "Receipt regime performance metrics must be complete."
            )
        has_performance = all(performance_present)
        memory = _optional_int(
            self.candidate_peak_memory_bytes, "candidate_peak_memory_bytes"
        )
        object.__setattr__(self, "candidate_peak_memory_bytes", memory)
        if has_performance:
            if (
                memory is None
                or self.observed_trials != self.expected_trials
                or self.failed_arms != 0
                or self.cold_trials == 0
                or self.warm_trials == 0
            ):
                raise SpeculativeCellContractError(
                    "Receipt performance metrics require complete successful evidence."
                )
        elif memory is not None or any(
            getattr(self, name) is not None for name in acceptance_names
        ):
            raise SpeculativeCellContractError(
                "Receipt cannot claim partial performance evidence."
            )

        complete_acceptance = all(
            getattr(self, name) is not None for name in acceptance_names
        )
        if decision == "qualified":
            if not has_performance or not complete_acceptance or self.output_mismatches:
                raise SpeculativeCellContractError(
                    "Qualified receipt evidence is internally inconsistent."
                )
        elif decision == "rejected":
            if (
                not has_performance
                or not complete_acceptance
                or not set(reasons).issubset(REJECTION_REASON_CODES)
                or (
                    (self.output_mismatches > 0) != ("exact_output_mismatch" in reasons)
                )
            ):
                raise SpeculativeCellContractError(
                    "Rejected receipt evidence is internally inconsistent."
                )
        elif not set(reasons).issubset(ABSTENTION_REASON_CODES):
            raise SpeculativeCellContractError(
                "Abstained receipt reason codes are invalid."
            )

        incomplete = self.observed_trials < self.expected_trials
        if incomplete != ("evidence_incomplete" in reasons):
            raise SpeculativeCellContractError(
                "Receipt completeness reason is inconsistent."
            )
        failed = self.failed_arms > 0
        if failed != ("evidence_execution_failed" in reasons):
            raise SpeculativeCellContractError(
                "Receipt execution-failure reason is inconsistent."
            )
        if "candidate_acceptance_missing" in reasons:
            if not has_performance or complete_acceptance:
                raise SpeculativeCellContractError(
                    "Receipt acceptance reason is inconsistent."
                )
        elif decision == "abstained" and has_performance:
            raise SpeculativeCellContractError(
                "Abstained performance evidence requires a missing-acceptance reason."
            )
        if self.activation_authorized is not False:
            raise SpeculativeCellContractError("Receipt cannot authorize activation.")
        if self.authority != "host_attested_unsigned_advisory":
            raise SpeculativeCellContractError("Receipt authority is invalid.")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "evidence_sha256": self.evidence_sha256,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "expected_trials": self.expected_trials,
            "expected_cases": self.expected_cases,
            "trials_per_case": self.trials_per_case,
            "observed_trials": self.observed_trials,
            "unique_cases": self.unique_cases,
            "cold_trials": self.cold_trials,
            "warm_trials": self.warm_trials,
            "failed_arms": self.failed_arms,
            "output_mismatches": self.output_mismatches,
            "cold_median_speedup_ratio": self.cold_median_speedup_ratio,
            "cold_p95_latency_ratio": self.cold_p95_latency_ratio,
            "cold_p95_ttft_ratio": self.cold_p95_ttft_ratio,
            "cold_candidate_acceptance_rate": (self.cold_candidate_acceptance_rate),
            "warm_median_speedup_ratio": self.warm_median_speedup_ratio,
            "warm_p95_latency_ratio": self.warm_p95_latency_ratio,
            "warm_p95_ttft_ratio": self.warm_p95_ttft_ratio,
            "warm_candidate_acceptance_rate": (self.warm_candidate_acceptance_rate),
            "candidate_peak_memory_bytes": self.candidate_peak_memory_bytes,
            "activation_authorized": self.activation_authorized,
            "authority": self.authority,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def speculative_execution_binding_from_payload(
    value: object,
) -> SpeculativeExecutionBinding:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "runtime",
            "runtime_revision_sha256",
            "runtime_binary_sha256",
            "runtime_binding_manifest_sha256",
            "hardware_sha256",
            "target_model_sha256",
            "shared_runtime_config_sha256",
            "request_policy_sha256",
            "regime_protocol_sha256",
            "harness_sha256",
            "collector_sha256",
            "adapter_contract_sha256",
            "qualifier_contract_sha256",
            "digest",
        },
        "execution binding",
    )
    return _call(SpeculativeExecutionBinding, **raw)


def speculative_cell_binding_from_payload(value: object) -> SpeculativeCellBinding:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "cell_id",
            "draft_model_sha256",
            "speculation_config_sha256",
            "speculation_mode",
            "digest",
        },
        "cell binding",
    )
    return _call(SpeculativeCellBinding, **raw)


def speculative_policy_from_payload(value: object) -> SpeculativeQualificationPolicy:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "trials_per_case",
            "minimum_median_speedup_ratio",
            "maximum_p95_latency_ratio",
            "maximum_p95_ttft_ratio",
            "minimum_acceptance_rate",
            "maximum_candidate_peak_memory_bytes",
            "digest",
        },
        "qualification policy",
    )
    return _call(SpeculativeQualificationPolicy, **raw)


def speculative_plan_from_payload(value: object) -> SpeculativeQualificationPlan:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "plan_id",
            "execution",
            "baseline",
            "candidate",
            "workload_sha256",
            "case_sha256s",
            "order_seed_sha256",
            "policy",
            "required_regimes",
            "authority",
            "digest",
        },
        "qualification plan",
    )
    raw["execution"] = speculative_execution_binding_from_payload(raw["execution"])
    raw["baseline"] = speculative_cell_binding_from_payload(raw["baseline"])
    raw["candidate"] = speculative_cell_binding_from_payload(raw["candidate"])
    raw["policy"] = speculative_policy_from_payload(raw["policy"])
    return _call(SpeculativeQualificationPlan, **raw)


def speculative_arm_from_payload(value: object) -> SpeculativeArmMeasurement:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "cell_sha256",
            "success",
            "output_sha256",
            "ttft_ms",
            "total_latency_ms",
            "predicted_tokens",
            "predicted_ms",
            "peak_memory_bytes",
            "draft_generated_tokens",
            "draft_accepted_tokens",
            "error_code",
            "digest",
        },
        "arm measurement",
    )
    return _call(SpeculativeArmMeasurement, **raw)


def speculative_trial_from_payload(value: object) -> SpeculativeTrial:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "plan_sha256",
            "sequence_index",
            "case_sha256",
            "repetition",
            "regime",
            "order",
            "baseline",
            "candidate",
            "digest",
        },
        "trial",
    )
    raw["baseline"] = speculative_arm_from_payload(raw["baseline"])
    raw["candidate"] = speculative_arm_from_payload(raw["candidate"])
    return _call(SpeculativeTrial, **raw)


def speculative_receipt_from_payload(value: object) -> SpeculativeQualificationReceipt:
    raw = _strict_record(
        value,
        {
            "schema_version",
            "plan_sha256",
            "evidence_sha256",
            "decision",
            "reason_codes",
            "expected_trials",
            "expected_cases",
            "trials_per_case",
            "observed_trials",
            "unique_cases",
            "cold_trials",
            "warm_trials",
            "failed_arms",
            "output_mismatches",
            "cold_median_speedup_ratio",
            "cold_p95_latency_ratio",
            "cold_p95_ttft_ratio",
            "cold_candidate_acceptance_rate",
            "warm_median_speedup_ratio",
            "warm_p95_latency_ratio",
            "warm_p95_ttft_ratio",
            "warm_candidate_acceptance_rate",
            "candidate_peak_memory_bytes",
            "activation_authorized",
            "authority",
            "digest",
        },
        "qualification receipt",
    )
    return _call(SpeculativeQualificationReceipt, **raw)


def _strict_record(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpeculativeCellContractError(f"{label} must be an object.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(str(item) for item in actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise SpeculativeCellContractError(
            f"Invalid {label} fields: {'; '.join(details)}."
        )
    return dict(value)
