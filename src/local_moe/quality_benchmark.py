from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable
from urllib import error
from urllib.parse import urlparse

from .config import (
    DistilledRoutingConfig,
    ExpertConfig,
    MoEConfig,
    RoutingConfig,
    SemanticRoutingConfig,
    load_config,
)
from .deterministic_evaluator import (
    EVALUATOR_VERSION,
    BenchmarkCase,
    QualityBenchmarkError,
    evaluate_case_output,
    load_benchmark_cases,
)
from .http_boundary import open_model_endpoint
from .orchestrator import LocalMoE
from .providers import ProviderError


SCHEMA_VERSION = 1
SUPPORTED_VARIANTS = {"single_general", "moe_top1", "moe_top2"}
DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS = 1.0
RUNTIME_PACKAGE_DISTRIBUTIONS = (
    "local-moe-orchestrator",
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "transformers",
    "huggingface-hub",
)
MemorySampler = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class BenchmarkSpec:
    manifest_path: Path
    source_config_path: Path
    dataset_path: Path
    general_expert_id: str
    variants: tuple[str, ...]
    repetitions: int
    endpoint_timeout_seconds: float
    model_match: str
    generation_overrides: dict[str, Any]
    evaluator: dict[str, Any]
    decision: dict[str, Any]
    store_outputs: bool


def sample_host_memory() -> dict[str, Any]:
    """Return metadata-only host RAM/swap counters using stdlib facilities.

    Availability is explicit because no portable stdlib API exposes both RAM and
    swap on every supported operating system. The function never guesses missing
    counters: unsupported or failed components are marked unavailable.
    """

    system = platform.system()
    try:
        if system == "Linux":
            return _sample_linux_host_memory()
        if system == "Darwin":
            return _sample_macos_host_memory()
        if system == "Windows":
            return _sample_windows_host_memory()
        return _sample_posix_host_memory(source="posix_sysconf")
    except Exception:
        return _unavailable_memory_sample(
            source="stdlib_best_effort",
            reason_code="platform_sample_failed",
        )


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    manifest_path = Path(path)
    raw = _read_json_object(manifest_path, "benchmark manifest")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise QualityBenchmarkError(
            f"Unsupported benchmark schema_version: {raw.get('schema_version')!r}."
        )

    variants = tuple(str(item) for item in raw.get("variants", []))
    if not variants or len(variants) != len(set(variants)):
        raise QualityBenchmarkError("variants must contain unique entries.")
    unsupported = sorted(set(variants) - SUPPORTED_VARIANTS)
    if unsupported:
        raise QualityBenchmarkError(f"Unsupported benchmark variants: {unsupported}.")

    evaluator = raw.get("evaluator", {})
    if not isinstance(evaluator, dict) or evaluator.get("type") != "deterministic_rubric":
        raise QualityBenchmarkError("evaluator.type must be 'deterministic_rubric'.")
    quality_threshold = float(evaluator.get("quality_pass_threshold", 0.7))
    if not 0.0 <= quality_threshold <= 1.0:
        raise QualityBenchmarkError("evaluator.quality_pass_threshold must be between 0 and 1.")

    repetitions = int(raw.get("repetitions", 1))
    if repetitions < 1:
        raise QualityBenchmarkError("repetitions must be >= 1.")

    readiness = raw.get("readiness", {})
    if not isinstance(readiness, dict):
        raise QualityBenchmarkError("readiness must be an object.")
    model_match = str(readiness.get("model_match", "exact"))
    if model_match not in {"exact", "endpoint_only"}:
        raise QualityBenchmarkError("readiness.model_match must be 'exact' or 'endpoint_only'.")
    timeout_seconds = float(readiness.get("timeout_seconds", 2.0))
    if timeout_seconds <= 0.0:
        raise QualityBenchmarkError("readiness.timeout_seconds must be > 0.")

    generation_overrides = raw.get("generation_overrides", {})
    decision = raw.get("decision", {})
    if not isinstance(generation_overrides, dict):
        raise QualityBenchmarkError("generation_overrides must be an object.")
    if not isinstance(decision, dict):
        raise QualityBenchmarkError("decision must be an object.")

    source_config = _required_path(raw, "source_config")
    dataset = _required_path(raw, "dataset")
    general_expert_id = str(raw.get("general_expert_id", "")).strip()
    if not general_expert_id:
        raise QualityBenchmarkError("general_expert_id is required.")

    return BenchmarkSpec(
        manifest_path=manifest_path,
        source_config_path=Path(source_config),
        dataset_path=Path(dataset),
        general_expert_id=general_expert_id,
        variants=variants,
        repetitions=repetitions,
        endpoint_timeout_seconds=timeout_seconds,
        model_match=model_match,
        generation_overrides=dict(generation_overrides),
        evaluator=dict(evaluator),
        decision=dict(decision),
        store_outputs=bool(raw.get("store_outputs", True)),
    )


def build_variant_config(
    source: MoEConfig,
    variant: str,
    *,
    general_expert_id: str,
    generation_overrides: dict[str, Any] | None = None,
) -> MoEConfig:
    if variant not in SUPPORTED_VARIANTS:
        raise QualityBenchmarkError(f"Unsupported benchmark variant: {variant}.")
    experts_by_id = source.experts_by_id
    if general_expert_id not in experts_by_id:
        raise QualityBenchmarkError(f"Unknown general expert: {general_expert_id}.")

    overrides = generation_overrides or {}
    experts = tuple(
        replace(expert, params={**expert.params, **overrides}) for expert in source.experts
    )
    if variant == "single_general":
        general = next(expert for expert in experts if expert.id == general_expert_id)
        routing = RoutingConfig(
            top_k=1,
            fallback_order=(general_expert_id,),
            aggregation="best",
            strategy="rules",
            semantic=SemanticRoutingConfig(enabled=False),
            distilled=DistilledRoutingConfig(enabled=False),
        )
        rules = tuple(rule for rule in source.rules if rule.expert_id == general_expert_id)
        return MoEConfig(routing=routing, experts=(general,), rules=rules)

    if variant == "moe_top2" and len(experts) < 2:
        raise QualityBenchmarkError("moe_top2 requires at least two configured experts.")
    top_k = 1 if variant == "moe_top1" else 2
    aggregation = "best" if variant == "moe_top1" else "compare"
    routing = replace(source.routing, top_k=top_k, aggregation=aggregation)
    return MoEConfig(routing=routing, experts=experts, rules=source.rules)


def check_benchmark_readiness(
    config: MoEConfig,
    *,
    timeout_seconds: float,
    model_match: str,
    opener: Callable[..., Any] = open_model_endpoint,
) -> dict[str, Any]:
    checks = [
        _check_expert_readiness(
            expert,
            timeout_seconds=timeout_seconds,
            model_match=model_match,
            opener=opener,
        )
        for expert in config.experts
    ]
    ready = all(item["status"] in {"ready", "not_required"} for item in checks)
    return {
        "status": "ready" if ready else "blocked",
        "checked_at": _now_iso(),
        "model_match": model_match,
        "experts": checks,
    }


def run_quality_benchmark(
    spec: BenchmarkSpec,
    *,
    limit: int = 0,
    readiness_checker: Callable[..., dict[str, Any]] = check_benchmark_readiness,
    moe_factory: Callable[[MoEConfig], Any] = LocalMoE,
    memory_sampler: MemorySampler = sample_host_memory,
    memory_sample_interval_seconds: float = DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if limit < 0:
        raise QualityBenchmarkError("limit must be >= 0.")
    if (
        isinstance(memory_sample_interval_seconds, bool)
        or not isinstance(memory_sample_interval_seconds, (int, float))
        or not math.isfinite(memory_sample_interval_seconds)
        or memory_sample_interval_seconds < 0.0
    ):
        raise QualityBenchmarkError("memory_sample_interval_seconds must be finite and >= 0.")
    source = load_config(spec.source_config_path)
    cases = load_benchmark_cases(spec.dataset_path)
    if limit:
        cases = cases[:limit]
    validation = validate_benchmark_inputs(spec, source, cases)
    provenance = build_benchmark_provenance(
        spec,
        cases,
        source=source,
        memory_sampler=memory_sampler,
        memory_sample_interval_seconds=memory_sample_interval_seconds,
    )
    if validation["status"] != "passed":
        return _blocked_payload(
            provenance,
            validation,
            {"status": "not_run", "experts": []},
            reason="deterministic input validation failed",
        )

    readiness = readiness_checker(
        source,
        timeout_seconds=spec.endpoint_timeout_seconds,
        model_match=spec.model_match,
    )
    if readiness["status"] != "ready":
        return _blocked_payload(
            provenance,
            validation,
            readiness,
            reason="one or more required endpoints or models are unavailable",
        )

    run_memory_before = _safe_memory_sample(memory_sampler)
    variant_configs = {
        variant: build_variant_config(
            source,
            variant,
            general_expert_id=spec.general_expert_id,
            generation_overrides=spec.generation_overrides,
        )
        for variant in spec.variants
    }
    runners = {variant: moe_factory(config) for variant, config in variant_configs.items()}
    records: list[dict[str, Any]] = []
    threshold = float(spec.evaluator.get("quality_pass_threshold", 0.7))

    for repetition in range(1, spec.repetitions + 1):
        for case_index, case in enumerate(cases):
            offset = (case_index + repetition - 1) % len(spec.variants)
            variant_order = (*spec.variants[offset:], *spec.variants[:offset])
            for variant in variant_order:
                records.append(
                    _run_record(
                        runners[variant],
                        variant,
                        case,
                        repetition,
                        quality_pass_threshold=threshold,
                        store_output=spec.store_outputs,
                        memory_sampler=memory_sampler,
                        memory_sample_interval_seconds=memory_sample_interval_seconds,
                    )
                )

    run_memory_after = _safe_memory_sample(memory_sampler)
    run_memory = _summarize_run_memory(
        run_memory_before,
        run_memory_after,
        records,
    )
    metrics = summarize_records(
        records,
        spec.variants,
        general_expert_id=spec.general_expert_id,
    )
    comparisons = compare_to_baseline(metrics, spec.decision)
    gate = evaluate_benchmark_gate(
        metrics,
        comparisons,
        spec.decision,
        host_memory=run_memory,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _now_iso(),
        "provenance": provenance,
        "deterministic_validation": validation,
        "readiness": readiness,
        "execution": {
            "status": "complete",
            "planned_records": len(cases) * spec.repetitions * len(spec.variants),
            "records": records,
            "host_memory": run_memory,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
    }


def validate_benchmark_inputs(
    spec: BenchmarkSpec,
    source: MoEConfig,
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: Any) -> None:
        checks.append({"id": check_id, "passed": passed, "detail": detail})

    add(
        "general_expert_exists",
        spec.general_expert_id in source.experts_by_id,
        spec.general_expert_id,
    )
    add("dataset_nonempty", bool(cases), {"case_count": len(cases)})
    add(
        "case_ids_unique",
        len({case.id for case in cases}) == len(cases),
        {"case_count": len(cases)},
    )
    add(
        "top2_has_two_experts",
        "moe_top2" not in spec.variants or len(source.experts) >= 2,
        {"expert_count": len(source.experts)},
    )
    add(
        "all_cases_have_task_and_quality_checks",
        all(case.task_checks and case.quality_rubric for case in cases),
        {"case_count": len(cases)},
    )
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "checks": checks,
    }


def summarize_records(
    records: list[dict[str, Any]],
    variants: Iterable[str],
    *,
    general_expert_id: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        items = [item for item in records if item["variant"] == variant]
        successes = [item for item in items if item["execution"]["status"] == "ok"]
        latencies = [float(item["execution"]["latency_seconds"]) for item in successes]
        task_scores = [1.0 if item["task_validation"]["passed"] else 0.0 for item in items]
        quality_scores = [float(item["quality_judgment"]["score"]) for item in items]
        quality_passes = [
            1.0 if item["quality_judgment"]["passed"] else 0.0 for item in items
        ]
        non_general_routes = [
            1.0
            if item["execution"].get("selected_experts")
            and general_expert_id not in item["execution"]["selected_experts"]
            else 0.0
            for item in items
        ]
        complete_compares = [
            1.0
            if len(item["execution"].get("actual_experts", [])) >= 2
            else 0.0
            for item in items
        ]
        disagreement_reports = [
            1.0 if item["execution"].get("disagreement_reported") else 0.0
            for item in items
        ]
        response_errors = [
            1.0 if item["execution"].get("errors") else 0.0 for item in items
        ]
        route_fulfillments = [
            1.0 if _route_fulfilled(item["execution"]) else 0.0
            for item in items
        ]
        finish_reasons = [
            reason
            for item in successes
            for reason in _record_finish_reasons(item["execution"])
        ]
        finish_reason_counts = {
            reason: finish_reasons.count(reason)
            for reason in sorted(set(finish_reasons))
        }
        truncations = [
            1.0
            if any(
                _is_truncation_finish_reason(reason)
                for reason in _record_finish_reasons(item["execution"])
            )
            else 0.0
            for item in items
        ]
        completion_tokens = [
            int(item["execution"]["completion_tokens"])
            for item in successes
            if item["execution"].get("completion_tokens") is not None
        ]
        total = len(items)
        failed = total - len(successes)
        latency_seconds_by_case = {
            f"{item['case_id']}#{item['repetition']}": float(
                item["execution"]["latency_seconds"]
            )
            for item in successes
        }
        non_general_case_keys = [
            f"{item['case_id']}#{item['repetition']}"
            for item in successes
            if item["execution"].get("selected_experts")
            and general_expert_id not in item["execution"]["selected_experts"]
        ]
        result[variant] = {
            "planned": total,
            "completed": len(successes),
            "failures": failed,
            "failure_rate": round(failed / max(total, 1), 4),
            "failure_rate_ci95": _wilson_interval(failed, total),
            "task_success_rate": round(sum(task_scores) / max(total, 1), 4),
            "task_success_rate_ci95": _wilson_interval(int(sum(task_scores)), total),
            "quality_pass_rate": round(sum(quality_passes) / max(total, 1), 4),
            "quality_score": round(sum(quality_scores) / max(total, 1), 4),
            "non_general_route_rate": round(
                sum(non_general_routes) / max(total, 1), 4
            ),
            "non_general_case_keys": non_general_case_keys,
            "complete_compare_rate": round(
                sum(complete_compares) / max(total, 1), 4
            ),
            "disagreement_report_rate": round(
                sum(disagreement_reports) / max(total, 1), 4
            ),
            "response_error_rate": round(
                sum(response_errors) / max(total, 1), 4
            ),
            "route_fulfillment_rate": round(
                sum(route_fulfillments) / max(total, 1), 4
            ),
            "finish_reason_counts": finish_reason_counts,
            "truncation_rate": round(sum(truncations) / max(total, 1), 4),
            "latency_seconds": {
                "mean": _mean(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "latency_seconds_by_case": latency_seconds_by_case,
            "completion_tokens_mean": _mean([float(item) for item in completion_tokens]),
            "host_memory": _summarize_variant_memory(items),
            "by_complexity": _group_metrics(items, "complexity"),
            "by_category": _group_metrics(items, "category"),
        }
    return result


def compare_to_baseline(metrics: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    baseline_id = str(decision.get("baseline_variant", "single_general"))
    baseline = metrics.get(baseline_id)
    if baseline is None:
        return {"status": "invalid", "reason": f"missing baseline variant: {baseline_id}"}

    candidates: dict[str, Any] = {}
    for variant, current in metrics.items():
        if variant == baseline_id:
            continue
        baseline_latency = baseline["latency_seconds"]["mean"]
        current_latency = current["latency_seconds"]["mean"]
        latency_ratio = None
        if baseline_latency not in {None, 0.0} and current_latency is not None:
            latency_ratio = round(current_latency / baseline_latency, 4)
        routed_keys = current.get("non_general_case_keys", [])
        baseline_by_case = baseline.get("latency_seconds_by_case", {})
        current_by_case = current.get("latency_seconds_by_case", {})
        paired_keys = [
            key
            for key in routed_keys
            if key in baseline_by_case and key in current_by_case
        ]
        routed_latency_ratios_by_case = {
            key: round(float(current_by_case[key]) / float(baseline_by_case[key]), 4)
            for key in paired_keys
            if float(baseline_by_case[key]) > 0.0
        }
        routed_latency_ratios = list(routed_latency_ratios_by_case.values())
        routed_latency_ratio = _median(routed_latency_ratios)
        candidates[variant] = {
            "quality_delta": round(current["quality_score"] - baseline["quality_score"], 4),
            "task_success_delta": round(
                current["task_success_rate"] - baseline["task_success_rate"], 4
            ),
            "failure_rate_delta": round(
                current["failure_rate"] - baseline["failure_rate"], 4
            ),
            "latency_ratio": latency_ratio,
            "routed_latency_ratio": routed_latency_ratio,
            "routed_latency_ratio_statistic": "median_of_per_case_ratios",
            "routed_latency_ratios_by_case": routed_latency_ratios_by_case,
            "routed_case_count": len(routed_latency_ratios_by_case),
        }
    return {"status": "complete", "baseline_variant": baseline_id, "candidates": candidates}


def evaluate_benchmark_gate(
    metrics: dict[str, Any],
    comparisons: dict[str, Any],
    decision: dict[str, Any],
    *,
    host_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if comparisons.get("status") != "complete":
        return {"status": "failed", "passed": False, "reason": comparisons.get("reason")}

    baseline_id = str(decision.get("baseline_variant", "single_general"))
    value_variant = str(decision.get("value_variant", "moe_top1"))
    diagnostic_variants = tuple(
        str(item) for item in decision.get("diagnostic_variants", ["moe_top2"])
    )
    required_operational_variants = tuple(
        str(item)
        for item in decision.get(
            "required_operational_variants",
            [baseline_id, value_variant],
        )
    )
    min_task = float(decision.get("minimum_task_success_rate", 0.8))
    min_quality_pass = float(decision.get("minimum_quality_pass_rate", 0.8))
    min_quality = float(decision.get("minimum_quality_score", 0.7))
    max_failures = float(decision.get("maximum_failure_rate", 0.05))
    max_truncation_rate = float(decision.get("maximum_truncation_rate", 0.0))
    raw_max_operational_latency = decision.get(
        "maximum_operational_mean_latency_seconds"
    )
    if raw_max_operational_latency is None:
        max_operational_latency = None
        operational_latency_threshold_valid = True
    else:
        try:
            max_operational_latency = float(raw_max_operational_latency)
        except (TypeError, ValueError):
            max_operational_latency = None
            operational_latency_threshold_valid = False
        else:
            operational_latency_threshold_valid = bool(
                math.isfinite(max_operational_latency)
                and max_operational_latency > 0.0
            )
    min_delta = float(
        decision.get(
            "minimum_top1_quality_delta",
            decision.get("minimum_moe_quality_delta", 0.0),
        )
    )
    min_task_delta = float(decision.get("minimum_top1_task_success_delta", 0.0))
    max_failure_delta = float(decision.get("maximum_top1_failure_rate_delta", 0.0))
    max_latency_ratio = float(
        decision.get(
            "maximum_top1_latency_ratio",
            decision.get("maximum_moe_latency_ratio", 1.0),
        )
    )
    max_routed_latency_ratio = float(
        decision.get("maximum_top1_routed_latency_ratio", max_latency_ratio)
    )
    min_routed_case_count = int(
        decision.get("minimum_top1_routed_case_count", 1)
    )
    min_non_general_route_rate = float(
        decision.get("minimum_top1_non_general_route_rate", 0.0)
    )
    min_route_fulfillment_rate = float(
        decision.get("minimum_top1_route_fulfillment_rate", 1.0)
    )
    max_top1_response_error_rate = float(
        decision.get("maximum_top1_response_error_rate", 0.0)
    )
    max_top1_truncation_rate = float(
        decision.get("maximum_top1_truncation_rate", max_truncation_rate)
    )
    min_complete_compare_rate = float(
        decision.get("minimum_top2_complete_compare_rate", 0.0)
    )
    min_disagreement_report_rate = float(
        decision.get("minimum_top2_disagreement_report_rate", 0.0)
    )
    max_response_error_rate = float(
        decision.get("maximum_top2_response_error_rate", 1.0)
    )
    operational_checks = []
    for variant in required_operational_variants:
        values = metrics.get(variant)
        if values is None:
            operational_checks.append(
                {"variant": variant, "passed": False, "missing": True}
            )
            continue
        mean_latency = values.get("latency_seconds", {}).get("mean")
        absolute_latency_passed = bool(
            operational_latency_threshold_valid
            and (
                max_operational_latency is None
                or (
                    isinstance(mean_latency, (int, float))
                    and not isinstance(mean_latency, bool)
                    and math.isfinite(float(mean_latency))
                    and float(mean_latency) <= max_operational_latency
                )
            )
        )
        passed = (
            values["task_success_rate"] >= min_task
            and values["quality_pass_rate"] >= min_quality_pass
            and values["quality_score"] >= min_quality
            and values["failure_rate"] <= max_failures
            and values.get("truncation_rate", 0.0) <= max_truncation_rate
            and absolute_latency_passed
        )
        operational_checks.append(
            {
                "variant": variant,
                "passed": passed,
                "task_success_rate": values["task_success_rate"],
                "quality_pass_rate": values["quality_pass_rate"],
                "quality_score": values["quality_score"],
                "failure_rate": values["failure_rate"],
                "truncation_rate": values.get("truncation_rate", 0.0),
                "mean_latency_seconds": mean_latency,
                "maximum_mean_latency_seconds": max_operational_latency,
                "absolute_latency_passed": absolute_latency_passed,
            }
        )

    comparison = comparisons["candidates"].get(value_variant)
    value_metrics = metrics.get(value_variant)
    value_check: dict[str, Any]
    if comparison is None or value_metrics is None:
        value_check = {"variant": value_variant, "passed": False, "missing": True}
    else:
        latency_ok = (
            comparison["latency_ratio"] is not None
            and comparison["latency_ratio"] <= max_latency_ratio
        )
        routed_latency_ok = (
            comparison["routed_latency_ratio"] is not None
            and comparison["routed_latency_ratio"] <= max_routed_latency_ratio
            and comparison["routed_case_count"] >= min_routed_case_count
        )
        value_check = {
            "variant": value_variant,
            "passed": (
                comparison["quality_delta"] >= min_delta
                and comparison["task_success_delta"] >= min_task_delta
                and comparison["failure_rate_delta"] <= max_failure_delta
                and value_metrics["non_general_route_rate"]
                >= min_non_general_route_rate
                and value_metrics["route_fulfillment_rate"]
                >= min_route_fulfillment_rate
                and value_metrics["response_error_rate"]
                <= max_top1_response_error_rate
                and value_metrics.get("truncation_rate", 0.0)
                <= max_top1_truncation_rate
                and latency_ok
                and routed_latency_ok
            ),
            **comparison,
            "non_general_route_rate": value_metrics["non_general_route_rate"],
            "route_fulfillment_rate": value_metrics["route_fulfillment_rate"],
            "response_error_rate": value_metrics["response_error_rate"],
            "truncation_rate": value_metrics.get("truncation_rate", 0.0),
        }

    diagnostic_checks = []
    for variant in diagnostic_variants:
        values = metrics.get(variant)
        if values is None:
            diagnostic_checks.append(
                {"variant": variant, "passed": False, "missing": True}
            )
            continue
        diagnostic_checks.append(
            {
                "variant": variant,
                "passed": (
                    values["complete_compare_rate"] >= min_complete_compare_rate
                    and values["disagreement_report_rate"]
                    >= min_disagreement_report_rate
                    and values["response_error_rate"] <= max_response_error_rate
                ),
                "evaluation_scope": "aggregated_compare_output_diagnostic_only",
                "complete_compare_rate": values["complete_compare_rate"],
                "disagreement_report_rate": values["disagreement_report_rate"],
                "response_error_rate": values["response_error_rate"],
            }
        )

    operational_passed = all(item["passed"] for item in operational_checks)
    moe_value_demonstrated = bool(value_check["passed"])
    diagnostics_passed = all(item["passed"] for item in diagnostic_checks)
    memory_check = evaluate_host_memory_gate(
        host_memory,
        decision.get("host_memory", {}),
    )
    passed = (
        operational_passed
        and moe_value_demonstrated
        and diagnostics_passed
        and memory_check["passed"]
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "operational_thresholds": {
            "minimum_task_success_rate": min_task,
            "minimum_quality_pass_rate": min_quality_pass,
            "minimum_quality_score": min_quality,
            "maximum_failure_rate": max_failures,
            "maximum_truncation_rate": max_truncation_rate,
            "maximum_operational_mean_latency_seconds": max_operational_latency,
            "operational_latency_threshold_valid": (
                operational_latency_threshold_valid
            ),
        },
        "value_thresholds": {
            "value_variant": value_variant,
            "minimum_top1_quality_delta": min_delta,
            "minimum_top1_task_success_delta": min_task_delta,
            "maximum_top1_failure_rate_delta": max_failure_delta,
            "maximum_top1_latency_ratio": max_latency_ratio,
            "maximum_top1_routed_latency_ratio": max_routed_latency_ratio,
            "minimum_top1_routed_case_count": min_routed_case_count,
            "minimum_top1_non_general_route_rate": min_non_general_route_rate,
            "minimum_top1_route_fulfillment_rate": min_route_fulfillment_rate,
            "maximum_top1_response_error_rate": max_top1_response_error_rate,
            "maximum_top1_truncation_rate": max_top1_truncation_rate,
        },
        "diagnostic_thresholds": {
            "diagnostic_variants": list(diagnostic_variants),
            "minimum_top2_complete_compare_rate": min_complete_compare_rate,
            "minimum_top2_disagreement_report_rate": min_disagreement_report_rate,
            "maximum_top2_response_error_rate": max_response_error_rate,
        },
        "operational_checks": operational_checks,
        "moe_value_checks": [value_check],
        "diagnostic_checks": diagnostic_checks,
        "host_memory_check": memory_check,
        "moe_value_demonstrated": moe_value_demonstrated,
        "diagnostics_passed": diagnostics_passed,
    }


def evaluate_host_memory_gate(
    host_memory: dict[str, Any] | None,
    raw_policy: Any,
) -> dict[str, Any]:
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    required = policy.get("required") is True
    if not required:
        return {
            "status": "not_required",
            "passed": True,
            "required": False,
        }

    try:
        max_swap_growth = int(policy["maximum_swap_growth_bytes"])
        max_peak_ram_percent = float(policy["maximum_peak_ram_used_percent"])
        min_sample_coverage = float(policy.get("minimum_sample_coverage", 1.0))
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid_policy",
            "passed": False,
            "required": True,
            "reason": "host memory thresholds must be numeric",
        }
    if (
        max_swap_growth < 0
        or not math.isfinite(max_peak_ram_percent)
        or not 0.0 < max_peak_ram_percent <= 100.0
        or not math.isfinite(min_sample_coverage)
        or not 0.0 < min_sample_coverage <= 1.0
    ):
        return {
            "status": "invalid_policy",
            "passed": False,
            "required": True,
            "reason": "host memory thresholds are outside their valid range",
        }

    observed = host_memory if isinstance(host_memory, dict) else {}
    before = observed.get("before", {})
    after = observed.get("after", {})
    peak = observed.get("peak_observed", {})
    before_swap = _memory_component_value(before, "swap", "used_bytes")
    after_swap = _memory_component_value(after, "swap", "used_bytes")
    total_ram = _memory_component_value(before, "memory", "total_bytes")
    peak_ram = _memory_component_value(peak, "memory", "used_bytes")
    counters_available = all(
        value is not None
        for value in (before_swap, after_swap, total_ram, peak_ram)
    )
    sample_count = _nonnegative_int(observed.get("sample_count"))
    available_sample_count = _nonnegative_int(
        observed.get("available_sample_count")
    )
    if sample_count is None and available_sample_count is None:
        sample_coverage = 1.0 if observed.get("status") == "available" else 0.0
        sample_counts_valid = observed.get("status") == "available"
    else:
        sample_counts_valid = bool(
            sample_count is not None
            and sample_count > 0
            and available_sample_count is not None
            and available_sample_count <= sample_count
        )
        sample_coverage = (
            float(available_sample_count) / float(sample_count)
            if sample_counts_valid and sample_count is not None
            and available_sample_count is not None
            else 0.0
        )
    sampling_complete = bool(
        sample_counts_valid and sample_coverage >= min_sample_coverage
    )
    swap_growth = (
        int(after_swap) - int(before_swap)
        if before_swap is not None and after_swap is not None
        else None
    )
    peak_ram_percent = (
        round(float(peak_ram) * 100.0 / float(total_ram), 4)
        if peak_ram is not None and total_ram not in {None, 0}
        else None
    )
    swap_ok = swap_growth is not None and swap_growth <= max_swap_growth
    peak_ok = (
        peak_ram_percent is not None
        and peak_ram_percent <= max_peak_ram_percent
    )
    passed = bool(sampling_complete and counters_available and swap_ok and peak_ok)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required": True,
        "sampling_status": observed.get("status", "missing"),
        "sampling_complete": sampling_complete,
        "sample_count": sample_count,
        "available_sample_count": available_sample_count,
        "sample_coverage": round(sample_coverage, 4),
        "minimum_sample_coverage": min_sample_coverage,
        "counters_available": counters_available,
        "swap_growth_bytes": swap_growth,
        "maximum_swap_growth_bytes": max_swap_growth,
        "peak_ram_used_percent": peak_ram_percent,
        "maximum_peak_ram_used_percent": max_peak_ram_percent,
        "unavailable_semantics": (
            "required RAM or swap counters make the release benchmark fail; "
            "offline CI may skip only a blocked or absent live artifact"
        ),
    }


def _memory_component_value(
    observation: Any,
    component: str,
    field: str,
) -> int | None:
    if not isinstance(observation, dict):
        return None
    raw_component = observation.get(component)
    if not isinstance(raw_component, dict) or raw_component.get("status") != "available":
        return None
    return _nonnegative_int(raw_component.get(field))


def collect_runtime_environment_provenance() -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    for distribution in RUNTIME_PACKAGE_DISTRIBUTIONS:
        try:
            package_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = {"status": "unavailable", "version": None}
        else:
            packages[distribution] = {
                "status": "installed",
                "version": package_version,
            }
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "scope": "benchmark_process",
        "server_runtime_identity": {
            "status": "unverified",
            "reason_code": "openai_models_endpoint_has_no_server_version",
            "release_policy": (
                "model snapshot revisions are required; OpenAI-compatible server "
                "package identity remains explicitly unverified"
            ),
        },
    }


def collect_model_snapshot_provenance(config: MoEConfig) -> list[dict[str, Any]]:
    return [_model_snapshot_identity(expert) for expert in config.experts]


def _model_snapshot_identity(expert: ExpertConfig) -> dict[str, Any]:
    base = {
        "expert_id": expert.id,
        "provider": expert.provider,
        "model": expert.model,
        "runtime_backend": str(expert.params.get("runtime_backend", "unknown")),
    }
    if expert.provider == "synthetic":
        return {
            **base,
            "status": "not_required",
            "identity_type": "synthetic",
            "revision": None,
        }

    revision = _huggingface_cached_revision(expert.model)
    if revision is not None:
        return {
            **base,
            "status": "resolved",
            "identity_type": "huggingface_snapshot_revision",
            "revision": revision,
            "serving_match_status": "unverified",
            "serving_match_reason_code": (
                "openai_models_endpoint_exposes_model_id_not_revision"
            ),
        }
    return {
        **base,
        "status": "unresolved",
        "identity_type": "unknown",
        "revision": None,
        "reason_code": "model_revision_not_exposed_by_endpoint_or_local_cache",
        "serving_match_status": "unverified",
        "serving_match_reason_code": (
            "openai_models_endpoint_exposes_model_id_not_revision"
        ),
    }


def _huggingface_cached_revision(model_id: str) -> str | None:
    if model_id.count("/") != 1 or any(part in model_id for part in ("\\", "..")):
        return None
    cache_roots: list[Path] = []
    explicit_cache = os.environ.get("HF_HUB_CACHE")
    if explicit_cache:
        cache_roots.append(Path(explicit_cache).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(Path("~/.cache/huggingface/hub").expanduser())

    repository_dir = "models--" + model_id.replace("/", "--")
    for cache_root in dict.fromkeys(cache_roots):
        ref_path = cache_root / repository_dir / "refs" / "main"
        try:
            revision = ref_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not _is_hex_revision(revision):
            continue
        snapshot_path = cache_root / repository_dir / "snapshots" / revision
        if snapshot_path.is_dir():
            return revision.lower()
    return None


def _is_hex_revision(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def build_benchmark_provenance(
    spec: BenchmarkSpec,
    cases: list[BenchmarkCase],
    *,
    source: MoEConfig,
    memory_sampler: MemorySampler = sample_host_memory,
    memory_sample_interval_seconds: float = DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    return {
        "manifest_path": str(spec.manifest_path),
        "manifest_sha256": _file_sha256(spec.manifest_path),
        "source_config_path": str(spec.source_config_path),
        "source_config_sha256": _file_sha256(spec.source_config_path),
        "dataset_path": str(spec.dataset_path),
        "dataset_sha256": _file_sha256(spec.dataset_path),
        "case_ids_sha256": _sha256_json([case.id for case in cases]),
        "case_count": len(cases),
        "variants": list(spec.variants),
        "repetitions": spec.repetitions,
        "generation_overrides": spec.generation_overrides,
        "execution_order": "deterministic rotating variant order by case and repetition",
        "evaluator": {**spec.evaluator, "implementation_version": EVALUATOR_VERSION},
        "benchmark_implementation_sha256": _file_sha256(Path(__file__)),
        "evaluator_implementation_sha256": _file_sha256(
            Path(__file__).with_name("deterministic_evaluator.py")
        ),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime_environment": collect_runtime_environment_provenance(),
        "model_snapshots": collect_model_snapshot_provenance(source),
        "host_memory_sampling": {
            "scope": "host",
            "content": "metadata_only",
            "unit": "bytes",
            "strategy": (
                "stdlib_best_effort"
                if memory_sampler is sample_host_memory
                else "injected"
            ),
            "sampler": _sampler_name(memory_sampler),
            "record_sample_interval_seconds": memory_sample_interval_seconds,
            "peak_semantics": "maximum used_bytes among samples actually observed",
            "unavailable_semantics": "missing counters remain unavailable; no values are imputed",
        },
    }


def _run_record(
    runner: Any,
    variant: str,
    case: BenchmarkCase,
    repetition: int,
    *,
    quality_pass_threshold: float,
    store_output: bool,
    memory_sampler: MemorySampler,
    memory_sample_interval_seconds: float,
) -> dict[str, Any]:
    correlation_id = f"quality-{case.id}-{variant}-{repetition}"
    memory_tracker = _HostMemoryTracker(
        memory_sampler,
        sample_interval_seconds=memory_sample_interval_seconds,
    )
    memory_tracker.start()
    started = time.perf_counter()
    execution: dict[str, Any] | None = None
    try:
        response = runner.generate(case.prompt, correlation_id=correlation_id)
        elapsed = time.perf_counter() - started
        content = str(response.content)
        task_validation, quality_judgment = evaluate_case_output(
            case,
            content,
            quality_pass_threshold=quality_pass_threshold,
        )
        completion_tokens = sum(
            item.completion_tokens or 0 for item in response.results
        ) or None
        finish_reasons = [
            {
                "expert_id": item.expert_id,
                "finish_reason": getattr(item, "finish_reason", None) or "unknown",
            }
            for item in response.results
        ]
        execution = {
            "status": "ok",
            "latency_seconds": round(elapsed, 4),
            "selected_experts": [item.expert_id for item in response.route.selected],
            "actual_experts": [item.expert_id for item in response.results],
            "errors": list(response.errors),
            "disagreement_reported": response.disagreement is not None,
            "completion_tokens": completion_tokens,
            "finish_reasons": finish_reasons,
            "truncated": any(
                _is_truncation_finish_reason(item["finish_reason"])
                for item in finish_reasons
            ),
            "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        if store_output:
            execution["output"] = content
    except (ProviderError, OSError, RuntimeError) as exc:
        elapsed = time.perf_counter() - started
        task_validation = {
            "status": "failed",
            "passed": False,
            "checks": [],
            "reason": "generation_failed",
        }
        quality_judgment = {
            "status": "not_evaluated",
            "passed": False,
            "score": 0.0,
            "threshold": quality_pass_threshold,
            "evaluator": EVALUATOR_VERSION,
            "criteria": [],
            "reason": "generation_failed",
        }
        execution = {
            "status": "failed",
            "latency_seconds": round(elapsed, 4),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        memory_observation = memory_tracker.stop()
        if execution is not None:
            execution["host_memory"] = memory_observation

    if execution is None:
        raise RuntimeError("record execution did not produce a result")

    return {
        "variant": variant,
        "case_id": case.id,
        "category": case.category,
        "complexity": case.complexity,
        "repetition": repetition,
        "correlation_id": correlation_id,
        "execution": execution,
        "task_validation": task_validation,
        "quality_judgment": quality_judgment,
    }


class _HostMemoryTracker:
    def __init__(
        self,
        sampler: MemorySampler,
        *,
        sample_interval_seconds: float,
    ) -> None:
        self._sampler = sampler
        self._sample_interval_seconds = sample_interval_seconds
        self._samples: list[dict[str, Any]] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._append_sample()
        if self._sample_interval_seconds <= 0.0:
            return
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="quality-benchmark-host-memory",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._append_sample()
        with self._samples_lock:
            samples = list(self._samples)
        return _summarize_memory_samples(samples)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            self._append_sample()

    def _append_sample(self) -> None:
        sample = _safe_memory_sample(self._sampler)
        with self._samples_lock:
            self._samples.append(sample)


def _safe_memory_sample(sampler: MemorySampler) -> dict[str, Any]:
    try:
        raw = sampler()
    except Exception:
        return _unavailable_memory_sample(
            source="injected_sampler",
            reason_code="sampler_raised",
        )
    return _normalize_memory_sample(raw)


def _normalize_memory_sample(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _unavailable_memory_sample(
            source="injected_sampler",
            reason_code="invalid_sampler_payload",
        )

    memory = _normalize_memory_component(raw.get("memory"), available_key="available_bytes")
    swap = _normalize_memory_component(raw.get("swap"), available_key="free_bytes")
    available_components = sum(
        component["status"] == "available" for component in (memory, swap)
    )
    status = (
        "available"
        if available_components == 2
        else "partial"
        if available_components == 1
        else "unavailable"
    )
    result: dict[str, Any] = {
        "status": status,
        "scope": "host",
        "source": _metadata_identifier(raw.get("source"), default="unknown_sampler"),
        "memory": memory,
        "swap": swap,
    }
    reason_code = _reason_code(raw.get("reason_code"))
    if status == "unavailable" and reason_code is not None:
        result["reason_code"] = reason_code
    return result


def _normalize_memory_component(raw: Any, *, available_key: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("status") != "available":
        return {"status": "unavailable"}
    total = _nonnegative_int(raw.get("total_bytes"))
    available = _nonnegative_int(raw.get(available_key))
    used = _nonnegative_int(raw.get("used_bytes"))
    if (
        total is None
        or available is None
        or used is None
        or available > total
        or used > total
        or available + used != total
    ):
        return {"status": "unavailable"}
    return {
        "status": "available",
        "total_bytes": total,
        available_key: available,
        "used_bytes": used,
        "used_percent": round(used * 100.0 / total, 4) if total else 0.0,
    }


def _summarize_memory_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    before = samples[0] if samples else _unavailable_memory_sample(
        source="unknown_sampler",
        reason_code="no_samples",
    )
    after = samples[-1] if samples else before
    peaks = _peak_components_from_samples(samples)
    return {
        "status": _memory_observation_status(samples, peaks),
        "scope": "host",
        "unit": "bytes",
        "sample_count": len(samples),
        "available_sample_count": sum(
            sample.get("status") != "unavailable" for sample in samples
        ),
        "sources": sorted(
            {
                str(sample["source"])
                for sample in samples
                if sample.get("source") is not None
            }
        ),
        "before": before,
        "after": after,
        "peak_observed": peaks,
        "peak_semantics": "maximum used_bytes among samples actually observed",
    }


def _summarize_run_memory(
    before: dict[str, Any],
    after: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [
        item.get("execution", {}).get("host_memory", {})
        for item in records
        if isinstance(item.get("execution", {}).get("host_memory"), dict)
    ]
    base = _summarize_memory_samples([before, after])
    peak_candidates = [base["peak_observed"]] + [
        summary.get("peak_observed", {}) for summary in summaries
    ]
    peaks = _merge_component_peaks(peak_candidates)
    sample_statuses = [before, after] + [
        {"status": summary.get("status", "unavailable")} for summary in summaries
    ]
    base.update(
        {
            "status": _memory_observation_status(sample_statuses, peaks),
            "sample_count": 2
            + sum(int(summary.get("sample_count", 0)) for summary in summaries),
            "available_sample_count": sum(
                sample.get("status") != "unavailable" for sample in (before, after)
            )
            + sum(
                int(summary.get("available_sample_count", 0)) for summary in summaries
            ),
            "sources": sorted(
                {
                    *base["sources"],
                    *(
                        source
                        for summary in summaries
                        for source in summary.get("sources", [])
                    ),
                }
            ),
            "peak_observed": peaks,
        }
    )
    return base


def _summarize_variant_memory(items: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [
        item.get("execution", {}).get("host_memory", {})
        for item in items
        if isinstance(item.get("execution", {}).get("host_memory"), dict)
    ]
    statuses = [str(summary.get("status", "unavailable")) for summary in summaries]
    observed = sum(status != "unavailable" for status in statuses)
    status = (
        "available"
        if summaries and all(value == "available" for value in statuses)
        else "partial"
        if observed
        else "unavailable"
    )
    return {
        "status": status,
        "scope": "host",
        "unit": "bytes",
        "records_observed": observed,
        "records_unavailable": len(items) - observed,
        "sources": sorted(
            {
                source
                for summary in summaries
                for source in summary.get("sources", [])
            }
        ),
        "memory_used_bytes": _summarize_component_across_records(
            summaries,
            component="memory",
        ),
        "swap_used_bytes": _summarize_component_across_records(
            summaries,
            component="swap",
        ),
        "peak_semantics": "maximum used_bytes among samples actually observed",
    }


def _summarize_component_across_records(
    summaries: list[dict[str, Any]],
    *,
    component: str,
) -> dict[str, Any]:
    before_values = _component_values(summaries, "before", component)
    after_values = _component_values(summaries, "after", component)
    peak_values = [
        int(value)
        for summary in summaries
        if (
            value := summary.get("peak_observed", {})
            .get(component, {})
            .get("used_bytes")
        )
        is not None
    ]
    if not before_values and not after_values and not peak_values:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "before_mean": _mean([float(value) for value in before_values]),
        "after_mean": _mean([float(value) for value in after_values]),
        "peak_observed_mean": _mean([float(value) for value in peak_values]),
        "peak_observed_max": max(peak_values) if peak_values else None,
    }


def _component_values(
    summaries: list[dict[str, Any]],
    point: str,
    component: str,
) -> list[int]:
    values: list[int] = []
    for summary in summaries:
        raw = summary.get(point, {}).get(component, {}).get("used_bytes")
        value = _nonnegative_int(raw)
        if value is not None:
            values.append(value)
    return values


def _peak_components_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in ("memory", "swap"):
        values = [
            value
            for sample in samples
            if (value := _nonnegative_int(sample.get(component, {}).get("used_bytes")))
            is not None
        ]
        result[component] = (
            {"status": "available", "used_bytes": max(values)}
            if values
            else {"status": "unavailable"}
        )
    return result


def _merge_component_peaks(peaks: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in ("memory", "swap"):
        values = [
            value
            for peak in peaks
            if (
                value := _nonnegative_int(peak.get(component, {}).get("used_bytes"))
            )
            is not None
        ]
        result[component] = (
            {"status": "available", "used_bytes": max(values)}
            if values
            else {"status": "unavailable"}
        )
    return result


def _memory_observation_status(
    samples: list[dict[str, Any]],
    peaks: dict[str, Any],
) -> str:
    available_components = sum(
        peaks.get(component, {}).get("status") == "available"
        for component in ("memory", "swap")
    )
    if available_components == 0:
        return "unavailable"
    if available_components == 2 and all(
        sample.get("status") == "available" for sample in samples
    ):
        return "available"
    return "partial"


def _sample_linux_host_memory() -> dict[str, Any]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            match = re.fullmatch(r"([^:]+):\s+(\d+)\s+kB", line.strip())
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    except (OSError, UnicodeDecodeError, ValueError):
        return _sample_posix_host_memory(source="linux_sysconf")

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    memory = _measured_memory_component(
        total,
        available,
        available_key="available_bytes",
    )
    swap = _measured_memory_component(
        swap_total,
        swap_free,
        available_key="free_bytes",
    )
    return _compose_memory_sample(source="linux_proc_meminfo", memory=memory, swap=swap)


def _sample_macos_host_memory() -> dict[str, Any]:
    memory: dict[str, Any] = {"status": "unavailable"}
    swap: dict[str, Any] = {"status": "unavailable"}
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        vm_stat = _metadata_command("vm_stat")
        page_counts = {
            match.group(1): int(match.group(2))
            for line in vm_stat.splitlines()
            if (match := re.fullmatch(r"([^:]+):\s+(\d+)\.", line.strip()))
        }
        if "Pages free" in page_counts:
            available_pages = sum(
                page_counts.get(key, 0)
                for key in ("Pages free", "Pages inactive", "Pages speculative")
            )
            memory = _measured_memory_component(
                physical_pages * page_size,
                available_pages * page_size,
                available_key="available_bytes",
            )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        memory = {"status": "unavailable"}

    try:
        swap_usage = _metadata_command("sysctl", "-n", "vm.swapusage")
        match = re.search(
            r"total\s*=\s*([0-9.]+[KMGTP]?B?)\s+used\s*=\s*([0-9.]+[KMGTP]?B?)",
            swap_usage,
            flags=re.IGNORECASE,
        )
        if match:
            swap_total = _parse_byte_quantity(match.group(1))
            swap_used = _parse_byte_quantity(match.group(2))
            swap = _measured_memory_component(
                swap_total,
                None if swap_total is None or swap_used is None else swap_total - swap_used,
                available_key="free_bytes",
            )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        swap = {"status": "unavailable"}

    return _compose_memory_sample(
        source="macos_vm_stat_and_sysctl",
        memory=memory,
        swap=swap,
    )


def _sample_windows_host_memory() -> dict[str, Any]:
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        memory = _measured_memory_component(
            int(status.ullTotalPhys),
            int(status.ullAvailPhys),
            available_key="available_bytes",
        )
    except (AttributeError, OSError, ValueError):
        memory = {"status": "unavailable"}
    return _compose_memory_sample(
        source="windows_global_memory_status_ex",
        memory=memory,
        swap={"status": "unavailable"},
    )


def _sample_posix_host_memory(*, source: str) -> dict[str, Any]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        memory = _measured_memory_component(
            total_pages * page_size,
            available_pages * page_size,
            available_key="available_bytes",
        )
    except (AttributeError, OSError, ValueError):
        memory = {"status": "unavailable"}
    return _compose_memory_sample(
        source=source,
        memory=memory,
        swap={"status": "unavailable"},
    )


def _measured_memory_component(
    total: int | None,
    available: int | None,
    *,
    available_key: str,
) -> dict[str, Any]:
    total_value = _nonnegative_int(total)
    available_value = _nonnegative_int(available)
    if (
        total_value is None
        or available_value is None
        or available_value > total_value
    ):
        return {"status": "unavailable"}
    used = total_value - available_value
    return {
        "status": "available",
        "total_bytes": total_value,
        available_key: available_value,
        "used_bytes": used,
        "used_percent": round(used * 100.0 / total_value, 4) if total_value else 0.0,
    }


def _compose_memory_sample(
    *,
    source: str,
    memory: dict[str, Any],
    swap: dict[str, Any],
) -> dict[str, Any]:
    available_components = sum(
        component.get("status") == "available" for component in (memory, swap)
    )
    return {
        "status": (
            "available"
            if available_components == 2
            else "partial"
            if available_components == 1
            else "unavailable"
        ),
        "scope": "host",
        "source": source,
        "memory": memory,
        "swap": swap,
    }


def _unavailable_memory_sample(*, source: str, reason_code: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "scope": "host",
        "source": _metadata_identifier(source, default="unknown_sampler"),
        "memory": {"status": "unavailable"},
        "swap": {"status": "unavailable"},
        "reason_code": _reason_code(reason_code) or "unavailable",
    }


def _metadata_command(*args: str) -> str:
    completed = subprocess.run(
        list(args),
        capture_output=True,
        check=False,
        text=True,
        timeout=1.0,
    )
    if completed.returncode != 0:
        raise OSError(f"metadata command failed: {args[0]}")
    return completed.stdout


def _parse_byte_quantity(raw: str) -> int | None:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)(?:B)?", raw.strip(), re.I)
    if not match:
        return None
    value = float(match.group(1))
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2).upper()]
    result = value * (1024**power)
    if not math.isfinite(result) or result < 0.0:
        return None
    return int(round(result))


def _nonnegative_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if not math.isfinite(float(raw)) or raw < 0 or int(raw) != raw:
        return None
    return int(raw)


def _metadata_identifier(raw: Any, *, default: str) -> str:
    if not isinstance(raw, str):
        return default
    sanitized = "".join(
        character
        for character in raw.lower()
        if character.isascii() and (character.isalnum() or character in "._-")
    )[:64]
    return sanitized or default


def _reason_code(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    sanitized = _metadata_identifier(raw, default="")
    return sanitized or None


def _sampler_name(sampler: MemorySampler) -> str:
    try:
        raw = getattr(sampler, "__name__", type(sampler).__name__)
    except Exception:
        raw = "custom_sampler"
    return _metadata_identifier(raw, default="custom_sampler")


def _check_expert_readiness(
    expert: ExpertConfig,
    *,
    timeout_seconds: float,
    model_match: str,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    base = {
        "expert_id": expert.id,
        "provider": expert.provider,
        "model": expert.model,
        "base_url": expert.base_url,
    }
    if expert.provider != "openai_compatible":
        return {**base, "status": "not_required", "message": "No HTTP preflight required."}
    models_url = _models_url(expert.base_url)
    if models_url is None:
        return {**base, "status": "blocked", "message": "Missing or malformed base_url."}

    started = time.perf_counter()
    try:
        with opener(models_url, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
    except (OSError, error.URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "blocked",
            "models_url": models_url,
            "message": str(exc),
        }
    model_ids = _model_ids(parsed)
    if model_ids is None:
        return {
            **base,
            "status": "blocked",
            "models_url": models_url,
            "message": "Models endpoint returned an invalid payload.",
        }
    model_available = expert.model in model_ids
    ready = model_match == "endpoint_only" or model_available
    return {
        **base,
        "status": "ready" if ready else "blocked",
        "models_url": models_url,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "available_model_ids": sorted(model_ids),
        "model_available": model_available,
        "message": (
            "Configured model is available."
            if model_available
            else "Endpoint responded but configured model is not listed."
        ),
    }


def _models_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    base = base_url.rstrip("/")
    return base + "/models" if parsed.path.rstrip("/").endswith("/v1") else base + "/v1/models"


def _model_ids(payload: Any) -> set[str] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    result = set()
    for item in payload["data"]:
        if isinstance(item, dict) and item.get("id") is not None:
            result.add(str(item["id"]))
    return result


def _group_metrics(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item[key]), []).append(item)
    return {
        group: {
            "count": len(group_items),
            "task_success_rate": round(
                sum(1.0 if item["task_validation"]["passed"] else 0.0 for item in group_items)
                / len(group_items),
                4,
            ),
            "quality_score": round(
                sum(float(item["quality_judgment"]["score"]) for item in group_items)
                / len(group_items),
                4,
            ),
        }
        for group, group_items in sorted(groups.items())
    }


def _blocked_payload(
    provenance: dict[str, Any],
    validation: dict[str, Any],
    readiness: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "created_at": _now_iso(),
        "provenance": provenance,
        "deterministic_validation": validation,
        "readiness": readiness,
        "execution": {"status": "not_run", "planned_records": 0, "records": []},
        "metrics": {},
        "comparisons": {"status": "not_run"},
        "gate": {"status": "blocked", "passed": False, "reason": reason},
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityBenchmarkError(f"Cannot read {label} {path}: {exc}.") from exc
    if not isinstance(raw, dict):
        raise QualityBenchmarkError(f"{label.capitalize()} must be a JSON object.")
    return raw


def _required_path(raw: dict[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise QualityBenchmarkError(f"{key} is required.")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _route_fulfilled(execution: dict[str, Any]) -> bool:
    selected = execution.get("selected_experts")
    actual = execution.get("actual_experts")
    if not isinstance(selected, list) or not selected or not isinstance(actual, list):
        return False
    return len(selected) == len(actual) and set(selected) == set(actual)


def _record_finish_reasons(execution: dict[str, Any]) -> list[str]:
    raw_reasons = execution.get("finish_reasons")
    reasons: list[str] = []
    if isinstance(raw_reasons, list):
        for item in raw_reasons:
            raw = item.get("finish_reason") if isinstance(item, dict) else item
            value = str(raw).strip().lower() if raw is not None else ""
            reasons.append(value or "unknown")
    if reasons:
        return reasons

    actual = execution.get("actual_experts")
    if isinstance(actual, list) and actual:
        return ["unknown"] * len(actual)
    return []


def _is_truncation_finish_reason(reason: Any) -> bool:
    return str(reason).strip().lower() in {
        "length",
        "max_length",
        "max_tokens",
        "token_limit",
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[midpoint], 4)
    return round((ordered[midpoint - 1] + ordered[midpoint]) / 2.0, 4)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 4)


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 4),
        "upper": round(min(1.0, center + margin), 4),
    }
