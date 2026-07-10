from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable
from urllib import error, request
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
from .orchestrator import LocalMoE
from .providers import ProviderError


SCHEMA_VERSION = 1
SUPPORTED_VARIANTS = {"single_general", "moe_top1", "moe_top2"}


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
    opener: Callable[..., Any] = request.urlopen,
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
) -> dict[str, Any]:
    if limit < 0:
        raise QualityBenchmarkError("limit must be >= 0.")
    source = load_config(spec.source_config_path)
    cases = load_benchmark_cases(spec.dataset_path)
    if limit:
        cases = cases[:limit]
    validation = validate_benchmark_inputs(spec, source, cases)
    provenance = build_benchmark_provenance(spec, cases)
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
                    )
                )

    metrics = summarize_records(records, spec.variants)
    comparisons = compare_to_baseline(metrics, spec.decision)
    gate = evaluate_benchmark_gate(metrics, comparisons, spec.decision)
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


def summarize_records(records: list[dict[str, Any]], variants: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        items = [item for item in records if item["variant"] == variant]
        successes = [item for item in items if item["execution"]["status"] == "ok"]
        latencies = [float(item["execution"]["latency_seconds"]) for item in successes]
        task_scores = [1.0 if item["task_validation"]["passed"] else 0.0 for item in items]
        quality_scores = [float(item["quality_judgment"]["score"]) for item in items]
        completion_tokens = [
            int(item["execution"]["completion_tokens"])
            for item in successes
            if item["execution"].get("completion_tokens") is not None
        ]
        total = len(items)
        failed = total - len(successes)
        result[variant] = {
            "planned": total,
            "completed": len(successes),
            "failures": failed,
            "failure_rate": round(failed / max(total, 1), 4),
            "failure_rate_ci95": _wilson_interval(failed, total),
            "task_success_rate": round(sum(task_scores) / max(total, 1), 4),
            "task_success_rate_ci95": _wilson_interval(int(sum(task_scores)), total),
            "quality_score": round(sum(quality_scores) / max(total, 1), 4),
            "latency_seconds": {
                "mean": _mean(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "completion_tokens_mean": _mean([float(item) for item in completion_tokens]),
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
        candidates[variant] = {
            "quality_delta": round(current["quality_score"] - baseline["quality_score"], 4),
            "task_success_delta": round(
                current["task_success_rate"] - baseline["task_success_rate"], 4
            ),
            "failure_rate_delta": round(
                current["failure_rate"] - baseline["failure_rate"], 4
            ),
            "latency_ratio": latency_ratio,
        }
    return {"status": "complete", "baseline_variant": baseline_id, "candidates": candidates}


def evaluate_benchmark_gate(
    metrics: dict[str, Any],
    comparisons: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    if comparisons.get("status") != "complete":
        return {"status": "failed", "passed": False, "reason": comparisons.get("reason")}

    min_task = float(decision.get("minimum_task_success_rate", 0.8))
    min_quality = float(decision.get("minimum_quality_score", 0.7))
    max_failures = float(decision.get("maximum_failure_rate", 0.05))
    min_delta = float(decision.get("minimum_moe_quality_delta", 0.02))
    max_latency_ratio = float(decision.get("maximum_moe_latency_ratio", 2.0))
    operational_checks = []
    for variant, values in metrics.items():
        passed = (
            values["task_success_rate"] >= min_task
            and values["quality_score"] >= min_quality
            and values["failure_rate"] <= max_failures
        )
        operational_checks.append(
            {
                "variant": variant,
                "passed": passed,
                "task_success_rate": values["task_success_rate"],
                "quality_score": values["quality_score"],
                "failure_rate": values["failure_rate"],
            }
        )

    value_checks = []
    for variant, values in comparisons["candidates"].items():
        latency_ok = values["latency_ratio"] is not None and values["latency_ratio"] <= max_latency_ratio
        passed = (
            values["quality_delta"] >= min_delta
            and values["task_success_delta"] >= 0.0
            and values["failure_rate_delta"] <= 0.0
            and latency_ok
        )
        value_checks.append({"variant": variant, "passed": passed, **values})

    operational_passed = all(item["passed"] for item in operational_checks)
    moe_value_demonstrated = any(item["passed"] for item in value_checks)
    passed = operational_passed and moe_value_demonstrated
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "operational_thresholds": {
            "minimum_task_success_rate": min_task,
            "minimum_quality_score": min_quality,
            "maximum_failure_rate": max_failures,
        },
        "value_thresholds": {
            "minimum_moe_quality_delta": min_delta,
            "maximum_moe_latency_ratio": max_latency_ratio,
        },
        "operational_checks": operational_checks,
        "moe_value_checks": value_checks,
        "moe_value_demonstrated": moe_value_demonstrated,
    }


def build_benchmark_provenance(spec: BenchmarkSpec, cases: list[BenchmarkCase]) -> dict[str, Any]:
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
    }


def _run_record(
    runner: Any,
    variant: str,
    case: BenchmarkCase,
    repetition: int,
    *,
    quality_pass_threshold: float,
    store_output: bool,
) -> dict[str, Any]:
    correlation_id = f"quality-{case.id}-{variant}-{repetition}"
    started = time.perf_counter()
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
        execution = {
            "status": "ok",
            "latency_seconds": round(elapsed, 4),
            "selected_experts": [item.expert_id for item in response.route.selected],
            "actual_experts": [item.expert_id for item in response.results],
            "errors": list(response.errors),
            "completion_tokens": completion_tokens,
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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


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
