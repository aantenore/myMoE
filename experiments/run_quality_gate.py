from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

from local_moe.config import MoEConfig, load_config
from local_moe.deterministic_evaluator import (
    BenchmarkCase,
    evaluate_case_output,
    load_benchmark_cases,
)
from local_moe.evaluator import evaluate_router, load_eval_cases
from local_moe.evaluation_integrity import analyze_route_holdout
from local_moe.quality_benchmark import (
    collect_model_snapshot_provenance,
    collect_runtime_environment_provenance,
    compare_to_baseline,
    evaluate_benchmark_gate,
    summarize_records,
)


SUPPORTED_GATE_PROFILES = {"release", "ci_offline"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quality-gate.json")
    parser.add_argument("--out", default="outputs/quality-gate.json")
    args = parser.parse_args()

    gate_config = _load_gate_config(Path(args.config))
    profile = str(gate_config.get("profile", ""))
    checks = _run_gate_checks(gate_config)
    report = _summarize_gate(profile, checks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "profile": profile,
                "passed": report["passed"],
                "checks_passed": report["checks_passed"],
                "release_ready": report["release_ready"],
                "out": str(out),
            },
            indent=2,
        )
    )

    if not report["passed"]:
        sys.exit(1)


def _run_gate_checks(gate_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _check_gate_profile(gate_config),
        _check_required_files(gate_config.get("required_files", [])),
        _check_routing_eval(gate_config.get("routing_eval", {})),
        _check_routing_holdout(gate_config.get("routing_holdout", {})),
        _check_quality_benchmark(gate_config.get("quality_benchmark", {})),
        _check_forbidden_listeners(gate_config.get("forbidden_listeners", [])),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gate_config(
    path: Path,
    *,
    _parents: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in _parents:
        chain = " -> ".join(str(item) for item in (*_parents, resolved))
        raise ValueError(f"Circular quality-gate config inheritance: {chain}")

    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Quality-gate config must be an object: {path}")
    extends = raw.get("extends")
    if extends is None:
        return raw
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("quality-gate extends must be a non-empty path")

    parent_path = Path(extends)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = _load_gate_config(parent_path, _parents=(*_parents, resolved))
    override = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_merge(parent, override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _check_gate_profile(config: object) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {
            "name": "gate_profile",
            "passed": False,
            "error": "quality-gate config must be an object",
        }
    profile = str(config.get("profile", ""))
    quality = config.get("quality_benchmark", {})
    mode = quality.get("mode") if isinstance(quality, dict) else None
    expected_mode = {
        "release": "required",
        "ci_offline": "offline_optional",
    }.get(profile)
    passed = profile in SUPPORTED_GATE_PROFILES and mode == expected_mode
    return {
        "name": "gate_profile",
        "profile": profile,
        "quality_benchmark_mode": mode,
        "expected_quality_benchmark_mode": expected_mode,
        "passed": passed,
        "error": (
            None
            if passed
            else "profile must be release/required or ci_offline/offline_optional"
        ),
    }


def _summarize_gate(profile: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    checks_passed = all(check.get("passed") is True for check in checks)
    release_ready = bool(
        checks_passed
        and profile == "release"
        and all(check.get("release_eligible", True) for check in checks)
    )
    passed = (
        release_ready
        if profile == "release"
        else checks_passed
        if profile == "ci_offline"
        else False
    )
    return {
        "profile": profile,
        "passed": passed,
        "checks_passed": checks_passed,
        "release_ready": release_ready,
        "checks": checks,
    }


def _check_required_files(files: object) -> dict[str, Any]:
    missing = [path for path in files if not Path(str(path)).exists()]
    return {
        "name": "required_files",
        "passed": not missing,
        "missing": missing,
    }


def _check_routing_eval(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "routing_eval",
            "passed": False,
            "error": "routing_eval config must be an object",
        }

    paths = {
        "result": Path(str(raw.get("result_path", ""))),
        "config": Path(str(raw.get("config_path", ""))),
        "eval": Path(str(raw.get("eval_path", ""))),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "name": "routing_eval",
            "passed": False,
            "error": f"Missing routing eval inputs: {', '.join(sorted(missing))}",
        }

    try:
        result = _read_json(paths["result"])
        if not isinstance(result, dict):
            raise ValueError("routing eval result must be an object")
        config = load_config(paths["config"])
        cases = load_eval_cases(paths["eval"])
        recomputed = evaluate_router(config, cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "name": "routing_eval",
            "passed": False,
            "error": f"Cannot recompute routing eval: {exc}",
        }

    provenance = result.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    provenance_matches = {
        "config_path": _same_path(provenance.get("config_path"), paths["config"]),
        "config_content": (
            provenance.get("config_sha256") == _file_sha256(paths["config"])
        ),
        "eval_path": _same_path(provenance.get("eval_path"), paths["eval"]),
        "eval_content": (
            provenance.get("eval_sha256") == _file_sha256(paths["eval"])
        ),
    }
    evaluation_fields = (
        "accuracy",
        "accuracy_ci95",
        "total",
        "by_complexity",
        "results",
    )
    report_matches_recomputed = all(
        result.get(field) == recomputed[field] for field in evaluation_fields
    )
    min_accuracy = float(raw.get("min_accuracy", 0.0))
    min_total = int(raw.get("min_total", 0))
    required_complexities = {str(item) for item in raw.get("required_complexities", [])}
    observed_complexities = set(result.get("by_complexity", {}).keys())

    accuracy = float(result.get("accuracy", 0.0))
    total = int(result.get("total", 0))
    missing_complexities = sorted(required_complexities - observed_complexities)
    failed_cases = [
        item
        for item in result.get("results", [])
        if isinstance(item, dict) and not item.get("passed")
    ]

    passed = (
        all(provenance_matches.values())
        and report_matches_recomputed
        and accuracy >= min_accuracy
        and total >= min_total
        and not missing_complexities
        and not failed_cases
    )

    return {
        "name": "routing_eval",
        "passed": passed,
        "accuracy": accuracy,
        "min_accuracy": min_accuracy,
        "total": total,
        "min_total": min_total,
        "missing_complexities": missing_complexities,
        "failed_case_ids": [item.get("id") for item in failed_cases],
        "report_matches_recomputed": report_matches_recomputed,
        "provenance_matches": provenance_matches,
    }


def _check_routing_holdout(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "routing_holdout",
            "passed": False,
            "error": "routing_holdout config must be an object",
        }

    paths = {
        "result": Path(str(raw.get("result_path", ""))),
        "config": Path(str(raw.get("config_path", ""))),
        "holdout": Path(str(raw.get("eval_path", ""))),
        "training": Path(str(raw.get("training_labels_path", ""))),
        "artifact": Path(str(raw.get("artifact_path", ""))),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {
            "name": "routing_holdout",
            "passed": False,
            "error": f"Missing routing holdout inputs: {', '.join(sorted(missing))}",
        }

    result = _read_json(paths["result"])
    holdout_records = _read_jsonl(paths["holdout"])
    training_records = _read_jsonl(paths["training"])
    artifact = _read_json(paths["artifact"])
    moe_config = load_config(paths["config"])
    recomputed_result = evaluate_router(moe_config, load_eval_cases(paths["holdout"]))
    integrity = analyze_route_holdout(training_records, holdout_records)
    artifact_training_sha = str(artifact.get("training_data_sha256", ""))
    artifact_sha = _file_sha256(paths["artifact"])
    artifact_matches_training = bool(
        artifact_training_sha
        and artifact_training_sha == integrity["training_data_sha256"]
    )
    configured_artifact_path = Path(moe_config.routing.distilled.artifact_path)
    artifact_path_matches_config = bool(
        moe_config.routing.distilled.enabled
        and configured_artifact_path.resolve() == paths["artifact"].resolve()
    )
    provenance = result.get("provenance", {})
    provenance_matches = {
        "config": provenance.get("config_sha256") == _file_sha256(paths["config"]),
        "holdout": (
            provenance.get("holdout_data_sha256")
            == integrity["holdout_data_sha256"]
        ),
        "training": (
            provenance.get("training_data_sha256")
            == integrity["training_data_sha256"]
        ),
        "artifact": (
            provenance.get("artifact_training_data_sha256")
            == artifact_training_sha
        ),
        "artifact_content": provenance.get("artifact_sha256") == artifact_sha,
        "artifact_path": _same_path(
            provenance.get("artifact_path"), paths["artifact"]
        ),
    }
    evaluation_fields = (
        "accuracy",
        "accuracy_ci95",
        "total",
        "by_complexity",
        "results",
    )
    report_matches_recomputed = all(
        result.get(field) == recomputed_result[field] for field in evaluation_fields
    )

    min_accuracy = float(raw.get("min_accuracy", 0.0))
    min_total = int(raw.get("min_total", 0))
    required_complexities = {str(item) for item in raw.get("required_complexities", [])}
    observed_complexities = set(result.get("by_complexity", {}).keys())
    accuracy = float(result.get("accuracy", 0.0))
    total = int(result.get("total", 0))
    missing_complexities = sorted(required_complexities - observed_complexities)
    failed_cases = [
        item
        for item in result.get("results", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    passed = bool(
        integrity["passed"]
        and artifact_matches_training
        and artifact_path_matches_config
        and all(provenance_matches.values())
        and report_matches_recomputed
        and accuracy >= min_accuracy
        and total >= min_total
        and not missing_complexities
    )
    return {
        "name": "routing_holdout",
        "passed": passed,
        "accuracy": accuracy,
        "accuracy_ci95": result.get("accuracy_ci95", {}),
        "min_accuracy": min_accuracy,
        "total": total,
        "min_total": min_total,
        "missing_complexities": missing_complexities,
        "failed_case_ids": [item.get("id") for item in failed_cases],
        "integrity": integrity,
        "artifact_matches_training": artifact_matches_training,
        "artifact_path_matches_config": artifact_path_matches_config,
        "report_matches_recomputed": report_matches_recomputed,
        "provenance_matches": provenance_matches,
    }


def _check_quality_benchmark(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "quality_benchmark",
            "passed": False,
            "release_eligible": False,
            "error": "quality_benchmark config must be an object",
        }

    mode = str(raw.get("mode", "required"))
    if mode not in {"required", "offline_optional"}:
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": "quality_benchmark.mode must be required or offline_optional",
        }

    result_path = Path(str(raw.get("result_path", "")))
    if not result_path.is_file():
        if mode == "offline_optional":
            return _offline_benchmark_skip(mode, "artifact_missing", result_path)
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": f"Missing quality benchmark result: {result_path}",
        }

    try:
        result = _read_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": f"Cannot read quality benchmark result {result_path}: {exc}",
        }
    if not isinstance(result, dict):
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": "Quality benchmark result must be an object",
        }

    artifact_status = str(result.get("status", "missing"))
    if artifact_status == "blocked" and mode == "offline_optional":
        blocked_validation = _validate_blocked_benchmark_artifact(result)
        if blocked_validation["passed"]:
            return _offline_benchmark_skip(
                mode,
                "artifact_blocked",
                result_path,
                artifact_status=artifact_status,
                validation=blocked_validation,
            )
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "status": "failed",
            "passed": False,
            "release_eligible": False,
            "artifact_status": artifact_status,
            "error": "Blocked quality benchmark artifact is structurally invalid",
            "blocked_artifact_validation": blocked_validation,
        }
    if artifact_status != "complete":
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "status": "failed",
            "passed": False,
            "release_eligible": False,
            "artifact_status": artifact_status,
            "error": (
                "Quality benchmark artifact status must be complete"
                + (
                    "; offline_optional only permits a missing artifact or a valid "
                    "blocked artifact"
                    if mode == "offline_optional"
                    else ""
                )
            ),
        }

    configured_paths = {
        "manifest": Path(str(raw.get("manifest_path", ""))),
        "benchmark_implementation": Path(
            str(raw.get("benchmark_implementation_path", ""))
        ),
        "evaluator_implementation": Path(
            str(raw.get("evaluator_implementation_path", ""))
        ),
    }
    missing_configured = [
        name for name, path in configured_paths.items() if not path.is_file()
    ]
    if missing_configured:
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": (
                "Missing quality benchmark inputs: "
                + ", ".join(sorted(missing_configured))
            ),
        }

    try:
        manifest = _read_json(configured_paths["manifest"])
        if not isinstance(manifest, dict):
            raise ValueError("benchmark manifest must be an object")
        source_config_path = Path(str(manifest.get("source_config", "")))
        dataset_path = Path(str(manifest.get("dataset", "")))
        if not source_config_path.is_file() or not dataset_path.is_file():
            raise ValueError("manifest source_config and dataset must exist")
        source_config = load_config(source_config_path)
        dataset_records = _read_jsonl(dataset_path)
        if not all(isinstance(item, dict) for item in dataset_records):
            raise ValueError("benchmark dataset records must be objects")
        benchmark_cases = load_benchmark_cases(dataset_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "name": "quality_benchmark",
            "mode": mode,
            "passed": False,
            "release_eligible": False,
            "error": f"Cannot validate quality benchmark inputs: {exc}",
        }

    case_ids = [case.id for case in benchmark_cases]
    variants_raw = manifest.get("variants", [])
    variants = (
        [str(item) for item in variants_raw]
        if isinstance(variants_raw, list)
        else []
    )
    try:
        repetitions = int(manifest.get("repetitions", 0))
    except (TypeError, ValueError):
        repetitions = 0
    general_expert_id = str(manifest.get("general_expert_id", "")).strip()
    decision = manifest.get("decision", {})
    evaluator = manifest.get("evaluator", {})
    try:
        quality_pass_threshold = float(
            evaluator.get("quality_pass_threshold", 0.7)
            if isinstance(evaluator, dict)
            else 0.7
        )
    except (TypeError, ValueError):
        quality_pass_threshold = -1.0
    store_outputs = manifest.get("store_outputs") is True
    dataset_is_coherent = bool(
        case_ids
        and all(case_ids)
        and len(case_ids) == len(set(case_ids))
        and variants
        and len(variants) == len(set(variants))
        and repetitions > 0
        and general_expert_id
        and isinstance(decision, dict)
        and isinstance(evaluator, dict)
        and evaluator.get("type") == "deterministic_rubric"
        and 0.0 <= quality_pass_threshold <= 1.0
    )
    expected_record_count = len(case_ids) * len(variants) * repetitions

    provenance = result.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    git_evidence = _check_git_evidence(
        provenance,
        raw.get("runtime_dependency_paths", []),
        mode=mode,
    )
    runtime_provenance = _check_runtime_provenance(
        provenance,
        source_config,
        raw.get("runtime_provenance", {}),
        mode=mode,
    )
    provenance_matches = {
        "manifest_path": _same_path(
            provenance.get("manifest_path"), configured_paths["manifest"]
        ),
        "manifest_content": (
            provenance.get("manifest_sha256")
            == _file_sha256(configured_paths["manifest"])
        ),
        "source_config_path": _same_path(
            provenance.get("source_config_path"), source_config_path
        ),
        "source_config_content": (
            provenance.get("source_config_sha256")
            == _file_sha256(source_config_path)
        ),
        "dataset_path": _same_path(provenance.get("dataset_path"), dataset_path),
        "dataset_content": (
            provenance.get("dataset_sha256") == _file_sha256(dataset_path)
        ),
        "case_ids": provenance.get("case_ids_sha256") == _sha256_json(case_ids),
        "case_count": provenance.get("case_count") == len(case_ids),
        "variants": provenance.get("variants") == variants,
        "repetitions": provenance.get("repetitions") == repetitions,
        "generation_overrides": (
            provenance.get("generation_overrides")
            == manifest.get("generation_overrides", {})
        ),
        "benchmark_implementation": (
            provenance.get("benchmark_implementation_sha256")
            == _file_sha256(configured_paths["benchmark_implementation"])
        ),
        "evaluator_implementation": (
            provenance.get("evaluator_implementation_sha256")
            == _file_sha256(configured_paths["evaluator_implementation"])
        ),
        "git_evidence": git_evidence["passed"],
        "runtime_environment": runtime_provenance["passed"],
    }

    execution = result.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    records_raw = execution.get("records", [])
    records = records_raw if isinstance(records_raw, list) else []
    expected_keys = Counter(
        (variant, case_id, repetition)
        for repetition in range(1, repetitions + 1)
        for case_id in case_ids
        for variant in variants
    )
    observed_keys: Counter[tuple[str, str, int] | None] = Counter()
    for record in records:
        if not isinstance(record, dict):
            observed_keys[None] += 1
            continue
        try:
            repetition = int(record.get("repetition"))
        except (TypeError, ValueError):
            observed_keys[None] += 1
            continue
        observed_keys[
            (str(record.get("variant", "")), str(record.get("case_id", "")), repetition)
        ] += 1

    record_count_matches = bool(
        dataset_is_coherent
        and execution.get("planned_records") == expected_record_count
        and len(records) == expected_record_count
    )
    record_coverage_matches = bool(dataset_is_coherent and observed_keys == expected_keys)
    records_per_variant = len(case_ids) * repetitions
    metrics = result.get("metrics", {})
    metrics_planned_matches = bool(
        isinstance(metrics, dict)
        and set(metrics) == set(variants)
        and all(
            isinstance(metrics.get(variant), dict)
            and metrics[variant].get("planned") == records_per_variant
            for variant in variants
        )
    )

    gate = result.get("gate", {})
    deterministic_validation = result.get("deterministic_validation", {})
    readiness = result.get("readiness", {})
    comparisons = result.get("comparisons", {})
    recomputation = _recompute_quality_evidence(
        records,
        variants,
        cases_by_id={case.id: case for case in benchmark_cases},
        general_expert_id=general_expert_id,
        quality_pass_threshold=quality_pass_threshold,
        decision=decision if isinstance(decision, dict) else {},
        host_memory=execution.get("host_memory"),
        can_recompute=record_count_matches and record_coverage_matches,
    )
    evidence_matches = {
        "metrics": (
            recomputation.get("status") == "complete"
            and metrics == recomputation.get("metrics")
        ),
        "comparisons": (
            recomputation.get("status") == "complete"
            and comparisons == recomputation.get("comparisons")
        ),
        "gate": (
            recomputation.get("status") == "complete"
            and gate == recomputation.get("gate")
        ),
    }
    checks = {
        "artifact_complete": artifact_status == "complete",
        "schema_matches_manifest": (
            result.get("schema_version") == manifest.get("schema_version")
        ),
        "deterministic_validation_passed": (
            isinstance(deterministic_validation, dict)
            and deterministic_validation.get("status") == "passed"
        ),
        "readiness_ready": (
            isinstance(readiness, dict) and readiness.get("status") == "ready"
        ),
        "execution_complete": execution.get("status") == "complete",
        "record_count_matches": record_count_matches,
        "record_coverage_matches": record_coverage_matches,
        "release_outputs_stored": mode != "required" or store_outputs,
        "metrics_planned_matches": metrics_planned_matches,
        "record_judgments_match": (
            recomputation.get("judgment_validation", {}).get("passed") is True
        ),
        "evidence_recomputed": recomputation.get("status") == "complete",
        "metrics_match_recomputed": evidence_matches["metrics"],
        "comparisons_match_recomputed": evidence_matches["comparisons"],
        "gate_matches_recomputed": evidence_matches["gate"],
        "comparisons_complete": (
            isinstance(comparisons, dict) and comparisons.get("status") == "complete"
        ),
        "gate_passed": (
            isinstance(gate, dict)
            and gate.get("status") == "passed"
            and gate.get("passed") is True
        ),
        "provenance_matches": all(provenance_matches.values()),
    }
    passed = all(checks.values())
    return {
        "name": "quality_benchmark",
        "mode": mode,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "release_eligible": bool(mode == "required" and passed),
        "artifact_status": artifact_status,
        "expected_record_count": expected_record_count,
        "observed_record_count": len(records),
        "checks": checks,
        "provenance_matches": provenance_matches,
        "git_evidence": git_evidence,
        "runtime_provenance": runtime_provenance,
        "evidence_matches": evidence_matches,
        "judgment_validation": recomputation.get("judgment_validation"),
        "recomputation_error": recomputation.get("error"),
    }


def _recompute_quality_evidence(
    records: list[dict[str, Any]],
    variants: list[str],
    *,
    cases_by_id: dict[str, BenchmarkCase],
    general_expert_id: str,
    quality_pass_threshold: float,
    decision: dict[str, Any],
    host_memory: Any,
    can_recompute: bool,
) -> dict[str, Any]:
    if not can_recompute:
        return {
            "status": "failed",
            "error": "record count or coverage does not match the benchmark manifest",
        }
    judgment_validation = _reevaluate_record_judgments(
        records,
        cases_by_id,
        quality_pass_threshold=quality_pass_threshold,
    )
    if not judgment_validation["passed"]:
        return {
            "status": "failed",
            "judgment_validation": judgment_validation,
            "error": "record judgments do not match deterministic output re-evaluation",
        }
    try:
        metrics = summarize_records(
            records,
            variants,
            general_expert_id=general_expert_id,
        )
        comparisons = compare_to_baseline(metrics, decision)
        gate = evaluate_benchmark_gate(
            metrics,
            comparisons,
            decision,
            host_memory=host_memory if isinstance(host_memory, dict) else None,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "judgment_validation": judgment_validation,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "complete",
        "judgment_validation": judgment_validation,
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
    }


def _reevaluate_record_judgments(
    records: list[dict[str, Any]],
    cases_by_id: dict[str, BenchmarkCase],
    *,
    quality_pass_threshold: float,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for record in records:
        if not isinstance(record, dict):
            mismatches.append({"record": None, "fields": ["record_object"]})
            continue
        record_key = {
            "variant": record.get("variant"),
            "case_id": record.get("case_id"),
            "repetition": record.get("repetition"),
        }
        execution = record.get("execution")
        if not isinstance(execution, dict):
            mismatches.append({"record": record_key, "fields": ["execution"]})
            continue
        if execution.get("status") != "ok":
            continue
        checked += 1
        case_id = str(record.get("case_id", ""))
        case = cases_by_id.get(case_id)
        output = execution.get("output")
        if case is None or not isinstance(output, str):
            fields = []
            if case is None:
                fields.append("case_id")
            if not isinstance(output, str):
                fields.append("execution.output")
            mismatches.append({"record": record_key, "fields": fields})
            continue
        expected_task, expected_quality = evaluate_case_output(
            case,
            output,
            quality_pass_threshold=quality_pass_threshold,
        )
        fields = []
        if record.get("category") != case.category:
            fields.append("category")
        if record.get("complexity") != case.complexity:
            fields.append("complexity")
        if record.get("task_validation") != expected_task:
            fields.append("task_validation")
        if record.get("quality_judgment") != expected_quality:
            fields.append("quality_judgment")
        if fields:
            mismatches.append({"record": record_key, "fields": fields})
    return {
        "passed": not mismatches,
        "successful_records_checked": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _validate_blocked_benchmark_artifact(result: dict[str, Any]) -> dict[str, Any]:
    provenance = result.get("provenance")
    deterministic_validation = result.get("deterministic_validation")
    readiness = result.get("readiness")
    execution = result.get("execution")
    comparisons = result.get("comparisons")
    gate = result.get("gate")
    checks = {
        "schema_version_present": (
            isinstance(result.get("schema_version"), int)
            and not isinstance(result.get("schema_version"), bool)
        ),
        "created_at_present": bool(
            isinstance(result.get("created_at"), str) and result["created_at"].strip()
        ),
        "provenance_present": isinstance(provenance, dict) and bool(provenance),
        "deterministic_validation_explicit": (
            isinstance(deterministic_validation, dict)
            and deterministic_validation.get("status") in {"passed", "failed"}
        ),
        "readiness_explicit": (
            isinstance(readiness, dict)
            and readiness.get("status") in {"blocked", "not_run"}
        ),
        "execution_not_run": (
            isinstance(execution, dict)
            and execution.get("status") == "not_run"
            and execution.get("planned_records") == 0
            and execution.get("records") == []
        ),
        "metrics_empty": result.get("metrics") == {},
        "comparisons_not_run": (
            isinstance(comparisons, dict)
            and comparisons.get("status") == "not_run"
        ),
        "gate_blocked": (
            isinstance(gate, dict)
            and gate.get("status") == "blocked"
            and gate.get("passed") is False
            and isinstance(gate.get("reason"), str)
            and bool(gate["reason"].strip())
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _check_runtime_provenance(
    provenance: dict[str, Any],
    source_config: MoEConfig,
    raw_policy: object,
    *,
    mode: str,
) -> dict[str, Any]:
    if not isinstance(raw_policy, dict) or not raw_policy:
        return {
            "status": "not_required",
            "passed": True,
            "release_eligible": mode == "required",
        }

    required_packages_raw = raw_policy.get("required_packages", [])
    required_packages = (
        [str(item).strip() for item in required_packages_raw]
        if isinstance(required_packages_raw, list)
        else []
    )
    require_model_revision = raw_policy.get("require_model_snapshot_revision") is True
    verify_current_release = raw_policy.get("verify_current_environment_in_release") is True
    policy_valid = bool(
        required_packages
        and all(required_packages)
        and len(required_packages) == len(set(required_packages))
        and require_model_revision
        and verify_current_release
    )

    stored_runtime = provenance.get("runtime_environment")
    runtime = stored_runtime if isinstance(stored_runtime, dict) else {}
    stored_python = runtime.get("python")
    python_valid = bool(
        isinstance(stored_python, dict)
        and isinstance(stored_python.get("implementation"), str)
        and bool(stored_python["implementation"].strip())
        and isinstance(stored_python.get("version"), str)
        and bool(stored_python["version"].strip())
    )
    stored_platform = runtime.get("platform")
    platform_valid = bool(
        isinstance(stored_platform, dict)
        and isinstance(stored_platform.get("system"), str)
        and bool(stored_platform["system"].strip())
        and isinstance(stored_platform.get("machine"), str)
        and bool(stored_platform["machine"].strip())
    )
    stored_packages = runtime.get("packages")
    packages_valid = bool(
        isinstance(stored_packages, dict)
        and all(
            isinstance(stored_packages.get(package), dict)
            and stored_packages[package].get("status") == "installed"
            and isinstance(stored_packages[package].get("version"), str)
            and bool(stored_packages[package]["version"].strip())
            for package in required_packages
        )
    )
    server_runtime = runtime.get("server_runtime_identity")
    server_runtime_explicit = bool(
        isinstance(server_runtime, dict)
        and server_runtime.get("status") == "unverified"
        and isinstance(server_runtime.get("reason_code"), str)
        and bool(server_runtime["reason_code"].strip())
    )

    snapshots_raw = provenance.get("model_snapshots")
    snapshots = snapshots_raw if isinstance(snapshots_raw, list) else []
    snapshots_valid = _stored_model_snapshots_valid(
        snapshots,
        source_config,
        require_revision=require_model_revision,
    )

    current_runtime_matches: bool | None = None
    current_snapshots_match: bool | None = None
    current_runtime: dict[str, Any] | None = None
    current_snapshots: list[dict[str, Any]] | None = None
    if mode == "required" and verify_current_release:
        current_runtime = collect_runtime_environment_provenance()
        current_snapshots = collect_model_snapshot_provenance(source_config)
        current_runtime_matches = bool(
            python_valid
            and platform_valid
            and packages_valid
            and stored_python == current_runtime.get("python")
            and stored_platform == current_runtime.get("platform")
            and all(
                stored_packages.get(package)
                == current_runtime.get("packages", {}).get(package)
                for package in required_packages
            )
        )
        current_snapshots_match = snapshots == current_snapshots

    stored_evidence_valid = bool(
        policy_valid
        and python_valid
        and platform_valid
        and packages_valid
        and server_runtime_explicit
        and snapshots_valid
    )
    if mode == "required":
        current_verification_passed = bool(
            current_runtime_matches is True and current_snapshots_match is True
        )
    else:
        current_verification_passed = True
    passed = bool(stored_evidence_valid and current_verification_passed)
    return {
        "status": (
            "passed"
            if passed and mode == "required"
            else "deferred_non_release"
            if passed
            else "failed"
        ),
        "passed": passed,
        "release_eligible": bool(mode == "required" and passed),
        "policy_valid": policy_valid,
        "python_valid": python_valid,
        "platform_valid": platform_valid,
        "required_packages": required_packages,
        "packages_valid": packages_valid,
        "server_runtime_identity_explicitly_unverified": server_runtime_explicit,
        "model_snapshots_valid": snapshots_valid,
        "current_runtime_matches": current_runtime_matches,
        "current_model_snapshots_match": current_snapshots_match,
        "verification_scope": (
            "current_environment_and_local_model_cache"
            if mode == "required"
            else "stored_evidence_only_non_release"
        ),
        "server_runtime_limitation": (
            "The OpenAI-compatible models endpoint does not expose inference "
            "server package identity; this remains explicitly unverified."
        ),
    }


def _stored_model_snapshots_valid(
    snapshots: list[Any],
    source_config: MoEConfig,
    *,
    require_revision: bool,
) -> bool:
    if len(snapshots) != len(source_config.experts):
        return False
    by_id = {
        str(item.get("expert_id", "")): item
        for item in snapshots
        if isinstance(item, dict)
    }
    if len(by_id) != len(snapshots):
        return False
    for expert in source_config.experts:
        item = by_id.get(expert.id)
        if item is None:
            return False
        if (
            item.get("provider") != expert.provider
            or item.get("model") != expert.model
            or item.get("runtime_backend")
            != str(expert.params.get("runtime_backend", "unknown"))
        ):
            return False
        if expert.provider == "synthetic":
            if item.get("status") != "not_required":
                return False
            continue
        revision = item.get("revision")
        serving_match_explicit = bool(
            item.get("serving_match_status") == "unverified"
            and isinstance(item.get("serving_match_reason_code"), str)
            and bool(item["serving_match_reason_code"].strip())
        )
        if require_revision and not (
            item.get("status") == "resolved"
            and item.get("identity_type") == "huggingface_snapshot_revision"
            and isinstance(revision, str)
            and len(revision) in {40, 64}
            and all(character in "0123456789abcdefABCDEF" for character in revision)
            and serving_match_explicit
        ):
            return False
    return True


def _check_git_evidence(
    provenance: dict[str, Any],
    raw_dependency_paths: object,
    *,
    mode: str,
) -> dict[str, Any]:
    dependency_paths = _validated_dependency_paths(raw_dependency_paths)
    commit = provenance.get("git_commit")
    commit_text = commit.strip() if isinstance(commit, str) else ""
    clean_at_generation = provenance.get("git_dirty") is False
    commit_format_valid = bool(
        len(commit_text) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in commit_text)
    )
    root = _git_repository_root()
    base = {
        "git_dirty_is_false": clean_at_generation,
        "git_commit": commit_text or None,
        "git_commit_format_valid": commit_format_valid,
        "runtime_dependency_paths_valid": dependency_paths is not None,
        "runtime_dependency_count": len(dependency_paths or ()),
    }
    if root is None or dependency_paths is None or not clean_at_generation or not commit_format_valid:
        return {
            **base,
            "passed": False,
            "status": "failed",
            "error": "invalid git provenance or runtime dependency configuration",
        }

    current_head = _git_text(root, "rev-parse", "HEAD")
    shallow = _git_text(root, "rev-parse", "--is-shallow-repository") == "true"
    resolved_commit = _git_text(root, "rev-parse", "--verify", f"{commit_text}^{{commit}}")
    commit_available = resolved_commit == commit_text.lower()
    if not commit_available:
        if mode == "offline_optional" and shallow:
            current_hashes = _current_dependency_hashes(root, dependency_paths)
            current_dependencies_valid = bool(
                current_hashes
                and len(current_hashes) == len(dependency_paths)
                and all(
                    item.get("current_head_sha256")
                    and item.get("working_tree_clean") is True
                    for item in current_hashes.values()
                )
            )
            return {
                **base,
                "passed": current_dependencies_valid,
                "status": (
                    "deferred_shallow_non_release"
                    if current_dependencies_valid
                    else "failed"
                ),
                "repository_root": str(root),
                "current_head": current_head,
                "repository_shallow": True,
                "commit_available": False,
                "ancestor_of_head": None,
                "runtime_dependencies": current_hashes,
                "release_eligible": False,
                "reason": (
                    "historical commit is unavailable in a shallow offline CI checkout; "
                    "release verification remains disabled"
                ),
            }
        return {
            **base,
            "passed": False,
            "status": "failed",
            "repository_root": str(root),
            "current_head": current_head,
            "repository_shallow": shallow,
            "commit_available": False,
            "error": "benchmark git commit is unavailable",
        }

    ancestor = _git_returncode(
        root,
        "merge-base",
        "--is-ancestor",
        commit_text,
        "HEAD",
    ) == 0
    dependencies = _compare_dependencies_at_commit(root, commit_text, dependency_paths)
    dependencies_match = bool(
        len(dependencies) == len(dependency_paths)
        and all(item.get("matches") is True for item in dependencies.values())
    )
    passed = bool(ancestor and dependencies_match)
    return {
        **base,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "repository_root": str(root),
        "current_head": current_head,
        "repository_shallow": shallow,
        "commit_available": True,
        "ancestor_of_head": ancestor,
        "runtime_dependencies": dependencies,
        "release_eligible": passed,
    }


def _validated_dependency_paths(raw: object) -> tuple[str, ...] | None:
    if not isinstance(raw, list) or not raw:
        return None
    values = tuple(str(item).strip() for item in raw)
    if any(not item for item in values) or len(values) != len(set(values)):
        return None
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in values):
        return None
    return values


def _git_repository_root() -> Path | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return Path(completed.stdout.strip()).resolve()


def _git_text(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_returncode(root: Path, *args: str) -> int:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
    ).returncode


def _git_blob(root: Path, commit: str, relative_path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def _current_dependency_hashes(
    root: Path,
    dependency_paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_path in dependency_paths:
        path = (root / raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            result[raw_path] = {
                "current_head_sha256": None,
                "working_tree_clean": False,
                "error": "outside repository",
            }
            continue
        head_blob = _git_blob(root, "HEAD", relative)
        result[relative] = {
            "current_head_sha256": (
                sha256(head_blob).hexdigest() if head_blob is not None else None
            ),
            "working_tree_clean": bool(
                path.is_file()
                and _git_returncode(
                    root,
                    "diff",
                    "--quiet",
                    "--no-ext-diff",
                    "HEAD",
                    "--",
                    relative,
                )
                == 0
            ),
        }
    return result


def _compare_dependencies_at_commit(
    root: Path,
    commit: str,
    dependency_paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    current = _current_dependency_hashes(root, dependency_paths)
    result: dict[str, dict[str, Any]] = {}
    for relative, values in current.items():
        blob = _git_blob(root, commit, relative)
        committed_sha = sha256(blob).hexdigest() if blob is not None else None
        current_sha = values.get("current_head_sha256")
        working_tree_clean = values.get("working_tree_clean") is True
        result[relative] = {
            "current_head_sha256": current_sha,
            "benchmark_commit_sha256": committed_sha,
            "working_tree_clean": working_tree_clean,
            "matches": bool(
                current_sha
                and committed_sha
                and current_sha == committed_sha
                and working_tree_clean
            ),
        }
    return result


def _offline_benchmark_skip(
    mode: str,
    reason: str,
    result_path: Path,
    *,
    artifact_status: str = "missing",
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": "quality_benchmark",
        "mode": mode,
        "status": "skipped",
        "passed": True,
        "release_eligible": False,
        "artifact_status": artifact_status,
        "result_path": str(result_path),
        "reason": reason,
    }
    if validation is not None:
        payload["blocked_artifact_validation"] = validation
    return payload


def _check_forbidden_listeners(listeners: object) -> dict[str, Any]:
    active: list[dict[str, object]] = []
    for raw in listeners:
        if not isinstance(raw, dict):
            continue
        host = str(raw.get("host", "127.0.0.1"))
        port = int(raw["port"])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                active.append(
                    {
                        "host": host,
                        "port": port,
                        "name": raw.get("name", "unnamed"),
                    }
                )

    return {
        "name": "forbidden_listeners",
        "passed": not active,
        "active": active,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _same_path(raw: object, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    return Path(raw).resolve() == expected.resolve()


if __name__ == "__main__":
    main()
