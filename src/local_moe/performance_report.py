from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .performance import BenchmarkManifest, load_benchmark_manifest, summarize_benchmarks


def build_performance_report(
    *,
    manifest_path: str | Path = "configs/model-benchmark.json",
    benchmark_path: str | Path = "outputs/performance-benchmark.json",
    hardware_profile_path: str | Path = "outputs/hardware-profile.json",
    decision_markdown_path: str | Path = "outputs/performance-decision.md",
) -> dict[str, Any]:
    manifest_artifact = _load_manifest(manifest_path)
    if manifest_artifact["status"] != "available":
        return {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "status": "blocked",
            "manifest": manifest_artifact,
            "benchmark": _read_json_artifact(benchmark_path, include_data=False),
            "hardware_profile": _read_json_artifact(hardware_profile_path, include_data=True),
            "decision_markdown": _text_artifact(decision_markdown_path),
            "coverage": {
                "status": "missing_manifest",
                "manifest_candidate_count": 0,
                "ranked_candidate_count": 0,
                "measured_result_count": 0,
                "ok_count": 0,
                "failed_count": 0,
                "not_run_count": 0,
                "missing_result_ids": [],
                "failed_candidate_ids": [],
                "not_run_candidate_ids": [],
            },
            "decision": {
                "primary_general": None,
                "fast_fallback": None,
                "recommended_architecture": "",
            },
            "ranked": [],
            "recommendations": ["Restore configs/model-benchmark.json before trusting local model decisions."],
        }

    manifest = manifest_artifact["manifest_object"]
    benchmark_artifact = _read_json_artifact(benchmark_path, include_data=True)
    benchmark_data = benchmark_artifact.get("data") if benchmark_artifact["status"] == "available" else {}
    if not isinstance(benchmark_data, dict):
        benchmark_data = {}
    raw_results = benchmark_data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    raw_summary = benchmark_data.get("summary")
    if not isinstance(raw_summary, dict):
        raw_summary = summarize_benchmarks(manifest, raw_results)

    summary = _sanitize_summary(raw_summary)
    coverage = _coverage(manifest, summary["ranked"], raw_results)
    decision = _decision(summary)
    status = _status(benchmark_artifact["status"], decision, coverage)

    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "status": status,
        "manifest": _manifest_payload(manifest, manifest_path),
        "benchmark": _benchmark_payload(benchmark_artifact, benchmark_data, raw_results),
        "hardware_profile": _read_json_artifact(hardware_profile_path, include_data=True),
        "decision_markdown": _text_artifact(decision_markdown_path),
        "coverage": coverage,
        "decision": decision,
        "ranked": summary["ranked"],
        "recommendations": _recommendations(status, coverage, decision),
    }


def performance_report_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"mymoe-performance-report-{stamp}.md"


def render_performance_report_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision", {})
    coverage = report.get("coverage", {})
    benchmark = report.get("benchmark", {})
    hardware = report.get("hardware_profile", {})
    hardware_data = hardware.get("data", {}) if isinstance(hardware, dict) else {}
    lines = [
        "# myMoE Performance Report",
        "",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Generated: `{report.get('generated_at', 'unknown')}`",
        f"Benchmark: `{benchmark.get('path', 'unknown')}` / `{benchmark.get('status', 'unknown')}`",
        f"Benchmark created: `{benchmark.get('created_at', 'unknown')}`",
        f"Machine: `{hardware_data.get('cpu_brand', 'unknown')}` / `{hardware_data.get('machine', 'unknown')}` / `{hardware_data.get('memory_gib', 'unknown')} GiB RAM`",
        "",
        "## Decision",
        "",
        f"- Primary general expert: `{_label(decision.get('primary_general'))}`",
        f"- Fast fallback/compaction expert: `{_label(decision.get('fast_fallback'))}`",
        f"- Architecture: {decision.get('recommended_architecture') or 'No recommendation available.'}",
        "",
        "## Coverage",
        "",
        f"- Coverage status: `{coverage.get('status', 'unknown')}`",
        f"- Manifest candidates: `{coverage.get('manifest_candidate_count', 0)}`",
        f"- Measured results: `{coverage.get('measured_result_count', 0)}`",
        f"- OK: `{coverage.get('ok_count', 0)}`",
        f"- Failed: `{coverage.get('failed_count', 0)}`",
        f"- Not run: `{coverage.get('not_run_count', 0)}`",
        "",
        "## Ranked Results",
        "",
        "| Rank | Candidate | Role | Status | Score | Tok/s | Peak GB | Load s |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(report.get("ranked", []), start=1):
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        score = item.get("score", {}) if isinstance(item, dict) else {}
        lines.append(
            "| {rank} | `{label}` | `{role}` | `{status}` | {score} | {tps} | {mem} | {load} |".format(
                rank=index,
                label=item.get("label", item.get("candidate_id", "unknown")),
                role=item.get("role", "unknown"),
                status=item.get("status", "unknown"),
                score=_fmt(score.get("overall")),
                tps=_fmt(metrics.get("generation_tps_avg")),
                mem=_fmt(metrics.get("peak_memory_gb")),
                load=_fmt(metrics.get("load_seconds")),
            )
        )
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines) + "\n"


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"path": str(manifest_path), "status": "missing"}
    try:
        manifest = load_benchmark_manifest(manifest_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(manifest_path), "status": "invalid", "error": str(exc)}
    payload = _manifest_payload(manifest, manifest_path)
    payload["manifest_object"] = manifest
    return payload


def _manifest_payload(manifest: BenchmarkManifest, path: str | Path) -> dict[str, Any]:
    categories = sorted({prompt.category for prompt in manifest.prompts})
    return {
        "path": str(path),
        "status": "available",
        "hardware_budget_gb": manifest.hardware_budget_gb,
        "prompt_count": len(manifest.prompts),
        "prompt_categories": categories,
        "candidate_count": len(manifest.candidates),
        "candidates": [
            {
                "candidate_id": candidate.id,
                "label": candidate.label,
                "role": candidate.role,
                "runtime": candidate.runtime,
                "repo": candidate.repo,
                "estimated_memory_gb": candidate.estimated_memory_gb,
                "quality_prior": candidate.quality_prior,
                "speed_target_tps": candidate.speed_target_tps,
                "source_url": candidate.source_url,
            }
            for candidate in manifest.candidates
        ],
    }


def _read_json_artifact(path: str | Path, *, include_data: bool) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"path": str(artifact_path), "status": "missing"}
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(artifact_path), "status": "invalid_json", "error": str(exc)}
    payload: dict[str, Any] = {"path": str(artifact_path), "status": "available"}
    if include_data:
        payload["data"] = data
    return payload


def _text_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"path": str(artifact_path), "status": "missing", "size_bytes": 0}
    return {
        "path": str(artifact_path),
        "status": "available",
        "size_bytes": artifact_path.stat().st_size,
    }


def _benchmark_payload(
    artifact: dict[str, Any],
    data: dict[str, Any],
    results: list[object],
) -> dict[str, Any]:
    if artifact["status"] != "available":
        return artifact
    return {
        "path": artifact["path"],
        "status": "available",
        "created_at": data.get("created_at"),
        "manifest": data.get("manifest"),
        "max_tokens": data.get("max_tokens"),
        "max_kv_size": data.get("max_kv_size"),
        "prompt_count": data.get("prompt_count"),
        "result_count": len(results),
    }


def _sanitize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    ranked = summary.get("ranked", [])
    if not isinstance(ranked, list):
        ranked = []
    return {
        "ranked": [_ranked_payload(item) for item in ranked if isinstance(item, dict)],
        "decision": summary.get("decision", {}) if isinstance(summary.get("decision"), dict) else {},
    }


def _ranked_payload(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
    aggregate = result.get("aggregate", {}) if isinstance(result.get("aggregate"), dict) else {}
    return {
        "candidate_id": item.get("candidate_id"),
        "label": item.get("label"),
        "role": item.get("role"),
        "repo": item.get("repo"),
        "status": item.get("status"),
        "score": item.get("score", {}),
        "metrics": {
            "generation_tps_avg": aggregate.get("generation_tps_avg"),
            "prompt_tps_avg": aggregate.get("prompt_tps_avg"),
            "peak_memory_gb": aggregate.get("peak_memory_gb"),
            "latency_seconds_avg": aggregate.get("latency_seconds_avg"),
            "successful_prompts": aggregate.get("successful_prompts"),
            "failed_prompts": aggregate.get("failed_prompts"),
            "generation_tokens_total": aggregate.get("generation_tokens_total"),
            "load_seconds": result.get("load_seconds"),
            "total_seconds": result.get("total_seconds"),
        },
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    ranked_by_id = {item.get("candidate_id"): item for item in summary["ranked"]}
    raw_decision = summary.get("decision", {})
    primary_raw = raw_decision.get("primary_general")
    fallback_raw = raw_decision.get("fast_fallback")
    primary_id = primary_raw.get("candidate_id") if isinstance(primary_raw, dict) else None
    fallback_id = fallback_raw.get("candidate_id") if isinstance(fallback_raw, dict) else None
    return {
        "primary_general": ranked_by_id.get(primary_id),
        "fast_fallback": ranked_by_id.get(fallback_id),
        "recommended_architecture": raw_decision.get("recommended_architecture", ""),
    }


def _coverage(
    manifest: BenchmarkManifest,
    ranked: list[dict[str, Any]],
    raw_results: list[object],
) -> dict[str, Any]:
    manifest_ids = {candidate.id for candidate in manifest.candidates}
    ranked_ids = {str(item.get("candidate_id")) for item in ranked if item.get("candidate_id")}
    measured_ids = {
        str(item.get("candidate_id"))
        for item in raw_results
        if isinstance(item, dict) and item.get("candidate_id")
    }
    failed_ids = sorted(
        str(item.get("candidate_id"))
        for item in ranked
        if item.get("status") == "failed" and item.get("candidate_id")
    )
    not_run_ids = sorted(
        str(item.get("candidate_id"))
        for item in ranked
        if item.get("status") == "not_run" and item.get("candidate_id")
    )
    missing_result_ids = sorted(manifest_ids - measured_ids)
    coverage_status = "complete" if not missing_result_ids and not failed_ids else "partial"
    if not measured_ids:
        coverage_status = "missing"
    return {
        "status": coverage_status,
        "manifest_candidate_count": len(manifest_ids),
        "ranked_candidate_count": len(ranked_ids),
        "measured_result_count": len(measured_ids),
        "ok_count": sum(1 for item in ranked if item.get("status") == "ok"),
        "failed_count": len(failed_ids),
        "not_run_count": len(not_run_ids),
        "missing_result_ids": missing_result_ids,
        "failed_candidate_ids": failed_ids,
        "not_run_candidate_ids": not_run_ids,
    }


def _status(
    benchmark_status: str,
    decision: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    if benchmark_status != "available":
        return "missing"
    if not decision.get("primary_general") or not decision.get("fast_fallback"):
        return "attention"
    if coverage.get("status") == "complete":
        return "ready"
    return "ready_partial"


def _recommendations(
    status: str,
    coverage: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if status == "missing":
        recommendations.append("Run make benchmark-small or experiments/benchmark_models.py before changing default model policy.")
    if status == "attention":
        recommendations.append("Rebuild outputs/performance-benchmark.json until both primary and fallback decisions are available.")
    if coverage.get("not_run_candidate_ids"):
        recommendations.append("Benchmark not-run stretch candidates only when they are realistic for the local memory budget.")
    if coverage.get("failed_candidate_ids"):
        recommendations.append("Keep failed candidates out of default profiles until the failure is reproduced and fixed.")
    if decision.get("primary_general") and decision.get("fast_fallback"):
        recommendations.append("Keep one heavy resident general expert and one small resident fallback unless new evals beat this policy.")
    return recommendations


def _label(item: object) -> str:
    if not isinstance(item, dict):
        return "none"
    return str(item.get("label") or item.get("candidate_id") or "unknown")


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
