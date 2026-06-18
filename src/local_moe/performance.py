from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCandidate:
    id: str
    label: str
    role: str
    runtime: str
    repo: str
    estimated_size_gb: float
    estimated_memory_gb: float
    quality_prior: float
    speed_target_tps: float
    source_url: str
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkManifest:
    hardware_budget_gb: float
    decision_weights: dict[str, float]
    prompts: list[BenchmarkPrompt]
    candidates: list[BenchmarkCandidate]


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkManifest(
        hardware_budget_gb=float(raw.get("hardware_budget_gb", 24.0)),
        decision_weights={key: float(value) for key, value in raw.get("decision_weights", {}).items()},
        prompts=[
            BenchmarkPrompt(
                id=str(item["id"]),
                category=str(item.get("category", "general")),
                prompt=str(item["prompt"]),
            )
            for item in raw.get("prompts", [])
        ],
        candidates=[
            BenchmarkCandidate(
                id=str(item["id"]),
                label=str(item["label"]),
                role=str(item["role"]),
                runtime=str(item["runtime"]),
                repo=str(item["repo"]),
                estimated_size_gb=float(item.get("estimated_size_gb", 0.0)),
                estimated_memory_gb=float(item.get("estimated_memory_gb", 0.0)),
                quality_prior=float(item.get("quality_prior", 0.0)),
                speed_target_tps=float(item.get("speed_target_tps", 1.0)),
                source_url=str(item.get("source_url", "")),
                notes=str(item.get("notes", "")),
            )
            for item in raw.get("candidates", [])
        ],
    )


def score_candidate(
    candidate: BenchmarkCandidate,
    result: dict[str, Any],
    *,
    hardware_budget_gb: float,
    weights: dict[str, float],
) -> dict[str, float]:
    if result.get("status") != "ok":
        return {
            "overall": 0.0,
            "quality_prior": candidate.quality_prior,
            "speed": 0.0,
            "memory_headroom": 0.0,
            "load_time": 0.0,
            "reliability": 0.0,
        }

    aggregate = result.get("aggregate", {}) if isinstance(result.get("aggregate"), dict) else {}
    generation_tps = _float(aggregate.get("generation_tps_avg"))
    peak_memory_gb = _float(aggregate.get("peak_memory_gb"))
    load_seconds = _float(result.get("load_seconds"))

    memory_gb = peak_memory_gb or candidate.estimated_memory_gb
    speed_score = _clamp(generation_tps / max(candidate.speed_target_tps, 1.0))
    headroom_score = _clamp((hardware_budget_gb - memory_gb) / 8.0)
    load_score = _clamp(1.0 - (load_seconds / 180.0))

    parts = {
        "quality_prior": _clamp(candidate.quality_prior),
        "speed": speed_score,
        "memory_headroom": headroom_score,
        "load_time": load_score,
        "reliability": 1.0,
    }
    total_weight = sum(weights.get(key, 0.0) for key in parts) or 1.0
    overall = sum(parts[key] * weights.get(key, 0.0) for key in parts) / total_weight
    parts["overall"] = overall
    return {key: round(value, 4) for key, value in parts.items()}


def summarize_benchmarks(manifest: BenchmarkManifest, results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["candidate_id"]: item for item in results}
    scored: list[dict[str, Any]] = []

    for candidate in manifest.candidates:
        result = by_id.get(candidate.id, {"candidate_id": candidate.id, "status": "not_run"})
        score = score_candidate(
            candidate,
            result,
            hardware_budget_gb=manifest.hardware_budget_gb,
            weights=manifest.decision_weights,
        )
        scored.append(
            {
                "candidate_id": candidate.id,
                "label": candidate.label,
                "role": candidate.role,
                "repo": candidate.repo,
                "status": result.get("status", "not_run"),
                "score": score,
                "result": result,
            }
        )

    ranked = sorted(scored, key=lambda item: item["score"]["overall"], reverse=True)
    primary_roles = {"primary_general", "primary_general_alternative"}
    fallback_roles = {"fast_compaction_or_fallback"}
    primary = _first_ok([item for item in ranked if item["role"] in primary_roles])
    fallback = _first_ok([item for item in ranked if item["role"] in fallback_roles])

    return {
        "hardware_budget_gb": manifest.hardware_budget_gb,
        "ranked": ranked,
        "decision": {
            "primary_general": primary,
            "fast_fallback": fallback,
            "recommended_architecture": (
                "one resident heavy general expert plus one small resident "
                "fallback/compaction expert; cold-load specialists only after eval wins"
            ),
        },
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    primary = decision.get("primary_general")
    fallback = decision.get("fast_fallback")
    hardware = summary.get("hardware", {})
    lines = [
        "# Performance Decision",
        "",
        f"Hardware budget: `{summary['hardware_budget_gb']:.1f} GiB` unified memory.",
        f"Tested machine: `{hardware.get('cpu_brand', 'unknown')}` / `{hardware.get('machine', 'unknown')}` / `{hardware.get('memory_gib', 'unknown')} GiB RAM`.",
        "",
        "## Decision",
        "",
        f"- Primary general expert: `{_label(primary)}`",
        f"- Fast fallback/compaction expert: `{_label(fallback)}`",
        f"- Architecture: {decision['recommended_architecture']}.",
        "",
        "## Ranked Results",
        "",
        "| Rank | Candidate | Role | Status | Score | Tok/s | Peak GB | Load s |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(summary["ranked"], start=1):
        result = item.get("result", {})
        aggregate = result.get("aggregate", {}) if isinstance(result, dict) else {}
        lines.append(
            "| {rank} | `{label}` | `{role}` | `{status}` | {score:.3f} | {tps} | {mem} | {load} |".format(
                rank=index,
                label=item["label"],
                role=item["role"],
                status=item["status"],
                score=item["score"]["overall"],
                tps=_fmt(aggregate.get("generation_tps_avg")),
                mem=_fmt(aggregate.get("peak_memory_gb")),
                load=_fmt(result.get("load_seconds")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Scores combine measured local performance with a documented quality prior.",
            "- A failed or skipped model gets zero reliability and cannot be selected.",
            "- The score intentionally rewards memory headroom because this app must remain usable while the OS, UI, and context/memory layers are active.",
        ]
    )
    return "\n".join(lines) + "\n"


def _first_ok(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in items:
        if item.get("status") == "ok":
            return item
    return None


def _label(item: dict[str, Any] | None) -> str:
    if not item:
        return "none"
    return str(item.get("label") or item.get("candidate_id") or "unknown")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"
