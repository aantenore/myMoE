from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_profiles import recommend_config_profile
from .performance_report import build_performance_report
from .run_log import RunLogStore, run_log_summary

DEFAULT_P95_LATENCY_ATTENTION_MS = 30000
DEFAULT_AVG_LATENCY_WATCH_MS = 15000


def build_runtime_optimizer_report(
    *,
    config_path: str,
    app_config: object,
    app_config_path: str = "configs/app.json",
    run_log_path: str | Path | None = None,
    run_limit: int = 100,
    p95_latency_attention_ms: int = DEFAULT_P95_LATENCY_ATTENTION_MS,
    avg_latency_watch_ms: int = DEFAULT_AVG_LATENCY_WATCH_MS,
) -> dict[str, Any]:
    """Build a read-only local optimization report from existing runtime evidence."""

    store = RunLogStore(run_log_path or _run_log_path(app_config))
    run_report = store.read_report(limit=run_limit)
    summary = run_log_summary(run_report.records)
    profile_report = recommend_config_profile(
        active_config_path=config_path,
        app_config=app_config,
        app_config_path=app_config_path,
    )
    performance_report = build_performance_report()
    signals = _signals(
        summary=summary,
        run_report=run_report,
        profile_recommendation=profile_report.get("recommendation", {}),
        performance=performance_report,
        p95_latency_attention_ms=p95_latency_attention_ms,
        avg_latency_watch_ms=avg_latency_watch_ms,
    )
    actions = _actions(
        config_path=config_path,
        app_config_path=app_config_path,
        summary=summary,
        run_report=run_report,
        profile_recommendation=profile_report.get("recommendation", {}),
        performance=performance_report,
        p95_latency_attention_ms=p95_latency_attention_ms,
    )
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "status": _overall_status(signals),
        "mode": "read_only",
        "thresholds": {
            "p95_latency_attention_ms": p95_latency_attention_ms,
            "avg_latency_watch_ms": avg_latency_watch_ms,
        },
        "active_config_path": config_path,
        "run_log": {
            "path": str(store.path),
            "limit": max(1, min(run_limit, 500)),
            "diagnostics": {
                "status": "attention" if run_report.skipped_count else "ready",
                "returned_records": len(run_report.records),
                "valid_records": run_report.valid_count,
                "skipped_records": run_report.skipped_count,
                "total_lines": run_report.total_lines,
            },
            "summary": summary,
        },
        "profile_recommendation": profile_report.get("recommendation", {}),
        "performance": _performance_snapshot(performance_report),
        "signals": signals,
        "actions": actions,
    }


def runtime_optimizer_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"mymoe-runtime-optimizer-{stamp}.md"


def render_runtime_optimizer_markdown(report: dict[str, Any]) -> str:
    run_log = report.get("run_log", {})
    summary = run_log.get("summary", {}) if isinstance(run_log, dict) else {}
    latency = summary.get("latency_ms", {}) if isinstance(summary, dict) else {}
    profile = report.get("profile_recommendation", {})
    performance = report.get("performance", {})
    lines = [
        "# myMoE Runtime Optimizer Report",
        "",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Generated: `{report.get('generated_at', 'unknown')}`",
        f"Mode: `{report.get('mode', 'read_only')}`",
        f"Active config: `{report.get('active_config_path', 'unknown')}`",
        "",
        "## Signals",
        "",
    ]
    for signal in report.get("signals", []):
        lines.append(
            "- `{id}` / `{status}` / `{severity}`: {message}".format(
                id=signal.get("id", "unknown"),
                status=signal.get("status", "unknown"),
                severity=signal.get("severity", "unknown"),
                message=signal.get("message", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime Evidence",
            "",
            f"- Run records: `{summary.get('record_count', 0)}`",
            f"- P95 latency: `{_fmt(latency.get('p95'))} ms`",
            f"- Average latency: `{_fmt(latency.get('avg'))} ms`",
            f"- Skipped run-log records: `{run_log.get('diagnostics', {}).get('skipped_records', 0)}`",
            f"- Performance status: `{performance.get('status', 'unknown')}`",
            f"- Recommended profile: `{profile.get('profile_path', '') or 'none'}`",
            f"- Profile recommendation status: `{profile.get('status', 'unknown')}`",
            "",
            "## Actions",
            "",
            "| Priority | Action | Confirmation | Side effects |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for action in report.get("actions", []):
        lines.append(
            "| {priority} | `{id}`: {label} | `{confirm}` | `{side}` |".format(
                priority=action.get("priority", 0),
                id=action.get("id", "unknown"),
                label=action.get("label", ""),
                confirm=bool(action.get("requires_confirmation")),
                side=action.get("side_effects", "unknown"),
            )
        )
    return "\n".join(lines) + "\n"


def _signals(
    *,
    summary: dict[str, Any],
    run_report: object,
    profile_recommendation: dict[str, Any],
    performance: dict[str, Any],
    p95_latency_attention_ms: int,
    avg_latency_watch_ms: int,
) -> list[dict[str, Any]]:
    latency = summary.get("latency_ms", {})
    errors = summary.get("errors", {})
    context = summary.get("context", {})
    record_count = int(summary.get("record_count") or 0)
    signals = [
        _signal(
            "run_observations",
            "ready" if record_count else "watch",
            "info" if record_count else "medium",
            f"{record_count} recent metadata-only generation run(s) are available.",
            {"record_count": record_count},
        )
    ]
    skipped = int(getattr(run_report, "skipped_count", 0) or 0)
    if skipped:
        signals.append(
            _signal(
                "run_log_integrity",
                "attention",
                "medium",
                f"{skipped} malformed or legacy run-log record(s) were skipped.",
                {"skipped_records": skipped},
            )
        )
    else:
        signals.append(_signal("run_log_integrity", "ready", "info", "Run-log records are readable.", {}))

    p95 = _optional_float(latency.get("p95"))
    avg = _optional_float(latency.get("avg"))
    if p95 is not None and p95 > p95_latency_attention_ms:
        signals.append(
            _signal(
                "latency",
                "attention",
                "high",
                f"P95 latency is {int(p95)} ms, above the {p95_latency_attention_ms} ms local target.",
                {"p95_ms": p95, "target_ms": p95_latency_attention_ms},
            )
        )
    elif avg is not None and avg > avg_latency_watch_ms:
        signals.append(
            _signal(
                "latency",
                "watch",
                "medium",
                f"Average latency is {round(avg, 2)} ms; continue monitoring local responsiveness.",
                {"avg_ms": avg, "watch_ms": avg_latency_watch_ms},
            )
        )
    else:
        signals.append(_signal("latency", "ready", "info", "Recent latency is within the local target.", {}))

    error_total = int(errors.get("total") or 0)
    if error_total:
        signals.append(
            _signal(
                "generation_errors",
                "attention",
                "high",
                f"{error_total} generation error(s) were recorded in recent run metadata.",
                {"error_total": error_total},
            )
        )
    else:
        signals.append(_signal("generation_errors", "ready", "info", "No generation errors were recorded.", {}))

    compactions = int(context.get("compaction_needed_count") or 0)
    dropped_turns = int(context.get("dropped_turns_total") or 0)
    if compactions or dropped_turns:
        signals.append(
            _signal(
                "context_pressure",
                "watch",
                "medium",
                "Recent runs needed context compaction or dropped older turns.",
                {"compaction_needed_count": compactions, "dropped_turns_total": dropped_turns},
            )
        )
    else:
        signals.append(_signal("context_pressure", "ready", "info", "No recent context pressure was recorded.", {}))

    profile_status = str(profile_recommendation.get("status") or "unknown")
    if profile_status == "unavailable":
        signals.append(
            _signal(
                "profile_recommendation",
                "attention",
                "high",
                "No valid local runtime profile is available for optimization.",
                {"status": profile_status},
            )
        )
    elif profile_recommendation.get("profile_path") and not profile_recommendation.get("active"):
        signals.append(
            _signal(
                "profile_recommendation",
                "watch",
                "medium",
                "A different local profile is recommended for this machine.",
                {"profile_path": profile_recommendation.get("profile_path")},
            )
        )
    else:
        signals.append(_signal("profile_recommendation", "ready", "info", "The active profile is locally recommended.", {}))

    perf_status = str(performance.get("status") or "unknown")
    coverage = performance.get("coverage", {}) if isinstance(performance.get("coverage"), dict) else {}
    if perf_status in {"blocked", "missing"}:
        signals.append(
            _signal(
                "performance_evidence",
                "attention",
                "medium",
                "Local benchmark evidence is missing or blocked.",
                {"status": perf_status},
            )
        )
    elif coverage.get("status") != "complete":
        signals.append(
            _signal(
                "performance_evidence",
                "watch",
                "low",
                "Benchmark coverage is partial; recommendations remain conservative.",
                {"status": perf_status, "coverage": coverage.get("status")},
            )
        )
    else:
        signals.append(_signal("performance_evidence", "ready", "info", "Benchmark evidence is available.", {}))
    return signals


def _actions(
    *,
    config_path: str,
    app_config_path: str,
    summary: dict[str, Any],
    run_report: object,
    profile_recommendation: dict[str, Any],
    performance: dict[str, Any],
    p95_latency_attention_ms: int,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    latency = summary.get("latency_ms", {})
    errors = summary.get("errors", {})
    p95 = _optional_float(latency.get("p95"))

    if int(summary.get("record_count") or 0) == 0:
        actions.append(
            _command_action(
                "run_generation_smoke",
                "Run a local generation smoke test",
                "Collect the first local generation observation before changing runtime policy.",
                [".venv/bin/python", "-m", "local_moe.cli", "--app-config", app_config_path, "--config", config_path, "--smoke-generate"],
                priority=60,
                side_effects="local_model_call",
                requires_confirmation=False,
            )
        )

    if p95 is not None and p95 > p95_latency_attention_ms:
        recommended_path = str(profile_recommendation.get("profile_path") or "")
        if recommended_path and not profile_recommendation.get("default"):
            actions.append(
                _command_action(
                    "activate_recommended_profile",
                    "Activate the locally recommended profile for the next start",
                    "High observed latency makes the profile recommendation worth applying after review.",
                    [
                        ".venv/bin/python",
                        "-m",
                        "local_moe.cli",
                        "--app-config",
                        app_config_path,
                        "--config",
                        config_path,
                        "--activate-recommended-profile",
                        "--profile-confirm",
                    ],
                    priority=95,
                    side_effects="writes_default_profile",
                    requires_confirmation=True,
                )
            )
        actions.append(
            _command_action(
                "review_performance_report",
                "Review benchmark-backed model decision",
                "Compare observed latency against the latest local benchmark before changing resident experts.",
                [".venv/bin/python", "-m", "local_moe.cli", "--performance-report"],
                priority=75,
                side_effects="none",
                requires_confirmation=False,
            )
        )

    if profile_recommendation.get("requires_setup"):
        actions.append(
            _command_action(
                "prepare_recommended_profile",
                "Prepare missing assets for the recommended profile",
                "Download or validate model assets through the guarded setup runner.",
                [
                    ".venv/bin/python",
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    app_config_path,
                    "--config",
                    config_path,
                    "--prepare-recommended-profile",
                    "--prepare-execute",
                    "--prepare-download-models",
                    "--prepare-confirm",
                ],
                priority=85,
                side_effects="installs_dependencies_and_downloads_models",
                requires_confirmation=True,
            )
        )

    if int(errors.get("total") or 0):
        actions.append(
            _command_action(
                "inspect_model_logs",
                "Inspect sanitized model logs",
                "Use bounded sanitized log tails to diagnose generation failures.",
                [".venv/bin/python", "-m", "local_moe.cli", "--app-config", app_config_path, "--config", config_path, "--models-logs"],
                priority=90,
                side_effects="none",
                requires_confirmation=False,
            )
        )

    skipped = int(getattr(run_report, "skipped_count", 0) or 0)
    if skipped:
        actions.append(
            _command_action(
                "prune_run_log",
                "Rewrite the run log without skipped records",
                "Keep recent valid metadata records and remove malformed or legacy rows.",
                [
                    ".venv/bin/python",
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    app_config_path,
                    "--config",
                    config_path,
                    "--runs-prune",
                    "--runs-confirm",
                    "--runs-keep",
                    "1000",
                ],
                priority=70,
                side_effects="rewrites_run_log_metadata",
                requires_confirmation=True,
            )
        )

    if not actions and performance.get("status") == "ready":
        actions.append(
            _command_action(
                "keep_current_profile",
                "Keep the current local runtime policy",
                "Recent observations and benchmark evidence do not require a profile change.",
                [".venv/bin/python", "-m", "local_moe.cli", "--doctor"],
                priority=20,
                side_effects="none",
                requires_confirmation=False,
            )
        )

    return sorted(actions, key=lambda item: (-int(item.get("priority", 0)), item.get("id", "")))


def _performance_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision", {})
    coverage = report.get("coverage", {})
    ranked = report.get("ranked", [])
    return {
        "status": report.get("status", "unknown"),
        "generated_at": report.get("generated_at"),
        "coverage": {
            "status": coverage.get("status") if isinstance(coverage, dict) else "unknown",
            "measured_result_count": coverage.get("measured_result_count") if isinstance(coverage, dict) else 0,
            "manifest_candidate_count": coverage.get("manifest_candidate_count") if isinstance(coverage, dict) else 0,
            "failed_count": coverage.get("failed_count") if isinstance(coverage, dict) else 0,
        },
        "decision": {
            "primary_general": _candidate_label(decision.get("primary_general")) if isinstance(decision, dict) else None,
            "fast_fallback": _candidate_label(decision.get("fast_fallback")) if isinstance(decision, dict) else None,
            "recommended_architecture": decision.get("recommended_architecture", "") if isinstance(decision, dict) else "",
        },
        "top_ranked": [_candidate_label(item) for item in ranked[:3] if isinstance(item, dict)],
        "recommendations": list(report.get("recommendations", []))[:3],
    }


def _command_action(
    action_id: str,
    label: str,
    description: str,
    argv: list[str],
    *,
    priority: int,
    side_effects: str,
    requires_confirmation: bool,
) -> dict[str, Any]:
    env = {"PYTHONPATH": "src"}
    return {
        "id": action_id,
        "label": label,
        "description": description,
        "priority": priority,
        "requires_confirmation": requires_confirmation,
        "side_effects": side_effects,
        "argv": argv,
        "env": env,
        "display": f"PYTHONPATH=src {' '.join(argv)}",
    }


def _signal(
    signal_id: str,
    status: str,
    severity: str,
    message: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": signal_id,
        "status": status,
        "severity": severity,
        "message": message,
        "evidence": evidence,
    }


def _overall_status(signals: list[dict[str, Any]]) -> str:
    if any(signal.get("status") == "attention" and signal.get("severity") == "high" for signal in signals):
        return "attention"
    if any(signal.get("status") == "attention" for signal in signals):
        return "attention"
    if any(signal.get("status") == "watch" for signal in signals):
        return "watch"
    return "ready"


def _run_log_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/runs.jsonl"


def _candidate_label(candidate: object) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    return {
        "candidate_id": candidate.get("candidate_id"),
        "label": candidate.get("label", candidate.get("candidate_id")),
        "role": candidate.get("role"),
        "status": candidate.get("status"),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "-"
    if numeric.is_integer():
        return str(int(numeric))
    return str(round(numeric, 2))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
