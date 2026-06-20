from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .app_config import app_config_payload
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .config_profiles import build_hardware_fit
from .extensions import (
    ExtensionRegistry,
    audit_extension_registry,
    load_extension_registry,
    registry_payload,
)
from .health import check_runtime_health, runtime_health_payload
from .hardware import HardwareProfile
from .model_servers import ModelServerManager
from .scheduler import cron_status
from .setup_status import inspect_setup_status, setup_status_payload
from .storage import DEFAULT_MIN_FREE_GIB, build_storage_report

DOCTOR_REPORT_FILENAME = "mymoe-doctor-report.md"


def build_doctor_report(
    *,
    config_path: str,
    config: object,
    app_config: object,
    app_config_path: str = "configs/app.json",
    registry: ExtensionRegistry | None = None,
    model_manager: ModelServerManager | None = None,
    include_health: bool = True,
    hardware_profile: HardwareProfile | None = None,
    candidate_paths: tuple[str, ...] | None = None,
    storage_min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    registry = registry or _load_registry(app_config)
    runtime_plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
    setup = setup_status_payload(
        inspect_setup_status(
            config_path,
            config,
            app_config,
            app_config_path=app_config_path,
        )
    )
    health = (
        runtime_health_payload(check_runtime_health(config))
        if include_health
        else {"status": "skipped", "checked_at": _now_iso(), "experts": []}
    )
    audit = audit_extension_registry(registry)
    hardware_fit = build_hardware_fit(
        config,
        hardware_profile=hardware_profile,
        candidate_paths=candidate_paths,
    )
    processes = (
        model_manager
        or ModelServerManager.from_config(
            config,
            preferred_backends=app_config.runtime.preferred_backends,
            work_dir=app_config.runtime.work_dir,
        )
    ).status()
    cron = cron_status(registry.cron_jobs, state_path=_cron_state_path(app_config))
    storage = build_storage_report(app_config, min_free_gib=storage_min_free_gib)

    checks = [
        _setup_check(setup),
        _health_check(health),
        _extension_check(audit),
        _hardware_fit_check(hardware_fit),
        _storage_check(storage),
        _process_check(processes, health),
        _cron_check(cron, app_config),
    ]
    status = _overall_status(checks)
    recommendations = _recommendations(setup, health, audit, hardware_fit, storage, processes, cron)

    return {
        "status": status,
        "checked_at": _now_iso(),
        "summary": {
            "passed": sum(1 for item in checks if item["status"] == "pass"),
            "warnings": sum(1 for item in checks if item["status"] == "warn"),
            "failed": sum(1 for item in checks if item["status"] == "fail"),
        },
        "checks": checks,
        "recommendations": recommendations,
        "app": app_config_payload(app_config),
        "runtime": runtime_plan_payload(runtime_plan),
        "setup": setup,
        "health": health,
        "hardware_fit": hardware_fit,
        "storage": storage,
        "model_processes": processes,
        "extension_audit": audit,
        "extensions": registry_payload(registry),
        "cron": cron,
    }


def doctor_report_filename() -> str:
    return DOCTOR_REPORT_FILENAME


def render_doctor_report_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    hardware_fit = report.get("hardware_fit", {}) if isinstance(report.get("hardware_fit"), dict) else {}
    storage = report.get("storage", {}) if isinstance(report.get("storage"), dict) else {}
    setup = report.get("setup", {}) if isinstance(report.get("setup"), dict) else {}
    health = report.get("health", {}) if isinstance(report.get("health"), dict) else {}
    runtime = report.get("runtime", {}) if isinstance(report.get("runtime"), dict) else {}
    model_processes = report.get("model_processes", {}) if isinstance(report.get("model_processes"), dict) else {}
    extension_audit = report.get("extension_audit", {}) if isinstance(report.get("extension_audit"), dict) else {}
    cron = report.get("cron", {}) if isinstance(report.get("cron"), dict) else {}
    checks = [item for item in report.get("checks", []) if isinstance(item, dict)]
    recommendations = [str(item) for item in report.get("recommendations", [])]
    lines = [
        "# myMoE System Doctor Report",
        "",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Checked at: `{report.get('checked_at', 'unknown')}`",
        "",
        "## Summary",
        "",
        f"- Passed checks: `{summary.get('passed', 0)}`",
        f"- Warnings: `{summary.get('warnings', 0)}`",
        f"- Failed checks: `{summary.get('failed', 0)}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        detail = str(check.get("detail") or check.get("message") or "")
        lines.append(
            "| `{id}` | `{status}` | `{severity}` | {detail} |".format(
                id=_md_cell(check.get("id", "")),
                status=_md_cell(check.get("status", "")),
                severity=_md_cell(check.get("severity", "")),
                detail=_md_cell(detail),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No recommendations.")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Backend: `{runtime.get('backend', 'unknown')}`",
            f"- Platform: `{runtime.get('platform_key', 'unknown')}`",
            f"- Setup status: `{setup.get('status', 'unknown')}`",
            f"- Health status: `{health.get('status', 'unknown')}`",
            f"- Model process count: `{model_processes.get('count', 0)}`",
            f"- Extension issues: `{extension_audit.get('issue_count', 0)}`",
            f"- Cron jobs: `{len(cron.get('jobs', [])) if isinstance(cron.get('jobs', []), list) else 0}`",
            f"- Storage status: `{storage.get('status', 'unknown')}`",
            "",
            "## Hardware Fit",
            "",
            f"- Status: `{hardware_fit.get('status', 'unknown')}`",
            f"- Summary: {hardware_fit.get('summary', 'unknown')}",
            f"- Estimated memory: `{hardware_fit.get('estimated_memory_gb', 'unknown')} GiB`",
            f"- Detected memory: `{hardware_fit.get('memory_gib', 'unknown')} GiB`",
            f"- Resident large experts: `{hardware_fit.get('resident_large_experts', 0)}`",
            "",
            "## Storage",
            "",
            f"- Status: `{storage.get('status', 'unknown')}`",
            f"- Minimum free space: `{storage.get('min_free_gib', 'unknown')} GiB`",
        ]
    )
    storage_paths = storage.get("paths", []) if isinstance(storage.get("paths"), list) else []
    for item in storage_paths:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{label}`: `{status}`, `{free}` GiB free at `{path}`".format(
                label=_md_cell(item.get("label", "")),
                status=_md_cell(item.get("status", "")),
                free=_md_cell(item.get("free_gib", "unknown")),
                path=_md_cell(item.get("expanded_path", "")),
            )
        )
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This report is metadata-only. It does not include chat transcripts, memory records, environment variables, API keys, model log bodies, or benchmark response excerpts.",
            "",
        ]
    )
    return "\n".join(lines)


def _setup_check(setup: dict[str, Any]) -> dict[str, Any]:
    if setup.get("status") == "ready":
        return _check("setup", "pass", "Model assets are ready.", severity="required")
    return _check(
        "setup",
        "fail",
        "Model assets or runtime dependencies need setup.",
        severity="required",
        detail=str(setup.get("error") or setup.get("status") or "needs_setup"),
    )


def _health_check(health: dict[str, Any]) -> dict[str, Any]:
    if health.get("status") == "ready":
        return _check("health", "pass", "Configured model endpoints are healthy.", severity="required")
    failed = [
        f"{item.get('expert_id')}:{item.get('status')}"
        for item in health.get("experts", [])
        if item.get("provider") == "openai_compatible" and item.get("status") != "ok"
    ]
    return _check(
        "health",
        "fail",
        "One or more configured model endpoints are not reachable.",
        severity="required",
        detail=", ".join(failed) or str(health.get("status", "degraded")),
    )


def _extension_check(audit: dict[str, Any]) -> dict[str, Any]:
    issue_count = int(audit.get("issue_count", 0))
    if issue_count == 0:
        return _check("extensions", "pass", "Extension registry references are valid.", severity="required")
    return _check(
        "extensions",
        "fail",
        "Extension registry has invalid plugin references.",
        severity="required",
        detail=f"{issue_count} issue(s)",
    )


def _hardware_fit_check(fit: dict[str, Any]) -> dict[str, Any]:
    status = str(fit.get("status") or "unknown")
    summary = str(fit.get("summary") or status)
    if status in {"recommended", "fits", "compatible"}:
        return _check(
            "hardware_fit",
            "pass",
            "Active profile fits the detected machine.",
            severity="required",
            detail=summary,
        )
    if status == "too_large":
        return _check(
            "hardware_fit",
            "fail",
            "Active profile is too large for the detected machine.",
            severity="required",
            detail=summary,
        )
    return _check(
        "hardware_fit",
        "warn",
        "Active profile needs hardware-fit review.",
        severity="optional",
        detail=summary,
    )


def _process_check(processes: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    servers = [item for item in processes.get("servers", []) if isinstance(item, dict)]
    if not servers:
        return _check("model_processes", "pass", "No local model process commands are required.", severity="optional")
    if all(bool(item.get("endpoint_reachable")) for item in servers):
        return _check("model_processes", "pass", "Configured model endpoints are reachable.", severity="required")
    if health.get("status") == "ready":
        return _check(
            "model_processes",
            "warn",
            "Health is ready, but process manager cannot confirm every endpoint.",
            severity="optional",
        )
    return _check(
        "model_processes",
        "fail",
        "Configured model endpoints are not running.",
        severity="required",
    )


def _cron_check(cron: dict[str, Any], app_config: object) -> dict[str, Any]:
    jobs = cron.get("jobs", [])
    if not jobs:
        return _check("cron", "warn", "No cron jobs are configured.", severity="optional")
    if bool(app_config.runtime.cron_auto_run):
        return _check("cron", "pass", "Background safe-job automation is configured.", severity="optional")
    return _check("cron", "warn", "Background cron automation is disabled.", severity="optional")


def _storage_check(storage: dict[str, Any]) -> dict[str, Any]:
    if storage.get("status") == "ready":
        return _check("storage", "pass", "Configured runtime storage has enough free space.", severity="optional")
    summary = storage.get("summary", {}) if isinstance(storage.get("summary"), dict) else {}
    return _check(
        "storage",
        "warn",
        "Configured runtime storage needs attention.",
        severity="optional",
        detail=(
            f"{summary.get('attention', 0)} attention path(s), "
            f"{summary.get('unavailable', 0)} unavailable path(s)"
        ),
    )


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    severity: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "severity": severity,
        "message": message,
        "detail": detail,
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    required_failures = [
        item for item in checks if item["status"] == "fail" and item["severity"] == "required"
    ]
    if required_failures:
        return "blocked"
    if any(item["status"] in {"fail", "warn"} for item in checks):
        return "attention"
    return "ready"


def _recommendations(
    setup: dict[str, Any],
    health: dict[str, Any],
    audit: dict[str, Any],
    hardware_fit: dict[str, Any],
    storage: dict[str, Any],
    processes: dict[str, Any],
    cron: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if setup.get("status") != "ready":
        command = str(setup.get("download_command_display") or "").strip()
        recommendations.append(
            f"Run runtime preparation: {command}" if command else "Run guarded runtime preparation."
        )
    if health.get("status") != "ready":
        recommendations.append("Start configured local model servers and refresh runtime health.")
    if int(audit.get("issue_count", 0)):
        recommendations.append("Fix plugin references reported by the extension registry audit.")
    fit_status = str(hardware_fit.get("status") or "unknown")
    if fit_status == "too_large":
        recommendations.append("Switch to a smaller runtime profile before starting local model servers.")
    elif fit_status == "stretch":
        recommendations.append("Treat the active profile as stretch: close extra model processes and monitor memory.")
    elif fit_status == "unknown":
        recommendations.append("Add a model candidate memory estimate or run a local benchmark for this profile.")
    if storage.get("status") != "ready":
        recommendations.extend(str(item) for item in storage.get("recommendations", []))
    servers = [item for item in processes.get("servers", []) if isinstance(item, dict)]
    if servers and not all(bool(item.get("endpoint_reachable")) for item in servers):
        recommendations.append("Use Advanced Runtime or CLI --start-models --models-confirm to start endpoints.")
    due = [item.get("id") for item in cron.get("jobs", []) if isinstance(item, dict) and item.get("due")]
    if due:
        recommendations.append(f"Run due cron jobs: {', '.join(str(item) for item in due)}.")
    if not recommendations:
        recommendations.append("System doctor passed; the local runtime is ready.")
    return recommendations


def _load_registry(app_config: object) -> ExtensionRegistry:
    return load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )


def _cron_state_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/cron-state.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
