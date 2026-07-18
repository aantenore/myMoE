from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from .app_config import app_config_payload
from .execution_scope import ExecutionScopeGuard
from .extensions import ExtensionRegistry, audit_extension_registry, load_extension_registry

SECURITY_AUDIT_PREFIX = "mymoe-security-audit"

WRITE_RISK_CLASSES = {
    "write_local",
    "write_internal",
    "write_external",
    "financial",
    "communication",
    "identity_access",
    "security_sensitive",
    "process_execution",
    "network_open_world",
    "destructive",
    "privileged_admin",
}


def build_security_audit_report(
    *,
    config_path: str,
    config: object,
    app_config: object,
    app_config_path: str = "configs/app.json",
    registry: ExtensionRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or _load_registry(app_config)
    extension_audit = audit_extension_registry(registry)
    permissions = app_config.permissions
    cron_summary = _cron_summary(app_config, registry)
    mcp_summary = _mcp_summary(registry)
    model_summary = _model_endpoint_summary(config)
    tool_summary = _tool_summary(registry)
    plugin_summary = _plugin_summary(registry)

    checks = [
        _check_local_model_mode(app_config),
        _check_write_policy(permissions.default_write_policy),
        _check_process_execution(permissions.allow_process_execution),
        _check_connector_policy(permissions.connector_install_policy),
        _check_external_policy(permissions.external_communication_policy),
        _check_model_endpoints(model_summary),
        _check_mcp_env(mcp_summary),
        _check_mcp_allowlists(mcp_summary),
        _check_cron_write_policy(cron_summary),
        _check_extension_audit(extension_audit),
    ]
    status = _overall_status(checks)
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "status": status,
        "summary": {
            "passed": sum(1 for item in checks if item["status"] == "pass"),
            "warnings": sum(1 for item in checks if item["status"] == "warn"),
            "failed": sum(1 for item in checks if item["status"] == "fail"),
        },
        "privacy": {
            "includes": [
                "application security posture metadata",
                "permission policy values",
                "MCP server counts and env counts",
                "cron risk-class counts",
                "tool and plugin risk-class counts",
                "model endpoint locality metadata",
                "extension registry audit summary",
            ],
            "excludes": [
                "chat transcripts",
                "memory records",
                "environment variable names and values",
                "API keys or secrets",
                "model log contents",
                "MCP tool results",
                "local data bundle contents",
            ],
        },
        "app": {
            "config_path": app_config_path,
            "moe_config": config_path,
            "mode": app_config.mode,
            "permissions": dict(app_config_payload(app_config)["permissions"]),
        },
        "mcp": mcp_summary,
        "cron": cron_summary,
        "tools": tool_summary,
        "plugins": plugin_summary,
        "model_endpoints": model_summary,
        "extension_audit": extension_audit,
        "checks": checks,
        "recommendations": _recommendations(checks),
    }


def security_audit_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{SECURITY_AUDIT_PREFIX}-{stamp}.md"


def render_security_audit_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    app = report.get("app", {}) if isinstance(report.get("app"), dict) else {}
    mcp = report.get("mcp", {}) if isinstance(report.get("mcp"), dict) else {}
    cron = report.get("cron", {}) if isinstance(report.get("cron"), dict) else {}
    tools = report.get("tools", {}) if isinstance(report.get("tools"), dict) else {}
    models = report.get("model_endpoints", {}) if isinstance(report.get("model_endpoints"), dict) else {}
    checks = [item for item in report.get("checks", []) if isinstance(item, dict)]
    recommendations = [str(item) for item in report.get("recommendations", [])]
    lines = [
        "# myMoE Security Audit",
        "",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Generated: `{report.get('generated_at', 'unknown')}`",
        "",
        "## Summary",
        "",
        f"- Passed checks: `{summary.get('passed', 0)}`",
        f"- Warnings: `{summary.get('warnings', 0)}`",
        f"- Failed checks: `{summary.get('failed', 0)}`",
        f"- App mode: `{app.get('mode', 'unknown')}`",
        f"- Remote model endpoints: `{models.get('remote_count', 0)}`",
        f"- Scope-blocked model endpoints: `{models.get('blocked_count', 0)}`",
        f"- MCP env values configured: `{mcp.get('env_var_count', 0)}`",
        f"- Cron write-risk jobs enabled: `{cron.get('enabled_write_risk_count', 0)}`",
        f"- Enabled write-risk tools: `{tools.get('enabled_write_risk_count', 0)}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Message |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        lines.append(
            "| `{id}` | `{status}` | `{severity}` | {message} |".format(
                id=_md_cell(check.get("id", "")),
                status=_md_cell(check.get("status", "")),
                severity=_md_cell(check.get("severity", "")),
                message=_md_cell(check.get("message", "")),
            )
        )
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No recommendations.")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This audit is metadata-only. It does not include chat transcripts, memory records, environment variable names or values, API keys, model log contents, MCP tool results, or local data bundle contents.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_registry(app_config: object) -> ExtensionRegistry:
    return load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )


def _cron_summary(app_config: object, registry: ExtensionRegistry) -> dict[str, Any]:
    jobs = registry.cron_jobs
    enabled = [job for job in jobs if job.enabled]
    write_risk = [job for job in jobs if job.risk_class in WRITE_RISK_CLASSES]
    enabled_write_risk = [job for job in enabled if job.risk_class in WRITE_RISK_CLASSES]
    return {
        "auto_run": bool(app_config.runtime.cron_auto_run),
        "confirm_writes": bool(app_config.runtime.cron_confirm_writes),
        "job_count": len(jobs),
        "enabled_count": len(enabled),
        "write_risk_count": len(write_risk),
        "enabled_write_risk_count": len(enabled_write_risk),
        "enabled_write_risk_jobs": [job.id for job in enabled_write_risk],
    }


def _mcp_summary(registry: ExtensionRegistry) -> dict[str, Any]:
    servers = registry.mcp_servers
    enabled = [server for server in servers if server.enabled]
    env_servers = [server for server in servers if server.env]
    tool_servers_without_allowlist = [
        server
        for server in enabled
        if "tools" in server.capabilities and not server.allowed_tools
    ]
    return {
        "server_count": len(servers),
        "enabled_count": len(enabled),
        "env_server_count": len(env_servers),
        "env_var_count": sum(len(server.env) for server in servers),
        "tool_servers_without_allowlist": [server.name for server in tool_servers_without_allowlist],
        "servers": [
            {
                "name": server.name,
                "enabled": server.enabled,
                "risk_class": server.risk_class,
                "transport": server.transport,
                "capabilities": list(server.capabilities),
                "allowed_tool_count": len(server.allowed_tools),
                "env_configured": bool(server.env),
                "env_count": len(server.env),
            }
            for server in servers
        ],
    }


def _tool_summary(registry: ExtensionRegistry) -> dict[str, Any]:
    enabled = [tool for tool in registry.tools if tool.enabled]
    enabled_write_risk = [tool for tool in enabled if tool.risk_class in WRITE_RISK_CLASSES]
    return {
        "tool_count": len(registry.tools),
        "enabled_count": len(enabled),
        "enabled_write_risk_count": len(enabled_write_risk),
        "enabled_write_risk_tools": [tool.name for tool in enabled_write_risk],
    }


def _plugin_summary(registry: ExtensionRegistry) -> dict[str, Any]:
    write_risk_plugins = [
        plugin.id
        for plugin in registry.plugins
        if str(plugin.permissions.get("risk_class", "read_only")) in WRITE_RISK_CLASSES
    ]
    return {
        "plugin_count": len(registry.plugins),
        "write_risk_count": len(write_risk_plugins),
        "write_risk_plugins": write_risk_plugins,
    }


def _model_endpoint_summary(config: object) -> dict[str, Any]:
    endpoints = []
    remote_count = 0
    local_count = 0
    allowed_count = 0
    blocked_count = 0
    guard = ExecutionScopeGuard(getattr(config, "execution_policy", None))
    for expert in getattr(config, "experts", ()):
        base_url = getattr(expert, "base_url", None)
        provider = str(getattr(expert, "provider", ""))
        parsed = urlparse(str(base_url or ""))
        host = parsed.hostname or ""
        local = _is_local_host(host) if base_url else provider in {"synthetic", "ollama"}
        if local:
            local_count += 1
        elif base_url:
            remote_count += 1
        target = expert.execution_target
        eligibility = guard.evaluate(target)
        if eligibility.allowed:
            allowed_count += 1
        else:
            blocked_count += 1
        endpoints.append(
            {
                "expert_id": str(getattr(expert, "id", "")),
                "provider": provider,
                "has_base_url": bool(base_url),
                "host": host,
                "local": local,
                "execution_allowed": eligibility.allowed,
                "execution_scope": (
                    eligibility.scope.value if eligibility.scope is not None else None
                ),
                "execution_transport": (
                    eligibility.transport.value
                    if eligibility.transport is not None
                    else None
                ),
                "reason_code": eligibility.reason_code,
            }
        )
    return {
        "expert_count": len(getattr(config, "experts", ())),
        "local_count": local_count,
        "remote_count": remote_count,
        "allowed_count": allowed_count,
        "blocked_count": blocked_count,
        "endpoints": endpoints,
    }


def _check_local_model_mode(app_config: object) -> dict[str, Any]:
    if app_config.mode == "local_model_required":
        return _check("app_mode", "pass", "Application mode requires local models.")
    return _check("app_mode", "warn", f"Application mode is `{app_config.mode}`.", severity="recommended")


def _check_write_policy(default_write_policy: str) -> dict[str, Any]:
    if default_write_policy == "approval_required":
        return _check("write_policy", "pass", "Default write policy requires approval.")
    return _check("write_policy", "warn", "Default write policy does not require approval.", severity="required")


def _check_process_execution(allow_process_execution: bool) -> dict[str, Any]:
    if not allow_process_execution:
        return _check("process_execution", "pass", "Arbitrary process execution is disabled by default.")
    return _check("process_execution", "warn", "Process execution is enabled; keep MCP allowlists narrow.", severity="required")


def _check_connector_policy(connector_install_policy: str) -> dict[str, Any]:
    if connector_install_policy == "approval_required":
        return _check("connector_policy", "pass", "Connector installation requires approval.")
    return _check("connector_policy", "warn", "Connector installation policy is not approval_required.", severity="recommended")


def _check_external_policy(external_communication_policy: str) -> dict[str, Any]:
    if external_communication_policy in {"draft_only", "approval_required"}:
        return _check("external_communication", "pass", "External communication is guarded.")
    return _check("external_communication", "warn", "External communication is not guarded.", severity="required")


def _check_model_endpoints(summary: dict[str, Any]) -> dict[str, Any]:
    blocked_count = int(summary.get("blocked_count", 0))
    remote_count = int(summary.get("remote_count", 0))
    if blocked_count:
        return _check(
            "model_endpoints",
            "warn",
            f"{blocked_count} configured model endpoint(s) are blocked by the execution policy.",
            severity="required",
        )
    if remote_count == 0:
        return _check("model_endpoints", "pass", "Configured model endpoints are local or provider-managed.")
    return _check("model_endpoints", "warn", f"{remote_count} configured model endpoint(s) are remote.", severity="required")


def _check_mcp_env(summary: dict[str, Any]) -> dict[str, Any]:
    env_count = int(summary.get("env_var_count", 0))
    if env_count == 0:
        return _check("mcp_env", "pass", "No MCP environment values are configured.")
    return _check("mcp_env", "warn", f"{env_count} MCP environment value(s) are configured and redacted from public payloads.")


def _check_mcp_allowlists(summary: dict[str, Any]) -> dict[str, Any]:
    servers = summary.get("tool_servers_without_allowlist", [])
    if not servers:
        return _check("mcp_allowlists", "pass", "Enabled MCP tool servers have explicit tool allowlists or no tool surface.")
    return _check("mcp_allowlists", "warn", "One or more enabled MCP tool servers have no allowlisted tools.", severity="required")


def _check_cron_write_policy(summary: dict[str, Any]) -> dict[str, Any]:
    enabled_write_count = int(summary.get("enabled_write_risk_count", 0))
    if not summary.get("auto_run"):
        return _check("cron_write_policy", "pass", "Cron auto-run is disabled.")
    if enabled_write_count == 0:
        return _check("cron_write_policy", "pass", "Cron auto-run has no enabled write-risk jobs.")
    if not summary.get("confirm_writes"):
        return _check("cron_write_policy", "pass", "Write-risk cron jobs are enabled but auto-run write confirmation is disabled.")
    return _check("cron_write_policy", "warn", "Write-risk cron jobs can run automatically because cron_confirm_writes is enabled.", severity="required")


def _check_extension_audit(audit: dict[str, Any]) -> dict[str, Any]:
    issue_count = int(audit.get("issue_count", 0))
    if issue_count == 0:
        return _check("extension_registry", "pass", "Extension registry audit has no issues.")
    return _check("extension_registry", "fail", f"Extension registry audit found {issue_count} issue(s).", severity="required")


def _recommendations(checks: list[dict[str, Any]]) -> list[str]:
    messages = [check["message"] for check in checks if check["status"] in {"warn", "fail"}]
    return messages or ["No security posture changes recommended."]


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "fail" for item in checks):
        return "blocked"
    if any(item["status"] == "warn" for item in checks):
        return "attention"
    return "ready"


def _check(check_id: str, status: str, message: str, *, severity: str = "recommended") -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "severity": severity,
        "message": message,
    }


def _is_local_host(host: str) -> bool:
    normalized = host.lower().strip("[]")
    return normalized in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
