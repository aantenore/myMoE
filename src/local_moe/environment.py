from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .app_config import app_config_payload
from .hardware import detect_hardware

ENVIRONMENT_REPORT_PREFIX = "mymoe-environment-report"


def build_environment_report(
    *,
    config_path: str,
    config: object,
    app_config: object,
    app_config_path: str = "configs/app.json",
) -> dict[str, Any]:
    hardware = detect_hardware()
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "privacy": {
            "includes": [
                "application mode and config paths",
                "platform and Python metadata",
                "selected package versions",
                "git revision metadata",
                "configured local model identities",
                "hardware summary",
            ],
            "excludes": [
                "chat transcripts",
                "memory records",
                "environment variables",
                "API keys or secrets",
                "model log contents",
                "benchmark prompt response excerpts",
            ],
        },
        "app": app_config_payload(app_config),
        "paths": {
            "working_directory": str(Path.cwd()),
            "app_config": app_config_path,
            "moe_config": config_path,
            "work_dir": str(app_config.runtime.work_dir),
            "model_cache_dir": str(app_config.runtime.model_cache_dir),
        },
        "system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "virtualenv": sys.prefix != sys.base_prefix,
        },
        "packages": _package_versions(
            [
                "local-moe-orchestrator",
                "mlx",
                "mlx-lm",
                "mlx-metal",
                "huggingface-hub",
                "transformers",
                "tokenizers",
                "numpy",
                "psutil",
                "llama-cpp-python",
                "playwright",
            ]
        ),
        "git": _git_info(),
        "hardware": asdict(hardware),
        "runtime": {
            "backend_preferences": dict(app_config.runtime.preferred_backends),
            "context_policy_config": app_config.runtime.context_policy_config,
            "context_policy_profile": app_config.runtime.context_policy_profile,
            "cron_auto_run": app_config.runtime.cron_auto_run,
            "cron_poll_seconds": app_config.runtime.cron_poll_seconds,
            "routing": {
                "strategy": config.routing.strategy,
                "aggregation": config.routing.aggregation,
                "top_k": config.routing.top_k,
                "fallback_order": list(config.routing.fallback_order),
                "semantic_enabled": config.routing.semantic.enabled,
                "distilled_enabled": config.routing.distilled.enabled,
                "distilled_artifact_path": config.routing.distilled.artifact_path,
            },
            "expert_count": len(config.experts),
            "experts": [_expert_payload(expert) for expert in config.experts],
        },
        "recommendations": _recommendations(),
    }


def environment_report_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ENVIRONMENT_REPORT_PREFIX}-{stamp}.md"


def render_environment_report_markdown(report: dict[str, Any]) -> str:
    app = report.get("app", {}) if isinstance(report.get("app"), dict) else {}
    paths = report.get("paths", {}) if isinstance(report.get("paths"), dict) else {}
    system = report.get("system", {}) if isinstance(report.get("system"), dict) else {}
    python = report.get("python", {}) if isinstance(report.get("python"), dict) else {}
    git = report.get("git", {}) if isinstance(report.get("git"), dict) else {}
    hardware = report.get("hardware", {}) if isinstance(report.get("hardware"), dict) else {}
    runtime = report.get("runtime", {}) if isinstance(report.get("runtime"), dict) else {}
    packages = report.get("packages", {}) if isinstance(report.get("packages"), dict) else {}
    experts = runtime.get("experts", []) if isinstance(runtime.get("experts"), list) else []
    lines = [
        "# myMoE Environment Snapshot",
        "",
        f"Generated: `{report.get('generated_at', 'unknown')}`",
        f"Schema: `{report.get('schema_version', 'unknown')}`",
        "",
        "## Application",
        "",
        f"- Name: `{app.get('name', 'unknown')}`",
        f"- Mode: `{app.get('mode', 'unknown')}`",
        f"- App config: `{paths.get('app_config', 'unknown')}`",
        f"- MoE config: `{paths.get('moe_config', 'unknown')}`",
        f"- Work directory: `{paths.get('work_dir', 'unknown')}`",
        f"- Model cache: `{paths.get('model_cache_dir', 'unknown')}`",
        "",
        "## System",
        "",
        f"- Platform: `{system.get('platform', 'unknown')}`",
        f"- Machine: `{system.get('machine', 'unknown')}`",
        f"- Python: `{python.get('implementation', 'unknown')} {python.get('version', 'unknown')}`",
        f"- Virtualenv: `{python.get('virtualenv', 'unknown')}`",
        f"- Hardware: `{hardware.get('cpu_brand', 'unknown')}` / `{hardware.get('memory_gib', 'unknown')} GiB RAM`",
        f"- Strategy: `{hardware.get('recommended_strategy', 'unknown')}`",
        "",
        "## Git",
        "",
        f"- Status: `{git.get('status', 'unknown')}`",
        f"- Branch: `{git.get('branch', 'unknown')}`",
        f"- Commit: `{git.get('commit', 'unknown')}`",
        f"- Dirty worktree: `{git.get('dirty', 'unknown')}`",
        "",
        "## Runtime",
        "",
        f"- Routing strategy: `{runtime.get('routing', {}).get('strategy', 'unknown') if isinstance(runtime.get('routing'), dict) else 'unknown'}`",
        f"- Aggregation: `{runtime.get('routing', {}).get('aggregation', 'unknown') if isinstance(runtime.get('routing'), dict) else 'unknown'}`",
        f"- Top K: `{runtime.get('routing', {}).get('top_k', 'unknown') if isinstance(runtime.get('routing'), dict) else 'unknown'}`",
        f"- Expert count: `{runtime.get('expert_count', 0)}`",
        "",
        "## Experts",
        "",
        "| Expert | Provider | Role | Model | Base URL | Parameters |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for expert in experts:
        if not isinstance(expert, dict):
            continue
        lines.append(
            "| `{id}` | `{provider}` | `{role}` | `{model}` | `{base_url}` | {params} |".format(
                id=_md_cell(expert.get("id", "")),
                provider=_md_cell(expert.get("provider", "")),
                role=_md_cell(expert.get("role", "")),
                model=_md_cell(expert.get("model", "")),
                base_url=_md_cell(expert.get("base_url") or "n/a"),
                params=_md_cell(", ".join(expert.get("safe_param_keys", [])) or "none"),
            )
        )
    lines.extend(
        [
            "",
            "## Packages",
            "",
            "| Package | Version |",
            "| --- | --- |",
        ]
    )
    for name, info in sorted(packages.items()):
        if not isinstance(info, dict):
            continue
        lines.append(f"| `{_md_cell(name)}` | `{_md_cell(_package_label(info))}` |")
    recommendations = report.get("recommendations", [])
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
            "This snapshot is metadata-only. It does not include chat transcripts, memory records, environment variables, API keys, model log contents, or benchmark response excerpts.",
            "",
        ]
    )
    return "\n".join(lines)


def _expert_payload(expert: object) -> dict[str, Any]:
    params = getattr(expert, "params", {}) if isinstance(getattr(expert, "params", {}), dict) else {}
    safe_params = {
        key: _sanitize_param_value(value)
        for key, value in params.items()
        if _is_safe_param_key(str(key))
    }
    return {
        "id": expert.id,
        "provider": expert.provider,
        "model": expert.model,
        "role": expert.role,
        "base_url": expert.base_url,
        "timeout_seconds": expert.timeout_seconds,
        "safe_params": safe_params,
        "safe_param_keys": sorted(safe_params),
    }


def _package_versions(names: list[str]) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for name in names:
        try:
            versions[name] = {"status": "installed", "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            versions[name] = {"status": "not_installed", "version": ""}
    return versions


def _package_label(info: dict[str, object]) -> str:
    version = str(info.get("version") or "").strip()
    if version:
        return version
    return str(info.get("status") or "unknown")


def _git_info() -> dict[str, Any]:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git(["rev-parse", "HEAD"])
    remote = _git(["config", "--get", "remote.origin.url"])
    porcelain = _git(["status", "--porcelain"])
    if not commit:
        return {"status": "unavailable", "branch": "", "commit": "", "dirty": None, "remote": ""}
    return {
        "status": "available",
        "branch": branch,
        "commit": commit,
        "short_commit": commit[:7],
        "dirty": bool(porcelain.strip()),
        "remote": remote,
    }


def _git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _is_safe_param_key(key: str) -> bool:
    lowered = key.lower()
    secret_markers = ("api", "key", "token", "secret", "password", "credential", "auth")
    return not any(marker in lowered for marker in secret_markers)


def _sanitize_param_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): (_sanitize_param_value(nested) if _is_safe_param_key(str(key)) else "[redacted]")
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_param_value(item) for item in value]
    return str(type(value).__name__)


def _recommendations() -> list[str]:
    git = _git_info()
    items: list[str] = []
    if git.get("dirty"):
        items.append("Commit or stash local changes before sharing this snapshot as a release baseline.")
    return items


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
