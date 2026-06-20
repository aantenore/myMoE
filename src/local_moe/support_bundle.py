from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .doctor import build_doctor_report
from .environment import build_environment_report
from .extensions import ExtensionRegistry
from .model_inventory import build_model_asset_inventory
from .model_servers import ModelServerManager
from .performance_report import build_performance_report
from .runtime_optimizer import build_runtime_optimizer_report
from .security_audit import build_security_audit_report


def build_support_bundle(
    *,
    config_path: str,
    config: object,
    app_config: object,
    app_config_path: str = "configs/app.json",
    registry: ExtensionRegistry | None = None,
    model_manager: ModelServerManager | None = None,
    quality_gate_path: str | Path = "outputs/quality-gate.json",
    hardware_profile_path: str | Path = "outputs/hardware-profile.json",
) -> dict[str, Any]:
    doctor = build_doctor_report(
        config_path=config_path,
        config=config,
        app_config=app_config,
        app_config_path=app_config_path,
        registry=registry,
        model_manager=model_manager,
    )
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "privacy": {
            "includes": [
                "system doctor report",
                "environment snapshot",
                "quality gate status",
                "sanitized performance report",
                "read-only runtime optimizer summary",
                "read-only security audit summary",
                "storage capacity summary",
                "model asset inventory",
                "hardware profile",
                "configured model server log paths",
                "generation run log path",
                "runtime file paths",
            ],
            "excludes": [
                "chat transcripts",
                "memory records",
                "generation run log contents",
                "environment variables",
                "model log contents",
                "benchmark prompt response excerpts",
                "API keys or secrets",
                "MCP environment variable names and values",
            ],
        },
        "doctor": doctor,
        "environment": build_environment_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=app_config_path,
        ),
        "quality_gate": _read_json_artifact(quality_gate_path),
        "performance": build_performance_report(),
        "model_inventory": build_model_asset_inventory(
            config_path=config_path,
            config=config,
            app_config=app_config,
        ),
        "runtime_optimizer": build_runtime_optimizer_report(
            config_path=config_path,
            app_config=app_config,
            app_config_path=app_config_path,
        ),
        "security_audit": build_security_audit_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=app_config_path,
            registry=registry,
        ),
        "hardware_profile": _read_json_artifact(hardware_profile_path),
        "runtime_files": _runtime_files(app_config),
        "log_paths": _log_paths(doctor),
    }


def support_bundle_filename() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"mymoe-support-bundle-{stamp}.json"


def _read_json_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"path": str(artifact_path), "status": "missing"}
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(artifact_path), "status": "invalid_json", "error": str(exc)}
    return {"path": str(artifact_path), "status": "available", "data": data}


def _runtime_files(app_config: object) -> dict[str, str]:
    work_dir = str(app_config.runtime.work_dir).rstrip("/")
    return {
        "work_dir": work_dir,
        "chat_store": f"{work_dir}/chats.json",
        "memory_store": f"{work_dir}/memory.jsonl",
        "run_log": f"{work_dir}/runs.jsonl",
        "cron_state": f"{work_dir}/cron-state.json",
    }


def _log_paths(doctor: dict[str, Any]) -> list[dict[str, str]]:
    servers = doctor.get("model_processes", {}).get("servers", [])
    paths = []
    for server in servers:
        if not isinstance(server, dict):
            continue
        log_path = str(server.get("log_path") or "").strip()
        if not log_path:
            continue
        paths.append(
            {
                "expert_id": str(server.get("expert_id", "")),
                "model": str(server.get("model", "")),
                "path": log_path,
            }
        )
    return paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
