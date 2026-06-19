from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LanguagePolicy:
    mode: str
    reply_in_user_language: bool
    supported: tuple[str, ...]


@dataclass(frozen=True)
class RuntimePolicy:
    auto_configure: bool
    preferred_backends: dict[str, str]
    model_cache_dir: str
    work_dir: str
    context_policy_config: str
    context_policy_profile: str


@dataclass(frozen=True)
class ExtensionPaths:
    plugins_dir: str
    skills_dir: str
    tools_config: str
    mcp_config: str
    cron_config: str


@dataclass(frozen=True)
class PermissionPolicy:
    default_write_policy: str
    allow_process_execution: bool
    connector_install_policy: str
    external_communication_policy: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    mode: str
    default_moe_config: str
    language: LanguagePolicy
    runtime: RuntimePolicy
    extensions: ExtensionPaths
    permissions: PermissionPolicy


def load_app_config(path: str | Path = "configs/app.json") -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    language_raw = raw.get("language", {})
    runtime_raw = raw.get("runtime", {})
    extensions_raw = raw.get("extensions", {})
    permissions_raw = raw.get("permissions", {})
    return AppConfig(
        name=str(raw.get("name", "myMoE")),
        mode=str(raw.get("mode", "local_model_required")),
        default_moe_config=str(raw.get("default_moe_config", "configs/moe.live.general-mlx.example.json")),
        language=LanguagePolicy(
            mode=str(language_raw.get("mode", "auto")),
            reply_in_user_language=bool(language_raw.get("reply_in_user_language", True)),
            supported=tuple(str(item) for item in language_raw.get("supported", ["auto", "en"])),
        ),
        runtime=RuntimePolicy(
            auto_configure=bool(runtime_raw.get("auto_configure", True)),
            preferred_backends={
                str(key): str(value)
                for key, value in runtime_raw.get("preferred_backends", {}).items()
            },
            model_cache_dir=str(runtime_raw.get("model_cache_dir", "~/.cache/huggingface")),
            work_dir=str(runtime_raw.get("work_dir", "work/runtime")),
            context_policy_config=str(runtime_raw.get("context_policy_config", "configs/context-policy.json")),
            context_policy_profile=str(runtime_raw.get("context_policy_profile", "default")),
        ),
        extensions=ExtensionPaths(
            plugins_dir=str(extensions_raw.get("plugins_dir", "plugins")),
            skills_dir=str(extensions_raw.get("skills_dir", "skills")),
            tools_config=str(extensions_raw.get("tools_config", "configs/tools.json")),
            mcp_config=str(extensions_raw.get("mcp_config", "configs/mcp.json")),
            cron_config=str(extensions_raw.get("cron_config", "configs/cron.json")),
        ),
        permissions=PermissionPolicy(
            default_write_policy=str(permissions_raw.get("default_write_policy", "approval_required")),
            allow_process_execution=bool(permissions_raw.get("allow_process_execution", False)),
            connector_install_policy=str(permissions_raw.get("connector_install_policy", "approval_required")),
            external_communication_policy=str(permissions_raw.get("external_communication_policy", "draft_only")),
        ),
    )


def app_config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "mode": config.mode,
        "default_moe_config": config.default_moe_config,
        "language": {
            "mode": config.language.mode,
            "reply_in_user_language": config.language.reply_in_user_language,
            "supported": list(config.language.supported),
        },
        "runtime": {
            "auto_configure": config.runtime.auto_configure,
            "preferred_backends": dict(config.runtime.preferred_backends),
            "model_cache_dir": config.runtime.model_cache_dir,
            "work_dir": config.runtime.work_dir,
            "context_policy_config": config.runtime.context_policy_config,
            "context_policy_profile": config.runtime.context_policy_profile,
        },
        "extensions": config.extensions.__dict__,
        "permissions": config.permissions.__dict__,
    }
