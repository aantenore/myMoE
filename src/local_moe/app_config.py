from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any

from .secure_files import read_bounded_regular_file


MAX_APP_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LanguagePolicy:
    mode: str
    respond_in_user_language: bool
    supported: tuple[str, ...]


@dataclass(frozen=True)
class RuntimePolicy:
    auto_configure: bool
    preferred_backends: dict[str, str]
    model_cache_dir: str
    work_dir: str
    context_policy_config: str
    context_policy_profile: str
    profile_dir: str
    evaluation_dir: str
    cron_auto_run: bool
    cron_poll_seconds: float
    cron_confirm_writes: bool


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
    assistant_bridge_execution_policy: str
    connector_install_policy: str
    external_communication_policy: str


@dataclass(frozen=True)
class GatewayPolicy:
    enabled: bool
    model_alias: str
    max_request_bytes: int
    max_response_bytes: int
    allow_non_loopback: bool
    api_key_env: str


@dataclass(frozen=True)
class AdvisorPolicy:
    enabled: bool
    catalog_path: str
    evaluation_contract_path: str
    allowed_profiles: tuple[str, ...]
    default_profile: str
    workload_id: str
    capabilities: tuple[str, ...]
    tool_surfaces: tuple[str, ...]
    risk_class: str
    context_tokens: int
    max_request_bytes: int
    max_task_chars: int


@dataclass(frozen=True)
class AppConfig:
    name: str
    mode: str
    default_moe_config: str
    language: LanguagePolicy
    runtime: RuntimePolicy
    extensions: ExtensionPaths
    permissions: PermissionPolicy
    gateway: GatewayPolicy
    advisor: AdvisorPolicy


def load_app_config(path: str | Path = "configs/app.json") -> AppConfig:
    try:
        content = read_bounded_regular_file(
            path,
            maximum_bytes=MAX_APP_CONFIG_BYTES,
            label="app config",
        )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        owner_root = Path(path).expanduser().absolute().parent.resolve(strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(
            "App config must be bounded, strict UTF-8 JSON in a regular file."
        ) from exc
    _reject_unknown_keys(
        "root",
        raw,
        {
            "name",
            "mode",
            "default_moe_config",
            "language",
            "runtime",
            "extensions",
            "permissions",
            "gateway",
            "advisor",
        },
    )
    language_raw = raw.get("language", {})
    _reject_unknown_keys(
        "language",
        language_raw,
        {"mode", "respond_in_user_language", "supported"},
    )
    runtime_raw = raw.get("runtime", {})
    _reject_unknown_keys(
        "runtime",
        runtime_raw,
        {
            "auto_configure",
            "preferred_backends",
            "model_cache_dir",
            "work_dir",
            "context_policy_config",
            "context_policy_profile",
            "profile_dir",
            "evaluation_dir",
            "cron_auto_run",
            "cron_poll_seconds",
            "cron_confirm_writes",
        },
    )
    extensions_raw = raw.get("extensions", {})
    _reject_unknown_keys(
        "extensions",
        extensions_raw,
        {"plugins_dir", "skills_dir", "tools_config", "mcp_config", "cron_config"},
    )
    permissions_raw = raw.get("permissions", {})
    _reject_unknown_keys(
        "permissions",
        permissions_raw,
        {
            "default_write_policy",
            "allow_process_execution",
            "assistant_bridge_execution_policy",
            "connector_install_policy",
            "external_communication_policy",
        },
    )
    gateway_raw = raw.get("gateway", {})
    _reject_unknown_keys(
        "gateway",
        gateway_raw,
        {
            "enabled",
            "model_alias",
            "max_request_bytes",
            "max_response_bytes",
            "allow_non_loopback",
            "api_key_env",
        },
    )
    advisor_raw = raw.get("advisor", {})
    _reject_unknown_keys(
        "advisor",
        advisor_raw,
        {
            "enabled",
            "catalog_path",
            "evaluation_contract_path",
            "allowed_profiles",
            "default_profile",
            "workload_id",
            "capabilities",
            "tool_surfaces",
            "risk_class",
            "context_tokens",
            "max_request_bytes",
            "max_task_chars",
        },
    )
    assistant_bridge_execution_policy = str(
        permissions_raw.get(
            "assistant_bridge_execution_policy",
            "disabled",
        )
    )
    if assistant_bridge_execution_policy not in {
        "disabled",
        "local_only",
        "hybrid_receipt_confirmation",
    }:
        raise ValueError(
            "permissions.assistant_bridge_execution_policy must be disabled, local_only, or hybrid_receipt_confirmation."
        )
    gateway_enabled = _strict_bool(
        gateway_raw.get("enabled", True),
        "gateway.enabled",
    )
    gateway_model_alias = str(gateway_raw.get("model_alias", "mymoe")).strip()
    if not gateway_model_alias or len(gateway_model_alias) > 80:
        raise ValueError("gateway.model_alias must contain between 1 and 80 characters.")
    if any(character.isspace() for character in gateway_model_alias):
        raise ValueError("gateway.model_alias cannot contain whitespace.")
    gateway_max_request_bytes = _bounded_positive_int(
        gateway_raw.get("max_request_bytes", 8 * 1024 * 1024),
        "gateway.max_request_bytes",
        maximum=64 * 1024 * 1024,
    )
    gateway_max_response_bytes = _bounded_positive_int(
        gateway_raw.get("max_response_bytes", 32 * 1024 * 1024),
        "gateway.max_response_bytes",
        maximum=256 * 1024 * 1024,
    )
    gateway_allow_non_loopback = _strict_bool(
        gateway_raw.get("allow_non_loopback", False),
        "gateway.allow_non_loopback",
    )
    gateway_api_key_env = str(gateway_raw.get("api_key_env", "")).strip()
    if gateway_allow_non_loopback and not gateway_api_key_env:
        raise ValueError(
            "gateway.api_key_env is required when gateway.allow_non_loopback=true."
        )
    advisor = _load_advisor_policy(advisor_raw)
    advisor = replace(
        advisor,
        catalog_path=_resolve_owned_path(advisor.catalog_path, owner_root),
        evaluation_contract_path=_resolve_owned_path(
            advisor.evaluation_contract_path,
            owner_root,
        ),
    )
    return AppConfig(
        name=str(raw.get("name", "myMoE")),
        mode=str(raw.get("mode", "local_model_required")),
        default_moe_config=_resolve_owned_path(
            str(
                raw.get(
                    "default_moe_config",
                    "configs/moe.live.general-mlx.example.json",
                )
            ),
            owner_root,
        ),
        language=LanguagePolicy(
            mode=str(language_raw.get("mode", "auto")),
            respond_in_user_language=bool(
                language_raw.get("respond_in_user_language", True)
            ),
            supported=tuple(
                str(item) for item in language_raw.get("supported", ["auto", "en"])
            ),
        ),
        runtime=RuntimePolicy(
            auto_configure=_strict_bool(
                runtime_raw.get("auto_configure", True),
                "runtime.auto_configure",
            ),
            preferred_backends={
                str(key): str(value)
                for key, value in runtime_raw.get("preferred_backends", {}).items()
            },
            model_cache_dir=_resolve_owned_path(
                str(runtime_raw.get("model_cache_dir", "~/.cache/huggingface")),
                owner_root,
            ),
            work_dir=_resolve_owned_path(
                str(runtime_raw.get("work_dir", "work/runtime")),
                owner_root,
            ),
            context_policy_config=_resolve_owned_path(
                str(
                    runtime_raw.get(
                        "context_policy_config",
                        "configs/context-policy.json",
                    )
                ),
                owner_root,
            ),
            context_policy_profile=str(
                runtime_raw.get("context_policy_profile", "default")
            ),
            profile_dir=_resolve_owned_path(
                str(runtime_raw.get("profile_dir", "configs")),
                owner_root,
            ),
            evaluation_dir=_resolve_owned_path(
                str(runtime_raw.get("evaluation_dir", "experiments")),
                owner_root,
            ),
            cron_auto_run=_strict_bool(
                runtime_raw.get("cron_auto_run", False),
                "runtime.cron_auto_run",
            ),
            cron_poll_seconds=float(runtime_raw.get("cron_poll_seconds", 300)),
            cron_confirm_writes=_strict_bool(
                runtime_raw.get("cron_confirm_writes", False),
                "runtime.cron_confirm_writes",
            ),
        ),
        extensions=ExtensionPaths(
            plugins_dir=_resolve_owned_path(
                str(extensions_raw.get("plugins_dir", "plugins")),
                owner_root,
            ),
            skills_dir=_resolve_owned_path(
                str(extensions_raw.get("skills_dir", "skills")),
                owner_root,
            ),
            tools_config=_resolve_owned_path(
                str(extensions_raw.get("tools_config", "configs/tools.json")),
                owner_root,
            ),
            mcp_config=_resolve_owned_path(
                str(extensions_raw.get("mcp_config", "configs/mcp.json")),
                owner_root,
            ),
            cron_config=_resolve_owned_path(
                str(extensions_raw.get("cron_config", "configs/cron.json")),
                owner_root,
            ),
        ),
        permissions=PermissionPolicy(
            default_write_policy=str(
                permissions_raw.get("default_write_policy", "approval_required")
            ),
            allow_process_execution=_strict_bool(
                permissions_raw.get("allow_process_execution", False),
                "permissions.allow_process_execution",
            ),
            assistant_bridge_execution_policy=assistant_bridge_execution_policy,
            connector_install_policy=str(
                permissions_raw.get("connector_install_policy", "approval_required")
            ),
            external_communication_policy=str(
                permissions_raw.get("external_communication_policy", "draft_only")
            ),
        ),
        gateway=GatewayPolicy(
            enabled=gateway_enabled,
            model_alias=gateway_model_alias,
            max_request_bytes=gateway_max_request_bytes,
            max_response_bytes=gateway_max_response_bytes,
            allow_non_loopback=gateway_allow_non_loopback,
            api_key_env=gateway_api_key_env,
        ),
        advisor=advisor,
    )


def app_config_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "mode": config.mode,
        "default_moe_config": config.default_moe_config,
        "language": {
            "mode": config.language.mode,
            "respond_in_user_language": config.language.respond_in_user_language,
            "supported": list(config.language.supported),
        },
        "runtime": {
            "auto_configure": config.runtime.auto_configure,
            "preferred_backends": dict(config.runtime.preferred_backends),
            "model_cache_dir": config.runtime.model_cache_dir,
            "work_dir": config.runtime.work_dir,
            "context_policy_config": config.runtime.context_policy_config,
            "context_policy_profile": config.runtime.context_policy_profile,
            "profile_dir": config.runtime.profile_dir,
            "evaluation_dir": config.runtime.evaluation_dir,
            "cron_auto_run": config.runtime.cron_auto_run,
            "cron_poll_seconds": config.runtime.cron_poll_seconds,
            "cron_confirm_writes": config.runtime.cron_confirm_writes,
        },
        "extensions": config.extensions.__dict__,
        "permissions": config.permissions.__dict__,
        "gateway": config.gateway.__dict__,
        "advisor": advisor_policy_payload(config.advisor),
    }


def advisor_policy_payload(policy: AdvisorPolicy) -> dict[str, Any]:
    """Return the browser-safe Advisor policy without local source paths."""

    return {
        "enabled": policy.enabled,
        "allowed_profiles": list(policy.allowed_profiles),
        "default_profile": policy.default_profile,
        "workload": {
            "id": policy.workload_id,
            "capabilities": list(policy.capabilities),
            "tool_surfaces": list(policy.tool_surfaces),
            "risk_class": policy.risk_class,
            "context_tokens": policy.context_tokens,
        },
        "limits": {
            "max_request_bytes": policy.max_request_bytes,
            "max_task_chars": policy.max_task_chars,
        },
    }


def _resolve_owned_path(value: str, owner_root: Path) -> str:
    """Resolve only explicit app-owned paths while preserving legacy CWD paths."""

    if not (value.startswith("./") or value.startswith(".\\")):
        return value
    candidate = (owner_root / value).resolve()
    if not candidate.is_relative_to(owner_root):
        raise ValueError("App-owned paths must stay beside their app config.")
    return str(candidate)


def _reject_unknown_keys(
    section: str,
    raw: object,
    allowed: set[str],
) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"App config section {section!r} must be an object.")
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown app config keys in {section!r}: {names}.")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate app config key is not allowed: {key}.")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite app config number is not allowed: {value}.")


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _bounded_positive_int(value: object, label: str, *, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ValueError(f"{label} must be an integer between 1 and {maximum}.")
    return value


_ADVISOR_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")


def _load_advisor_policy(raw: dict[str, Any]) -> AdvisorPolicy:
    enabled = _strict_bool(raw.get("enabled", False), "advisor.enabled")
    catalog_path = _strict_config_path(
        raw.get("catalog_path", ""),
        "advisor.catalog_path",
        required=enabled,
    )
    evaluation_contract_path = _strict_config_path(
        raw.get("evaluation_contract_path", ""),
        "advisor.evaluation_contract_path",
        required=enabled,
    )
    allowed_profiles = _strict_identifier_tuple(
        raw.get("allowed_profiles", ["balanced"]),
        "advisor.allowed_profiles",
        non_empty=True,
    )
    default_profile = _strict_identifier(
        raw.get("default_profile", "balanced"),
        "advisor.default_profile",
    )
    if default_profile not in allowed_profiles:
        raise ValueError(
            "advisor.default_profile must be included in advisor.allowed_profiles."
        )
    workload_id = _strict_identifier(
        raw.get("workload_id", "local-summary"),
        "advisor.workload_id",
    )
    capabilities = _strict_identifier_tuple(
        raw.get("capabilities", ["summarization"]),
        "advisor.capabilities",
        non_empty=True,
    )
    tool_surfaces = _strict_identifier_tuple(
        raw.get("tool_surfaces", []),
        "advisor.tool_surfaces",
        non_empty=False,
    )
    risk_class = _strict_identifier(
        raw.get("risk_class", "compute_only"),
        "advisor.risk_class",
    )
    context_tokens = _bounded_positive_int(
        raw.get("context_tokens", 4096),
        "advisor.context_tokens",
        maximum=1_048_576,
    )
    max_request_bytes = _bounded_positive_int(
        raw.get("max_request_bytes", 64 * 1024),
        "advisor.max_request_bytes",
        maximum=262_144,
    )
    max_task_chars = _bounded_positive_int(
        raw.get("max_task_chars", 16 * 1024),
        "advisor.max_task_chars",
        maximum=131_072,
    )
    return AdvisorPolicy(
        enabled=enabled,
        catalog_path=catalog_path,
        evaluation_contract_path=evaluation_contract_path,
        allowed_profiles=allowed_profiles,
        default_profile=default_profile,
        workload_id=workload_id,
        capabilities=capabilities,
        tool_surfaces=tool_surfaces,
        risk_class=risk_class,
        context_tokens=context_tokens,
        max_request_bytes=max_request_bytes,
        max_task_chars=max_task_chars,
    )


def _strict_config_path(value: object, label: str, *, required: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if rendered != value or any(
        ord(character) < 32 or ord(character) == 127 for character in rendered
    ):
        raise ValueError(f"{label} must be a plain filesystem path.")
    if required and not rendered:
        raise ValueError(f"{label} is required when advisor.enabled=true.")
    if len(rendered) > 4096:
        raise ValueError(f"{label} must contain at most 4096 characters.")
    return rendered


def _strict_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{label} must be a string identifier.")
    if _ADVISOR_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier.")
    return value


def _strict_identifier_tuple(
    value: object,
    label: str,
    *,
    non_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array of identifiers.")
    items = tuple(_strict_identifier(item, label) for item in value)
    if non_empty and not items:
        raise ValueError(f"{label} must not be empty.")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates.")
    return items
