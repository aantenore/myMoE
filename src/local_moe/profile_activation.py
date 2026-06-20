from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .app_config import AppConfig
from .config import load_config
from .config_profiles import recommend_config_profile
from .hardware import HardwareProfile


class ProfileActivationError(ValueError):
    """Raised when a runtime profile cannot be activated safely."""


def activate_config_profile(
    profile_path: str,
    *,
    active_config_path: str,
    app_config: AppConfig,
    app_config_path: str = "configs/app.json",
    confirm: bool = False,
) -> dict[str, Any]:
    target_path = _display_path(profile_path)
    _validate_profile(target_path)
    previous_default = _display_path(app_config.default_moe_config)
    active_path = _display_path(active_config_path)
    restart_required = active_path != target_path
    restart_command = _restart_command(app_config_path, target_path)

    base_payload = {
        "schema_version": "1.0",
        "status": "confirmation_required" if not confirm else "ok",
        "activated": False,
        "app_config_path": _display_path(app_config_path),
        "previous_default_config": previous_default,
        "new_default_config": target_path,
        "active_config_path": active_path,
        "restart_required": restart_required,
        "current_process_changed": False,
        "restart_command": restart_command,
        "restart_command_display": _display_command(restart_command, env={"PYTHONPATH": "src"}),
        "message": "",
    }
    if not confirm:
        base_payload["message"] = "Profile activation requires confirm=true because it writes the app config file."
        return base_payload

    _write_default_profile(app_config_path, target_path)
    return {
        **base_payload,
        "status": "ok",
        "activated": True,
        "message": (
            "Default runtime profile updated. Restart the app to use the new profile."
            if restart_required
            else "Default runtime profile updated; the running app is already using this profile."
        ),
    }


def activate_recommended_config_profile(
    *,
    active_config_path: str,
    app_config: AppConfig,
    app_config_path: str = "configs/app.json",
    config_dir: str | Path = "configs",
    hardware_profile: HardwareProfile | None = None,
    candidate_paths: tuple[str | Path, ...] | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    recommendation = recommend_config_profile(
        active_config_path=active_config_path,
        app_config=app_config,
        app_config_path=app_config_path,
        config_dir=config_dir,
        hardware_profile=hardware_profile,
        candidate_paths=candidate_paths,
    )["recommendation"]
    profile_path = str(recommendation.get("profile_path") or "")
    if not profile_path:
        raise ProfileActivationError("No recommended runtime profile is available.")
    result = activate_config_profile(
        profile_path,
        active_config_path=active_config_path,
        app_config=app_config,
        app_config_path=app_config_path,
        confirm=confirm,
    )
    return {**result, "recommendation": recommendation}


def _validate_profile(profile_path: str) -> None:
    path = Path(profile_path)
    if not path.exists():
        raise ProfileActivationError(f"Runtime profile does not exist: {profile_path}")
    try:
        load_config(path)
    except Exception as exc:
        raise ProfileActivationError(f"Runtime profile is invalid: {exc}") from exc


def _write_default_profile(app_config_path: str | Path, profile_path: str) -> None:
    path = Path(app_config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["default_moe_config"] = profile_path
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _restart_command(app_config_path: str, profile_path: str) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        "local_moe.web",
        "--app-config",
        _display_path(app_config_path),
        "--config",
        profile_path,
        "--port",
        "8089",
    ]


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _display_command(argv: list[str], *, env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={value}" for key, value in env.items())
    body = " ".join(_quote_arg(item) for item in argv)
    return f"{prefix} {body}".strip()


def _quote_arg(value: str) -> str:
    if not value:
        return "''"
    if any(char.isspace() for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value
