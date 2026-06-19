from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .app_config import AppConfig
from .config import load_config
from .setup_status import inspect_setup_status, setup_status_payload


def discover_config_profiles(
    *,
    active_config_path: str,
    app_config: AppConfig,
    app_config_path: str = "configs/app.json",
    config_dir: str | Path = "configs",
) -> dict[str, Any]:
    """Return read-only metadata for runnable local MoE profiles."""

    paths = list(_candidate_paths(config_dir))
    active_path = Path(active_config_path)
    if active_path.exists() and not _contains_path(paths, active_path):
        paths.insert(0, active_path)

    profiles = [
        _profile_payload(
            path,
            active_config_path=active_config_path,
            default_config_path=app_config.default_moe_config,
            app_config=app_config,
            app_config_path=app_config_path,
        )
        for path in paths
    ]
    return {
        "schema_version": "1.0",
        "active_config_path": _display_path(active_config_path),
        "default_config_path": _display_path(app_config.default_moe_config),
        "config_dir": _display_path(config_dir),
        "count": len(profiles),
        "profiles": profiles,
    }


def _candidate_paths(config_dir: str | Path) -> tuple[Path, ...]:
    root = Path(config_dir)
    if not root.exists():
        return ()
    paths = [
        path
        for path in root.glob("*.json")
        if path.name.startswith(("moe.", "single."))
    ]
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _profile_payload(
    path: Path,
    *,
    active_config_path: str,
    default_config_path: str,
    app_config: AppConfig,
    app_config_path: str,
) -> dict[str, Any]:
    display_path = _display_path(path)
    payload: dict[str, Any] = {
        "path": display_path,
        "name": path.stem,
        "active": _same_path(path, active_config_path),
        "default": _same_path(path, default_config_path),
        "status": "invalid",
        "error": "",
    }
    try:
        config = load_config(path)
        setup = inspect_setup_status(
            display_path,
            config,
            app_config,
            app_config_path=app_config_path,
        )
    except Exception as exc:
        payload["error"] = str(exc)
        return payload

    setup_payload = setup_status_payload(setup)
    model_status_counts = Counter(str(item["status"]) for item in setup_payload["models"])
    runtime_backends = sorted(
        {
            str(expert.params.get("runtime_backend") or "provider_default")
            for expert in config.experts
        }
    )
    payload.update(
        {
            "status": "valid",
            "expert_count": len(config.experts),
            "provider_count": len({expert.provider for expert in config.experts}),
            "backend": setup.runtime_plan.backend,
            "runtime_backends": runtime_backends,
            "routing": {
                "strategy": config.routing.strategy,
                "aggregation": config.routing.aggregation,
                "top_k": config.routing.top_k,
                "semantic": config.routing.semantic.enabled,
                "distilled": config.routing.distilled.enabled,
            },
            "experts": [
                {
                    "id": expert.id,
                    "provider": expert.provider,
                    "model": expert.model,
                    "role": expert.role,
                    "runtime_backend": str(expert.params.get("runtime_backend") or "provider_default"),
                    "base_url": expert.base_url,
                }
                for expert in config.experts
            ],
            "setup": {
                "status": setup_payload["status"],
                "model_count": len(setup_payload["models"]),
                "model_status_counts": dict(sorted(model_status_counts.items())),
                "download_command_display": setup_payload["download_command_display"],
                "error": setup_payload["error"],
            },
            "launch_commands": _launch_commands(
                display_path,
                app_config_path=app_config_path,
            ),
        }
    )
    return payload


def _launch_commands(config_path: str, *, app_config_path: str) -> list[dict[str, Any]]:
    python = ".venv/bin/python"
    env = {"PYTHONPATH": "src"}
    commands = [
        {
            "id": "inspect_setup",
            "label": "Inspect setup",
            "description": "Preview setup readiness for this profile without side effects.",
            "argv": [
                python,
                "-m",
                "local_moe.cli",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--setup",
            ],
            "side_effects": "none",
            "requires_confirmation": False,
        },
        {
            "id": "prepare_runtime",
            "label": "Prepare runtime",
            "description": "Install runtime dependencies and download configured model assets.",
            "argv": [
                python,
                "scripts/bootstrap_runtime.py",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--execute",
                "--download-models",
            ],
            "side_effects": "installs_dependencies_and_downloads_models",
            "requires_confirmation": True,
        },
        {
            "id": "start_models",
            "label": "Start models",
            "description": "Start the model servers configured by this profile in the foreground.",
            "argv": [
                python,
                "scripts/start_local_models.py",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
            ],
            "side_effects": "starts_local_model_processes",
            "requires_confirmation": True,
        },
        {
            "id": "start_ui",
            "label": "Start UI",
            "description": "Run the web UI with this profile.",
            "argv": [
                python,
                "-m",
                "local_moe.web",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--port",
                "8089",
            ],
            "side_effects": "starts_local_web_server",
            "requires_confirmation": False,
        },
        {
            "id": "open_cli",
            "label": "Open CLI",
            "description": "Open an interactive CLI session with this profile.",
            "argv": [
                python,
                "-m",
                "local_moe.cli",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--interactive",
            ],
            "side_effects": "starts_interactive_cli",
            "requires_confirmation": False,
        },
    ]
    return [{**command, "env": env, "display": _display_command(command["argv"], env=env)} for command in commands]


def _contains_path(paths: list[Path], target: Path) -> bool:
    return any(_same_path(path, target) for path in paths)


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left).as_posix() == Path(right).as_posix()


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
    if not value or any(char.isspace() for char in value):
        return json.dumps(value)
    return value
