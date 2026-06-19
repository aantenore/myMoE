from __future__ import annotations

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
        }
    )
    return payload


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
