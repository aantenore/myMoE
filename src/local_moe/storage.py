from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

DEFAULT_MIN_FREE_GIB = 5.0


def build_storage_report(
    app_config: object,
    *,
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """Return read-only disk capacity diagnostics for local runtime paths."""

    paths = [
        _path_report(
            "model_cache_dir",
            str(app_config.runtime.model_cache_dir),
            min_free_gib=min_free_gib,
        ),
        _path_report(
            "work_dir",
            str(app_config.runtime.work_dir),
            min_free_gib=min_free_gib,
        ),
    ]
    attention = [item for item in paths if item["status"] != "ready"]
    return {
        "schema_version": "1.0",
        "status": "attention" if attention else "ready",
        "min_free_gib": min_free_gib,
        "summary": {
            "path_count": len(paths),
            "ready": sum(1 for item in paths if item["status"] == "ready"),
            "attention": sum(1 for item in paths if item["status"] == "attention"),
            "unavailable": sum(1 for item in paths if item["status"] == "unavailable"),
            "lowest_free_gib": _lowest_free_gib(paths),
        },
        "paths": paths,
        "recommendations": _recommendations(paths, min_free_gib=min_free_gib),
    }


def _path_report(label: str, raw_path: str, *, min_free_gib: float) -> dict[str, Any]:
    expanded = Path(raw_path).expanduser()
    probe_path = _nearest_existing_path(expanded)
    if probe_path is None:
        return {
            "label": label,
            "path": raw_path,
            "expanded_path": str(expanded),
            "exists": expanded.exists(),
            "probe_path": "",
            "status": "unavailable",
            "detail": "No existing parent path is available for disk usage inspection.",
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "total_gib": None,
            "used_gib": None,
            "free_gib": None,
        }
    try:
        usage = shutil.disk_usage(probe_path)
    except OSError as exc:
        return {
            "label": label,
            "path": raw_path,
            "expanded_path": str(expanded),
            "exists": expanded.exists(),
            "probe_path": str(probe_path),
            "status": "unavailable",
            "detail": str(exc),
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "total_gib": None,
            "used_gib": None,
            "free_gib": None,
        }
    free_gib = _bytes_to_gib(usage.free)
    status = "ready" if free_gib >= min_free_gib else "attention"
    detail = (
        "Storage has enough free space for local runtime operations."
        if status == "ready"
        else f"Free space is below the {min_free_gib:g} GiB local runtime threshold."
    )
    if not expanded.exists():
        detail += " The configured path does not exist yet; nearest existing parent was inspected."
    return {
        "label": label,
        "path": raw_path,
        "expanded_path": str(expanded),
        "exists": expanded.exists(),
        "probe_path": str(probe_path),
        "status": status,
        "detail": detail,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_gib": _bytes_to_gib(usage.total),
        "used_gib": _bytes_to_gib(usage.used),
        "free_gib": free_gib,
    }


def _nearest_existing_path(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _recommendations(paths: list[dict[str, Any]], *, min_free_gib: float) -> list[str]:
    recommendations: list[str] = []
    for item in paths:
        label = str(item.get("label", "path"))
        if item.get("status") == "attention":
            recommendations.append(
                f"Free at least {min_free_gib:g} GiB for {label}; current free space is {item.get('free_gib')} GiB."
            )
        elif item.get("status") == "unavailable":
            recommendations.append(f"Check storage permissions or parent directories for {label}.")
    if not recommendations:
        recommendations.append("Storage checks passed for configured local runtime paths.")
    return recommendations


def _lowest_free_gib(paths: list[dict[str, Any]]) -> float | None:
    values = [item.get("free_gib") for item in paths if item.get("free_gib") is not None]
    return min(float(value) for value in values) if values else None


def _bytes_to_gib(value: int) -> float:
    return round(value / 1024**3, 2)
