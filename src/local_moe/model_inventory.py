from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from .bootstrap import build_runtime_plan
from .model_downloads import ModelDownloadRequest, build_model_download_requests

DEFAULT_MAX_FILES = 20_000
READY_STATUSES = {"available", "cached", "runtime_managed"}


def build_model_asset_inventory(
    *,
    config_path: str,
    config: object,
    app_config: object,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Return a read-only inventory for model assets required by the active config."""

    try:
        plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
        requests = build_model_download_requests(config, plan.backend)
    except ValueError as exc:
        return {
            "schema_version": "1.0",
            "status": "attention",
            "config_path": config_path,
            "model_cache_dir": str(app_config.runtime.model_cache_dir),
            "summary": _summary([]),
            "assets": [],
            "recommendations": [str(exc)],
        }

    assets = [
        _asset_inventory(request, str(app_config.runtime.model_cache_dir), max_files=max_files)
        for request in requests
    ]
    status = "ready" if all(item["status"] in READY_STATUSES for item in assets) else "attention"
    return {
        "schema_version": "1.0",
        "status": status,
        "config_path": config_path,
        "model_cache_dir": str(app_config.runtime.model_cache_dir),
        "max_files": max_files,
        "summary": _summary(assets),
        "assets": assets,
        "recommendations": _recommendations(assets),
    }


def _asset_inventory(request: ModelDownloadRequest, model_cache_dir: str, *, max_files: int) -> dict[str, Any]:
    if request.kind == "local_file":
        path = Path(request.model).expanduser()
        return _local_file_inventory(request, path)
    if request.kind == "huggingface_snapshot":
        return _huggingface_inventory(request, model_cache_dir, max_files=max_files)
    if request.kind == "ollama_pull":
        return {
            **_base_asset(request),
            "status": "runtime_managed",
            "detail": "Ollama manages this model outside the configured Hugging Face cache.",
            "path": "",
            "exists": None,
            "file_count": None,
            "matched_file_count": None,
            "cache_size_bytes": None,
            "cache_size_gib": None,
            "configured_size_bytes": None,
            "configured_size_gib": None,
            "truncated": False,
            "command": list(request.command),
        }
    return {
        **_base_asset(request),
        "status": "unsupported",
        "detail": f"Unsupported model asset request kind: {request.kind}.",
        "path": "",
        "exists": None,
        "file_count": None,
        "matched_file_count": None,
        "cache_size_bytes": None,
        "cache_size_gib": None,
        "configured_size_bytes": None,
        "configured_size_gib": None,
        "truncated": False,
        "command": list(request.command),
    }


def _local_file_inventory(request: ModelDownloadRequest, path: Path) -> dict[str, Any]:
    exists = path.exists()
    size = _file_size(path) if exists else None
    return {
        **_base_asset(request),
        "status": "available" if exists else "missing",
        "detail": "Local model file exists." if exists else "Local model file is missing.",
        "path": str(path),
        "exists": exists,
        "file_count": 1 if exists else 0,
        "matched_file_count": 1 if exists else 0,
        "cache_size_bytes": size,
        "cache_size_gib": _bytes_to_gib(size),
        "configured_size_bytes": size,
        "configured_size_gib": _bytes_to_gib(size),
        "truncated": False,
        "command": list(request.command),
    }


def _huggingface_inventory(
    request: ModelDownloadRequest,
    model_cache_dir: str,
    *,
    max_files: int,
) -> dict[str, Any]:
    if not request.repo_id:
        return {
            **_base_asset(request),
            "status": "missing_repo",
            "detail": "Hugging Face snapshot request has no repo id.",
            "path": "",
            "exists": None,
            "file_count": None,
            "matched_file_count": None,
            "cache_size_bytes": None,
            "cache_size_gib": None,
            "configured_size_bytes": None,
            "configured_size_gib": None,
            "truncated": False,
            "command": list(request.command),
        }

    repo_cache = _repo_cache_path(request.repo_id, model_cache_dir)
    if not repo_cache.exists():
        return {
            **_base_asset(request),
            "status": "missing",
            "detail": "Hugging Face cache folder was not found.",
            "path": str(repo_cache),
            "exists": False,
            "file_count": 0,
            "matched_file_count": 0,
            "cache_size_bytes": 0,
            "cache_size_gib": 0.0,
            "configured_size_bytes": 0,
            "configured_size_gib": 0.0,
            "truncated": False,
            "command": list(request.command),
        }

    scan = _scan_files(repo_cache, request.allow_patterns, max_files=max_files)
    status = "cached"
    detail = "Configured Hugging Face model files are present in the local cache."
    if scan["error"]:
        status = "unavailable"
        detail = str(scan["error"])
    elif scan["file_count"] == 0:
        status = "partial"
        detail = "Hugging Face cache folder exists but contains no files."
    elif request.allow_patterns and scan["matched_file_count"] == 0:
        status = "partial"
        detail = "Cache exists, but no file matches the configured allow patterns."

    configured_size = (
        scan["matched_size_bytes"] if request.allow_patterns else scan["cache_size_bytes"]
    )
    return {
        **_base_asset(request),
        "status": status,
        "detail": detail,
        "path": str(repo_cache),
        "exists": True,
        "file_count": scan["file_count"],
        "matched_file_count": scan["matched_file_count"],
        "cache_size_bytes": scan["cache_size_bytes"],
        "cache_size_gib": _bytes_to_gib(scan["cache_size_bytes"]),
        "configured_size_bytes": configured_size,
        "configured_size_gib": _bytes_to_gib(configured_size),
        "truncated": scan["truncated"],
        "command": list(request.command),
    }


def _base_asset(request: ModelDownloadRequest) -> dict[str, Any]:
    return {
        "kind": request.kind,
        "backend": request.backend,
        "model": request.model,
        "repo_id": request.repo_id,
        "allow_patterns": list(request.allow_patterns),
    }


def _scan_files(root: Path, allow_patterns: tuple[str, ...], *, max_files: int) -> dict[str, Any]:
    cache_size = 0
    matched_size = 0
    file_count = 0
    matched_count = 0
    truncated = False
    seen: set[str] = set()
    try:
        iterator = root.rglob("*")
        for path in iterator:
            if not path.is_file():
                continue
            file_count += 1
            if file_count > max_files:
                truncated = True
                break
            size = _deduped_size(path, seen)
            cache_size += size
            if not allow_patterns or any(fnmatch.fnmatchcase(path.name, pattern) for pattern in allow_patterns):
                matched_count += 1
                matched_size += size
    except OSError as exc:
        return {
            "error": str(exc),
            "file_count": file_count,
            "matched_file_count": matched_count,
            "cache_size_bytes": cache_size,
            "matched_size_bytes": matched_size,
            "truncated": truncated,
        }
    return {
        "error": "",
        "file_count": file_count,
        "matched_file_count": matched_count,
        "cache_size_bytes": cache_size,
        "matched_size_bytes": matched_size,
        "truncated": truncated,
    }


def _deduped_size(path: Path, seen: set[str]) -> int:
    try:
        key = str(path.resolve(strict=True))
    except OSError:
        key = str(path.absolute())
    if key in seen:
        return 0
    seen.add(key)
    return _file_size(path)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    configured_size = sum(int(item.get("configured_size_bytes") or 0) for item in assets)
    cache_size = sum(int(item.get("cache_size_bytes") or 0) for item in assets)
    return {
        "asset_count": len(assets),
        "ready": sum(1 for item in assets if item["status"] in READY_STATUSES),
        "attention": sum(1 for item in assets if item["status"] not in READY_STATUSES),
        "configured_size_bytes": configured_size,
        "configured_size_gib": _bytes_to_gib(configured_size),
        "cache_size_bytes": cache_size,
        "cache_size_gib": _bytes_to_gib(cache_size),
    }


def _recommendations(assets: list[dict[str, Any]]) -> list[str]:
    recommendations = []
    for item in assets:
        status = item.get("status")
        if status in READY_STATUSES:
            continue
        model = item.get("model") or item.get("repo_id") or "model"
        if status == "missing":
            recommendations.append(f"Download or prepare the configured asset for {model}.")
        elif status == "partial":
            recommendations.append(f"Re-run runtime preparation for {model}; the cache looks incomplete.")
        elif status == "missing_repo":
            recommendations.append(f"Check the model repository identifier for {model}.")
        else:
            recommendations.append(f"Inspect model asset status for {model}: {item.get('detail', status)}")
    if not recommendations:
        recommendations.append("Configured model assets are present or managed by their runtime.")
    return recommendations


def _repo_cache_path(repo_id: str, model_cache_dir: str) -> Path:
    return _hub_cache_dir(model_cache_dir) / f"models--{repo_id.replace('/', '--')}"


def _hub_cache_dir(model_cache_dir: str) -> Path:
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()
    root = Path(model_cache_dir).expanduser()
    if root.name == "hub":
        return root
    return root / "hub"


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1024**3, 3)
