from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .app_config import AppConfig
from .bootstrap import RuntimePlan, build_runtime_plan, runtime_plan_payload
from .config import MoEConfig
from .model_downloads import ModelDownloadRequest, build_model_download_requests


READY_MODEL_STATUSES = {"available", "cached"}
READY_RUNTIME_STATUSES = {"available", "not_required"}


@dataclass(frozen=True)
class ModelAssetStatus:
    kind: str
    backend: str
    model: str
    status: str
    detail: str
    repo_id: str | None = None
    allow_patterns: tuple[str, ...] = ()
    cache_path: str | None = None
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    backend: str
    status: str
    detail: str
    module: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeSetupStatus:
    status: str
    config_path: str
    model_cache_dir: str
    download_command: tuple[str, ...]
    runtime_plan: RuntimePlan
    models: tuple[ModelAssetStatus, ...]
    runtime_dependencies: tuple[RuntimeDependencyStatus, ...] = ()
    error: str = ""


def inspect_setup_status(
    config_path: str,
    config: MoEConfig,
    app_config: AppConfig,
    *,
    app_config_path: str = "configs/app.json",
) -> RuntimeSetupStatus:
    plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
    command = _download_command(config_path, app_config_path)
    try:
        requests = build_model_download_requests(config, plan.backend)
    except ValueError as exc:
        return RuntimeSetupStatus(
            status="needs_setup",
            config_path=config_path,
            model_cache_dir=app_config.runtime.model_cache_dir,
            download_command=command,
            runtime_plan=plan,
            models=(),
            runtime_dependencies=_inspect_runtime_dependencies(plan),
            error=str(exc),
        )

    models = tuple(_inspect_request(item, app_config.runtime.model_cache_dir) for item in requests)
    runtime_dependencies = _inspect_runtime_dependencies(plan)
    ready = all(item.status in READY_MODEL_STATUSES for item in models)
    runtime_ready = all(
        item.status in READY_RUNTIME_STATUSES for item in runtime_dependencies
    )
    return RuntimeSetupStatus(
        status="ready" if ready and runtime_ready else "needs_setup",
        config_path=config_path,
        model_cache_dir=app_config.runtime.model_cache_dir,
        download_command=command,
        runtime_plan=plan,
        models=models,
        runtime_dependencies=runtime_dependencies,
    )


def setup_status_payload(status: RuntimeSetupStatus) -> dict[str, Any]:
    return {
        "status": status.status,
        "config_path": status.config_path,
        "model_cache_dir": status.model_cache_dir,
        "download_command": list(status.download_command),
        "download_command_display": _format_python_command(status.download_command),
        "runtime": runtime_plan_payload(status.runtime_plan),
        "models": [
            {
                "kind": item.kind,
                "backend": item.backend,
                "model": item.model,
                "repo_id": item.repo_id,
                "allow_patterns": list(item.allow_patterns),
                "status": item.status,
                "detail": item.detail,
                "cache_path": item.cache_path,
                "command": list(item.command),
                "command_display": _format_command(item.command) if item.command else "",
            }
            for item in status.models
        ],
        "runtime_dependencies": [
            {
                "backend": item.backend,
                "module": item.module,
                "status": item.status,
                "detail": item.detail,
                "command": list(item.command),
                "command_display": _format_command(item.command) if item.command else "",
            }
            for item in status.runtime_dependencies
        ],
        "error": status.error,
    }


def _inspect_runtime_dependencies(
    plan: RuntimePlan,
) -> tuple[RuntimeDependencyStatus, ...]:
    probes: dict[tuple[str, str], tuple[str, ...]] = {}
    for command in plan.model_commands:
        if len(command) >= 3 and command[1] == "-m":
            module = command[2]
            backend = _module_backend(module)
            if backend:
                probes[(backend, module)] = (command[0], "-c", f"import {module}")
    if not probes:
        return ()

    return tuple(
        _inspect_python_module(backend, module, command)
        for (backend, module), command in sorted(probes.items())
    )


def _module_backend(module: str) -> str:
    if module == "mlx_lm.server":
        return "mlx_lm"
    if module == "mlx_vlm.server":
        return "mlx_vlm"
    return ""


def _inspect_python_module(
    backend: str,
    module: str,
    command: tuple[str, ...],
) -> RuntimeDependencyStatus:
    executable = command[0]
    if not _python_executable_exists(executable):
        return RuntimeDependencyStatus(
            backend=backend,
            module=module,
            status="missing_runtime",
            detail=f"Python runtime was not found at {executable}.",
            command=command,
        )

    status, detail = _probe_python_module(executable, module)
    return RuntimeDependencyStatus(
        backend=backend,
        module=module,
        status=status,
        detail=detail,
        command=command,
    )


@lru_cache(maxsize=16)
def _probe_python_module(executable: str, module: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            (executable, "-c", f"import {module}"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            "missing_runtime",
            f"Python runtime cannot import {module}: {type(exc).__name__}: {exc}",
        )

    if completed.returncode == 0:
        return ("available", f"Python runtime can import {module}.")
    detail = _last_nonempty_line(completed.stderr) or _last_nonempty_line(
        completed.stdout
    )
    if not detail:
        detail = f"exit code {completed.returncode}"
    return ("missing_runtime", f"Python runtime cannot import {module}: {detail}")


def _python_executable_exists(executable: str) -> bool:
    return Path(executable).exists()


def _inspect_request(request: ModelDownloadRequest, model_cache_dir: str) -> ModelAssetStatus:
    if request.kind == "local_file":
        path = Path(request.model).expanduser()
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            status="available" if path.exists() else "missing",
            detail="Local model file exists." if path.exists() else "Local model file is missing.",
            cache_path=str(path),
        )

    if request.kind == "ollama_pull":
        if shutil.which("ollama") is None:
            return ModelAssetStatus(
                kind=request.kind,
                backend=request.backend,
                model=request.model,
                status="missing_runtime",
                detail="Ollama is not installed or not on PATH.",
                command=request.command,
            )
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            status="pull_required",
            detail="Run the pull command unless the model is already present in Ollama.",
            command=request.command,
        )

    if request.kind == "huggingface_snapshot":
        return _inspect_huggingface_snapshot(request, model_cache_dir)

    return ModelAssetStatus(
        kind=request.kind,
        backend=request.backend,
        model=request.model,
        status="unsupported",
        detail=f"Unsupported model setup request kind: {request.kind}.",
    )


def _inspect_huggingface_snapshot(
    request: ModelDownloadRequest,
    model_cache_dir: str,
) -> ModelAssetStatus:
    if not request.repo_id:
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            status="missing_repo",
            detail="Hugging Face snapshot request has no repo id.",
            allow_patterns=request.allow_patterns,
        )

    repo_cache = _repo_cache_path(request.repo_id, model_cache_dir)
    if not repo_cache.exists():
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            repo_id=request.repo_id,
            allow_patterns=request.allow_patterns,
            status="missing",
            detail="Hugging Face cache folder was not found.",
            cache_path=str(repo_cache),
        )

    files = tuple(_iter_files(repo_cache))
    if not files:
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            repo_id=request.repo_id,
            allow_patterns=request.allow_patterns,
            status="partial",
            detail="Hugging Face cache folder exists but contains no files.",
            cache_path=str(repo_cache),
        )

    if request.allow_patterns and not _has_matching_file(files, request.allow_patterns):
        return ModelAssetStatus(
            kind=request.kind,
            backend=request.backend,
            model=request.model,
            repo_id=request.repo_id,
            allow_patterns=request.allow_patterns,
            status="partial",
            detail="Cache exists, but no file matches the configured allow patterns.",
            cache_path=str(repo_cache),
        )

    return ModelAssetStatus(
        kind=request.kind,
        backend=request.backend,
        model=request.model,
        repo_id=request.repo_id,
        allow_patterns=request.allow_patterns,
        status="cached",
        detail="Matching Hugging Face snapshot files are present in the local cache.",
        cache_path=str(repo_cache),
    )


def _repo_cache_path(repo_id: str, model_cache_dir: str) -> Path:
    hub_cache = _hub_cache_dir(model_cache_dir)
    return hub_cache / f"models--{repo_id.replace('/', '--')}"


def _hub_cache_dir(model_cache_dir: str) -> Path:
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()
    root = Path(model_cache_dir).expanduser()
    if root.name == "hub":
        return root
    return root / "hub"


def _iter_files(root: Path) -> tuple[Path, ...]:
    try:
        return tuple(path for path in root.rglob("*") if path.is_file())
    except OSError:
        return ()


def _has_matching_file(files: tuple[Path, ...], allow_patterns: tuple[str, ...]) -> bool:
    for path in files:
        name = path.name
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in allow_patterns):
            return True
    return False


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _download_command(config_path: str, app_config_path: str) -> tuple[str, ...]:
    command = [
        _venv_python(),
        "scripts/bootstrap_runtime.py",
        "--app-config",
        app_config_path,
        "--config",
        config_path,
        "--execute",
        "--download-models",
    ]
    return tuple(command)


def _venv_python() -> str:
    return ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(command)


def _format_python_command(command: tuple[str, ...]) -> str:
    if os.name == "nt":
        return f"set PYTHONPATH=src && {' '.join(command)}"
    return f"PYTHONPATH=src {' '.join(command)}"
