from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any, Callable

from .app_config import load_app_config
from .config import load_config
from .model_downloads import build_model_download_requests, validate_local_file_request
from .setup_status import RuntimeSetupStatus, inspect_setup_status, setup_status_payload


CommandRunner = Callable[[tuple[str, ...]], None]
SnapshotDownloader = Callable[..., object]


@dataclass(frozen=True)
class SetupRunStep:
    phase: str
    status: str
    message: str
    command: tuple[str, ...] = ()
    model: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSetupRun:
    status: str
    ok: bool
    execute: bool
    download_models: bool
    confirmed: bool
    setup_before: RuntimeSetupStatus
    setup_after: RuntimeSetupStatus
    steps: tuple[SetupRunStep, ...]


def run_runtime_setup(
    *,
    config_path: str,
    app_config_path: str = "configs/app.json",
    execute: bool = False,
    download_models: bool = False,
    confirm: bool = False,
    command_runner: CommandRunner | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
) -> RuntimeSetupRun:
    app_config = load_app_config(app_config_path)
    config = load_config(config_path)
    setup_before = inspect_setup_status(
        config_path,
        config,
        app_config,
        app_config_path=app_config_path,
    )

    requested_side_effects = execute or download_models
    if requested_side_effects and not confirm:
        step = SetupRunStep(
            phase="setup",
            status="confirmation_required",
            message="Runtime preparation requires confirm=true before installs or downloads run.",
        )
        return RuntimeSetupRun(
            status="confirmation_required",
            ok=False,
            execute=execute,
            download_models=download_models,
            confirmed=confirm,
            setup_before=setup_before,
            setup_after=setup_before,
            steps=(step,),
        )

    runner = command_runner or _default_command_runner
    steps: list[SetupRunStep] = []
    if execute:
        steps.extend(_run_install_commands(setup_before.runtime_plan.install_commands, runner))
    if download_models and not _has_blocking_failure(steps):
        steps.extend(
            _run_model_downloads(
                config,
                setup_before.runtime_plan.backend,
                runner,
                snapshot_downloader=snapshot_downloader,
            )
        )

    setup_after = inspect_setup_status(
        config_path,
        config,
        app_config,
        app_config_path=app_config_path,
    )
    status = _overall_status(steps, setup_after.status, requested_side_effects=requested_side_effects)
    return RuntimeSetupRun(
        status=status,
        ok=status in {"planned", "ready"},
        execute=execute,
        download_models=download_models,
        confirmed=confirm,
        setup_before=setup_before,
        setup_after=setup_after,
        steps=tuple(steps),
    )


def setup_run_payload(run: RuntimeSetupRun) -> dict[str, Any]:
    return {
        "status": run.status,
        "ok": run.ok,
        "execute": run.execute,
        "download_models": run.download_models,
        "confirmed": run.confirmed,
        "setup_before": setup_status_payload(run.setup_before),
        "setup_after": setup_status_payload(run.setup_after),
        "steps": [
            {
                "phase": step.phase,
                "status": step.status,
                "message": step.message,
                "command": list(step.command),
                "model": step.model,
                "payload": step.payload,
            }
            for step in run.steps
        ],
    }


def _run_install_commands(
    commands: tuple[tuple[str, ...], ...],
    command_runner: CommandRunner,
) -> list[SetupRunStep]:
    steps: list[SetupRunStep] = []
    for command in commands:
        if command and command[0] == "install":
            steps.append(
                SetupRunStep(
                    phase="install",
                    status="manual_required",
                    command=command,
                    message=f"Manual install required: {' '.join(command)}",
                )
            )
            continue
        if tuple(command[:2]) == ("uv", "venv") and _venv_exists():
            steps.append(
                SetupRunStep(
                    phase="install",
                    status="skipped",
                    command=command,
                    message="Existing virtual environment detected.",
                )
            )
            continue
        try:
            command_runner(command)
        except Exception as exc:
            steps.append(
                SetupRunStep(
                    phase="install",
                    status="error",
                    command=command,
                    message=str(exc),
                )
            )
            break
        steps.append(
            SetupRunStep(
                phase="install",
                status="ok",
                command=command,
                message="Install command completed.",
            )
        )
    return steps


def _run_model_downloads(
    config: object,
    default_backend: str,
    command_runner: CommandRunner,
    *,
    snapshot_downloader: SnapshotDownloader | None = None,
) -> list[SetupRunStep]:
    try:
        requests = build_model_download_requests(config, default_backend)
    except ValueError as exc:
        return [SetupRunStep(phase="download", status="error", message=str(exc))]

    steps: list[SetupRunStep] = []
    downloader = snapshot_downloader
    for request in requests:
        if request.kind == "local_file":
            try:
                validate_local_file_request(request)
            except FileNotFoundError as exc:
                steps.append(
                    SetupRunStep(
                        phase="download",
                        status="error",
                        model=request.model,
                        message=str(exc),
                    )
                )
                break
            steps.append(
                SetupRunStep(
                    phase="download",
                    status="ok",
                    model=request.model,
                    message="Using existing local model file.",
                )
            )
            continue

        if request.kind == "ollama_pull":
            try:
                command_runner(request.command)
            except Exception as exc:
                steps.append(
                    SetupRunStep(
                        phase="download",
                        status="error",
                        command=request.command,
                        model=request.model,
                        message=str(exc),
                    )
                )
                break
            steps.append(
                SetupRunStep(
                    phase="download",
                    status="ok",
                    command=request.command,
                    model=request.model,
                    message="Ollama model pull completed.",
                )
            )
            continue

        if request.kind == "huggingface_snapshot":
            if not request.repo_id:
                steps.append(
                    SetupRunStep(
                        phase="download",
                        status="error",
                        model=request.model,
                        message="Missing Hugging Face repo id.",
                    )
                )
                break
            if downloader is None:
                try:
                    from huggingface_hub import snapshot_download
                except Exception as exc:
                    steps.append(
                        SetupRunStep(
                            phase="download",
                            status="error",
                            model=request.model,
                            message=f"huggingface_hub is required: {exc}",
                        )
                    )
                    break
                downloader = snapshot_download
            kwargs: dict[str, object] = {}
            if request.allow_patterns:
                kwargs["allow_patterns"] = list(request.allow_patterns)
            try:
                downloader(request.repo_id, **kwargs)
            except Exception as exc:
                steps.append(
                    SetupRunStep(
                        phase="download",
                        status="error",
                        model=request.model,
                        message=str(exc),
                        payload={"repo_id": request.repo_id},
                    )
                )
                break
            steps.append(
                SetupRunStep(
                    phase="download",
                    status="ok",
                    model=request.model,
                    message="Hugging Face snapshot is available.",
                    payload={"repo_id": request.repo_id, "allow_patterns": list(request.allow_patterns)},
                )
            )
            continue

        steps.append(
            SetupRunStep(
                phase="download",
                status="error",
                model=request.model,
                message=f"Unsupported model download request: {request.kind}",
            )
        )
        break
    return steps


def _overall_status(
    steps: list[SetupRunStep],
    setup_after_status: str,
    *,
    requested_side_effects: bool,
) -> str:
    statuses = {step.status for step in steps}
    if "error" in statuses:
        return "error"
    if "confirmation_required" in statuses:
        return "confirmation_required"
    if not requested_side_effects:
        return "planned"
    if "manual_required" in statuses:
        return "manual_required"
    return "ready" if setup_after_status == "ready" else "needs_setup"


def _has_blocking_failure(steps: list[SetupRunStep]) -> bool:
    return any(step.status in {"error", "confirmation_required"} for step in steps)


def _default_command_runner(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


def _venv_exists() -> bool:
    return Path(".venv/bin/python").exists() or Path(".venv/Scripts/python.exe").exists()
