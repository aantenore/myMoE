from __future__ import annotations

import time
from typing import Any

from .app_config import load_app_config
from .config import load_config
from .doctor import build_doctor_report
from .extensions import ExtensionRegistry
from .model_servers import (
    ModelServerManager,
    model_server_action_payload,
)
from .setup_runner import (
    CommandRunner,
    SnapshotDownloader,
    run_runtime_setup,
    setup_run_payload,
)
from .setup_status import inspect_setup_status, setup_status_payload


def run_startup_readiness(
    *,
    config_path: str,
    app_config_path: str = "configs/app.json",
    prepare: bool = False,
    download_models: bool = False,
    start_models: bool = False,
    confirm: bool = False,
    only_first: bool = False,
    model_start_wait_seconds: float = 45.0,
    model_start_poll_seconds: float = 2.0,
    model_manager: ModelServerManager | None = None,
    registry: ExtensionRegistry | None = None,
    command_runner: CommandRunner | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
) -> dict[str, Any]:
    app_config = load_app_config(app_config_path)
    config = load_config(config_path)
    manager = model_manager or ModelServerManager.from_config(
        config,
        preferred_backends=app_config.runtime.preferred_backends,
        work_dir=app_config.runtime.work_dir,
    )
    requested_side_effects = prepare or download_models or start_models

    setup_before = setup_status_payload(
        inspect_setup_status(
            config_path,
            config,
            app_config,
            app_config_path=app_config_path,
        )
    )
    steps: list[dict[str, Any]] = [
        {
            "id": "inspect",
            "status": setup_before.get("status", "unknown"),
            "message": "Inspected runtime setup and model assets.",
        }
    ]

    if requested_side_effects and not confirm:
        steps.append(
            {
                "id": "confirm",
                "status": "confirmation_required",
                "message": "Startup preparation and model starts require confirm=true.",
            }
        )
        doctor = _doctor(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=app_config_path,
            registry=registry,
            model_manager=manager,
        )
        return _startup_payload(
            status="confirmation_required",
            ok=False,
            confirmed=False,
            prepare=prepare,
            download_models=download_models,
            start_models=start_models,
            only_first=only_first,
            setup=setup_before,
            doctor=doctor,
            model_processes=manager.status(),
            steps=steps,
        )

    setup_run: dict[str, Any] | None = None
    if prepare or download_models:
        result = run_runtime_setup(
            config_path=config_path,
            app_config_path=app_config_path,
            execute=prepare,
            download_models=download_models,
            confirm=confirm,
            command_runner=command_runner,
            snapshot_downloader=snapshot_downloader,
        )
        setup_run = setup_run_payload(result)
        steps.append(
            {
                "id": "prepare",
                "status": setup_run.get("status", "unknown"),
                "message": "Runtime preparation completed." if result.ok else "Runtime preparation needs attention.",
            }
        )
    else:
        steps.append(
            {
                "id": "prepare",
                "status": "skipped",
                "message": "Runtime preparation was not requested.",
            }
        )

    model_action: dict[str, Any] | None = None
    setup_after = setup_run.get("setup_after") if isinstance(setup_run, dict) else setup_before
    setup_ready = isinstance(setup_after, dict) and setup_after.get("status") == "ready"
    if start_models and setup_ready:
        action = manager.start(confirm=True, only_first=only_first)
        model_action = model_server_action_payload(action)
        steps.append(
            {
                "id": "start_models",
                "status": model_action.get("status", "unknown"),
                "message": "Model start action completed." if action.ok else "Model start action needs attention.",
            }
        )
        wait_status = _wait_for_reachable_model_endpoints(
            manager,
            timeout_seconds=model_start_wait_seconds,
            poll_seconds=model_start_poll_seconds,
        )
        steps.append(
            {
                "id": "wait_models",
                "status": wait_status["status"],
                "message": wait_status["message"],
            }
        )
    elif start_models:
        model_action = {
            "status": "skipped",
            "ok": False,
            "confirmed": confirm,
            "only_first": only_first,
            "message": "Model start skipped because setup is not ready.",
            "results": [],
        }
        steps.append(
            {
                "id": "start_models",
                "status": "skipped",
                "message": "Model start skipped because setup is not ready.",
            }
        )
        steps.append(
            {
                "id": "wait_models",
                "status": "skipped",
                "message": "Model endpoint wait skipped because setup is not ready.",
            }
        )
    else:
        steps.append(
            {
                "id": "start_models",
                "status": "skipped",
                "message": "Model start was not requested.",
            }
        )
        steps.append(
            {
                "id": "wait_models",
                "status": "skipped",
                "message": "No model start was requested.",
            }
        )

    doctor = _doctor(
        config_path=config_path,
        config=config,
        app_config=app_config,
        app_config_path=app_config_path,
        registry=registry,
        model_manager=manager,
    )
    steps.append(
        {
            "id": "doctor",
            "status": doctor.get("status", "unknown"),
            "message": "System Doctor completed.",
        }
    )
    status = _overall_startup_status(
        requested_side_effects=requested_side_effects,
        setup_run=setup_run,
        model_action=model_action,
        doctor=doctor,
    )
    return _startup_payload(
        status=status,
        ok=status == "ready" or (status == "planned" and doctor.get("status") == "ready"),
        confirmed=confirm,
        prepare=prepare,
        download_models=download_models,
        start_models=start_models,
        only_first=only_first,
        setup=setup_after if isinstance(setup_after, dict) else setup_before,
        setup_run=setup_run,
        model_action=model_action,
        doctor=doctor,
        model_processes=manager.status(),
        steps=steps,
    )


def _doctor(
    *,
    config_path: str,
    config: object,
    app_config: object,
    app_config_path: str,
    registry: ExtensionRegistry | None,
    model_manager: ModelServerManager,
) -> dict[str, Any]:
    return build_doctor_report(
        config_path=config_path,
        config=config,
        app_config=app_config,
        app_config_path=app_config_path,
        registry=registry,
        model_manager=model_manager,
    )


def _startup_payload(
    *,
    status: str,
    ok: bool,
    confirmed: bool,
    prepare: bool,
    download_models: bool,
    start_models: bool,
    only_first: bool,
    setup: dict[str, Any],
    doctor: dict[str, Any],
    model_processes: dict[str, object],
    steps: list[dict[str, Any]],
    setup_run: dict[str, Any] | None = None,
    model_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "ok": ok,
        "confirmed": confirmed,
        "prepare": prepare,
        "download_models": download_models,
        "start_models": start_models,
        "only_first": only_first,
        "setup": setup,
        "setup_run": setup_run,
        "model_action": model_action,
        "doctor": doctor,
        "model_processes": model_processes,
        "steps": steps,
    }


def _wait_for_reachable_model_endpoints(
    manager: ModelServerManager,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, str]:
    if timeout_seconds <= 0:
        return {"status": "skipped", "message": "Model endpoint wait was disabled."}
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses = manager.status()["servers"]
        servers = [item for item in statuses if isinstance(item, dict)]
        if not servers:
            return {"status": "skipped", "message": "No model endpoints are configured."}
        if all(bool(item.get("endpoint_reachable")) for item in servers):
            return {"status": "ready", "message": "Configured model endpoints are reachable."}
        if time.monotonic() >= deadline:
            return {"status": "timeout", "message": "Timed out waiting for model endpoints."}
        time.sleep(max(0.25, min(poll_seconds, 5.0)))


def _overall_startup_status(
    *,
    requested_side_effects: bool,
    setup_run: dict[str, Any] | None,
    model_action: dict[str, Any] | None,
    doctor: dict[str, Any],
) -> str:
    if setup_run and setup_run.get("status") in {"error", "manual_required", "needs_setup"}:
        return str(setup_run.get("status"))
    if model_action and not bool(model_action.get("ok")):
        return str(model_action.get("status") or "model_start_failed")
    if not requested_side_effects:
        return "planned"
    return "ready" if doctor.get("status") == "ready" else "needs_attention"
