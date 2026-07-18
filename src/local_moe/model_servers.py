from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

from .bootstrap import RuntimePlan, build_runtime_plan, endpoint_is_reachable
from .config import MoEConfig
from .execution_scope import (
    ExecutionScopeGuard,
    ExecutionTarget,
    ExecutionTransport,
)


ReachabilityChecker = Callable[[str], bool]
ProcessFactory = Callable[[tuple[str, ...], Path], Any]


@dataclass(frozen=True)
class ModelServerSpec:
    expert_id: str
    model: str
    base_url: str
    command: tuple[str, ...]
    log_path: str
    execution_target: ExecutionTarget | None = None


@dataclass(frozen=True)
class ModelServerStatus:
    expert_id: str
    model: str
    base_url: str
    command: tuple[str, ...]
    log_path: str
    pid: int | None
    managed: bool
    running: bool
    endpoint_reachable: bool
    status: str
    message: str = ""


@dataclass(frozen=True)
class ModelServerAction:
    status: str
    ok: bool
    confirmed: bool
    only_first: bool = False
    results: tuple[ModelServerStatus, ...] = field(default_factory=tuple)
    message: str = ""


@dataclass(frozen=True)
class ModelServerLog:
    expert_id: str
    model: str
    log_path: str
    exists: bool
    status: str
    lines: tuple[str, ...] = ()
    max_lines: int = 120
    max_bytes: int = 65536
    bytes_total: int = 0
    bytes_read: int = 0
    truncated: bool = False
    sanitized: bool = True
    message: str = ""


class ModelServerManager:
    def __init__(
        self,
        specs: tuple[ModelServerSpec, ...],
        *,
        reachability_checker: ReachabilityChecker | None = None,
        process_factory: ProcessFactory | None = None,
        execution_guard: ExecutionScopeGuard | None = None,
    ) -> None:
        self._specs = specs
        self._reachability_checker = reachability_checker or endpoint_is_reachable
        self._process_factory = process_factory or _start_process
        self._execution_guard = execution_guard or ExecutionScopeGuard()
        self._processes: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        config: MoEConfig,
        *,
        preferred_backends: dict[str, str] | None = None,
        work_dir: str | Path = "work/runtime",
        reachability_checker: ReachabilityChecker | None = None,
        process_factory: ProcessFactory | None = None,
        execution_guard: ExecutionScopeGuard | None = None,
    ) -> "ModelServerManager":
        plan = build_runtime_plan(config, preferred_backends)
        guard = execution_guard or ExecutionScopeGuard(config.execution_policy)
        return cls(
            build_model_server_specs(
                config,
                plan,
                work_dir=work_dir,
            ),
            reachability_checker=reachability_checker,
            process_factory=process_factory,
            execution_guard=guard,
        )

    def status(self) -> dict[str, object]:
        return {
            "count": len(self._specs),
            "servers": [model_server_status_payload(item) for item in self._statuses()],
        }

    def logs(
        self,
        *,
        expert_id: str | None = None,
        max_lines: int = 120,
        max_bytes: int = 65536,
    ) -> dict[str, object]:
        selected = tuple(
            spec
            for spec in self._specs
            if expert_id is None or spec.expert_id == expert_id
        )
        return model_server_logs_payload(
            tuple(
                read_model_server_log(spec, max_lines=max_lines, max_bytes=max_bytes)
                for spec in selected
            ),
            expert_id=expert_id,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    def start(self, *, confirm: bool = False, only_first: bool = False) -> ModelServerAction:
        if not confirm:
            return ModelServerAction(
                status="confirmation_required",
                ok=False,
                confirmed=False,
                only_first=only_first,
                message="Starting model servers requires confirm=true.",
            )
        selected = self._specs[:1] if only_first else self._specs
        if not selected:
            return ModelServerAction(
                status="no_commands",
                ok=True,
                confirmed=True,
                only_first=only_first,
                message="No model server commands are configured for this profile.",
            )

        results: list[ModelServerStatus] = []
        for spec in selected:
            execution_allowed, execution_reason = self._execution_decision(spec)
            if not execution_allowed:
                results.append(
                    self._status_for_spec(
                        spec,
                        status_override="scope_blocked",
                        message=execution_reason
                        or "Model endpoint is outside the execution policy.",
                        execution_allowed=False,
                        execution_reason=execution_reason,
                        endpoint_reachable=False,
                    )
                )
                continue
            current = self._processes.get(spec.expert_id)
            if current is not None and _process_running(current):
                results.append(
                    self._status_for_spec(
                        spec,
                        message="Model server is already managed.",
                        execution_allowed=True,
                    )
                )
                continue
            if spec.base_url and self._reachability_checker(spec.base_url):
                results.append(
                    self._status_for_spec(
                        spec,
                        status_override="external_running",
                        message="Endpoint is already reachable; start skipped.",
                        execution_allowed=True,
                        endpoint_reachable=True,
                    )
                )
                continue
            if not _locally_manageable(spec):
                results.append(
                    self._status_for_spec(
                        spec,
                        status_override="external_unreachable",
                        message=(
                            "The attested external transport is unreachable and "
                            "cannot be started by the local model manager."
                        ),
                        execution_allowed=True,
                        endpoint_reachable=False,
                    )
                )
                continue
            execution_allowed, execution_reason = self._execution_decision(spec)
            if not execution_allowed:
                results.append(
                    self._status_for_spec(
                        spec,
                        status_override="scope_blocked",
                        message=execution_reason
                        or "Model endpoint is outside the execution policy.",
                        execution_allowed=False,
                        execution_reason=execution_reason,
                        endpoint_reachable=False,
                    )
                )
                continue
            process = self._process_factory(spec.command, Path(spec.log_path))
            self._processes[spec.expert_id] = process
            results.append(
                self._status_for_spec(
                    spec,
                    message="Model server started.",
                    execution_allowed=True,
                )
            )

        all_scope_blocked = bool(results) and all(
            item.status == "scope_blocked" for item in results
        )
        all_external_unreachable = bool(results) and all(
            item.status in {"scope_blocked", "external_unreachable"}
            for item in results
        )
        status = (
            "scope_blocked"
            if all_scope_blocked
            else (
                "external_unreachable"
                if all_external_unreachable
                else (
                    "started"
                    if any(item.status == "managed_running" for item in results)
                    else "skipped"
                )
            )
        )
        return ModelServerAction(
            status=status,
            ok=not (all_scope_blocked or all_external_unreachable),
            confirmed=True,
            only_first=only_first,
            results=tuple(results),
        )

    def stop(self, *, confirm: bool = False, timeout_seconds: float = 10) -> ModelServerAction:
        if not confirm:
            return ModelServerAction(
                status="confirmation_required",
                ok=False,
                confirmed=False,
                message="Stopping managed model servers requires confirm=true.",
            )
        results: list[ModelServerStatus] = []
        for spec in self._specs:
            process = self._processes.get(spec.expert_id)
            if process is None or not _process_running(process):
                results.append(self._status_for_spec(spec, message="No managed process is running."))
                continue
            _terminate_process(process, timeout_seconds=timeout_seconds)
            results.append(
                self._status_for_spec(
                    spec,
                    status_override="stopped",
                    message="Managed model server stopped.",
                )
            )
        return ModelServerAction(status="stopped", ok=True, confirmed=True, results=tuple(results))

    def close(self) -> None:
        for process in self._processes.values():
            if _process_running(process):
                _terminate_process(process, timeout_seconds=5)

    def _statuses(self) -> tuple[ModelServerStatus, ...]:
        return tuple(self._status_for_spec(spec) for spec in self._specs)

    def _status_for_spec(
        self,
        spec: ModelServerSpec,
        *,
        status_override: str | None = None,
        message: str = "",
        execution_allowed: bool | None = None,
        execution_reason: str = "",
        endpoint_reachable: bool | None = None,
    ) -> ModelServerStatus:
        if execution_allowed is None:
            execution_allowed, execution_reason = self._execution_decision(spec)
        process = self._processes.get(spec.expert_id)
        managed_running = process is not None and _process_running(process)
        pid = int(getattr(process, "pid", 0)) if managed_running else None
        if endpoint_reachable is None:
            endpoint_reachable = bool(
                execution_allowed
                and spec.base_url
                and self._reachability_checker(spec.base_url)
            )
        if not execution_allowed:
            status_override = status_override or "scope_blocked"
            message = message or execution_reason
        status = status_override or _server_status(managed_running, endpoint_reachable)
        return ModelServerStatus(
            expert_id=spec.expert_id,
            model=spec.model,
            base_url=spec.base_url,
            command=spec.command,
            log_path=spec.log_path,
            pid=pid,
            managed=managed_running,
            running=managed_running or endpoint_reachable,
            endpoint_reachable=endpoint_reachable,
            status=status,
            message=message,
        )

    def _execution_decision(self, spec: ModelServerSpec) -> tuple[bool, str]:
        if spec.execution_target is None:
            return True, ""
        eligibility = self._execution_guard.evaluate(spec.execution_target)
        return eligibility.allowed, eligibility.detail


def build_model_server_specs(
    config: MoEConfig,
    plan: RuntimePlan,
    *,
    work_dir: str | Path,
) -> tuple[ModelServerSpec, ...]:
    work = Path(work_dir)
    experts = tuple(expert for expert in config.experts if expert.provider == "openai_compatible")
    specs: list[ModelServerSpec] = []
    for index, expert in enumerate(experts):
        command = plan.model_commands[index] if index < len(plan.model_commands) else ()
        specs.append(
            ModelServerSpec(
                expert_id=expert.id,
                model=expert.model,
                base_url=expert.base_url or _base_url_from_command(command),
                command=command,
                log_path=str(work / f"model-{index + 1}.log"),
                execution_target=expert.execution_target,
            )
        )
    return tuple(specs)


def _locally_manageable(spec: ModelServerSpec) -> bool:
    target = spec.execution_target
    if target is None:
        return True
    return target.declaration.transport == ExecutionTransport.DIRECT_LOCAL


def model_server_action_payload(action: ModelServerAction) -> dict[str, object]:
    return {
        "status": action.status,
        "ok": action.ok,
        "confirmed": action.confirmed,
        "only_first": action.only_first,
        "message": action.message,
        "results": [model_server_status_payload(item) for item in action.results],
    }


def model_server_status_payload(status: ModelServerStatus) -> dict[str, object]:
    return {
        "expert_id": status.expert_id,
        "model": status.model,
        "base_url": status.base_url,
        "command": list(status.command),
        "command_display": " ".join(status.command),
        "log_path": status.log_path,
        "pid": status.pid,
        "managed": status.managed,
        "running": status.running,
        "endpoint_reachable": status.endpoint_reachable,
        "status": status.status,
        "message": status.message,
    }


def read_model_server_log(
    spec: ModelServerSpec,
    *,
    max_lines: int = 120,
    max_bytes: int = 65536,
) -> ModelServerLog:
    line_limit = _clamp_int(max_lines, minimum=1, maximum=1000)
    byte_limit = _clamp_int(max_bytes, minimum=1024, maximum=1048576)
    path = Path(spec.log_path)
    if not path.exists():
        return ModelServerLog(
            expert_id=spec.expert_id,
            model=spec.model,
            log_path=str(path),
            exists=False,
            status="missing",
            max_lines=line_limit,
            max_bytes=byte_limit,
            message="Log file has not been created yet.",
        )
    if not path.is_file():
        return ModelServerLog(
            expert_id=spec.expert_id,
            model=spec.model,
            log_path=str(path),
            exists=False,
            status="invalid",
            max_lines=line_limit,
            max_bytes=byte_limit,
            message="Log path is not a regular file.",
        )
    try:
        bytes_total = path.stat().st_size
        raw = _read_tail_bytes(path, max_bytes=byte_limit)
    except OSError as exc:
        return ModelServerLog(
            expert_id=spec.expert_id,
            model=spec.model,
            log_path=str(path),
            exists=True,
            status="error",
            max_lines=line_limit,
            max_bytes=byte_limit,
            message=str(exc),
        )
    text = raw.decode("utf-8", errors="replace")
    lines = tuple(_sanitize_log_line(line) for line in text.splitlines()[-line_limit:])
    return ModelServerLog(
        expert_id=spec.expert_id,
        model=spec.model,
        log_path=str(path),
        exists=True,
        status="ready",
        lines=lines,
        max_lines=line_limit,
        max_bytes=byte_limit,
        bytes_total=bytes_total,
        bytes_read=len(raw),
        truncated=bytes_total > len(raw) or len(text.splitlines()) > line_limit,
        sanitized=True,
    )


def model_server_logs_payload(
    logs: tuple[ModelServerLog, ...],
    *,
    expert_id: str | None = None,
    max_lines: int = 120,
    max_bytes: int = 65536,
) -> dict[str, object]:
    return {
        "count": len(logs),
        "expert_id": expert_id,
        "max_lines": _clamp_int(max_lines, minimum=1, maximum=1000),
        "max_bytes": _clamp_int(max_bytes, minimum=1024, maximum=1048576),
        "sanitized": True,
        "logs": [model_server_log_payload(item) for item in logs],
    }


def model_server_log_payload(log: ModelServerLog) -> dict[str, object]:
    return {
        "expert_id": log.expert_id,
        "model": log.model,
        "log_path": log.log_path,
        "exists": log.exists,
        "status": log.status,
        "lines": list(log.lines),
        "line_count": len(log.lines),
        "max_lines": log.max_lines,
        "max_bytes": log.max_bytes,
        "bytes_total": log.bytes_total,
        "bytes_read": log.bytes_read,
        "truncated": log.truncated,
        "sanitized": log.sanitized,
        "message": log.message,
    }


def _server_status(managed_running: bool, endpoint_reachable: bool) -> str:
    if managed_running:
        return "managed_running"
    if endpoint_reachable:
        return "external_running"
    return "stopped"


def _read_tail_bytes(path: Path, *, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(size - max_bytes, 0))
        return handle.read()


def _sanitize_log_line(line: str) -> str:
    sanitized = line
    sanitized = re.sub(r"hf_[A-Za-z0-9_-]{12,}", "[REDACTED_HF_TOKEN]", sanitized)
    sanitized = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}", r"\1[REDACTED_TOKEN]", sanitized)
    sanitized = re.sub(
        r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED_SECRET]",
        sanitized,
    )
    return sanitized


def _clamp_int(raw: object, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = minimum
    return max(minimum, min(value, maximum))


def _base_url_from_command(command: tuple[str, ...]) -> str:
    host = _value_after(command, "--host") or "127.0.0.1"
    port = _value_after(command, "--port")
    if not port:
        return ""
    return f"http://{host}:{port}/v1"


def _value_after(command: tuple[str, ...], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return command[index + 1]


def _start_process(command: tuple[str, ...], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    try:
        return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    finally:
        log_file.close()


def _process_running(process: Any) -> bool:
    return process.poll() is None


def _terminate_process(process: Any, *, timeout_seconds: float) -> None:
    if not _process_running(process):
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


def wait_for_managed_processes(manager: ModelServerManager) -> None:
    try:
        while True:
            statuses = manager.status()["servers"]
            running = [item for item in statuses if isinstance(item, dict) and item.get("managed")]
            if not running:
                raise SystemExit("All managed model servers exited.")
            time.sleep(2)
    except KeyboardInterrupt:
        print()
    finally:
        manager.close()
