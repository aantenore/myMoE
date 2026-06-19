from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Callable

from .extensions import CronJobDefinition, ExtensionRegistry
from .memory import FileMemoryStore, memory_maintenance_payload, memory_prune_payload


WRITE_CONFIRMATION_RISK_CLASSES = frozenset(
    {
        "write_local",
        "write_internal",
        "write_external",
        "destructive",
        "process_execution",
        "privileged_admin",
    }
)


@dataclass(frozen=True)
class DueJob:
    id: str
    command: tuple[str, ...]
    risk_class: str
    reason: str


@dataclass(frozen=True)
class CronRunResult:
    id: str
    status: str
    reason: str
    command: tuple[str, ...]
    message: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CronRunSummary:
    now_epoch: float
    state_path: str
    due: tuple[str, ...]
    results: tuple[CronRunResult, ...]
    last_run_epoch: dict[str, float]


class BackgroundCronRunner:
    def __init__(
        self,
        jobs: tuple[CronJobDefinition, ...] | list[CronJobDefinition],
        *,
        state_path: str | Path,
        poll_seconds: float = 300,
        confirm_writes: bool = False,
        enabled: bool = False,
        registry: ExtensionRegistry | None = None,
        now_func: Callable[[], float] | None = None,
    ) -> None:
        self._jobs = tuple(jobs)
        self._state_path = state_path
        self._poll_seconds = max(float(poll_seconds), 1.0)
        self._confirm_writes = bool(confirm_writes)
        self._enabled = bool(enabled)
        self._registry = registry
        self._now_func = now_func or (lambda: datetime.now(timezone.utc).timestamp())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._run_count = 0
        self._last_run_epoch: float | None = None
        self._last_error: str | None = None
        self._last_summary: dict[str, object] | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="myMoE-background-cron",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def replace_registry(self, registry: ExtensionRegistry) -> None:
        with self._lock:
            self._jobs = tuple(registry.cron_jobs)
            self._registry = registry

    def run_once(self) -> CronRunSummary | None:
        now = self._now_func()
        try:
            summary = run_due_jobs(
                auto_runnable_jobs(self._jobs, confirm_writes=self._confirm_writes),
                state_path=self._state_path,
                now_epoch=now,
                confirm_writes=self._confirm_writes,
                registry=self._registry,
            )
            payload = cron_summary_payload(summary)
            with self._lock:
                self._run_count += 1
                self._last_run_epoch = now
                self._last_error = None
                self._last_summary = payload
            return summary
        except Exception as exc:  # pragma: no cover - defensive background guard
            with self._lock:
                self._run_count += 1
                self._last_run_epoch = now
                self._last_error = str(exc)
                self._last_summary = None
            return None

    def status_payload(self) -> dict[str, object]:
        auto_jobs = auto_runnable_jobs(self._jobs, confirm_writes=self._confirm_writes)
        auto_ids = {job.id for job in auto_jobs}
        skipped = tuple(
            job.id
            for job in self._jobs
            if job.enabled and job.id not in auto_ids and requires_write_confirmation(job.risk_class)
        )
        with self._lock:
            run_count = self._run_count
            last_run_epoch = self._last_run_epoch
            last_error = self._last_error
            last_summary = self._last_summary
        return {
            "enabled": self._enabled,
            "running": self.running,
            "poll_seconds": self._poll_seconds,
            "confirm_writes": self._confirm_writes,
            "policy": "all_allowlisted_jobs" if self._confirm_writes else "safe_jobs_only",
            "auto_job_ids": [job.id for job in auto_jobs],
            "skipped_job_ids": list(skipped),
            "run_count": run_count,
            "last_run_epoch": last_run_epoch,
            "last_error": last_error,
            "last_summary": last_summary,
        }

    def _loop(self) -> None:
        self.run_once()
        while not self._stop_event.wait(self._poll_seconds):
            self.run_once()


def auto_runnable_jobs(
    jobs: tuple[CronJobDefinition, ...] | list[CronJobDefinition],
    *,
    confirm_writes: bool = False,
) -> tuple[CronJobDefinition, ...]:
    if confirm_writes:
        return tuple(job for job in jobs if job.enabled)
    return tuple(
        job for job in jobs if job.enabled and not requires_write_confirmation(job.risk_class)
    )


def requires_write_confirmation(risk_class: str) -> bool:
    return risk_class in WRITE_CONFIRMATION_RISK_CLASSES


def due_jobs(
    jobs: tuple[CronJobDefinition, ...] | list[CronJobDefinition],
    last_run_epoch: dict[str, float],
    *,
    now_epoch: float | None = None,
) -> list[DueJob]:
    now = now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()
    due: list[DueJob] = []
    for job in jobs:
        if not job.enabled:
            continue
        schedule_type = str(job.schedule.get("type", "manual"))
        if schedule_type == "startup" and job.id not in last_run_epoch:
            due.append(DueJob(job.id, job.command, job.risk_class, "startup"))
            continue
        if schedule_type == "interval":
            seconds = float(job.schedule.get("seconds", 0))
            if seconds <= 0:
                continue
            previous = float(last_run_epoch.get(job.id, 0))
            if now - previous >= seconds:
                due.append(DueJob(job.id, job.command, job.risk_class, f"interval:{int(seconds)}"))
    return due


def cron_status(
    jobs: tuple[CronJobDefinition, ...] | list[CronJobDefinition],
    *,
    state_path: str | Path,
    now_epoch: float | None = None,
) -> dict[str, object]:
    now = now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()
    state = _load_state(state_path)
    due = due_jobs(jobs, state, now_epoch=now)
    due_ids = {job.id for job in due}
    return {
        "now_epoch": now,
        "state_path": str(state_path),
        "last_run_epoch": state,
        "jobs": [
            {
                "id": job.id,
                "enabled": job.enabled,
                "schedule": job.schedule,
                "command": list(job.command),
                "risk_class": job.risk_class,
                "auto_safe": not requires_write_confirmation(job.risk_class),
                "due": job.id in due_ids,
            }
            for job in jobs
        ],
    }


def run_due_jobs(
    jobs: tuple[CronJobDefinition, ...] | list[CronJobDefinition],
    *,
    state_path: str | Path,
    now_epoch: float | None = None,
    dry_run: bool = False,
    confirm_writes: bool = False,
    registry: ExtensionRegistry | None = None,
) -> CronRunSummary:
    now = now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()
    state = _load_state(state_path)
    due = due_jobs(jobs, state, now_epoch=now)
    results: list[CronRunResult] = []
    for job in due:
        if dry_run:
            result = CronRunResult(
                id=job.id,
                status="dry_run",
                reason=job.reason,
                command=job.command,
                message="Job is due but was not executed.",
            )
        elif not _is_allowlisted_action(job):
            result = _cron_error(job, f"Unsupported cron action: {job.command[0] if job.command else ''}")
        elif requires_write_confirmation(job.risk_class) and not confirm_writes:
            result = CronRunResult(
                id=job.id,
                status="needs_confirmation",
                reason=job.reason,
                command=job.command,
                message="Job writes local files and requires confirm_writes=true.",
            )
        else:
            result = _run_allowed_job(job, registry=registry)
            if result.status == "ok":
                state[job.id] = now
        results.append(result)

    if not dry_run:
        _save_state(state_path, state)

    return CronRunSummary(
        now_epoch=now,
        state_path=str(state_path),
        due=tuple(job.id for job in due),
        results=tuple(results),
        last_run_epoch=state,
    )


def cron_summary_payload(summary: CronRunSummary) -> dict[str, object]:
    return {
        "now_epoch": summary.now_epoch,
        "state_path": summary.state_path,
        "due": list(summary.due),
        "results": [
            {
                "id": result.id,
                "status": result.status,
                "reason": result.reason,
                "command": list(result.command),
                "message": result.message,
                "payload": result.payload,
            }
            for result in summary.results
        ],
        "last_run_epoch": dict(summary.last_run_epoch),
    }


def _run_allowed_job(job: DueJob, *, registry: ExtensionRegistry | None = None) -> CronRunResult:
    if not job.command:
        return _cron_error(job, "Cron job has no command.")

    action = job.command[0]
    args = _parse_command_args(job.command[1:])
    if action == "memory.maintenance":
        memory_path = args.get("memory-path", "work/runtime/memory.jsonl")
        report = FileMemoryStore(memory_path).maintenance_report(now=_now_iso())
        return CronRunResult(
            id=job.id,
            status="ok",
            reason=job.reason,
            command=job.command,
            message="Memory maintenance completed.",
            payload=memory_maintenance_payload(report),
        )

    if action == "memory.prune_expired":
        memory_path = args.get("memory-path", "work/runtime/memory.jsonl")
        report = FileMemoryStore(memory_path).prune_expired(now=_now_iso())
        return CronRunResult(
            id=job.id,
            status="ok",
            reason=job.reason,
            command=job.command,
            message="Expired memory records pruned.",
            payload=memory_prune_payload(report),
        )

    if action == "extension.audit":
        from .extensions import audit_extension_registry, load_extension_registry

        return CronRunResult(
            id=job.id,
            status="ok",
            reason=job.reason,
            command=job.command,
            message="Extension registry audit completed.",
            payload=audit_extension_registry(registry or load_extension_registry()),
        )

    if action == "router.distill":
        return _run_router_distill(job, args)

    return _cron_error(job, f"Unsupported cron action: {action}")


def _is_allowlisted_action(job: DueJob) -> bool:
    if not job.command:
        return False
    return job.command[0] in {"memory.maintenance", "memory.prune_expired", "extension.audit", "router.distill"}


def _run_router_distill(job: DueJob, args: dict[str, str]) -> CronRunResult:
    from .distilled_router import (
        labels_from_eval_cases,
        train_distilled_router_artifact,
        write_distilled_router_artifact,
        write_route_labels,
    )

    eval_path = args.get("eval")
    labels_path = args.get("labels")
    artifact_path = args.get("artifact")
    if not eval_path or not labels_path or not artifact_path:
        return _cron_error(job, "router.distill requires --eval, --labels, and --artifact.")

    raw_cases = [
        json.loads(line)
        for line in Path(eval_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = labels_from_eval_cases(raw_cases, teacher_source=args.get("teacher-source", "cron_curated_eval"))
    write_route_labels(labels, labels_path)
    artifact = train_distilled_router_artifact(labels)
    write_distilled_router_artifact(artifact, artifact_path)
    return CronRunResult(
        id=job.id,
        status="ok",
        reason=job.reason,
        command=job.command,
        message="Router distillation completed.",
        payload={
            "labels": len(labels),
            "labels_path": labels_path,
            "artifact_path": artifact_path,
            "experts": sorted(artifact["expert_counts"]),
        },
    )


def _cron_error(job: DueJob, message: str) -> CronRunResult:
    return CronRunResult(
        id=job.id,
        status="error",
        reason=job.reason,
        command=job.command,
        message=message,
    )


def _parse_command_args(args: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    index = 0
    while index < len(args):
        item = args[index]
        if item.startswith("--") and index + 1 < len(args):
            parsed[item[2:]] = args[index + 1]
            index += 2
        else:
            index += 1
    return parsed


def _load_state(path: str | Path) -> dict[str, float]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in raw.get("last_run_epoch", {}).items()}


def _save_state(path: str | Path, last_run_epoch: dict[str, float]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"last_run_epoch": last_run_epoch}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
