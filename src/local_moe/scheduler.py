from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .extensions import CronJobDefinition


@dataclass(frozen=True)
class DueJob:
    id: str
    command: tuple[str, ...]
    risk_class: str
    reason: str


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
