from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    id: str
    created_at: str
    action: str
    status: str
    risk_class: str
    subject: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditPruneReport:
    keep: int
    before_count: int
    after_count: int
    removed_count: int
    path: str
    event: AuditEvent


class AuditLogStore:
    """Local JSONL audit trail for sensitive host-side actions."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        status: str,
        *,
        risk_class: str = "read_only",
        subject: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = _build_event(
            action,
            status,
            risk_class=risk_class,
            subject=subject,
            metadata=metadata,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit_event_payload(event), ensure_ascii=True) + "\n")
        return event

    def list_events(
        self,
        *,
        limit: int = 100,
        action: str | None = None,
        status: str | None = None,
    ) -> list[AuditEvent]:
        clean_action = _clean(action)
        clean_status = _clean(status)
        with self._lock:
            events = _read_events(self.path)
        if clean_action:
            events = [event for event in events if event.action == clean_action]
        if clean_status:
            events = [event for event in events if event.status == clean_status]
        events.reverse()
        return events[: max(1, min(limit, 500))]

    def prune(self, *, keep: int = 500) -> AuditPruneReport:
        bounded_keep = max(1, min(int(keep), 50000))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = _read_events(self.path)
            before_count = len(events)
            after_count = min(before_count + 1, bounded_keep)
            removed_count = before_count + 1 - after_count
            event = _build_event(
                "audit.prune",
                "ok",
                risk_class="write_local",
                metadata={
                    "keep": bounded_keep,
                    "before_count": before_count,
                    "after_count": after_count,
                    "removed_count": removed_count,
                },
            )
            kept_events = [*events, event][-bounded_keep:]
            _write_events(self.path, kept_events)
        return AuditPruneReport(
            keep=bounded_keep,
            before_count=before_count,
            after_count=after_count,
            removed_count=removed_count,
            path=str(self.path),
            event=event,
        )


def audit_event_payload(event: AuditEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "created_at": event.created_at,
        "action": event.action,
        "status": event.status,
        "risk_class": event.risk_class,
        "subject": event.subject,
        "metadata": event.metadata,
    }


def audit_log_payload(events: list[AuditEvent]) -> dict[str, Any]:
    return {
        "count": len(events),
        "events": [audit_event_payload(event) for event in events],
    }


def audit_prune_payload(report: AuditPruneReport) -> dict[str, Any]:
    return {
        "keep": report.keep,
        "before_count": report.before_count,
        "after_count": report.after_count,
        "removed_count": report.removed_count,
        "path": report.path,
        "event": audit_event_payload(report.event),
    }


def _read_events(path: Path) -> list[AuditEvent]:
    if not path.exists():
        return []
    events: list[AuditEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata", {})
        events.append(
            AuditEvent(
                id=str(raw["id"]),
                created_at=str(raw["created_at"]),
                action=str(raw["action"]),
                status=str(raw["status"]),
                risk_class=str(raw.get("risk_class", "read_only")),
                subject=str(raw.get("subject", "")),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return events


def _write_events(path: Path, events: list[AuditEvent]) -> None:
    payload = "".join(json.dumps(audit_event_payload(event), ensure_ascii=True) + "\n" for event in events)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _build_event(
    action: str,
    status: str,
    *,
    risk_class: str = "read_only",
    subject: str = "",
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    return AuditEvent(
        id=str(uuid4()),
        created_at=_now_iso(),
        action=_clean(action),
        status=_clean(status),
        risk_class=_clean(risk_class) or "read_only",
        subject=_clean(subject),
        metadata=_safe_metadata(metadata or {}),
    )


def _safe_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        safe[_clean(key)[:80]] = _safe_value(value)
    return safe


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        clean = _clean(value)
        return clean if len(clean) <= 240 else clean[:237] + "..."
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return _safe_metadata(value)
    return _safe_value(str(value))


def _clean(value: object | None) -> str:
    return " ".join(str(value or "").strip().split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
