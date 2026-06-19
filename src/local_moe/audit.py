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
        event = AuditEvent(
            id=str(uuid4()),
            created_at=_now_iso(),
            action=_clean(action),
            status=_clean(status),
            risk_class=_clean(risk_class) or "read_only",
            subject=_clean(subject),
            metadata=_safe_metadata(metadata or {}),
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
