from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: str
    kind: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _now_iso())
    valid_from: str | None = None
    valid_until: str | None = None


@dataclass(frozen=True)
class MemoryMaintenanceReport:
    path: str
    total_records: int
    active_records: int
    expired_records: int


class FileMemoryStore:
    """Append-only local memory store; simple first layer before vector/graph backends."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(
        self,
        text: str,
        *,
        scope: str = "default",
        kind: str = "fact",
        metadata: dict[str, object] | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid4()),
            scope=scope,
            kind=kind,
            text=text,
            metadata=metadata or {},
            valid_from=valid_from,
            valid_until=valid_until,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")
        return record

    def list(self, *, scope: str | None = None) -> list[MemoryRecord]:
        records = _read_records(self.path)
        if scope is not None:
            records = [record for record in records if record.scope == scope]
        return records

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 8,
        now: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        query_terms = _terms(query)
        scored: list[tuple[MemoryRecord, float]] = []
        for record in self.list(scope=scope):
            if not _is_valid_at(record, now):
                continue
            score = _score(query_terms, _terms(record.text))
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0].created_at, item[0].id))
        return scored[:limit]

    def maintenance_report(self, *, now: str | None = None) -> MemoryMaintenanceReport:
        records = self.list()
        active = [record for record in records if _is_valid_at(record, now)]
        return MemoryMaintenanceReport(
            path=str(self.path),
            total_records=len(records),
            active_records=len(active),
            expired_records=len(records) - len(active),
        )


def _read_records(path: Path) -> list[MemoryRecord]:
    if not path.exists():
        return []
    records: list[MemoryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(
            MemoryRecord(
                id=str(raw["id"]),
                scope=str(raw["scope"]),
                kind=str(raw["kind"]),
                text=str(raw["text"]),
                metadata=dict(raw.get("metadata", {})),
                created_at=str(raw["created_at"]),
                valid_from=raw.get("valid_from"),
                valid_until=raw.get("valid_until"),
            )
        )
    return records


def _terms(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'`").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'`")
    }


def _score(query_terms: Iterable[str], text_terms: set[str]) -> float:
    query = set(query_terms)
    if not query:
        return 0.0
    return len(query & text_terms) / len(query)


def _is_valid_at(record: MemoryRecord, now: str | None) -> bool:
    if now is None:
        return record.valid_until is None
    if record.valid_from and record.valid_from > now:
        return False
    if record.valid_until and record.valid_until <= now:
        return False
    return True


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
