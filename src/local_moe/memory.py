from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Iterable
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
    checked_at: str
    total_records: int
    active_records: int
    pending_records: int
    expired_records: int


@dataclass(frozen=True)
class MemoryPruneReport:
    path: str
    checked_at: str
    before_count: int
    removed_count: int
    remaining_count: int
    removed_ids: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeIngestReport:
    document_id: str
    title: str
    scope: str
    chunk_count: int
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryForgetReport:
    target: str
    removed_count: int
    remaining_count: int
    removed_ids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryRestoreReport:
    mode: str
    imported_count: int
    updated_count: int
    skipped_count: int
    total_records: int


def memory_record_payload(record: MemoryRecord, *, score: float | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": record.id,
        "scope": record.scope,
        "kind": record.kind,
        "text": record.text,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "valid_from": record.valid_from,
        "valid_until": record.valid_until,
    }
    if score is not None:
        payload["score"] = score
    return payload


def memory_maintenance_payload(report: MemoryMaintenanceReport) -> dict[str, object]:
    return {
        "path": report.path,
        "checked_at": report.checked_at,
        "total_records": report.total_records,
        "active_records": report.active_records,
        "pending_records": report.pending_records,
        "expired_records": report.expired_records,
    }


def memory_prune_payload(report: MemoryPruneReport) -> dict[str, object]:
    return {
        "path": report.path,
        "checked_at": report.checked_at,
        "before_count": report.before_count,
        "removed_count": report.removed_count,
        "remaining_count": report.remaining_count,
        "removed_ids": list(report.removed_ids),
    }


class FileMemoryStore:
    """Local JSONL memory store; simple first layer before vector/graph backends."""

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

    def forget_record(self, record_id: str) -> MemoryForgetReport:
        target = str(record_id).strip()
        if not target:
            raise ValueError("record_id is required.")
        return self._forget(lambda record: record.id == target, target=target)

    def forget_document(self, document_id: str) -> MemoryForgetReport:
        target = str(document_id).strip()
        if not target:
            raise ValueError("document_id is required.")
        return self._forget(
            lambda record: str(record.metadata.get("document_id", "")) == target,
            target=target,
        )

    def restore_records(self, raw_records: list[object], *, mode: str = "merge") -> MemoryRestoreReport:
        if mode not in {"merge", "replace"}:
            raise ValueError("mode must be merge or replace.")
        if not isinstance(raw_records, list):
            raise ValueError("records must be a JSON array.")
        imported = [_parse_record(item) for item in raw_records]
        seen: set[str] = set()
        for record in imported:
            if record.id in seen:
                raise ValueError(f"Duplicate memory record id in bundle: {record.id}")
            seen.add(record.id)

        existing = [] if mode == "replace" else _read_records(self.path)
        by_id = {record.id: index for index, record in enumerate(existing)}
        updated_count = 0
        imported_count = 0
        for record in imported:
            if record.id in by_id:
                existing[by_id[record.id]] = record
                updated_count += 1
            else:
                existing.append(record)
                imported_count += 1
        _write_records(self.path, existing)
        return MemoryRestoreReport(
            mode=mode,
            imported_count=imported_count,
            updated_count=updated_count,
            skipped_count=0,
            total_records=len(existing),
        )

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

    def ingest_document(
        self,
        content: str,
        *,
        title: str,
        scope: str = "default",
        chunk_chars: int = 1200,
        metadata: dict[str, object] | None = None,
    ) -> KnowledgeIngestReport:
        clean_title = _clean_text(title)
        clean_content = _clean_text(content)
        if not clean_title:
            raise ValueError("title is required.")
        if not clean_content:
            raise ValueError("content is required.")
        if chunk_chars < 200 or chunk_chars > 8000:
            raise ValueError("chunk_chars must be between 200 and 8000.")

        document_id = str(uuid4())
        chunks = _chunk_text(clean_content, chunk_chars=chunk_chars)
        records = []
        base_metadata = dict(metadata or {})
        for index, chunk in enumerate(chunks, start=1):
            record = self.add(
                f"{clean_title}\n\n{chunk}",
                scope=scope,
                kind="knowledge",
                metadata={
                    **base_metadata,
                    "document_id": document_id,
                    "title": clean_title,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            )
            records.append(record)
        return KnowledgeIngestReport(
            document_id=document_id,
            title=clean_title,
            scope=scope,
            chunk_count=len(records),
            record_ids=tuple(record.id for record in records),
        )

    def maintenance_report(self, *, now: str | None = None) -> MemoryMaintenanceReport:
        checked_at = now or _now_iso()
        records = self.list()
        active = [record for record in records if _is_valid_at(record, checked_at)]
        pending = [record for record in records if _is_pending_at(record, checked_at)]
        expired = [record for record in records if _is_expired_at(record, checked_at)]
        return MemoryMaintenanceReport(
            path=str(self.path),
            checked_at=checked_at,
            total_records=len(records),
            active_records=len(active),
            pending_records=len(pending),
            expired_records=len(expired),
        )

    def prune_expired(self, *, now: str | None = None) -> MemoryPruneReport:
        checked_at = now or _now_iso()
        records = _read_records(self.path)
        kept: list[MemoryRecord] = []
        removed: list[MemoryRecord] = []
        for record in records:
            if _is_expired_at(record, checked_at):
                removed.append(record)
            else:
                kept.append(record)
        if removed:
            _write_records(self.path, kept)
        return MemoryPruneReport(
            path=str(self.path),
            checked_at=checked_at,
            before_count=len(records),
            removed_count=len(removed),
            remaining_count=len(kept),
            removed_ids=tuple(record.id for record in removed),
        )

    def _forget(self, predicate: Callable[[MemoryRecord], bool], *, target: str) -> MemoryForgetReport:
        records = _read_records(self.path)
        kept: list[MemoryRecord] = []
        removed: list[MemoryRecord] = []
        for record in records:
            if predicate(record):
                removed.append(record)
            else:
                kept.append(record)
        if removed:
            _write_records(self.path, kept)
        return MemoryForgetReport(
            target=target,
            removed_count=len(removed),
            remaining_count=len(kept),
            removed_ids=tuple(record.id for record in removed),
        )


def _read_records(path: Path) -> list[MemoryRecord]:
    if not path.exists():
        return []
    records: list[MemoryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(_parse_record(json.loads(line)))
    return records


def _parse_record(raw: object) -> MemoryRecord:
    if not isinstance(raw, dict):
        raise ValueError("Memory record must be a JSON object.")
    return MemoryRecord(
        id=str(raw["id"]),
        scope=str(raw["scope"]),
        kind=str(raw["kind"]),
        text=str(raw["text"]),
        metadata=dict(raw.get("metadata", {})),
        created_at=str(raw["created_at"]),
        valid_from=raw.get("valid_from"),
        valid_until=raw.get("valid_until"),
    )


def _write_records(path: Path, records: list[MemoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")
    tmp.replace(path)


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


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_wrap(paragraph, chunk_chars=chunk_chars))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())
    return chunks or [text[:chunk_chars]]


def _hard_wrap(text: str, *, chunk_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word[:chunk_chars]
        remainder = word[chunk_chars:]
        while remainder:
            chunks.append(current)
            current = remainder[:chunk_chars]
            remainder = remainder[chunk_chars:]
    if current:
        chunks.append(current)
    return chunks


def _clean_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines()).strip()


def _is_valid_at(record: MemoryRecord, now: str | None) -> bool:
    checked_at = now or _now_iso()
    if record.valid_from and record.valid_from > checked_at:
        return False
    if record.valid_until and record.valid_until <= checked_at:
        return False
    return True


def _is_pending_at(record: MemoryRecord, now: str) -> bool:
    return bool(record.valid_from and record.valid_from > now)


def _is_expired_at(record: MemoryRecord, now: str) -> bool:
    return bool(record.valid_until and record.valid_until <= now)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
