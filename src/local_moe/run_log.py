from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class GenerationRunRecord:
    id: str
    created_at: str
    mode: str
    correlation_id: str
    session_id: str = ""
    prompt_sha256: str = ""
    prompt_chars: int = 0
    selected_experts: tuple[str, ...] = ()
    fallback_order: tuple[str, ...] = ()
    result_models: tuple[str, ...] = ()
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    predicted_tokens_per_second: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    disagreement_status: str = ""


@dataclass(frozen=True)
class RunLogPruneReport:
    keep: int
    before_count: int
    after_count: int
    removed_count: int
    path: str


class RunLogStore:
    """Append-only metadata log for generation observability."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def record_generation(
        self,
        *,
        mode: str,
        prompt: str,
        response_payload: dict[str, Any],
        context_payload: dict[str, Any] | None = None,
        session_id: str = "",
        latency_ms: int | None = None,
    ) -> GenerationRunRecord:
        record = _record_from_payload(
            mode=mode,
            prompt=prompt,
            response_payload=response_payload,
            context_payload=context_payload or {},
            session_id=session_id,
            latency_ms=latency_ms,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_record_payload(record), ensure_ascii=True) + "\n")
        return record

    def list_records(self, *, limit: int = 100) -> list[GenerationRunRecord]:
        with self._lock:
            records = _read_records(self.path)
        records.reverse()
        return records[: max(1, min(limit, 500))]

    def prune(self, *, keep: int = 1000) -> RunLogPruneReport:
        bounded_keep = max(1, min(int(keep), 100000))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = _read_records(self.path)
            before_count = len(records)
            kept = records[-bounded_keep:]
            _write_records(self.path, kept)
        return RunLogPruneReport(
            keep=bounded_keep,
            before_count=before_count,
            after_count=len(kept),
            removed_count=max(0, before_count - len(kept)),
            path=str(self.path),
        )


def run_record_payload(record: GenerationRunRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "created_at": record.created_at,
        "mode": record.mode,
        "correlation_id": record.correlation_id,
        "session_id": record.session_id,
        "prompt_sha256": record.prompt_sha256,
        "prompt_chars": record.prompt_chars,
        "selected_experts": list(record.selected_experts),
        "fallback_order": list(record.fallback_order),
        "result_models": list(record.result_models),
        "latency_ms": record.latency_ms,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "predicted_tokens_per_second": record.predicted_tokens_per_second,
        "context": record.context,
        "error_count": record.error_count,
        "disagreement_status": record.disagreement_status,
    }


def run_log_payload(records: list[GenerationRunRecord], *, path: str | Path) -> dict[str, Any]:
    return {
        "count": len(records),
        "path": str(path),
        "privacy": "metadata_only",
        "records": [run_record_payload(record) for record in records],
    }


def run_log_prune_payload(report: RunLogPruneReport) -> dict[str, Any]:
    return {
        "keep": report.keep,
        "before_count": report.before_count,
        "after_count": report.after_count,
        "removed_count": report.removed_count,
        "path": report.path,
    }


def _record_from_payload(
    *,
    mode: str,
    prompt: str,
    response_payload: dict[str, Any],
    context_payload: dict[str, Any],
    session_id: str,
    latency_ms: int | None,
) -> GenerationRunRecord:
    route = response_payload.get("route", {})
    selected = route.get("selected", []) if isinstance(route, dict) else []
    fallback = route.get("fallback_order", []) if isinstance(route, dict) else []
    results = response_payload.get("results", [])
    errors = response_payload.get("errors", [])
    disagreement = response_payload.get("disagreement")
    return GenerationRunRecord(
        id=str(uuid4()),
        created_at=_now_iso(),
        mode=_clean(mode) or "generate",
        correlation_id=str(response_payload.get("correlation_id", "")),
        session_id=_clean(session_id),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_chars=len(prompt),
        selected_experts=tuple(str(item.get("expert_id", "")) for item in selected if isinstance(item, dict)),
        fallback_order=tuple(str(item) for item in fallback),
        result_models=tuple(str(item.get("model", "")) for item in results if isinstance(item, dict)),
        latency_ms=latency_ms,
        prompt_tokens=_sum_int(results, "prompt_tokens"),
        completion_tokens=_sum_int(results, "completion_tokens"),
        predicted_tokens_per_second=_first_float(results, "predicted_tokens_per_second"),
        context=_safe_context(context_payload),
        error_count=len(errors) if isinstance(errors, list) else 0,
        disagreement_status=str(disagreement.get("status", "")) if isinstance(disagreement, dict) else "",
    )


def _safe_context(context: dict[str, Any]) -> dict[str, Any]:
    sections = context.get("sections", {})
    return {
        "token_estimate": _optional_int(context.get("token_estimate")),
        "budget_tokens": _optional_int(context.get("budget_tokens")),
        "compaction_needed": bool(context.get("compaction_needed", False)),
        "dropped_turns": _optional_int(context.get("dropped_turns")) or 0,
        "section_tokens": sections if isinstance(sections, dict) else {},
        "memory_ids": [str(item) for item in context.get("memory_ids", []) if str(item).strip()],
    }


def _read_records(path: Path) -> list[GenerationRunRecord]:
    if not path.exists():
        return []
    records: list[GenerationRunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            continue
        context = raw.get("context", {})
        records.append(
            GenerationRunRecord(
                id=str(raw["id"]),
                created_at=str(raw["created_at"]),
                mode=str(raw.get("mode", "generate")),
                correlation_id=str(raw.get("correlation_id", "")),
                session_id=str(raw.get("session_id", "")),
                prompt_sha256=str(raw.get("prompt_sha256", "")),
                prompt_chars=int(raw.get("prompt_chars", 0)),
                selected_experts=tuple(str(item) for item in raw.get("selected_experts", [])),
                fallback_order=tuple(str(item) for item in raw.get("fallback_order", [])),
                result_models=tuple(str(item) for item in raw.get("result_models", [])),
                latency_ms=_optional_int(raw.get("latency_ms")),
                prompt_tokens=_optional_int(raw.get("prompt_tokens")),
                completion_tokens=_optional_int(raw.get("completion_tokens")),
                predicted_tokens_per_second=_optional_float(raw.get("predicted_tokens_per_second")),
                context=context if isinstance(context, dict) else {},
                error_count=int(raw.get("error_count", 0)),
                disagreement_status=str(raw.get("disagreement_status", "")),
            )
        )
    return records


def _write_records(path: Path, records: list[GenerationRunRecord]) -> None:
    payload = "".join(json.dumps(run_record_payload(record), ensure_ascii=True) + "\n" for record in records)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _sum_int(items: object, key: str) -> int | None:
    if not isinstance(items, list):
        return None
    total = 0
    seen = False
    for item in items:
        if not isinstance(item, dict) or item.get(key) is None:
            continue
        try:
            total += int(item[key])
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def _first_float(items: object, key: str) -> float | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get(key) is None:
            continue
        value = _optional_float(item.get(key))
        if value is not None:
            return value
    return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: object) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
