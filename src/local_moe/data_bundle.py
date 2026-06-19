from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .chat_store import FileChatStore, chat_session_payload
from .memory import FileMemoryStore, memory_record_payload


SCHEMA_VERSION = "mymoe.local-data.v1"


@dataclass(frozen=True)
class LocalDataRestoreReport:
    mode: str
    chats: dict[str, Any]
    memory: dict[str, Any]


def build_local_data_bundle(
    *,
    chat_store: FileChatStore,
    memory_store: FileMemoryStore,
) -> dict[str, Any]:
    """Build a portable local data bundle containing user-owned chats and memory."""

    sessions = [chat_session_payload(session) for session in chat_store.list_full_sessions()]
    records = [memory_record_payload(record) for record in memory_store.list()]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "privacy": {
            "contains_user_content": True,
            "contains_chat_transcripts": True,
            "contains_memory_records": True,
            "contains_environment": False,
            "contains_model_logs": False,
        },
        "counts": {
            "chat_sessions": len(sessions),
            "chat_messages": sum(len(session.get("messages", [])) for session in sessions),
            "memory_records": len(records),
        },
        "data": {
            "chats": {
                "sessions": sessions,
            },
            "memory": {
                "records": records,
            },
        },
    }


def restore_local_data_bundle(
    bundle: dict[str, Any],
    *,
    chat_store: FileChatStore,
    memory_store: FileMemoryStore,
    mode: str = "merge",
) -> LocalDataRestoreReport:
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be a JSON object.")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported local data bundle schema: {bundle.get('schema_version')}")
    if mode not in {"merge", "replace"}:
        raise ValueError("mode must be merge or replace.")
    data = bundle.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("bundle.data must be a JSON object.")
    chats = data.get("chats", {})
    memory = data.get("memory", {})
    if not isinstance(chats, dict):
        raise ValueError("bundle.data.chats must be a JSON object.")
    if not isinstance(memory, dict):
        raise ValueError("bundle.data.memory must be a JSON object.")

    chat_report = chat_store.restore_sessions(chats.get("sessions", []), mode=mode)
    memory_report = memory_store.restore_records(memory.get("records", []), mode=mode)
    return LocalDataRestoreReport(
        mode=mode,
        chats=asdict(chat_report),
        memory=asdict(memory_report),
    )


def local_data_bundle_filename(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"mymoe-local-data-{stamp}.json"


def local_data_restore_payload(report: LocalDataRestoreReport) -> dict[str, Any]:
    return {
        "mode": report.mode,
        "chats": report.chats,
        "memory": report.memory,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
