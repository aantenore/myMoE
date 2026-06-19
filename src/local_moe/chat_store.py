from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ChatMessage:
    id: str
    role: str
    content: str
    created_at: str = field(default_factory=lambda: _now_iso())
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class ChatSessionSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class FileChatStore:
    """Small local JSON chat store for the web UI."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def list_sessions(self, *, limit: int = 50) -> list[ChatSessionSummary]:
        with self._lock:
            sessions = self._read_unlocked()
        summaries = [
            ChatSessionSummary(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=len(session.messages),
            )
            for session in sessions
        ]
        summaries.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        return summaries[:limit]

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._lock:
            for session in self._read_unlocked():
                if session.id == session_id:
                    return session
        return None

    def create_session(self, *, title: str | None = None) -> ChatSession:
        now = _now_iso()
        session = ChatSession(
            id=str(uuid4()),
            title=_clean_title(title) or "New chat",
            created_at=now,
            updated_at=now,
            messages=(),
        )
        with self._lock:
            sessions = self._read_unlocked()
            sessions.append(session)
            self._write_unlocked(sessions)
        return session

    def append_exchange(
        self,
        *,
        session_id: str | None,
        user_content: str,
        assistant_content: str,
        assistant_meta: dict[str, Any] | None = None,
    ) -> ChatSession:
        now = _now_iso()
        user_message = ChatMessage(id=str(uuid4()), role="user", content=user_content, created_at=now)
        assistant_message = ChatMessage(
            id=str(uuid4()),
            role="assistant",
            content=assistant_content,
            created_at=now,
            meta=assistant_meta or {},
        )
        with self._lock:
            sessions = self._read_unlocked()
            index = _find_session_index(sessions, session_id) if session_id else None
            if index is None:
                session = ChatSession(
                    id=str(uuid4()),
                    title=_title_from_prompt(user_content),
                    created_at=now,
                    updated_at=now,
                    messages=(user_message, assistant_message),
                )
                sessions.append(session)
            else:
                current = sessions[index]
                title = current.title if current.title != "New chat" else _title_from_prompt(user_content)
                session = ChatSession(
                    id=current.id,
                    title=title,
                    created_at=current.created_at,
                    updated_at=now,
                    messages=(*current.messages, user_message, assistant_message),
                )
                sessions[index] = session
            self._write_unlocked(sessions)
        return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            sessions = self._read_unlocked()
            kept = [session for session in sessions if session.id != session_id]
            if len(kept) == len(sessions):
                return False
            self._write_unlocked(kept)
        return True

    def _read_unlocked(self) -> list[ChatSession]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Chat store must contain a JSON object.")
        sessions_raw = raw.get("sessions", [])
        if not isinstance(sessions_raw, list):
            raise ValueError("Chat store sessions must be a JSON array.")
        return [_parse_session(item) for item in sessions_raw]

    def _write_unlocked(self, sessions: list[ChatSession]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [chat_session_payload(session) for session in sessions]}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self.path)


def chat_summary_payload(summary: ChatSessionSummary) -> dict[str, Any]:
    return {
        "id": summary.id,
        "title": summary.title,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "message_count": summary.message_count,
    }


def chat_session_payload(session: ChatSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
                "meta": message.meta,
            }
            for message in session.messages
        ],
    }


def _parse_session(raw: object) -> ChatSession:
    if not isinstance(raw, dict):
        raise ValueError("Chat session must be a JSON object.")
    messages_raw = raw.get("messages", [])
    if not isinstance(messages_raw, list):
        raise ValueError("Chat session messages must be a JSON array.")
    return ChatSession(
        id=str(raw["id"]),
        title=str(raw.get("title", "New chat")),
        created_at=str(raw["created_at"]),
        updated_at=str(raw["updated_at"]),
        messages=tuple(_parse_message(item) for item in messages_raw),
    )


def _parse_message(raw: object) -> ChatMessage:
    if not isinstance(raw, dict):
        raise ValueError("Chat message must be a JSON object.")
    meta = raw.get("meta", {})
    return ChatMessage(
        id=str(raw["id"]),
        role=str(raw["role"]),
        content=str(raw["content"]),
        created_at=str(raw["created_at"]),
        meta=meta if isinstance(meta, dict) else {},
    )


def _find_session_index(sessions: list[ChatSession], session_id: str | None) -> int | None:
    if not session_id:
        return None
    for index, session in enumerate(sessions):
        if session.id == session_id:
            return index
    raise KeyError(session_id)


def _title_from_prompt(prompt: str) -> str:
    title = " ".join(prompt.strip().split())
    if not title:
        return "New chat"
    if len(title) <= 56:
        return title
    return title[:53].rstrip() + "..."


def _clean_title(title: str | None) -> str:
    return " ".join(str(title or "").strip().split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
