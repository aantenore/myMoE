from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from .chat_store import ChatSession, FileChatStore, chat_session_payload
from .context import ContextBundle, ConversationTurn, MemorySnippet, build_context_bundle
from .memory import FileMemoryStore
from .orchestrator import LocalMoE
from .run_log import RunLogStore


def generate_chat_turn(
    *,
    moe: LocalMoE,
    chat_store: FileChatStore,
    memory_store: FileMemoryStore,
    run_log_store: RunLogStore,
    context_policy: object,
    prompt: str,
    session_id: str | None = None,
    correlation_id: str | None = None,
    mode: str = "generate",
) -> dict[str, Any]:
    session = _session_for_id(chat_store, session_id)
    model_context = build_chat_model_context(
        session,
        prompt,
        context_policy,
        memory_store=memory_store,
    )
    started_at = time.monotonic()
    response = moe.generate(
        model_context["prompt"],
        correlation_id=correlation_id,
        route_prompt=prompt,
    )
    return persist_chat_response(
        chat_store=chat_store,
        run_log_store=run_log_store,
        prompt=prompt,
        response=response,
        context_payload=model_context["payload"],
        session_id=session_id,
        mode=mode,
        started_at=started_at,
    )


def persist_chat_response(
    *,
    chat_store: FileChatStore,
    run_log_store: RunLogStore,
    prompt: str,
    response: object,
    context_payload: dict[str, Any],
    session_id: str | None,
    mode: str,
    started_at: float,
) -> dict[str, Any]:
    response_payload = chat_response_payload(response)
    response_payload["context"] = context_payload
    session = chat_store.append_exchange(
        session_id=session_id,
        user_content=prompt,
        assistant_content=response.content,
        assistant_meta={
            "correlation_id": response_payload["correlation_id"],
            "route": response_payload["route"],
            "results": response_payload["results"],
            "errors": response_payload["errors"],
            "disagreement": response_payload["disagreement"],
            "context": response_payload["context"],
        },
    )
    response_payload["session_id"] = session.id
    response_payload["session"] = chat_session_payload(session)
    run_log_store.record_generation(
        mode=mode,
        prompt=prompt,
        response_payload=response_payload,
        context_payload=response_payload["context"],
        session_id=session.id,
        latency_ms=_elapsed_ms(started_at),
    )
    return response_payload


def build_chat_model_context(
    session: ChatSession | None,
    prompt: str,
    context_policy: object,
    *,
    memory_store: FileMemoryStore,
) -> dict[str, Any]:
    memories = _memory_snippets(memory_store, prompt, limit=context_policy.max_memory_items)
    if session is None or not session.messages:
        bundle = build_context_bundle(
            system_prompt="",
            current_prompt=prompt,
            turns=(),
            memories=memories,
            policy=context_policy,
        )
        model_prompt = bundle.as_prompt() if memories else prompt
        return {"prompt": model_prompt, "payload": context_payload(bundle, memories=memories)}

    bundle = build_context_bundle(
        system_prompt=(
            "You are continuing a local chat session. "
            "Use prior messages only as context for the current user message."
        ),
        current_prompt=f"Current user message:\n{prompt}",
        turns=_conversation_turns(session),
        summary=session.summary,
        memories=memories,
        policy=context_policy,
    )
    return {"prompt": bundle.as_prompt(), "payload": context_payload(bundle, memories=memories)}


def chat_response_payload(response: object) -> dict[str, Any]:
    return {
        "content": response.content,
        "correlation_id": response.correlation_id,
        "route": route_payload(response.route),
        "results": [item.__dict__ for item in response.results],
        "errors": list(response.errors),
        "disagreement": asdict(response.disagreement) if response.disagreement else None,
        "context": {},
    }


def route_payload(route: object) -> dict[str, Any]:
    return {
        "selected": [item.__dict__ for item in route.selected],
        "fallback_order": list(route.fallback_order),
    }


def context_payload(
    bundle: ContextBundle,
    *,
    memories: tuple[MemorySnippet, ...] = (),
) -> dict[str, Any]:
    return {
        "token_estimate": bundle.token_estimate,
        "budget_tokens": bundle.budget_tokens,
        "compaction_needed": bundle.compaction_needed,
        "dropped_turns": bundle.dropped_turns,
        "sections": bundle.by_section(),
        "memory_ids": [memory.id for memory in memories],
    }


def _session_for_id(chat_store: FileChatStore, session_id: str | None) -> ChatSession | None:
    if not session_id:
        return None
    session = chat_store.get_session(session_id)
    if session is None:
        raise KeyError(session_id)
    return session


def _memory_snippets(
    memory_store: FileMemoryStore,
    prompt: str,
    *,
    limit: int,
) -> tuple[MemorySnippet, ...]:
    return tuple(
        MemorySnippet(id=record.id, text=record.text, score=score)
        for record, score in memory_store.search(prompt, scope="default", limit=limit)
    )


def _conversation_turns(session: ChatSession) -> tuple[ConversationTurn, ...]:
    return tuple(
        ConversationTurn(role=message.role, content=message.content)
        for message in session.messages
        if message.role in {"user", "assistant"}
    )


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)
