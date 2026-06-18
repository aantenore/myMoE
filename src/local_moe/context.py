from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ContextSection(str, Enum):
    SYSTEM = "system"
    MEMORY = "memory"
    SUMMARY = "summary"
    RECENT_TURNS = "recent_turns"
    CURRENT_PROMPT = "current_prompt"


@dataclass(frozen=True)
class ContextPolicy:
    context_limit_tokens: int
    reserved_output_tokens: int = 1024
    compaction_trigger_ratio: float = 0.75
    max_recent_turns: int = 12
    max_memory_items: int = 8

    @property
    def input_budget_tokens(self) -> int:
        return max(self.context_limit_tokens - self.reserved_output_tokens, 1)

    @property
    def compaction_trigger_tokens(self) -> int:
        return int(self.context_limit_tokens * self.compaction_trigger_ratio)


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True)
class MemorySnippet:
    id: str
    text: str
    score: float = 1.0


@dataclass(frozen=True)
class ContextPart:
    section: ContextSection
    content: str
    token_estimate: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBundle:
    parts: tuple[ContextPart, ...]
    token_estimate: int
    budget_tokens: int
    compaction_needed: bool
    dropped_turns: int

    def as_prompt(self) -> str:
        return "\n\n".join(part.content for part in self.parts if part.content.strip())

    def by_section(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for part in self.parts:
            totals[part.section.value] = totals.get(part.section.value, 0) + part.token_estimate
        return totals


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def build_context_bundle(
    *,
    system_prompt: str,
    current_prompt: str,
    policy: ContextPolicy,
    turns: Iterable[ConversationTurn] = (),
    memories: Iterable[MemorySnippet] = (),
    summary: str = "",
) -> ContextBundle:
    memory_items = tuple(sorted(memories, key=lambda item: (-item.score, item.id)))[
        : policy.max_memory_items
    ]
    recent_turns = tuple(turns)[-policy.max_recent_turns :]

    stable_parts = [
        _part(ContextSection.SYSTEM, system_prompt),
        _part(ContextSection.MEMORY, _format_memories(memory_items)),
        _part(ContextSection.SUMMARY, summary),
    ]
    current_part = _part(ContextSection.CURRENT_PROMPT, current_prompt)

    fixed_tokens = sum(part.token_estimate for part in stable_parts) + current_part.token_estimate
    remaining = max(policy.input_budget_tokens - fixed_tokens, 0)
    selected_turns = _select_recent_turns(recent_turns, remaining)
    dropped_turns = len(recent_turns) - len(selected_turns)

    parts = tuple(
        part
        for part in [
            *stable_parts,
            _part(ContextSection.RECENT_TURNS, _format_turns(selected_turns)),
            current_part,
        ]
        if part.content.strip()
    )
    total = sum(part.token_estimate for part in parts)

    return ContextBundle(
        parts=parts,
        token_estimate=total,
        budget_tokens=policy.input_budget_tokens,
        compaction_needed=total >= policy.compaction_trigger_tokens or dropped_turns > 0,
        dropped_turns=dropped_turns,
    )


def build_compaction_prompt(
    *,
    turns: Iterable[ConversationTurn],
    existing_summary: str = "",
) -> str:
    transcript = _format_turns(tuple(turns))
    return (
        "Summarize this local-agent session for future continuation.\n"
        "Preserve exact file paths, model ids, decisions, risks, failed attempts, "
        "test status, and next actions. Merge with the existing summary without "
        "dropping still-relevant facts.\n\n"
        "Existing summary:\n"
        f"{existing_summary or '(none)'}\n\n"
        "New transcript slice:\n"
        f"{transcript}"
    )


def _select_recent_turns(
    turns: tuple[ConversationTurn, ...], budget_tokens: int
) -> tuple[ConversationTurn, ...]:
    selected: list[ConversationTurn] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn.content) + estimate_tokens(turn.role) + 4
        if selected and used + cost > budget_tokens:
            break
        if not selected and cost > budget_tokens:
            break
        selected.append(turn)
        used += cost
    selected.reverse()
    return tuple(selected)


def _part(section: ContextSection, content: str) -> ContextPart:
    return ContextPart(section=section, content=content, token_estimate=estimate_tokens(content))


def _format_memories(memories: tuple[MemorySnippet, ...]) -> str:
    if not memories:
        return ""
    lines = ["Relevant memory snippets:"]
    for memory in memories:
        lines.append(f"- [{memory.id}] {memory.text}")
    return "\n".join(lines)


def _format_turns(turns: tuple[ConversationTurn, ...]) -> str:
    if not turns:
        return ""
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
