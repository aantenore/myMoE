from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .config import ExpertConfig, MoEConfig
from .context import ConversationTurn, build_compaction_prompt
from .providers import GenerationRequest, build_provider


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    expert_id: str
    model: str
    correlation_id: str


class LocalCompactionProvider:
    def __init__(self, config: MoEConfig, expert_id: str | None = None):
        self._config = config
        self._expert = _select_compaction_expert(config, expert_id)
        self._provider = build_provider(self._expert.provider)

    @property
    def expert_id(self) -> str:
        return self._expert.id

    def compact(
        self,
        *,
        turns: tuple[ConversationTurn, ...],
        existing_summary: str = "",
        correlation_id: str | None = None,
    ) -> CompactionResult:
        cid = correlation_id or str(uuid4())
        prompt = build_compaction_prompt(turns=turns, existing_summary=existing_summary)
        result = self._provider.generate(self._expert, GenerationRequest(prompt=prompt, correlation_id=cid))
        return CompactionResult(
            summary=result.content,
            expert_id=result.expert_id,
            model=result.model,
            correlation_id=result.correlation_id,
        )


def _select_compaction_expert(config: MoEConfig, expert_id: str | None) -> ExpertConfig:
    if expert_id:
        return config.experts_by_id[expert_id]

    role_markers = ("compaction", "summary", "summarization", "fallback")
    fallback_ids = set(config.routing.fallback_order)
    ranked = sorted(
        config.experts,
        key=lambda expert: (
            not _is_compaction_candidate(expert, role_markers, fallback_ids),
            expert.weight,
            expert.id,
        ),
    )
    return ranked[0]


def _is_compaction_candidate(
    expert: ExpertConfig,
    role_markers: tuple[str, ...],
    fallback_ids: set[str],
) -> bool:
    role = expert.role.lower()
    return expert.id in fallback_ids or any(marker in role for marker in role_markers)
