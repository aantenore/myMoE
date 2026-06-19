from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import re
from uuid import uuid4

from .config import MoEConfig
from .providers import (
    ExpertResult,
    GenerationRequest,
    ProviderError,
    build_provider,
)
from .router import RuleRouter, RouteDecision


@dataclass(frozen=True)
class MoEResponse:
    content: str
    correlation_id: str
    route: RouteDecision
    results: tuple[ExpertResult, ...]
    errors: tuple[str, ...]
    disagreement: DisagreementReport | None = None


@dataclass(frozen=True)
class PairwiseOverlap:
    left_expert_id: str
    right_expert_id: str
    lexical_overlap: float


@dataclass(frozen=True)
class DisagreementReport:
    status: str
    compared_experts: tuple[str, ...]
    minimum_lexical_overlap: float
    maximum_length_delta: float
    pairwise_overlaps: tuple[PairwiseOverlap, ...]
    unique_terms_by_expert: dict[str, tuple[str, ...]]


class LocalMoE:
    def __init__(self, config: MoEConfig):
        self._config = config
        self._router = RuleRouter(config)
        self._providers = {
            expert.id: build_provider(expert.provider) for expert in config.experts
        }

    def generate(self, prompt: str, correlation_id: str | None = None) -> MoEResponse:
        cid = correlation_id or str(uuid4())
        route = self._router.route(prompt)
        req = GenerationRequest(prompt=prompt, correlation_id=cid)

        expert_order = [score.expert_id for score in route.selected]
        for fallback in route.fallback_order:
            if fallback not in expert_order:
                expert_order.append(fallback)

        results: list[ExpertResult] = []
        errors: list[str] = []
        experts_by_id = self._config.experts_by_id

        if self._config.routing.aggregation in {"concat", "compare"}:
            results, errors = self._generate_many(expert_order, req)
        else:
            results, errors = self._generate_best(expert_order, req)

        if not results:
            raise ProviderError("; ".join(errors) or "No expert produced a result.")

        disagreement = self._build_disagreement_report(results)
        content = self._aggregate(results, disagreement)
        return MoEResponse(
            content=content,
            correlation_id=cid,
            route=route,
            results=tuple(results),
            errors=tuple(errors),
            disagreement=disagreement,
        )

    def _generate_best(
        self, expert_order: list[str], req: GenerationRequest
    ) -> tuple[list[ExpertResult], list[str]]:
        results: list[ExpertResult] = []
        errors: list[str] = []
        experts_by_id = self._config.experts_by_id

        for expert_id in expert_order:
            expert = experts_by_id[expert_id]
            provider = self._providers[expert_id]
            try:
                results.append(provider.generate(expert, req))
            except ProviderError as exc:
                errors.append(f"{expert_id}: {exc}")
                continue

            break

        return results, errors

    def _generate_many(
        self, expert_order: list[str], req: GenerationRequest
    ) -> tuple[list[ExpertResult], list[str]]:
        results: list[ExpertResult] = []
        errors: list[str] = []
        experts_by_id = self._config.experts_by_id

        with ThreadPoolExecutor(max_workers=max(len(expert_order), 1)) as pool:
            futures = {}
            for expert_id in expert_order:
                expert = experts_by_id[expert_id]
                provider = self._providers[expert_id]
                futures[pool.submit(provider.generate, expert, req)] = expert_id

            for future in as_completed(futures):
                expert_id = futures[future]
                try:
                    results.append(future.result())
                except ProviderError as exc:
                    errors.append(f"{expert_id}: {exc}")

        order = {expert_id: index for index, expert_id in enumerate(expert_order)}
        results.sort(key=lambda item: order[item.expert_id])
        return results, errors

    def _aggregate(self, results: list[ExpertResult], disagreement: DisagreementReport | None) -> str:
        if len(results) == 1:
            return results[0].content

        sections = [
            f"### {result.expert_id} ({result.model})\n{result.content}"
            for result in results
        ]
        if disagreement is not None:
            sections.insert(0, _format_disagreement_report(disagreement))
        return "\n\n".join(sections)

    def _build_disagreement_report(self, results: list[ExpertResult]) -> DisagreementReport | None:
        if self._config.routing.aggregation != "compare" or len(results) < 2:
            return None
        return build_disagreement_report(results)


def build_disagreement_report(results: list[ExpertResult]) -> DisagreementReport:
    token_sets = {result.expert_id: _content_terms(result.content) for result in results}
    lengths = {expert_id: len(tokens) for expert_id, tokens in token_sets.items()}
    pairwise: list[PairwiseOverlap] = []

    for left, right in combinations(results, 2):
        overlap = _jaccard(token_sets[left.expert_id], token_sets[right.expert_id])
        pairwise.append(
            PairwiseOverlap(
                left_expert_id=left.expert_id,
                right_expert_id=right.expert_id,
                lexical_overlap=round(overlap, 3),
            )
        )

    overlaps = [item.lexical_overlap for item in pairwise]
    minimum_overlap = min(overlaps) if overlaps else 1.0
    maximum_length_delta = _maximum_length_delta(tuple(lengths.values()))
    status = "review_recommended" if minimum_overlap < 0.2 or maximum_length_delta > 0.75 else "agreement_likely"
    unique_terms = _unique_terms(token_sets)
    return DisagreementReport(
        status=status,
        compared_experts=tuple(result.expert_id for result in results),
        minimum_lexical_overlap=round(minimum_overlap, 3),
        maximum_length_delta=round(maximum_length_delta, 3),
        pairwise_overlaps=tuple(pairwise),
        unique_terms_by_expert=unique_terms,
    )


def _format_disagreement_report(report: DisagreementReport) -> str:
    unique_lines = []
    for expert_id, terms in report.unique_terms_by_expert.items():
        rendered = ", ".join(terms) if terms else "none"
        unique_lines.append(f"- {expert_id} unique terms: {rendered}")

    return "\n".join(
        [
            "### Deterministic disagreement report",
            f"- Status: {report.status}",
            f"- Compared experts: {', '.join(report.compared_experts)}",
            f"- Minimum lexical overlap: {report.minimum_lexical_overlap:.3f}",
            f"- Maximum length delta: {report.maximum_length_delta:.3f}",
            *unique_lines,
        ]
    )


_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "has",
    "have",
    "into",
    "that",
    "the",
    "this",
    "with",
    "you",
}


def _content_terms(content: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(content.lower())
        if token not in _STOPWORDS and not token.isdigit()
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _maximum_length_delta(lengths: tuple[int, ...]) -> float:
    if not lengths:
        return 0.0
    longest = max(lengths)
    shortest = min(lengths)
    if longest == 0:
        return 0.0
    return (longest - shortest) / longest


def _unique_terms(token_sets: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for expert_id, tokens in token_sets.items():
        others: set[str] = set()
        for other_id, other_tokens in token_sets.items():
            if other_id != expert_id:
                others.update(other_tokens)
        result[expert_id] = tuple(sorted(tokens - others)[:8])
    return result
