from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import re
from typing import Iterator
from uuid import uuid4

from .config import MoEConfig
from .execution_scope import ExecutionScopeGuard
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
class MoEStreamEvent:
    kind: str
    content: str = ""
    response: MoEResponse | None = None
    route: RouteDecision | None = None
    error: str = ""


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
    def __init__(
        self,
        config: MoEConfig,
        *,
        execution_guard: ExecutionScopeGuard | None = None,
    ):
        self._config = config
        self._execution_guard = execution_guard or ExecutionScopeGuard(
            config.execution_policy
        )
        self._router = RuleRouter(
            config,
            execution_guard=self._execution_guard,
        )
        self._providers = {
            expert.id: build_provider(expert.provider) for expert in config.experts
        }

    def generate(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        route_prompt: str | None = None,
    ) -> MoEResponse:
        cid = correlation_id or str(uuid4())
        route = self._router.route(route_prompt or prompt)
        req = GenerationRequest(prompt=prompt, correlation_id=cid)

        selected_order = [score.expert_id for score in route.selected]
        fallback_order = []
        for fallback in route.fallback_order:
            if fallback not in selected_order and fallback not in fallback_order:
                fallback_order.append(fallback)

        results: list[ExpertResult] = []
        errors: list[str] = []

        if self._config.routing.aggregation in {"concat", "compare"}:
            results, errors = self._generate_many(selected_order, req)
            missing = max(0, len(selected_order) - len(results))
            if missing:
                fallback_results, fallback_errors = self._generate_fallbacks(
                    fallback_order,
                    req,
                    limit=missing,
                )
                results.extend(fallback_results)
                errors.extend(fallback_errors)
        else:
            expert_order = [*selected_order, *fallback_order]
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

    def generate_stream(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        route_prompt: str | None = None,
    ) -> Iterator[MoEStreamEvent]:
        cid = correlation_id or str(uuid4())
        route = self._router.route(route_prompt or prompt)
        yield MoEStreamEvent(kind="route", route=route)

        if self._config.routing.aggregation in {"concat", "compare"}:
            response = self.generate(prompt, correlation_id=cid, route_prompt=route_prompt)
            yield MoEStreamEvent(kind="content", content=response.content)
            yield MoEStreamEvent(kind="final", content=response.content, response=response)
            return

        req = GenerationRequest(prompt=prompt, correlation_id=cid)
        expert_order = [score.expert_id for score in route.selected]
        for fallback in route.fallback_order:
            if fallback not in expert_order:
                expert_order.append(fallback)

        errors: list[str] = []
        experts_by_id = self._config.experts_by_id
        for expert_id in expert_order:
            expert = experts_by_id[expert_id]
            provider = self._providers[expert_id]
            stream_generate = getattr(provider, "stream_generate", None)
            try:
                if callable(stream_generate):
                    self._execution_guard.require_allowed(expert.execution_target)
                    final_result = None
                    for event in stream_generate(expert, req):
                        if event.result is not None:
                            final_result = event.result
                            continue
                        yield MoEStreamEvent(kind="content", content=event.content)
                    if final_result is None:
                        raise ProviderError(f"Expert {expert_id} stream ended without a final result.")
                    response = MoEResponse(
                        content=final_result.content,
                        correlation_id=cid,
                        route=route,
                        results=(final_result,),
                        errors=tuple(errors),
                    )
                    yield MoEStreamEvent(kind="final", content=response.content, response=response)
                    return

                result = self._invoke_provider(expert_id, req)
                response = MoEResponse(
                    content=result.content,
                    correlation_id=cid,
                    route=route,
                    results=(result,),
                    errors=tuple(errors),
                )
                yield MoEStreamEvent(kind="content", content=response.content)
                yield MoEStreamEvent(kind="final", content=response.content, response=response)
                return
            except ProviderError as exc:
                errors.append(f"{expert_id}: {exc}")
                continue

        raise ProviderError("; ".join(errors) or "No expert produced a stream result.")

    def _generate_best(
        self, expert_order: list[str], req: GenerationRequest
    ) -> tuple[list[ExpertResult], list[str]]:
        results: list[ExpertResult] = []
        errors: list[str] = []

        for expert_id in expert_order:
            try:
                results.append(self._invoke_provider(expert_id, req))
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

        with ThreadPoolExecutor(max_workers=max(len(expert_order), 1)) as pool:
            futures = {}
            for expert_id in expert_order:
                futures[pool.submit(self._invoke_provider, expert_id, req)] = expert_id

            for future in as_completed(futures):
                expert_id = futures[future]
                try:
                    results.append(future.result())
                except ProviderError as exc:
                    errors.append(f"{expert_id}: {exc}")

        order = {expert_id: index for index, expert_id in enumerate(expert_order)}
        results.sort(key=lambda item: order[item.expert_id])
        return results, errors

    def _generate_fallbacks(
        self,
        expert_order: list[str],
        req: GenerationRequest,
        *,
        limit: int,
    ) -> tuple[list[ExpertResult], list[str]]:
        results: list[ExpertResult] = []
        errors: list[str] = []

        for expert_id in expert_order:
            try:
                results.append(self._invoke_provider(expert_id, req))
            except ProviderError as exc:
                errors.append(f"{expert_id}: {exc}")
                continue
            if len(results) >= limit:
                break

        return results, errors

    def _invoke_provider(
        self,
        expert_id: str,
        req: GenerationRequest,
    ) -> ExpertResult:
        expert = self._config.experts_by_id[expert_id]
        self._execution_guard.require_allowed(expert.execution_target)
        return self._providers[expert_id].generate(expert, req)

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
