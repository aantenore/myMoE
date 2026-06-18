from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        content = self._aggregate(results)
        return MoEResponse(
            content=content,
            correlation_id=cid,
            route=route,
            results=tuple(results),
            errors=tuple(errors),
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

    def _aggregate(self, results: list[ExpertResult]) -> str:
        if len(results) == 1:
            return results[0].content

        sections = [
            f"### {result.expert_id} ({result.model})\n{result.content}"
            for result in results
        ]
        return "\n\n".join(sections)
