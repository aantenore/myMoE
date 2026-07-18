from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from typing import Any, Callable
from urllib import error
from urllib.parse import urlparse

from .config import ExpertConfig, MoEConfig
from .execution_scope import ExecutionScopeGuard
from .http_boundary import open_model_endpoint


@dataclass(frozen=True)
class ExpertHealth:
    expert_id: str
    provider: str
    model: str
    role: str
    base_url: str | None
    status: str
    latency_ms: float | None = None
    checked_url: str | None = None
    message: str = ""


@dataclass(frozen=True)
class RuntimeHealth:
    status: str
    checked_at: str
    experts: tuple[ExpertHealth, ...]


def check_runtime_health(
    config: MoEConfig,
    *,
    timeout_seconds: float = 1.5,
    opener: Callable[..., Any] = open_model_endpoint,
    execution_guard: ExecutionScopeGuard | None = None,
) -> RuntimeHealth:
    guard = execution_guard or ExecutionScopeGuard(config.execution_policy)
    experts = tuple(
        _check_expert(
            expert,
            timeout_seconds=timeout_seconds,
            opener=opener,
            execution_guard=guard,
        )
        for expert in config.experts
    )
    required = [expert for expert in experts if expert.provider == "openai_compatible"]
    status = "ready" if all(item.status == "ok" for item in required) else "degraded"
    if not required:
        status = "ready"
    return RuntimeHealth(status=status, checked_at=_now_iso(), experts=experts)


def runtime_health_payload(health: RuntimeHealth) -> dict[str, Any]:
    return {
        "status": health.status,
        "checked_at": health.checked_at,
        "experts": [
            {
                "expert_id": expert.expert_id,
                "provider": expert.provider,
                "model": expert.model,
                "role": expert.role,
                "base_url": expert.base_url,
                "status": expert.status,
                "latency_ms": expert.latency_ms,
                "checked_url": expert.checked_url,
                "message": expert.message,
            }
            for expert in health.experts
        ],
    }


def _check_expert(
    expert: ExpertConfig,
    *,
    timeout_seconds: float,
    opener: Callable[..., Any],
    execution_guard: ExecutionScopeGuard,
) -> ExpertHealth:
    if expert.provider != "openai_compatible":
        return ExpertHealth(
            expert_id=expert.id,
            provider=expert.provider,
            model=expert.model,
            role=expert.role,
            base_url=expert.base_url,
            status="skipped",
            message="Provider does not require a local HTTP endpoint.",
        )
    if not expert.base_url:
        return ExpertHealth(
            expert_id=expert.id,
            provider=expert.provider,
            model=expert.model,
            role=expert.role,
            base_url=None,
            status="missing_base_url",
            message="OpenAI-compatible expert has no base_url.",
        )

    probe_urls = _probe_urls(expert.base_url)
    if not probe_urls:
        return ExpertHealth(
            expert_id=expert.id,
            provider=expert.provider,
            model=expert.model,
            role=expert.role,
            base_url=expert.base_url,
            status="malformed_base_url",
            message="OpenAI-compatible expert base_url must include scheme and host.",
        )

    eligibility = execution_guard.evaluate(expert.execution_target)
    if not eligibility.allowed:
        return ExpertHealth(
            expert_id=expert.id,
            provider=expert.provider,
            model=expert.model,
            role=expert.role,
            base_url=expert.base_url,
            status="scope_blocked",
            message=eligibility.detail or "Endpoint is outside the execution policy.",
        )

    last_error = "Endpoint did not respond."
    for url in probe_urls:
        started = time.perf_counter()
        try:
            with opener(url, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            message = _health_message(body)
            return ExpertHealth(
                expert_id=expert.id,
                provider=expert.provider,
                model=expert.model,
                role=expert.role,
                base_url=expert.base_url,
                status="ok",
                latency_ms=latency_ms,
                checked_url=url,
                message=message,
            )
        except (OSError, error.URLError) as exc:
            last_error = str(exc)
            continue

    return ExpertHealth(
        expert_id=expert.id,
        provider=expert.provider,
        model=expert.model,
        role=expert.role,
        base_url=expert.base_url,
        status="unreachable",
        message=last_error,
    )


def _probe_urls(base_url: str) -> tuple[str, ...]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return ()

    base = base_url.rstrip("/")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        models_url = base + "/models"
        health_base = base[: -len("/v1")] or f"{parsed.scheme}://{parsed.netloc}"
    else:
        models_url = base + "/v1/models"
        health_base = base

    origin = f"{parsed.scheme}://{parsed.netloc}"
    urls = [models_url, health_base.rstrip("/") + "/health", origin + "/health"]
    return tuple(dict.fromkeys(urls))


def _health_message(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return "Endpoint responded."
    if isinstance(parsed, dict) and parsed.get("status"):
        return f"Endpoint status: {parsed['status']}"
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        return f"Model endpoint responded with {len(parsed['data'])} models."
    return "Endpoint responded."


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
