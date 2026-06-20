from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import time
from typing import Any

from .config import MoEConfig
from .orchestrator import LocalMoE
from .providers import ProviderError

DEFAULT_SMOKE_PROMPT = "Reply in one short English sentence: myMoE local generation smoke test passed."


def build_generation_smoke_report(
    config: MoEConfig,
    *,
    prompt: str = DEFAULT_SMOKE_PROMPT,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    prompt = prompt.strip() or DEFAULT_SMOKE_PROMPT
    try:
        response = LocalMoE(config).generate(
            prompt,
            correlation_id=correlation_id,
            route_prompt=prompt,
        )
    except ProviderError as exc:
        return {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "status": "fail",
            "latency_ms": _elapsed_ms(started),
            "prompt": prompt,
            "error": str(exc),
            "route": None,
            "results": [],
            "content": "",
            "content_chars": 0,
            "recommendations": [
                "Start configured local model servers and rerun the generation smoke test.",
                "Inspect System Doctor and model logs if endpoints are reachable but generation fails.",
            ],
        }

    content = response.content.strip()
    status = "pass" if content else "fail"
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "status": status,
        "latency_ms": _elapsed_ms(started),
        "prompt": prompt,
        "correlation_id": response.correlation_id,
        "route": {
            "selected": [item.__dict__ for item in response.route.selected],
            "fallback_order": list(response.route.fallback_order),
        },
        "results": [_result_payload(item) for item in response.results],
        "errors": list(response.errors),
        "disagreement": asdict(response.disagreement) if response.disagreement else None,
        "content": content,
        "content_chars": len(content),
        "recommendations": _recommendations(status),
    }


def _result_payload(result: object) -> dict[str, Any]:
    return {
        "expert_id": result.expert_id,
        "model": result.model,
        "content_chars": len(str(result.content).strip()),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "predicted_tokens_per_second": result.predicted_tokens_per_second,
    }


def _recommendations(status: str) -> list[str]:
    if status == "pass":
        return ["Generation smoke test passed; the selected local expert returned visible content."]
    return [
        "The selected expert returned no visible content.",
        "Inspect provider response parsing, reasoning-channel stripping, and model log tails.",
    ]


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
