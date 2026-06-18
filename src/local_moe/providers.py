from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol
from urllib import request, error

from .config import ExpertConfig


class ProviderError(RuntimeError):
    """Raised when an expert provider fails."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    correlation_id: str


@dataclass(frozen=True)
class ExpertResult:
    expert_id: str
    model: str
    content: str
    correlation_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    predicted_tokens_per_second: float | None = None


class Provider(Protocol):
    def generate(self, expert: ExpertConfig, req: GenerationRequest) -> ExpertResult:
        ...


class MockProvider:
    def generate(self, expert: ExpertConfig, req: GenerationRequest) -> ExpertResult:
        content = (
            f"[{expert.role}:{expert.model}] handled correlation_id="
            f"{req.correlation_id}; prompt_chars={len(req.prompt)}"
        )
        return ExpertResult(
            expert_id=expert.id,
            model=expert.model,
            content=content,
            correlation_id=req.correlation_id,
            prompt_tokens=None,
            completion_tokens=None,
            predicted_tokens_per_second=None,
        )


class OpenAICompatibleProvider:
    def generate(self, expert: ExpertConfig, req: GenerationRequest) -> ExpertResult:
        if not expert.base_url:
            raise ProviderError(f"Expert {expert.id} has no base_url.")

        url = expert.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": expert.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local expert in a system-level MoE. "
                        "Return useful, direct answers. Reply in the user's language "
                        "unless they explicitly ask for another language. Preserve "
                        "correlation context."
                    ),
                },
                {"role": "user", "content": req.prompt},
            ],
        }
        payload.update(expert.params)

        data = json.dumps(payload).encode("utf-8")
        http_req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_req, timeout=expert.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise ProviderError(f"Expert {expert.id} request failed: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Expert {expert.id} returned invalid JSON.") from exc

        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Expert {expert.id} returned invalid payload.") from exc

        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        timings = parsed.get("timings") if isinstance(parsed, dict) else None

        return ExpertResult(
            expert_id=expert.id,
            model=expert.model,
            content=str(content),
            correlation_id=req.correlation_id,
            prompt_tokens=_maybe_int(usage, "prompt_tokens"),
            completion_tokens=_maybe_int(usage, "completion_tokens"),
            predicted_tokens_per_second=_maybe_float(timings, "predicted_per_second"),
        )


def build_provider(provider_name: str) -> Provider:
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider()
    raise ProviderError(f"Unsupported provider type: {provider_name}")


def _maybe_int(raw: object, key: str) -> int | None:
    if not isinstance(raw, dict) or raw.get(key) is None:
        return None
    try:
        return int(raw[key])
    except (TypeError, ValueError):
        return None


def _maybe_float(raw: object, key: str) -> float | None:
    if not isinstance(raw, dict) or raw.get(key) is None:
        return None
    try:
        return float(raw[key])
    except (TypeError, ValueError):
        return None
