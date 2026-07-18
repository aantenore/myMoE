from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterator, Protocol
from urllib import error, request

from .config import ExpertConfig
from .http_boundary import open_model_endpoint

LOCAL_PROVIDER_PARAMS = {
    "runtime_backend",
    "supports_thinking",
    "thinking_policy",
    "system_prompt",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a local expert in a system-level MoE. "
    "Return useful, direct answers. Respond in the user's language "
    "unless they explicitly ask for another language. Preserve "
    "correlation context."
)


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
    finish_reason: str | None = None


@dataclass(frozen=True)
class ProviderStreamEvent:
    content: str
    result: ExpertResult | None = None


class Provider(Protocol):
    def generate(self, expert: ExpertConfig, req: GenerationRequest) -> ExpertResult:
        ...


class SyntheticProvider:
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

    def stream_generate(
        self,
        expert: ExpertConfig,
        req: GenerationRequest,
    ) -> Iterator[ProviderStreamEvent]:
        result = self.generate(expert, req)
        yield ProviderStreamEvent(content=result.content)
        yield ProviderStreamEvent(content=result.content, result=result)


class OpenAICompatibleProvider:
    def __init__(self, *, opener: Callable[..., Any] | None = None):
        self._opener = opener or open_model_endpoint

    def generate(self, expert: ExpertConfig, req: GenerationRequest) -> ExpertResult:
        if not expert.base_url:
            raise ProviderError(f"Expert {expert.id} has no base_url.")

        http_req = _chat_request(expert, req)

        try:
            with self._opener(http_req, timeout=expert.timeout_seconds) as response:
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
            content=strip_reasoning_content(str(content)),
            correlation_id=req.correlation_id,
            prompt_tokens=_maybe_int(usage, "prompt_tokens"),
            completion_tokens=_maybe_int(usage, "completion_tokens"),
            predicted_tokens_per_second=_maybe_float(timings, "predicted_per_second"),
            finish_reason=_finish_reason(parsed),
        )

    def stream_generate(
        self,
        expert: ExpertConfig,
        req: GenerationRequest,
    ) -> Iterator[ProviderStreamEvent]:
        if not expert.base_url:
            raise ProviderError(f"Expert {expert.id} has no base_url.")

        http_req = _chat_request(expert, req, stream=True)
        raw_content = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        predicted_tokens_per_second: float | None = None
        finish_reason: str | None = None
        saw_sse = False
        non_sse_lines: list[str] = []

        try:
            with self._opener(http_req, timeout=expert.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        non_sse_lines.append(line)
                        continue
                    saw_sse = True
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Expert {expert.id} returned invalid stream JSON.") from exc

                    delta = _stream_delta(parsed)
                    if delta:
                        raw_content += delta
                        yield ProviderStreamEvent(content=strip_reasoning_content(raw_content))

                    usage = parsed.get("usage") if isinstance(parsed, dict) else None
                    timings = parsed.get("timings") if isinstance(parsed, dict) else None
                    prompt_tokens = _maybe_int(usage, "prompt_tokens") or prompt_tokens
                    completion_tokens = _maybe_int(usage, "completion_tokens") or completion_tokens
                    predicted_tokens_per_second = (
                        _maybe_float(timings, "predicted_per_second")
                        or predicted_tokens_per_second
                    )
                    finish_reason = _finish_reason(parsed) or finish_reason
        except error.URLError as exc:
            raise ProviderError(f"Expert {expert.id} stream request failed: {exc}") from exc

        if not saw_sse and non_sse_lines:
            result = _parse_non_streaming_result(expert, req, "\n".join(non_sse_lines))
            yield ProviderStreamEvent(content=result.content)
            yield ProviderStreamEvent(content=result.content, result=result)
            return

        result = ExpertResult(
            expert_id=expert.id,
            model=expert.model,
            content=strip_reasoning_content(raw_content),
            correlation_id=req.correlation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            predicted_tokens_per_second=predicted_tokens_per_second,
            finish_reason=finish_reason,
        )
        yield ProviderStreamEvent(content=result.content, result=result)


def build_provider(provider_name: str) -> Provider:
    if provider_name == "synthetic":
        return SyntheticProvider()
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider()
    raise ProviderError(f"Unsupported provider type: {provider_name}")


def _chat_request(
    expert: ExpertConfig,
    req: GenerationRequest,
    *,
    stream: bool = False,
) -> request.Request:
    url = str(expert.base_url).rstrip("/") + "/chat/completions"
    payload = {
        "model": expert.model,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(expert),
            },
            {"role": "user", "content": req.prompt},
        ],
    }
    payload.update(_remote_params(expert, req.prompt))
    if stream:
        payload["stream"] = True

    return request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _parse_non_streaming_result(
    expert: ExpertConfig,
    req: GenerationRequest,
    raw: str,
) -> ExpertResult:
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
        content=strip_reasoning_content(str(content)),
        correlation_id=req.correlation_id,
        prompt_tokens=_maybe_int(usage, "prompt_tokens"),
        completion_tokens=_maybe_int(usage, "completion_tokens"),
        predicted_tokens_per_second=_maybe_float(timings, "predicted_per_second"),
        finish_reason=_finish_reason(parsed),
    )


def _stream_delta(parsed: object) -> str:
    if not isinstance(parsed, dict):
        return ""
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta["content"])
    message = first.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    text = first.get("text")
    return str(text) if text is not None else ""


def _finish_reason(parsed: object) -> str | None:
    if not isinstance(parsed, dict):
        return None
    choices = parsed.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        return None
    raw = choices[0].get("finish_reason")
    if raw is None:
        return None
    value = str(raw).strip().lower()
    return value or None


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


def _remote_params(expert: ExpertConfig, prompt: str) -> dict[str, object]:
    params = {
        key: value
        for key, value in expert.params.items()
        if key not in LOCAL_PROVIDER_PARAMS
    }
    if not _supports_thinking(expert.params):
        return params

    policy = str(expert.params.get("thinking_policy", "off")).strip().lower()
    if policy in {"on", "always", "true", "1"}:
        enable_thinking = True
    elif policy == "auto":
        enable_thinking = _prompt_needs_thinking(prompt)
    else:
        enable_thinking = False

    chat_template_kwargs = dict(params.get("chat_template_kwargs", {}))
    chat_template_kwargs["enable_thinking"] = enable_thinking
    params["chat_template_kwargs"] = chat_template_kwargs
    return params


def _system_prompt(expert: ExpertConfig) -> str:
    raw = expert.params.get("system_prompt")
    if raw is None:
        return DEFAULT_SYSTEM_PROMPT
    if not isinstance(raw, str) or not raw.strip():
        raise ProviderError(
            f"Expert {expert.id} system_prompt must be a non-empty string."
        )
    return raw.strip()


def _supports_thinking(params: dict[str, object]) -> bool:
    raw = params.get("supports_thinking", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _prompt_needs_thinking(prompt: str) -> bool:
    normalized = prompt.lower()
    # Auto mode is intentionally conservative: broad words such as "plan",
    # "architecture", or "compare" made small local models spend minutes in
    # hidden reasoning on otherwise routine interactive work. Operators can
    # still select thinking_policy="on" for dedicated reasoning profiles.
    hard_markers = (
        "security",
        "threat",
        "formal proof",
        "prove that",
        "security review",
        "threat model",
    )
    simple_markers = (
        "summarize",
        "translate",
        "rewrite",
        "shorten",
        "format",
        "fix grammar",
    )
    if any(marker in normalized for marker in simple_markers):
        return False
    if any(marker in normalized for marker in hard_markers):
        return True
    return False


def strip_reasoning_content(content: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", content)
    cleaned = re.sub(r"(?is)<think>.*$", "", cleaned)
    cleaned = re.sub(
        r"(?is)<\|channel\|>analysis.*?(?=<\|channel\|>final|$)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?is)<\|channel>\s*(?:analysis|thought)\b.*?<channel\|>",
        "",
        cleaned,
    )
    if "<|channel|>final" in cleaned:
        cleaned = cleaned.split("<|channel|>final", maxsplit=1)[-1]
    if "<|channel>final" in cleaned:
        cleaned = cleaned.split("<|channel>final", maxsplit=1)[-1]
    cleaned = re.sub(r"(?is)<\|[^>]+?\|>", "", cleaned)
    cleaned = re.sub(r"(?is)<[^>]*\|>", "", cleaned)
    cleaned = re.sub(r"(?is)</?(?:start_of_turn|end_of_turn|eos)>", "", cleaned)
    return cleaned.strip()
