from __future__ import annotations

import json
import math
from typing import Any, Callable, Sequence
from urllib import error, request

from .agent_types import (
    AgentMessage,
    AgentModelOutput,
    AgentModelUsage,
    AgentToolCall,
    AgentToolSpec,
)
from .config import ExpertConfig, MoEConfig
from .execution_scope import ExecutionScopeGuard
from .http_boundary import is_loopback_http_url, open_model_endpoint
from .providers import ProviderError, _remote_params, strip_reasoning_content


MappingPayload = dict[str, Any]
AgentHttpTransport = Callable[[request.Request, float], MappingPayload]
MAX_AGENT_RESPONSE_BYTES = 4_000_000


class OpenAICompatibleAgentAdapter:
    """Chat-Completions tool-call adapter for local OpenAI-compatible servers."""

    def __init__(
        self,
        expert: ExpertConfig,
        *,
        transport: AgentHttpTransport | None = None,
        parallel_tool_calls: bool | None = False,
        execution_guard: ExecutionScopeGuard | None = None,
    ):
        if expert.provider != "openai_compatible":
            raise ProviderError(
                f"Agent adapter requires provider=openai_compatible, got {expert.provider}."
            )
        if not expert.base_url:
            raise ProviderError(f"Expert {expert.id} has no base_url.")
        self._expert = expert
        self._transport = transport or _default_transport
        self._parallel_tool_calls = parallel_tool_calls
        self._execution_guard = execution_guard or ExecutionScopeGuard()

    @property
    def expert(self) -> ExpertConfig:
        return self._expert

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentToolSpec],
        *,
        correlation_id: str,
        timeout_seconds: float | None = None,
    ) -> AgentModelOutput:
        payload = self.request_payload(messages, tools)
        url = str(self._expert.base_url).rstrip("/") + "/chat/completions"
        http_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._execution_guard.require_allowed(self._expert.execution_target)
        try:
            parsed = self._transport(
                http_request,
                _effective_timeout_seconds(
                    self._expert.timeout_seconds,
                    timeout_seconds,
                ),
            )
        except (OSError, error.URLError) as exc:
            raise ProviderError(
                f"Expert {self._expert.id} agent request failed: {exc}"
            ) from exc
        return parse_openai_compatible_output(
            parsed,
            model=self._expert.model,
            correlation_id=correlation_id,
        )

    def request_payload(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentToolSpec],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._expert.model,
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [
                spec.openai_definition()
                for spec in sorted(tools, key=lambda item: item.exposed_name)
            ]
            payload["tool_choice"] = "auto"
            if self._parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = self._parallel_tool_calls

        last_user_prompt = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )
        remote_params = _remote_params(self._expert, last_user_prompt)
        for reserved in (
            "model",
            "messages",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
        ):
            remote_params.pop(reserved, None)
        payload.update(remote_params)
        return payload


def build_openai_compatible_agent_adapter(
    config: MoEConfig,
    *,
    expert_id: str | None = None,
    transport: AgentHttpTransport | None = None,
    execution_guard: ExecutionScopeGuard | None = None,
) -> OpenAICompatibleAgentAdapter:
    expert = select_agent_expert(config, expert_id=expert_id)
    return OpenAICompatibleAgentAdapter(
        expert,
        transport=transport,
        execution_guard=execution_guard
        or ExecutionScopeGuard(config.execution_policy),
    )


def select_agent_expert(
    config: MoEConfig, *, expert_id: str | None = None
) -> ExpertConfig:
    if expert_id:
        expert = config.experts_by_id.get(expert_id)
        if expert is None:
            raise ProviderError(f"Unknown agent expert: {expert_id}")
        if expert.provider != "openai_compatible":
            raise ProviderError(f"Agent expert {expert_id} is not OpenAI-compatible.")
        return expert

    candidates = [
        expert for expert in config.experts if expert.provider == "openai_compatible"
    ]
    if not candidates:
        raise ProviderError(
            "No OpenAI-compatible expert is configured for the agent loop."
        )
    return next(
        (expert for expert in candidates if expert.role.strip().lower() == "general"),
        candidates[0],
    )


def validate_local_agent_endpoints(config: MoEConfig) -> None:
    """Reject agent experts that cannot satisfy the configured execution policy."""

    guard = ExecutionScopeGuard(config.execution_policy)
    blocked = sorted(
        expert.id
        for expert in config.experts
        if expert.provider == "openai_compatible"
        and not guard.evaluate(expert.execution_target).allowed
    )
    if blocked:
        rendered = ", ".join(blocked)
        raise ProviderError(
            "Agent mode requires execution-policy-compliant model endpoints when "
            "the app mode is local_model_required; blocked expert(s): "
            f"{rendered}."
        )


def is_loopback_base_url(base_url: str) -> bool:
    return is_loopback_http_url(base_url)


def parse_openai_compatible_output(
    payload: object,
    *,
    model: str,
    correlation_id: str,
) -> AgentModelOutput:
    if not isinstance(payload, dict):
        raise ProviderError("Agent provider returned a non-object payload.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderError("Agent provider returned an invalid choices payload.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProviderError("Agent provider returned an invalid message payload.")

    tool_calls: list[AgentToolCall] = []
    raw_calls = message.get("tool_calls", [])
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ProviderError("Agent provider returned invalid tool_calls.")
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise ProviderError("Agent provider returned an invalid tool call.")
        function = raw_call.get("function")
        if not isinstance(function, dict) or not str(function.get("name", "")).strip():
            raise ProviderError(
                "Agent provider returned a tool call without a function name."
            )
        raw_arguments = function.get("arguments", {})
        arguments: object
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(
                    raw_arguments,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError):
                # Preserve malformed arguments as data. The local schema
                # validator will produce the corresponding structured result.
                arguments = raw_arguments
        else:
            arguments = raw_arguments
        call_id = str(raw_call.get("id", "")).strip() or (
            f"{correlation_id}-tool-{index + 1}"
        )
        tool_calls.append(
            AgentToolCall(
                id=call_id,
                name=str(function["name"]).strip(),
                arguments=arguments,
            )
        )

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ProviderError("Agent provider returned non-text assistant content.")
    visible_content = strip_reasoning_content(content or "")
    final_answer = None
    if not tool_calls and content is not None:
        final_answer = visible_content

    usage = payload.get("usage")
    return AgentModelOutput(
        final_answer=final_answer,
        tool_calls=tuple(tool_calls),
        assistant_content=visible_content if tool_calls else "",
        model=model,
        usage=AgentModelUsage(
            prompt_tokens=_optional_int(usage, "prompt_tokens"),
            completion_tokens=_optional_int(usage, "completion_tokens"),
        ),
    )


def _message_payload(message: AgentMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _argument_text(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }
    if message.role == "tool":
        if not message.tool_call_id:
            raise ProviderError("Tool messages require a matching tool_call_id.")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.role not in {"system", "user", "assistant"}:
        raise ProviderError(f"Unsupported agent message role: {message.role}")
    return {"role": message.role, "content": message.content}


def _argument_text(arguments: object) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(
        arguments, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def _default_transport(
    http_request: request.Request, timeout_seconds: float
) -> MappingPayload:
    with open_model_endpoint(http_request, timeout=timeout_seconds) as response:
        raw_bytes = response.read(MAX_AGENT_RESPONSE_BYTES + 1)
    if len(raw_bytes) > MAX_AGENT_RESPONSE_BYTES:
        raise ProviderError("Agent provider response exceeded the size limit.")
    raw = raw_bytes.decode("utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Agent provider returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("Agent provider returned a non-object JSON payload.")
    return parsed


def _optional_int(value: object, key: str) -> int | None:
    if not isinstance(value, dict) or value.get(key) is None:
        return None
    try:
        return int(value[key])
    except (TypeError, ValueError):
        return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _effective_timeout_seconds(
    configured_timeout: float,
    remaining_timeout: float | None,
) -> float:
    try:
        configured = float(configured_timeout)
    except (TypeError, ValueError) as exc:
        raise ProviderError("Agent expert timeout_seconds must be numeric.") from exc
    if not math.isfinite(configured) or configured <= 0:
        raise ProviderError("Agent expert timeout_seconds must be finite and positive.")
    if remaining_timeout is None:
        return configured
    try:
        remaining = float(remaining_timeout)
    except (TypeError, ValueError) as exc:
        raise ProviderError("Agent remaining timeout must be numeric.") from exc
    if not math.isfinite(remaining) or remaining <= 0:
        raise ProviderError("Agent remaining timeout must be finite and positive.")
    return min(configured, remaining)
