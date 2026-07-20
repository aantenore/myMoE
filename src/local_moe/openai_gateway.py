from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from ipaddress import ip_address
import json
import os
import re
from typing import Any, Callable, Mapping
from urllib import request
from urllib.parse import urlsplit
from uuid import uuid4

from .app_config import GatewayPolicy
from .config import ExpertConfig, MoEConfig
from .execution_scope import ExecutionScopeGuard
from .http_boundary import open_model_endpoint
from .providers import provider_request_params
from .router import RuleRouter


GatewayOpener = Callable[..., Any]
MAX_GATEWAY_JSON_DEPTH = 64
_SAFE_CORRELATION_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z"
)
_CONTENT_URL_FIELDS = (
    "audio_url",
    "file_url",
    "image_url",
    "video_url",
)
_URL_CONTENT_TYPES = {
    "audio_url",
    "file_url",
    "image_url",
    "input_audio",
    "input_file",
    "input_image",
    "input_video",
    "video_url",
}


class GatewayRequestError(ValueError):
    """A caller-visible OpenAI-compatible request error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        parameter: str | None = None,
        status: int = 400,
        response_started: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.parameter = parameter
        self.status = status
        self.response_started = response_started


@dataclass(frozen=True)
class GatewayAuthorization:
    allowed: bool
    code: str
    message: str


@dataclass(frozen=True)
class PreparedChatCompletion:
    requested_model: str
    expert: ExpertConfig
    upstream_request: request.Request
    correlation_id: str
    request_sha256: str
    routing_sha256: str
    stream: bool
    route_selected: tuple[str, ...]


class OpenAIGatewayService:
    """Local OpenAI-compatible inference gateway for editor agent harnesses."""

    def __init__(
        self,
        config: MoEConfig,
        policy: GatewayPolicy,
        *,
        opener: GatewayOpener | None = None,
    ) -> None:
        self._config = config
        self._policy = policy
        self._opener = opener or open_model_endpoint
        self._guard = ExecutionScopeGuard(config.execution_policy)
        self._router = RuleRouter(config, execution_guard=self._guard)
        self._experts = tuple(
            expert for expert in config.experts if expert.provider == "openai_compatible"
        )
        self._experts_by_id = {expert.id: expert for expert in self._experts}

    @property
    def policy(self) -> GatewayPolicy:
        return self._policy

    def authorize(
        self,
        client_host: str,
        authorization_header: str | None,
        *,
        host_header: str | None = None,
        origin_header: str | None = None,
    ) -> GatewayAuthorization:
        if not self._policy.enabled:
            return GatewayAuthorization(False, "gateway_disabled", "The local gateway is disabled.")

        is_loopback = is_loopback_host(client_host)
        if not is_loopback and not self._policy.allow_non_loopback:
            return GatewayAuthorization(
                False,
                "non_loopback_forbidden",
                "The local gateway accepts loopback clients only.",
            )

        request_host = _header_host(host_header)
        if not self._policy.allow_non_loopback and not is_loopback_host(request_host):
            return GatewayAuthorization(
                False,
                "invalid_host",
                "The gateway Host header must identify a loopback address.",
            )

        if origin_header:
            origin_host = _origin_host(origin_header)
            if not is_loopback_host(origin_host):
                return GatewayAuthorization(
                    False,
                    "origin_forbidden",
                    "Browser requests from non-loopback origins are forbidden.",
                )

        key_env = self._policy.api_key_env
        if key_env:
            expected = os.environ.get(key_env, "")
            supplied = _bearer_token(authorization_header)
            if not expected:
                return GatewayAuthorization(
                    False,
                    "gateway_key_unavailable",
                    "The configured gateway key is unavailable.",
                )
            if not supplied or not hmac.compare_digest(supplied, expected):
                return GatewayAuthorization(
                    False,
                    "invalid_api_key",
                    "The gateway API key is missing or invalid.",
                )

        return GatewayAuthorization(True, "allowed", "Gateway access allowed.")

    def models_payload(self) -> dict[str, Any]:
        models: list[dict[str, Any]] = []
        if self._experts:
            models.append(
                {
                    "id": self._policy.model_alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "mymoe-local",
                    "permission": [],
                    "mymoe": {
                        "selection": "deterministic_route",
                        "execution_scope": "configured_policy",
                    },
                }
            )
        for expert in sorted(self._experts, key=lambda item: item.id):
            eligibility = self._guard.evaluate(expert.execution_target)
            models.append(
                {
                    "id": self._pinned_alias(expert.id),
                    "object": "model",
                    "created": 0,
                    "owned_by": "mymoe-local",
                    "permission": [],
                    "mymoe": {
                        "selection": "pinned",
                        "expert_id": expert.id,
                        "upstream_model": expert.model,
                        "role": expert.role,
                        "execution_scope": (
                            eligibility.scope.value if eligibility.scope is not None else ""
                        ),
                        "execution_transport": (
                            eligibility.transport.value
                            if eligibility.transport is not None
                            else ""
                        ),
                        "eligible": eligibility.allowed,
                    },
                }
            )
        return {"object": "list", "data": models}

    def prepare_chat_completion(
        self,
        payload: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> PreparedChatCompletion:
        if not self._policy.enabled:
            raise GatewayRequestError(
                "The local gateway is disabled.",
                code="gateway_disabled",
                status=404,
            )
        if not isinstance(payload, Mapping):
            raise GatewayRequestError("The request body must be a JSON object.")

        caller_payload = dict(payload)
        validate_json_depth(caller_payload)
        canonical_caller = _canonical_json_bytes(caller_payload)
        if len(canonical_caller) > self._policy.max_request_bytes:
            raise GatewayRequestError(
                "The request body exceeds the configured gateway limit.",
                code="request_too_large",
                status=413,
            )

        messages = caller_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayRequestError(
                "messages must be a non-empty array.",
                parameter="messages",
            )
        if len(messages) > 4096 or not all(isinstance(item, dict) for item in messages):
            raise GatewayRequestError(
                "messages must contain at most 4096 message objects.",
                parameter="messages",
            )
        _validate_message_content_urls(messages)

        stream = caller_payload.get("stream", False)
        if not isinstance(stream, bool):
            raise GatewayRequestError("stream must be boolean.", parameter="stream")

        tools = caller_payload.get("tools")
        if tools is not None and (
            not isinstance(tools, list)
            or len(tools) > 256
            or not all(isinstance(item, dict) for item in tools)
        ):
            raise GatewayRequestError(
                "tools must contain at most 256 tool definition objects.",
                parameter="tools",
            )

        requested_model = str(
            caller_payload.get("model", self._policy.model_alias)
        ).strip()
        if not requested_model or len(requested_model) > 256:
            raise GatewayRequestError(
                "model must contain between 1 and 256 characters.",
                parameter="model",
            )

        routing_text = _routing_text(messages)
        expert, selected_ids = self._select_expert(requested_model, routing_text)
        self._guard.require_allowed(expert.execution_target)
        if not expert.base_url:
            raise GatewayRequestError(
                f"Expert {expert.id} has no OpenAI-compatible base URL.",
                code="model_unavailable",
                parameter="model",
                status=503,
            )

        upstream_payload = _json_clone(caller_payload)
        upstream_payload["model"] = expert.model
        defaults = provider_request_params(expert, routing_text)
        reserved_fields = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
            "stream_options",
        }
        for key, value in defaults.items():
            if key not in reserved_fields and key not in upstream_payload:
                upstream_payload[key] = value

        upstream_bytes = _canonical_json_bytes(upstream_payload)
        if len(upstream_bytes) > self._policy.max_request_bytes:
            raise GatewayRequestError(
                "The provider-ready request exceeds the configured gateway limit.",
                code="request_too_large",
                status=413,
            )

        cid = _correlation_id(correlation_id)
        upstream_url = str(expert.base_url).rstrip("/") + "/chat/completions"
        upstream_request = request.Request(
            upstream_url,
            data=upstream_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "X-MyMoE-Correlation-ID": cid,
            },
            method="POST",
        )
        return PreparedChatCompletion(
            requested_model=requested_model,
            expert=expert,
            upstream_request=upstream_request,
            correlation_id=cid,
            request_sha256=hashlib.sha256(canonical_caller).hexdigest(),
            routing_sha256=hashlib.sha256(routing_text.encode("utf-8")).hexdigest(),
            stream=stream,
            route_selected=selected_ids,
        )

    def open_chat_completion(self, prepared: PreparedChatCompletion) -> Any:
        current = self._experts_by_id.get(prepared.expert.id)
        if current is None or current != prepared.expert:
            raise GatewayRequestError(
                "The selected model is no longer configured.",
                code="model_unavailable",
                status=503,
            )
        self._guard.require_allowed(current.execution_target)
        return self._opener(
            prepared.upstream_request,
            timeout=current.timeout_seconds,
        )

    def error_payload(self, error: GatewayRequestError) -> dict[str, Any]:
        return openai_error_payload(
            str(error),
            code=error.code,
            parameter=error.parameter,
        )

    def _select_expert(
        self,
        requested_model: str,
        routing_text: str,
    ) -> tuple[ExpertConfig, tuple[str, ...]]:
        if requested_model == self._policy.model_alias:
            if not self._experts:
                raise GatewayRequestError(
                    "No OpenAI-compatible local expert is configured.",
                    code="model_unavailable",
                    parameter="model",
                    status=503,
                )
            route = self._router.route(routing_text)
            selected_ids = tuple(item.expert_id for item in route.selected)
            expert = next(
                (
                    self._experts_by_id[expert_id]
                    for expert_id in selected_ids
                    if expert_id in self._experts_by_id
                ),
                None,
            )
            if expert is None:
                raise GatewayRequestError(
                    "The route selected no OpenAI-compatible local expert.",
                    code="model_unavailable",
                    parameter="model",
                    status=503,
                )
            return expert, selected_ids

        prefix = f"{self._policy.model_alias}/"
        if requested_model.startswith(prefix):
            expert_id = requested_model[len(prefix) :]
            expert = self._experts_by_id.get(expert_id)
            if expert is not None:
                return expert, (expert.id,)

        matching_models = [
            expert for expert in self._experts if expert.model == requested_model
        ]
        if len(matching_models) == 1:
            return matching_models[0], (matching_models[0].id,)

        raise GatewayRequestError(
            f"Unknown local model alias: {requested_model}",
            code="model_not_found",
            parameter="model",
            status=404,
        )

    def _pinned_alias(self, expert_id: str) -> str:
        return f"{self._policy.model_alias}/{expert_id}"


def openai_error_payload(
    message: str,
    *,
    code: str,
    parameter: str | None = None,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    return {
        "error": {
            "message": str(message),
            "type": error_type,
            "param": parameter,
            "code": code,
        }
    }


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise GatewayRequestError(
            "The request must contain finite JSON-compatible values.",
            code="invalid_json",
        ) from exc


def _json_clone(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def validate_json_depth(
    payload: object,
    *,
    max_depth: int = MAX_GATEWAY_JSON_DEPTH,
) -> None:
    """Reject structures that can exhaust recursive JSON encoders/decoders."""

    stack: list[tuple[object, int]] = [(payload, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > max_depth:
            raise GatewayRequestError(
                f"The request JSON exceeds the maximum depth of {max_depth}.",
                code="invalid_json",
            )
        if isinstance(value, Mapping):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend((item, depth + 1) for item in value)


def _validate_message_content_urls(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content")
        parts: tuple[object, ...]
        if isinstance(content, list):
            parts = tuple(content)
        elif isinstance(content, Mapping):
            parts = (content,)
        else:
            parts = ()
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            for content_url in _content_urls(part):
                try:
                    scheme = urlsplit(content_url.strip()).scheme.lower()
                except ValueError as exc:
                    raise GatewayRequestError(
                        "Message content contains an invalid URL.",
                        code="unsafe_content_url",
                        parameter="messages",
                    ) from exc
                if scheme != "data":
                    raise GatewayRequestError(
                        "Only inline data: URLs are allowed in message content.",
                        code="unsafe_content_url",
                        parameter="messages",
                    )


def _content_urls(part: Mapping[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    for field in _CONTENT_URL_FIELDS:
        raw = part.get(field)
        if isinstance(raw, str):
            urls.append(raw)
        elif isinstance(raw, Mapping):
            nested = raw.get("url")
            if isinstance(nested, str):
                urls.append(nested)

    part_type = str(part.get("type", "")).strip().lower()
    direct_url = part.get("url")
    if part_type in _URL_CONTENT_TYPES and isinstance(direct_url, str):
        urls.append(direct_url)
    return tuple(urls)


def _routing_text(messages: list[dict[str, Any]]) -> str:
    user_texts = [
        text
        for message in messages
        if str(message.get("role", "")).strip().lower() == "user"
        if (text := _message_content_text(message.get("content")))
    ]
    # The first user instruction is stable across the model/tool loop, which
    # keeps automatic routing session-sticky for normal coding-agent requests.
    return (user_texts[0] if user_texts else "local coding task")[:64_000]


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        raw = part.get("text")
        if isinstance(raw, str) and raw.strip():
            parts.append(raw.strip())
    return "\n".join(parts)


def _correlation_id(value: str | None) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if _SAFE_CORRELATION_ID_PATTERN.fullmatch(cleaned) is not None:
        return cleaned
    return str(uuid4())


def _bearer_token(header: str | None) -> str:
    raw = str(header or "").strip()
    prefix = "bearer "
    if not raw.lower().startswith(prefix):
        return ""
    return raw[len(prefix) :].strip()


def is_loopback_host(host: str) -> bool:
    normalized = str(host).strip().rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _header_host(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(urlsplit(f"//{raw}").hostname or "")
    except ValueError:
        return ""


def _origin_host(value: str) -> str:
    try:
        parsed = urlsplit(str(value).strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    return str(parsed.hostname or "")
