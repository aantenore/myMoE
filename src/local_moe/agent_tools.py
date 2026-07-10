from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from .agent_types import AgentToolCall, AgentToolSpec
from .agent_tool_schemas import LOCAL_TOOL_CONFIRMATIONS, LOCAL_TOOL_INPUT_SCHEMAS
from .extensions import ExtensionRegistry
from .redaction import REDACTED_VALUE
from .tool_runner import ToolExecutionError, ToolRunResult, tool_result_payload


AUTO_ALLOW_RISKS = ("read_only", "search_only", "compute_only", "draft_only")
APPROVAL_REQUIRED_RISKS = (
    "write_local",
    "write_internal",
    "write_external",
    "financial",
    "communication",
    "identity_access",
    "security_sensitive",
    "process_execution",
    "network_open_world",
    "destructive",
    "privileged_admin",
)


class ToolRunner(Protocol):
    def run(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolRunResult: ...


@dataclass(frozen=True)
class ApprovalRequest:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    arguments_sha256: str
    risk_class: str
    side_effects: str
    scope: str = "single_tool_call"


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""


ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision | bool]


@dataclass(frozen=True)
class AgentPermissionPolicy:
    auto_allow_risks: tuple[str, ...] = AUTO_ALLOW_RISKS
    approval_required_risks: tuple[str, ...] = APPROVAL_REQUIRED_RISKS
    denied_risks: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()
    approval_required_tools: tuple[str, ...] = ("data.export",)

    def decision(self, spec: AgentToolSpec) -> str:
        if spec.name in self.denied_tools:
            return "deny"
        if spec.risk_class in self.denied_risks:
            return "deny"
        if spec.name in self.approval_required_tools:
            return "approval_required"
        if spec.risk_class in self.auto_allow_risks:
            return "allow"
        if spec.risk_class in self.approval_required_risks:
            return "approval_required"
        return "deny"


@dataclass(frozen=True)
class AgentToolResult:
    call_id: str
    tool_name: str
    status: str
    code: str
    message: str
    risk_class: str
    data: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "risk_class": self.risk_class,
            "data": self.data,
        }


@dataclass(frozen=True)
class AgentToolExecution:
    result: AgentToolResult
    approval_request: ApprovalRequest | None = None
    pause_required: bool = False
    permission_decision: str = ""
    approval_status: str = ""


class AgentToolRegistry:
    """Resolve, validate, authorize, and execute strict local tool contracts."""

    def __init__(
        self,
        runner: ToolRunner,
        specs: Sequence[AgentToolSpec],
        *,
        permission_policy: AgentPermissionPolicy | None = None,
    ):
        self._runner = runner
        self._permission_policy = permission_policy or AgentPermissionPolicy()
        by_name: dict[str, AgentToolSpec] = {}
        by_exposed_name: dict[str, AgentToolSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate canonical tool name: {spec.name}")
            if spec.exposed_name in by_exposed_name:
                raise ValueError(
                    f"Duplicate model-visible tool name: {spec.exposed_name}"
                )
            by_name[spec.name] = spec
            by_exposed_name[spec.exposed_name] = spec
        self._by_name = by_name
        self._by_exposed_name = by_exposed_name

    @classmethod
    def from_local_tools(
        cls,
        runner: ToolRunner,
        registry: ExtensionRegistry,
        *,
        visible_tools: Sequence[str] | None = None,
        schemas: Mapping[str, Mapping[str, Any]] | None = None,
        permission_policy: AgentPermissionPolicy | None = None,
    ) -> AgentToolRegistry:
        schema_map = LOCAL_TOOL_INPUT_SCHEMAS if schemas is None else schemas
        requested = set(visible_tools) if visible_tools is not None else None
        configured = {tool.name: tool for tool in registry.tools if tool.enabled}
        if requested is not None:
            missing = requested - configured.keys()
            if missing:
                raise ValueError(
                    f"Requested tools are not enabled/configured: {sorted(missing)}"
                )
            missing_schemas = requested - schema_map.keys()
            if missing_schemas:
                raise ValueError(
                    f"Requested tools have no strict schema: {sorted(missing_schemas)}"
                )

        specs = []
        for name in sorted(configured):
            if requested is not None and name not in requested:
                continue
            if name not in schema_map:
                # A tool without a strict schema is deliberately not exposed to
                # the model. It can still be invoked manually through the runner.
                continue
            tool = configured[name]
            specs.append(
                AgentToolSpec(
                    name=name,
                    description=tool.description,
                    input_schema=schema_map[name],
                    risk_class=tool.risk_class,
                    side_effects=tool.side_effects,
                )
            )
        return cls(runner, specs, permission_policy=permission_policy)

    @property
    def specs(self) -> tuple[AgentToolSpec, ...]:
        return tuple(self._by_name[name] for name in sorted(self._by_name))

    def resolve(self, name: str) -> AgentToolSpec | None:
        return self._by_name.get(name) or self._by_exposed_name.get(name)

    def execute(
        self,
        call: AgentToolCall,
        *,
        approval_handler: ApprovalHandler | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentToolExecution:
        spec = self.resolve(call.name)
        if spec is None:
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name="unknown",
                    status="error",
                    code="unknown_tool",
                    message="The requested tool is not registered or visible.",
                    risk_class="unknown",
                ),
                permission_decision="not_evaluated",
            )

        errors = validate_json_arguments(spec.input_schema, call.arguments)
        if errors:
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="invalid_arguments",
                    message="; ".join(errors[:8]),
                    risk_class=spec.risk_class,
                ),
                permission_decision="not_evaluated",
            )

        arguments = dict(call.arguments)  # type: ignore[arg-type]
        if contains_non_finite_number(arguments):
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="invalid_arguments",
                    message="Tool arguments must contain only finite JSON numbers.",
                    risk_class=spec.risk_class,
                ),
                permission_decision="not_evaluated",
            )
        if _contains_secret_key(arguments):
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="secret_argument_forbidden",
                    message="Credentials and secret-like fields cannot be supplied by the model.",
                    risk_class=spec.risk_class,
                ),
                permission_decision="not_evaluated",
            )

        decision = self._permission_policy.decision(spec)
        if decision == "deny":
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="denied",
                    code="permission_denied",
                    message="Runtime policy denied this tool risk class.",
                    risk_class=spec.risk_class,
                ),
                permission_decision="deny",
            )

        approval_request: ApprovalRequest | None = None
        if decision == "approval_required":
            approval_request = ApprovalRequest(
                call_id=call.id,
                tool_name=spec.name,
                arguments=_sanitize_agent_value(arguments),  # type: ignore[arg-type]
                arguments_sha256=arguments_sha256(arguments),
                risk_class=spec.risk_class,
                side_effects=spec.side_effects,
            )
            if approval_handler is None:
                return AgentToolExecution(
                    AgentToolResult(
                        call_id=call.id,
                        tool_name=spec.name,
                        status="approval_required",
                        code="approval_required",
                        message="Explicit approval is required for this exact tool call.",
                        risk_class=spec.risk_class,
                        data={"arguments_sha256": approval_request.arguments_sha256},
                    ),
                    approval_request=approval_request,
                    pause_required=True,
                    permission_decision="approval_required",
                    approval_status="pending",
                )
            try:
                raw_decision = approval_handler(approval_request)
            except Exception:
                return AgentToolExecution(
                    AgentToolResult(
                        call_id=call.id,
                        tool_name=spec.name,
                        status="approval_required",
                        code="approval_handler_error",
                        message="The approval handler failed; the tool was not executed.",
                        risk_class=spec.risk_class,
                    ),
                    approval_request=approval_request,
                    pause_required=True,
                    permission_decision="approval_required",
                    approval_status="error",
                )
            valid_decision = type(raw_decision) is bool or (
                isinstance(raw_decision, ApprovalDecision)
                and type(raw_decision.approved) is bool
                and isinstance(raw_decision.reason, str)
            )
            if not valid_decision:
                return AgentToolExecution(
                    AgentToolResult(
                        call_id=call.id,
                        tool_name=spec.name,
                        status="approval_required",
                        code="approval_handler_error",
                        message="The approval handler returned an invalid decision; the tool was not executed.",
                        risk_class=spec.risk_class,
                    ),
                    approval_request=approval_request,
                    pause_required=True,
                    permission_decision="approval_required",
                    approval_status="error",
                )
            approved = (
                raw_decision if type(raw_decision) is bool else raw_decision.approved  # type: ignore[union-attr]
            )
            reason = (
                "" if type(raw_decision) is bool else raw_decision.reason  # type: ignore[union-attr]
            )
            if not approved:
                return AgentToolExecution(
                    AgentToolResult(
                        call_id=call.id,
                        tool_name=spec.name,
                        status="denied",
                        code="approval_denied",
                        message=redact_agent_text(reason.strip())
                        or "The exact tool call was not approved.",
                        risk_class=spec.risk_class,
                    ),
                    approval_request=approval_request,
                    permission_decision="approval_required",
                    approval_status="denied",
                )
            arguments.update(LOCAL_TOOL_CONFIRMATIONS.get(spec.name, {}))

        try:
            raw_result = self._runner.run(
                spec.name,
                arguments,
                timeout_seconds=timeout_seconds,
            )
        except ToolExecutionError as exc:
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="tool_execution_error",
                    message=redact_agent_text(str(exc)),
                    risk_class=spec.risk_class,
                ),
                approval_request=approval_request,
                permission_decision=decision,
                approval_status="approved" if approval_request is not None else "",
            )
        except Exception as exc:
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="internal_error",
                    message=f"Tool execution failed with {type(exc).__name__}.",
                    risk_class=spec.risk_class,
                ),
                approval_request=approval_request,
                permission_decision=decision,
                approval_status="approved" if approval_request is not None else "",
            )

        try:
            payload = tool_result_payload(raw_result)
            safe_payload = _sanitize_agent_value(payload.get("payload", {}))
            result_status = str(raw_result.status)
            result_message = str(raw_result.message)
        except Exception:
            return AgentToolExecution(
                AgentToolResult(
                    call_id=call.id,
                    tool_name=spec.name,
                    status="error",
                    code="invalid_tool_result",
                    message="The tool returned an invalid or unserializable structured result.",
                    risk_class=spec.risk_class,
                ),
                approval_request=approval_request,
                permission_decision=decision,
                approval_status="approved" if approval_request is not None else "",
            )
        tool_reported_error = (
            spec.name == "mcp.call_tool"
            and isinstance(safe_payload, dict)
            and safe_payload.get("is_error") is True
        )
        return AgentToolExecution(
            AgentToolResult(
                call_id=call.id,
                tool_name=spec.name,
                status=(
                    "error"
                    if tool_reported_error
                    else "success"
                    if result_status == "ok"
                    else result_status
                ),
                code=(
                    "tool_reported_error"
                    if tool_reported_error
                    else "ok"
                    if result_status == "ok"
                    else "tool_result"
                ),
                message=redact_agent_text(result_message),
                risk_class=spec.risk_class,
                data=safe_payload
                if isinstance(safe_payload, dict)
                else {"value": safe_payload},
            ),
            approval_request=approval_request,
            permission_decision=decision,
            approval_status="approved" if approval_request is not None else "",
        )


def arguments_sha256(arguments: object) -> str:
    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: type(value).__name__,
        )
    except (TypeError, ValueError, RecursionError):
        serialized = f"unserializable:{type(arguments).__name__}"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def argument_size_chars(arguments: object) -> int:
    try:
        return len(
            json.dumps(
                arguments,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=lambda value: type(value).__name__,
            )
        )
    except (TypeError, ValueError, RecursionError):
        return 2**63 - 1


def redact_agent_text(value: str) -> str:
    redacted = str(value)
    patterns = (
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}",
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|authorization|credential)\b\s*[:=]\s*[^\s,;]+",
        r"(?i)\b(?:sk|ghp|github_pat|xox[baprs])[-_][a-z0-9_-]{8,}\b",
    )
    for pattern in patterns:
        redacted = re.sub(pattern, REDACTED_VALUE, redacted)
    return redacted


def bound_tool_result(
    result: AgentToolResult,
    *,
    max_chars: int,
) -> tuple[AgentToolResult, str]:
    serialized = _serialize_result(result)
    if len(serialized) <= max_chars:
        return result, serialized

    bounded = replace(
        result,
        message="Tool result was truncated by the harness result-size budget.",
        data={"truncated": True, "original_chars": len(serialized)},
    )
    compact = _serialize_result(bounded)
    if len(compact) > max_chars:
        # AgentLoopBudget enforces at least 512 chars; this is defensive for
        # unusually long provider call IDs or custom tool names.
        bounded = replace(
            bounded,
            call_id=bounded.call_id[:96],
            tool_name=bounded.tool_name[:96],
            message="Result truncated.",
        )
        compact = _serialize_result(bounded)
    if len(compact) > max_chars:
        bounded = AgentToolResult(
            call_id="bounded",
            tool_name="bounded",
            status=result.status,
            code="result_truncated",
            message="Result truncated.",
            risk_class=result.risk_class,
            data={"truncated": True},
        )
        compact = _serialize_result(bounded)
    return bounded, compact


def validate_json_arguments(
    schema: Mapping[str, Any], value: object
) -> tuple[str, ...]:
    errors: list[str] = []
    _validate_schema(schema, value, "$", errors)
    return tuple(errors)


def _validate_schema(
    schema: Mapping[str, Any],
    value: object,
    path: str,
    errors: list[str],
) -> None:
    expected = schema.get("type")
    if expected and not _matches_type(str(expected), value):
        errors.append(f"{path} must be {expected}")
        return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {list(schema['enum'])}")

    if isinstance(value, dict):
        required = tuple(str(item) for item in schema.get("required", ()))
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")

        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")
        for key, nested in value.items():
            nested_schema = properties.get(key)
            if isinstance(nested_schema, Mapping):
                _validate_schema(nested_schema, nested, f"{path}.{key}", errors)

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if minimum_items is not None and len(value) < int(minimum_items):
            errors.append(f"{path} must contain at least {minimum_items} items")
        if maximum_items is not None and len(value) > int(maximum_items):
            errors.append(f"{path} must contain at most {maximum_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item_schema, item, f"{path}[{index}]", errors)

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path} is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path} is too long")
        if "pattern" in schema and re.fullmatch(str(schema["pattern"]), value) is None:
            errors.append(f"{path} has an invalid format")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            errors.append(f"{path} must be finite")
            return
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")

    alternatives = schema.get("oneOf")
    if isinstance(alternatives, list):
        matches = 0
        for alternative in alternatives:
            nested_errors: list[str] = []
            if isinstance(alternative, Mapping):
                _validate_schema(alternative, value, path, nested_errors)
            if not nested_errors:
                matches += 1
        if matches != 1:
            errors.append(f"{path} must match exactly one allowed argument shape")


def _matches_type(expected: str, value: object) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            _is_secret_field(str(key)) or _contains_secret_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    if isinstance(value, str):
        return redact_agent_text(value) != value
    return False


def contains_non_finite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_non_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_non_finite_number(item) for item in value)
    return False


def _sanitize_agent_value(value: object) -> object:
    if isinstance(value, str):
        return redact_agent_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): (
                REDACTED_VALUE
                if _is_secret_field(str(key))
                else _sanitize_agent_value(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_agent_value(item) for item in value]
    return type(value).__name__


def _is_secret_field(key: str) -> bool:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    exact = {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "session_token",
        "id_token",
        "token",
        "client_secret",
        "private_key",
        "ssh_private_key",
        "secret_access_key",
        "aws_secret_access_key",
        "signing_key",
        "encryption_key",
        "secret",
        "secret_value",
        "password",
        "passwd",
        "credential",
        "credentials",
        "authorization",
    }
    secret_suffixes = (
        "_api_key",
        "_access_token",
        "_refresh_token",
        "_auth_token",
        "_token",
        "_password",
        "_secret",
        "_credential",
        "_private_key",
        "_secret_access_key",
        "_signing_key",
        "_encryption_key",
    )
    return normalized in exact or normalized.endswith(secret_suffixes)


def _serialize_result(result: AgentToolResult) -> str:
    return json.dumps(
        result.payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
