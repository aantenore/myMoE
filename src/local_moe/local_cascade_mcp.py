from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import hmac
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


SCHEMA_VERSION = "1.0"
DEFAULT_MAX_OUTPUT_CHARS = 4_000
MAX_OUTPUT_CHARS = 16_000
MAX_TASK_CHARS = 32_768
MAX_STEPS = 16
MAX_STORED_PLANS = 128
MAX_STORED_RECEIPTS = 256
EFFICIENCY_PROFILES = frozenset({"economy", "balanced", "quality"})
TASK_KINDS = frozenset({"classification", "extraction", "summarization"})
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")

CASCADE_CONFIG_ENV = "MYMOE_LOCAL_CASCADE_CONFIG"
MOE_CONFIG_ENV = "MYMOE_LOCAL_CASCADE_MOE_CONFIG"
FINISH_REASON_RE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,63}\Z")
TERMINAL_FINISH_REASONS_PARAM = "local_cascade_terminal_finish_reasons"
NON_TERMINAL_FINISH_REASONS = frozenset(
    {
        "cancelled",
        "canceled",
        "content_filter",
        "error",
        "function_call",
        "length",
        "max_tokens",
        "timeout",
        "tool_calls",
    }
)
_PUBLIC_ERROR_MESSAGES = {
    "absolute_configuration_path_required": (
        "Local Cascade configuration variables must use absolute file paths."
    ),
    "adapter_unavailable": "Local cascade is not available.",
    "configuration_required": (
        "Set MYMOE_LOCAL_CASCADE_CONFIG and MYMOE_LOCAL_CASCADE_MOE_CONFIG "
        "before starting the MCP server."
    ),
    "device_only_required": (
        "Cascade experts must be configured for direct device-only execution."
    ),
    "invalid_configuration": "The local cascade configuration is invalid.",
    "invalid_efficiency_profile": "The efficiency profile is unsupported.",
    "invalid_max_steps": "The local cascade step limit is invalid.",
    "model_ref_not_configured": (
        "Every cascade tier model_ref must match one configured expert id."
    ),
    "model_ref_unavailable": "The planned local expert is no longer configured.",
    "cascade_busy": (
        "Another local cascade invocation is already running; try again later."
    ),
    "output_limit_below_verifier_maximum": (
        "max_output_chars must cover the configured verifier maximum."
    ),
    "output_limit_exceeded": (
        "The accepted local result exceeded the requested output limit."
    ),
    "verifier_output_exceeds_mcp_limit": (
        "The configured verifier maximum exceeds the MCP output limit."
    ),
    "numeric_loopback_required": ("Cascade experts must use a numeric loopback host."),
    "plan_binding_mismatch": (
        "The supplied plan digest does not match the stored plan."
    ),
    "plan_not_found": ("The in-memory local plan was not found; plan the task again."),
    "provider_binding_error": (
        "The configured local provider returned a different expert binding."
    ),
    "provider_contract_error": (
        "The configured local provider returned an invalid result."
    ),
    "task_binding_error": "The task digest does not match the task text.",
    "task_binding_mismatch": "The task text does not match the stored plan.",
    "unsupported_local_provider": (
        "Local Cascade alpha requires an OpenAI-compatible local provider."
    ),
    "unsupported_task_kind": (
        "task_kind must be classification, extraction, or summarization."
    ),
}
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "chain_of_thought",
        "content",
        "credential",
        "credentials",
        "env",
        "environment",
        "messages",
        "output",
        "password",
        "prompt",
        "raw_content",
        "raw_output",
        "raw_result",
        "raw_task",
        "reasoning",
        "result",
        "scratchpad",
        "secret",
        "task",
    }
)
_MACHINE_FIELDS = (
    "schema_version",
    "status",
    "machine",
    "system",
    "cpu",
    "hardware",
    "memory_gib",
    "accelerators",
    "resource_snapshot",
    "runtimes",
    "models",
    "recommendation",
    "inspection_sha256",
    "warnings",
)
_PLAN_FIELDS = (
    "schema_version",
    "status",
    "plan",
    "plan_id",
    "plan_sha256",
    "task_sha256",
    "selected_tier",
    "tier",
    "selected_model",
    "model",
    "route",
    "attempts",
    "reason_codes",
    "estimated_usage",
    "limits",
    "installation",
    "receipt",
    "evidence_sha256",
    "warnings",
)
_RUN_FIELDS = (
    "schema_version",
    "status",
    "model",
    "tier",
    "route",
    "usage",
    "latency_ms",
    "reason_codes",
    "receipt",
    "evidence_sha256",
    "warnings",
)
_RECEIPT_FIELDS = (
    "schema_version",
    "status",
    "receipt_id",
    "run_id",
    "id",
    "kind",
    "contract",
    "created_at",
    "plan_id",
    "plan_sha256",
    "task_sha256",
    "request_task_sha256",
    "config_sha256",
    "efficiency_profile",
    "finish_reason_policy_bound",
    "model",
    "tier",
    "route",
    "selected_tier_id",
    "attempt_count",
    "total_duration_ms",
    "attempts",
    "token_totals",
    "requested_execution_scope",
    "execution_scope_attestation",
    "parallel_attempts",
    "usage",
    "latency_ms",
    "reason_codes",
    "privacy",
    "core_receipt",
    "evidence_sha256",
    "warnings",
)


class CascadeAdapterUnavailable(RuntimeError):
    """Raised when the optional local-cascade core cannot be loaded."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "adapter_unavailable",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


class CascadeAdapterRejected(RuntimeError):
    """A stable, user-actionable rejection that contains no private values."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@dataclass(frozen=True)
class DelegatePlanRequest:
    task: str
    task_sha256: str
    task_kind: str
    efficiency_profile: str
    max_steps: int
    installation_mode: str


@dataclass(frozen=True)
class DelegateRunRequest:
    task: str
    task_sha256: str
    plan_id: str
    plan_sha256: str
    max_output_chars: int


@runtime_checkable
class CascadeAdapter(Protocol):
    """Narrow seam between the Codex plugin and the reusable cascade core."""

    def inspect_machine(self) -> object: ...

    def plan(self, request: DelegatePlanRequest) -> object: ...

    def run(self, request: DelegateRunRequest) -> object: ...

    def inspect_receipt(self, receipt_id: str) -> object: ...


class LazyCascadeAdapter:
    """Load the core only on the first tool call, not while Codex discovers tools."""

    def __init__(self, factory: Callable[[], CascadeAdapter] | None = None) -> None:
        self._factory = factory or load_default_adapter
        self._adapter: CascadeAdapter | None = None
        self._lock = Lock()

    def inspect_machine(self) -> object:
        return self._resolve().inspect_machine()

    def plan(self, request: DelegatePlanRequest) -> object:
        return self._resolve().plan(request)

    def run(self, request: DelegateRunRequest) -> object:
        return self._resolve().run(request)

    def inspect_receipt(self, receipt_id: str) -> object:
        return self._resolve().inspect_receipt(receipt_id)

    def _resolve(self) -> CascadeAdapter:
        if self._adapter is not None:
            return self._adapter
        with self._lock:
            if self._adapter is None:
                self._adapter = self._factory()
        return self._adapter


def load_default_adapter() -> CascadeAdapter:
    """Load the configuration-first core adapter on the first MCP tool call."""

    return LocalCascadeCoreAdapter.from_environment()


@dataclass(frozen=True)
class _ConfigurationBundle:
    cascade_config: Any
    moe_config: Any
    experts_by_id: Mapping[str, Any]
    cascade_config_sha256: str
    moe_config_sha256: str


@dataclass(frozen=True)
class _PlanRecord:
    plan_id: str
    plan_sha256: str
    raw_task: str
    raw_task_sha256: str
    task: Any
    filtered_config: Any
    profile: str
    max_steps: int


class _ConfiguredAttemptPort:
    def __init__(
        self,
        *,
        experts_by_id: Mapping[str, Any],
        verifier: Any,
        scope_guard: Any,
        provider_factory: Callable[[Any], Any],
    ) -> None:
        self._experts_by_id = experts_by_id
        self._verifier = verifier
        self._scope_guard = scope_guard
        self._provider_factory = provider_factory

    def attempt(self, request: Any) -> Any:
        from .local_cascade_contracts import (
            LocalCascadeAttemptResultV1,
            LocalCascadeTokenCountV1,
        )
        from .providers import ExpertResult, GenerationRequest, strip_reasoning_content

        expert = self._experts_by_id.get(request.tier.model_ref)
        if expert is None:
            raise CascadeAdapterRejected(
                "model_ref_unavailable",
                "The planned local expert is no longer configured.",
            )
        _require_numeric_device_only_expert(expert, self._scope_guard)
        provider = self._provider_factory(expert)
        correlation_id = f"{request.task.task_id}-attempt-{request.attempt_number}"
        result = provider.generate(
            expert,
            GenerationRequest(
                prompt=_attempt_prompt(request, self._verifier),
                correlation_id=correlation_id,
                max_output_tokens=request.tier.max_output_tokens,
            ),
        )
        if not isinstance(result, ExpertResult):
            raise CascadeAdapterRejected(
                "provider_contract_error",
                "The configured local provider returned an invalid result.",
            )
        if result.expert_id != expert.id:
            raise CascadeAdapterRejected(
                "provider_binding_error",
                "The configured local provider returned a different expert binding.",
            )
        if result.model != expert.model or result.correlation_id != correlation_id:
            raise CascadeAdapterRejected(
                "provider_binding_error",
                "The configured local provider returned a different expert binding.",
            )
        input_tokens = _actual_or_unknown_tokens(
            result.prompt_tokens,
            LocalCascadeTokenCountV1,
        )
        output_tokens = _actual_or_unknown_tokens(
            result.completion_tokens,
            LocalCascadeTokenCountV1,
        )
        finish_reason = (
            result.finish_reason.strip().casefold()
            if isinstance(result.finish_reason, str)
            else None
        )
        if finish_reason not in _terminal_finish_reasons(expert):
            return LocalCascadeAttemptResultV1(
                status="abstained",
                content=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return LocalCascadeAttemptResultV1(
            status="completed",
            content=strip_reasoning_content(result.content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class LocalCascadeCoreAdapter:
    """Configuration-first adapter around the reusable offline cascade core."""

    def __init__(
        self,
        bundle: _ConfigurationBundle,
        *,
        provider_factory: Callable[[Any], Any] | None = None,
        opener: Callable[..., Any] | None = None,
        hardware_probe: Callable[[], object] | None = None,
        resource_probe: Callable[[], object] | None = None,
    ) -> None:
        from .execution_scope import ExecutionScopeGuard
        from .hardware import detect_hardware
        from .resource_snapshot import collect_resource_snapshot

        self._bundle = bundle
        self._scope_guard = ExecutionScopeGuard(bundle.moe_config.execution_policy)
        self._opener = opener
        self._provider_factory = provider_factory or self._build_provider
        self._hardware_probe = hardware_probe or detect_hardware
        self._resource_probe = resource_probe or collect_resource_snapshot
        self._plans: OrderedDict[str, _PlanRecord] = OrderedDict()
        self._receipts: OrderedDict[str, bytes] = OrderedDict()
        self._state_lock = Lock()
        self._invocation_lock = Lock()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        provider_factory: Callable[[Any], Any] | None = None,
        opener: Callable[..., Any] | None = None,
        hardware_probe: Callable[[], object] | None = None,
        resource_probe: Callable[[], object] | None = None,
    ) -> LocalCascadeCoreAdapter:
        source = os.environ if environ is None else environ
        try:
            bundle = _load_configuration_bundle(source)
        except CascadeAdapterUnavailable:
            raise
        except CascadeAdapterRejected as exc:
            raise CascadeAdapterUnavailable(
                exc.public_message,
                code=exc.code,
            ) from exc
        except Exception as exc:
            raise CascadeAdapterUnavailable(
                "The local cascade configuration is invalid.",
                code="invalid_configuration",
            ) from exc
        return cls(
            bundle,
            provider_factory=provider_factory,
            opener=opener,
            hardware_probe=hardware_probe,
            resource_probe=resource_probe,
        )

    def inspect_machine(self) -> object:
        from .local_cascade_contracts import sha256_json

        hardware = self._hardware_probe()
        snapshot = _payload_mapping(self._resource_probe())
        models = [
            {
                "tier_id": tier.tier_id,
                "model_ref": tier.model_ref,
                "provider": self._bundle.experts_by_id[tier.model_ref].provider,
                "runtime_status": "configured_not_probed",
                "finish_reason_policy_bound": True,
            }
            for tier in self._bundle.cascade_config.ordered_tiers
        ]
        providers = sorted({str(item["provider"]) for item in models})
        inspection_binding = {
            "cascade_config_sha256": self._bundle.cascade_config_sha256,
            "moe_config_sha256": self._bundle.moe_config_sha256,
            "resource_source_sha256": snapshot.get("source_sha256"),
        }
        accelerator_kind = snapshot.get("accelerator_kind", "unknown")
        accelerator_memory = snapshot.get("accelerator_memory_available_bytes")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "configured",
            "machine": str(getattr(hardware, "machine", "unknown")),
            "system": str(snapshot.get("system", "unknown")),
            "cpu": {"brand": str(getattr(hardware, "cpu_brand", "unknown"))},
            "memory_gib": getattr(hardware, "memory_gib", 0.0),
            "accelerators": [
                {
                    "kind": accelerator_kind,
                    "available_memory_bytes": accelerator_memory,
                }
            ],
            "resource_snapshot": snapshot,
            "runtimes": [
                {"provider": provider, "status": "configured_not_probed"}
                for provider in providers
            ],
            "models": models,
            "recommendation": {
                "strategy": str(getattr(hardware, "recommended_strategy", "unknown")),
                "basis": "hardware_only",
            },
            "inspection_sha256": sha256_json(inspection_binding),
            "warnings": ["model_runtime_readiness_not_probed"],
        }

    def plan(self, request: DelegatePlanRequest) -> object:
        from .local_cascade_contracts import (
            LocalCascadeConfigV1,
            LocalCascadeTaskV1,
            sha256_json,
        )

        if request.task_sha256 != _task_sha256(request.task):
            raise CascadeAdapterRejected(
                "task_binding_error",
                "The task digest does not match the task text.",
            )
        if request.task_kind not in TASK_KINDS:
            raise CascadeAdapterRejected(
                "unsupported_task_kind",
                "task_kind must be classification, extraction, or summarization.",
            )
        if request.efficiency_profile not in EFFICIENCY_PROFILES:
            raise CascadeAdapterRejected(
                "invalid_efficiency_profile",
                "The efficiency profile is unsupported.",
            )
        if not 1 <= request.max_steps <= MAX_STEPS:
            raise CascadeAdapterRejected(
                "invalid_max_steps",
                "The local cascade step limit is invalid.",
            )
        if self._bundle.cascade_config.verifier.max_characters > MAX_OUTPUT_CHARS:
            raise CascadeAdapterRejected(
                "verifier_output_exceeds_mcp_limit",
                "The configured verifier maximum exceeds the MCP output limit.",
            )

        base = self._bundle.cascade_config
        eligible = tuple(base.ordered_tiers)
        if request.efficiency_profile == "economy":
            selected = eligible[:1]
            policy = "lowest_configured_cost_rank_single_attempt"
        elif request.efficiency_profile == "balanced":
            selected = eligible[: request.max_steps]
            policy = "configured_cost_rank_ascending_bounded_escalation"
        else:
            selected = (max(eligible, key=lambda tier: tier.cost_rank),)
            policy = "highest_configured_cost_rank_single_attempt"

        filtered_config = LocalCascadeConfigV1(
            cascade_id=base.cascade_id,
            tiers=selected,
            verifier=base.verifier,
            max_attempts=len(selected),
        )
        task = LocalCascadeTaskV1(
            task_id=f"cascade-task-{request.task_sha256[:40]}",
            kind=request.task_kind,
            instruction=request.task,
            output_format=base.verifier.output_format,
        )
        filtered_config_sha256 = sha256_json(filtered_config.payload())
        binding = {
            "task_sha256": request.task_sha256,
            "task_kind": request.task_kind,
            "cascade_config_sha256": self._bundle.cascade_config_sha256,
            "filtered_config_sha256": filtered_config_sha256,
            "moe_config_sha256": self._bundle.moe_config_sha256,
            "efficiency_profile": request.efficiency_profile,
            "max_steps": request.max_steps,
            "installation_mode": request.installation_mode,
        }
        plan_sha256 = sha256_json(binding)
        plan_id = f"cascade-plan-{plan_sha256}"
        record = _PlanRecord(
            plan_id=plan_id,
            plan_sha256=plan_sha256,
            raw_task=request.task,
            raw_task_sha256=request.task_sha256,
            task=task,
            filtered_config=filtered_config,
            profile=request.efficiency_profile,
            max_steps=request.max_steps,
        )
        reason_codes = (
            "strict_configuration_valid",
            "configured_direct_local_first_hop",
            policy,
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": plan_id,
            "kind": "plan",
            "status": "ready",
            "plan_id": plan_id,
            "plan_sha256": plan_sha256,
            "task_sha256": request.task_sha256,
            "config_sha256": filtered_config_sha256,
            "route": [
                {"tier_id": tier.tier_id, "model_ref": tier.model_ref}
                for tier in selected
            ],
            "reason_codes": list(reason_codes),
            "privacy": "metadata_only",
        }
        self._store_plan(record, receipt)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "plan_id": plan_id,
            "plan_sha256": plan_sha256,
            "task_sha256": request.task_sha256,
            "selected_tier": selected[0].tier_id,
            "selected_model": selected[0].model_ref,
            "route": {
                "policy": policy,
                "tier_ids": [tier.tier_id for tier in selected],
                "model_refs": [tier.model_ref for tier in selected],
                "max_attempts": len(selected),
                "requested_execution_scope": "offline_local",
                "execution_scope_attestation": "adapter_declared_unverified",
                "transport_boundary": "numeric_loopback_first_hop",
            },
            "limits": {
                "required_max_output_chars": base.verifier.max_characters,
                "mcp_max_output_chars": MAX_OUTPUT_CHARS,
                "finish_reason_policy_bound": True,
            },
            "reason_codes": list(reason_codes),
            "installation": {
                "mode": request.installation_mode,
                "status": "not_evaluated",
                "download_performed": False,
                "install_performed": False,
            },
            "receipt": receipt,
        }

    def run(self, request: DelegateRunRequest) -> object:
        from .local_cascade import run_local_cascade

        with self._state_lock:
            record = self._plans.get(request.plan_id)
        if record is None:
            raise CascadeAdapterRejected(
                "plan_not_found",
                "The in-memory local plan was not found; plan the task again.",
            )
        if not hmac.compare_digest(request.plan_sha256, record.plan_sha256):
            raise CascadeAdapterRejected(
                "plan_binding_mismatch",
                "The supplied plan digest does not match the stored plan.",
            )
        if (
            not hmac.compare_digest(request.task_sha256, record.raw_task_sha256)
            or request.task != record.raw_task
        ):
            raise CascadeAdapterRejected(
                "task_binding_mismatch",
                "The task text does not match the stored plan.",
            )
        if request.max_output_chars < record.filtered_config.verifier.max_characters:
            raise CascadeAdapterRejected(
                "output_limit_below_verifier_maximum",
                "max_output_chars must cover the configured verifier maximum.",
            )
        if not self._invocation_lock.acquire(blocking=False):
            raise CascadeAdapterRejected(
                "cascade_busy",
                "Another local cascade invocation is already running; try again later.",
            )

        try:
            attempt_port = _ConfiguredAttemptPort(
                experts_by_id=self._bundle.experts_by_id,
                verifier=record.filtered_config.verifier,
                scope_guard=self._scope_guard,
                provider_factory=self._provider_factory,
            )
            outcome = run_local_cascade(
                record.task,
                record.filtered_config,
                attempt_port,
            )
            content = outcome.content or ""
            if len(content) > request.max_output_chars:
                raise CascadeAdapterRejected(
                    "output_limit_exceeded",
                    "The accepted local result exceeded the requested output limit.",
                )
            receipt = outcome.receipt.payload()
            selected_model = None
            if outcome.receipt.selected_tier_id is not None:
                selected_model = next(
                    (
                        tier.model_ref
                        for tier in record.filtered_config.tiers
                        if tier.tier_id == outcome.receipt.selected_tier_id
                    ),
                    None,
                )
            wrapper_base = {
                "schema_version": SCHEMA_VERSION,
                "contract": "LocalCascadeMcpRunReceiptV1",
                "receipt_id": outcome.receipt.run_id,
                "run_id": outcome.receipt.run_id,
                "kind": "run",
                "status": outcome.receipt.status,
                "plan_id": record.plan_id,
                "plan_sha256": record.plan_sha256,
                "request_task_sha256": record.raw_task_sha256,
                "config_sha256": receipt.get("config_sha256"),
                "efficiency_profile": record.profile,
                "finish_reason_policy_bound": True,
                "model": selected_model,
                "tier": outcome.receipt.selected_tier_id,
                "route": [
                    {"tier_id": tier.tier_id, "model_ref": tier.model_ref}
                    for tier in record.filtered_config.tiers
                ],
                "privacy": "metadata_only",
                "core_receipt": receipt,
            }
            stored_receipt = _metadata_receipt_with_evidence(wrapper_base)
            self._store_receipt(outcome.receipt.run_id, stored_receipt)
            reason_codes = (
                []
                if outcome.receipt.status == "passed"
                else list(outcome.receipt.attempts[-1].verifier_reason_codes)
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "status": outcome.receipt.status,
                "content": content,
                "model": selected_model,
                "tier": outcome.receipt.selected_tier_id,
                "usage": outcome.receipt.token_totals.payload(),
                "latency_ms": outcome.receipt.total_duration_ms,
                "reason_codes": reason_codes,
                "receipt": stored_receipt,
            }
        finally:
            self._invocation_lock.release()

    def inspect_receipt(self, receipt_id: str) -> object:
        with self._state_lock:
            encoded = self._receipts.get(receipt_id)
        if encoded is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "not_found",
                "receipt_id": receipt_id,
                "privacy": "metadata_only",
            }
        return json.loads(encoded.decode("utf-8"))

    def _build_provider(self, expert: Any) -> Any:
        from .providers import OpenAICompatibleProvider

        if expert.provider != "openai_compatible":
            raise CascadeAdapterRejected(
                "unsupported_local_provider",
                "Local Cascade alpha requires an OpenAI-compatible local provider.",
            )
        return OpenAICompatibleProvider(opener=self._opener)

    def _store_plan(
        self,
        record: _PlanRecord,
        receipt: Mapping[str, object],
    ) -> None:
        encoded = _metadata_bytes(receipt)
        with self._state_lock:
            self._plans[record.plan_id] = record
            self._plans.move_to_end(record.plan_id)
            self._receipts[record.plan_id] = encoded
            self._receipts.move_to_end(record.plan_id)
            while len(self._plans) > MAX_STORED_PLANS:
                old_plan_id, _ = self._plans.popitem(last=False)
                self._receipts.pop(old_plan_id, None)
            while len(self._receipts) > MAX_STORED_RECEIPTS:
                self._receipts.popitem(last=False)

    def _store_receipt(
        self,
        receipt_id: str,
        receipt: Mapping[str, object],
    ) -> None:
        encoded = _metadata_bytes(receipt)
        with self._state_lock:
            self._receipts[receipt_id] = encoded
            self._receipts.move_to_end(receipt_id)
            while len(self._receipts) > MAX_STORED_RECEIPTS:
                self._receipts.popitem(last=False)


class LocalCascadeToolSurface:
    """Compact MCP-facing projection with no telemetry or implicit side effects."""

    def __init__(self, adapter: CascadeAdapter) -> None:
        self._adapter = adapter

    def machine_inspect(self) -> dict[str, object]:
        """Inspect read-only machine and configured-runtime metadata."""

        try:
            payload = _project_payload(self._adapter.inspect_machine(), _MACHINE_FIELDS)
        except CascadeAdapterUnavailable as exc:
            return _adapter_error(exc)
        except Exception:
            return _error("inspection_failed", "Machine inspection failed.")
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload["scope"] = "local_read_only"
        payload["privacy"] = "metadata_only"
        return payload

    def delegate_plan(
        self,
        task: str,
        task_kind: str = "summarization",
        efficiency_profile: str = "balanced",
        max_steps: int = 4,
        plan_model_assets: bool = True,
    ) -> dict[str, object]:
        """Plan bounded local inference; model installation is plan-only."""

        validation = _validate_task(task)
        if validation is not None:
            return validation
        if task_kind not in TASK_KINDS:
            return _error(
                "unsupported_task_kind",
                "task_kind must be classification, extraction, or summarization.",
            )
        if efficiency_profile not in EFFICIENCY_PROFILES:
            return _error(
                "invalid_efficiency_profile",
                "efficiency_profile must be economy, balanced, or quality.",
            )
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= MAX_STEPS
        ):
            return _error(
                "invalid_max_steps",
                f"max_steps must be between 1 and {MAX_STEPS}.",
            )

        task_digest = _task_sha256(task)
        request = DelegatePlanRequest(
            task=task,
            task_sha256=task_digest,
            task_kind=task_kind,
            efficiency_profile=efficiency_profile,
            max_steps=max_steps,
            installation_mode="plan_only" if plan_model_assets else "disabled",
        )
        try:
            raw = self._adapter.plan(request)
            if _reports_install_side_effect(_payload_mapping(raw)):
                return _error(
                    "installation_side_effect_rejected",
                    "The alpha plugin does not install or download model assets.",
                )
            payload = _project_payload(raw, _PLAN_FIELDS)
        except (CascadeAdapterRejected, CascadeAdapterUnavailable) as exc:
            return _adapter_error(exc)
        except Exception:
            return _error("planning_failed", "Local delegation planning failed.")

        if _reports_install_side_effect(payload):
            return _error(
                "installation_side_effect_rejected",
                "The alpha plugin does not install or download model assets.",
            )
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload["task_sha256"] = task_digest
        payload["installation_mode"] = request.installation_mode
        payload["installation_executed"] = False
        return payload

    def delegate_run(
        self,
        task: str,
        plan_id: str,
        plan_sha256: str,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> dict[str, object]:
        """Run one bound local plan and return content plus a metadata-only receipt."""

        validation = _validate_task(task)
        if validation is not None:
            return validation
        if not isinstance(plan_id, str) or IDENTIFIER_RE.fullmatch(plan_id) is None:
            return _error("invalid_plan_id", "plan_id has an invalid format.")
        if not isinstance(plan_sha256, str) or SHA256_RE.fullmatch(plan_sha256) is None:
            return _error(
                "invalid_plan_sha256",
                "plan_sha256 must be a SHA-256 digest.",
            )
        if (
            not isinstance(max_output_chars, int)
            or isinstance(max_output_chars, bool)
            or not 256 <= max_output_chars <= MAX_OUTPUT_CHARS
        ):
            return _error(
                "invalid_output_limit",
                f"max_output_chars must be between 256 and {MAX_OUTPUT_CHARS}.",
            )

        task_digest = _task_sha256(task)
        request = DelegateRunRequest(
            task=task,
            task_sha256=task_digest,
            plan_id=plan_id,
            plan_sha256=plan_sha256.lower(),
            max_output_chars=max_output_chars,
        )
        try:
            raw = self._adapter.run(request)
            content, metadata = _split_run_outcome(raw)
            if _reports_install_side_effect(metadata):
                return _error(
                    "installation_side_effect_rejected",
                    "The alpha plugin does not install or download model assets.",
                )
            payload = _project_payload(metadata, _RUN_FIELDS)
        except (CascadeAdapterRejected, CascadeAdapterUnavailable) as exc:
            return _adapter_error(exc)
        except Exception:
            return _error("delegation_failed", "Local delegation failed.")

        if content is None:
            return _error(
                "invalid_core_result", "Local delegation returned no content."
            )
        if "receipt" not in payload:
            return _error("missing_receipt", "Local delegation returned no receipt.")
        if _reports_install_side_effect(payload):
            return _error(
                "installation_side_effect_rejected",
                "The alpha plugin does not install or download model assets.",
            )

        if len(content) > max_output_chars:
            return _error(
                "output_limit_exceeded",
                "The accepted local result exceeded the requested output limit.",
            )
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload["content"] = content
        payload["content_truncated"] = False
        payload["task_sha256"] = task_digest
        return payload

    def receipt_inspect(self, receipt_id: str) -> dict[str, object]:
        """Inspect one local-cascade receipt without raw task or result content."""

        if (
            not isinstance(receipt_id, str)
            or IDENTIFIER_RE.fullmatch(receipt_id) is None
        ):
            return _error("invalid_receipt_id", "receipt_id has an invalid format.")
        try:
            payload = _project_payload(
                self._adapter.inspect_receipt(receipt_id),
                _RECEIPT_FIELDS,
            )
        except (CascadeAdapterRejected, CascadeAdapterUnavailable) as exc:
            return _adapter_error(exc)
        except Exception:
            return _error("receipt_inspection_failed", "Receipt inspection failed.")
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload["privacy"] = "metadata_only"
        return payload


def build_server(adapter: CascadeAdapter | None = None) -> Any:
    """Build the official MCP Python SDK v1 FastMCP stdio server."""

    server = _new_fastmcp()
    tools = LocalCascadeToolSurface(adapter or LazyCascadeAdapter())
    server.tool(name="machine_inspect")(tools.machine_inspect)
    server.tool(name="delegate_plan")(tools.delegate_plan)
    server.tool(name="delegate_run")(tools.delegate_run)
    server.tool(name="receipt_inspect")(tools.receipt_inspect)
    return server


def main() -> None:
    """Run the local-only MCP adapter over stdio."""

    build_server().run(transport="stdio")


def _new_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise CascadeAdapterUnavailable(
            "MCP Python SDK v1 is required to run the plugin server."
        ) from exc
    return FastMCP(
        "myMoE Local Cascade",
        instructions=(
            "Delegate bounded text work to configured local models. "
            "Never install model assets, execute external tools, expose secrets, "
            "or return hidden reasoning."
        ),
        json_response=True,
    )


def _load_configuration_bundle(
    environ: Mapping[str, str],
) -> _ConfigurationBundle:
    from .config import parse_config, runtime_config_sha256
    from .execution_scope import ExecutionScopeGuard
    from .local_cascade_contracts import LocalCascadeConfigV1, sha256_json
    from .providers import validate_provider_request_params
    from .secure_files import read_bounded_regular_file

    cascade_path = _required_absolute_config_path(environ, CASCADE_CONFIG_ENV)
    moe_path = _required_absolute_config_path(environ, MOE_CONFIG_ENV)

    def load_payload(path: Path, label: str) -> Mapping[str, object]:
        raw = read_bounded_regular_file(
            path,
            maximum_bytes=2 * 1024 * 1024,
            label=label,
        )
        try:
            parsed = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{label} must be strict JSON.") from exc
        if not isinstance(parsed, Mapping) or any(
            not isinstance(key, str) for key in parsed
        ):
            raise ValueError(f"{label} must be a JSON object.")
        return parsed

    cascade_config = LocalCascadeConfigV1.from_payload(
        load_payload(cascade_path, "local cascade configuration")
    )
    moe_payload = load_payload(moe_path, "local model configuration")
    moe_config = parse_config(dict(moe_payload))
    experts_by_id = moe_config.experts_by_id
    guard = ExecutionScopeGuard(moe_config.execution_policy)
    for tier in cascade_config.tiers:
        expert = experts_by_id.get(tier.model_ref)
        if expert is None:
            raise CascadeAdapterUnavailable(
                "Every cascade tier model_ref must match one configured expert id.",
                code="model_ref_not_configured",
            )
        _require_numeric_device_only_expert(expert, guard)
        _terminal_finish_reasons(expert)
        validate_provider_request_params(expert)
    return _ConfigurationBundle(
        cascade_config=cascade_config,
        moe_config=moe_config,
        experts_by_id=experts_by_id,
        cascade_config_sha256=sha256_json(cascade_config.payload()),
        moe_config_sha256=runtime_config_sha256(moe_config),
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _terminal_finish_reasons(expert: Any) -> tuple[str, ...]:
    params = getattr(expert, "params", None)
    raw = (
        params.get(TERMINAL_FINISH_REASONS_PARAM)
        if isinstance(params, Mapping)
        else None
    )
    if not isinstance(raw, (list, tuple)):
        raise CascadeAdapterRejected(
            "invalid_configuration",
            "The local cascade configuration is invalid.",
        )
    values = tuple(item for item in raw if isinstance(item, str))
    if (
        len(values) != len(raw)
        or not 1 <= len(values) <= 16
        or len(set(values)) != len(values)
        or any(item != item.strip().casefold() for item in values)
        or any(FINISH_REASON_RE.fullmatch(item) is None for item in values)
        or any(item in NON_TERMINAL_FINISH_REASONS for item in values)
    ):
        raise CascadeAdapterRejected(
            "invalid_configuration",
            "The local cascade configuration is invalid.",
        )
    return values


def _required_absolute_config_path(
    environ: Mapping[str, str],
    name: str,
) -> Path:
    raw = environ.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise CascadeAdapterUnavailable(
            f"Set {name} before starting the Local Cascade MCP server.",
            code="configuration_required",
        )
    path = Path(raw.strip())
    if not path.is_absolute():
        raise CascadeAdapterUnavailable(
            f"{name} must contain an absolute file path.",
            code="absolute_configuration_path_required",
        )
    return path


def _require_numeric_device_only_expert(expert: Any, scope_guard: Any) -> None:
    from .execution_scope import ExecutionScope, ExecutionTransport
    from .http_boundary import http_origin

    if expert.provider != "openai_compatible" or not expert.base_url:
        raise CascadeAdapterRejected(
            "unsupported_local_provider",
            "Cascade experts must use an OpenAI-compatible numeric loopback endpoint.",
        )
    origin = http_origin(str(expert.base_url))
    try:
        address = ip_address(origin.host)
    except ValueError as exc:
        raise CascadeAdapterRejected(
            "numeric_loopback_required",
            "Cascade experts must use a numeric loopback host.",
        ) from exc
    mapped = getattr(address, "ipv4_mapped", None)
    if not address.is_loopback and not (mapped and mapped.is_loopback):
        raise CascadeAdapterRejected(
            "numeric_loopback_required",
            "Cascade experts must use a numeric loopback host.",
        )
    attestation = scope_guard.require_allowed(expert.execution_target)
    if (
        attestation.scope != ExecutionScope.DEVICE_ONLY
        or attestation.transport != ExecutionTransport.DIRECT_LOCAL
    ):
        raise CascadeAdapterRejected(
            "device_only_required",
            "Cascade experts must be configured for direct device-only execution.",
        )


def _attempt_prompt(request: Any, verifier: Any) -> str:
    payload = {
        "task_kind": request.task.kind,
        "instruction": request.task.instruction,
        "output_format": request.task.output_format,
        "verification_contract": verifier.payload(),
        "prior_verifier_reason_codes": list(request.verifier_reason_codes),
    }
    return (
        "Complete the bounded task below. Return only the final answer and do not "
        "include hidden reasoning.\n"
        + json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _actual_or_unknown_tokens(value: object, token_type: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return token_type(source="actual", count=value)
    return token_type.unknown()


def _metadata_bytes(receipt: Mapping[str, object]) -> bytes:
    cleaned = _scrub(receipt)
    if not isinstance(cleaned, Mapping):
        raise TypeError("Receipt metadata must be a mapping.")
    return json.dumps(
        cleaned,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _metadata_receipt_with_evidence(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    from .local_cascade_contracts import sha256_json

    cleaned = _scrub(receipt)
    if not isinstance(cleaned, Mapping):
        raise TypeError("Receipt metadata must be a mapping.")
    stable = {str(key): value for key, value in cleaned.items()}
    return {**stable, "evidence_sha256": sha256_json(stable)}


def _adapter_error(
    error: CascadeAdapterRejected | CascadeAdapterUnavailable,
) -> dict[str, object]:
    code = error.code if error.code in _PUBLIC_ERROR_MESSAGES else "adapter_unavailable"
    return _error(code, _PUBLIC_ERROR_MESSAGES[code])


def _validate_task(task: object) -> dict[str, object] | None:
    if not isinstance(task, str) or not task.strip():
        return _error("invalid_task", "task must be non-empty text.")
    if "\x00" in task or len(task) > MAX_TASK_CHARS:
        return _error(
            "invalid_task",
            f"task must contain at most {MAX_TASK_CHARS} characters and no NUL bytes.",
        )
    return None


def _task_sha256(task: str) -> str:
    return sha256(task.encode("utf-8")).hexdigest()


def _split_run_outcome(value: object) -> tuple[str | None, object]:
    if isinstance(value, Mapping):
        content = value.get("content")
        metadata = {key: item for key, item in value.items() if key != "content"}
    else:
        content = getattr(value, "content", None)
        receipt = getattr(value, "receipt", None)
        metadata = {
            "schema_version": getattr(value, "schema_version", SCHEMA_VERSION),
            "status": getattr(value, "status", "complete"),
            "model": getattr(value, "model", None),
            "tier": getattr(value, "tier", None),
            "usage": getattr(value, "usage", None),
            "latency_ms": getattr(value, "latency_ms", None),
            "receipt": receipt,
        }
    return content if isinstance(content, str) else None, metadata


def _project_payload(value: object, fields: tuple[str, ...]) -> dict[str, object]:
    mapping = _payload_mapping(value)
    projected: dict[str, object] = {}
    for field in fields:
        if field not in mapping or mapping[field] is None:
            continue
        cleaned = _scrub(mapping[field])
        if cleaned is not None:
            projected[field] = cleaned
    return projected


def _payload_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    payload = getattr(value, "payload", None)
    if callable(payload):
        rendered = payload()
        if isinstance(rendered, Mapping):
            return {str(key): item for key, item in rendered.items()}
    raise TypeError("Core payload must be a mapping or expose payload().")


def _scrub(value: object, *, depth: int = 0) -> object | None:
    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1_024]
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _SENSITIVE_KEYS:
                continue
            rendered = _scrub(item, depth=depth + 1)
            if rendered is not None:
                cleaned[key] = rendered
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_items = [
            rendered
            for item in value[:32]
            if (rendered := _scrub(item, depth=depth + 1)) is not None
        ]
        return cleaned_items
    payload = getattr(value, "payload", None)
    if callable(payload):
        return _scrub(payload(), depth=depth + 1)
    return None


def _reports_install_side_effect(value: object) -> bool:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if (
                key
                in {
                    "download_performed",
                    "execution_started",
                    "install_performed",
                    "installation_executed",
                }
                and item is True
            ):
                return True
            if _reports_install_side_effect(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_reports_install_side_effect(item) for item in value)
    return False


def _error(code: str, message: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": {"code": code, "message": message},
    }


if __name__ == "__main__":
    main()
