from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Protocol
from urllib import error, request
from urllib.parse import urlsplit

from .adaptive_execution_gate import (
    AdaptiveCellExecutionPreviewReceipt,
    preview_cell_execution,
)
from .adaptive_advisor_cli import ProtectedRootIdentity
from .bound_cell_run_contracts import BoundCellRunPolicy, BoundCellRunReceipt
from .cell_contracts import CellPassport
from .cell_passport import load_cell_catalog
from .config import ExpertConfig, parse_config, runtime_config_sha256
from .execution_scope import ExecutionScope, ExecutionTransport
from .runtime_binding_contracts import BOUND_CELL_ADAPTER_ID
from .runtime_binding_inspector import (
    CellBindingInspectRequest,
    CellBindingInspectionBundle,
    inspect_cell_binding,
    load_cell_binding_inspect_request,
)
from .secure_files import read_bounded_regular_file
from .verified_routing_contracts import CONTRACT_VERSION, sha256_json


MAX_RUNTIME_CONFIG_BYTES = 2 * 1024 * 1024


class BoundCellRunTransportError(RuntimeError):
    def __init__(
        self, code: str, detail: str, *, response_received: bool = False
    ) -> None:
        self.code = code
        self.detail = detail
        self.response_received = response_received
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class ModelIdentityProbe:
    model_ids: tuple[str, ...]
    identity_set_sha256: str

    @classmethod
    def from_ids(
        cls, values: list[str] | tuple[str, ...], *, maximum: int
    ) -> "ModelIdentityProbe":
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise BoundCellRunTransportError(
                "model_probe_failed", "Invalid model limit."
            )
        if not isinstance(values, (list, tuple)) or not values or len(values) > maximum:
            raise BoundCellRunTransportError(
                "model_probe_failed", "Model identity set is invalid."
            )
        if any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in values
        ):
            raise BoundCellRunTransportError(
                "model_probe_failed", "Model identity is invalid."
            )
        ordered = tuple(sorted(values))
        if len(set(ordered)) != len(ordered):
            raise BoundCellRunTransportError(
                "model_probe_failed", "Model identities are duplicated."
            )
        digest = sha256_json(
            {"schema_version": CONTRACT_VERSION, "model_ids": list(ordered)}
        )
        return cls(model_ids=ordered, identity_set_sha256=digest)


class BoundCellRunTransport(Protocol):
    def probe_models(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        maximum_bytes: int,
        maximum_models: int,
    ) -> ModelIdentityProbe: ...

    def invoke(
        self,
        *,
        base_url: str,
        model: str,
        task_text: str,
        timeout_seconds: float,
        maximum_bytes: int,
        max_output_tokens: int,
    ) -> str: ...


class OpenAICompatibleLoopbackTransport:
    """One-attempt, same-origin loopback transport with bounded reads and no proxy."""

    def __init__(self, *, opener: Callable[..., object] | None = None) -> None:
        self._opener = opener or _open_loopback_without_redirects

    def probe_models(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        maximum_bytes: int,
        maximum_models: int,
    ) -> ModelIdentityProbe:
        self._require_loopback(base_url)
        target = base_url.rstrip("/") + "/models"
        raw = self._read(
            request.Request(
                target,
                headers={"Accept": "application/json", "Accept-Encoding": "identity"},
                method="GET",
            ),
            timeout_seconds=timeout_seconds,
            maximum_bytes=maximum_bytes,
            failure_code="model_probe_failed",
        )
        try:
            parsed = _strict_json(raw)
            if not isinstance(parsed, dict):
                raise TypeError
            entries = parsed["data"]
            if not isinstance(entries, list):
                raise TypeError
            ids = [item["id"] for item in entries if isinstance(item, dict)]
            if len(ids) != len(entries):
                raise TypeError
            return ModelIdentityProbe.from_ids(ids, maximum=maximum_models)
        except BoundCellRunTransportError:
            raise
        except (
            KeyError,
            OverflowError,
            RecursionError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise BoundCellRunTransportError(
                "model_probe_failed",
                "Invalid /models response.",
                response_received=True,
            ) from exc

    def invoke(
        self,
        *,
        base_url: str,
        model: str,
        task_text: str,
        timeout_seconds: float,
        maximum_bytes: int,
        max_output_tokens: int,
    ) -> str:
        self._require_loopback(base_url)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task_text}],
            "stream": False,
            "max_tokens": max_output_tokens,
        }
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        raw = self._read(
            request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
                method="POST",
            ),
            timeout_seconds=timeout_seconds,
            maximum_bytes=maximum_bytes,
            failure_code="transport_failed",
        )
        try:
            parsed = _strict_json(raw)
            if not isinstance(parsed, dict):
                raise TypeError
            reported_model = parsed.get("model")
            if reported_model is not None and reported_model != model:
                raise BoundCellRunTransportError(
                    "response_invalid",
                    "Completion reported a different model.",
                    response_received=True,
                )
            choices = parsed["choices"]
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
            ):
                raise TypeError
            if choices[0].get("finish_reason") in {"tool_calls", "function_call"}:
                raise BoundCellRunTransportError(
                    "response_invalid",
                    "Model reported a tool or function-call finish reason.",
                    response_received=True,
                )
            message = choices[0]["message"]
            if not isinstance(message, dict):
                raise TypeError
            tool_calls = message.get("tool_calls")
            if tool_calls not in (None, []):
                raise BoundCellRunTransportError(
                    "response_invalid",
                    "Model attempted a tool call.",
                    response_received=True,
                )
            if message.get("function_call") is not None:
                raise BoundCellRunTransportError(
                    "response_invalid",
                    "Model attempted a function call.",
                    response_received=True,
                )
            content = message["content"]
            if not isinstance(content, str) or not content:
                raise TypeError
            return content
        except BoundCellRunTransportError:
            raise
        except (
            KeyError,
            OverflowError,
            RecursionError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise BoundCellRunTransportError(
                "response_invalid",
                "Invalid chat completion response.",
                response_received=True,
            ) from exc

    def _read(
        self,
        req: request.Request,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        failure_code: str,
    ) -> bytes:
        try:
            with self._opener(req, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                if final_url != req.full_url:
                    raise BoundCellRunTransportError(
                        failure_code,
                        "Loopback response URL changed.",
                        response_received=failure_code == "transport_failed",
                    )
                headers = response.headers
                content_type = str(headers.get("Content-Type", ""))
                if content_type.split(";", 1)[0].strip().lower() != "application/json":
                    raise BoundCellRunTransportError(
                        failure_code,
                        "Loopback response is not application/json.",
                        response_received=failure_code == "transport_failed",
                    )
                content_encoding = str(headers.get("Content-Encoding", "identity"))
                if content_encoding.strip().lower() != "identity":
                    raise BoundCellRunTransportError(
                        failure_code,
                        "Encoded loopback responses are forbidden.",
                        response_received=failure_code == "transport_failed",
                    )
                declared_lengths = _header_values(headers, "Content-Length")
                transfer_encodings = _header_values(headers, "Transfer-Encoding")
                if transfer_encodings:
                    normalized_encodings = tuple(
                        token.strip().lower()
                        for value in transfer_encodings
                        for token in value.split(",")
                        if token.strip()
                    )
                    if declared_lengths or normalized_encodings != ("chunked",):
                        raise BoundCellRunTransportError(
                            failure_code,
                            "Loopback response uses ambiguous transfer framing.",
                            response_received=failure_code == "transport_failed",
                        )
                declared_length: int | None = None
                if declared_lengths:
                    try:
                        lengths = tuple(int(value) for value in declared_lengths)
                    except (TypeError, ValueError) as exc:
                        raise BoundCellRunTransportError(
                            failure_code,
                            "Loopback response Content-Length is invalid.",
                            response_received=failure_code == "transport_failed",
                        ) from exc
                    if (
                        len(set(lengths)) != 1
                        or lengths[0] < 0
                        or lengths[0] > maximum_bytes
                    ):
                        raise BoundCellRunTransportError(
                            failure_code,
                            "Loopback response Content-Length is outside the bound.",
                            response_received=failure_code == "transport_failed",
                        )
                    declared_length = lengths[0]
                raw = response.read(maximum_bytes + 1)
        except BoundCellRunTransportError:
            raise
        except (OSError, TimeoutError, error.URLError, ValueError) as exc:
            raise BoundCellRunTransportError(
                failure_code, "Loopback request failed."
            ) from exc
        if len(raw) > maximum_bytes:
            code = (
                "response_too_large"
                if failure_code == "transport_failed"
                else failure_code
            )
            raise BoundCellRunTransportError(
                code,
                "Loopback response exceeded its byte bound.",
                response_received=True,
            )
        if declared_length is not None and len(raw) != declared_length:
            raise BoundCellRunTransportError(
                failure_code,
                "Loopback response length did not match Content-Length.",
                response_received=failure_code == "transport_failed",
            )
        return raw

    @staticmethod
    def _require_loopback(base_url: str) -> None:
        if not _is_explicit_loopback_http_url(base_url):
            raise BoundCellRunTransportError(
                "endpoint_not_loopback",
                "Endpoint must use an explicit numeric loopback HTTP authority.",
            )


@dataclass(frozen=True)
class BoundCellResolvedTarget:
    request: CellBindingInspectRequest
    passport: CellPassport
    expert: ExpertConfig
    config_source_sha256: str
    runtime_config_sha256: str


@dataclass(frozen=True)
class BoundCellRunResult:
    receipt: BoundCellRunReceipt
    response_text: str | None = None
    publication_inputs: tuple[Path, ...] = field(default=(), repr=False, compare=False)
    publication_protected_roots: tuple[ProtectedRootIdentity, ...] = field(
        default=(), repr=False, compare=False
    )
    interruption: BaseException | None = field(default=None, repr=False, compare=False)


Previewer = Callable[..., AdaptiveCellExecutionPreviewReceipt]
Inspector = Callable[..., CellBindingInspectionBundle]
Resolver = Callable[[str | Path], BoundCellResolvedTarget]


def resolve_bound_cell_target(
    binding_request_path: str | Path,
) -> BoundCellResolvedTarget:
    request_path = Path(os.path.abspath(os.fspath(binding_request_path)))
    binding_request = load_cell_binding_inspect_request(request_path)
    root = request_path.parent
    config_path = root / binding_request.runtime_config_path
    raw_bytes = read_bounded_regular_file(
        config_path,
        root=root,
        maximum_bytes=MAX_RUNTIME_CONFIG_BYTES,
        label="bound cell runtime configuration",
    )
    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(raw, dict):
            raise TypeError
        config = parse_config(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BoundCellRunTransportError(
            "config_changed", "Runtime config is invalid."
        ) from exc
    experts = [item for item in config.experts if item.id == binding_request.expert_id]
    if len(experts) != 1:
        raise BoundCellRunTransportError(
            "expert_mismatch", "Expert is not uniquely configured."
        )
    catalog = load_cell_catalog(
        root / binding_request.catalog_path, confinement_root=root
    )
    passports = [
        item for item in catalog.cells if item.cell_id == binding_request.cell_id
    ]
    if len(passports) != 1:
        raise BoundCellRunTransportError(
            "selected_cell_mismatch", "Cell is not uniquely declared."
        )
    return BoundCellResolvedTarget(
        request=binding_request,
        passport=passports[0],
        expert=experts[0],
        config_source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        runtime_config_sha256=runtime_config_sha256(config),
    )


def run_bound_cell(
    source_advisor_receipt_path: str | Path,
    task_text: str,
    catalog_path: str | Path,
    evaluation_contract_path: str | Path,
    adaptive_policy_path: str | Path,
    binding_request_path: str | Path,
    *,
    confirmed: bool,
    policy: BoundCellRunPolicy | None = None,
    transport: BoundCellRunTransport | None = None,
    previewer: Previewer = preview_cell_execution,
    inspector: Inspector = inspect_cell_binding,
    resolver: Resolver = resolve_bound_cell_target,
    clock: Callable[[], datetime] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    publication_path: str | Path | None = None,
) -> BoundCellRunResult:
    """Run one explicitly confirmed, already-resident, evidence-bound cell once.

    This proves file/config/catalog continuity around the call. It does not attest
    the identity of the process listening on the loopback endpoint.
    """

    active_policy = policy or BoundCellRunPolicy()
    active_transport = transport or OpenAICompatibleLoopbackTransport()
    active_clock = clock or (lambda: datetime.now(timezone.utc))
    active_monotonic_clock = monotonic_clock or time.monotonic
    started_time = active_clock()
    started = _utc(started_time)
    started_tick = _monotonic_seconds(active_monotonic_clock())
    try:
        task_bytes_raw = (
            task_text.encode("utf-8") if isinstance(task_text, str) else b""
        )
    except UnicodeEncodeError:
        task_bytes_raw = b""
    publication_inputs = _base_publication_inputs(
        source_advisor_receipt_path,
        catalog_path,
        evaluation_contract_path,
        adaptive_policy_path,
        binding_request_path,
    )
    publication_roots: tuple[ProtectedRootIdentity, ...] = ()
    pending_interruption: BaseException | None = None
    state: dict[str, object] = {
        "policy_sha256": active_policy.digest,
        "started_at": started,
        "confirmed": confirmed if type(confirmed) is bool else False,
        "task_sha256": hashlib.sha256(task_bytes_raw).hexdigest(),
        "task_bytes": len(task_bytes_raw),
        "invocation_attempts": 0,
        "endpoint_probe_requests": 0,
        "delivery_status": "not_attempted",
    }

    def finish(
        status: str, reasons: set[str], response: str | None = None
    ) -> BoundCellRunResult:
        nonlocal pending_interruption
        response_bytes = response.encode("utf-8") if response is not None else None
        final_status = status
        final_reasons = set(reasons)
        try:
            completed_time = active_clock()
            _utc(completed_time)
            if completed_time < started_time:
                raise BoundCellRunTransportError(
                    "clock_invalid", "Clock moved backwards during the run."
                )
        except (KeyboardInterrupt, SystemExit) as exc:
            if pending_interruption is None:
                pending_interruption = exc
            completed_time = started_time
            final_reasons.update({"clock_invalid", "execution_interrupted"})
        except Exception:
            completed_time = started_time
            final_reasons.add("clock_invalid")
        try:
            completed_tick = _monotonic_seconds(active_monotonic_clock())
            if completed_tick < started_tick:
                raise ValueError("monotonic clock moved backwards")
            elapsed_ms = int((completed_tick - started_tick) * 1000)
        except (KeyboardInterrupt, SystemExit) as exc:
            if pending_interruption is None:
                pending_interruption = exc
            elapsed_ms = 0
            final_reasons.update({"clock_invalid", "execution_interrupted"})
        except Exception:
            elapsed_ms = 0
            final_reasons.add("clock_invalid")
        if final_reasons and final_status == "completed":
            final_status = "invalidated"
        receipt = BoundCellRunReceipt(
            **state,
            status=final_status,
            reason_codes=tuple(sorted(final_reasons)),
            completed_at=_utc(completed_time),
            response_sha256=(
                hashlib.sha256(response_bytes).hexdigest()
                if response_bytes is not None
                else None
            ),
            response_bytes=(
                len(response_bytes) if response_bytes is not None else None
            ),
            response_chars=(len(response) if response is not None else None),
            elapsed_ms=elapsed_ms,
        )
        return BoundCellRunResult(
            receipt=receipt,
            response_text=(
                response if final_status in {"completed", "invalidated"} else None
            ),
            publication_inputs=publication_inputs,
            publication_protected_roots=publication_roots,
            interruption=pending_interruption,
        )

    if confirmed is not True:
        return finish("blocked", {"confirmation_required"})
    if (
        not isinstance(task_text, str)
        or not task_text
        or not task_bytes_raw
        or len(task_bytes_raw) > active_policy.max_task_bytes
    ):
        return finish("blocked", {"task_invalid"})

    try:
        target = resolver(binding_request_path)
        pre = inspector(
            binding_request_path,
            now=active_clock(),
            publication_path=publication_path,
        )
        publication_roots = pre.publication_protected_roots
        _record_binding(state, pre, prefix="pre")
        _validate_binding(target, pre)
        publication_inputs = _merge_publication_inputs(
            publication_inputs,
            _binding_publication_inputs(binding_request_path, target.request),
        )
        state["declaration_sha256"] = target.passport.declaration.digest
        state["expert_id"] = target.expert.id
        state["pre_config_source_sha256"] = target.config_source_sha256
    except BoundCellRunTransportError as exc:
        return finish(
            "blocked", {_stable_reason(exc.code, "binding_inspection_failed")}
        )
    except Exception:
        return finish("blocked", {"binding_inspection_failed"})

    try:
        current = resolver(binding_request_path)
        if (
            current.request.digest != target.request.digest
            or current.config_source_sha256 != target.config_source_sha256
            or current.runtime_config_sha256 != target.runtime_config_sha256
            or current.passport.digest != target.passport.digest
            or current.expert != target.expert
        ):
            return finish("blocked", {"config_changed"})
    except BoundCellRunTransportError as exc:
        return finish(
            "blocked", {_stable_reason(exc.code, "binding_inspection_failed")}
        )
    except Exception:
        return finish("blocked", {"binding_inspection_failed"})

    # The adaptive preview is deliberately the last admission step before any
    # endpoint traffic; its read-only contract is not itself authorization.
    try:
        preview = previewer(
            source_advisor_receipt_path,
            task_text,
            catalog_path,
            evaluation_contract_path,
            adaptive_policy_path,
        )
        if not isinstance(preview, AdaptiveCellExecutionPreviewReceipt):
            raise TypeError
        state["preview_sha256"] = preview.digest
        state["selected_cell_id"] = preview.fresh_selected_cell_id
        state["passport_sha256"] = preview.fresh_passport_sha256
        if preview.status != "admission_passed":
            return finish("blocked", {"adaptive_admission_blocked"})
        if preview.task_chars != len(task_text):
            return finish("blocked", {"adaptive_preview_invalid"})
        _validate_preview_binding(preview, target, pre)
    except BoundCellRunTransportError as exc:
        return finish("blocked", {_stable_reason(exc.code, "adaptive_preview_invalid")})
    except Exception:
        return finish("blocked", {"adaptive_preview_invalid"})

    try:
        state["endpoint_probe_requests"] = 1
        probe_pre = active_transport.probe_models(
            base_url=_base_url(target.expert),
            timeout_seconds=active_policy.timeout_seconds,
            maximum_bytes=active_policy.max_probe_bytes,
            maximum_models=active_policy.max_models,
        )
        state["pre_model_identity_set_sha256"] = probe_pre.identity_set_sha256
        if target.expert.model not in probe_pre.model_ids:
            return finish("blocked", {"expected_model_missing"})
    except BoundCellRunTransportError as exc:
        return finish("blocked", {_stable_reason(exc.code, "model_probe_failed")})
    except Exception:
        return finish("blocked", {"model_probe_failed"})

    response: str | None = None
    transport_reason: str | None = None
    state["invocation_attempts"] = 1
    state["delivery_status"] = "attempted_unknown"
    try:
        response = active_transport.invoke(
            base_url=_base_url(target.expert),
            model=target.expert.model,
            task_text=task_text,
            timeout_seconds=active_policy.timeout_seconds,
            maximum_bytes=active_policy.max_response_bytes,
            max_output_tokens=active_policy.max_output_tokens,
        )
        if not isinstance(response, str) or not response:
            raise BoundCellRunTransportError(
                "response_invalid",
                "Transport returned no text.",
                response_received=True,
            )
        try:
            encoded_response = response.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BoundCellRunTransportError(
                "response_invalid",
                "Transport returned text that is not valid UTF-8.",
                response_received=True,
            ) from exc
        if len(encoded_response) > active_policy.max_response_bytes:
            raise BoundCellRunTransportError(
                "response_too_large",
                "Transport returned text beyond the response byte bound.",
                response_received=True,
            )
        state["delivery_status"] = "response_received"
    except BoundCellRunTransportError as exc:
        response = None
        transport_reason = _stable_reason(exc.code, "transport_failed")
        if exc.response_received:
            state["delivery_status"] = "response_received"
    except Exception:
        response = None
        transport_reason = "transport_failed"
    except (KeyboardInterrupt, SystemExit) as exc:
        pending_interruption = exc
        transport_reason = "execution_interrupted"

    invalidation: set[str] = set()
    try:
        state["endpoint_probe_requests"] = 2
        probe_post = active_transport.probe_models(
            base_url=_base_url(target.expert),
            timeout_seconds=active_policy.timeout_seconds,
            maximum_bytes=active_policy.max_probe_bytes,
            maximum_models=active_policy.max_models,
        )
        state["post_model_identity_set_sha256"] = probe_post.identity_set_sha256
        if target.expert.model not in probe_post.model_ids:
            invalidation.add("expected_model_missing")
        if probe_post.identity_set_sha256 != probe_pre.identity_set_sha256:
            invalidation.add("model_identity_changed")
    except (KeyboardInterrupt, SystemExit) as exc:
        if pending_interruption is None:
            pending_interruption = exc
        invalidation.update({"execution_interrupted", "model_probe_failed"})
    except Exception:
        invalidation.add("model_probe_failed")

    try:
        post = inspector(
            binding_request_path,
            now=active_clock(),
            publication_path=publication_path,
        )
        _record_binding(state, post, prefix="post")
        publication_roots = _merge_protected_roots(
            publication_roots, post.publication_protected_roots
        )
        after = resolver(binding_request_path)
        state["post_config_source_sha256"] = after.config_source_sha256
        _validate_binding(after, post)
        _validate_preview_binding(preview, after, post)
        if after.config_source_sha256 != target.config_source_sha256:
            invalidation.add("config_changed")
        if (
            post.manifest.digest != pre.manifest.digest
            or post.request_sha256 != pre.request_sha256
        ):
            invalidation.add("binding_changed")
        if post.publication_protected_roots != pre.publication_protected_roots:
            invalidation.add("binding_changed")
        publication_inputs = _merge_publication_inputs(
            publication_inputs,
            _binding_publication_inputs(binding_request_path, after.request),
        )
    except BoundCellRunTransportError as exc:
        invalidation.add(_stable_reason(exc.code, "post_binding_inspection_failed"))
    except (KeyboardInterrupt, SystemExit) as exc:
        if pending_interruption is None:
            pending_interruption = exc
        invalidation.update({"execution_interrupted", "post_binding_inspection_failed"})
    except Exception:
        invalidation.add("post_binding_inspection_failed")

    if invalidation:
        if transport_reason is not None:
            invalidation.add(transport_reason)
        return finish("invalidated", invalidation, response=response)
    if transport_reason is not None:
        return finish("failed", {transport_reason})
    return finish("completed", set(), response=response)


def _validate_binding(
    target: BoundCellResolvedTarget,
    bundle: CellBindingInspectionBundle,
) -> None:
    manifest = bundle.manifest
    if bundle.request_sha256 != target.request.digest:
        raise BoundCellRunTransportError(
            "binding_changed", "Inspection request does not match the resolved target."
        )
    if bundle.receipt.status != "verified":
        raise BoundCellRunTransportError(
            "binding_not_verified", "Inspection did not verify."
        )
    if manifest.cell_id != target.request.cell_id:
        raise BoundCellRunTransportError(
            "selected_cell_mismatch", "Selected cell does not match binding."
        )
    if manifest.declaration_sha256 != target.passport.declaration.digest:
        raise BoundCellRunTransportError(
            "declaration_mismatch", "Declaration does not match binding."
        )
    if (
        manifest.expert_id != target.request.expert_id
        or target.expert.id != target.request.expert_id
    ):
        raise BoundCellRunTransportError(
            "expert_mismatch", "Expert does not match binding."
        )
    if manifest.config_source_sha256 != target.config_source_sha256:
        raise BoundCellRunTransportError(
            "config_changed", "Config bytes do not match inspection."
        )
    if manifest.runtime_config_sha256 != target.runtime_config_sha256:
        raise BoundCellRunTransportError(
            "runtime_config_mismatch", "Effective config does not match inspection."
        )
    declaration = target.passport.declaration
    if declaration.risk_classes != ("compute_only",):
        raise BoundCellRunTransportError(
            "risk_class_blocked", "Cell exceeds compute-only."
        )
    if declaration.tool_surfaces:
        raise BoundCellRunTransportError("tool_surface_blocked", "Cell declares tools.")
    if manifest.adapter_id != BOUND_CELL_ADAPTER_ID:
        raise BoundCellRunTransportError(
            "transport_blocked", "Binding adapter is unsupported."
        )
    if (
        target.expert.provider != "openai_compatible"
        or target.expert.execution.scope != ExecutionScope.DEVICE_ONLY
        or target.expert.execution.transport != ExecutionTransport.DIRECT_LOCAL
    ):
        raise BoundCellRunTransportError(
            "transport_blocked", "Expert exceeds direct-local scope."
        )
    _base_url(target.expert)


def _validate_preview_binding(
    preview: AdaptiveCellExecutionPreviewReceipt,
    target: BoundCellResolvedTarget,
    bundle: CellBindingInspectionBundle,
) -> None:
    if preview.fresh_selected_cell_id != target.request.cell_id:
        raise BoundCellRunTransportError(
            "selected_cell_mismatch", "Adaptive cell does not match binding."
        )
    if preview.fresh_passport_sha256 != target.passport.digest:
        raise BoundCellRunTransportError(
            "passport_mismatch", "Adaptive passport does not match binding catalog."
        )
    if bundle.manifest.declaration_sha256 != target.passport.declaration.digest:
        raise BoundCellRunTransportError(
            "declaration_mismatch", "Adaptive declaration does not match binding."
        )


def _record_binding(
    state: dict[str, object], bundle: CellBindingInspectionBundle, *, prefix: str
) -> None:
    if not isinstance(bundle, CellBindingInspectionBundle):
        raise BoundCellRunTransportError(
            "binding_inspection_failed", "Inspection returned invalid evidence."
        )
    state[f"{prefix}_binding_bundle_sha256"] = bundle.digest
    state[f"{prefix}_binding_request_sha256"] = bundle.request_sha256
    state[f"{prefix}_binding_manifest_sha256"] = bundle.manifest.digest
    state[f"{prefix}_inspection_receipt_sha256"] = bundle.receipt.digest


def _base_publication_inputs(*values: str | Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in values:
        try:
            paths.append(Path(os.path.abspath(os.fspath(value))))
        except (OSError, OverflowError, TypeError, ValueError):
            continue
    return _merge_publication_inputs((), paths)


def _binding_publication_inputs(
    binding_request_path: str | Path,
    binding_request: CellBindingInspectRequest,
) -> tuple[Path, ...]:
    request_path = Path(os.path.abspath(os.fspath(binding_request_path)))
    root = request_path.parent
    return (
        request_path,
        root / binding_request.catalog_path,
        root / binding_request.runtime_config_path,
    )


def _merge_publication_inputs(
    first: tuple[Path, ...],
    second: tuple[Path, ...] | list[Path],
) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in (*first, *second):
        absolute = Path(os.path.abspath(os.fspath(path)))
        key = os.path.normcase(os.fspath(absolute))
        if key not in seen:
            result.append(absolute)
            seen.add(key)
    return tuple(result)


def _merge_protected_roots(
    first: tuple[ProtectedRootIdentity, ...],
    second: tuple[ProtectedRootIdentity, ...],
) -> tuple[ProtectedRootIdentity, ...]:
    result: list[ProtectedRootIdentity] = []
    for identity in (*first, *second):
        if identity not in result:
            result.append(identity)
    return tuple(result)


def _base_url(expert: ExpertConfig) -> str:
    if not isinstance(expert.base_url, str) or not _is_explicit_loopback_http_url(
        expert.base_url
    ):
        raise BoundCellRunTransportError(
            "endpoint_not_loopback",
            "Expert endpoint must use an explicit numeric loopback HTTP authority.",
        )
    return expert.base_url


def _stable_reason(code: str, fallback: str) -> str:
    allowed = {
        "binding_changed",
        "binding_inspection_failed",
        "binding_not_verified",
        "config_changed",
        "declaration_mismatch",
        "endpoint_not_loopback",
        "expert_mismatch",
        "expected_model_missing",
        "model_identity_changed",
        "execution_interrupted",
        "model_probe_failed",
        "passport_mismatch",
        "post_binding_inspection_failed",
        "response_invalid",
        "response_too_large",
        "risk_class_blocked",
        "runtime_config_mismatch",
        "selected_cell_mismatch",
        "tool_surface_blocked",
        "transport_blocked",
        "transport_failed",
    }
    return code if code in allowed else fallback


def _utc(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise BoundCellRunTransportError(
            "transport_failed", "Clock must return UTC datetimes."
        )
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _monotonic_seconds(value: object) -> float:
    if isinstance(value, bool):
        raise BoundCellRunTransportError(
            "clock_invalid", "Monotonic clock must return finite seconds."
        )
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundCellRunTransportError(
            "clock_invalid", "Monotonic clock must return finite seconds."
        ) from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise BoundCellRunTransportError(
            "clock_invalid", "Monotonic clock must return finite seconds."
        )
    return seconds


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Unsupported JSON constant: {value}")


def _strict_json(raw: bytes) -> object:
    parsed = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    _validate_json_tree(parsed)
    return parsed


def _validate_json_tree(
    value: object, *, maximum_depth: int = 64, maximum_nodes: int = 100_000
) -> None:
    remaining = maximum_nodes

    def visit(item: object, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > maximum_depth:
            raise ValueError("JSON structure exceeds its validation bound.")
        if isinstance(item, str):
            item.encode("utf-8")
            return
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON number must be finite.")
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object key must be text.")
                key.encode("utf-8")
                visit(child, depth + 1)
            return
        raise ValueError("Unsupported JSON value.")

    visit(value, 0)


def _header_values(headers: object, name: str) -> tuple[str, ...]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name, [])
        return tuple(str(value).strip() for value in values)
    get = getattr(headers, "get", None)
    if not callable(get):
        return ()
    value = get(name)
    return () if value is None else (str(value).strip(),)


def _is_explicit_loopback_http_url(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.lower() != "http"
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/", "/v1", "/v1/"}
    ):
        return False
    try:
        address = ip_address(parsed.hostname)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_loopback or (mapped and mapped.is_loopback))


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _open_loopback_without_redirects(target: request.Request, *, timeout: float):
    target_url = target.full_url
    try:
        parsed = urlsplit(target_url)
        path = parsed.path
        for suffix in ("/chat/completions", "/models"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        else:
            raise ValueError("unsupported API path")
        base_url = parsed._replace(path=path, query="", fragment="").geturl()
    except (AttributeError, TypeError, ValueError) as exc:
        raise BoundCellRunTransportError(
            "endpoint_not_loopback", "Endpoint uses an unsupported API path."
        ) from exc
    if not _is_explicit_loopback_http_url(base_url):
        raise BoundCellRunTransportError(
            "endpoint_not_loopback", "Endpoint is not loopback."
        )
    opener = request.build_opener(request.ProxyHandler({}), _NoRedirectHandler())
    return opener.open(target, timeout=timeout)


__all__ = [
    "BoundCellResolvedTarget",
    "BoundCellRunResult",
    "BoundCellRunTransport",
    "BoundCellRunTransportError",
    "ModelIdentityProbe",
    "OpenAICompatibleLoopbackTransport",
    "resolve_bound_cell_target",
    "run_bound_cell",
]
