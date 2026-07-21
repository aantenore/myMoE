from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


BOUND_CELL_INSPECTOR_CONTRACT = "BoundCellInspector"
BOUND_CELL_ADAPTER_ID = "managed_direct_local_openai_v1"
BOUND_CELL_EXECUTION_SCOPE = "device_only"
BOUND_CELL_TRANSPORT = "direct_local"
BOUND_CELL_RUNTIME_BACKENDS = frozenset({"llama_cpp", "mlx_lm", "mlx_vlm"})
BOUND_CELL_RUNTIME_COMPONENT_ROLES = frozenset(
    {"driver", "harness", "runtime_executable"}
)
BOUND_CELL_ENDPOINT_SCHEME = "http"
BOUND_CELL_ENDPOINT_PATHS = frozenset({"", "/", "/v1", "/v1/"})
BOUND_CELL_RISK_CLASSES = ("compute_only",)
BOUND_CELL_FORBIDDEN_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "download",
        "download.exe",
        "fish",
        "ollama",
        "ollama.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sh",
        "zsh",
    }
)
MAX_INSPECTION_TTL_SECONDS = 120
MODEL_IDENTITY_STATUSES = frozenset({"verified"})
MODEL_ARTIFACT_KINDS = frozenset({"directory", "file"})
INSPECTION_STATUSES = frozenset({"abstained", "verified"})
_REQUIRED_RUNTIME_ROLES = BOUND_CELL_RUNTIME_COMPONENT_ROLES
_ALLOWED_COMPONENT_ROLES = _REQUIRED_RUNTIME_ROLES | {"model_artifact"}
INSPECTION_REASON_CODES = frozenset(
    {
        "harness_identity_mismatch",
        "harness_identity_unknown",
        "model_identity_mismatch",
        "model_identity_unknown",
        "runtime_identity_mismatch",
        "runtime_identity_unknown",
        "tool_contract_identity_mismatch",
        "tool_contract_identity_unknown",
    }
)
_MODEL_COMPONENT_PATH = re.compile(r"\Aartifact-[0-9a-f]{64}(?:\.gguf)?\Z")


def bound_cell_adapter_contract_payload() -> dict[str, object]:
    """Return the complete v1 semantics committed by the adapter digest."""

    return {
        "schema_version": CONTRACT_VERSION,
        "adapter_id": BOUND_CELL_ADAPTER_ID,
        "provider": "openai_compatible",
        "execution_policy": {
            "max_scope": BOUND_CELL_EXECUTION_SCOPE,
            "allowed_scopes": [BOUND_CELL_EXECUTION_SCOPE],
            "allow_scope_widening": False,
        },
        "expert_execution": {
            "scope": BOUND_CELL_EXECUTION_SCOPE,
            "transport": BOUND_CELL_TRANSPORT,
        },
        "runtime_backends": sorted(BOUND_CELL_RUNTIME_BACKENDS),
        "runtime_model_source": "local",
        "runtime_components": sorted(BOUND_CELL_RUNTIME_COMPONENT_ROLES),
        "runtime_component_locations_unique": True,
        "runtime_executable_binds_launch_argv": True,
        "launch_working_directory": "inspection_request_root",
        "endpoint": {
            "scheme": BOUND_CELL_ENDPOINT_SCHEME,
            "loopback_only": True,
            "explicit_port": True,
            "credentials": False,
            "query": False,
            "fragment": False,
            "allowed_paths": sorted(BOUND_CELL_ENDPOINT_PATHS),
            "binds_launch_host_and_port": True,
        },
        "model_artifacts": {
            "llama_cpp": "single_local_gguf_file",
            "mlx_lm": "local_directory",
            "mlx_vlm": "local_directory",
            "artifact_roots": "real_directories",
            "public_paths": "deterministic_sha256_pseudonyms",
            "public_manifest_recomputable": True,
            "remote_fetch_arguments": False,
        },
        "cell_declaration": {
            "offline_capable": True,
            "risk_classes": list(BOUND_CELL_RISK_CLASSES),
            "tool_surfaces": [],
        },
        "forbidden_launch_executables": sorted(BOUND_CELL_FORBIDDEN_EXECUTABLES),
        "network_used": False,
        "processes_started": 0,
        "process_mutations": False,
        "model_invocations": 0,
        "authorizes_execution": False,
    }


def _empty_tool_contract_sha256() -> str:
    return sha256_json({"schema_version": CONTRACT_VERSION, "tool_surfaces": []})


BOUND_CELL_ADAPTER_CONTRACT_SHA256 = sha256_json(bound_cell_adapter_contract_payload())
EMPTY_TOOL_CONTRACT_SHA256 = _empty_tool_contract_sha256()


class RuntimeBindingContractError(VerifiedRoutingError):
    """Raised when runtime-binding evidence violates its public contract."""


def _call(function, *args):
    try:
        return function(*args)
    except RuntimeBindingContractError:
        raise
    except (VerifiedRoutingError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingContractError(str(exc)) from exc


def _safe(value: object, label: str) -> str:
    return _call(require_safe_id, value, label)


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _integer(value: object, label: str) -> int:
    return _call(require_non_negative_int, value, label)


def _positive(value: object, label: str) -> int:
    rendered = _integer(value, label)
    if rendered < 1:
        raise RuntimeBindingContractError(f"{label} must be positive.")
    return rendered


def _timestamp(value: object, label: str) -> str:
    return _call(require_utc_timestamp, value, label)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeBindingContractError(f"{label} must be a boolean.")
    return value


def _enum(value: object, allowed: frozenset[str], label: str) -> str:
    rendered = str(value or "")
    if rendered not in allowed:
        raise RuntimeBindingContractError(f"{label} is not supported.")
    return rendered


def _relative_path(value: object, label: str) -> str:
    rendered = str(value or "")
    if not rendered or "\\" in rendered:
        raise RuntimeBindingContractError(f"{label} must be a relative POSIX path.")
    path = PurePosixPath(rendered)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeBindingContractError(f"{label} must stay below its logical root.")
    return path.as_posix()


def _digest(value: object, content: dict[str, object], label: str) -> str:
    try:
        expected = sha256_json(content)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise RuntimeBindingContractError(
            f"Unable to canonicalize {label} content."
        ) from exc
    if value not in (None, "") and _sha(value, label) != expected:
        raise RuntimeBindingContractError(f"{label} does not match its content.")
    return expected


def _schema(value: object, label: str) -> None:
    if value != CONTRACT_VERSION:
        raise RuntimeBindingContractError(f"Unsupported {label} schema_version.")


@dataclass(frozen=True)
class RuntimeComponentEvidence:
    role: str
    root_id: str
    path: str
    size_bytes: int
    sha256: str
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime component")
        object.__setattr__(self, "role", _safe(self.role, "component role"))
        object.__setattr__(self, "root_id", _safe(self.root_id, "component root_id"))
        object.__setattr__(self, "path", _relative_path(self.path, "component path"))
        object.__setattr__(self, "size_bytes", _positive(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "sha256", _sha(self.sha256, "component sha256"))
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "component digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "root_id": self.root_id,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def model_artifact_evidence_sha256(
    kind: object,
    components: tuple[RuntimeComponentEvidence, ...] | list[RuntimeComponentEvidence],
) -> str:
    """Derive the public model identity from its path-redacted evidence."""

    artifact_kind = _enum(kind, MODEL_ARTIFACT_KINDS, "model_artifact_kind")
    items = tuple(components) if isinstance(components, (tuple, list)) else ()
    if not items or any(
        not isinstance(item, RuntimeComponentEvidence)
        or item.role != "model_artifact"
        or item.root_id != "model"
        for item in items
    ):
        raise RuntimeBindingContractError(
            "Model artifact evidence must contain model-root components."
        )
    ordered = tuple(sorted(items, key=lambda item: item.path))
    if len({item.path for item in ordered}) != len(ordered):
        raise RuntimeBindingContractError(
            "Model artifact evidence paths must be unique."
        )
    if artifact_kind == "file":
        if len(ordered) != 1 or not ordered[0].path.endswith(".gguf"):
            raise RuntimeBindingContractError(
                "File model evidence must contain one pseudonymous GGUF component."
            )
    if any(
        _MODEL_COMPONENT_PATH.fullmatch(item.path) is None
        or (artifact_kind == "directory" and item.path.endswith(".gguf"))
        for item in ordered
    ):
        raise RuntimeBindingContractError(
            "Model artifact evidence paths must use the v1 pseudonymous form."
        )
    return sha256_json(
        {
            "schema_version": CONTRACT_VERSION,
            "kind": artifact_kind,
            "entries": [
                {
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in ordered
            ],
            "file_count": len(ordered),
            "total_bytes": sum(item.size_bytes for item in ordered),
        }
    )


@dataclass(frozen=True)
class CellRuntimeBindingManifest:
    cell_id: str
    declaration_sha256: str
    config_source_sha256: str
    runtime_config_sha256: str
    expert_id: str
    expert_config_sha256: str
    adapter_id: str
    adapter_contract_sha256: str
    platform_key: str
    components: tuple[RuntimeComponentEvidence, ...]
    launch_plan_sha256: str
    endpoint_authority_sha256: str
    model_reference_sha256: str
    model_artifact_kind: str
    model_artifact_manifest_sha256: str | None
    model_identity_sha256: str
    runtime_identity_sha256: str
    harness_identity_sha256: str
    tool_contract_identity_sha256: str
    model_identity_status: str
    execution_scope: str
    transport: str
    producer_id: str
    producer_version: str
    producer_code_sha256: str
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime binding manifest")
        for name in (
            "cell_id",
            "expert_id",
            "adapter_id",
            "platform_key",
            "execution_scope",
            "transport",
            "producer_id",
            "producer_version",
        ):
            object.__setattr__(self, name, _safe(getattr(self, name), name))
        for name in (
            "declaration_sha256",
            "config_source_sha256",
            "runtime_config_sha256",
            "expert_config_sha256",
            "adapter_contract_sha256",
            "launch_plan_sha256",
            "endpoint_authority_sha256",
            "model_reference_sha256",
            "model_identity_sha256",
            "runtime_identity_sha256",
            "harness_identity_sha256",
            "tool_contract_identity_sha256",
            "producer_code_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(
            self,
            "model_artifact_manifest_sha256",
            _optional_sha(
                self.model_artifact_manifest_sha256,
                "model_artifact_manifest_sha256",
            ),
        )
        status = _enum(
            self.model_identity_status,
            MODEL_IDENTITY_STATUSES,
            "model_identity_status",
        )
        object.__setattr__(self, "model_identity_status", status)
        artifact_kind = _enum(
            self.model_artifact_kind,
            MODEL_ARTIFACT_KINDS,
            "model_artifact_kind",
        )
        object.__setattr__(self, "model_artifact_kind", artifact_kind)
        components = (
            tuple(self.components) if isinstance(self.components, (tuple, list)) else ()
        )
        if not components or any(
            not isinstance(item, RuntimeComponentEvidence) for item in components
        ):
            raise RuntimeBindingContractError(
                "components must contain runtime component evidence."
            )
        expected = tuple(
            sorted(components, key=lambda item: (item.role, item.root_id, item.path))
        )
        if components != expected:
            raise RuntimeBindingContractError(
                "components must be sorted by role, root_id, and path."
            )
        locations = [(item.root_id, item.path) for item in components]
        if len(locations) != len(set(locations)):
            raise RuntimeBindingContractError("component locations must be unique.")
        roles = {item.role for item in components}
        unsupported_roles = roles - _ALLOWED_COMPONENT_ROLES
        if unsupported_roles:
            raise RuntimeBindingContractError(
                "components contain unsupported roles: "
                + ", ".join(sorted(unsupported_roles))
                + "."
            )
        by_runtime_role = {
            role: [item for item in components if item.role == role]
            for role in _REQUIRED_RUNTIME_ROLES
        }
        invalid_runtime_roles = [
            role
            for role, items in by_runtime_role.items()
            if len(items) != 1 or items[0].root_id != "runtime"
        ]
        if invalid_runtime_roles:
            raise RuntimeBindingContractError(
                "components require exactly one runtime-root item for roles: "
                + ", ".join(sorted(invalid_runtime_roles))
                + "."
            )
        model_components = [
            item for item in components if item.role == "model_artifact"
        ]
        if any(item.root_id != "model" for item in model_components):
            raise RuntimeBindingContractError(
                "model artifacts must use the model logical root."
            )
        artifact_digest = self.model_artifact_manifest_sha256
        if not model_components or artifact_digest is None:
            raise RuntimeBindingContractError(
                "V1 model identity requires model artifacts and their manifest digest."
            )
        expected_artifact_digest = model_artifact_evidence_sha256(
            artifact_kind,
            model_components,
        )
        if artifact_digest != expected_artifact_digest:
            raise RuntimeBindingContractError(
                "model_artifact_manifest_sha256 does not match model evidence."
            )
        if self.adapter_id != BOUND_CELL_ADAPTER_ID:
            raise RuntimeBindingContractError("adapter_id is outside the v1 contract.")
        if self.execution_scope != BOUND_CELL_EXECUTION_SCOPE:
            raise RuntimeBindingContractError(
                "execution_scope is outside the v1 contract."
            )
        if self.transport != BOUND_CELL_TRANSPORT:
            raise RuntimeBindingContractError("transport is outside the v1 contract.")
        if self.adapter_contract_sha256 != BOUND_CELL_ADAPTER_CONTRACT_SHA256:
            raise RuntimeBindingContractError(
                "adapter_contract_sha256 does not match the v1 adapter contract."
            )
        executable = by_runtime_role["runtime_executable"][0]
        driver = by_runtime_role["driver"][0]
        harness = by_runtime_role["harness"][0]
        expected_runtime_identity = sha256_json(
            {
                "schema_version": CONTRACT_VERSION,
                "runtime_executable_sha256": executable.sha256,
                "driver_sha256": driver.sha256,
            }
        )
        if self.runtime_identity_sha256 != expected_runtime_identity:
            raise RuntimeBindingContractError(
                "runtime_identity_sha256 does not match its runtime components."
            )
        if self.harness_identity_sha256 != harness.sha256:
            raise RuntimeBindingContractError(
                "harness_identity_sha256 does not match its harness component."
            )
        if self.model_identity_sha256 != artifact_digest:
            raise RuntimeBindingContractError(
                "model_identity_sha256 does not match its artifact manifest."
            )
        if self.tool_contract_identity_sha256 != EMPTY_TOOL_CONTRACT_SHA256:
            raise RuntimeBindingContractError(
                "tool_contract_identity_sha256 does not match the empty v1 contract."
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "runtime binding digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "declaration_sha256": self.declaration_sha256,
            "config_source_sha256": self.config_source_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "expert_id": self.expert_id,
            "expert_config_sha256": self.expert_config_sha256,
            "adapter_id": self.adapter_id,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "platform_key": self.platform_key,
            "components": [item.payload() for item in self.components],
            "launch_plan_sha256": self.launch_plan_sha256,
            "endpoint_authority_sha256": self.endpoint_authority_sha256,
            "model_reference_sha256": self.model_reference_sha256,
            "model_artifact_kind": self.model_artifact_kind,
            "model_artifact_manifest_sha256": self.model_artifact_manifest_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "harness_identity_sha256": self.harness_identity_sha256,
            "tool_contract_identity_sha256": self.tool_contract_identity_sha256,
            "model_identity_status": self.model_identity_status,
            "execution_scope": self.execution_scope,
            "transport": self.transport,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "producer_code_sha256": self.producer_code_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellRuntimeInspectionReceipt:
    binding_manifest_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    captured_at: str
    expires_at: str
    component_count: int
    observed_component_count: int
    residency_status: str = "unknown"
    applied: bool = False
    network_used: bool = False
    processes_started: int = 0
    model_invocations: int = 0
    process_mutations: bool = False
    authorizes_execution: bool = False
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime inspection receipt")
        object.__setattr__(
            self,
            "binding_manifest_sha256",
            _sha(self.binding_manifest_sha256, "binding_manifest_sha256"),
        )
        status = _enum(self.status, INSPECTION_STATUSES, "inspection status")
        object.__setattr__(self, "status", status)
        reasons = _call(require_identifier_tuple, self.reason_codes, "reason_codes")
        reasons = tuple(sorted(reasons))
        if not set(reasons).issubset(INSPECTION_REASON_CODES):
            raise RuntimeBindingContractError(
                "reason_codes contain values outside the v1 contract."
            )
        object.__setattr__(self, "reason_codes", reasons)
        captured = _timestamp(self.captured_at, "captured_at")
        expires = _timestamp(self.expires_at, "expires_at")
        captured_time = datetime.fromisoformat(captured)
        expires_time = datetime.fromisoformat(expires)
        if expires_time <= captured_time:
            raise RuntimeBindingContractError("expires_at must be after captured_at.")
        if (expires_time - captured_time).total_seconds() > MAX_INSPECTION_TTL_SECONDS:
            raise RuntimeBindingContractError(
                "Inspection receipt TTL exceeds the v1 safety bound."
            )
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "expires_at", expires)
        count = _positive(self.component_count, "component_count")
        observed = _integer(self.observed_component_count, "observed_component_count")
        if observed != count:
            raise RuntimeBindingContractError(
                "observed_component_count must equal component_count in v1."
            )
        object.__setattr__(self, "component_count", count)
        object.__setattr__(self, "observed_component_count", observed)
        if self.residency_status != "unknown":
            raise RuntimeBindingContractError(
                "residency_status must remain unknown for static inspection."
            )
        for name in (
            "applied",
            "network_used",
            "process_mutations",
            "authorizes_execution",
        ):
            value = _bool(getattr(self, name), name)
            if value:
                raise RuntimeBindingContractError(f"{name} must remain false.")
            object.__setattr__(self, name, value)
        processes = _integer(self.processes_started, "processes_started")
        if processes != 0:
            raise RuntimeBindingContractError("processes_started must remain zero.")
        object.__setattr__(self, "processes_started", processes)
        invocations = _integer(self.model_invocations, "model_invocations")
        if invocations != 0:
            raise RuntimeBindingContractError("model_invocations must remain zero.")
        object.__setattr__(self, "model_invocations", invocations)
        if status == "verified":
            if reasons or observed != count:
                raise RuntimeBindingContractError(
                    "Verified inspection requires every component and no reason codes."
                )
        elif not reasons:
            raise RuntimeBindingContractError(
                "Abstained inspection requires at least one blocker."
            )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "inspection receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "binding_manifest_sha256": self.binding_manifest_sha256,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
            "component_count": self.component_count,
            "observed_component_count": self.observed_component_count,
            "residency_status": self.residency_status,
            "applied": self.applied,
            "network_used": self.network_used,
            "processes_started": self.processes_started,
            "model_invocations": self.model_invocations,
            "process_mutations": self.process_mutations,
            "authorizes_execution": self.authorizes_execution,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


__all__ = [
    "BOUND_CELL_ADAPTER_CONTRACT_SHA256",
    "BOUND_CELL_ADAPTER_ID",
    "BOUND_CELL_ENDPOINT_PATHS",
    "BOUND_CELL_ENDPOINT_SCHEME",
    "BOUND_CELL_EXECUTION_SCOPE",
    "BOUND_CELL_FORBIDDEN_EXECUTABLES",
    "BOUND_CELL_INSPECTOR_CONTRACT",
    "BOUND_CELL_RISK_CLASSES",
    "BOUND_CELL_RUNTIME_BACKENDS",
    "BOUND_CELL_RUNTIME_COMPONENT_ROLES",
    "BOUND_CELL_TRANSPORT",
    "CellRuntimeBindingManifest",
    "CellRuntimeInspectionReceipt",
    "EMPTY_TOOL_CONTRACT_SHA256",
    "INSPECTION_REASON_CODES",
    "INSPECTION_STATUSES",
    "MAX_INSPECTION_TTL_SECONDS",
    "MODEL_ARTIFACT_KINDS",
    "MODEL_IDENTITY_STATUSES",
    "RuntimeBindingContractError",
    "RuntimeComponentEvidence",
    "bound_cell_adapter_contract_payload",
    "model_artifact_evidence_sha256",
]
