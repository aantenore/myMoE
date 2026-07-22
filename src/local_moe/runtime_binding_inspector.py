from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping
from urllib.parse import urlsplit

from .adaptive_advisor_cli import (
    ProtectedRootIdentity,
    ProtectedRootIdentityError,
    capture_protected_root_identity,
    protected_root_identity_is_current,
)
from .artifact_tree import (
    ArtifactTreeIdentity,
    ArtifactTreeLimitError,
    ArtifactTreeLimits,
    hash_artifact_tree,
)
from .bootstrap import ExpertRuntimeCommand, build_runtime_plan
from .cell_contracts import CellContractError, CellDeclaration
from .cell_passport import load_cell_catalog
from .config import ConfigError, ExpertConfig, parse_config, runtime_config_sha256
from .execution_scope import ExecutionScope, ExecutionTransport, is_loopback_endpoint
from .runtime_binding_contracts import (
    BOUND_CELL_ADAPTER_CONTRACT_SHA256,
    BOUND_CELL_ADAPTER_ID,
    BOUND_CELL_ENDPOINT_PATHS,
    BOUND_CELL_ENDPOINT_SCHEME,
    BOUND_CELL_FORBIDDEN_EXECUTABLES,
    BOUND_CELL_INSPECTOR_CONTRACT,
    BOUND_CELL_RISK_CLASSES,
    BOUND_CELL_RUNTIME_BACKENDS,
    BOUND_CELL_RUNTIME_COMPONENT_ROLES,
    EMPTY_TOOL_CONTRACT_SHA256,
    INSPECTION_REASON_CODES,
    MAX_INSPECTION_TTL_SECONDS,
    CellRuntimeBindingManifest,
    CellRuntimeInspectionReceipt,
    RuntimeBindingContractError,
    RuntimeComponentEvidence,
    model_artifact_evidence_sha256,
)
from .secure_files import read_bounded_regular_file
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_safe_id,
    require_sha256,
    sha256_json,
)


REQUEST_CONTRACT = "CellBindingInspectRequest"
BUNDLE_CONTRACT = BOUND_CELL_INSPECTOR_CONTRACT
ADAPTER_ID = BOUND_CELL_ADAPTER_ID
PRODUCER_ID = "mymoe.bound_cell_inspector"
PRODUCER_TRUST_BOUNDARY = (
    "_win32_fs.py",
    "adaptive_advisor_cli.py",
    "artifact_tree.py",
    "bootstrap.py",
    "cell_contracts.py",
    "cell_passport.py",
    "config.py",
    "execution_scope.py",
    "http_boundary.py",
    "model_servers.py",
    "path_security.py",
    "runtime_binding_contracts.py",
    "runtime_binding_inspector.py",
    "secure_files.py",
    "verified_routing_contracts.py",
)


def _package_version() -> str:
    try:
        return version("local-moe-orchestrator")
    except PackageNotFoundError:
        return "0.14.0a1"


PRODUCER_VERSION = _package_version()
MAX_REQUEST_BYTES = 64 * 1024
MAX_RUNTIME_CONFIG_BYTES = 2 * 1024 * 1024
MAX_OBSERVATION_TTL_SECONDS = MAX_INSPECTION_TTL_SECONDS
SUPPORTED_RUNTIME_BACKENDS = BOUND_CELL_RUNTIME_BACKENDS
RUNTIME_COMPONENT_ROLES = BOUND_CELL_RUNTIME_COMPONENT_ROLES
IDENTITY_REASON_CODES = INSPECTION_REASON_CODES


class RuntimeBindingInspectionError(ValueError):
    """Stable fail-closed error at the local runtime inspection boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class RuntimeComponentRequest:
    role: str
    path: str

    def __post_init__(self) -> None:
        role = _safe_id(self.role, "runtime component role")
        if role not in RUNTIME_COMPONENT_ROLES:
            raise RuntimeBindingInspectionError(
                "request_invalid",
                "Runtime component roles are outside the v1 contract.",
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "path",
            _relative_posix_path(self.path, "runtime component path"),
        )

    def payload(self) -> dict[str, object]:
        return {"role": self.role, "path": self.path}


@dataclass(frozen=True)
class CellBindingInspectRequest:
    cell_id: str
    expert_id: str
    adapter_id: str
    catalog_path: str
    runtime_config_path: str
    runtime_root: str
    model_artifact_root: str
    runtime_components: tuple[RuntimeComponentRequest, ...]
    observation_ttl_seconds: int
    hash_limits: ArtifactTreeLimits
    contract: str = REQUEST_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.contract != REQUEST_CONTRACT:
            raise RuntimeBindingInspectionError(
                "request_invalid", "Inspection request contract is unsupported."
            )
        object.__setattr__(self, "cell_id", _safe_id(self.cell_id, "cell_id"))
        object.__setattr__(self, "expert_id", _safe_id(self.expert_id, "expert_id"))
        if self.adapter_id != ADAPTER_ID:
            raise RuntimeBindingInspectionError(
                "adapter_unsupported", "Inspection adapter is unsupported."
            )
        for name in (
            "catalog_path",
            "runtime_config_path",
            "runtime_root",
            "model_artifact_root",
        ):
            object.__setattr__(
                self,
                name,
                _relative_posix_path(getattr(self, name), name),
            )
        components = (
            tuple(self.runtime_components)
            if isinstance(self.runtime_components, (tuple, list))
            else ()
        )
        if any(not isinstance(item, RuntimeComponentRequest) for item in components):
            raise RuntimeBindingInspectionError(
                "request_invalid", "Runtime components are invalid."
            )
        roles = [item.role for item in components]
        if len(components) != len(RUNTIME_COMPONENT_ROLES) or set(roles) != set(
            RUNTIME_COMPONENT_ROLES
        ):
            raise RuntimeBindingInspectionError(
                "request_invalid",
                "Inspection requires exactly one runtime executable, driver, and harness.",
            )
        if len(roles) != len(set(roles)) or len(
            {item.path for item in components}
        ) != len(components):
            raise RuntimeBindingInspectionError(
                "request_invalid", "Runtime component roles and paths must be unique."
            )
        object.__setattr__(
            self,
            "runtime_components",
            tuple(sorted(components, key=lambda item: (item.role, item.path))),
        )
        ttl = _integer(self.observation_ttl_seconds, "observation_ttl_seconds")
        if not 1 <= ttl <= MAX_OBSERVATION_TTL_SECONDS:
            raise RuntimeBindingInspectionError(
                "request_invalid", "Observation TTL exceeds the v1 safety bound."
            )
        object.__setattr__(self, "observation_ttl_seconds", ttl)
        if not isinstance(self.hash_limits, ArtifactTreeLimits):
            raise RuntimeBindingInspectionError(
                "request_invalid", "Artifact hash limits are invalid."
            )

    @property
    def digest(self) -> str:
        return sha256_json(self.content_payload())

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "cell_id": self.cell_id,
            "expert_id": self.expert_id,
            "adapter_id": self.adapter_id,
            "catalog_path": self.catalog_path,
            "runtime_config_path": self.runtime_config_path,
            "runtime_root": self.runtime_root,
            "model_artifact_root": self.model_artifact_root,
            "runtime_components": [item.payload() for item in self.runtime_components],
            "observation_ttl_seconds": self.observation_ttl_seconds,
            "hash_limits": _hash_limits_payload(self.hash_limits),
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellBindingInspectionBundle:
    request_sha256: str
    manifest: CellRuntimeBindingManifest
    receipt: CellRuntimeInspectionReceipt
    publication_protected_roots: tuple[ProtectedRootIdentity, ...] = field(
        repr=False,
        compare=False,
    )
    contract: str = BUNDLE_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION or self.contract != BUNDLE_CONTRACT:
            raise RuntimeBindingInspectionError(
                "bundle_invalid", "Inspection bundle contract is unsupported."
            )
        try:
            request_sha = require_sha256(self.request_sha256, "request_sha256")
        except VerifiedRoutingError as exc:
            raise RuntimeBindingInspectionError(
                "bundle_invalid", "Inspection request digest is invalid."
            ) from exc
        if not isinstance(self.manifest, CellRuntimeBindingManifest) or not isinstance(
            self.receipt, CellRuntimeInspectionReceipt
        ):
            raise RuntimeBindingInspectionError(
                "bundle_invalid", "Inspection bundle members are invalid."
            )
        if self.receipt.binding_manifest_sha256 != self.manifest.digest:
            raise RuntimeBindingInspectionError(
                "bundle_invalid", "Inspection receipt does not bind its manifest."
            )
        if self.receipt.component_count != len(self.manifest.components):
            raise RuntimeBindingInspectionError(
                "bundle_invalid",
                "Inspection receipt component count does not match its manifest.",
            )
        roots = (
            tuple(self.publication_protected_roots)
            if isinstance(self.publication_protected_roots, (tuple, list))
            else ()
        )
        if len(roots) != 2 or any(
            not isinstance(item, ProtectedRootIdentity) for item in roots
        ):
            raise RuntimeBindingInspectionError(
                "bundle_invalid", "Inspection publication boundary is invalid."
            )
        object.__setattr__(self, "request_sha256", request_sha)
        object.__setattr__(self, "publication_protected_roots", roots)

    @property
    def digest(self) -> str:
        return sha256_json(self.content_payload())

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "request_sha256": self.request_sha256,
            "binding_manifest": self.manifest.payload(),
            "inspection_receipt": self.receipt.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def load_cell_binding_inspect_request(
    path: str | Path,
) -> CellBindingInspectRequest:
    raw = _load_json_object(
        path,
        maximum_bytes=MAX_REQUEST_BYTES,
        label="inspection request",
        code="request_invalid",
    )
    data = _strict(raw, _field_names(CellBindingInspectRequest), "inspection request")
    components_raw = data["runtime_components"]
    if not isinstance(components_raw, list):
        raise RuntimeBindingInspectionError(
            "request_invalid", "runtime_components must be a list."
        )
    components: list[RuntimeComponentRequest] = []
    for raw_component in components_raw:
        item = _strict(
            raw_component,
            _field_names(RuntimeComponentRequest),
            "runtime component",
        )
        components.append(
            RuntimeComponentRequest(role=item["role"], path=item["path"])  # type: ignore[arg-type]
        )
    limits_raw = _strict(
        data["hash_limits"],
        _field_names(ArtifactTreeLimits),
        "hash limits",
    )
    try:
        limits = ArtifactTreeLimits(**limits_raw)  # type: ignore[arg-type]
    except RuntimeBindingInspectionError:
        raise
    except (CellContractError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "request_invalid", "Artifact hash limits are invalid."
        ) from exc
    try:
        return CellBindingInspectRequest(
            cell_id=data["cell_id"],  # type: ignore[arg-type]
            expert_id=data["expert_id"],  # type: ignore[arg-type]
            adapter_id=data["adapter_id"],  # type: ignore[arg-type]
            catalog_path=data["catalog_path"],  # type: ignore[arg-type]
            runtime_config_path=data["runtime_config_path"],  # type: ignore[arg-type]
            runtime_root=data["runtime_root"],  # type: ignore[arg-type]
            model_artifact_root=data["model_artifact_root"],  # type: ignore[arg-type]
            runtime_components=tuple(components),
            observation_ttl_seconds=data["observation_ttl_seconds"],  # type: ignore[arg-type]
            hash_limits=limits,
            contract=data["contract"],  # type: ignore[arg-type]
            schema_version=data["schema_version"],  # type: ignore[arg-type]
        )
    except RuntimeBindingInspectionError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "request_invalid", "Inspection request is invalid."
        ) from exc


def inspect_cell_binding(
    request_path: str | Path,
    *,
    now: datetime | None = None,
    publication_path: str | Path | None = None,
) -> CellBindingInspectionBundle:
    """Inspect one declared local cell without executing or contacting it."""

    request_file = _request_path(request_path)
    request = load_cell_binding_inspect_request(request_file)
    root = request_file.parent
    catalog_file = _within_request_root(root, request.catalog_path, "catalog")
    config_file = _within_request_root(
        root, request.runtime_config_path, "runtime config"
    )
    runtime_root = _within_request_root(root, request.runtime_root, "runtime root")
    model_root = _within_request_root(
        root, request.model_artifact_root, "model artifact root"
    )
    try:
        publication_protected_roots = tuple(
            capture_protected_root_identity(path) for path in (runtime_root, model_root)
        )
    except ProtectedRootIdentityError as exc:
        raise RuntimeBindingInspectionError(
            "artifact_root_invalid",
            "Runtime and model artifact roots must be real directories.",
        ) from exc
    _validate_publication_path(
        publication_path,
        request_file=request_file,
        catalog_file=catalog_file,
        config_file=config_file,
        runtime_root=runtime_root,
        model_root=model_root,
    )

    config_raw, config_source_sha = _load_runtime_config(config_file, root=root)
    try:
        config = parse_config(config_raw)
    except (
        AttributeError,
        ConfigError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeBindingInspectionError(
            "runtime_config_invalid", "Runtime configuration is invalid."
        ) from exc
    try:
        catalog = load_cell_catalog(catalog_file, confinement_root=root)
    except (CellContractError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "catalog_invalid", "Adaptive cell catalog is invalid."
        ) from exc

    passports = [item for item in catalog.cells if item.cell_id == request.cell_id]
    if len(passports) != 1:
        raise RuntimeBindingInspectionError(
            "cell_not_found", "Inspection cell is not uniquely declared."
        )
    declaration = passports[0].declaration
    experts = [item for item in config.experts if item.id == request.expert_id]
    if len(experts) != 1:
        raise RuntimeBindingInspectionError(
            "expert_not_found", "Inspection expert is not uniquely configured."
        )
    expert = experts[0]
    raw_expert = _selected_raw_expert(config_raw, request.expert_id)
    backend, runtime_executable = _validate_scope_and_runtime(
        config, expert, raw_expert, declaration, request
    )
    endpoint = _validate_endpoint(expert)
    model_path, model_relative = _validate_model_path(
        root=root,
        model_root=model_root,
        expert=expert,
        backend=backend,
    )
    command, platform_key = _selected_runtime_command(
        config=config,
        expert=expert,
        backend=backend,
        endpoint=endpoint,
        model_reference=expert.model,
        runtime_executable=runtime_executable,
    )

    runtime_evidence, runtime_trees = _hash_runtime_components(
        trusted_root=root,
        runtime_root=runtime_root,
        request=request,
    )
    model_limits = _remaining_artifact_limits(request.hash_limits, runtime_trees)
    model_tree = _hash_tree(
        model_path,
        root=root,
        limits=model_limits,
        code="model_artifact_invalid",
        detail="Model artifacts could not be inspected.",
    )
    _validate_model_artifact_shape(model_tree, backend=backend)
    _enforce_aggregate_limits(
        request.hash_limits,
        (*runtime_trees, model_tree),
    )
    model_evidence = _model_component_evidence(
        model_tree,
        model_relative=model_relative,
    )
    components = tuple(
        sorted(
            (*runtime_evidence, *model_evidence),
            key=lambda item: (item.role, item.root_id, item.path),
        )
    )

    by_role = {item.role: item for item in runtime_evidence}
    observed_runtime = runtime_identity_sha256(
        by_role["runtime_executable"], by_role["driver"]
    )
    observed_harness = harness_identity_sha256(by_role["harness"])
    observed_model = model_artifact_evidence_sha256(
        model_tree.kind,
        model_evidence,
    )
    observed_tools = empty_tool_contract_sha256()
    reasons = _identity_reasons(
        declaration,
        observed_model=observed_model,
        observed_runtime=observed_runtime,
        observed_harness=observed_harness,
        observed_tools=observed_tools,
    )

    try:
        manifest = CellRuntimeBindingManifest(
            cell_id=declaration.cell_id,
            declaration_sha256=declaration.digest,
            config_source_sha256=config_source_sha,
            runtime_config_sha256=runtime_config_sha256(config),
            expert_id=expert.id,
            expert_config_sha256=_expert_config_sha256(expert),
            adapter_id=request.adapter_id,
            adapter_contract_sha256=_adapter_contract_sha256(),
            platform_key=platform_key,
            components=components,
            launch_plan_sha256=_launch_plan_sha256(command),
            endpoint_authority_sha256=_endpoint_authority_sha256(endpoint),
            model_reference_sha256=sha256_json(
                {"schema_version": CONTRACT_VERSION, "path": model_relative}
            ),
            model_artifact_kind=model_tree.kind,
            model_artifact_manifest_sha256=observed_model,
            model_identity_sha256=observed_model,
            runtime_identity_sha256=observed_runtime,
            harness_identity_sha256=observed_harness,
            tool_contract_identity_sha256=observed_tools,
            model_identity_status="verified",
            execution_scope="device_only",
            transport="direct_local",
            producer_id=PRODUCER_ID,
            producer_version=PRODUCER_VERSION,
            producer_code_sha256=_producer_code_sha256(),
        )
        captured = _inspection_time(now)
        try:
            expires = captured + timedelta(seconds=request.observation_ttl_seconds)
        except OverflowError as exc:
            raise RuntimeBindingInspectionError(
                "clock_invalid", "Inspection clock cannot represent the receipt TTL."
            ) from exc
        receipt = CellRuntimeInspectionReceipt(
            binding_manifest_sha256=manifest.digest,
            status="verified" if not reasons else "abstained",
            reason_codes=tuple(sorted(reasons)),
            captured_at=captured.isoformat(),
            expires_at=expires.isoformat(),
            component_count=len(components),
            observed_component_count=len(components),
        )
        if not all(
            protected_root_identity_is_current(identity)
            for identity in publication_protected_roots
        ):
            raise RuntimeBindingInspectionError(
                "artifact_root_changed",
                "An inspected artifact root changed during inspection.",
            )
        return CellBindingInspectionBundle(
            request_sha256=request.digest,
            manifest=manifest,
            receipt=receipt,
            publication_protected_roots=publication_protected_roots,
        )
    except RuntimeBindingInspectionError:
        raise
    except (RuntimeBindingContractError, VerifiedRoutingError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "inspection_contract_invalid", "Inspection evidence is invalid."
        ) from exc


def runtime_identity_sha256(
    executable: RuntimeComponentEvidence,
    driver: RuntimeComponentEvidence,
) -> str:
    """Return the stable identity of the runtime executable and its driver."""

    if executable.role != "runtime_executable" or driver.role != "driver":
        raise RuntimeBindingInspectionError(
            "inspection_contract_invalid", "Runtime identity roles are invalid."
        )
    return sha256_json(
        {
            "schema_version": CONTRACT_VERSION,
            "runtime_executable_sha256": executable.sha256,
            "driver_sha256": driver.sha256,
        }
    )


def harness_identity_sha256(harness: RuntimeComponentEvidence) -> str:
    """Return the content identity of the declared harness."""

    if harness.role != "harness":
        raise RuntimeBindingInspectionError(
            "inspection_contract_invalid", "Harness identity role is invalid."
        )
    return harness.sha256


def empty_tool_contract_sha256() -> str:
    """Return the v1 identity for a cell with no tool authority."""

    return EMPTY_TOOL_CONTRACT_SHA256


def _load_runtime_config(
    path: Path,
    *,
    root: Path,
) -> tuple[dict[str, object], str]:
    try:
        content = read_bounded_regular_file(
            path,
            root=root,
            maximum_bytes=MAX_RUNTIME_CONFIG_BYTES,
            label="runtime configuration",
        )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except RuntimeBindingInspectionError:
        raise
    except (
        CellContractError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeBindingInspectionError(
            "runtime_config_invalid", "Runtime configuration is invalid."
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeBindingInspectionError(
            "runtime_config_invalid", "Runtime configuration must be an object."
        )
    return raw, hashlib.sha256(content).hexdigest()


def _selected_raw_expert(
    raw_config: Mapping[str, object], expert_id: str
) -> dict[str, object]:
    raw_experts = raw_config.get("experts")
    if not isinstance(raw_experts, list):
        raise RuntimeBindingInspectionError(
            "runtime_config_invalid", "Runtime experts are invalid."
        )
    matches = [
        item
        for item in raw_experts
        if isinstance(item, dict) and item.get("id") == expert_id
    ]
    if len(matches) != 1:
        raise RuntimeBindingInspectionError(
            "expert_not_found", "Inspection expert is not uniquely configured."
        )
    return dict(matches[0])


def _validate_scope_and_runtime(
    config: object,
    expert: ExpertConfig,
    raw_expert: Mapping[str, object],
    declaration: CellDeclaration,
    request: CellBindingInspectRequest,
) -> tuple[str, str]:
    policy = getattr(config, "execution_policy")
    if (
        policy.max_scope != ExecutionScope.DEVICE_ONLY
        or policy.allowed_scopes != (ExecutionScope.DEVICE_ONLY,)
        or policy.allow_scope_widening
    ):
        raise RuntimeBindingInspectionError(
            "execution_policy_unsupported",
            "Inspection requires a device-only non-widening execution policy.",
        )
    if expert.provider != "openai_compatible":
        raise RuntimeBindingInspectionError(
            "provider_unsupported", "Inspection requires an OpenAI-compatible expert."
        )
    execution = raw_expert.get("execution")
    if not isinstance(execution, dict) or execution != {
        "scope": "device_only",
        "transport": "direct_local",
    }:
        raise RuntimeBindingInspectionError(
            "expert_execution_unsupported",
            "Expert execution must explicitly be device-only and direct-local.",
        )
    if (
        expert.execution.scope != ExecutionScope.DEVICE_ONLY
        or expert.execution.transport != ExecutionTransport.DIRECT_LOCAL
    ):
        raise RuntimeBindingInspectionError(
            "expert_execution_unsupported",
            "Effective expert execution is outside the local adapter contract.",
        )
    params = raw_expert.get("params")
    if not isinstance(params, dict):
        raise RuntimeBindingInspectionError(
            "runtime_backend_unsupported", "Runtime parameters must be explicit."
        )
    backend = params.get("runtime_backend")
    if backend not in SUPPORTED_RUNTIME_BACKENDS:
        raise RuntimeBindingInspectionError(
            "runtime_backend_unsupported", "Runtime backend is unsupported."
        )
    if params.get("runtime_model_source") != "local":
        raise RuntimeBindingInspectionError(
            "runtime_model_source_unsupported",
            "Runtime model source must explicitly remain local.",
        )
    raw_runtime_executable = params.get("runtime_executable")
    if not isinstance(raw_runtime_executable, str):
        raise RuntimeBindingInspectionError(
            "runtime_executable_mismatch",
            "Runtime executable must be explicitly bound to inspected evidence.",
        )
    try:
        runtime_executable = _relative_posix_path(
            raw_runtime_executable, "runtime executable"
        )
    except RuntimeBindingInspectionError as exc:
        raise RuntimeBindingInspectionError(
            "runtime_executable_mismatch",
            "Runtime executable must be explicitly bound to inspected evidence.",
        ) from exc
    declared_executables = [
        item.path
        for item in request.runtime_components
        if item.role == "runtime_executable"
    ]
    expected_runtime_executable = (
        PurePosixPath(request.runtime_root) / declared_executables[0]
    ).as_posix()
    if runtime_executable != expected_runtime_executable:
        raise RuntimeBindingInspectionError(
            "runtime_executable_mismatch",
            "Runtime command and inspected executable do not share the request root.",
        )
    if not declaration.offline_capable:
        raise RuntimeBindingInspectionError(
            "cell_contract_unsupported", "Cell is not declared offline-capable."
        )
    if declaration.risk_classes != BOUND_CELL_RISK_CLASSES:
        raise RuntimeBindingInspectionError(
            "cell_contract_unsupported", "Cell risk classes exceed compute-only."
        )
    if declaration.tool_surfaces:
        raise RuntimeBindingInspectionError(
            "cell_contract_unsupported", "Cell declares tool authority."
        )
    return str(backend), runtime_executable


def _validate_endpoint(expert: ExpertConfig):
    raw = expert.base_url
    if not isinstance(raw, str) or raw != raw.strip():
        raise RuntimeBindingInspectionError(
            "endpoint_unsupported", "Expert endpoint is invalid."
        )
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeBindingInspectionError(
            "endpoint_unsupported", "Expert endpoint is invalid."
        ) from exc
    if (
        parsed.scheme.lower() != BOUND_CELL_ENDPOINT_SCHEME
        or not is_loopback_endpoint(raw)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in raw
        or "#" in raw
        or not parsed.hostname
        or port is None
    ):
        raise RuntimeBindingInspectionError(
            "endpoint_unsupported",
            "Expert endpoint must be an explicit uncredentialed loopback authority.",
        )
    if parsed.path not in BOUND_CELL_ENDPOINT_PATHS:
        raise RuntimeBindingInspectionError(
            "endpoint_unsupported", "Expert endpoint path is unsupported."
        )
    return parsed


def _validate_model_path(
    *,
    root: Path,
    model_root: Path,
    expert: ExpertConfig,
    backend: str,
) -> tuple[Path, str]:
    model_reference = _relative_posix_path(expert.model, "expert model")
    model_path = _within_request_root(root, model_reference, "expert model")
    try:
        model_relative = model_path.relative_to(model_root).as_posix()
    except ValueError as exc:
        raise RuntimeBindingInspectionError(
            "model_path_invalid", "Expert model leaves the artifact root."
        ) from exc
    if model_relative == ".":
        model_relative = "root"
    if (
        backend == "llama_cpp"
        and PurePosixPath(model_reference).suffix.lower() != ".gguf"
    ):
        raise RuntimeBindingInspectionError(
            "model_artifact_invalid", "llama.cpp model artifact must be a GGUF file."
        )
    return model_path, model_relative


def _validate_model_artifact_shape(
    tree: ArtifactTreeIdentity,
    *,
    backend: str,
) -> None:
    if backend in {"mlx_lm", "mlx_vlm"} and tree.kind != "directory":
        raise RuntimeBindingInspectionError(
            "model_artifact_invalid", "MLX model artifact must be a directory."
        )
    if backend == "llama_cpp" and tree.kind != "file":
        raise RuntimeBindingInspectionError(
            "model_artifact_invalid", "llama.cpp model artifact must be a GGUF file."
        )


def _selected_runtime_command(
    *,
    config: object,
    expert: ExpertConfig,
    backend: str,
    endpoint: object,
    model_reference: str,
    runtime_executable: str,
) -> tuple[ExpertRuntimeCommand, str]:
    try:
        plan = build_runtime_plan(config)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime plan could not be constructed."
        ) from exc
    matches = [item for item in plan.expert_commands if item.expert_id == expert.id]
    if len(matches) != 1 or matches[0].backend != backend:
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime plan does not uniquely bind the expert."
        )
    command = matches[0]
    argv = command.argv
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime command is invalid."
        )
    executable = PurePosixPath(argv[0].replace("\\", "/")).name.casefold()
    if executable in BOUND_CELL_FORBIDDEN_EXECUTABLES:
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Shell-wrapper runtime commands are forbidden."
        )
    normalized_argv_executable = argv[0].replace("\\", "/")
    if normalized_argv_executable != runtime_executable:
        raise RuntimeBindingInspectionError(
            "runtime_executable_mismatch",
            "Runtime command and inspected executable are not bound.",
        )
    lowered = tuple(item.casefold() for item in argv)
    if (
        any(item == "-hf" or item.startswith("-hf=") for item in lowered)
        or "pull" in lowered
        or any(item in {"download", "ollama"} for item in lowered)
        or any(item.startswith("-") and "download" in item for item in lowered)
        or any(item.startswith(("http://", "https://")) for item in lowered)
    ):
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime command could fetch remote artifacts."
        )
    model_flag = "-m" if backend == "llama_cpp" else "--model"
    if _flag_values(argv, model_flag) != (model_reference,):
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime command does not bind the local model."
        )
    host_values = _flag_values(argv, "--host")
    port_values = _flag_values(argv, "--port")
    hostname = str(getattr(endpoint, "hostname"))
    port = str(getattr(endpoint, "port"))
    if host_values != (hostname,) or port_values != (port,):
        raise RuntimeBindingInspectionError(
            "runtime_plan_invalid", "Runtime command does not bind the endpoint."
        )
    return command, plan.platform_key


def _hash_runtime_components(
    *,
    trusted_root: Path,
    runtime_root: Path,
    request: CellBindingInspectRequest,
) -> tuple[tuple[RuntimeComponentEvidence, ...], tuple[ArtifactTreeIdentity, ...]]:
    _require_distinct_runtime_component_locations(runtime_root, request)
    evidence: list[RuntimeComponentEvidence] = []
    trees: list[ArtifactTreeIdentity] = []
    for component in request.runtime_components:
        remaining_limits = _remaining_artifact_limits(request.hash_limits, trees)
        tree = _hash_tree(
            runtime_root / component.path,
            root=trusted_root,
            limits=remaining_limits,
            code="runtime_component_invalid",
            detail="Runtime component could not be inspected.",
        )
        if tree.kind != "file" or len(tree.entries) != 1:
            raise RuntimeBindingInspectionError(
                "runtime_component_invalid", "Runtime component must be one file."
            )
        entry = tree.entries[0]
        try:
            evidence.append(
                RuntimeComponentEvidence(
                    role=component.role,
                    root_id="runtime",
                    path=component.path,
                    size_bytes=entry.size_bytes,
                    sha256=entry.sha256,
                )
            )
        except RuntimeBindingContractError as exc:
            raise RuntimeBindingInspectionError(
                "runtime_component_invalid", "Runtime component evidence is invalid."
            ) from exc
        trees.append(tree)
    _require_distinct_runtime_component_locations(runtime_root, request)
    return tuple(evidence), tuple(trees)


def _require_distinct_runtime_component_locations(
    runtime_root: Path,
    request: CellBindingInspectRequest,
) -> None:
    paths = [runtime_root / component.path for component in request.runtime_components]
    try:
        for index, first in enumerate(paths):
            for second in paths[index + 1 :]:
                if os.path.samefile(first, second):
                    raise RuntimeBindingInspectionError(
                        "runtime_component_invalid",
                        "Runtime component locations must be physically distinct.",
                    )
    except RuntimeBindingInspectionError:
        raise
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeBindingInspectionError(
            "runtime_component_invalid",
            "Runtime component locations could not be compared safely.",
        ) from exc


def _model_component_evidence(
    tree: ArtifactTreeIdentity,
    *,
    model_relative: str,
) -> tuple[RuntimeComponentEvidence, ...]:
    del model_relative  # The selected model root is deliberately not disclosed.
    result: list[RuntimeComponentEvidence] = []
    for entry in tree.entries:
        path_identity = sha256_json(
            {"schema_version": CONTRACT_VERSION, "path": entry.path}
        )
        suffix = ".gguf" if tree.kind == "file" else ""
        logical_path = f"artifact-{path_identity}{suffix}"
        try:
            result.append(
                RuntimeComponentEvidence(
                    role="model_artifact",
                    root_id="model",
                    path=logical_path,
                    size_bytes=entry.size_bytes,
                    sha256=entry.sha256,
                )
            )
        except RuntimeBindingContractError as exc:
            raise RuntimeBindingInspectionError(
                "model_artifact_invalid", "Model artifact evidence is invalid."
            ) from exc
    if not result:
        raise RuntimeBindingInspectionError(
            "model_artifact_invalid", "Model artifact tree is empty."
        )
    return tuple(result)


def _hash_tree(
    path: Path,
    *,
    root: Path,
    limits: ArtifactTreeLimits,
    code: str,
    detail: str,
) -> ArtifactTreeIdentity:
    try:
        relative = path.relative_to(root)
        return hash_artifact_tree(relative, root=root, limits=limits)
    except ArtifactTreeLimitError as exc:
        raise RuntimeBindingInspectionError(
            "artifact_limits_exceeded",
            "Combined inspection artifacts exceed the configured bounds.",
        ) from exc
    except (
        CellContractError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeBindingInspectionError(code, detail) from exc


def _remaining_artifact_limits(
    limits: ArtifactTreeLimits,
    trees: tuple[ArtifactTreeIdentity, ...] | list[ArtifactTreeIdentity],
) -> ArtifactTreeLimits:
    remaining_files = limits.max_files - sum(item.file_count for item in trees)
    remaining_bytes = limits.max_total_bytes - sum(item.total_bytes for item in trees)
    if remaining_files < 1 or remaining_bytes < 1:
        raise RuntimeBindingInspectionError(
            "artifact_limits_exceeded",
            "Combined inspection artifacts exceed the configured bounds.",
        )
    try:
        return ArtifactTreeLimits(
            max_files=remaining_files,
            max_total_bytes=remaining_bytes,
            max_depth=limits.max_depth,
            max_file_bytes=min(limits.max_file_bytes, remaining_bytes),
        )
    except CellContractError as exc:
        raise RuntimeBindingInspectionError(
            "artifact_limits_exceeded",
            "Combined inspection artifacts exceed the configured bounds.",
        ) from exc


def _enforce_aggregate_limits(
    limits: ArtifactTreeLimits,
    trees: tuple[ArtifactTreeIdentity, ...],
) -> None:
    file_count = sum(item.file_count for item in trees)
    total_bytes = sum(item.total_bytes for item in trees)
    if file_count > limits.max_files or total_bytes > limits.max_total_bytes:
        raise RuntimeBindingInspectionError(
            "artifact_limits_exceeded",
            "Combined inspection artifacts exceed the configured bounds.",
        )


def _identity_reasons(
    declaration: CellDeclaration,
    *,
    observed_model: str,
    observed_runtime: str,
    observed_harness: str,
    observed_tools: str,
) -> tuple[str, ...]:
    pairs = (
        ("model", declaration.expected_model_sha256, observed_model),
        ("runtime", declaration.expected_runtime_sha256, observed_runtime),
        ("harness", declaration.expected_harness_sha256, observed_harness),
        (
            "tool_contract",
            declaration.expected_tool_contract_sha256,
            observed_tools,
        ),
    )
    reasons: list[str] = []
    for name, expected, observed in pairs:
        if expected is None:
            reasons.append(f"{name}_identity_unknown")
        elif expected != observed:
            reasons.append(f"{name}_identity_mismatch")
    if not set(reasons).issubset(IDENTITY_REASON_CODES):
        raise RuntimeBindingInspectionError(
            "inspection_contract_invalid", "Inspection reason codes are invalid."
        )
    return tuple(sorted(reasons))


def _expert_config_sha256(expert: ExpertConfig) -> str:
    return sha256_json(
        {
            "id": expert.id,
            "provider": expert.provider,
            "model": expert.model,
            "role": expert.role,
            "weight": expert.weight,
            "timeout_seconds": expert.timeout_seconds,
            "base_url": expert.base_url,
            "params": expert.params,
            "execution": {
                "scope": (
                    expert.execution.scope.value
                    if expert.execution.scope is not None
                    else None
                ),
                "transport": (
                    expert.execution.transport.value
                    if expert.execution.transport is not None
                    else None
                ),
            },
        }
    )


def _adapter_contract_sha256() -> str:
    return BOUND_CELL_ADAPTER_CONTRACT_SHA256


def _launch_plan_sha256(command: ExpertRuntimeCommand) -> str:
    return sha256_json(
        {
            "schema_version": CONTRACT_VERSION,
            "expert_id": command.expert_id,
            "backend": command.backend,
            "working_directory": "inspection_request_root",
            "argv": list(command.argv),
        }
    )


def _endpoint_authority_sha256(endpoint: object) -> str:
    return sha256_json(
        {
            "schema_version": CONTRACT_VERSION,
            "scheme": str(getattr(endpoint, "scheme")).lower(),
            "hostname": str(getattr(endpoint, "hostname")).rstrip(".").lower(),
            "port": int(getattr(endpoint, "port")),
            "base_path": str(getattr(endpoint, "path")) or "/",
        }
    )


def _producer_code_sha256() -> str:
    module_root = Path(__file__).parent
    module_digests: dict[str, str] = {}
    try:
        for module_name in PRODUCER_TRUST_BOUNDARY:
            content = read_bounded_regular_file(
                module_root / module_name,
                root=module_root,
                maximum_bytes=2 * 1024 * 1024,
                label="inspector producer module",
            )
            module_digests[module_name] = hashlib.sha256(content).hexdigest()
    except CellContractError as exc:
        raise RuntimeBindingInspectionError(
            "producer_identity_unavailable", "Inspector identity is unavailable."
        ) from exc
    return sha256_json(
        {
            "schema_version": CONTRACT_VERSION,
            "producer_id": PRODUCER_ID,
            "modules": module_digests,
        }
    )


def _validate_publication_path(
    publication_path: str | Path | None,
    *,
    request_file: Path,
    catalog_file: Path,
    config_file: Path,
    runtime_root: Path,
    model_root: Path,
) -> None:
    if publication_path is None:
        return
    try:
        supplied = Path(os.path.abspath(os.fspath(publication_path)))
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "output_path_conflict", "Inspection output path is invalid."
        ) from exc
    protected_files = (request_file, catalog_file, config_file)
    protected_roots = (runtime_root, model_root)
    if supplied in protected_files or any(
        _is_within(supplied, root) for root in protected_roots
    ):
        raise RuntimeBindingInspectionError(
            "output_path_conflict",
            "Inspection output must stay outside inspected inputs and artifacts.",
        )
    if _publication_aliases_protected_location(
        supplied,
        protected_files=protected_files,
        protected_roots=protected_roots,
    ):
        raise RuntimeBindingInspectionError(
            "output_path_conflict",
            "Inspection output must stay outside inspected inputs and artifacts.",
        )
    try:
        canonical = supplied.parent.resolve(strict=True) / supplied.name
        canonical_files = tuple(item.resolve(strict=True) for item in protected_files)
        canonical_roots = tuple(item.resolve(strict=True) for item in protected_roots)
    except (OSError, RuntimeError, ValueError):
        return
    if canonical in canonical_files or any(
        _is_within(canonical, root) for root in canonical_roots
    ):
        raise RuntimeBindingInspectionError(
            "output_path_conflict",
            "Inspection output must stay outside inspected inputs and artifacts.",
        )


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _publication_aliases_protected_location(
    supplied: Path,
    *,
    protected_files: tuple[Path, ...],
    protected_roots: tuple[Path, ...],
) -> bool:
    if supplied.exists() and any(
        _same_existing_location(supplied, item) for item in protected_files
    ):
        return True
    current = supplied.parent
    while True:
        if current.exists() and any(
            _same_existing_location(current, root) for root in protected_roots
        ):
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _same_existing_location(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeBindingInspectionError(
            "output_path_conflict",
            "Inspection output location could not be compared safely.",
        ) from exc


def _inspection_time(value: datetime | None) -> datetime:
    captured = value or datetime.now(timezone.utc)
    if not isinstance(captured, datetime) or captured.tzinfo is None:
        raise RuntimeBindingInspectionError(
            "clock_invalid", "Inspection clock must be timezone-aware."
        )
    return captured.astimezone(timezone.utc).replace(microsecond=0)


def _request_path(value: str | Path) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "request_invalid", "Inspection request path is invalid."
        ) from exc
    if not path.name:
        raise RuntimeBindingInspectionError(
            "request_invalid", "Inspection request path is invalid."
        )
    return path


def _within_request_root(root: Path, relative: str, label: str) -> Path:
    try:
        target = Path(os.path.abspath(os.fspath(root / relative)))
        target.relative_to(root)
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise RuntimeBindingInspectionError(
            "path_escape", f"{label.capitalize()} leaves the request directory."
        ) from exc
    return target


def _relative_posix_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RuntimeBindingInspectionError(
            "path_escape", f"{label} must be a relative POSIX path."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise RuntimeBindingInspectionError(
            "path_escape", f"{label} must stay below its declared root."
        )
    return value


def _load_json_object(
    path: str | Path,
    *,
    maximum_bytes: int,
    label: str,
    code: str,
) -> dict[str, object]:
    try:
        content = read_bounded_regular_file(
            path,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except RuntimeBindingInspectionError:
        raise
    except (
        CellContractError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeBindingInspectionError(
            code, f"{label.capitalize()} is invalid."
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeBindingInspectionError(
            code, f"{label.capitalize()} must be an object."
        )
    return raw


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeBindingInspectionError(
                "json_duplicate_key", "Duplicate JSON keys are not allowed."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    del value
    raise RuntimeBindingInspectionError(
        "json_non_finite", "Non-finite JSON numbers are not allowed."
    )


def _strict(raw: object, allowed: set[str], label: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise RuntimeBindingInspectionError(
            "request_invalid", f"{label.capitalize()} must be an object."
        )
    data = dict(raw)
    if any(not isinstance(key, str) for key in data):
        raise RuntimeBindingInspectionError(
            "request_invalid", f"{label.capitalize()} fields are invalid."
        )
    if set(data) != allowed:
        raise RuntimeBindingInspectionError(
            "request_invalid",
            f"{label.capitalize()} must contain exactly the v1 fields.",
        )
    return data


def _field_names(cls: type[object]) -> set[str]:
    return {item.name for item in fields(cls)}


def _safe_id(value: object, label: str) -> str:
    try:
        return require_safe_id(value, label)
    except VerifiedRoutingError as exc:
        raise RuntimeBindingInspectionError(
            "request_invalid", f"{label} is invalid."
        ) from exc


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeBindingInspectionError(
            "request_invalid", f"{label} must be an integer."
        )
    return value


def _hash_limits_payload(limits: ArtifactTreeLimits) -> dict[str, object]:
    return {
        "max_files": limits.max_files,
        "max_total_bytes": limits.max_total_bytes,
        "max_depth": limits.max_depth,
        "max_file_bytes": limits.max_file_bytes,
    }


def _flag_values(argv: tuple[str, ...], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv):
                return ()
            values.append(argv[index + 1])
    return tuple(values)


__all__ = [
    "ADAPTER_ID",
    "BUNDLE_CONTRACT",
    "CellBindingInspectRequest",
    "CellBindingInspectionBundle",
    "IDENTITY_REASON_CODES",
    "MAX_OBSERVATION_TTL_SECONDS",
    "PRODUCER_TRUST_BOUNDARY",
    "REQUEST_CONTRACT",
    "RUNTIME_COMPONENT_ROLES",
    "RuntimeBindingInspectionError",
    "RuntimeComponentRequest",
    "empty_tool_contract_sha256",
    "harness_identity_sha256",
    "inspect_cell_binding",
    "load_cell_binding_inspect_request",
    "runtime_identity_sha256",
]
