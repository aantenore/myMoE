from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import sys
import sysconfig
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import assistant_bridge_workspace as _workspace_security
from .deterministic_evaluator import (
    QualityBenchmarkError,
    evaluate_check,
    validate_checks,
)
from .assistant_bridge_ledger import (
    BridgeLedgerError,
    BridgeStateLedger,
    PremiumBudgetLease,
    budget_key,
)
from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    LauncherChainIdentity,
    ProcessCleanupError,
    ProcessExecutionPolicy,
    ProcessLaunchLifecycleError,
    ProcessLaunchNotAuthorizedError,
    ProcessLaunchPermit,
    execute_process,
    fingerprint_environment,
    resolve_executable,
    resolve_launcher_chain,
    runtime_capabilities,
    validate_environment_name,
)
from .assistant_bridge_resources import (
    VerifierResourceCapabilities,
    VerifierResourceEnforcementReport,
    VerifierResourceError,
    VerifierResourcePlan,
    VerifierResourcePolicy,
    build_verifier_resource_enforcement_report,
    build_verifier_resource_plan,
    verifier_resource_capabilities,
)
from .assistant_bridge_secrets import (
    ResidualAssuranceUnavailableError,
    SecretRedactionPolicy,
    redact_text,
    redact_user_controlled_fields,
)
from .assistant_bridge_verifier_isolation import (
    VerifierIsolationError,
    VerifierIsolationPlan,
    VerifierIsolationPolicy,
    build_verifier_isolation_plan,
    expand_runtime_read_roots,
    verifier_isolation_capability,
)
from .assistant_bridge_workspace import (
    IgnoredPathRule,
    MaterializedWorkspace,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    WorkspaceWriteCapability,
    WorkspaceChange,
    WorkspaceFile,
    WorkspaceSnapshot,
    apply_changeset,
    build_changeset,
    materialize_workspace,
    snapshot_workspace,
    trusted_git_executable,
    trusted_git_session,
    workspace_write_capability,
)


BRIDGE_SCHEMA_VERSION = "2.0"
ROUTES = {"blocked", "local", "local_then_verify", "premium"}
PROFILES = {"balanced", "economy", "offline", "privacy", "quality"}
RISK_LEVELS = {
    "read_only": 0,
    "compute_only": 1,
    "write_local": 2,
    "write_external": 3,
    "destructive": 4,
    "privileged": 5,
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+./:_@-]{0,255}$")
_SAFE_ENV = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_GIT_BYTES = 16 * 1024 * 1024
_MAX_STREAM_BYTES = 2 * 1024 * 1024
_MAX_FINAL_BYTES = 1024 * 1024
_PYTHON_UNITTEST_BOOTSTRAP = r"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


def _read_file(path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("runner entry is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("runner entry changed during attestation")
    return [
        path.name,
        int(after.st_size),
        stat.S_IMODE(after.st_mode),
        digest.hexdigest(),
    ]


def _manifest(root):
    records = []
    total = 0
    count = 0
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directories.sort()
        files.sort()
        for name in directories:
            metadata = current.joinpath(name).lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError("runner tree contains an unsafe directory")
        for name in files:
            path = current.joinpath(name)
            record = _read_file(path)
            record[0] = path.relative_to(root).as_posix()
            records.append(record)
            count += 1
            total += record[1]
            if count > 4096 or total > 64 * 1024 * 1024:
                raise RuntimeError("runner tree exceeds its attestation bound")
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


runner_root = Path(sys.argv[1]).resolve(strict=True)
expected_manifest = sys.argv[2]
if _manifest(runner_root) != expected_manifest:
    raise SystemExit(86)
import unittest
if Path(unittest.__file__).resolve(strict=True) != runner_root.joinpath("__init__.py"):
    raise SystemExit(87)
path_count = int(sys.argv[3])
candidate_roots = sys.argv[4:4 + path_count]
runner_args = sys.argv[4 + path_count:]
sys.path[:0] = candidate_roots
sys.argv = ["unittest", *runner_args]
program = unittest.main(module=None, exit=False)
result = getattr(program, "result", None)
raise SystemExit(0 if result is not None and result.wasSuccessful() else 1)
""".strip()
_PYTHON_RUNNER_BOOTSTRAP_ENV = "MYMOE_TYPED_RUNNER_BOOTSTRAP"
_PYTHON_RUNNER_LAUNCH = (
    "exec(__import__('os').environ['MYMOE_TYPED_RUNNER_BOOTSTRAP'])"
)
_PYTHON_RUNNER_MAX_FILES = 4096
_PYTHON_RUNNER_MAX_BYTES = 64 * 1024 * 1024
_BASE_ENV_KEYS = {
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "WINDIR",
}
class AssistantBridgeError(ValueError):
    """Raised when a bridge contract, policy, or binding is invalid."""


@dataclass(frozen=True)
class CapabilityDemand:
    required: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    risk_class: str = "read_only"

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "tools", tuple(self.tools))
        _validate_identifiers("required capability", self.required)
        _validate_identifiers("required tool", self.tools)
        if self.risk_class not in RISK_LEVELS:
            supported = ", ".join(sorted(RISK_LEVELS))
            raise AssistantBridgeError(
                f"Unsupported risk class {self.risk_class!r}; use one of: {supported}."
            )

    def payload(self) -> dict[str, object]:
        return {
            "required": list(self.required),
            "tools": list(self.tools),
            "risk_class": self.risk_class,
        }


@dataclass(frozen=True)
class AssistantTaskBudget:
    max_premium_calls: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_premium_calls, bool):
            raise AssistantBridgeError(
                "max_premium_calls must be an integer, not boolean."
            )
        if self.max_premium_calls is not None and not 0 <= self.max_premium_calls <= 1:
            raise AssistantBridgeError(
                "max_premium_calls must be 0 or 1 for this single-escalation contract."
            )


@dataclass(frozen=True)
class AssistantTaskEnvelope:
    objective: str = field(repr=False)
    profile: str = "balanced"
    capability_demand: CapabilityDemand = field(default_factory=CapabilityDemand)
    constraints: tuple[str, ...] = field(repr=False, default=())
    required_verifier_ids: tuple[str, ...] = ()
    no_change_expected: bool = False
    allow_remote: bool | None = None
    allow_remote_workspace: bool = False
    budget: AssistantTaskBudget = field(default_factory=AssistantTaskBudget)
    task_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.no_change_expected, bool):
            raise AssistantBridgeError("no_change_expected must be boolean.")
        if self.allow_remote is not None and not isinstance(self.allow_remote, bool):
            raise AssistantBridgeError("allow_remote must be boolean or omitted.")
        if not isinstance(self.allow_remote_workspace, bool):
            raise AssistantBridgeError("allow_remote_workspace must be boolean.")
        if self.allow_remote_workspace and self.allow_remote is not True:
            raise AssistantBridgeError(
                "allow_remote_workspace requires explicit allow_remote=true."
            )
        objective = self.objective.strip()
        if not objective:
            raise AssistantBridgeError("Assistant task objective is required.")
        if len(objective) > 200_000:
            raise AssistantBridgeError(
                "Assistant task objective exceeds 200000 characters."
            )
        if self.profile not in PROFILES:
            supported = ", ".join(sorted(PROFILES))
            raise AssistantBridgeError(
                f"Unsupported assistant profile {self.profile!r}; use one of: {supported}."
            )
        if self.task_id and _SAFE_ID.fullmatch(self.task_id) is None:
            raise AssistantBridgeError(
                "task_id must contain 1-96 safe identifier characters."
            )
        if len(self.constraints) > 32:
            raise AssistantBridgeError("Assistant task accepts at most 32 constraints.")
        clean_constraints = tuple(
            item.strip() for item in self.constraints if item.strip()
        )
        if any(len(item) > 4000 for item in clean_constraints):
            raise AssistantBridgeError(
                "Each assistant constraint is limited to 4000 characters."
            )
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "constraints", clean_constraints)
        object.__setattr__(
            self, "required_verifier_ids", tuple(self.required_verifier_ids)
        )
        _validate_identifiers("required verifier", self.required_verifier_ids)
        if len(set(self.required_verifier_ids)) != len(self.required_verifier_ids):
            raise AssistantBridgeError("required_verifier_ids contains duplicates.")
        if not self.task_id:
            object.__setattr__(self, "task_id", f"task-{self.task_fingerprint[:24]}")

    @property
    def objective_sha256(self) -> str:
        return _sha256_text(self.objective)

    @property
    def task_fingerprint(self) -> str:
        return _sha256_text(
            _canonical_json(
                {
                    "objective": self.objective,
                    "profile": self.profile,
                    "capability_demand": self.capability_demand.payload(),
                    "constraints": list(self.constraints),
                    "no_change_expected": self.no_change_expected,
                    "required_verifier_ids": list(self.required_verifier_ids),
                    "allow_remote": self.allow_remote,
                    "allow_remote_workspace": self.allow_remote_workspace,
                    "max_premium_calls": self.budget.max_premium_calls,
                }
            )
        )

    def metadata_payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "objective_sha256": self.objective_sha256,
            "task_fingerprint": self.task_fingerprint,
            "objective_chars": len(self.objective),
            "profile": self.profile,
            "capability_demand": self.capability_demand.payload(),
            "constraint_count": len(self.constraints),
            "no_change_expected": self.no_change_expected,
            "required_verifier_ids": list(self.required_verifier_ids),
            "allow_remote": self.allow_remote,
            "allow_remote_workspace": self.allow_remote_workspace,
            "max_premium_calls": self.budget.max_premium_calls,
        }


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    mode: str
    executable: str = field(repr=False)
    capabilities: tuple[str, ...]
    tools: tuple[str, ...]
    max_risk: str
    adapter: str = "codex_cli"
    execution_scope: str = "device_only"
    local_provider: str = ""
    codex_profile: str = ""
    model: str = ""
    sandbox: str = "workspace-write"
    workspace_access: str = "read_write"
    network_access: bool = False
    timeout_seconds: float = 900.0
    launcher_args: tuple[str, ...] = ()
    launcher_entrypoint: str = field(default="", repr=False)
    launcher_companions: tuple[str, ...] = field(default=(), repr=False)
    extra_args: tuple[str, ...] = ()
    environment_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "capabilities",
            "tools",
            "launcher_args",
            "launcher_companions",
            "extra_args",
            "environment_allowlist",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if _SAFE_ID.fullmatch(self.id) is None:
            raise AssistantBridgeError(
                "Provider id must contain 1-96 safe identifier characters."
            )
        if self.mode not in {"local", "premium"}:
            raise AssistantBridgeError("Provider mode must be local or premium.")
        if self.adapter != "codex_cli":
            raise AssistantBridgeError(
                f"Provider {self.id} uses unsupported adapter {self.adapter!r}."
            )
        if self.execution_scope not in {"device_only", "remote"}:
            raise AssistantBridgeError(
                f"Provider {self.id} has unsupported execution_scope."
            )
        if self.mode == "local" and self.execution_scope != "device_only":
            raise AssistantBridgeError(
                "Local providers must use execution_scope=device_only."
            )
        if self.mode == "premium" and self.execution_scope != "remote":
            raise AssistantBridgeError(
                "Premium providers must use execution_scope=remote."
            )
        if not self.executable or "\x00" in self.executable:
            raise AssistantBridgeError(
                "Provider executable is required and cannot contain NUL."
            )
        _validate_identifiers("provider capability", self.capabilities)
        _validate_identifiers("provider tool", self.tools)
        if self.max_risk not in RISK_LEVELS:
            raise AssistantBridgeError(
                f"Provider {self.id} has an unsupported max_risk."
            )
        if RISK_LEVELS[self.max_risk] > RISK_LEVELS["write_local"]:
            raise AssistantBridgeError(
                f"Provider {self.id} cannot receive authority above write_local."
            )
        if self.mode == "local" and self.local_provider not in {"lmstudio", "ollama"}:
            raise AssistantBridgeError(
                f"Local provider {self.id} must choose local_provider=ollama or lmstudio."
            )
        if self.codex_profile:
            raise AssistantBridgeError(
                f"Provider {self.id} codex_profile is incompatible with isolated execution; "
                "use a trusted executable adapter instead."
            )
        if not self.model.strip():
            raise AssistantBridgeError(
                f"Provider {self.id} requires an explicit model for capability attestation."
            )
        if self.sandbox not in {"read-only", "workspace-write"}:
            raise AssistantBridgeError(
                f"Provider {self.id} has an unsupported sandbox ceiling."
            )
        if self.workspace_access not in {"capsule_only", "read_only", "read_write"}:
            raise AssistantBridgeError(
                f"Provider {self.id} has unsupported workspace_access."
            )
        if self.mode == "local" and self.workspace_access == "capsule_only":
            raise AssistantBridgeError("Local provider requires workspace access.")
        if not isinstance(self.network_access, bool):
            raise AssistantBridgeError(
                f"Provider {self.id} network_access must be boolean."
            )
        if self.mode == "local" and self.network_access:
            raise AssistantBridgeError(
                "Local providers must disable agent-tool network access; model traffic stays loopback."
            )
        if not 1 <= self.timeout_seconds <= 86_400:
            raise AssistantBridgeError(
                f"Provider {self.id} timeout_seconds must be between 1 and 86400."
            )
        for label, values in (
            ("launcher_args", self.launcher_args),
            ("launcher_companions", self.launcher_companions),
            ("extra_args", self.extra_args),
        ):
            if any(not item or "\x00" in item for item in values):
                raise AssistantBridgeError(
                    f"Provider {self.id} {label} contains an invalid value."
                )
        if "\x00" in self.launcher_entrypoint:
            raise AssistantBridgeError(
                f"Provider {self.id} launcher_entrypoint contains an invalid value."
            )
        if len(set(self.launcher_companions)) != len(self.launcher_companions):
            raise AssistantBridgeError(
                f"Provider {self.id} launcher_companions contains duplicates."
            )
        _validate_safe_extra_args(self.extra_args, provider_id=self.id)
        if len(set(self.environment_allowlist)) != len(self.environment_allowlist):
            raise AssistantBridgeError(
                f"Provider {self.id} environment_allowlist contains duplicates."
            )
        for name in self.environment_allowlist:
            if _SAFE_ENV.fullmatch(name) is None:
                raise AssistantBridgeError(
                    f"Provider {self.id} environment allowlist contains an invalid name."
                )
            try:
                validate_environment_name(name)
            except (TypeError, ValueError):
                raise AssistantBridgeError(
                    f"Provider {self.id} environment allowlist contains a denied injection variable."
                ) from None

    def metadata_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "mode": self.mode,
            "adapter": self.adapter,
            "execution_scope": self.execution_scope,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools),
            "max_risk": self.max_risk,
            "local_provider": self.local_provider or None,
            "codex_profile": self.codex_profile or None,
            "model": self.model,
            "sandbox": self.sandbox,
            "workspace_access": self.workspace_access,
            "network_access": self.network_access,
            "timeout_seconds": self.timeout_seconds,
            "launcher_arg_count": len(self.launcher_args),
            "launcher_entrypoint_declared": bool(self.launcher_entrypoint),
            "launcher_companion_count": len(self.launcher_companions),
            "extra_arg_count": len(self.extra_args),
            "environment_keys": list(self.environment_allowlist),
            "planning_probe": "none",
        }


@dataclass(frozen=True)
class ProfilePolicy:
    name: str
    initial_route: str
    remote_allowed: bool
    explicit_remote_opt_in: bool
    max_premium_calls: int

    def __post_init__(self) -> None:
        if self.name not in PROFILES:
            raise AssistantBridgeError(f"Unknown profile policy {self.name!r}.")
        if self.initial_route not in {"local", "local_then_verify", "premium"}:
            raise AssistantBridgeError(
                f"Profile {self.name} has unsupported initial_route {self.initial_route!r}."
            )
        if not isinstance(self.remote_allowed, bool) or not isinstance(
            self.explicit_remote_opt_in, bool
        ):
            raise AssistantBridgeError(
                f"Profile {self.name} remote policy values must be boolean."
            )
        if (
            isinstance(self.max_premium_calls, bool)
            or not 0 <= self.max_premium_calls <= 1
        ):
            raise AssistantBridgeError(
                f"Profile {self.name} max_premium_calls must be 0 or 1."
            )
        if self.name == "offline" and (self.remote_allowed or self.max_premium_calls):
            raise AssistantBridgeError(
                "Offline profile must hard-disable remote calls."
            )
        if self.name == "privacy" and not self.explicit_remote_opt_in:
            raise AssistantBridgeError(
                "Privacy profile requires explicit remote opt-in."
            )


@dataclass(frozen=True)
class CapsulePolicy:
    max_chars: int = 8000
    max_objective_chars: int = 3000
    max_constraint_chars: int = 1000
    max_diff_chars: int = 2500
    secret_redaction: SecretRedactionPolicy = field(
        default_factory=SecretRedactionPolicy,
        repr=False,
    )

    def __post_init__(self) -> None:
        values = (
            self.max_chars,
            self.max_objective_chars,
            self.max_constraint_chars,
            self.max_diff_chars,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise AssistantBridgeError("Capsule limits must be integers.")
        if not 1000 <= self.max_chars <= 100_000:
            raise AssistantBridgeError(
                "capsule.max_chars must be between 1000 and 100000."
            )
        for name, value in (
            ("max_objective_chars", self.max_objective_chars),
            ("max_constraint_chars", self.max_constraint_chars),
            ("max_diff_chars", self.max_diff_chars),
        ):
            if not 0 <= value <= self.max_chars:
                raise AssistantBridgeError(f"capsule.{name} must fit inside max_chars.")
        if not self.secret_redaction.require_residual_assurance:
            raise AssistantBridgeError(
                "Capsule residual secret assurance cannot be disabled."
            )


@dataclass(frozen=True)
class PythonRunnerIdentity:
    name: str
    module_root: str = field(repr=False)
    module_root_sha256: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    device_id: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)

    def __post_init__(self) -> None:
        if self.name != "unittest":
            raise AssistantBridgeError("Typed Python runner identity is unsupported.")
        _require_sha256(self.module_root_sha256, "runner module_root_sha256")
        _require_sha256(self.manifest_sha256, "runner manifest_sha256")
        if self.file_count < 1 or self.total_bytes < 1:
            raise AssistantBridgeError("Typed Python runner manifest is empty.")

    def binding_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module_root": self.module_root,
            "module_root_sha256": self.module_root_sha256,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "device_id": self.device_id,
            "inode": self.inode,
            "mode": self.mode,
        }

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module_root_sha256": self.module_root_sha256,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "identity_sha256": _sha256_text(
                _canonical_json(self.binding_payload())
            ),
        }


@dataclass(frozen=True)
class CommandVerifierSpec:
    id: str
    argv: tuple[str, ...] = field(repr=False)
    timeout_seconds: float
    kind: str = "command"
    purpose: str = "hygiene"
    execution_boundary: str = "hard_sandbox"
    network_policy: str = "denied"
    launcher_entrypoint: str = field(default="", repr=False)
    launcher_companions: tuple[str, ...] = field(default=(), repr=False)
    runtime_read_roots: tuple[str, ...] = field(
        default=("{python_runtime}",), repr=False
    )
    python_runner: str = ""
    workspace_python_paths: tuple[str, ...] = field(default=(), repr=False)
    environment_allowlist: tuple[str, ...] = ()
    required_for_capabilities: tuple[str, ...] = ()
    required_for_tools: tuple[str, ...] = ()
    required_for_risks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "argv",
            "launcher_companions",
            "runtime_read_roots",
            "workspace_python_paths",
            "environment_allowlist",
            "required_for_capabilities",
            "required_for_tools",
            "required_for_risks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if _SAFE_ID.fullmatch(self.id) is None:
            raise AssistantBridgeError("Command verifier id must be a safe identifier.")
        if self.kind not in {"command", "trusted_git_diff_check"}:
            raise AssistantBridgeError(
                f"Command verifier {self.id} has an unsupported kind."
            )
        if self.kind == "command" and (
            not self.argv or any(not item or "\x00" in item for item in self.argv)
        ):
            raise AssistantBridgeError(
                f"Command verifier {self.id} requires safe argv values."
            )
        if self.kind == "trusted_git_diff_check" and self.argv:
            raise AssistantBridgeError(
                f"Trusted Git verifier {self.id} does not accept configurable argv."
            )
        if "\x00" in self.launcher_entrypoint or any(
            not item or "\x00" in item for item in self.launcher_companions
        ):
            raise AssistantBridgeError(
                f"Command verifier {self.id} launcher declarations are invalid."
            )
        if len(set(self.launcher_companions)) != len(self.launcher_companions):
            raise AssistantBridgeError(
                f"Command verifier {self.id} launcher companions contain duplicates."
            )
        if not 1 <= self.timeout_seconds <= 3600:
            raise AssistantBridgeError(
                f"Command verifier {self.id} timeout_seconds must be between 1 and 3600."
            )
        if self.purpose not in {"hygiene", "task"}:
            raise AssistantBridgeError(
                f"Command verifier {self.id} purpose must be hygiene or task."
            )
        expected_boundary = (
            "trusted_git_session"
            if self.kind == "trusted_git_diff_check"
            else "hard_sandbox"
        )
        if self.execution_boundary != expected_boundary:
            raise AssistantBridgeError(
                f"Command verifier {self.id} requires the {expected_boundary} boundary."
            )
        if self.network_policy != "denied":
            raise AssistantBridgeError(
                f"Command verifier {self.id} has unsupported network policy."
            )
        for name in self.environment_allowlist:
            if _SAFE_ENV.fullmatch(name) is None:
                raise AssistantBridgeError(
                    f"Command verifier {self.id} environment allowlist is invalid."
                )
            try:
                validate_environment_name(name)
            except (TypeError, ValueError):
                raise AssistantBridgeError(
                    f"Command verifier {self.id} environment allowlist contains "
                    "a denied injection variable."
                ) from None
        if self.kind == "trusted_git_diff_check" and any(
            (
                self.launcher_entrypoint,
                self.launcher_companions,
                self.runtime_read_roots,
                self.python_runner,
                self.workspace_python_paths,
                self.environment_allowlist,
            )
        ):
            raise AssistantBridgeError(
                f"Trusted Git verifier {self.id} cannot extend its fixed execution contract."
            )
        if self.kind == "trusted_git_diff_check" and self.purpose != "hygiene":
            raise AssistantBridgeError(
                f"Trusted Git verifier {self.id} must be a hygiene verifier."
            )
        for root in self.runtime_read_roots:
            if not root or "\x00" in root:
                raise AssistantBridgeError(
                    f"Command verifier {self.id} runtime read roots are invalid."
                )
        if len(set(self.workspace_python_paths)) != len(
            self.workspace_python_paths
        ):
            raise AssistantBridgeError(
                f"Command verifier {self.id} workspace Python paths contain duplicates."
            )
        for relative in self.workspace_python_paths:
            path = PurePosixPath(relative)
            if (
                not relative
                or "\x00" in relative
                or "\\" in relative
                or ":" in relative
                or bool(path.drive)
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != relative
            ):
                raise AssistantBridgeError(
                    f"Command verifier {self.id} workspace Python paths must be "
                    "safe relative paths."
                )
        if self.python_runner not in {"", "unittest"}:
            raise AssistantBridgeError(
                f"Command verifier {self.id} has an unsupported typed Python runner."
            )
        if self.workspace_python_paths and not self.python_runner:
            raise AssistantBridgeError(
                f"Command verifier {self.id} workspace Python paths require a "
                "typed Python runner."
            )
        if self.python_runner and (
            self.kind != "command"
            or len(self.argv) < 3
            or self.argv[:3] != ("{python}", "-m", self.python_runner)
            or bool(self.launcher_entrypoint)
            or bool(self.launcher_companions)
        ):
            raise AssistantBridgeError(
                f"Command verifier {self.id} typed Python runner contract is invalid."
            )
        _validate_identifiers("verifier capability", self.required_for_capabilities)
        _validate_identifiers("verifier tool", self.required_for_tools)
        unknown_risks = sorted(set(self.required_for_risks).difference(RISK_LEVELS))
        if unknown_risks:
            raise AssistantBridgeError(
                f"Command verifier {self.id} has unsupported risks: {', '.join(unknown_risks)}."
            )
        if self.purpose == "task" and any(
            (
                self.required_for_capabilities,
                self.required_for_tools,
                self.required_for_risks,
            )
        ):
            raise AssistantBridgeError(
                f"Task verifier {self.id} must be selected explicitly by id."
            )

    @property
    def spec_sha256(self) -> str:
        return _sha256_text(_canonical_json(self.payload()))

    def applies_to(self, demand: CapabilityDemand) -> bool:
        if self.purpose != "hygiene":
            return False
        return bool(
            set(self.required_for_capabilities).intersection(demand.required)
            or set(self.required_for_tools).intersection(demand.tools)
            or demand.risk_class in self.required_for_risks
        )

    def payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "argv_sha256": _sha256_text(_canonical_json(list(self.argv))),
            "timeout_seconds": self.timeout_seconds,
            "purpose": self.purpose,
            "execution_boundary": self.execution_boundary,
            "network_policy": self.network_policy,
            "launcher_entrypoint_sha256": (
                _sha256_text(self.launcher_entrypoint)
                if self.launcher_entrypoint
                else None
            ),
            "launcher_companions_sha256": _sha256_text(
                _canonical_json(list(self.launcher_companions))
            ),
            "launcher_companion_count": len(self.launcher_companions),
            "runtime_read_roots_sha256": _sha256_text(
                _canonical_json(list(self.runtime_read_roots))
            ),
            "runtime_read_root_count": len(self.runtime_read_roots),
            "python_runner": self.python_runner or None,
            "workspace_python_paths_sha256": _sha256_text(
                _canonical_json(list(self.workspace_python_paths))
            ),
            "workspace_python_path_count": len(self.workspace_python_paths),
            "environment_keys": list(self.environment_allowlist),
            "required_for_capabilities": list(self.required_for_capabilities),
            "required_for_tools": list(self.required_for_tools),
            "required_for_risks": list(self.required_for_risks),
        }


@dataclass(frozen=True)
class ExternalVerifierSpec:
    id: str
    verifier: str
    spec_sha256: str

    def __post_init__(self) -> None:
        if (
            _SAFE_ID.fullmatch(self.id) is None
            or _SAFE_ID.fullmatch(self.verifier) is None
        ):
            raise AssistantBridgeError(
                "External verifier identity must use safe identifiers."
            )
        _require_sha256(self.spec_sha256, "external verifier spec_sha256")


@dataclass(frozen=True)
class WorkspaceAttestation:
    root: str = field(repr=False)
    fingerprint: str
    git_repository: bool
    head_sha: str
    index_sha256: str
    status_sha256: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    scope: str = "tracked_untracked_nonignored_plus_declared_ignored"

    def payload(self) -> dict[str, object]:
        return {
            "root_sha256": _sha256_text(self.root),
            "fingerprint": self.fingerprint,
            "git_repository": self.git_repository,
            "head_sha": self.head_sha or None,
            "index_sha256": self.index_sha256,
            "status_sha256": self.status_sha256,
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class DiffEvidence:
    sha256: str
    characters: int
    excerpt: str = field(repr=False)
    truncated: bool
    staged_sha256: str
    unstaged_sha256: str
    untracked_manifest_sha256: str

    @property
    def available(self) -> bool:
        return self.characters > 0

    def payload(self) -> dict[str, object]:
        return {
            "available": self.available,
            "sha256": self.sha256 or None,
            "characters": self.characters,
            "excerpt": self.excerpt if self.available else None,
            "truncated": self.truncated,
            "staged_sha256": self.staged_sha256 or None,
            "unstaged_sha256": self.unstaged_sha256 or None,
            "untracked_manifest_sha256": self.untracked_manifest_sha256 or None,
        }


@dataclass(frozen=True)
class VerificationEvidence:
    id: str
    verifier: str
    kind: str
    passed: bool
    code: str
    artifact_sha256: str
    task_fingerprint: str
    workspace_fingerprint: str
    verifier_spec_sha256: str
    observed_chars: int = 0
    evidence_ref: str = field(repr=False, default="")

    def __post_init__(self) -> None:
        for name, value in (
            ("id", self.id),
            ("verifier", self.verifier),
            ("code", self.code),
        ):
            if _SAFE_ID.fullmatch(value) is None:
                raise AssistantBridgeError(
                    f"Verification evidence {name} must contain safe identifier characters."
                )
        if self.kind not in {
            "builtin",
            "command",
            "external",
            "output",
            "process",
            "policy",
        }:
            raise AssistantBridgeError("Verification evidence kind is unsupported.")
        if not isinstance(self.passed, bool):
            raise AssistantBridgeError("Verification evidence passed must be boolean.")
        for label, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("task_fingerprint", self.task_fingerprint),
            ("workspace_fingerprint", self.workspace_fingerprint),
            ("verifier_spec_sha256", self.verifier_spec_sha256),
        ):
            _require_sha256(value, f"verification {label}")
        if isinstance(self.observed_chars, bool) or self.observed_chars < 0:
            raise AssistantBridgeError(
                "Verification observed_chars cannot be negative."
            )
        if self.evidence_ref and _SAFE_REF.fullmatch(self.evidence_ref) is None:
            raise AssistantBridgeError(
                "Verification evidence_ref contains unsafe characters."
            )

    def payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "verifier": self.verifier,
            "kind": self.kind,
            "passed": self.passed,
            "code": self.code,
            "artifact_sha256": self.artifact_sha256,
            "observed_chars": self.observed_chars,
            "evidence_ref": self.evidence_ref or None,
            "task_fingerprint": self.task_fingerprint,
            "workspace_fingerprint": self.workspace_fingerprint,
            "verifier_spec_sha256": self.verifier_spec_sha256,
        }


@dataclass(frozen=True)
class BridgeRuntimePolicy:
    require_tree_isolation: bool = True
    require_psutil: bool = True
    require_launcher_chain: bool = True
    stdin_limit_bytes: int = 8 * 1024 * 1024
    stdout_limit_bytes: int = _MAX_STREAM_BYTES
    stderr_limit_bytes: int = _MAX_STREAM_BYTES
    cleanup_grace_seconds: float = 0.25
    cleanup_kill_seconds: float = 0.75

    def __post_init__(self) -> None:
        if any(
            value is not True
            for value in (
                self.require_tree_isolation,
                self.require_psutil,
                self.require_launcher_chain,
            )
        ):
            raise AssistantBridgeError(
                "Bridge runtime must require tree isolation, psutil, and launcher chains."
            )
        try:
            self.process_policy()
        except ValueError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def process_policy(self, *, stdin_limit_bytes: int | None = None) -> ProcessExecutionPolicy:
        return ProcessExecutionPolicy(
            stdin_limit_bytes=(
                self.stdin_limit_bytes
                if stdin_limit_bytes is None
                else stdin_limit_bytes
            ),
            stdout_limit_bytes=self.stdout_limit_bytes,
            stderr_limit_bytes=self.stderr_limit_bytes,
            cleanup_grace_seconds=self.cleanup_grace_seconds,
            cleanup_kill_seconds=self.cleanup_kill_seconds,
            require_tree_isolation=self.require_tree_isolation,
            require_psutil=self.require_psutil,
            require_launcher_chain=self.require_launcher_chain,
        )

    def payload(self) -> dict[str, object]:
        return {
            "require_tree_isolation": self.require_tree_isolation,
            "require_psutil": self.require_psutil,
            "require_launcher_chain": self.require_launcher_chain,
            "stdin_limit_bytes": self.stdin_limit_bytes,
            "stdout_limit_bytes": self.stdout_limit_bytes,
            "stderr_limit_bytes": self.stderr_limit_bytes,
            "cleanup_grace_seconds": self.cleanup_grace_seconds,
            "cleanup_kill_seconds": self.cleanup_kill_seconds,
        }


@dataclass(frozen=True)
class BridgeStatePolicy:
    ledger_path: str = field(repr=False)
    namespace: str
    confirmation_ttl_seconds: float = 300.0
    lock_timeout_seconds: float = 5.0
    stale_lock_seconds: float = 120.0
    budget_retention_seconds: float = 90 * 24 * 60 * 60
    max_budget_entries: int = 4096
    confirmation_retention_seconds: float = 24 * 60 * 60
    max_confirmation_entries: int = 4096
    budget_lease_ttl_seconds: float = 60.0

    def __post_init__(self) -> None:
        try:
            self.ledger()
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(str(exc)) from None
        if not 1 <= self.confirmation_ttl_seconds <= 3600:
            raise AssistantBridgeError(
                "state.confirmation_ttl_seconds must be between 1 and 3600."
            )

    def ledger(self) -> BridgeStateLedger:
        return BridgeStateLedger(
            self.ledger_path,
            namespace=self.namespace,
            lock_timeout_seconds=self.lock_timeout_seconds,
            stale_lock_seconds=self.stale_lock_seconds,
            budget_retention_seconds=self.budget_retention_seconds,
            max_budget_entries=self.max_budget_entries,
            confirmation_retention_seconds=self.confirmation_retention_seconds,
            max_confirmation_entries=self.max_confirmation_entries,
            budget_lease_ttl_seconds=self.budget_lease_ttl_seconds,
        )

    def effective_descriptor(self) -> dict[str, object]:
        descriptor = self.ledger().effective_descriptor()
        descriptor["confirmation_ttl_seconds"] = self.confirmation_ttl_seconds
        return descriptor


@dataclass(frozen=True)
class BridgeWorkspacePolicy:
    scope: WorkspaceScopePolicy
    transaction_state_dir: str = field(repr=False)
    transaction_lock_ttl_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.transaction_state_dir:
            raise AssistantBridgeError(
                "workspace.transaction_state_dir is required."
            )
        if not 1 <= self.transaction_lock_ttl_seconds <= 86_400:
            raise AssistantBridgeError(
                "workspace.transaction_lock_ttl_seconds is outside safe bounds."
            )

    def effective_descriptor(self) -> dict[str, object]:
        return {
            "max_files": self.scope.max_files,
            "max_total_bytes": self.scope.max_total_bytes,
            "max_file_bytes": self.scope.max_file_bytes,
            "ignored_paths": [
                {"path": item.path, "direction": item.direction}
                for item in self.scope.ignored_paths
            ],
            "transaction_state_dir_sha256": _sha256_text(
                str(Path(self.transaction_state_dir).resolve())
            ),
            "transaction_lock_ttl_seconds": self.transaction_lock_ttl_seconds,
        }


@dataclass(frozen=True)
class AssistantBridgeConfig:
    local: ProviderSpec
    premium: ProviderSpec
    profiles: Mapping[str, ProfilePolicy]
    verification_checks: tuple[Mapping[str, Any], ...]
    command_verifiers: tuple[CommandVerifierSpec, ...]
    external_verifiers: Mapping[str, ExternalVerifierSpec]
    independent_capabilities: tuple[str, ...]
    independent_tools: tuple[str, ...]
    independent_risks: tuple[str, ...]
    verifier_isolation: VerifierIsolationPolicy
    verifier_resources: VerifierResourcePolicy
    capsule: CapsulePolicy
    runtime: BridgeRuntimePolicy
    state: BridgeStatePolicy
    workspace: BridgeWorkspacePolicy
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(
            self,
            "verification_checks",
            tuple(_deep_freeze(dict(check)) for check in self.verification_checks),
        )
        object.__setattr__(
            self,
            "external_verifiers",
            MappingProxyType(dict(self.external_verifiers)),
        )
        for name in (
            "command_verifiers",
            "independent_capabilities",
            "independent_tools",
            "independent_risks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        missing = sorted(PROFILES.difference(self.profiles))
        if missing:
            raise AssistantBridgeError(
                f"Assistant bridge config is missing profiles: {', '.join(missing)}."
            )
        if self.local.mode != "local" or self.premium.mode != "premium":
            raise AssistantBridgeError(
                "Bridge providers must define local and premium modes."
            )
        if not self.verification_checks:
            raise AssistantBridgeError(
                "At least one deterministic output check is required."
            )
        _validate_identifiers("independent capability", self.independent_capabilities)
        _validate_identifiers("independent tool", self.independent_tools)
        if set(self.independent_risks).difference(RISK_LEVELS):
            raise AssistantBridgeError(
                "Independent evidence policy contains unknown risks."
            )

    @property
    def budget_ledger_path(self) -> str:
        """Compatibility alias for callers that inspected the v1 config."""

        return self.state.ledger_path


@dataclass(frozen=True)
class RouteDecisionReceipt:
    receipt_id: str
    task: Mapping[str, object]
    route: str
    local_provider: str
    premium_provider: str | None
    local_gaps: tuple[str, ...]
    premium_gaps: tuple[str, ...]
    remote_allowed: bool
    premium_call_budget: int
    rationale_codes: tuple[str, ...]
    expected_flow: tuple[str, ...]
    config_sha256: str
    workspace: WorkspaceAttestation
    local_runtime: Mapping[str, object]
    premium_runtime: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise AssistantBridgeError(
                "RouteDecisionReceipt contains an invalid route."
            )
        object.__setattr__(self, "task", _deep_freeze(self.task))
        object.__setattr__(self, "local_runtime", _deep_freeze(self.local_runtime))
        object.__setattr__(self, "premium_runtime", _deep_freeze(self.premium_runtime))
        for name in (
            "local_gaps",
            "premium_gaps",
            "rationale_codes",
            "expected_flow",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "contract": "RouteDecisionReceipt",
            "receipt_id": self.receipt_id,
            "task": _deep_thaw(self.task),
            "route": self.route,
            "local_provider": self.local_provider,
            "premium_provider": self.premium_provider,
            "local_gaps": list(self.local_gaps),
            "premium_gaps": list(self.premium_gaps),
            "remote_allowed": self.remote_allowed,
            "premium_call_budget": self.premium_call_budget,
            "rationale_codes": list(self.rationale_codes),
            "expected_flow": list(self.expected_flow),
            "config_sha256": self.config_sha256,
            "workspace": self.workspace.payload(),
            "local_runtime": _deep_thaw(self.local_runtime),
            "premium_runtime": _deep_thaw(self.premium_runtime),
        }


@dataclass(frozen=True)
class CommandPlan:
    provider_id: str
    mode: str
    argv: tuple[str, ...] = field(repr=False)
    stdin_sha256: str
    stdin_chars: int
    workspace: str = field(repr=False)
    output_path: str = field(repr=False)
    command_sha256: str
    sandbox: str
    network_access: bool
    workspace_access: str
    model: str
    local_provider: str
    environment_allowlist: tuple[str, ...]
    executable_identity: ExecutableIdentity = field(repr=False)
    environment_sha256: str
    runtime: Mapping[str, object]
    runtime_policy: BridgeRuntimePolicy = field(repr=False)
    launcher_chain: LauncherChainIdentity = field(repr=False)
    launcher_authority_sha256: str
    ephemeral_workspace: bool
    launcher_artifact_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self, "environment_allowlist", tuple(self.environment_allowlist)
        )
        object.__setattr__(
            self, "launcher_artifact_sha256", tuple(self.launcher_artifact_sha256)
        )
        object.__setattr__(self, "runtime", _deep_freeze(self.runtime))

    def payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "mode": self.mode,
            "argv_sha256": _sha256_text(_canonical_json(list(self.argv))),
            "argv_shape": _redacted_argv_shape(self.argv),
            "stdin": {
                "transport": "stdin",
                "sha256": self.stdin_sha256,
                "characters": self.stdin_chars,
                "content_in_argv": False,
            },
            "workspace_sha256": _sha256_text(self.workspace),
            "output_path_sha256": (
                _sha256_text(self.output_path) if self.output_path else None
            ),
            "command_sha256": self.command_sha256,
            "sandbox": self.sandbox,
            "network_access": self.network_access,
            "workspace_access": self.workspace_access,
            "model": self.model,
            "local_provider": self.local_provider or None,
            "environment_keys": list(self.environment_allowlist),
            "executable": _public_executable_payload(self.executable_identity),
            "environment_sha256": self.environment_sha256,
            "runtime": _deep_thaw(self.runtime),
            "runtime_policy": self.runtime_policy.payload(),
            "launcher_chain": self.launcher_chain.payload(),
            "launcher_authority_sha256": self.launcher_authority_sha256,
            "launcher_artifact_sha256": list(self.launcher_artifact_sha256),
        }


@dataclass(frozen=True)
class BoundVerifierPlan:
    spec: CommandVerifierSpec
    argv: tuple[str, ...] = field(repr=False)
    executable_identity: ExecutableIdentity = field(repr=False)
    environment_sha256: str
    environment: Mapping[str, str] = field(repr=False)
    launcher_chain: LauncherChainIdentity | None = field(repr=False)
    launcher_authority_sha256: str
    launcher_artifact_sha256: tuple[str, ...]
    python_runner_identity: PythonRunnerIdentity | None = field(repr=False)
    isolation: VerifierIsolationPlan | None = field(repr=False)
    sandbox_launcher_chain: LauncherChainIdentity | None = field(repr=False)
    sandbox_launcher_authority_sha256: str
    sandbox_launcher_artifact_sha256: tuple[str, ...]
    resources: VerifierResourcePlan | None = field(repr=False)
    plan_sha256: str
    runtime_policy: BridgeRuntimePolicy = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(
            self, "launcher_artifact_sha256", tuple(self.launcher_artifact_sha256)
        )
        object.__setattr__(
            self,
            "sandbox_launcher_artifact_sha256",
            tuple(self.sandbox_launcher_artifact_sha256),
        )

    def payload(self) -> dict[str, object]:
        return {
            "id": self.spec.id,
            "kind": self.spec.kind,
            "purpose": self.spec.purpose,
            "spec_sha256": self.spec.spec_sha256,
            "plan_sha256": self.plan_sha256,
            "executable": _public_executable_payload(self.executable_identity),
            "environment_sha256": self.environment_sha256,
            "launcher_chain": (
                None if self.launcher_chain is None else self.launcher_chain.payload()
            ),
            "launcher_authority_sha256": self.launcher_authority_sha256,
            "launcher_artifact_sha256": list(self.launcher_artifact_sha256),
            "python_runner": (
                None
                if self.python_runner_identity is None
                else self.python_runner_identity.payload()
            ),
            "isolation": None if self.isolation is None else self.isolation.payload(),
            "sandbox_launcher_chain": (
                None
                if self.sandbox_launcher_chain is None
                else self.sandbox_launcher_chain.payload()
            ),
            "sandbox_launcher_authority_sha256": (
                self.sandbox_launcher_authority_sha256 or None
            ),
            "sandbox_launcher_artifact_sha256": list(
                self.sandbox_launcher_artifact_sha256
            ),
            "resources": None if self.resources is None else self.resources.payload(),
            "execution_boundary": self.spec.execution_boundary,
            "network_policy": self.spec.network_policy,
            "runtime_policy": self.runtime_policy.payload(),
        }


@dataclass(frozen=True)
class CommandResult:
    provider_id: str
    status: str
    code: str
    returncode: int | None
    duration_ms: int
    output: str = field(repr=False, default="")
    stdout_sha256: str = ""
    stdout_bytes: int = 0
    stderr_sha256: str = ""
    stderr_bytes: int = 0
    command_sha256: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def metadata_payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "status": self.status,
            "code": self.code,
            "returncode": self.returncode,
            "duration_ms": self.duration_ms,
            "output_sha256": _sha256_text(self.output) if self.output else None,
            "output_chars": len(self.output),
            "stdout_sha256": self.stdout_sha256 or None,
            "stdout_bytes": self.stdout_bytes,
            "stderr_sha256": self.stderr_sha256 or None,
            "stderr_bytes": self.stderr_bytes,
            "command_sha256": self.command_sha256 or None,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost": None,
                "cost_status": "not_computed_without_pricing_contract",
            },
        }


@dataclass(frozen=True)
class PremiumAuthAttestation:
    source_path: str = field(repr=False)
    sha256: str = field(repr=False)
    size_bytes: int
    content: bytes | None = field(default=None, repr=False)

    def binding_payload(self) -> dict[str, object]:
        return {
            "source_path_sha256": _sha256_text(self.source_path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class StagedPremiumAuthAttestation:
    path: str = field(repr=False)
    sha256: str = field(repr=False)
    size_bytes: int
    mtime_ns: int = field(repr=False)
    device_id: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    owner_uid: int | None = field(repr=False)
    owner_gid: int | None = field(repr=False)
    source_binding_sha256: str
    hard_containment: bool = field(default=False, init=False)
    verification_scope: str = field(
        default="pre_reservation_same_user_change_detection",
        init=False,
    )

    def binding_payload(self) -> dict[str, object]:
        return {
            "path_sha256": _sha256_text(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device_id": self.device_id,
            "inode": self.inode,
            "mode": self.mode,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid,
            "source_binding_sha256": self.source_binding_sha256,
            "hard_containment": self.hard_containment,
            "verification_scope": self.verification_scope,
        }

    def metadata_payload(self) -> dict[str, object]:
        return {
            "path_sha256": _sha256_text(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_binding_sha256": self.source_binding_sha256,
            "hard_containment": False,
            "verification_scope": self.verification_scope,
        }


@dataclass(frozen=True)
class EscalationCapsule:
    capsule_id: str
    task_id: str
    task_fingerprint: str
    objective: str = field(repr=False)
    objective_sha256: str
    capability_demand: CapabilityDemand = field(repr=False)
    constraints: tuple[str, ...] = field(repr=False)
    route_receipt_id: str
    workspace_fingerprint: str
    verification: tuple[VerificationEvidence, ...] = field(repr=False)
    failure_codes: tuple[str, ...] = field(repr=False)
    diff: DiffEvidence = field(repr=False)
    redaction_count: int
    residual_assured: bool
    residual_detector: str
    truncated: bool
    public_payload: Mapping[str, object] = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("constraints", "verification", "failure_codes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "public_payload", _deep_freeze(self.public_payload))

    def payload(self) -> dict[str, object]:
        payload = _deep_thaw(self.public_payload)
        if not isinstance(payload, dict):
            raise AssistantBridgeError("Escalation capsule public payload is invalid.")
        return payload

    def metadata_payload(self) -> dict[str, object]:
        public = self.payload()
        serialized = _canonical_json(public)
        failure_codes = public.get("failure_codes", [])
        return {
            "capsule_id": self.capsule_id,
            "sha256": _sha256_text(serialized),
            "characters": len(serialized),
            "objective_sha256": self.objective_sha256,
            "constraint_count": len(self.constraints),
            "verification_count": len(self.verification),
            "failure_codes": list(failure_codes)
            if isinstance(failure_codes, list)
            else [],
            "diff_sha256": self.diff.sha256 or None,
            "redaction_count": self.redaction_count,
            "residual_assured": self.residual_assured,
            "residual_detector": self.residual_detector,
            "truncated": self.truncated,
            "content_in_metadata": False,
        }


@dataclass(frozen=True)
class BridgeRunResult:
    status: str
    code: str
    receipt: RouteDecisionReceipt
    prior_verification: tuple[VerificationEvidence, ...] = ()
    verification: tuple[VerificationEvidence, ...] = ()
    commands: tuple[CommandResult, ...] = ()
    capsule: EscalationCapsule | None = None
    final_provider: str | None = None
    final_output: str = field(repr=False, default="")
    premium_calls_used: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_verification",
            tuple(self.prior_verification),
        )
        object.__setattr__(self, "verification", tuple(self.verification))
        object.__setattr__(self, "commands", tuple(self.commands))

    def metadata_payload(self) -> dict[str, object]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "mode": "assistant_bridge",
            "status": self.status,
            "code": self.code,
            "route_receipt": self.receipt.payload(),
            "verification": {
                "prior": [
                    item.payload() for item in self.prior_verification
                ],
                "final": [item.payload() for item in self.verification],
            },
            "commands": [item.metadata_payload() for item in self.commands],
            "capsule": self.capsule.metadata_payload() if self.capsule else None,
            "final_provider": self.final_provider,
            "premium_calls_used": self.premium_calls_used,
            "privacy": "metadata_only",
        }

    def user_payload(self) -> dict[str, object]:
        return {
            "telemetry": self.metadata_payload(),
            "result": {
                "content": self.final_output,
                "sha256": _sha256_text(self.final_output)
                if self.final_output
                else None,
                "characters": len(self.final_output),
            },
        }


def load_assistant_task(path: str | Path) -> AssistantTaskEnvelope:
    raw = _load_json_object(path, label="assistant task")
    _reject_unknown(
        "assistant task",
        raw,
        {
            "allow_remote",
            "allow_remote_workspace",
            "budget",
            "capability_demand",
            "constraints",
            "objective",
            "profile",
            "schema_version",
            "task_id",
            "no_change_expected",
            "required_verifier_ids",
        },
    )
    if str(raw.get("schema_version", BRIDGE_SCHEMA_VERSION)) != BRIDGE_SCHEMA_VERSION:
        raise AssistantBridgeError("Unsupported assistant task schema_version.")
    demand_raw = _as_object(raw.get("capability_demand", {}), "capability_demand")
    _reject_unknown(
        "capability_demand", demand_raw, {"required", "risk_class", "tools"}
    )
    budget_raw = _as_object(raw.get("budget", {}), "budget")
    _reject_unknown("budget", budget_raw, {"max_premium_calls"})
    allow_remote = raw.get("allow_remote")
    if allow_remote is not None and not isinstance(allow_remote, bool):
        raise AssistantBridgeError("allow_remote must be true, false, or omitted.")
    return AssistantTaskEnvelope(
        task_id=_string_value(raw.get("task_id", ""), "task_id"),
        objective=_string_value(raw.get("objective", ""), "objective"),
        profile=_string_value(raw.get("profile", "balanced"), "profile"),
        constraints=_string_tuple(raw.get("constraints", []), "constraints"),
        no_change_expected=_bool_value(
            raw.get("no_change_expected", False),
            "no_change_expected",
        ),
        required_verifier_ids=_identifier_tuple(
            raw.get("required_verifier_ids", []), "required_verifier_ids"
        ),
        allow_remote=allow_remote,
        allow_remote_workspace=_bool_value(
            raw.get("allow_remote_workspace", False),
            "allow_remote_workspace",
        ),
        budget=AssistantTaskBudget(
            max_premium_calls=_optional_int(
                budget_raw.get("max_premium_calls"),
                "budget.max_premium_calls",
            )
        ),
        capability_demand=CapabilityDemand(
            required=_identifier_tuple(demand_raw.get("required", []), "required"),
            tools=_identifier_tuple(demand_raw.get("tools", []), "tools"),
            risk_class=_string_value(
                demand_raw.get("risk_class", "read_only"),
                "capability_demand.risk_class",
            ),
        ),
    )


def build_assistant_task(
    objective: str,
    *,
    profile: str = "balanced",
    required_capabilities: Sequence[str] = (),
    required_tools: Sequence[str] = (),
    risk_class: str = "read_only",
    constraints: Sequence[str] = (),
    no_change_expected: bool = False,
    required_verifier_ids: Sequence[str] = (),
    allow_remote: bool | None = None,
    allow_remote_workspace: bool = False,
    max_premium_calls: int | None = None,
) -> AssistantTaskEnvelope:
    return AssistantTaskEnvelope(
        objective=objective,
        profile=profile,
        capability_demand=CapabilityDemand(
            required=tuple(required_capabilities),
            tools=tuple(required_tools),
            risk_class=risk_class,
        ),
        constraints=tuple(constraints),
        no_change_expected=no_change_expected,
        required_verifier_ids=tuple(required_verifier_ids),
        allow_remote=allow_remote,
        allow_remote_workspace=allow_remote_workspace,
        budget=AssistantTaskBudget(max_premium_calls=max_premium_calls),
    )


def load_assistant_bridge_config(path: str | Path) -> AssistantBridgeConfig:
    source = Path(path).expanduser().resolve()
    raw = _load_json_object(source, label="assistant bridge config")
    _reject_unknown(
        "assistant bridge config",
        raw,
        {
            "capsule",
            "profiles",
            "providers",
            "runtime",
            "schema_version",
            "state",
            "verification",
            "workspace",
        },
    )
    if str(raw.get("schema_version", "")) != BRIDGE_SCHEMA_VERSION:
        raise AssistantBridgeError(
            "Unsupported assistant bridge config schema_version."
        )
    providers = _as_object(raw.get("providers"), "providers")
    _reject_unknown("providers", providers, {"local", "premium"})
    profiles_raw = _as_object(raw.get("profiles"), "profiles")
    _reject_unknown("profiles", profiles_raw, PROFILES)
    verification_raw = _as_object(raw.get("verification", {}), "verification")
    _reject_unknown(
        "verification",
        verification_raw,
        {
            "command_verifiers",
            "external_verifiers",
            "independent_required_for",
            "isolation",
            "output_checks",
            "resources",
        },
    )
    isolation_raw = _as_object(
        verification_raw.get("isolation", {}),
        "verification.isolation",
    )
    _reject_unknown(
        "verification.isolation",
        isolation_raw,
        {"linux_backend", "macos_backend", "required"},
    )
    try:
        verifier_isolation = VerifierIsolationPolicy(
            required=_bool_value(
                isolation_raw.get("required", True),
                "verification.isolation.required",
            ),
            macos_backend=_string_value(
                isolation_raw.get("macos_backend", "/usr/bin/sandbox-exec"),
                "verification.isolation.macos_backend",
            ),
            linux_backend=_string_value(
                isolation_raw.get("linux_backend", "/usr/bin/bwrap"),
                "verification.isolation.linux_backend",
            ),
        )
    except VerifierIsolationError as exc:
        raise AssistantBridgeError(str(exc)) from None
    resources_raw = _as_object(
        verification_raw.get("resources", {}),
        "verification.resources",
    )
    _reject_unknown(
        "verification.resources",
        resources_raw,
        {
            "cpu_quota_percent",
            "cpu_time_seconds",
            "file_size_bytes",
            "linux_backend",
            "memory_bytes",
            "open_files",
            "processes",
            "required",
            "required_strengths",
            "workspace_growth_bytes",
        },
    )
    required_strengths_raw = _as_object(
        resources_raw.get(
            "required_strengths",
            dict(VerifierResourcePolicy().required_strengths),
        ),
        "verification.resources.required_strengths",
    )
    try:
        verifier_resources = VerifierResourcePolicy(
            required=_bool_value(
                resources_raw.get("required", True),
                "verification.resources.required",
            ),
            cpu_time_seconds=_int_value(
                resources_raw.get("cpu_time_seconds", 900),
                "verification.resources.cpu_time_seconds",
            ),
            cpu_quota_percent=_int_value(
                resources_raw.get("cpu_quota_percent", 200),
                "verification.resources.cpu_quota_percent",
            ),
            file_size_bytes=_int_value(
                resources_raw.get("file_size_bytes", 64 * 1024 * 1024),
                "verification.resources.file_size_bytes",
            ),
            open_files=_int_value(
                resources_raw.get("open_files", 256),
                "verification.resources.open_files",
            ),
            memory_bytes=_int_value(
                resources_raw.get("memory_bytes", 2 * 1024 * 1024 * 1024),
                "verification.resources.memory_bytes",
            ),
            processes=_int_value(
                resources_raw.get("processes", 256),
                "verification.resources.processes",
            ),
            workspace_growth_bytes=_int_value(
                resources_raw.get(
                    "workspace_growth_bytes", 256 * 1024 * 1024
                ),
                "verification.resources.workspace_growth_bytes",
            ),
            required_strengths={
                str(name): _string_value(
                    value,
                    f"verification.resources.required_strengths.{name}",
                )
                for name, value in required_strengths_raw.items()
            },
            linux_backend=_string_value(
                resources_raw.get("linux_backend", "/usr/bin/systemd-run"),
                "verification.resources.linux_backend",
            ),
        )
    except VerifierResourceError as exc:
        raise AssistantBridgeError(str(exc)) from None
    independent_raw = _as_object(
        verification_raw.get("independent_required_for", {}),
        "verification.independent_required_for",
    )
    _reject_unknown(
        "verification.independent_required_for",
        independent_raw,
        {"capabilities", "risks", "tools"},
    )
    capsule_raw = _as_object(raw.get("capsule", {}), "capsule")
    _reject_unknown(
        "capsule",
        capsule_raw,
        {
            "max_chars",
            "max_constraint_chars",
            "max_diff_chars",
            "max_objective_chars",
            "secret_redaction",
        },
    )
    secret_raw = _as_object(
        capsule_raw.get("secret_redaction", {}), "capsule.secret_redaction"
    )
    _reject_unknown(
        "capsule.secret_redaction",
        secret_raw,
        {"require_residual_assurance", "residual_plugins"},
    )
    runtime_raw = _as_object(raw.get("runtime", {}), "runtime")
    _reject_unknown(
        "runtime",
        runtime_raw,
        {
            "cleanup_grace_seconds",
            "cleanup_kill_seconds",
            "require_launcher_chain",
            "require_psutil",
            "require_tree_isolation",
            "stderr_limit_bytes",
            "stdin_limit_bytes",
            "stdout_limit_bytes",
        },
    )
    state_raw = _as_object(raw.get("state", {}), "state")
    _reject_unknown(
        "state",
        state_raw,
        {
            "confirmation_ttl_seconds",
            "ledger_path",
            "lock_timeout_seconds",
            "namespace",
            "stale_lock_seconds",
            "budget_retention_seconds",
            "max_budget_entries",
            "confirmation_retention_seconds",
            "max_confirmation_entries",
            "budget_lease_ttl_seconds",
        },
    )
    workspace_raw = _as_object(raw.get("workspace", {}), "workspace")
    _reject_unknown(
        "workspace",
        workspace_raw,
        {
            "ignored_paths",
            "max_file_bytes",
            "max_files",
            "max_total_bytes",
            "transaction_lock_ttl_seconds",
            "transaction_state_dir",
        },
    )
    profiles = {
        name: _parse_profile(
            name, _as_object(profiles_raw.get(name), f"profiles.{name}")
        )
        for name in sorted(PROFILES)
    }
    command_verifiers = _parse_command_verifiers(
        verification_raw.get("command_verifiers", [])
    )
    external_verifiers = _parse_external_verifiers(
        verification_raw.get("external_verifiers", [])
    )
    ledger_raw = _string_value(
        state_raw.get("ledger_path", "work/runtime/assistant-bridge-state.json"),
        "state.ledger_path",
    )
    ledger_path = Path(ledger_raw).expanduser()
    if not ledger_path.is_absolute():
        ledger_path = source.parent.parent / ledger_path
    transaction_raw = _string_value(
        workspace_raw.get(
            "transaction_state_dir", "work/runtime/assistant-transactions"
        ),
        "workspace.transaction_state_dir",
    )
    transaction_path = Path(transaction_raw).expanduser()
    if not transaction_path.is_absolute():
        transaction_path = source.parent.parent / transaction_path
    ignored_raw = workspace_raw.get("ignored_paths", [])
    if not isinstance(ignored_raw, list):
        raise AssistantBridgeError("workspace.ignored_paths must be a list.")
    ignored_rules: list[IgnoredPathRule] = []
    for index, item in enumerate(ignored_raw):
        ignored = _as_object(item, f"workspace.ignored_paths[{index}]")
        _reject_unknown(
            f"workspace.ignored_paths[{index}]", ignored, {"direction", "path"}
        )
        try:
            ignored_rules.append(
                IgnoredPathRule(
                    path=_string_value(
                        ignored.get("path", ""),
                        f"workspace.ignored_paths[{index}].path",
                    ),
                    direction=_string_value(
                        ignored.get("direction", "input_only"),
                        f"workspace.ignored_paths[{index}].direction",
                    ),
                )
            )
        except WorkspaceSecurityError as exc:
            raise AssistantBridgeError(str(exc)) from None
    try:
        workspace_policy = BridgeWorkspacePolicy(
            scope=WorkspaceScopePolicy(
                max_files=_int_value(
                    workspace_raw.get("max_files", 5000), "workspace.max_files"
                ),
                max_total_bytes=_int_value(
                    workspace_raw.get("max_total_bytes", 256 * 1024 * 1024),
                    "workspace.max_total_bytes",
                ),
                max_file_bytes=_int_value(
                    workspace_raw.get("max_file_bytes", 64 * 1024 * 1024),
                    "workspace.max_file_bytes",
                ),
                ignored_paths=tuple(ignored_rules),
            ),
            transaction_state_dir=str(transaction_path.resolve()),
            transaction_lock_ttl_seconds=_number_value(
                workspace_raw.get("transaction_lock_ttl_seconds", 120),
                "workspace.transaction_lock_ttl_seconds",
            ),
        )
    except WorkspaceSecurityError as exc:
        raise AssistantBridgeError(str(exc)) from None
    state_policy = BridgeStatePolicy(
        ledger_path=str(ledger_path.resolve()),
        namespace=_string_value(
            state_raw.get("namespace", "assistant-bridge-v2"),
            "state.namespace",
        ),
        confirmation_ttl_seconds=_number_value(
            state_raw.get("confirmation_ttl_seconds", 300),
            "state.confirmation_ttl_seconds",
        ),
        lock_timeout_seconds=_number_value(
            state_raw.get("lock_timeout_seconds", 5),
            "state.lock_timeout_seconds",
        ),
        stale_lock_seconds=_number_value(
            state_raw.get("stale_lock_seconds", 120),
            "state.stale_lock_seconds",
        ),
        budget_retention_seconds=_number_value(
            state_raw.get("budget_retention_seconds", 90 * 24 * 60 * 60),
            "state.budget_retention_seconds",
        ),
        max_budget_entries=_int_value(
            state_raw.get("max_budget_entries", 4096),
            "state.max_budget_entries",
        ),
        confirmation_retention_seconds=_number_value(
            state_raw.get("confirmation_retention_seconds", 24 * 60 * 60),
            "state.confirmation_retention_seconds",
        ),
        max_confirmation_entries=_int_value(
            state_raw.get("max_confirmation_entries", 4096),
            "state.max_confirmation_entries",
        ),
        budget_lease_ttl_seconds=_number_value(
            state_raw.get("budget_lease_ttl_seconds", 60),
            "state.budget_lease_ttl_seconds",
        ),
    )
    runtime_policy = BridgeRuntimePolicy(
        require_tree_isolation=_bool_value(
            runtime_raw.get("require_tree_isolation", True),
            "runtime.require_tree_isolation",
        ),
        require_psutil=_bool_value(
            runtime_raw.get("require_psutil", True), "runtime.require_psutil"
        ),
        require_launcher_chain=_bool_value(
            runtime_raw.get("require_launcher_chain", True),
            "runtime.require_launcher_chain",
        ),
        stdin_limit_bytes=_int_value(
            runtime_raw.get("stdin_limit_bytes", 8 * 1024 * 1024),
            "runtime.stdin_limit_bytes",
        ),
        stdout_limit_bytes=_int_value(
            runtime_raw.get("stdout_limit_bytes", _MAX_STREAM_BYTES),
            "runtime.stdout_limit_bytes",
        ),
        stderr_limit_bytes=_int_value(
            runtime_raw.get("stderr_limit_bytes", _MAX_STREAM_BYTES),
            "runtime.stderr_limit_bytes",
        ),
        cleanup_grace_seconds=_number_value(
            runtime_raw.get("cleanup_grace_seconds", 0.25),
            "runtime.cleanup_grace_seconds",
        ),
        cleanup_kill_seconds=_number_value(
            runtime_raw.get("cleanup_kill_seconds", 0.75),
            "runtime.cleanup_kill_seconds",
        ),
    )
    effective_sha256 = _sha256_text(
        _canonical_json(
            {
                "declared": raw,
                "state": state_policy.effective_descriptor(),
                "workspace": workspace_policy.effective_descriptor(),
                "runtime_policy": runtime_policy.payload(),
                "runtime_capabilities": runtime_capabilities().payload(),
                "verifier_isolation_policy": verifier_isolation.payload(),
                "verifier_isolation_capability": verifier_isolation_capability(
                    verifier_isolation
                ).payload(),
                "verifier_resource_policy": verifier_resources.payload(),
                "verifier_resource_capabilities": verifier_resource_capabilities(
                    verifier_resources
                ).payload(),
                "secret_assurance": {
                    "require_residual_assurance": _bool_value(
                        secret_raw.get("require_residual_assurance", True),
                        "capsule.secret_redaction.require_residual_assurance",
                    ),
                    "residual_plugins": list(
                        _string_tuple(
                            secret_raw.get(
                                "residual_plugins",
                                list(SecretRedactionPolicy().residual_plugins),
                            ),
                            "capsule.secret_redaction.residual_plugins",
                        )
                    ),
                },
            }
        )
    )
    return AssistantBridgeConfig(
        local=_parse_provider(_as_object(providers.get("local"), "providers.local")),
        premium=_parse_provider(
            _as_object(providers.get("premium"), "providers.premium")
        ),
        profiles=profiles,
        verification_checks=_verification_checks(
            verification_raw.get("output_checks", [])
        ),
        command_verifiers=command_verifiers,
        external_verifiers={item.id: item for item in external_verifiers},
        independent_capabilities=_identifier_tuple(
            independent_raw.get("capabilities", []),
            "verification independent capabilities",
        ),
        independent_tools=_identifier_tuple(
            independent_raw.get("tools", []),
            "verification independent tools",
        ),
        independent_risks=_string_tuple(
            independent_raw.get("risks", []),
            "verification independent risks",
        ),
        verifier_isolation=verifier_isolation,
        verifier_resources=verifier_resources,
        capsule=CapsulePolicy(
            max_chars=_int_value(
                capsule_raw.get("max_chars", 8000), "capsule.max_chars"
            ),
            max_objective_chars=_int_value(
                capsule_raw.get("max_objective_chars", 3000),
                "capsule.max_objective_chars",
            ),
            max_constraint_chars=_int_value(
                capsule_raw.get("max_constraint_chars", 1000),
                "capsule.max_constraint_chars",
            ),
            max_diff_chars=_int_value(
                capsule_raw.get("max_diff_chars", 2500),
                "capsule.max_diff_chars",
            ),
            secret_redaction=SecretRedactionPolicy(
                require_residual_assurance=_bool_value(
                    secret_raw.get("require_residual_assurance", True),
                    "capsule.secret_redaction.require_residual_assurance",
                ),
                residual_plugins=_string_tuple(
                    secret_raw.get(
                        "residual_plugins",
                        list(SecretRedactionPolicy().residual_plugins),
                    ),
                    "capsule.secret_redaction.residual_plugins",
                ),
            ),
        ),
        runtime=runtime_policy,
        state=state_policy,
        workspace=workspace_policy,
        source_sha256=effective_sha256,
    )


def _receipt_workspace_attestation(snapshot: WorkspaceSnapshot) -> WorkspaceAttestation:
    state = {
        "git_repository": snapshot.git_repository,
        "head_sha": snapshot.head_sha,
        "index_sha256": snapshot.index_sha256,
        "status_sha256": snapshot.status_sha256,
        "manifest_sha256": snapshot.manifest_sha256,
        "file_count": len(snapshot.files),
        "total_bytes": snapshot.total_bytes,
        "scope": "tracked_untracked_nonignored_plus_declared_ignored",
    }
    return WorkspaceAttestation(
        root=snapshot.root,
        fingerprint=_sha256_text(_canonical_json(state)),
        git_repository=snapshot.git_repository,
        head_sha=snapshot.head_sha,
        index_sha256=snapshot.index_sha256,
        status_sha256=snapshot.status_sha256,
        manifest_sha256=snapshot.manifest_sha256,
        file_count=len(snapshot.files),
        total_bytes=snapshot.total_bytes,
    )


def plan_assistant_route(
    task: AssistantTaskEnvelope,
    config: AssistantBridgeConfig,
    *,
    workspace: str | Path = ".",
    local_provider_override: str | None = None,
    workspace_snapshot: WorkspaceSnapshot | None = None,
) -> RouteDecisionReceipt:
    if workspace_snapshot is None:
        try:
            workspace_snapshot = snapshot_workspace(workspace, config.workspace.scope)
        except WorkspaceSecurityError as exc:
            raise AssistantBridgeError(str(exc)) from None
    workspace_attestation = _receipt_workspace_attestation(workspace_snapshot)
    profile = config.profiles[task.profile]
    local_gaps = provider_gaps(config.local, task)
    premium_gaps = provider_gaps(config.premium, task)
    budget = profile.max_premium_calls
    if task.budget.max_premium_calls is not None:
        budget = min(budget, task.budget.max_premium_calls)

    remote_allowed = profile.remote_allowed
    rationale: list[str] = [f"profile_{profile.name}"]
    if profile.name == "offline":
        remote_allowed = False
        rationale.append("offline_remote_forbidden")
    elif profile.explicit_remote_opt_in:
        remote_allowed = task.allow_remote is True
        rationale.append(
            "explicit_remote_opt_in_present"
            if remote_allowed
            else "explicit_remote_opt_in_missing"
        )
    elif task.allow_remote is False:
        remote_allowed = False
        rationale.append("task_remote_denied")
    elif task.allow_remote is True:
        remote_allowed = True
        rationale.append("task_remote_allowed")

    premium_available = remote_allowed and budget > 0 and not premium_gaps
    if budget == 0:
        rationale.append("premium_budget_zero")
    if premium_gaps:
        rationale.append("premium_capability_or_authority_gap")
    if local_gaps:
        rationale.append("local_capability_or_authority_gap")

    initial = profile.initial_route
    if initial == "local":
        if not local_gaps:
            route = "local"
        elif premium_available:
            route = "premium"
            rationale.append("known_gap_escalation")
        else:
            route = "blocked"
    elif initial == "premium":
        if premium_available:
            route = "premium"
        elif not local_gaps:
            route = "local"
            rationale.append("premium_unavailable_local_fallback")
        else:
            route = "blocked"
    elif not local_gaps:
        route = "local_then_verify" if premium_available else "local"
    elif premium_available:
        route = "premium"
    else:
        route = "blocked"

    if route == "blocked":
        rationale.append("no_policy_compliant_provider")
    elif route == "local":
        rationale.append("local_only")
    elif route == "local_then_verify":
        rationale.append("failure_driven_escalation")
    else:
        rationale.append("premium_selected")

    expected_flow = {
        "blocked": ("stop",),
        "local": ("local", "verify", "stop"),
        "local_then_verify": (
            "local",
            "verify",
            "stop_or_capsule",
            "premium",
            "verify",
        ),
        "premium": ("capsule", "premium", "verify"),
    }[route]
    local_runtime = _provider_runtime_attestation(
        config.local,
        task,
        local_provider_override=local_provider_override,
    )
    premium_runtime = _provider_runtime_attestation(config.premium, task)
    base_payload = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "contract": "RouteDecisionReceipt",
        "task": task.metadata_payload(),
        "route": route,
        "local_provider": config.local.id,
        "premium_provider": config.premium.id
        if route in {"local_then_verify", "premium"}
        else None,
        "local_gaps": list(local_gaps),
        "premium_gaps": list(premium_gaps),
        "remote_allowed": remote_allowed,
        "premium_call_budget": budget,
        "rationale_codes": rationale,
        "expected_flow": list(expected_flow),
        "config_sha256": config.source_sha256,
        "workspace": workspace_attestation.payload(),
        "local_runtime": local_runtime,
        "premium_runtime": premium_runtime,
    }
    return RouteDecisionReceipt(
        receipt_id=f"route-{_sha256_text(_canonical_json(base_payload))[:32]}",
        task=task.metadata_payload(),
        route=route,
        local_provider=config.local.id,
        premium_provider=(
            config.premium.id if route in {"local_then_verify", "premium"} else None
        ),
        local_gaps=local_gaps,
        premium_gaps=premium_gaps,
        remote_allowed=remote_allowed,
        premium_call_budget=budget,
        rationale_codes=tuple(rationale),
        expected_flow=expected_flow,
        config_sha256=config.source_sha256,
        workspace=workspace_attestation,
        local_runtime=local_runtime,
        premium_runtime=premium_runtime,
    )


def provider_gaps(
    provider: ProviderSpec,
    task_or_demand: AssistantTaskEnvelope | CapabilityDemand,
) -> tuple[str, ...]:
    task = task_or_demand if isinstance(task_or_demand, AssistantTaskEnvelope) else None
    demand = task.capability_demand if task is not None else task_or_demand
    gaps = [
        f"capability:{item}"
        for item in demand.required
        if item not in provider.capabilities
    ]
    gaps.extend(f"tool:{item}" for item in demand.tools if item not in provider.tools)
    if RISK_LEVELS[demand.risk_class] > RISK_LEVELS[provider.max_risk]:
        gaps.append(f"risk:{demand.risk_class}")
    if RISK_LEVELS[demand.risk_class] > RISK_LEVELS["write_local"]:
        gaps.append(f"authority:{demand.risk_class}")
    if demand.risk_class == "write_local" and provider.sandbox != "workspace-write":
        gaps.append("authority:workspace_write")
    if provider.mode == "local" and demand.risk_class == "write_local":
        if provider.workspace_access != "read_write":
            gaps.append("workspace:read_write")
    needs_web = "web" in demand.required or "web" in demand.tools
    if needs_web and not provider.network_access:
        gaps.append("network:web")
    if (
        provider.mode == "premium"
        and demand.risk_class == "write_local"
        and (task is None or not task.allow_remote_workspace)
    ):
        gaps.append("authority:remote_workspace_opt_in")
    if provider.mode == "premium" and demand.risk_class == "write_local":
        if provider.workspace_access != "read_write":
            gaps.append("workspace:read_write")
    return tuple(sorted(set(gaps)))


def _confirmation_binding_sha256(
    receipt: RouteDecisionReceipt,
    execution_binding: Mapping[str, object],
) -> str:
    payload = {
        "contract": "AssistantBridgeExecutionConfirmation",
        "receipt_id": receipt.receipt_id,
        "task_fingerprint": receipt.task["task_fingerprint"],
        "workspace_fingerprint": receipt.workspace.fingerprint,
        "config_sha256": receipt.config_sha256,
        "local_runtime": _deep_thaw(receipt.local_runtime),
        "premium_runtime": _deep_thaw(receipt.premium_runtime),
        "execution_binding": _deep_thaw(execution_binding),
    }
    return _sha256_text(_canonical_json(payload))


def build_local_prompt(task: AssistantTaskEnvelope) -> str:
    payload = {
        "objective": task.objective,
        "constraints": list(task.constraints),
        "capability_demand": task.capability_demand.payload(),
        "completion_contract": {
            "do_not_claim_success_on_failure": True,
            "independent_verification_decides_completion": True,
        },
    }
    return (
        "Work as the local execution tier for this bounded assistant task. "
        "Use the workspace as the source of truth, respect the constraints, and report "
        "failures honestly. A separate mechanical verifier decides completion.\n\n"
        + _canonical_json(payload)
    )


def build_premium_prompt(capsule: EscalationCapsule) -> str:
    return (
        "Continue from the minimal escalation capsule below. It contains operational "
        "evidence, not hidden reasoning. Do not assume omitted conversation context or "
        "workspace authority. Report failures honestly; verification is independent.\n\n"
        + _canonical_json(capsule.payload())
    )


def attest_workspace(workspace: str | Path) -> WorkspaceAttestation:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise AssistantBridgeError("Assistant workspace must be an existing directory.")
    try:
        snapshot = snapshot_workspace(root, WorkspaceScopePolicy())
    except WorkspaceSecurityError as exc:
        raise AssistantBridgeError(str(exc)) from None
    return _receipt_workspace_attestation(snapshot)


def collect_git_evidence(
    workspace: str | Path,
    policy: CapsulePolicy,
    *,
    include_excerpt: bool,
    expected_snapshot: WorkspaceSnapshot | None = None,
    workspace_policy: WorkspaceScopePolicy | None = None,
) -> DiffEvidence:
    scope = workspace_policy or WorkspaceScopePolicy()
    try:
        before = snapshot_workspace(workspace, scope)
    except WorkspaceSecurityError as exc:
        raise AssistantBridgeError(str(exc)) from None
    if (
        expected_snapshot is not None
        and before.fingerprint != expected_snapshot.fingerprint
    ):
        raise AssistantBridgeError(
            "Workspace changed before Git evidence collection."
        )
    attestation = _receipt_workspace_attestation(before)
    if not attestation.git_repository:
        return DiffEvidence(
            sha256=attestation.manifest_sha256,
            characters=0,
            excerpt="",
            truncated=False,
            staged_sha256="",
            unstaged_sha256="",
            untracked_manifest_sha256="",
        )
    root = Path(attestation.root)
    try:
        git = trusted_git_session(root)
        staged = git.staged_diff(max_output_bytes=_MAX_GIT_BYTES)
        unstaged = git.unstaged_diff(max_output_bytes=_MAX_GIT_BYTES)
        untracked = git.untracked_paths(
            max_output_bytes=_MAX_GIT_BYTES,
        )
        manifest = _untracked_manifest(untracked, before)
        after = snapshot_workspace(root, scope)
    except WorkspaceSecurityError as exc:
        raise AssistantBridgeError("Could not produce complete Git evidence.") from exc
    if after.fingerprint != before.fingerprint:
        raise AssistantBridgeError(
            "Workspace changed during Git evidence collection."
        )
    combined = (
        b"STAGED\0" + staged + b"UNSTAGED\0" + unstaged + b"UNTRACKED\0" + manifest
    )
    text = combined.decode("utf-8", errors="replace")
    excerpt = ""
    redacted_truncated = False
    if include_excerpt and text:
        excerpt, _, redacted_truncated = redact_and_bound(text, policy.max_diff_chars)
    return DiffEvidence(
        sha256=_sha256_bytes(combined),
        characters=len(text),
        excerpt=excerpt,
        truncated=bool(text) and (not include_excerpt or redacted_truncated),
        staged_sha256=_sha256_bytes(staged),
        unstaged_sha256=_sha256_bytes(unstaged),
        untracked_manifest_sha256=_sha256_bytes(manifest),
    )


def collect_git_diff(workspace: str | Path, *, max_chars: int = 100_000) -> str:
    """Compatibility helper returning a bounded staged+unstaged evidence excerpt."""

    policy = CapsulePolicy(
        max_chars=max(1000, min(100_000, max_chars + 1000)),
        max_objective_chars=0,
        max_constraint_chars=0,
        max_diff_chars=max(0, min(max_chars, 100_000)),
    )
    return collect_git_evidence(workspace, policy, include_excerpt=True).excerpt


def build_codex_command_plan(
    provider: ProviderSpec,
    *,
    prompt: str,
    workspace: str | Path,
    demand: CapabilityDemand | None = None,
    output_path: str | Path | None = None,
    local_provider_override: str | None = None,
    workspace_access: str | None = None,
    runtime_policy: BridgeRuntimePolicy | None = None,
    ephemeral_workspace: bool = False,
) -> CommandPlan:
    demand = demand or CapabilityDemand()
    selected_runtime_policy = runtime_policy or BridgeRuntimePolicy()
    resolved_workspace = str(Path(workspace).expanduser().resolve())
    if not Path(resolved_workspace).is_dir():
        raise AssistantBridgeError("Command workspace must be an existing directory.")
    planning_environment = _sanitized_environment(provider.environment_allowlist)
    try:
        executable_identity = resolve_executable(
            provider.executable,
            env=planning_environment,
        )
    except (AssistantBridgeRuntimeError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Provider {provider.id} executable attestation failed."
        ) from None
    runtime = runtime_capabilities().payload()
    selected_local = ""
    argv = [executable_identity.resolved_path, *provider.launcher_args]
    argv.append("--strict-config")
    if provider.mode == "local":
        selected_local = local_provider_override or provider.local_provider
        if selected_local not in {"lmstudio", "ollama"}:
            raise AssistantBridgeError(
                "Local provider override must be ollama or lmstudio."
            )
        argv.extend(("--oss", "--local-provider", selected_local))
    argv.extend(("--model", provider.model))
    if provider.codex_profile:
        argv.extend(("--profile", provider.codex_profile))
    sandbox = _effective_sandbox(provider, demand)
    effective_workspace_access = workspace_access or _effective_workspace_access(
        provider,
        demand,
        allow_remote_workspace=False,
    )
    web_enabled = _requires_web(demand)
    if web_enabled and not provider.network_access:
        raise AssistantBridgeError(
            "Selected provider cannot materialize required web access."
        )
    argv.extend(
        (
            "--sandbox",
            sandbox,
            "--cd",
            resolved_workspace,
            "--ask-for-approval",
            "never",
            "--config",
            f"sandbox_workspace_write.network_access={'true' if web_enabled else 'false'}",
            "--config",
            "shell_environment_policy.inherit=none",
        )
    )
    if web_enabled:
        argv.append("--search")
    argv.extend(provider.extra_args)
    argv.extend(("exec", "--ephemeral", "--ignore-user-config", "--ignore-rules"))
    if not (Path(resolved_workspace) / ".git").exists():
        argv.append("--skip-git-repo-check")
    argv.extend(("--color", "never"))
    resolved_output = ""
    if output_path is not None:
        resolved_output = str(Path(output_path).expanduser().resolve())
        argv.extend(("--output-last-message", resolved_output))
    argv.append("-")
    effective_ephemeral_workspace = (
        ephemeral_workspace or effective_workspace_access == "capsule_only"
    )
    try:
        launcher_chain = _build_bridge_launcher_chain(
            executable_identity,
            tuple(argv[1:]),
            entrypoint=provider.launcher_entrypoint,
            companions=provider.launcher_companions,
            workspace=resolved_workspace,
            environment=planning_environment,
        )
    except (AssistantBridgeError, AssistantBridgeRuntimeError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Provider {provider.id} launcher-chain attestation failed."
        ) from None
    launcher_authority_sha256 = _launcher_chain_authority_sha256(
        launcher_chain,
        executable=executable_identity,
        workspace=resolved_workspace,
        output_path=resolved_output,
        environment=planning_environment,
        ephemeral_workspace=effective_ephemeral_workspace,
        ephemeral_environment_keys=("CODEX_HOME", "HOME"),
    )
    launcher_artifacts = _launcher_artifact_authority_digests(
        launcher_chain,
        workspace=resolved_workspace,
        ephemeral_workspace=effective_ephemeral_workspace,
    )
    semantic_argv = _normalize_ephemeral_paths(
        tuple(argv),
        ephemeral_workspace=effective_ephemeral_workspace,
    )
    command_payload = {
        "provider_id": provider.id,
        "mode": provider.mode,
        "argv": list(semantic_argv),
        "stdin_sha256": _sha256_text(prompt),
        "workspace_sha256": (
            _sha256_text("ephemeral_workspace")
            if ephemeral_workspace or effective_workspace_access == "capsule_only"
            else _sha256_text(resolved_workspace)
        ),
        "sandbox": sandbox,
        "network_access": web_enabled,
        "workspace_access": effective_workspace_access,
        "model": provider.model,
        "local_provider": selected_local,
        "environment_keys": list(provider.environment_allowlist),
        "environment_sha256": executable_identity.resolution_environment.sha256,
        "executable": executable_identity.binding_payload(),
        "runtime": runtime,
        "runtime_policy": selected_runtime_policy.payload(),
        "launcher_authority_sha256": launcher_authority_sha256,
        "launcher_artifact_sha256": list(launcher_artifacts),
    }
    return CommandPlan(
        provider_id=provider.id,
        mode=provider.mode,
        argv=tuple(argv),
        stdin_sha256=_sha256_text(prompt),
        stdin_chars=len(prompt),
        workspace=resolved_workspace,
        output_path=resolved_output,
        command_sha256=_sha256_text(_canonical_json(command_payload)),
        sandbox=sandbox,
        network_access=web_enabled,
        workspace_access=effective_workspace_access,
        model=provider.model,
        local_provider=selected_local,
        environment_allowlist=provider.environment_allowlist,
        executable_identity=executable_identity,
        environment_sha256=executable_identity.resolution_environment.sha256,
        runtime=runtime,
        runtime_policy=selected_runtime_policy,
        launcher_chain=launcher_chain,
        launcher_authority_sha256=launcher_authority_sha256,
        ephemeral_workspace=effective_ephemeral_workspace,
        launcher_artifact_sha256=launcher_artifacts,
    )


def _preflight_process_runtime(
    plan: CommandPlan,
    resolution_environment: Mapping[str, str],
) -> None:
    """Validate every observable launch prerequisite without starting a process."""

    policy = plan.runtime_policy.process_policy()
    capabilities = runtime_capabilities()
    if policy.require_psutil and not capabilities.psutil_available:
        raise AssistantBridgeRuntimeError(
            "Execution policy requires unavailable process-tree observation."
        )
    if policy.require_tree_isolation and not capabilities.strict_tree_supported:
        raise AssistantBridgeRuntimeError(
            "Strict process-tree isolation is unavailable on this host."
        )
    current = resolve_executable(
        plan.executable_identity.requested,
        env=resolution_environment,
    )
    if current != plan.executable_identity:
        raise AssistantBridgeRuntimeError(
            "Executable identity no longer matches the confirmed plan."
        )


def _blocked_command_result(plan: CommandPlan, code: str) -> CommandResult:
    return CommandResult(
        provider_id=plan.provider_id,
        status="blocked",
        code=code,
        returncode=None,
        duration_ms=0,
        stdout_sha256=_sha256_bytes(b""),
        stdout_bytes=0,
        stderr_sha256=_sha256_bytes(b""),
        stderr_bytes=0,
        command_sha256=plan.command_sha256,
    )


def execute_codex_command(
    plan: CommandPlan,
    *,
    prompt: str,
    output_path: str | Path,
    timeout_seconds: float,
    environment_overrides: Mapping[str, str] | None = None,
    reserve_launch: Callable[[], ProcessLaunchPermit | None] | None = None,
) -> CommandResult:
    if _sha256_text(prompt) != plan.stdin_sha256 or len(prompt) != plan.stdin_chars:
        raise AssistantBridgeError(
            "Execution prompt does not match the inspected command plan."
        )
    resolved_output = str(Path(output_path).expanduser().resolve())
    if not plan.output_path or resolved_output != plan.output_path:
        raise AssistantBridgeError(
            "Execution output path does not match the command plan."
        )
    base_env = _sanitized_environment(plan.environment_allowlist)
    if fingerprint_environment(base_env).sha256 != plan.environment_sha256:
        raise AssistantBridgeError(
            "Execution environment no longer matches the confirmed plan."
        )
    env = _sanitized_environment(
        plan.environment_allowlist,
        overrides=environment_overrides or {},
    )
    try:
        _preflight_process_runtime(plan, base_env)
        launcher_chain = _rebind_bridge_launcher_chain(
            plan.launcher_chain,
            plan.executable_identity,
            plan.argv[1:],
            workspace=plan.workspace,
            environment=env,
        )
        launcher_authority_sha256 = _launcher_chain_authority_sha256(
            launcher_chain,
            executable=plan.executable_identity,
            workspace=plan.workspace,
            output_path=resolved_output,
            environment=env,
            ephemeral_workspace=plan.ephemeral_workspace,
            ephemeral_environment_keys=("CODEX_HOME", "HOME"),
        )
        if launcher_authority_sha256 != plan.launcher_authority_sha256:
            raise AssistantBridgeRuntimeError(
                "Launcher chain no longer matches the confirmed plan."
            )
    except (AssistantBridgeError, AssistantBridgeRuntimeError, OSError, ValueError):
        return _blocked_command_result(plan, "runtime_attestation_failed")
    try:
        outcome = execute_process(
            plan.executable_identity,
            plan.argv[1:],
            stdin=prompt.encode("utf-8"),
            cwd=plan.workspace,
            env=env,
            timeout_seconds=timeout_seconds,
            policy=plan.runtime_policy.process_policy(),
            launcher_chain=launcher_chain,
            reserve_launch=reserve_launch,
        )
    except ProcessLaunchNotAuthorizedError:
        return _blocked_command_result(plan, "launch_not_authorized")
    except ProcessLaunchLifecycleError:
        return _blocked_command_result(plan, "launch_lifecycle_failed")
    except ProcessCleanupError:
        raise
    except AssistantBridgeRuntimeError:
        return _blocked_command_result(plan, "runtime_attestation_failed")
    status = "completed" if outcome.ok else "failed"
    if status == "completed":
        code = "launcher_completed"
    elif outcome.code in {"stdout_limit_exceeded", "stderr_limit_exceeded"}:
        code = "launcher_output_limit_exceeded"
    else:
        code = f"launcher_{_safe_code(outcome.code)}"
    output = ""
    if status == "completed":
        try:
            target = Path(resolved_output)
            if not target.is_file() or target.is_symlink():
                status, code = "failed", "missing_final_output"
            elif target.stat().st_size > _MAX_FINAL_BYTES:
                status, code = "failed", "final_output_limit_exceeded"
            else:
                output = target.read_text(encoding="utf-8", errors="replace")
                if len(output.encode("utf-8")) > _MAX_FINAL_BYTES:
                    output = ""
                    status, code = "failed", "final_output_limit_exceeded"
        except OSError:
            status, code = "failed", "final_output_unreadable"
    return CommandResult(
        provider_id=plan.provider_id,
        status=status,
        code=code,
        returncode=outcome.returncode,
        duration_ms=outcome.duration_ms,
        output=output,
        stdout_sha256=outcome.stdout_sha256,
        stdout_bytes=outcome.stdout_bytes,
        stderr_sha256=outcome.stderr_sha256,
        stderr_bytes=outcome.stderr_bytes,
        command_sha256=plan.command_sha256,
    )


def load_verification_evidence(path: str | Path) -> tuple[VerificationEvidence, ...]:
    raw = _load_json_object(path, label="verification evidence")
    _reject_unknown("verification evidence", raw, {"checks", "schema_version"})
    if str(raw.get("schema_version", "")) != BRIDGE_SCHEMA_VERSION:
        raise AssistantBridgeError("Unsupported verification evidence schema_version.")
    items = raw.get("checks")
    if not isinstance(items, list) or not items:
        raise AssistantBridgeError(
            "Verification evidence requires a non-empty checks list."
        )
    parsed: list[VerificationEvidence] = []
    for index, item in enumerate(items):
        check = _as_object(item, f"verification checks[{index}]")
        _reject_unknown(
            f"verification checks[{index}]",
            check,
            {
                "artifact_sha256",
                "code",
                "evidence_ref",
                "id",
                "kind",
                "observed_chars",
                "passed",
                "task_fingerprint",
                "verifier",
                "verifier_spec_sha256",
                "workspace_fingerprint",
            },
        )
        if not isinstance(check.get("passed"), bool):
            raise AssistantBridgeError(
                f"Verification checks[{index}].passed must be boolean."
            )
        kind = _string_value(check.get("kind", "external"), f"checks[{index}].kind")
        if kind != "external":
            raise AssistantBridgeError(
                "Evidence files may contain only external evidence."
            )
        parsed.append(
            VerificationEvidence(
                id=_string_value(check.get("id", ""), f"checks[{index}].id"),
                verifier=_string_value(
                    check.get("verifier", ""),
                    f"checks[{index}].verifier",
                ),
                kind=kind,
                passed=bool(check["passed"]),
                code=_string_value(check.get("code", ""), f"checks[{index}].code"),
                artifact_sha256=_string_value(
                    check.get("artifact_sha256", ""),
                    f"checks[{index}].artifact_sha256",
                ),
                observed_chars=_int_value(
                    check.get("observed_chars", 0),
                    f"checks[{index}].observed_chars",
                ),
                evidence_ref=_string_value(
                    check.get("evidence_ref", ""),
                    f"checks[{index}].evidence_ref",
                ),
                task_fingerprint=_string_value(
                    check.get("task_fingerprint", ""),
                    f"checks[{index}].task_fingerprint",
                ),
                workspace_fingerprint=_string_value(
                    check.get("workspace_fingerprint", ""),
                    f"checks[{index}].workspace_fingerprint",
                ),
                verifier_spec_sha256=_string_value(
                    check.get("verifier_spec_sha256", ""),
                    f"checks[{index}].verifier_spec_sha256",
                ),
            )
        )
    if len({item.id for item in parsed}) != len(parsed):
        raise AssistantBridgeError("External verification evidence ids must be unique.")
    return tuple(parsed)


def verify_command_result(
    result: CommandResult,
    config: AssistantBridgeConfig,
    *,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
    external_evidence: Sequence[VerificationEvidence] = (),
    verifier_workspace: str | Path,
    verifier_plans: Sequence[BoundVerifierPlan] = (),
) -> tuple[VerificationEvidence, ...]:
    evidence: list[VerificationEvidence] = []
    if result.status != "completed":
        evidence.append(
            VerificationEvidence(
                id=f"launch-{result.provider_id}",
                verifier="process",
                kind="process",
                passed=False,
                code=_safe_code(result.code),
                artifact_sha256=_sha256_text(result.code),
                task_fingerprint=task.task_fingerprint,
                workspace_fingerprint=workspace.fingerprint,
                verifier_spec_sha256=result.command_sha256,
            )
        )
        return tuple(evidence)

    output_digest = _sha256_text(result.output)
    for frozen_check in config.verification_checks:
        check = _deep_thaw(frozen_check)
        assert isinstance(check, dict)
        evaluated = evaluate_check(result.output, check)
        check_spec = _sha256_text(_canonical_json(check))
        evidence.append(
            VerificationEvidence(
                id=str(evaluated["id"]),
                verifier=f"output-{evaluated['type']}",
                kind="output",
                passed=bool(evaluated["passed"]),
                code="check_passed" if bool(evaluated["passed"]) else "check_failed",
                artifact_sha256=output_digest,
                observed_chars=len(result.output),
                task_fingerprint=task.task_fingerprint,
                workspace_fingerprint=workspace.fingerprint,
                verifier_spec_sha256=check_spec,
            )
        )

    for plan in verifier_plans:
        evidence.append(
            _run_bound_verifier(
                plan,
                task=task,
                workspace=workspace,
                verifier_workspace=verifier_workspace,
            )
        )

    for item in external_evidence:
        _validate_external_evidence(item, config, task, workspace)
        evidence.append(item)

    if _requires_independent_evidence(task.capability_demand, config) and not any(
        item.kind == "external" and item.passed
        for item in evidence
    ):
        policy_spec = _sha256_text(
            _canonical_json(
                {
                    "capabilities": list(config.independent_capabilities),
                    "tools": list(config.independent_tools),
                    "risks": list(config.independent_risks),
                }
            )
        )
        evidence.append(
            VerificationEvidence(
                id="independent-evidence",
                verifier="completion-policy",
                kind="policy",
                passed=False,
                code="independent_evidence_missing",
                artifact_sha256=_sha256_text("missing"),
                task_fingerprint=task.task_fingerprint,
                workspace_fingerprint=workspace.fingerprint,
                verifier_spec_sha256=policy_spec,
            )
        )
    return tuple(evidence)


def _redact_capsule_text(
    value: str,
    max_chars: int,
    policy: SecretRedactionPolicy,
) -> tuple[str, int, bool, str]:
    try:
        result = redact_text(value, policy)
    except ResidualAssuranceUnavailableError:
        raise AssistantBridgeError(
            "Residual secret assurance is unavailable; capsule creation failed closed."
        ) from None
    if not result.residual_assured or not result.residual_detector:
        raise AssistantBridgeError(
            "Residual secret assurance is required for escalation capsules."
        )
    redacted = str(result.value)
    truncated = len(redacted) > max_chars
    if truncated:
        redacted = redacted[: max(0, max_chars - 16)] + "...[truncated]"
    return redacted, result.redaction_count, truncated, result.residual_detector


def _redact_public_capsule_payload(
    payload: Mapping[str, object],
    policy: SecretRedactionPolicy,
) -> tuple[dict[str, object], int, str]:
    """Redact every public user string and every mapping key, then assure residue."""

    try:
        keyed, key_count, key_detector = _redact_capsule_mapping_keys(
            payload,
            policy,
        )
        structural_plugins = tuple(
            name
            for name in policy.residual_plugins
            if name
            not in {
                "Base64HighEntropyString",
                "HexHighEntropyString",
                "KeywordDetector",
            }
        )
        public_policy = replace(
            policy,
            user_controlled_fields=frozenset({"*"}),
            require_residual_assurance=True,
            residual_plugins=structural_plugins,
        )
        result = redact_user_controlled_fields(keyed, public_policy)
    except ResidualAssuranceUnavailableError:
        raise AssistantBridgeError(
            "Residual secret assurance is unavailable; capsule creation failed closed."
        ) from None
    except AssistantBridgeError:
        raise
    except Exception:
        raise AssistantBridgeError(
            "Recursive capsule secret assurance failed closed."
        ) from None
    if not isinstance(result.value, Mapping):
        raise AssistantBridgeError("Redacted capsule payload is not an object.")
    if not result.residual_assured or not result.residual_detector:
        raise AssistantBridgeError(
            "Residual secret assurance is required for escalation capsules."
        )
    detector = result.residual_detector or key_detector or ""
    return (
        dict(result.value),
        key_count + result.redaction_count,
        detector,
    )


def _redact_capsule_mapping_keys(
    value: object,
    policy: SecretRedactionPolicy,
) -> tuple[object, int, str]:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        count = 0
        detector = ""
        key_policy = replace(policy, require_residual_assurance=False)
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise AssistantBridgeError(
                    "Capsule public mappings require string keys."
                )
            key_result = redact_text(raw_key, key_policy)
            if key_result.redaction_count:
                raise AssistantBridgeError(
                    "Capsule mapping key failed secret assurance."
                )
            safe_nested, nested_count, nested_detector = (
                _redact_capsule_mapping_keys(nested, policy)
            )
            output[raw_key] = safe_nested
            count += nested_count
            detector = (
                key_result.residual_detector or nested_detector or detector
            )
        return output, count, detector
    if isinstance(value, (list, tuple)):
        output_list: list[object] = []
        count = 0
        detector = ""
        for nested in value:
            safe_nested, nested_count, nested_detector = (
                _redact_capsule_mapping_keys(nested, policy)
            )
            output_list.append(safe_nested)
            count += nested_count
            detector = nested_detector or detector
        return output_list, count, detector
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value, 0, ""
    raise AssistantBridgeError(
        "Capsule public payload contains a non-JSON value."
    )


def build_escalation_capsule(
    task: AssistantTaskEnvelope,
    receipt: RouteDecisionReceipt,
    verification: Sequence[VerificationEvidence],
    policy: CapsulePolicy,
    *,
    failure_codes: Sequence[str],
    diff_text: str = "",
    diff_evidence: DiffEvidence | None = None,
    workspace_fingerprint: str | None = None,
) -> EscalationCapsule:
    if len(task.objective) > policy.max_objective_chars:
        raise AssistantBridgeError(
            "Objective cannot be represented safely inside the escalation capsule."
        )
    objective, objective_redactions, objective_truncated, residual_detector = (
        _redact_capsule_text(
        task.objective,
        policy.max_objective_chars,
        policy.secret_redaction,
        )
    )
    if objective_truncated:
        raise AssistantBridgeError(
            "Objective cannot be represented safely inside the escalation capsule."
        )
    constraints: list[str] = []
    redaction_count = objective_redactions
    remaining = policy.max_constraint_chars
    for raw in task.constraints:
        if len(raw) > remaining:
            raise AssistantBridgeError(
                "Constraints cannot be represented safely inside the escalation capsule."
            )
        value, count, was_truncated, detector = _redact_capsule_text(
            raw, remaining, policy.secret_redaction
        )
        residual_detector = detector or residual_detector
        if was_truncated:
            raise AssistantBridgeError(
                "Constraints cannot be represented safely inside the escalation capsule."
            )
        redaction_count += count
        constraints.append(value)
        remaining -= len(value)

    if diff_evidence is None:
        diff_excerpt, count, diff_truncated, detector = _redact_capsule_text(
            diff_text,
            policy.max_diff_chars,
            policy.secret_redaction,
        )
        residual_detector = detector or residual_detector
        redaction_count += count
        diff_evidence = DiffEvidence(
            sha256=_sha256_text(diff_text) if diff_text else "",
            characters=len(diff_text),
            excerpt=diff_excerpt if diff_text else "",
            truncated=diff_truncated,
            staged_sha256="",
            unstaged_sha256="",
            untracked_manifest_sha256="",
        )
    elif diff_evidence.excerpt:
        diff_excerpt, count, diff_truncated, detector = _redact_capsule_text(
            diff_evidence.excerpt,
            policy.max_diff_chars,
            policy.secret_redaction,
        )
        redaction_count += count
        residual_detector = detector or residual_detector
        diff_evidence = replace(
            diff_evidence,
            excerpt=diff_excerpt,
            truncated=diff_evidence.truncated or diff_truncated,
        )
    safe_failures = tuple(sorted({_safe_code(item) for item in failure_codes if item}))
    effective_workspace = workspace_fingerprint or receipt.workspace.fingerprint
    safe_verification: list[VerificationEvidence] = []
    for item in verification:
        if item.task_fingerprint != task.task_fingerprint:
            raise AssistantBridgeError(
                "Escalation evidence is bound to a different task."
            )
        evidence_ref = item.evidence_ref
        if evidence_ref:
            _, count, _, detector = _redact_capsule_text(
                evidence_ref,
                len(evidence_ref),
                policy.secret_redaction,
            )
            residual_detector = detector or residual_detector
            redaction_count += count
            if count:
                evidence_ref = ""
        safe_verification.append(replace(item, evidence_ref=evidence_ref))
    raw_public: dict[str, object] = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "contract": "EscalationCapsule",
        "capsule_id": "",
        "task_id": task.task_id,
        "task_fingerprint": task.task_fingerprint,
        "objective": objective,
        "objective_sha256": task.objective_sha256,
        "capability_demand": task.capability_demand.payload(),
        "constraints": constraints,
        "route_receipt_id": receipt.receipt_id,
        "workspace_fingerprint": effective_workspace,
        "verification": [item.payload() for item in safe_verification],
        "failure_codes": list(safe_failures),
        "diff": diff_evidence.payload(),
        "redaction": {
            "count": redaction_count,
            "residual_assured": True,
            "residual_detector": residual_detector,
            "truncated": diff_evidence.truncated,
        },
        "excluded": [
            "conversation_history",
            "hidden_reasoning",
            "local_execution_transcript",
            "command_output",
            "credentials",
        ],
    }
    public_payload, final_count, final_detector = _redact_public_capsule_payload(
        raw_public,
        policy.secret_redaction,
    )
    redaction_count += final_count
    redaction = public_payload.get("redaction")
    if not isinstance(redaction, dict):
        raise AssistantBridgeError("Capsule redaction metadata is invalid.")
    redaction["count"] = redaction_count
    redaction["residual_assured"] = True
    residual_detector = final_detector or residual_detector
    redaction["residual_detector"] = residual_detector
    identity_payload = dict(public_payload)
    identity_payload.pop("capsule_id", None)
    capsule_id = f"capsule-{_sha256_text(_canonical_json(identity_payload))[:32]}"
    public_payload["capsule_id"] = capsule_id
    capsule = EscalationCapsule(
        capsule_id=capsule_id,
        task_id=task.task_id,
        task_fingerprint=task.task_fingerprint,
        objective=objective,
        objective_sha256=task.objective_sha256,
        capability_demand=task.capability_demand,
        constraints=tuple(constraints),
        route_receipt_id=receipt.receipt_id,
        workspace_fingerprint=effective_workspace,
        verification=tuple(safe_verification),
        failure_codes=safe_failures,
        diff=diff_evidence,
        redaction_count=redaction_count,
        residual_assured=True,
        residual_detector=residual_detector,
        truncated=diff_evidence.truncated,
        public_payload=public_payload,
    )
    if len(_canonical_json(capsule.payload())) > policy.max_chars:
        raise AssistantBridgeError(
            "Escalation capsule exceeds max_chars; reduce evidence or capsule limits."
        )
    return capsule


@dataclass(frozen=True)
class _PreparedExecution:
    receipt: RouteDecisionReceipt
    source_snapshot: WorkspaceSnapshot = field(repr=False)
    workspace_write_capability: WorkspaceWriteCapability
    prior_evidence: tuple[VerificationEvidence, ...] = field(repr=False)
    commands: tuple[CommandPlan, ...] = field(repr=False)
    verifier_plans: tuple[BoundVerifierPlan, ...] = field(repr=False)
    confirmation_binding_sha256: str
    premium_auth: PremiumAuthAttestation | None = field(repr=False)


class AssistantBridgeRunner:
    def __init__(
        self,
        config: AssistantBridgeConfig,
        *,
        state_ledger: BridgeStateLedger | None = None,
    ) -> None:
        self.config = config
        self.state_ledger = state_ledger or config.state.ledger()

    def plan(
        self,
        task: AssistantTaskEnvelope,
        *,
        workspace: str | Path,
        local_provider_override: str | None = None,
        external_evidence: Sequence[VerificationEvidence] = (),
        include_diff: bool = False,
        capsule_out: str | Path | None = None,
    ) -> dict[str, object]:
        prepared = self._prepare_execution(
            task,
            workspace=workspace,
            local_provider_override=local_provider_override,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )
        try:
            ticket = self.state_ledger.issue_confirmation(
                prepared.confirmation_binding_sha256,
                ttl_seconds=self.config.state.confirmation_ttl_seconds,
            )
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(str(exc)) from None
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "mode": "assistant_bridge_plan",
            "execute": False,
            "route_receipt": prepared.receipt.payload(),
            "confirmation_id": ticket.token,
            "confirmation": ticket.metadata_payload(),
            "commands": [command.payload() for command in prepared.commands],
            "verifiers": [item.payload() for item in prepared.verifier_plans],
            "authority": {
                "process_execution": "requires_one_shot_confirmation_ticket",
                "workspace": str(
                    prepared.receipt.local_runtime["workspace_access"]
                ),
                "workspace_write_capability": (
                    prepared.workspace_write_capability.payload()
                ),
                "remote_workspace": str(
                    prepared.receipt.premium_runtime["workspace_access"]
                ),
                "external_effects": "forbidden",
            },
            "privacy": "metadata_only",
        }

    def inspect_route(
        self,
        task: AssistantTaskEnvelope,
        *,
        workspace: str | Path,
        local_provider_override: str | None = None,
        external_evidence: Sequence[VerificationEvidence] = (),
        include_diff: bool = False,
        capsule_out: str | Path | None = None,
    ) -> RouteDecisionReceipt:
        """Inspect effective route authority without issuing a confirmation ticket."""

        return self._prepare_execution(
            task,
            workspace=workspace,
            local_provider_override=local_provider_override,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        ).receipt

    def _prepare_execution(
        self,
        task: AssistantTaskEnvelope,
        *,
        workspace: str | Path,
        local_provider_override: str | None,
        external_evidence: Sequence[VerificationEvidence],
        include_diff: bool,
        capsule_out: str | Path | None,
    ) -> _PreparedExecution:
        self._validate_verifier_selection(task)
        try:
            source_snapshot = snapshot_workspace(workspace, self.config.workspace.scope)
        except WorkspaceSecurityError as exc:
            raise AssistantBridgeError(str(exc)) from None
        receipt = plan_assistant_route(
            task,
            self.config,
            workspace=workspace,
            local_provider_override=local_provider_override,
            workspace_snapshot=source_snapshot,
        )
        two_phase_required = _requires_independent_evidence(
            task.capability_demand,
            self.config,
        )
        if two_phase_required:
            receipt = _receipt_with_route(
                receipt,
                route="blocked",
                rationale_code="two_phase_required",
            )
        write_capability = workspace_write_capability()
        if (
            task.capability_demand.risk_class == "write_local"
            and not write_capability.supported
        ):
            receipt = _receipt_with_route(
                receipt,
                route="blocked",
                rationale_code="workspace_write_capability_unavailable",
            )
        selected_verifiers = (
            () if two_phase_required else self._selected_verifiers(task)
        )
        verifier_plans = tuple(
            _build_verifier_plan(
                spec,
                workspace=workspace,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
                resource_policy=self.config.verifier_resources,
            )
            for spec in selected_verifiers
        )
        if any(
            plan.spec.kind == "command"
            and (
                plan.isolation is None
                or not plan.isolation.capability.supported
            )
            for plan in verifier_plans
        ):
            receipt = _receipt_with_route(
                receipt,
                route="blocked",
                rationale_code="verifier_isolation_unavailable",
            )
        if any(
            plan.spec.kind == "command"
            and (plan.resources is None or not plan.resources.runnable)
            for plan in verifier_plans
        ):
            receipt = _receipt_with_route(
                receipt,
                route="blocked",
                rationale_code="verifier_resource_governance_unavailable",
            )
        premium_auth: PremiumAuthAttestation | None = None
        if not two_phase_required and receipt.route in {
            "local_then_verify",
            "premium",
        }:
            try:
                premium_auth = _attest_premium_auth()
            except AssistantBridgeError:
                if receipt.route == "local_then_verify" and not receipt.local_gaps:
                    receipt = _receipt_with_route(
                        receipt,
                        route="local",
                        rationale_code="premium_auth_unavailable",
                    )
                else:
                    receipt = _receipt_with_route(
                        receipt,
                        route="blocked",
                        rationale_code="premium_auth_unavailable",
                    )
        bound_external = self._validate_external(
            external_evidence,
            task,
            receipt.workspace,
        )
        commands: list[CommandPlan] = []
        if not two_phase_required and receipt.route in {
            "local",
            "local_then_verify",
        }:
            local_prompt = build_local_prompt(task)
            commands.append(
                build_codex_command_plan(
                    self.config.local,
                    prompt=local_prompt,
                    workspace=workspace,
                    demand=task.capability_demand,
                    output_path=_preview_output_path("local"),
                    local_provider_override=local_provider_override,
                    workspace_access=str(receipt.local_runtime["workspace_access"]),
                    runtime_policy=self.config.runtime,
                    ephemeral_workspace=True,
                )
            )
        if not two_phase_required and receipt.route == "premium":
            try:
                with materialize_workspace(
                    source_snapshot, self.config.workspace.scope
                ) as preview_candidate:
                    preview_diff = collect_git_evidence(
                        preview_candidate.root,
                        self.config.capsule,
                        include_excerpt=include_diff,
                        workspace_policy=self.config.workspace.scope,
                    )
            except WorkspaceSecurityError as exc:
                raise AssistantBridgeError(str(exc)) from None
            preview_capsule = build_escalation_capsule(
                task,
                receipt,
                bound_external,
                self.config.capsule,
                failure_codes=(
                    "policy_selected_premium",
                    *(item.code for item in bound_external if not item.passed),
                ),
                diff_evidence=preview_diff,
            )
            premium_prompt = build_premium_prompt(preview_capsule)
            commands.append(
                _premium_preview_plan(
                    self.config.premium,
                    task,
                    premium_prompt,
                    receipt,
                    workspace=workspace,
                    runtime_policy=self.config.runtime,
                )
            )
        elif not two_phase_required and receipt.route == "local_then_verify":
            commands.append(
                _premium_preview_plan(
                    self.config.premium,
                    task,
                    "<dynamic-capsule-bound-at-runtime>",
                    receipt,
                    workspace=workspace,
                    runtime_policy=self.config.runtime,
                )
            )
        execution_binding = _execution_binding(
            external_evidence=bound_external,
            include_diff=include_diff,
            capsule_out=capsule_out,
            commands=commands,
            verifier_plans=verifier_plans,
            source_snapshot=source_snapshot,
            config=self.config,
            premium_auth=premium_auth,
            workspace_write_capability=write_capability,
        )
        return _PreparedExecution(
            receipt=receipt,
            source_snapshot=source_snapshot,
            workspace_write_capability=write_capability,
            prior_evidence=bound_external,
            commands=tuple(commands),
            verifier_plans=verifier_plans,
            confirmation_binding_sha256=_confirmation_binding_sha256(
                receipt, execution_binding
            ),
            premium_auth=premium_auth,
        )

    def _validate_verifier_selection(self, task: AssistantTaskEnvelope) -> None:
        catalog = {item.id: item for item in self.config.command_verifiers}
        for verifier_id in task.required_verifier_ids:
            spec = catalog.get(verifier_id)
            if spec is None:
                raise AssistantBridgeError(
                    f"Unknown required task verifier {verifier_id!r}."
                )
            if spec.purpose != "task":
                raise AssistantBridgeError(
                    f"Required verifier {verifier_id!r} is not a task verifier."
                )

    def _selected_verifiers(
        self, task: AssistantTaskEnvelope
    ) -> tuple[CommandVerifierSpec, ...]:
        requested = set(task.required_verifier_ids)
        return tuple(
            item
            for item in self.config.command_verifiers
            if (item.purpose == "task" and item.id in requested)
            or (item.purpose == "hygiene" and item.applies_to(task.capability_demand))
        )

    def run(
        self,
        task: AssistantTaskEnvelope,
        *,
        workspace: str | Path,
        confirmation: str,
        local_provider_override: str | None = None,
        external_evidence: Sequence[VerificationEvidence] = (),
        include_diff: bool = False,
        capsule_out: str | Path | None = None,
    ) -> BridgeRunResult:
        prepared = self._prepare_execution(
            task,
            workspace=workspace,
            local_provider_override=local_provider_override,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )
        try:
            transaction_id = self.state_ledger.consume_confirmation(
                confirmation,
                prepared.confirmation_binding_sha256,
            )
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(
                "Execution confirmation is invalid, expired, consumed, or no longer bound."
            ) from None
        receipt = prepared.receipt
        if receipt.route == "blocked":
            return BridgeRunResult(
                status="blocked",
                code="route_blocked",
                receipt=receipt,
                prior_verification=prepared.prior_evidence,
            )

        try:
            with materialize_workspace(
                prepared.source_snapshot, self.config.workspace.scope
            ) as candidate:
                if receipt.route == "premium":
                    return self._execute_premium_candidate(
                        task,
                        prepared,
                        candidate,
                        transaction_id=transaction_id,
                        include_diff=include_diff,
                        capsule_out=capsule_out,
                        prior_commands=(),
                        prior_evidence=prepared.prior_evidence,
                        failure_codes=(
                            "policy_selected_premium",
                            *(
                                item.code
                                for item in prepared.prior_evidence
                                if not item.passed
                            ),
                        ),
                        expected_plan=prepared.commands[0],
                    )

                local_result = self._execute_local_candidate(
                    task,
                    prepared,
                    candidate,
                    local_provider_override=local_provider_override,
                )
                local_evidence, candidate_files, changes = self._verify_candidate(
                    task,
                    candidate,
                    local_result,
                    prepared.verifier_plans,
                )
                prior_failed = tuple(
                    item for item in prepared.prior_evidence if not item.passed
                )
                if local_result.status == "blocked":
                    return BridgeRunResult(
                        status="blocked",
                        code="local_runtime_unavailable",
                        receipt=receipt,
                        prior_verification=prepared.prior_evidence,
                        verification=local_evidence,
                        commands=(local_result,),
                        final_provider=self.config.local.id,
                    )
                if _all_passed(local_evidence) and not prior_failed:
                    self._apply_verified_candidate(
                        task,
                        prepared,
                        candidate,
                        candidate_files,
                        changes,
                        transaction_id=transaction_id,
                    )
                    return BridgeRunResult(
                        status="completed",
                        code="local_verification_passed",
                        receipt=receipt,
                        prior_verification=prepared.prior_evidence,
                        verification=local_evidence,
                        commands=(local_result,),
                        final_provider=self.config.local.id,
                        final_output=local_result.output,
                    )
                if any(
                    item.code == "workspace_mutation_forbidden"
                    for item in local_evidence
                ):
                    return BridgeRunResult(
                        status="failed",
                        code="workspace_authority_violated",
                        receipt=receipt,
                        prior_verification=prepared.prior_evidence,
                        verification=local_evidence,
                        commands=(local_result,),
                        final_provider=self.config.local.id,
                    )
                if receipt.route == "local":
                    return BridgeRunResult(
                        status="failed",
                        code="local_verification_failed_remote_forbidden",
                        receipt=receipt,
                        prior_verification=prepared.prior_evidence,
                        verification=local_evidence,
                        commands=(local_result,),
                        final_provider=self.config.local.id,
                        final_output=local_result.output,
                    )
                return self._execute_premium_candidate(
                    task,
                    prepared,
                    candidate,
                    transaction_id=transaction_id,
                    include_diff=include_diff,
                    capsule_out=capsule_out,
                    prior_commands=(local_result,),
                    prior_evidence=(*prepared.prior_evidence, *local_evidence),
                    failure_codes=tuple(
                        item.code
                        for item in (*prior_failed, *local_evidence)
                        if not item.passed
                    ),
                    expected_plan=(
                        prepared.commands[1]
                        if len(prepared.commands) > 1
                        else None
                    ),
                )
        except WorkspaceSecurityError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def _execute_local_candidate(
        self,
        task: AssistantTaskEnvelope,
        prepared: _PreparedExecution,
        candidate: MaterializedWorkspace,
        *,
        local_provider_override: str | None,
    ) -> CommandResult:
        prompt = build_local_prompt(task)
        with tempfile.TemporaryDirectory(prefix="mymoe-assistant-output-") as tmp:
            output_path = Path(tmp) / "local-final.txt"
            plan = build_codex_command_plan(
                self.config.local,
                prompt=prompt,
                workspace=candidate.root,
                demand=task.capability_demand,
                output_path=output_path,
                local_provider_override=local_provider_override,
                workspace_access=str(
                    prepared.receipt.local_runtime["workspace_access"]
                ),
                runtime_policy=self.config.runtime,
                ephemeral_workspace=True,
            )
            if (
                not prepared.commands
                or plan.command_sha256 != prepared.commands[0].command_sha256
            ):
                raise AssistantBridgeError(
                    "Local command no longer matches the confirmed plan."
                )
            with _isolated_codex_home(copy_auth=False) as codex_home:
                return execute_codex_command(
                    plan,
                    prompt=prompt,
                    output_path=output_path,
                    timeout_seconds=self.config.local.timeout_seconds,
                    environment_overrides={
                        "CODEX_HOME": str(codex_home),
                        "HOME": str(codex_home),
                    },
                )

    def _execute_premium_candidate(
        self,
        task: AssistantTaskEnvelope,
        prepared: _PreparedExecution,
        candidate: MaterializedWorkspace,
        *,
        transaction_id: str,
        include_diff: bool,
        capsule_out: str | Path | None,
        prior_commands: tuple[CommandResult, ...],
        prior_evidence: tuple[VerificationEvidence, ...],
        failure_codes: Sequence[str],
        expected_plan: CommandPlan | None,
    ) -> BridgeRunResult:
        receipt = prepared.receipt
        current_snapshot = snapshot_workspace(
            candidate.root, self.config.workspace.scope
        )
        current_attestation = _receipt_workspace_attestation(current_snapshot)
        capsule_workspace_fingerprint = (
            current_attestation.fingerprint
            if prior_commands
            else receipt.workspace.fingerprint
        )
        bound_external = prepared.prior_evidence
        capsule_evidence = list(prior_evidence)
        existing = {(item.kind, item.id) for item in capsule_evidence}
        capsule_evidence.extend(
            item
            for item in bound_external
            if (item.kind, item.id) not in existing
        )
        diff = collect_git_evidence(
            candidate.root,
            self.config.capsule,
            include_excerpt=include_diff,
            expected_snapshot=current_snapshot,
            workspace_policy=self.config.workspace.scope,
        )
        capsule = build_escalation_capsule(
            task,
            receipt,
            capsule_evidence,
            self.config.capsule,
            failure_codes=failure_codes,
            diff_evidence=diff,
            workspace_fingerprint=capsule_workspace_fingerprint,
        )
        self._write_capsule(capsule, capsule_out)
        premium_result, premium_accounted = self._execute_premium_command(
            task,
            prepared,
            candidate,
            capsule,
            expected_plan=expected_plan,
        )
        if premium_result.code == "durable_premium_budget_exhausted":
            return BridgeRunResult(
                status="blocked",
                code="durable_premium_budget_exhausted",
                receipt=receipt,
                prior_verification=tuple(capsule_evidence),
                verification=(),
                commands=prior_commands,
                capsule=capsule,
                final_provider=(
                    prior_commands[-1].provider_id if prior_commands else None
                ),
            )
        premium_evidence, candidate_files, changes = self._verify_candidate(
            task,
            candidate,
            premium_result,
            prepared.verifier_plans,
        )
        if premium_result.status == "blocked":
            status, code = "blocked", "premium_runtime_unavailable"
        elif _all_passed(premium_evidence):
            self._apply_verified_candidate(
                task,
                prepared,
                candidate,
                candidate_files,
                changes,
                transaction_id=transaction_id,
            )
            status, code = "completed", "premium_verification_passed"
        else:
            status, code = "failed", "premium_verification_failed"
        return BridgeRunResult(
            status=status,
            code=code,
            receipt=receipt,
            prior_verification=tuple(capsule_evidence),
            verification=premium_evidence,
            commands=(*prior_commands, premium_result),
            capsule=capsule,
            final_provider=self.config.premium.id,
            final_output=premium_result.output,
            premium_calls_used=int(premium_accounted),
        )

    def _execute_premium_command(
        self,
        task: AssistantTaskEnvelope,
        prepared: _PreparedExecution,
        candidate: MaterializedWorkspace,
        capsule: EscalationCapsule,
        *,
        expected_plan: CommandPlan | None,
    ) -> tuple[CommandResult, bool]:
        if prepared.premium_auth is None:
            raise AssistantBridgeError(
                "Premium authentication was not bound to the plan."
            )
        prompt = build_premium_prompt(capsule)
        with (
            _premium_workspace(
                self.config.premium,
                task,
                capsule,
                original_workspace=candidate.root,
            ) as premium_workspace,
            tempfile.TemporaryDirectory(prefix="mymoe-assistant-output-") as tmp,
        ):
            output_path = Path(tmp) / "premium-final.txt"
            workspace_access = _effective_workspace_access(
                self.config.premium,
                task.capability_demand,
                allow_remote_workspace=task.allow_remote_workspace,
            )
            plan = build_codex_command_plan(
                self.config.premium,
                prompt=prompt,
                workspace=premium_workspace,
                demand=task.capability_demand,
                output_path=output_path,
                workspace_access=workspace_access,
                runtime_policy=self.config.runtime,
                ephemeral_workspace=True,
            )
            if expected_plan is not None:
                exact = prepared.receipt.route == "premium"
                matches = (
                    plan.command_sha256 == expected_plan.command_sha256
                    if exact
                    else _command_authority_sha256(plan)
                    == _command_authority_sha256(expected_plan)
                )
                if not matches:
                    raise AssistantBridgeError(
                        "Premium command no longer matches the confirmed plan."
                    )
            with _isolated_codex_home(
                copy_auth=True,
                expected_auth=prepared.premium_auth,
            ) as codex_home:
                staged_auth = _attest_staged_premium_auth(
                    Path(codex_home) / "auth.json",
                    expected=prepared.premium_auth,
                )
                premium_accounted = False
                authorization_requested = False

                def reserve_premium_launch() -> ProcessLaunchPermit | None:
                    nonlocal authorization_requested, premium_accounted
                    if authorization_requested:
                        raise AssistantBridgeError(
                            "Premium launch authorization was requested more than once."
                        )
                    authorization_requested = True
                    _verify_staged_premium_auth(
                        staged_auth,
                        expected=prepared.premium_auth,
                    )
                    lease = self._reserve_budget(task, prepared)
                    if lease is None:
                        return None
                    premium_accounted = True

                    def commit() -> None:
                        self._commit_budget(lease)

                    def release_after_popen_failure() -> None:
                        nonlocal premium_accounted
                        self._release_budget_after_popen_failure(lease)
                        premium_accounted = False

                    return ProcessLaunchPermit(
                        commit_after_popen=commit,
                        release_after_popen_failure=release_after_popen_failure,
                    )

                result = execute_codex_command(
                    plan,
                    prompt=prompt,
                    output_path=output_path,
                    timeout_seconds=self.config.premium.timeout_seconds,
                    environment_overrides={
                        "CODEX_HOME": str(codex_home),
                        "HOME": str(codex_home),
                    },
                    reserve_launch=reserve_premium_launch,
                )
                if result.code == "launch_not_authorized":
                    result = replace(
                        result,
                        code="durable_premium_budget_exhausted",
                    )
                return result, premium_accounted

    def _verify_candidate(
        self,
        task: AssistantTaskEnvelope,
        candidate: MaterializedWorkspace,
        result: CommandResult,
        verifier_plans: Sequence[BoundVerifierPlan],
    ) -> tuple[
        tuple[VerificationEvidence, ...],
        tuple[WorkspaceFile, ...],
        tuple[WorkspaceChange, ...],
    ]:
        final_snapshot = snapshot_workspace(
            candidate.root, self.config.workspace.scope
        )
        candidate_files = candidate.snapshot()
        changes = build_changeset(candidate.baseline_files, candidate_files)
        attestation = _receipt_workspace_attestation(final_snapshot)
        with _disposable_verifier_workspace(
            candidate.root,
            source_snapshot=candidate.source_snapshot,
            baseline_files=candidate.baseline_files,
            expected_snapshot=final_snapshot,
            candidate_files=candidate_files,
            changes=changes,
            policy=self.config.workspace.scope,
        ) as verifier_workspace:
            evidence = verify_command_result(
                result,
                self.config,
                task=task,
                workspace=attestation,
                external_evidence=(),
                verifier_workspace=verifier_workspace,
                verifier_plans=verifier_plans,
            )
        evidence = self._enforce_completion_contract(
            task,
            evidence,
            attestation,
            change_count=len(changes),
        )
        return evidence, candidate_files, changes

    def _enforce_completion_contract(
        self,
        task: AssistantTaskEnvelope,
        evidence: Sequence[VerificationEvidence],
        workspace: WorkspaceAttestation,
        *,
        change_count: int,
    ) -> tuple[VerificationEvidence, ...]:
        result = list(evidence)
        if task.capability_demand.risk_class != "write_local" and change_count:
            result.append(
                self._policy_evidence(
                    task,
                    workspace,
                    code="workspace_mutation_forbidden",
                    artifact=str(change_count),
                )
            )
            return tuple(result)
        if task.capability_demand.risk_class == "write_local":
            result.append(
                self._policy_evidence(
                    task,
                    workspace,
                    code="two_phase_required",
                    artifact="independent-attestation",
                )
            )
            if change_count == 0 and not task.no_change_expected:
                result.append(
                    self._policy_evidence(
                        task,
                        workspace,
                        code="workspace_delta_required",
                        artifact="no-delta",
                    )
                )
        return tuple(result)

    def _policy_evidence(
        self,
        task: AssistantTaskEnvelope,
        workspace: WorkspaceAttestation,
        *,
        code: str,
        artifact: str,
    ) -> VerificationEvidence:
        return VerificationEvidence(
            id=_safe_code(code),
            verifier="completion-policy",
            kind="policy",
            passed=False,
            code=_safe_code(code),
            artifact_sha256=_sha256_text(artifact),
            task_fingerprint=task.task_fingerprint,
            workspace_fingerprint=workspace.fingerprint,
            verifier_spec_sha256=_sha256_text(
                _canonical_json(
                    {
                        "risk": task.capability_demand.risk_class,
                        "no_change_expected": task.no_change_expected,
                        "required_verifier_ids": list(
                            task.required_verifier_ids
                        ),
                    }
                )
            ),
        )

    def _apply_verified_candidate(
        self,
        task: AssistantTaskEnvelope,
        prepared: _PreparedExecution,
        candidate: MaterializedWorkspace,
        candidate_files: Sequence[WorkspaceFile],
        changes: Sequence[WorkspaceChange],
        *,
        transaction_id: str,
    ) -> None:
        if not changes:
            return
        if task.capability_demand.risk_class != "write_local":
            raise AssistantBridgeError(
                "A non-write task cannot apply candidate workspace changes."
            )
        if _requires_independent_evidence(task.capability_demand, self.config):
            raise AssistantBridgeError(
                "Independent tasks must use the two-phase resume path."
            )
        apply_changeset(
            source_snapshot=prepared.source_snapshot,
            candidate_root=candidate.root,
            candidate_files=candidate_files,
            changes=changes,
            policy=self.config.workspace.scope,
            state_dir=self.config.workspace.transaction_state_dir,
            transaction_id=transaction_id,
            lock_ttl_seconds=self.config.workspace.transaction_lock_ttl_seconds,
        )

    def _reserve_budget(
        self,
        task: AssistantTaskEnvelope,
        prepared: _PreparedExecution,
    ) -> PremiumBudgetLease | None:
        try:
            key = budget_key(
                namespace=self.config.state.namespace,
                task_fingerprint=task.task_fingerprint,
                config_sha256=prepared.receipt.config_sha256,
                workspace_fingerprint=prepared.source_snapshot.fingerprint,
            )
            return self.state_ledger.reserve_budget(
                key,
                prepared.receipt.premium_call_budget,
                ttl_seconds=self.config.state.budget_lease_ttl_seconds,
            )
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def _commit_budget(self, lease: PremiumBudgetLease) -> None:
        try:
            self.state_ledger.commit_budget(lease)
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def _release_budget_after_popen_failure(
        self,
        lease: PremiumBudgetLease,
    ) -> None:
        try:
            self.state_ledger.release_budget_after_popen_failure(lease)
        except BridgeLedgerError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def _validate_external(
        self,
        evidence: Sequence[VerificationEvidence],
        task: AssistantTaskEnvelope,
        workspace: WorkspaceAttestation,
    ) -> tuple[VerificationEvidence, ...]:
        for item in evidence:
            _validate_external_evidence(item, self.config, task, workspace)
        return tuple(evidence)

    @staticmethod
    def _write_capsule(
        capsule: EscalationCapsule,
        capsule_out: str | Path | None,
    ) -> None:
        if capsule_out is None:
            return
        payload = (json.dumps(capsule.payload(), indent=2) + "\n").encode("utf-8")
        _write_capsule_atomic(capsule_out, payload)


def _write_capsule_atomic(target: str | Path, payload: bytes) -> None:
    raw = Path(target).expanduser()
    path = Path(os.path.abspath(os.fspath(raw)))
    if path.name in {"", ".", ".."}:
        raise AssistantBridgeError("Capsule output must name a regular file.")
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        _validate_capsule_parent(parent)
        if os.name == "nt":
            _write_capsule_windows(path, payload)
        else:
            _write_capsule_posix(path, payload)
    except AssistantBridgeError:
        raise
    except OSError as exc:
        raise AssistantBridgeError(
            "Could not persist escalation capsule atomically."
        ) from exc


def _validate_capsule_parent(parent: Path) -> None:
    try:
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise AssistantBridgeError("Capsule output parent is unavailable.") from exc
    if not resolved.is_dir():
        raise AssistantBridgeError("Capsule output parent is not a directory.")
    current = parent
    while True:
        try:
            details = current.lstat()
        except OSError as exc:
            raise AssistantBridgeError(
                "Capsule output parent could not be attested."
            ) from exc
        if _is_link_or_reparse(details):
            if not _trusted_system_path_alias(current, details):
                raise AssistantBridgeError(
                    "Capsule output parent cannot traverse a symbolic link or reparse point."
                )
        elif not stat.S_ISDIR(details.st_mode):
            raise AssistantBridgeError(
                "Capsule output parent must contain only real directories."
            )
        if current == current.parent:
            break
        current = current.parent


def _trusted_system_path_alias(path: Path, details: os.stat_result) -> bool:
    """Allow only root-managed POSIX aliases such as macOS ``/var``."""

    if os.name != "posix" or not stat.S_ISLNK(details.st_mode):
        return False
    try:
        container = path.parent.lstat()
    except OSError:
        return False
    return (
        int(getattr(details, "st_uid", -1)) == 0
        and int(getattr(container, "st_uid", -1)) == 0
        and not container.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _write_capsule_posix(path: Path, payload: bytes) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, directory_flags)
    temporary_name = ""
    try:
        before = _capsule_target_state(path.name, directory_fd=parent_fd)
        temporary_name, temporary_fd = _open_capsule_temp(
            path.name,
            directory_fd=parent_fd,
        )
        try:
            _write_and_sync_file(temporary_fd, payload)
        finally:
            os.close(temporary_fd)
        after = _capsule_target_state(path.name, directory_fd=parent_fd)
        if before != after:
            raise AssistantBridgeError(
                "Capsule output changed while the atomic write was prepared."
            )
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = ""
        os.fsync(parent_fd)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _write_capsule_windows(path: Path, payload: bytes) -> None:
    """Windows backend using an exclusive peer temp and write-through replace."""

    before = _capsule_target_state(path)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_raw)
    try:
        opened = os.fstat(descriptor)
        observed = temporary.lstat()
        if (
            _is_link_or_reparse(observed)
            or not stat.S_ISREG(observed.st_mode)
            or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
        ):
            raise AssistantBridgeError(
                "Capsule temporary file failed exclusive identity validation."
            )
        try:
            os.fchmod(descriptor, 0o600)
        except AttributeError:  # pragma: no cover - absent on some Windows builds.
            pass
        _write_and_sync_file(descriptor, payload)
        os.close(descriptor)
        descriptor = -1
        _validate_capsule_parent(path.parent)
        after = _capsule_target_state(path)
        if before != after:
            raise AssistantBridgeError(
                "Capsule output changed while the atomic write was prepared."
            )
        _windows_replace_write_through(temporary, path, replace=before is not None)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _open_capsule_temp(name: str, *, directory_fd: int) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(32):
        candidate = f".{name}.{secrets.token_hex(16)}.tmp"
        try:
            return candidate, os.open(
                candidate,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
    raise AssistantBridgeError("Could not allocate an exclusive capsule temp file.")


def _write_and_sync_file(descriptor: int, payload: bytes) -> None:
    try:
        os.fchmod(descriptor, 0o600)
    except AttributeError:  # pragma: no cover - absent on some Windows builds.
        pass
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("capsule write made no progress")
        offset += written
    os.fsync(descriptor)


def _capsule_target_state(
    target: str | Path,
    *,
    directory_fd: int | None = None,
) -> tuple[int, int, int, int, int] | None:
    try:
        if directory_fd is None:
            details = Path(target).lstat()
        else:
            details = os.stat(
                os.fspath(target),
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
    except FileNotFoundError:
        return None
    if _is_link_or_reparse(details) or not stat.S_ISREG(details.st_mode):
        raise AssistantBridgeError(
            "Capsule output cannot replace a link, reparse point, or non-file target."
        )
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
    )


def _is_link_or_reparse(details: os.stat_result) -> bool:
    attributes = int(getattr(details, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse)


def _windows_replace_write_through(
    source: Path,
    target: Path,
    *,
    replace: bool,
) -> None:
    if os.name != "nt":  # pragma: no cover - guarded by backend dispatch.
        raise AssistantBridgeError("Windows capsule backend used on another platform.")
    import ctypes

    move_file_replace_existing = 0x1
    move_file_write_through = 0x8
    flags = move_file_write_through
    if replace:
        flags |= move_file_replace_existing
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file = kernel32.MoveFileExW
    move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    move_file.restype = ctypes.c_int
    if not move_file(str(source), str(target), flags):
        raise ctypes.WinError(ctypes.get_last_error())


def redact_and_bound(value: str, max_chars: int) -> tuple[str, int, bool]:
    redacted, count, truncated, _ = _redact_capsule_text(
        value,
        max_chars,
        SecretRedactionPolicy(),
    )
    return redacted, count, truncated


def _parse_provider(raw: Mapping[str, Any]) -> ProviderSpec:
    _reject_unknown(
        "provider",
        raw,
        {
            "adapter",
            "capabilities",
            "codex_profile",
            "environment_allowlist",
            "executable",
            "execution_scope",
            "extra_args",
            "id",
            "launcher_args",
            "launcher_companions",
            "launcher_entrypoint",
            "local_provider",
            "max_risk",
            "mode",
            "model",
            "network_access",
            "sandbox",
            "timeout_seconds",
            "tools",
            "workspace_access",
        },
    )
    return ProviderSpec(
        id=_string_value(raw.get("id", ""), "provider.id"),
        mode=_string_value(raw.get("mode", ""), "provider.mode"),
        executable=_string_value(raw.get("executable", "codex"), "provider.executable"),
        capabilities=_identifier_tuple(raw.get("capabilities", []), "capabilities"),
        tools=_identifier_tuple(raw.get("tools", []), "tools"),
        max_risk=_string_value(raw.get("max_risk", "read_only"), "provider.max_risk"),
        adapter=_string_value(raw.get("adapter", "codex_cli"), "provider.adapter"),
        execution_scope=_string_value(
            raw.get("execution_scope", "device_only"),
            "provider.execution_scope",
        ),
        local_provider=_string_value(
            raw.get("local_provider", ""),
            "provider.local_provider",
        ),
        codex_profile=_string_value(
            raw.get("codex_profile", ""),
            "provider.codex_profile",
        ),
        model=_string_value(raw.get("model", ""), "provider.model"),
        sandbox=_string_value(
            raw.get("sandbox", "workspace-write"), "provider.sandbox"
        ),
        workspace_access=_string_value(
            raw.get("workspace_access", "read_write"),
            "provider.workspace_access",
        ),
        network_access=_bool_value(
            raw.get("network_access", False),
            "provider.network_access",
        ),
        timeout_seconds=_number_value(
            raw.get("timeout_seconds", 900),
            "provider.timeout_seconds",
        ),
        launcher_args=_string_tuple(raw.get("launcher_args", []), "launcher_args"),
        launcher_entrypoint=_string_value(
            raw.get("launcher_entrypoint", ""), "launcher_entrypoint"
        ),
        launcher_companions=_string_tuple(
            raw.get("launcher_companions", []), "launcher_companions"
        ),
        extra_args=_string_tuple(raw.get("extra_args", []), "extra_args"),
        environment_allowlist=_string_tuple(
            raw.get("environment_allowlist", []),
            "environment_allowlist",
        ),
    )


def _parse_profile(name: str, raw: Mapping[str, Any]) -> ProfilePolicy:
    _reject_unknown(
        f"profile {name}",
        raw,
        {
            "explicit_remote_opt_in",
            "initial_route",
            "max_premium_calls",
            "remote_allowed",
        },
    )
    return ProfilePolicy(
        name=name,
        initial_route=_string_value(
            raw.get("initial_route", "local_then_verify"),
            f"profiles.{name}.initial_route",
        ),
        remote_allowed=_bool_value(
            raw.get("remote_allowed", True),
            f"profiles.{name}.remote_allowed",
        ),
        explicit_remote_opt_in=_bool_value(
            raw.get("explicit_remote_opt_in", False),
            f"profiles.{name}.explicit_remote_opt_in",
        ),
        max_premium_calls=_int_value(
            raw.get("max_premium_calls", 1),
            f"profiles.{name}.max_premium_calls",
        ),
    )


def _parse_command_verifiers(raw: object) -> tuple[CommandVerifierSpec, ...]:
    if not isinstance(raw, list):
        raise AssistantBridgeError("verification.command_verifiers must be a list.")
    parsed: list[CommandVerifierSpec] = []
    for index, item in enumerate(raw):
        value = _as_object(item, f"command_verifiers[{index}]")
        _reject_unknown(
            f"command_verifiers[{index}]",
            value,
            {
                "argv",
                "id",
                "kind",
                "environment_allowlist",
                "execution_boundary",
                "launcher_companions",
                "launcher_entrypoint",
                "network_policy",
                "purpose",
                "required_for_capabilities",
                "required_for_risks",
                "required_for_tools",
                "runtime_read_roots",
                "python_runner",
                "workspace_python_paths",
                "timeout_seconds",
            },
        )
        parsed.append(
            CommandVerifierSpec(
                id=_string_value(value.get("id", ""), f"command_verifiers[{index}].id"),
                argv=_string_tuple(
                    value.get("argv", []), f"command_verifiers[{index}].argv"
                ),
                timeout_seconds=_number_value(
                    value.get("timeout_seconds", 120),
                    f"command_verifiers[{index}].timeout_seconds",
                ),
                kind=_string_value(
                    value.get("kind", "command"),
                    f"command_verifiers[{index}].kind",
                ),
                purpose=_string_value(
                    value.get("purpose", "hygiene"),
                    f"command_verifiers[{index}].purpose",
                ),
                execution_boundary=_string_value(
                    value.get("execution_boundary", "hard_sandbox"),
                    f"command_verifiers[{index}].execution_boundary",
                ),
                network_policy=_string_value(
                    value.get("network_policy", "denied"),
                    f"command_verifiers[{index}].network_policy",
                ),
                launcher_entrypoint=_string_value(
                    value.get("launcher_entrypoint", ""),
                    f"command_verifiers[{index}].launcher_entrypoint",
                ),
                launcher_companions=_string_tuple(
                    value.get("launcher_companions", []),
                    f"command_verifiers[{index}].launcher_companions",
                ),
                runtime_read_roots=_string_tuple(
                    value.get("runtime_read_roots", ["{python_runtime}"]),
                    f"command_verifiers[{index}].runtime_read_roots",
                ),
                python_runner=_string_value(
                    value.get("python_runner", ""),
                    f"command_verifiers[{index}].python_runner",
                ),
                workspace_python_paths=_string_tuple(
                    value.get("workspace_python_paths", []),
                    f"command_verifiers[{index}].workspace_python_paths",
                ),
                environment_allowlist=_string_tuple(
                    value.get("environment_allowlist", []),
                    f"command_verifiers[{index}].environment_allowlist",
                ),
                required_for_capabilities=_identifier_tuple(
                    value.get("required_for_capabilities", []),
                    f"command_verifiers[{index}].required_for_capabilities",
                ),
                required_for_tools=_identifier_tuple(
                    value.get("required_for_tools", []),
                    f"command_verifiers[{index}].required_for_tools",
                ),
                required_for_risks=_string_tuple(
                    value.get("required_for_risks", []),
                    f"command_verifiers[{index}].required_for_risks",
                ),
            )
        )
    if len({item.id for item in parsed}) != len(parsed):
        raise AssistantBridgeError("Command verifier ids must be unique.")
    return tuple(parsed)


def _parse_external_verifiers(raw: object) -> tuple[ExternalVerifierSpec, ...]:
    if not isinstance(raw, list):
        raise AssistantBridgeError("verification.external_verifiers must be a list.")
    parsed: list[ExternalVerifierSpec] = []
    for index, item in enumerate(raw):
        value = _as_object(item, f"external_verifiers[{index}]")
        _reject_unknown(
            f"external_verifiers[{index}]",
            value,
            {"id", "spec_sha256", "verifier"},
        )
        parsed.append(
            ExternalVerifierSpec(
                id=_string_value(
                    value.get("id", ""), f"external_verifiers[{index}].id"
                ),
                verifier=_string_value(
                    value.get("verifier", ""),
                    f"external_verifiers[{index}].verifier",
                ),
                spec_sha256=_string_value(
                    value.get("spec_sha256", ""),
                    f"external_verifiers[{index}].spec_sha256",
                ),
            )
        )
    if len({item.id for item in parsed}) != len(parsed):
        raise AssistantBridgeError("External verifier ids must be unique.")
    return tuple(parsed)


def _verification_checks(raw: object) -> tuple[Mapping[str, Any], ...]:
    try:
        checks = validate_checks(raw, context="assistant-bridge", weighted=False)
    except QualityBenchmarkError as exc:
        raise AssistantBridgeError(str(exc)) from exc
    return tuple(dict(check) for check in checks)


def _provider_runtime_attestation(
    provider: ProviderSpec,
    task: AssistantTaskEnvelope,
    *,
    local_provider_override: str | None = None,
) -> dict[str, object]:
    local_provider = ""
    if provider.mode == "local":
        local_provider = local_provider_override or provider.local_provider
        if local_provider not in {"ollama", "lmstudio"}:
            raise AssistantBridgeError(
                "Local provider override must be ollama or lmstudio."
            )
    authorized = (
        RISK_LEVELS[task.capability_demand.risk_class] <= RISK_LEVELS["write_local"]
    )
    workspace_access = (
        _effective_workspace_access(
            provider,
            task.capability_demand,
            allow_remote_workspace=task.allow_remote_workspace,
        )
        if authorized
        else "not_authorized"
    )
    web_materialized = _requires_web(task.capability_demand) and provider.network_access
    runtime = {
        "provider_id": provider.id,
        "adapter": provider.adapter,
        "execution_scope": provider.execution_scope,
        "model": provider.model,
        "codex_profile": provider.codex_profile or None,
        "local_provider": local_provider or None,
        "sandbox": (
            _effective_sandbox(provider, task.capability_demand)
            if authorized
            else "not_authorized"
        ),
        "workspace_access": workspace_access,
        "agent_tool_network_access": web_materialized,
        "web_search_materialized": web_materialized,
        "user_config_ignored": True,
        "rules_ignored": True,
        "environment_keys": list(provider.environment_allowlist),
    }
    runtime["runtime_sha256"] = _sha256_text(_canonical_json(runtime))
    return runtime


def _effective_sandbox(provider: ProviderSpec, demand: CapabilityDemand) -> str:
    if RISK_LEVELS[demand.risk_class] > RISK_LEVELS["write_local"]:
        raise AssistantBridgeError(
            "Bridge execution cannot authorize external effects."
        )
    required = "workspace-write" if demand.risk_class == "write_local" else "read-only"
    if required == "workspace-write" and provider.sandbox != "workspace-write":
        raise AssistantBridgeError(
            "Provider sandbox ceiling cannot satisfy write_local authority."
        )
    return required


def _effective_workspace_access(
    provider: ProviderSpec,
    demand: CapabilityDemand,
    *,
    allow_remote_workspace: bool,
) -> str:
    if provider.mode == "local":
        required = "read_write" if demand.risk_class == "write_local" else "read_only"
        if required == "read_write" and provider.workspace_access != "read_write":
            raise AssistantBridgeError(
                "Local provider workspace ceiling cannot satisfy write_local authority."
            )
        if provider.workspace_access not in {"read_only", "read_write"}:
            raise AssistantBridgeError(
                "Local provider has an invalid workspace ceiling."
            )
        return required
    if demand.risk_class == "write_local" and allow_remote_workspace:
        if provider.workspace_access != "read_write":
            raise AssistantBridgeError(
                "Premium provider cannot receive write workspace authority."
            )
        return "read_write"
    return "capsule_only"


def _requires_web(demand: CapabilityDemand) -> bool:
    return "web" in demand.required or "web" in demand.tools


def _requires_independent_evidence(
    demand: CapabilityDemand,
    config: AssistantBridgeConfig,
) -> bool:
    return bool(
        set(demand.required).intersection(config.independent_capabilities)
        or set(demand.tools).intersection(config.independent_tools)
        or demand.risk_class in config.independent_risks
    )


def _validate_external_evidence(
    evidence: VerificationEvidence,
    config: AssistantBridgeConfig,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
) -> None:
    if evidence.kind != "external":
        raise AssistantBridgeError(
            "Only external evidence may cross the bridge boundary."
        )
    spec = config.external_verifiers.get(evidence.id)
    if spec is None or spec.verifier != evidence.verifier:
        raise AssistantBridgeError(
            "External verification source is not trusted by configuration."
        )
    if evidence.verifier_spec_sha256 != spec.spec_sha256:
        raise AssistantBridgeError(
            "External verification spec binding does not match configuration."
        )
    if evidence.task_fingerprint != task.task_fingerprint:
        raise AssistantBridgeError(
            "External verification is bound to a different task."
        )
    if evidence.workspace_fingerprint != workspace.fingerprint:
        raise AssistantBridgeError(
            "External verification is bound to a different workspace state."
        )


def _build_verifier_plan(
    spec: CommandVerifierSpec,
    *,
    workspace: str | Path,
    runtime_policy: BridgeRuntimePolicy,
    isolation_policy: VerifierIsolationPolicy = VerifierIsolationPolicy(),
    resource_policy: VerifierResourcePolicy = VerifierResourcePolicy(),
    execution_environment: Mapping[str, str] | None = None,
) -> BoundVerifierPlan:
    root = str(Path(workspace).expanduser().resolve())

    if spec.kind == "trusted_git_diff_check":
        try:
            executable = trusted_git_executable()
        except (WorkspaceSecurityError, OSError, ValueError):
            raise AssistantBridgeError(
                f"Verifier {spec.id} trusted Git attestation failed."
            ) from None
        environment_sha256 = _sha256_text("trusted-git-environment/v1")
        payload = {
            "spec_sha256": spec.spec_sha256,
            "kind": spec.kind,
            "operation": "trusted-git-diff-check/v2",
            "executable": executable.binding_payload(),
            "environment_sha256": environment_sha256,
            "runtime_capabilities": runtime_capabilities().payload(),
            "runtime_policy": runtime_policy.payload(),
        }
        return BoundVerifierPlan(
            spec=spec,
            argv=(),
            executable_identity=executable,
            environment_sha256=environment_sha256,
            environment={},
            launcher_chain=None,
            launcher_authority_sha256="",
            launcher_artifact_sha256=(),
            python_runner_identity=None,
            isolation=None,
            sandbox_launcher_chain=None,
            sandbox_launcher_authority_sha256="",
            sandbox_launcher_artifact_sha256=(),
            resources=None,
            plan_sha256=_sha256_text(_canonical_json(payload)),
            runtime_policy=runtime_policy,
        )

    def expand(value: str) -> str:
        return (
            sys.executable
            if value == "{python}"
            else value.replace("{workspace}", root)
        )

    def expand_launcher(value: str) -> str:
        prefix = "{workspace}/"
        if value.startswith(prefix):
            return value[len(prefix) :]
        return expand(value)

    argv = tuple(expand(item) for item in spec.argv)
    temp_namespace = _sha256_text(spec.id)[:24]
    launcher_entrypoint = (
        expand_launcher(spec.launcher_entrypoint)
        if spec.launcher_entrypoint
        else ""
    )
    launcher_companions = tuple(
        expand_launcher(item) for item in spec.launcher_companions
    )
    provisional_environment = (
        _sanitized_verifier_environment(
            spec.environment_allowlist,
            workspace=root,
            executable_hint=argv[0],
            temp_namespace=temp_namespace,
        )
        if execution_environment is None
        else dict(execution_environment)
    )
    try:
        executable = resolve_executable(
            argv[0],
            env=provisional_environment,
        )
    except (AssistantBridgeRuntimeError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Verifier {spec.id} executable attestation failed."
        ) from None
    python_runner_identity: PythonRunnerIdentity | None = None
    if spec.python_runner:
        python_runner_identity = _attest_python_runner(
            spec.python_runner,
            executable=executable,
            runtime_read_roots=spec.runtime_read_roots,
        )
        workspace_python_paths = _resolved_workspace_python_paths(
            Path(root), spec.workspace_python_paths
        )
        argv = (
            executable.launch_path,
            "-I",
            "-c",
            _PYTHON_RUNNER_LAUNCH,
            python_runner_identity.module_root,
            python_runner_identity.manifest_sha256,
            str(len(workspace_python_paths)),
            *workspace_python_paths,
            *argv[3:],
        )
    else:
        argv = (executable.launch_path, *argv[1:])
    environment = (
        _sanitized_verifier_environment(
            spec.environment_allowlist,
            workspace=root,
            executable_hint=executable.launch_path,
            temp_namespace=temp_namespace,
        )
        if execution_environment is None
        else dict(execution_environment)
    )
    if python_runner_identity is not None:
        environment[_PYTHON_RUNNER_BOOTSTRAP_ENV] = _PYTHON_UNITTEST_BOOTSTRAP
    ephemeral_environment_keys = ("HOME", "TEMP", "TMP", "TMPDIR")
    environment_sha256 = _sha256_text(
        _canonical_json(
            _authority_environment(
                environment,
                ephemeral_keys=ephemeral_environment_keys,
            )
        )
    )
    try:
        launcher_chain = _build_bridge_launcher_chain(
            executable,
            argv[1:],
            entrypoint=launcher_entrypoint,
            companions=launcher_companions,
            workspace=root,
            environment=environment,
        )
    except (AssistantBridgeError, AssistantBridgeRuntimeError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Verifier {spec.id} launcher-chain attestation failed."
        ) from None
    launcher_authority_sha256 = _launcher_chain_authority_sha256(
        launcher_chain,
        executable=executable,
        workspace=root,
        output_path="",
        environment=environment,
        ephemeral_workspace=True,
        ephemeral_environment_keys=ephemeral_environment_keys,
    )
    artifacts = _launcher_artifact_authority_digests(
        launcher_chain,
        workspace=root,
        ephemeral_workspace=True,
    )
    read_artifacts = _verifier_launcher_read_artifacts(executable, launcher_chain)
    capability = verifier_isolation_capability(isolation_policy)
    try:
        isolation = build_verifier_isolation_plan(
            isolation_policy,
            capability,
            workspace=root,
            command_argv=argv,
            runtime_read_roots=spec.runtime_read_roots,
            temp_namespace=temp_namespace,
            attested_read_artifacts=read_artifacts,
        )
    except (VerifierIsolationError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Verifier {spec.id} isolation plan could not be attested."
        ) from None
    sandbox_launcher_chain: LauncherChainIdentity | None = None
    sandbox_launcher_authority_sha256 = ""
    sandbox_artifacts: tuple[str, ...] = ()
    sandbox_executable: ExecutableIdentity | None = None
    if isolation.capability.supported:
        sandbox_executable = isolation.capability.executable
        if sandbox_executable is None:
            raise AssistantBridgeError(
                f"Verifier {spec.id} isolation capability is incomplete."
            )
        try:
            sandbox_companions = tuple(
                (
                    str(Path(item).relative_to(root))
                    if item != executable.launch_path
                    and _path_is_within(item, Path(root))
                    else item
                )
                for item in read_artifacts
            )
            sandbox_launcher_chain = resolve_launcher_chain(
                sandbox_executable,
                ("bound-verifier-isolation",),
                companions=sandbox_companions,
                cwd=root,
                env=environment,
                strict=True,
            )
            sandbox_launcher_chain = replace(
                sandbox_launcher_chain,
                argv=isolation.argv,
            )
        except (
            AssistantBridgeError,
            AssistantBridgeRuntimeError,
            OSError,
            ValueError,
        ):
            raise AssistantBridgeError(
                f"Verifier {spec.id} sandbox launcher-chain attestation failed."
            ) from None
        sandbox_launcher_authority_sha256 = _sandbox_launcher_authority_sha256(
            sandbox_launcher_chain,
            isolation=isolation,
            executable=sandbox_executable,
            workspace=root,
            output_path="",
            environment=environment,
            ephemeral_workspace=True,
            ephemeral_environment_keys=ephemeral_environment_keys,
        )
        sandbox_artifacts = _launcher_artifact_authority_digests(
            sandbox_launcher_chain,
            workspace=root,
            ephemeral_workspace=True,
        )
    sandbox_command_binding_sha256 = _sha256_text(
        _canonical_json(
            {
                "isolation_binding_sha256": isolation.binding_sha256,
                "sandbox_launcher_authority_sha256": (
                    sandbox_launcher_authority_sha256
                ),
                "sandbox_launcher_artifact_sha256": list(sandbox_artifacts),
            }
        )
    )
    try:
        resource_capabilities = verifier_resource_capabilities(resource_policy)
        resources = build_verifier_resource_plan(
            resource_policy,
            resource_capabilities,
            workspace=root,
            command_executable=sandbox_executable,
            command_argv=isolation.argv,
            command_binding_sha256=sandbox_command_binding_sha256,
            environment=environment,
            sandbox_ready=bool(
                isolation.capability.supported
                and sandbox_executable is not None
                and sandbox_launcher_chain is not None
            ),
            wall_time_seconds=spec.timeout_seconds,
            command_companions=read_artifacts,
            command_launcher_chain=sandbox_launcher_chain,
        )
    except (VerifierResourceError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Verifier {spec.id} resource plan could not be attested."
        ) from None
    semantic_argv = [
        "<attested-verifier-executable>",
        *(item.replace(root, "<ephemeral-workspace>") for item in argv[1:]),
    ]
    payload = {
        "spec_sha256": spec.spec_sha256,
        "kind": spec.kind,
        "argv": semantic_argv,
        "executable": _semantic_executable_payload(
            executable,
            authority_environment=_authority_environment(
                environment,
                ephemeral_keys=ephemeral_environment_keys,
            ),
        ),
        "environment_sha256": environment_sha256,
        "launcher_authority_sha256": launcher_authority_sha256,
        "launcher_artifact_sha256": list(artifacts),
        "python_runner": (
            None
            if python_runner_identity is None
            else python_runner_identity.binding_payload()
        ),
        "isolation_binding_sha256": isolation.binding_sha256,
        "sandbox_launcher_authority_sha256": sandbox_launcher_authority_sha256,
        "sandbox_launcher_artifact_sha256": list(sandbox_artifacts),
        "resource_binding_sha256": resources.binding_sha256,
        "runtime_capabilities": runtime_capabilities().payload(),
        "runtime_policy": runtime_policy.payload(),
    }
    return BoundVerifierPlan(
        spec=spec,
        argv=argv,
        executable_identity=executable,
        environment_sha256=environment_sha256,
        environment=environment,
        launcher_chain=launcher_chain,
        launcher_authority_sha256=launcher_authority_sha256,
        launcher_artifact_sha256=artifacts,
        python_runner_identity=python_runner_identity,
        isolation=isolation,
        sandbox_launcher_chain=sandbox_launcher_chain,
        sandbox_launcher_authority_sha256=sandbox_launcher_authority_sha256,
        sandbox_launcher_artifact_sha256=sandbox_artifacts,
        resources=resources,
        plan_sha256=_sha256_text(_canonical_json(payload)),
        runtime_policy=runtime_policy,
    )


def _verifier_launcher_read_artifacts(
    executable: ExecutableIdentity,
    chain: LauncherChainIdentity,
) -> tuple[str, ...]:
    paths = [executable.launch_path]
    for identity in (chain.entrypoint, *chain.companions):
        if identity is not None:
            paths.append(identity.resolved_path)
    for identity in (chain.interpreter, chain.env_launcher):
        if identity is not None:
            paths.append(identity.resolved_path)
    result: list[str] = []
    seen_targets: set[str] = set()
    for item in paths:
        target = os.path.normcase(str(Path(item).resolve(strict=True)))
        if target not in seen_targets:
            seen_targets.add(target)
            result.append(item)
    return tuple(result)


def _sandbox_launcher_authority_sha256(
    chain: LauncherChainIdentity,
    *,
    isolation: VerifierIsolationPlan,
    **kwargs: Any,
) -> str:
    payload = _launcher_chain_authority_payload(chain, **kwargs)
    payload["argv"] = ["<bound-verifier-isolation>", isolation.argv_sha256]
    payload["isolation_binding_sha256"] = isolation.binding_sha256
    return _sha256_text(_canonical_json(payload))


def _run_bound_verifier(
    plan: BoundVerifierPlan,
    *,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
    verifier_workspace: str | Path,
) -> VerificationEvidence:
    if plan.spec.kind == "trusted_git_diff_check":
        return _run_trusted_git_verifier(
            plan,
            task=task,
            workspace=workspace,
            verifier_workspace=verifier_workspace,
        )
    if plan.isolation is None:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} has no bound isolation plan."
        )
    if plan.resources is None:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} has no bound resource plan."
        )
    current = _build_verifier_plan(
        plan.spec,
        workspace=verifier_workspace,
        runtime_policy=plan.runtime_policy,
        isolation_policy=plan.isolation.policy,
        resource_policy=plan.resources.policy,
    )
    if current.plan_sha256 != plan.plan_sha256:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} no longer matches the confirmed plan."
        )
    if current.environment_sha256 != plan.environment_sha256:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} environment no longer matches the confirmed plan."
        )
    if current.isolation is None or not current.isolation.capability.supported:
        return _verifier_evidence(
            plan,
            task=task,
            workspace=workspace,
            passed=False,
            code="hard_sandbox_unavailable",
            observed=0,
            result_payload={
                "plan_sha256": plan.plan_sha256,
                "isolation": (
                    None
                    if current.isolation is None
                    else current.isolation.capability.binding_payload()
                ),
            },
        )
    if current.resources is None or not current.resources.runnable:
        return _verifier_evidence(
            plan,
            task=task,
            workspace=workspace,
            passed=False,
            code="resource_governance_unavailable",
            observed=0,
            result_payload={
                "plan_sha256": plan.plan_sha256,
                "resources": (
                    None
                    if current.resources is None
                    else current.resources.payload()
                ),
            },
        )
    sandbox_executable = current.isolation.capability.executable
    sandbox_launcher_chain = current.sandbox_launcher_chain
    if sandbox_executable is None or sandbox_launcher_chain is None:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} sandbox execution binding is incomplete."
        )
    resource_executable = current.resources.executable
    resource_launcher_chain = current.resources.launcher_chain
    if resource_executable is None or resource_launcher_chain is None:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} resource execution binding is incomplete."
        )
    environment = dict(current.resources.environment)
    internal_temp = Path(current.isolation.internal_temp)
    binding_payload = {
        "plan_sha256": plan.plan_sha256,
        "environment_sha256": current.environment_sha256,
        "executable_fingerprint": plan.executable_identity.fingerprint,
        "launcher_authority_sha256": current.launcher_authority_sha256,
        "launcher_chain_fingerprint": (
            None
            if current.launcher_chain is None
            else current.launcher_chain.fingerprint
        ),
        "launcher_artifact_sha256": list(plan.launcher_artifact_sha256),
        "python_runner": (
            None
            if current.python_runner_identity is None
            else current.python_runner_identity.binding_payload()
        ),
        "isolation_binding_sha256": current.isolation.binding_sha256,
        "sandbox_executable_fingerprint": sandbox_executable.fingerprint,
        "sandbox_launcher_authority_sha256": (
            current.sandbox_launcher_authority_sha256
        ),
        "sandbox_launcher_chain_fingerprint": sandbox_launcher_chain.fingerprint,
        "sandbox_launcher_artifact_sha256": list(
            current.sandbox_launcher_artifact_sha256
        ),
        "resource_binding_sha256": current.resources.binding_sha256,
        "resource_executable_fingerprint": resource_executable.fingerprint,
        "resource_launcher_chain_fingerprint": resource_launcher_chain.fingerprint,
    }
    outcome = None
    with _verifier_internal_temp(internal_temp, verifier_id=plan.spec.id):
        try:
            outcome = execute_process(
                resource_executable,
                current.resources.argv,
                stdin=b"",
                cwd=verifier_workspace,
                env=environment,
                timeout_seconds=plan.spec.timeout_seconds,
                policy=plan.runtime_policy.process_policy(stdin_limit_bytes=0),
                launcher_chain=resource_launcher_chain,
            )
        except ProcessCleanupError:
            raise
        except AssistantBridgeRuntimeError:
            passed = False
            code = "runtime_attestation_failed"
            result_payload = {**binding_payload, "code": code}
            observed = 0
    if outcome is not None:
        try:
            enforcement = build_verifier_resource_enforcement_report(
                current.resources,
                workspace=verifier_workspace,
                stdout_bytes=outcome.stdout_bytes,
                stderr_bytes=outcome.stderr_bytes,
                stdout_limit_bytes=plan.runtime_policy.stdout_limit_bytes,
                stderr_limit_bytes=plan.runtime_policy.stderr_limit_bytes,
                stdout_truncated=outcome.stdout_truncated,
                stderr_truncated=outcome.stderr_truncated,
                cleanup=outcome.cleanup.payload(),
            )
        except (VerifierResourceError, OSError, ValueError):
            raise AssistantBridgeError(
                f"Verifier {plan.spec.id} resource report could not be attested."
            ) from None
        passed = outcome.ok and enforcement.compliant
        if passed:
            code = "command_passed"
        elif outcome.ok:
            code = "resource_policy_violated"
        else:
            code = f"command_{_safe_code(outcome.code)}"
        result_payload = {
            **binding_payload,
            "returncode": outcome.returncode,
            "stdout_sha256": outcome.stdout_sha256,
            "stdout_bytes": outcome.stdout_bytes,
            "stderr_sha256": outcome.stderr_sha256,
            "stderr_bytes": outcome.stderr_bytes,
            "cleanup": outcome.cleanup.payload(),
            "resource_enforcement": enforcement.payload(),
        }
        observed = outcome.stdout_bytes + outcome.stderr_bytes
    return _verifier_evidence(
        plan,
        task=task,
        workspace=workspace,
        passed=passed,
        code=code,
        observed=observed,
        result_payload=result_payload,
    )


@contextmanager
def _verifier_internal_temp(
    path: Path,
    *,
    verifier_id: str,
) -> Iterator[None]:
    try:
        path.mkdir(mode=0o700)
        created = path.lstat()
    except OSError:
        raise AssistantBridgeError(
            f"Verifier {verifier_id} internal temporary root is unavailable."
        ) from None
    if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
        raise AssistantBridgeError(
            f"Verifier {verifier_id} internal temporary root is unsafe."
        )
    try:
        yield
    except BaseException as original:
        try:
            _cleanup_verifier_internal_temp(path, created, verifier_id)
        except AssistantBridgeError as cleanup_error:
            raise original from cleanup_error
        raise
    else:
        _cleanup_verifier_internal_temp(path, created, verifier_id)


def _cleanup_verifier_internal_temp(
    path: Path,
    created: os.stat_result,
    verifier_id: str,
) -> None:
    try:
        current = path.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_dev != created.st_dev
            or current.st_ino != created.st_ino
        ):
            raise OSError("verifier temporary root identity changed")
        shutil.rmtree(path)
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise OSError("verifier temporary root still exists")
    except OSError:
        raise AssistantBridgeError(
            f"Verifier {verifier_id} internal temporary cleanup could not be verified."
        ) from None


def _run_trusted_git_verifier(
    plan: BoundVerifierPlan,
    *,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
    verifier_workspace: str | Path,
) -> VerificationEvidence:
    current = _build_verifier_plan(
        plan.spec,
        workspace=verifier_workspace,
        runtime_policy=plan.runtime_policy,
        isolation_policy=VerifierIsolationPolicy(),
    )
    if current.plan_sha256 != plan.plan_sha256:
        raise AssistantBridgeError(
            f"Verifier {plan.spec.id} no longer matches the confirmed plan."
        )
    binding_payload = {
        "plan_sha256": plan.plan_sha256,
        "operation": "trusted-git-diff-check/v2",
        "executable_fingerprint": current.executable_identity.fingerprint,
    }
    try:
        session = trusted_git_session(verifier_workspace)
        if session.executable.fingerprint != current.executable_identity.fingerprint:
            raise WorkspaceSecurityError(
                "Trusted Git executable changed after confirmation."
            )
        outcome = session.diff_check(
            max_output_bytes=plan.runtime_policy.stdout_limit_bytes
        )
        passed = outcome.ok
        code = "builtin_passed" if passed else "builtin_check_failed"
        result_payload = {
            **binding_payload,
            "returncode": outcome.returncode,
            "stdout_sha256": outcome.stdout_sha256,
            "stdout_bytes": outcome.stdout_bytes,
            "stderr_sha256": outcome.stderr_sha256,
            "stderr_bytes": outcome.stderr_bytes,
            "cleanup": outcome.cleanup.payload(),
        }
        observed = outcome.stdout_bytes + outcome.stderr_bytes
    except ProcessCleanupError:
        raise
    except (AssistantBridgeRuntimeError, WorkspaceSecurityError):
        passed = False
        code = "runtime_attestation_failed"
        result_payload = {**binding_payload, "code": code}
        observed = 0
    return _verifier_evidence(
        plan,
        task=task,
        workspace=workspace,
        passed=passed,
        code=code,
        observed=observed,
        result_payload=result_payload,
    )


def _verifier_evidence(
    plan: BoundVerifierPlan,
    *,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
    passed: bool,
    code: str,
    observed: int,
    result_payload: Mapping[str, object],
) -> VerificationEvidence:
    kind = "builtin" if plan.spec.kind == "trusted_git_diff_check" else "command"
    scheme = "builtin" if kind == "builtin" else "command"
    artifact = _sha256_text(_canonical_json(result_payload))
    return VerificationEvidence(
        id=plan.spec.id,
        verifier=f"{scheme}-{plan.spec.purpose}",
        kind=kind,
        passed=passed,
        code=code,
        artifact_sha256=artifact,
        observed_chars=observed,
        evidence_ref=f"{scheme}://{plan.spec.id}",
        task_fingerprint=task.task_fingerprint,
        workspace_fingerprint=workspace.fingerprint,
        verifier_spec_sha256=plan.spec.spec_sha256,
    )


def _sanitized_verifier_environment(
    allowlist: Sequence[str],
    *,
    workspace: str | Path,
    executable_hint: str,
    temp_namespace: str,
) -> dict[str, str]:
    """Build a minimal verifier environment without inheriting host secrets."""

    explicit: dict[str, str] = {}
    for key in allowlist:
        try:
            validate_environment_name(key)
        except (TypeError, ValueError):
            raise AssistantBridgeError(
                "Verifier environment allowlist contains a denied injection variable."
            ) from None
        if re.search(
            r"(?:AUTH|CREDENTIAL|KEY|PASS(?:WORD)?|SECRET|SESSION|TOKEN)",
            key,
            flags=re.IGNORECASE,
        ):
            raise AssistantBridgeError(
                "Verifier environment allowlist cannot include secret-bearing names."
            )
        value = os.environ.get(key)
        if value is not None:
            explicit[key] = value
    root = Path(workspace).expanduser().resolve()
    internal_temp = root / f".mymoe-verifier-tmp-{temp_namespace}"
    path_entries: list[str] = []
    hinted = Path(executable_hint).expanduser()
    if hinted.is_absolute():
        path_entries.append(str(hinted.resolve(strict=False).parent))
    path_entries.extend(os.defpath.split(os.pathsep))
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if key in {"COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"}
        },
        **explicit,
        "HOME": str(internal_temp),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.pathsep.join(dict.fromkeys(path_entries)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(internal_temp),
        "TMP": str(internal_temp),
        "TMPDIR": str(internal_temp),
    }
    return env


def _resolved_workspace_python_paths(
    workspace: Path,
    relative_paths: Sequence[str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for relative in relative_paths:
        candidate = workspace / relative
        try:
            canonical = candidate.resolve(strict=True)
            canonical.relative_to(workspace)
        except (OSError, ValueError):
            raise AssistantBridgeError(
                "A configured verifier workspace Python path is unavailable or "
                "escapes the workspace."
            ) from None
        if canonical != candidate or not canonical.is_dir():
            raise AssistantBridgeError(
                "A configured verifier workspace Python path must be a real "
                "directory inside the workspace."
            )
        resolved.append(str(canonical))
    return tuple(resolved)


def _attest_python_runner(
    name: str,
    *,
    executable: ExecutableIdentity,
    runtime_read_roots: Sequence[str],
) -> PythonRunnerIdentity:
    if name != "unittest":
        raise AssistantBridgeError("Unsupported typed Python runner.")
    if Path(executable.launch_path).resolve(strict=True) != Path(
        sys.executable
    ).resolve(strict=True):
        raise AssistantBridgeError(
            "Typed Python runner must use the current attested Python runtime."
        )
    standard_library = sysconfig.get_path("stdlib")
    if not standard_library:
        raise AssistantBridgeError("Python standard-library root is unavailable.")
    module_root = (Path(standard_library) / name).resolve(strict=True)
    try:
        metadata = module_root.lstat()
    except OSError:
        raise AssistantBridgeError("Typed Python runner is unavailable.") from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise AssistantBridgeError("Typed Python runner root is unsafe.")
    runtime_roots = expand_runtime_read_roots(runtime_read_roots)
    if not any(
        root.is_dir() and _path_is_within(str(module_root), root)
        for root in runtime_roots
    ):
        raise AssistantBridgeError(
            "Typed Python runner is outside the declared runtime roots."
        )
    manifest_sha256, file_count, total_bytes = _python_runner_manifest(
        module_root
    )
    origin = module_root / "__init__.py"
    if not origin.is_file() or origin.is_symlink():
        raise AssistantBridgeError("Typed Python runner origin is unsafe.")
    return PythonRunnerIdentity(
        name=name,
        module_root=str(module_root),
        module_root_sha256=_sha256_text(str(module_root)),
        manifest_sha256=manifest_sha256,
        file_count=file_count,
        total_bytes=total_bytes,
        device_id=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _python_runner_manifest(root: Path) -> tuple[str, int, int]:
    records: list[list[object]] = []
    total = 0
    count = 0
    try:
        for directory, directories, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(directory)
            directories.sort()
            files.sort()
            for name in directories:
                metadata = (current / name).lstat()
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(
                    metadata.st_mode
                ):
                    raise AssistantBridgeError(
                        "Typed Python runner contains an unsafe directory."
                    )
            for name in files:
                path = current / name
                size, mode, digest = _attest_python_runner_file(path)
                records.append(
                    [path.relative_to(root).as_posix(), size, mode, digest]
                )
                count += 1
                total += size
                if (
                    count > _PYTHON_RUNNER_MAX_FILES
                    or total > _PYTHON_RUNNER_MAX_BYTES
                ):
                    raise AssistantBridgeError(
                        "Typed Python runner exceeds its attestation bound."
                    )
    except OSError:
        raise AssistantBridgeError(
            "Typed Python runner could not be attested."
        ) from None
    return _sha256_text(
        json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    ), count, total


def _attest_python_runner_file(path: Path) -> tuple[int, int, str]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise AssistantBridgeError(
                    "Typed Python runner entry is not a regular file."
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise AssistantBridgeError(
            "Typed Python runner entry could not be attested."
        ) from None
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise AssistantBridgeError(
            "Typed Python runner changed during attestation."
        )
    return int(after.st_size), stat.S_IMODE(after.st_mode), digest.hexdigest()


def _sanitized_environment(
    allowlist: Sequence[str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    for key in allowlist:
        try:
            validate_environment_name(key)
        except (TypeError, ValueError):
            raise AssistantBridgeError(
                "Execution environment allowlist contains a denied injection variable."
            ) from None
    allowed = _BASE_ENV_KEYS | set(allowlist)
    env = {key: value for key, value in os.environ.items() if key in allowed}
    if "PATH" not in env:
        env["PATH"] = os.defpath
    for key, value in (overrides or {}).items():
        if key not in {"CODEX_HOME", "HOME", "TEMP", "TMP", "TMPDIR"}:
            raise AssistantBridgeError(
                "Execution environment override is not permitted."
            )
        if not isinstance(value, str) or "\x00" in value:
            raise AssistantBridgeError("Execution environment override is invalid.")
        env[key] = value
    for proxy in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    ):
        env.pop(proxy, None)
    return env


def _premium_auth_open_flags(*, exclusive: bool) -> int:
    flags = os.O_RDWR if exclusive else os.O_RDONLY
    if exclusive:
        flags |= os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    return flags


def _premium_auth_stat_identity(metadata: os.stat_result) -> tuple[object, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        None
        if not hasattr(metadata, "st_uid")
        else int(getattr(metadata, "st_uid")),
        None
        if not hasattr(metadata, "st_gid")
        else int(getattr(metadata, "st_gid")),
    )


def _read_premium_auth_descriptor(
    descriptor: int,
    path: Path,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise AssistantBridgeError(
            "Premium authentication must be a regular non-symlink file."
        )
    if before.st_size > _MAX_JSON_BYTES:
        raise AssistantBridgeError("Premium authentication artifact is too large.")
    os.lseek(descriptor, 0, os.SEEK_SET)
    content = bytearray()
    while len(content) <= _MAX_JSON_BYTES:
        remaining = _MAX_JSON_BYTES + 1 - len(content)
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        content.extend(chunk)
    if len(content) > _MAX_JSON_BYTES:
        raise AssistantBridgeError("Premium authentication artifact is too large.")
    after = os.fstat(descriptor)
    try:
        observed = path.lstat()
    except OSError:
        raise AssistantBridgeError(
            "Premium authentication changed during attestation."
        ) from None
    if (
        _is_link_or_reparse(observed)
        or not stat.S_ISREG(observed.st_mode)
        or _premium_auth_stat_identity(before)
        != _premium_auth_stat_identity(after)
        or _premium_auth_stat_identity(after)
        != _premium_auth_stat_identity(observed)
        or len(content) != after.st_size
    ):
        raise AssistantBridgeError(
            "Premium authentication changed during attestation."
        )
    return bytes(content), after


def _read_premium_auth_file(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            _premium_auth_open_flags(exclusive=False),
        )
        return _read_premium_auth_descriptor(descriptor, path)
    except AssistantBridgeError:
        raise
    except OSError:
        raise AssistantBridgeError(
            "Premium authentication is unavailable for the confirmed route."
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _premium_auth_source_binding(expected: PremiumAuthAttestation) -> str:
    return _sha256_text(_canonical_json(expected.binding_payload()))


def _staged_premium_auth_attestation(
    path: Path,
    content: bytes,
    metadata: os.stat_result,
    *,
    expected: PremiumAuthAttestation,
) -> StagedPremiumAuthAttestation:
    digest = _sha256_bytes(content)
    if digest != expected.sha256 or len(content) != expected.size_bytes:
        raise AssistantBridgeError(
            "Staged premium authentication does not match the confirmed plan."
        )
    mode = int(metadata.st_mode)
    if os.name == "posix" and stat.S_IMODE(mode) != 0o600:
        raise AssistantBridgeError(
            "Staged premium authentication permissions are unsafe."
        )
    owner_uid = (
        None
        if not hasattr(metadata, "st_uid")
        else int(getattr(metadata, "st_uid"))
    )
    if (
        os.name == "posix"
        and hasattr(os, "geteuid")
        and owner_uid != int(os.geteuid())
    ):
        raise AssistantBridgeError(
            "Staged premium authentication ownership is unsafe."
        )
    return StagedPremiumAuthAttestation(
        path=str(path),
        sha256=digest,
        size_bytes=len(content),
        mtime_ns=int(metadata.st_mtime_ns),
        device_id=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        mode=mode,
        owner_uid=owner_uid,
        owner_gid=(
            None
            if not hasattr(metadata, "st_gid")
            else int(getattr(metadata, "st_gid"))
        ),
        source_binding_sha256=_premium_auth_source_binding(expected),
    )


def _attest_staged_premium_auth(
    path: str | Path,
    *,
    expected: PremiumAuthAttestation,
) -> StagedPremiumAuthAttestation:
    target = Path(path)
    content, metadata = _read_premium_auth_file(target)
    return _staged_premium_auth_attestation(
        target,
        content,
        metadata,
        expected=expected,
    )


def _verify_staged_premium_auth(
    staged: StagedPremiumAuthAttestation,
    *,
    expected: PremiumAuthAttestation,
) -> None:
    current = _attest_staged_premium_auth(staged.path, expected=expected)
    if current.binding_payload() != staged.binding_payload():
        raise AssistantBridgeError(
            "Staged premium authentication changed before launch authorization."
        )


def _sync_premium_auth_parent(parent: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_premium_auth(
    destination: Path,
    content: bytes,
    *,
    expected: PremiumAuthAttestation,
) -> StagedPremiumAuthAttestation:
    if len(content) > _MAX_JSON_BYTES:
        raise AssistantBridgeError("Premium authentication artifact is too large.")
    descriptor = -1
    try:
        descriptor = os.open(
            destination,
            _premium_auth_open_flags(exclusive=True),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
        except AttributeError:  # pragma: no cover - absent on some Windows builds.
            os.chmod(destination, 0o600)
        view = memoryview(content)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("premium authentication write made no progress")
            offset += written
        os.fsync(descriptor)
        readback, metadata = _read_premium_auth_descriptor(
            descriptor,
            destination,
        )
        staged = _staged_premium_auth_attestation(
            destination,
            readback,
            metadata,
            expected=expected,
        )
    except AssistantBridgeError:
        raise
    except OSError:
        raise AssistantBridgeError(
            "Could not stage isolated Codex authentication."
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        _sync_premium_auth_parent(destination.parent)
    except OSError:
        raise AssistantBridgeError(
            "Could not sync isolated Codex authentication."
        ) from None
    current = _attest_staged_premium_auth(destination, expected=expected)
    if current.binding_payload() != staged.binding_payload():
        raise AssistantBridgeError(
            "Staged premium authentication changed after durable staging."
        )
    return current


def _attest_premium_auth(*, include_content: bool = False) -> PremiumAuthAttestation:
    source_home = Path(
        os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
    ).expanduser()
    source = source_home / "auth.json"
    content, _ = _read_premium_auth_file(source)
    try:
        source_path = str(source.resolve(strict=True))
    except OSError:
        raise AssistantBridgeError(
            "Premium authentication changed during attestation."
        ) from None
    return PremiumAuthAttestation(
        source_path=source_path,
        sha256=_sha256_bytes(content),
        size_bytes=len(content),
        content=content if include_content else None,
    )


@contextmanager
def _isolated_codex_home(
    *,
    copy_auth: bool,
    expected_auth: PremiumAuthAttestation | None = None,
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="mymoe-codex-home-") as tmp:
        target = Path(tmp)
        try:
            target.chmod(0o700)
        except OSError:
            pass
        if copy_auth:
            if expected_auth is None:
                raise AssistantBridgeError(
                    "Premium authentication was not bound to the execution plan."
                )
            current = _attest_premium_auth(include_content=True)
            if current.binding_payload() != expected_auth.binding_payload():
                raise AssistantBridgeError(
                    "Premium authentication no longer matches the confirmed plan."
                )
            assert current.content is not None
            destination = target / "auth.json"
            _stage_premium_auth(
                destination,
                current.content,
                expected=current,
            )
        yield target


@contextmanager
def _premium_workspace(
    provider: ProviderSpec,
    task: AssistantTaskEnvelope,
    capsule: EscalationCapsule,
    *,
    original_workspace: str | Path,
) -> Iterator[Path]:
    access = _effective_workspace_access(
        provider,
        task.capability_demand,
        allow_remote_workspace=task.allow_remote_workspace,
    )
    if access == "read_write":
        yield Path(original_workspace).expanduser().resolve()
        return
    with tempfile.TemporaryDirectory(prefix="mymoe-capsule-") as tmp:
        root = Path(tmp)
        capsule_path = root / "capsule.json"
        capsule_path.write_text(
            json.dumps(capsule.payload(), indent=2), encoding="utf-8"
        )
        try:
            capsule_path.chmod(0o600)
        except OSError:
            pass
        yield root


@contextmanager
def _disposable_verifier_workspace(
    source: str | Path,
    *,
    source_snapshot: WorkspaceSnapshot,
    baseline_files: Sequence[WorkspaceFile],
    expected_snapshot: WorkspaceSnapshot,
    candidate_files: Sequence[WorkspaceFile],
    changes: Sequence[WorkspaceChange],
    policy: WorkspaceScopePolicy,
) -> Iterator[Path]:
    source_root = Path(source).resolve()
    expected_baseline = tuple(baseline_files)
    expected_candidate = tuple(candidate_files)
    expected_changes = tuple(changes)
    if expected_snapshot.files != expected_candidate:
        raise WorkspaceSecurityError(
            "Verifier candidate manifest does not match its workspace attestation."
        )
    if build_changeset(expected_baseline, expected_candidate) != expected_changes:
        raise WorkspaceSecurityError(
            "Verifier changes do not match the attested candidate manifest."
        )
    current_source = snapshot_workspace(source_root, policy)
    if current_source.fingerprint != expected_snapshot.fingerprint:
        raise WorkspaceSecurityError(
            "Candidate changed before verifier workspace materialization."
        )
    if current_source.files != expected_candidate:
        raise WorkspaceSecurityError(
            "Candidate manifest changed before verifier workspace materialization."
        )
    with materialize_workspace(source_snapshot, policy) as materialized:
        if materialized.baseline_files != expected_baseline:
            raise WorkspaceSecurityError(
                "Verifier baseline does not match the attested source manifest."
            )
        verifier_baseline = snapshot_workspace(materialized.root, policy)
        if verifier_baseline.files != expected_baseline:
            raise WorkspaceSecurityError(
                "Synthetic verifier HEAD does not match the attested baseline."
            )
        with tempfile.TemporaryDirectory(prefix="mymoe-verifier-state-") as state:
            _workspace_security.apply_changeset(
                source_snapshot=verifier_baseline,
                candidate_root=source_root,
                candidate_files=expected_candidate,
                changes=expected_changes,
                policy=policy,
                state_dir=state,
                transaction_id=secrets.token_hex(16),
            )
        current_source = snapshot_workspace(source_root, policy)
        if current_source.fingerprint != expected_snapshot.fingerprint:
            raise WorkspaceSecurityError(
                "Candidate changed while verifier workspace was materialized."
            )
        verifier_final = snapshot_workspace(materialized.root, policy)
        if verifier_final.files != expected_candidate:
            raise WorkspaceSecurityError(
                "Verifier workspace does not match the final candidate manifest."
            )
        if (
            verifier_final.head_sha != verifier_baseline.head_sha
            or verifier_final.index_sha256 != verifier_baseline.index_sha256
        ):
            raise WorkspaceSecurityError(
                "Verifier materialization unexpectedly changed synthetic HEAD or index."
            )
        yield materialized.root


def _premium_preview_plan(
    provider: ProviderSpec,
    task: AssistantTaskEnvelope,
    prompt: str,
    receipt: RouteDecisionReceipt,
    *,
    workspace: str | Path,
    runtime_policy: BridgeRuntimePolicy,
) -> CommandPlan:
    access = str(receipt.premium_runtime["workspace_access"])
    preview_workspace = (
        _preview_output_path("capsule-workspace").parent
        if access == "capsule_only"
        else Path(workspace).expanduser().resolve()
    )
    return build_codex_command_plan(
        provider,
        prompt=prompt,
        workspace=preview_workspace,
        demand=task.capability_demand,
        workspace_access=access,
        output_path=_preview_output_path("premium"),
        runtime_policy=runtime_policy,
        ephemeral_workspace=True,
    )


def _untracked_manifest(
    raw_paths: bytes,
    snapshot: WorkspaceSnapshot,
) -> bytes:
    files = {item.path: item for item in snapshot.files}
    records: list[dict[str, object]] = []
    for raw in sorted(item for item in raw_paths.split(b"\0") if item):
        relative = os.fsdecode(raw).replace(os.sep, "/")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise AssistantBridgeError("Git reported an unsafe untracked path.")
        item = files.get(relative)
        if item is None or item.kind != "file":
            raise AssistantBridgeError(
                "Git untracked evidence is absent from the attested workspace."
            )
        records.append(
            {
                "path": relative,
                "kind": item.kind,
                "size": item.size,
                "sha256": item.sha256,
            }
        )
    return _canonical_json(records).encode("utf-8")


def _validate_safe_extra_args(values: Sequence[str], *, provider_id: str) -> None:
    if values:
        raise AssistantBridgeError(
            f"Provider {provider_id} extra_args cannot be used inside the isolated authority boundary."
        )


def _bridge_launcher_path(value: str, *, workspace: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(workspace) / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _validate_bridge_launcher_declarations(
    *,
    entrypoint: str,
    companions: Sequence[str],
    workspace: str | Path,
) -> None:
    for value in ((entrypoint,) if entrypoint else ()) + tuple(companions):
        candidate = _bridge_launcher_path(value, workspace=workspace)
        try:
            metadata = candidate.lstat()
        except OSError:
            raise AssistantBridgeError(
                "A declared launcher artifact cannot be attested."
            ) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise AssistantBridgeError(
                "A declared launcher artifact must not be a symbolic link."
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise AssistantBridgeError(
                "A declared launcher artifact must be a regular file."
            )


def _build_bridge_launcher_chain(
    executable: ExecutableIdentity,
    argv: Sequence[str],
    *,
    entrypoint: str,
    companions: Sequence[str],
    workspace: str | Path,
    environment: Mapping[str, str],
) -> LauncherChainIdentity:
    _validate_bridge_launcher_declarations(
        entrypoint=entrypoint,
        companions=companions,
        workspace=workspace,
    )
    return resolve_launcher_chain(
        executable,
        argv,
        entrypoint=entrypoint or None,
        companions=companions,
        cwd=workspace,
        env=environment,
        strict=True,
    )


def _rebind_bridge_launcher_chain(
    planned: LauncherChainIdentity,
    executable: ExecutableIdentity,
    argv: Sequence[str],
    *,
    workspace: str | Path,
    environment: Mapping[str, str],
) -> LauncherChainIdentity:
    entrypoint = ""
    if (
        planned.entrypoint is not None
        and planned.entrypoint.resolved_path != executable.resolved_path
    ):
        entrypoint = planned.entrypoint.requested_path
    companions = tuple(item.requested_path for item in planned.companions)
    return _build_bridge_launcher_chain(
        executable,
        argv,
        entrypoint=entrypoint,
        companions=companions,
        workspace=workspace,
        environment=environment,
    )


def _authority_environment(
    environment: Mapping[str, str],
    *,
    ephemeral_keys: Sequence[str],
) -> dict[str, object]:
    normalized = dict(environment)
    for key in ephemeral_keys:
        normalized[key.upper() if os.name == "nt" else key] = (
            "<ephemeral-runtime>"
        )
    return fingerprint_environment(normalized).payload()


def _path_is_within(path: str, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def _semantic_authority_path(
    value: str,
    *,
    workspace: Path,
    ephemeral_workspace: bool,
) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    if ephemeral_workspace and _path_is_within(str(candidate), workspace):
        relative = candidate.relative_to(workspace)
        if relative == Path("."):
            return "<ephemeral-workspace>"
        return f"<ephemeral-workspace>/{relative.as_posix()}"
    return str(candidate)


def _semantic_executable_payload(
    identity: ExecutableIdentity | None,
    *,
    authority_environment: Mapping[str, object],
) -> dict[str, object] | None:
    if identity is None:
        return None
    payload = identity.binding_payload()
    payload["resolution_environment"] = dict(authority_environment)
    return payload


def _semantic_launcher_artifact_payload(
    identity: Any,
    *,
    workspace: Path,
    ephemeral_workspace: bool,
) -> dict[str, object]:
    payload = identity.binding_payload()
    internal = (
        ephemeral_workspace
        and not Path(identity.requested_path).expanduser().is_absolute()
        and _path_is_within(identity.resolved_path, workspace)
    )
    if internal:
        for key in ("requested_path", "launch_path", "resolved_path"):
            payload[key] = _semantic_authority_path(
                str(payload[key]),
                workspace=workspace,
                ephemeral_workspace=ephemeral_workspace,
            )
        for key in (
            "device_id",
            "inode",
            "launch_path_binding_sha256",
            "mtime_ns",
        ):
            payload.pop(key, None)
    return payload


def _semantic_launcher_argv(
    argv: Sequence[str],
    *,
    workspace: Path,
    output_path: str,
    ephemeral_workspace: bool,
) -> list[str]:
    normalized = list(
        _normalize_ephemeral_paths(
            tuple(argv),
            ephemeral_workspace=ephemeral_workspace,
        )
    )
    if ephemeral_workspace:
        root = str(workspace)
        normalized = [
            item.replace(root, "<ephemeral-workspace>") for item in normalized
        ]
    if output_path:
        normalized = [
            item.replace(output_path, "<ephemeral-output>") for item in normalized
        ]
    return normalized


def _launcher_chain_authority_payload(
    chain: LauncherChainIdentity,
    *,
    executable: ExecutableIdentity,
    workspace: str | Path,
    output_path: str,
    environment: Mapping[str, str],
    ephemeral_workspace: bool,
    ephemeral_environment_keys: Sequence[str],
) -> dict[str, object]:
    root = Path(workspace).expanduser().resolve()
    authority_environment = _authority_environment(
        environment,
        ephemeral_keys=ephemeral_environment_keys,
    )
    return {
        "schema_version": chain.schema_version,
        "executable": _semantic_executable_payload(
            executable,
            authority_environment=authority_environment,
        ),
        "argv": _semantic_launcher_argv(
            chain.argv,
            workspace=root,
            output_path=output_path,
            ephemeral_workspace=ephemeral_workspace,
        ),
        "cwd": (
            "<ephemeral-workspace>" if ephemeral_workspace else chain.cwd
        ),
        "environment": authority_environment,
        "entrypoint": (
            None
            if chain.entrypoint is None
            else _semantic_launcher_artifact_payload(
                chain.entrypoint,
                workspace=root,
                ephemeral_workspace=ephemeral_workspace,
            )
        ),
        "interpreter": _semantic_executable_payload(
            chain.interpreter,
            authority_environment=authority_environment,
        ),
        "env_launcher": _semantic_executable_payload(
            chain.env_launcher,
            authority_environment=authority_environment,
        ),
        "companions": [
            _semantic_launcher_artifact_payload(
                item,
                workspace=root,
                ephemeral_workspace=ephemeral_workspace,
            )
            for item in chain.companions
        ],
        "shebang": list(chain.shebang),
        "strict": chain.strict,
    }


def _launcher_chain_authority_sha256(
    chain: LauncherChainIdentity,
    **kwargs: Any,
) -> str:
    return _sha256_text(
        _canonical_json(_launcher_chain_authority_payload(chain, **kwargs))
    )


def _launcher_artifact_authority_digests(
    chain: LauncherChainIdentity,
    *,
    workspace: str | Path,
    ephemeral_workspace: bool,
) -> tuple[str, ...]:
    root = Path(workspace).expanduser().resolve()
    artifacts = (
        (() if chain.entrypoint is None else (chain.entrypoint,))
        + tuple(chain.companions)
    )
    return tuple(
        _sha256_text(
            _canonical_json(
                _semantic_launcher_artifact_payload(
                    item,
                    workspace=root,
                    ephemeral_workspace=ephemeral_workspace,
                )
            )
        )
        for item in artifacts
    )


def _public_executable_payload(identity: ExecutableIdentity) -> dict[str, object]:
    return dict(identity.payload())


def _normalize_ephemeral_paths(
    argv: tuple[str, ...],
    *,
    ephemeral_workspace: bool,
) -> tuple[str, ...]:
    normalized = list(argv)
    if ephemeral_workspace and "--cd" in normalized:
        index = normalized.index("--cd")
        if index + 1 < len(normalized):
            normalized[index + 1] = "<ephemeral-workspace>"
    if "--output-last-message" in normalized:
        index = normalized.index("--output-last-message")
        if index + 1 < len(normalized):
            normalized[index + 1] = "<ephemeral-output>"
    return tuple(normalized)


def _preview_output_path(label: str) -> Path:
    return Path(tempfile.gettempdir()).resolve() / f"mymoe-assistant-{label}-preview"


def _execution_binding(
    *,
    external_evidence: Sequence[VerificationEvidence],
    include_diff: bool,
    capsule_out: str | Path | None,
    commands: Sequence[CommandPlan],
    verifier_plans: Sequence[BoundVerifierPlan],
    source_snapshot: WorkspaceSnapshot,
    config: AssistantBridgeConfig,
    premium_auth: PremiumAuthAttestation | None,
    workspace_write_capability: WorkspaceWriteCapability,
) -> dict[str, object]:
    evidence_payload = [item.payload() for item in external_evidence]
    capsule_target = (
        str(Path(capsule_out).expanduser().resolve()) if capsule_out is not None else ""
    )
    return {
        "include_diff": include_diff,
        "capsule_out_sha256": (
            _sha256_text(capsule_target) if capsule_target else None
        ),
        "external_evidence_count": len(evidence_payload),
        "external_evidence_sha256": _sha256_text(_canonical_json(evidence_payload)),
        "initial_command_sha256": [item.command_sha256 for item in commands],
        "command_authority_sha256": [
            _command_authority_sha256(item) for item in commands
        ],
        "verifier_plan_sha256": [item.plan_sha256 for item in verifier_plans],
        "source_snapshot": source_snapshot.payload(),
        "source_snapshot_fingerprint": source_snapshot.fingerprint,
        "state": config.state.effective_descriptor(),
        "workspace_policy": config.workspace.effective_descriptor(),
        "workspace_write_capability": workspace_write_capability.payload(),
        "runtime_policy": config.runtime.payload(),
        "runtime_capabilities": runtime_capabilities().payload(),
        "premium_auth": (
            premium_auth.binding_payload() if premium_auth is not None else None
        ),
        "ephemeral_environment_overrides": {
            "CODEX_HOME": "isolated-runtime-placeholder",
            "HOME": "isolated-runtime-placeholder",
        },
    }


def _command_authority_sha256(plan: CommandPlan) -> str:
    semantic_argv = _normalize_ephemeral_paths(
        plan.argv,
        ephemeral_workspace=True,
    )
    return _sha256_text(
        _canonical_json(
            {
                "provider_id": plan.provider_id,
                "mode": plan.mode,
                "argv": list(semantic_argv),
                "sandbox": plan.sandbox,
                "network_access": plan.network_access,
                "workspace_access": plan.workspace_access,
                "model": plan.model,
                "local_provider": plan.local_provider,
                "environment_sha256": plan.environment_sha256,
                "executable_fingerprint": plan.executable_identity.fingerprint,
                "runtime": _deep_thaw(plan.runtime),
                "runtime_policy": plan.runtime_policy.payload(),
                "launcher_authority_sha256": plan.launcher_authority_sha256,
                "launcher_artifact_sha256": list(
                    plan.launcher_artifact_sha256
                ),
            }
        )
    )


def _receipt_with_route(
    receipt: RouteDecisionReceipt,
    *,
    route: str,
    rationale_code: str,
) -> RouteDecisionReceipt:
    expected_flow = {
        "blocked": ("stop",),
        "local": ("local", "verify", "stop"),
    }[route]
    candidate = replace(
        receipt,
        receipt_id="",
        route=route,
        premium_provider=None,
        rationale_codes=(*receipt.rationale_codes, rationale_code),
        expected_flow=expected_flow,
    )
    payload = candidate.payload()
    payload.pop("receipt_id", None)
    return replace(
        candidate,
        receipt_id=f"route-{_sha256_text(_canonical_json(payload))[:32]}",
    )


def _redacted_argv_shape(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    value_flags = {
        "--cd",
        "--config",
        "--local-provider",
        "--model",
        "--output-last-message",
        "--profile",
        "--sandbox",
    }
    skip_value = False
    for index, item in enumerate(argv):
        if index == 0:
            result.append("<executable>")
            continue
        if skip_value:
            result.append("<value>")
            skip_value = False
            continue
        if item.startswith("--") and "=" in item:
            result.append(f"{item.split('=', 1)[0]}=<value>")
        elif item.startswith("-") and not item.startswith("--") and len(item) > 2:
            result.append(f"{item[:2]}<attached-value>")
        else:
            result.append(
                item if item.startswith("-") or item in {"exec"} else "<launcher-arg>"
            )
        if item in value_flags:
            skip_value = True
    return result


def _all_passed(evidence: Sequence[VerificationEvidence]) -> bool:
    return bool(evidence) and all(item.passed for item in evidence)


def _safe_code(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())[:96]
    return normalized or "unspecified_failure"


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path).expanduser()
    try:
        if target.stat().st_size > _MAX_JSON_BYTES:
            raise AssistantBridgeError(f"{label} exceeds its size limit.")
        raw = json.loads(target.read_text(encoding="utf-8"))
    except AssistantBridgeError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise AssistantBridgeError(f"Could not load {label}.") from exc
    return dict(_as_object(raw, label))


def _as_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AssistantBridgeError(f"{label} must be an object.")
    return value


def _reject_unknown(label: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise AssistantBridgeError(f"Unknown {label} keys: {', '.join(unknown)}.")


def _identifier_tuple(value: object, label: str) -> tuple[str, ...]:
    items = _string_tuple(value, label)
    _validate_identifiers(label, items)
    return items


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AssistantBridgeError(f"{label} must be a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise AssistantBridgeError(f"{label} must contain only strings.")
    return tuple(str(item) for item in value)


def _validate_identifiers(label: str, values: Sequence[str]) -> None:
    if len(set(values)) != len(values):
        raise AssistantBridgeError(f"Duplicate {label} values are not allowed.")
    for item in values:
        if _SAFE_ID.fullmatch(item) is None:
            raise AssistantBridgeError(
                f"{label} {item!r} must contain 1-96 safe identifier characters."
            )


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, label)


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssistantBridgeError(f"{label} must be an integer.")
    return value


def _number_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssistantBridgeError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise AssistantBridgeError(f"{label} must be finite.")
    return result


def _bool_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise AssistantBridgeError(f"{label} must be boolean.")
    return value


def _string_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AssistantBridgeError(f"{label} must be a string.")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_deep_freeze(item) for item in value))
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise AssistantBridgeError(f"{label} must be lowercase SHA-256.")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
