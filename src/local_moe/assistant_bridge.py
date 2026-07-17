from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from .deterministic_evaluator import (
    QualityBenchmarkError,
    evaluate_check,
    validate_checks,
)
from .assistant_bridge_ledger import (
    BridgeLedgerError,
    BridgeStateLedger,
    budget_key,
)
from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    ProcessExecutionPolicy,
    execute_process,
    fingerprint_environment,
    inspect_executable,
    runtime_capabilities,
)
from .assistant_bridge_secrets import (
    ResidualAssuranceUnavailableError,
    SecretRedactionPolicy,
    redact_text,
)
from .assistant_bridge_workspace import (
    IgnoredPathRule,
    MaterializedWorkspace,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    WorkspaceSnapshot,
    apply_changeset,
    build_changeset,
    materialize_workspace,
    snapshot_materialized,
    snapshot_workspace,
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
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b(basic\s+)[A-Za-z0-9+/=]{8,}"),
    re.compile(
        r"""(?imx)
        (^|[,{\s])
        (["']?[A-Z0-9_.-]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY)[A-Z0-9_.-]*["']?)
        \s*[:=]\s*
        (?:"[^"]*"|'[^']*'|[^\r\n,;}]+)
        """
    ),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"
    ),
    re.compile(r"(?i)(https?://[^:/\s]+:)[^@/\s]+@"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[a-z]_[A-Za-z0-9]{12,})\b"),
    re.compile(
        r"-----BEGIN [^-]+-----.*?-----END [^-]+-----",
        flags=re.DOTALL,
    ),
)


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
    extra_args: tuple[str, ...] = ()
    environment_allowlist: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)

    def __post_init__(self) -> None:
        for name in (
            "capabilities",
            "tools",
            "launcher_args",
            "extra_args",
            "environment_allowlist",
            "version_args",
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
            ("extra_args", self.extra_args),
            ("version_args", self.version_args),
        ):
            if any(not item or "\x00" in item for item in values):
                raise AssistantBridgeError(
                    f"Provider {self.id} {label} contains an invalid value."
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
            "extra_arg_count": len(self.extra_args),
            "environment_keys": list(self.environment_allowlist),
            "version_args_sha256": _sha256_text(
                _canonical_json(list(self.version_args))
            ),
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
class CommandVerifierSpec:
    id: str
    argv: tuple[str, ...] = field(repr=False)
    timeout_seconds: float
    purpose: str = "hygiene"
    execution_boundary: str = "disposable_workspace"
    network_policy: str = "not_enforced"
    environment_allowlist: tuple[str, ...] = ()
    required_for_capabilities: tuple[str, ...] = ()
    required_for_tools: tuple[str, ...] = ()
    required_for_risks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "argv",
            "environment_allowlist",
            "required_for_capabilities",
            "required_for_tools",
            "required_for_risks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if _SAFE_ID.fullmatch(self.id) is None:
            raise AssistantBridgeError("Command verifier id must be a safe identifier.")
        if not self.argv or any(not item or "\x00" in item for item in self.argv):
            raise AssistantBridgeError(
                f"Command verifier {self.id} requires safe argv values."
            )
        if not 1 <= self.timeout_seconds <= 3600:
            raise AssistantBridgeError(
                f"Command verifier {self.id} timeout_seconds must be between 1 and 3600."
            )
        if self.purpose not in {"hygiene", "task"}:
            raise AssistantBridgeError(
                f"Command verifier {self.id} purpose must be hygiene or task."
            )
        if self.execution_boundary != "disposable_workspace":
            raise AssistantBridgeError(
                f"Command verifier {self.id} requires a disposable workspace boundary."
            )
        if self.network_policy != "not_enforced":
            raise AssistantBridgeError(
                f"Command verifier {self.id} has unsupported network policy."
            )
        for name in self.environment_allowlist:
            if _SAFE_ENV.fullmatch(name) is None:
                raise AssistantBridgeError(
                    f"Command verifier {self.id} environment allowlist is invalid."
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
            "argv_sha256": _sha256_text(_canonical_json(list(self.argv))),
            "timeout_seconds": self.timeout_seconds,
            "purpose": self.purpose,
            "execution_boundary": self.execution_boundary,
            "network_policy": self.network_policy,
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
    status_sha256: str
    staged_diff_sha256: str
    unstaged_diff_sha256: str
    untracked_manifest_sha256: str
    tracked_change_count: int
    untracked_count: int

    def payload(self) -> dict[str, object]:
        return {
            "root_sha256": _sha256_text(self.root),
            "fingerprint": self.fingerprint,
            "git_repository": self.git_repository,
            "head_sha": self.head_sha or None,
            "status_sha256": self.status_sha256,
            "staged_diff_sha256": self.staged_diff_sha256,
            "unstaged_diff_sha256": self.unstaged_diff_sha256,
            "untracked_manifest_sha256": self.untracked_manifest_sha256,
            "tracked_change_count": self.tracked_change_count,
            "untracked_count": self.untracked_count,
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
        if self.kind not in {"command", "external", "output", "process", "policy"}:
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
    stdin_limit_bytes: int = 8 * 1024 * 1024
    stdout_limit_bytes: int = _MAX_STREAM_BYTES
    stderr_limit_bytes: int = _MAX_STREAM_BYTES
    cleanup_grace_seconds: float = 0.25
    cleanup_kill_seconds: float = 0.75
    version_timeout_seconds: float = 3.0
    version_output_limit_bytes: int = 32 * 1024

    def __post_init__(self) -> None:
        if self.require_tree_isolation is not True or self.require_psutil is not True:
            raise AssistantBridgeError(
                "Bridge runtime must require tree isolation and psutil."
            )
        try:
            self.process_policy()
        except ValueError as exc:
            raise AssistantBridgeError(str(exc)) from None
        if not 0.1 <= self.version_timeout_seconds <= 30:
            raise AssistantBridgeError(
                "runtime.version_timeout_seconds is outside safe bounds."
            )
        if not 1024 <= self.version_output_limit_bytes <= 1024 * 1024:
            raise AssistantBridgeError(
                "runtime.version_output_limit_bytes is outside safe bounds."
            )

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
        )

    def payload(self) -> dict[str, object]:
        return {
            "require_tree_isolation": self.require_tree_isolation,
            "require_psutil": self.require_psutil,
            "stdin_limit_bytes": self.stdin_limit_bytes,
            "stdout_limit_bytes": self.stdout_limit_bytes,
            "stderr_limit_bytes": self.stderr_limit_bytes,
            "cleanup_grace_seconds": self.cleanup_grace_seconds,
            "cleanup_kill_seconds": self.cleanup_kill_seconds,
            "version_timeout_seconds": self.version_timeout_seconds,
            "version_output_limit_bytes": self.version_output_limit_bytes,
        }


@dataclass(frozen=True)
class BridgeStatePolicy:
    ledger_path: str = field(repr=False)
    namespace: str
    confirmation_ttl_seconds: float = 300.0
    lock_timeout_seconds: float = 5.0
    stale_lock_seconds: float = 120.0

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
            "executable": self.executable_identity.payload(),
            "environment_sha256": self.environment_sha256,
            "runtime": _deep_thaw(self.runtime),
            "runtime_policy": self.runtime_policy.payload(),
            "launcher_artifact_sha256": list(self.launcher_artifact_sha256),
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
class EscalationCapsule:
    capsule_id: str
    task_id: str
    task_fingerprint: str
    objective: str = field(repr=False)
    objective_sha256: str
    capability_demand: CapabilityDemand
    constraints: tuple[str, ...] = field(repr=False)
    route_receipt_id: str
    workspace_fingerprint: str
    verification: tuple[VerificationEvidence, ...]
    failure_codes: tuple[str, ...]
    diff: DiffEvidence = field(repr=False)
    redaction_count: int
    truncated: bool

    def __post_init__(self) -> None:
        for name in ("constraints", "verification", "failure_codes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "contract": "EscalationCapsule",
            "capsule_id": self.capsule_id,
            "task_id": self.task_id,
            "task_fingerprint": self.task_fingerprint,
            "objective": self.objective,
            "objective_sha256": self.objective_sha256,
            "capability_demand": self.capability_demand.payload(),
            "constraints": list(self.constraints),
            "route_receipt_id": self.route_receipt_id,
            "workspace_fingerprint": self.workspace_fingerprint,
            "verification": [item.payload() for item in self.verification],
            "failure_codes": list(self.failure_codes),
            "diff": self.diff.payload(),
            "redaction": {
                "count": self.redaction_count,
                "truncated": self.truncated,
            },
            "excluded": [
                "conversation_history",
                "hidden_reasoning",
                "local_execution_transcript",
                "command_output",
                "credentials",
            ],
        }

    def metadata_payload(self) -> dict[str, object]:
        serialized = _canonical_json(self.payload())
        return {
            "capsule_id": self.capsule_id,
            "sha256": _sha256_text(serialized),
            "characters": len(serialized),
            "objective_sha256": self.objective_sha256,
            "constraint_count": len(self.constraints),
            "verification_count": len(self.verification),
            "failure_codes": list(self.failure_codes),
            "diff_sha256": self.diff.sha256 or None,
            "redaction_count": self.redaction_count,
            "truncated": self.truncated,
            "content_in_metadata": False,
        }


@dataclass(frozen=True)
class BridgeRunResult:
    status: str
    code: str
    receipt: RouteDecisionReceipt
    verification: tuple[VerificationEvidence, ...] = ()
    commands: tuple[CommandResult, ...] = ()
    capsule: EscalationCapsule | None = None
    final_provider: str | None = None
    final_output: str = field(repr=False, default="")
    premium_calls_used: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "verification", tuple(self.verification))
        object.__setattr__(self, "commands", tuple(self.commands))

    def metadata_payload(self) -> dict[str, object]:
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "mode": "assistant_bridge",
            "status": self.status,
            "code": self.code,
            "route_receipt": self.receipt.payload(),
            "verification": [item.payload() for item in self.verification],
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


class PremiumBudgetLedger:
    """Cross-process, fail-closed call accounting using an atomic directory lock."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._thread_lock = threading.Lock()

    def consume(self, key: str, limit: int) -> bool:
        if limit <= 0:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self._file_lock():
            data = self._read()
            used = data.get(key, 0)
            if isinstance(used, bool) or not isinstance(used, int) or used < 0:
                raise AssistantBridgeError(
                    "Premium budget ledger contains invalid state."
                )
            if used >= limit:
                return False
            data[key] = used + 1
            self._write(data)
            return True

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                lock_path.mkdir(mode=0o700)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise AssistantBridgeError("Premium budget ledger is busy.")
                time.sleep(0.02)
            except OSError as exc:
                raise AssistantBridgeError(
                    "Could not lock premium budget ledger."
                ) from exc
        try:
            yield
        finally:
            try:
                lock_path.rmdir()
            except OSError:
                pass

    def _read(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        try:
            if self.path.stat().st_size > _MAX_JSON_BYTES:
                raise AssistantBridgeError(
                    "Premium budget ledger exceeds its size limit."
                )
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssistantBridgeError("Could not read premium budget ledger.") from exc
        if not isinstance(raw, dict) or set(raw).difference(
            {"schema_version", "usage"}
        ):
            raise AssistantBridgeError("Premium budget ledger has an invalid contract.")
        usage = raw.get("usage", {})
        if not isinstance(usage, dict):
            raise AssistantBridgeError("Premium budget ledger usage must be an object.")
        return {str(key): value for key, value in usage.items()}

    def _write(self, usage: Mapping[str, int]) -> None:
        payload = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "usage": dict(sorted(usage.items())),
        }
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            try:
                temp.chmod(0o600)
            except OSError:
                pass
            temp.replace(self.path)
        except OSError as exc:
            raise AssistantBridgeError(
                "Could not persist premium budget ledger."
            ) from exc


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
            "output_checks",
        },
    )
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
            "require_psutil",
            "require_tree_isolation",
            "stderr_limit_bytes",
            "stdin_limit_bytes",
            "stdout_limit_bytes",
            "version_output_limit_bytes",
            "version_timeout_seconds",
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
    )
    runtime_policy = BridgeRuntimePolicy(
        require_tree_isolation=_bool_value(
            runtime_raw.get("require_tree_isolation", True),
            "runtime.require_tree_isolation",
        ),
        require_psutil=_bool_value(
            runtime_raw.get("require_psutil", True), "runtime.require_psutil"
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
        version_timeout_seconds=_number_value(
            runtime_raw.get("version_timeout_seconds", 3),
            "runtime.version_timeout_seconds",
        ),
        version_output_limit_bytes=_int_value(
            runtime_raw.get("version_output_limit_bytes", 32 * 1024),
            "runtime.version_output_limit_bytes",
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


def plan_assistant_route(
    task: AssistantTaskEnvelope,
    config: AssistantBridgeConfig,
    *,
    workspace: str | Path = ".",
    local_provider_override: str | None = None,
) -> RouteDecisionReceipt:
    workspace_attestation = attest_workspace(workspace)
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


def confirmation_id(
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
    return f"confirm-{_sha256_text(_canonical_json(payload))}"


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
    head = _git_output(root, ("rev-parse", "--verify", "HEAD"), required=False)
    if head is None:
        base = {
            "root_sha256": _sha256_text(str(root)),
            "git_repository": False,
        }
        empty = _sha256_bytes(b"")
        return WorkspaceAttestation(
            root=str(root),
            fingerprint=_sha256_text(_canonical_json(base)),
            git_repository=False,
            head_sha="",
            status_sha256=empty,
            staged_diff_sha256=empty,
            unstaged_diff_sha256=empty,
            untracked_manifest_sha256=empty,
            tracked_change_count=0,
            untracked_count=0,
        )

    status = _git_output(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        required=True,
    )
    staged = _git_output(
        root,
        ("diff", "--cached", "--binary", "--no-ext-diff", "--", "."),
        required=True,
    )
    unstaged = _git_output(
        root,
        ("diff", "--binary", "--no-ext-diff", "--", "."),
        required=True,
    )
    untracked = _git_output(
        root,
        ("ls-files", "--others", "--exclude-standard", "-z"),
        required=True,
    )
    assert status is not None and staged is not None and unstaged is not None
    assert untracked is not None
    manifest = _untracked_manifest(root, untracked)
    head_text = head.decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", head_text) is None:
        raise AssistantBridgeError("Git HEAD attestation is malformed.")
    status_entries = [item for item in status.split(b"\0") if item]
    untracked_entries = [item for item in untracked.split(b"\0") if item]
    untracked_count = len(untracked_entries)
    tracked_count = sum(1 for item in status_entries if not item.startswith(b"?? "))
    fields = {
        "root_sha256": _sha256_text(str(root)),
        "git_repository": True,
        "head_sha": head_text.lower(),
        "status_sha256": _sha256_bytes(status),
        "staged_diff_sha256": _sha256_bytes(staged),
        "unstaged_diff_sha256": _sha256_bytes(unstaged),
        "untracked_manifest_sha256": _sha256_bytes(manifest),
        "tracked_change_count": tracked_count,
        "untracked_count": untracked_count,
    }
    return WorkspaceAttestation(
        root=str(root),
        fingerprint=_sha256_text(_canonical_json(fields)),
        git_repository=True,
        head_sha=head_text.lower(),
        status_sha256=str(fields["status_sha256"]),
        staged_diff_sha256=str(fields["staged_diff_sha256"]),
        unstaged_diff_sha256=str(fields["unstaged_diff_sha256"]),
        untracked_manifest_sha256=str(fields["untracked_manifest_sha256"]),
        tracked_change_count=tracked_count,
        untracked_count=untracked_count,
    )


def collect_git_evidence(
    workspace: str | Path,
    policy: CapsulePolicy,
    *,
    include_excerpt: bool,
) -> DiffEvidence:
    attestation = attest_workspace(workspace)
    if not attestation.git_repository:
        return DiffEvidence(
            sha256="",
            characters=0,
            excerpt="",
            truncated=False,
            staged_sha256=attestation.staged_diff_sha256,
            unstaged_sha256=attestation.unstaged_diff_sha256,
            untracked_manifest_sha256=attestation.untracked_manifest_sha256,
        )
    root = Path(attestation.root)
    staged = _git_output(
        root,
        ("diff", "--cached", "--binary", "--no-ext-diff", "--", "."),
        required=True,
    )
    unstaged = _git_output(
        root,
        ("diff", "--binary", "--no-ext-diff", "--", "."),
        required=True,
    )
    untracked = _git_output(
        root,
        ("ls-files", "--others", "--exclude-standard", "-z"),
        required=True,
    )
    assert staged is not None and unstaged is not None and untracked is not None
    manifest = _untracked_manifest(root, untracked)
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
) -> CommandPlan:
    demand = demand or CapabilityDemand()
    selected_runtime_policy = runtime_policy or BridgeRuntimePolicy()
    resolved_workspace = str(Path(workspace).expanduser().resolve())
    if not Path(resolved_workspace).is_dir():
        raise AssistantBridgeError("Command workspace must be an existing directory.")
    planning_environment = _sanitized_environment(provider.environment_allowlist)
    try:
        executable_identity = inspect_executable(
            provider.executable,
            env=planning_environment,
            version_args=provider.version_args,
            version_timeout_seconds=selected_runtime_policy.version_timeout_seconds,
            version_output_limit_bytes=(
                selected_runtime_policy.version_output_limit_bytes
            ),
            policy=selected_runtime_policy.process_policy(stdin_limit_bytes=0),
        )
    except (AssistantBridgeRuntimeError, OSError, ValueError):
        raise AssistantBridgeError(
            f"Provider {provider.id} executable attestation failed."
        ) from None
    runtime = runtime_capabilities().payload()
    if (
        executable_identity.version is None
        or executable_identity.version.status != "completed"
        or executable_identity.version.returncode != 0
        or executable_identity.version.truncated
    ):
        raise AssistantBridgeError(
            f"Provider {provider.id} version attestation failed."
        )
    launcher_artifacts = _launcher_artifact_digests(
        provider.launcher_args,
        workspace=resolved_workspace,
    )
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
    semantic_argv = _normalize_ephemeral_paths(
        tuple(argv),
        capsule_workspace=effective_workspace_access == "capsule_only",
    )
    command_payload = {
        "provider_id": provider.id,
        "mode": provider.mode,
        "argv": list(semantic_argv),
        "stdin_sha256": _sha256_text(prompt),
        "workspace_sha256": (
            _sha256_text("capsule_only")
            if effective_workspace_access == "capsule_only"
            else _sha256_text(resolved_workspace)
        ),
        "sandbox": sandbox,
        "network_access": web_enabled,
        "workspace_access": effective_workspace_access,
        "model": provider.model,
        "local_provider": selected_local,
        "environment_keys": list(provider.environment_allowlist),
        "environment_sha256": executable_identity.resolution_environment.sha256,
        "executable": executable_identity.payload(),
        "runtime": runtime,
        "runtime_policy": selected_runtime_policy.payload(),
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
        launcher_artifact_sha256=launcher_artifacts,
    )


def execute_codex_command(
    plan: CommandPlan,
    *,
    prompt: str,
    output_path: str | Path,
    timeout_seconds: float,
    environment_overrides: Mapping[str, str] | None = None,
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
    if (
        _launcher_artifact_digests(
            plan.argv[1 : plan.argv.index("--strict-config")],
            workspace=plan.workspace,
        )
        != plan.launcher_artifact_sha256
    ):
        raise AssistantBridgeError(
            "A launcher artifact no longer matches the confirmed plan."
        )
    env = _sanitized_environment(
        plan.environment_allowlist,
        overrides=environment_overrides or {},
    )
    try:
        outcome = execute_process(
            plan.executable_identity,
            plan.argv[1:],
            stdin=prompt.encode("utf-8"),
            cwd=plan.workspace,
            env=env,
            timeout_seconds=timeout_seconds,
            policy=plan.runtime_policy.process_policy(),
        )
    except AssistantBridgeRuntimeError:
        return CommandResult(
            provider_id=plan.provider_id,
            status="blocked",
            code="runtime_attestation_failed",
            returncode=None,
            duration_ms=0,
            stdout_sha256=_sha256_bytes(b""),
            stdout_bytes=0,
            stderr_sha256=_sha256_bytes(b""),
            stderr_bytes=0,
            command_sha256=plan.command_sha256,
        )
    status = "completed" if outcome.ok else "failed"
    code = (
        "launcher_completed"
        if status == "completed"
        else f"launcher_{_safe_code(outcome.code)}"
    )
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

    for spec in config.command_verifiers:
        if spec.applies_to(task.capability_demand):
            evidence.append(
                _run_command_verifier(
                    spec,
                    task=task,
                    workspace=workspace,
                    verifier_workspace=verifier_workspace,
                )
            )

    for item in external_evidence:
        _validate_external_evidence(item, config, task, workspace)
        evidence.append(item)

    if _requires_independent_evidence(task.capability_demand, config) and not any(
        item.kind in {"command", "external"} and item.passed for item in evidence
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
    objective, objective_redactions, objective_truncated = redact_and_bound(
        task.objective,
        policy.max_objective_chars,
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
        value, count, was_truncated = redact_and_bound(raw, remaining)
        if was_truncated:
            raise AssistantBridgeError(
                "Constraints cannot be represented safely inside the escalation capsule."
            )
        redaction_count += count
        constraints.append(value)
        remaining -= len(value)

    if diff_evidence is None:
        diff_excerpt, count, diff_truncated = redact_and_bound(
            diff_text,
            policy.max_diff_chars,
        )
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
    safe_failures = tuple(sorted({_safe_code(item) for item in failure_codes if item}))
    effective_workspace = workspace_fingerprint or receipt.workspace.fingerprint
    for item in verification:
        if item.task_fingerprint != task.task_fingerprint:
            raise AssistantBridgeError(
                "Escalation evidence is bound to a different task."
            )
    base = {
        "task_id": task.task_id,
        "task_fingerprint": task.task_fingerprint,
        "objective": objective,
        "objective_sha256": task.objective_sha256,
        "capability_demand": task.capability_demand.payload(),
        "constraints": constraints,
        "route_receipt_id": receipt.receipt_id,
        "workspace_fingerprint": effective_workspace,
        "verification": [item.payload() for item in verification],
        "failure_codes": list(safe_failures),
        "diff": diff_evidence.payload(),
        "redaction_count": redaction_count,
    }
    capsule = EscalationCapsule(
        capsule_id=f"capsule-{_sha256_text(_canonical_json(base))[:32]}",
        task_id=task.task_id,
        task_fingerprint=task.task_fingerprint,
        objective=objective,
        objective_sha256=task.objective_sha256,
        capability_demand=task.capability_demand,
        constraints=tuple(constraints),
        route_receipt_id=receipt.receipt_id,
        workspace_fingerprint=effective_workspace,
        verification=tuple(verification),
        failure_codes=safe_failures,
        diff=diff_evidence,
        redaction_count=redaction_count,
        truncated=diff_evidence.truncated,
    )
    if len(_canonical_json(capsule.payload())) > policy.max_chars:
        raise AssistantBridgeError(
            "Escalation capsule exceeds max_chars; reduce evidence or capsule limits."
        )
    return capsule


class AssistantBridgeRunner:
    def __init__(
        self,
        config: AssistantBridgeConfig,
        *,
        budget_ledger: PremiumBudgetLedger | None = None,
    ) -> None:
        self.config = config
        self.budget_ledger = budget_ledger or PremiumBudgetLedger(
            config.budget_ledger_path
        )

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
        receipt, commands, execution_binding = self._prepare_execution(
            task,
            workspace=workspace,
            local_provider_override=local_provider_override,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )
        return {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "mode": "assistant_bridge_plan",
            "execute": False,
            "route_receipt": receipt.payload(),
            "confirmation_id": confirmation_id(receipt, execution_binding),
            "execution_binding": _deep_thaw(execution_binding),
            "commands": [command.payload() for command in commands],
            "authority": {
                "process_execution": "requires_exact_confirmation_id",
                "workspace": str(receipt.local_runtime["workspace_access"]),
                "remote_workspace": str(receipt.premium_runtime["workspace_access"]),
                "external_effects": "forbidden",
            },
            "privacy": "metadata_only",
        }

    def _prepare_execution(
        self,
        task: AssistantTaskEnvelope,
        *,
        workspace: str | Path,
        local_provider_override: str | None,
        external_evidence: Sequence[VerificationEvidence],
        include_diff: bool,
        capsule_out: str | Path | None,
    ) -> tuple[
        RouteDecisionReceipt,
        tuple[CommandPlan, ...],
        Mapping[str, object],
    ]:
        receipt = plan_assistant_route(
            task,
            self.config,
            workspace=workspace,
            local_provider_override=local_provider_override,
        )
        bound_external = self._validate_external(
            external_evidence,
            task,
            receipt.workspace,
        )
        commands: list[CommandPlan] = []
        if receipt.route in {"local", "local_then_verify"}:
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
                )
            )
        if receipt.route == "premium":
            preview_diff = collect_git_evidence(
                workspace,
                self.config.capsule,
                include_excerpt=include_diff,
            )
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
                )
            )
        execution_binding = _execution_binding(
            external_evidence=bound_external,
            include_diff=include_diff,
            capsule_out=capsule_out,
            commands=commands,
        )
        return receipt, tuple(commands), _deep_freeze(execution_binding)

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
        receipt, initial_commands, execution_binding = self._prepare_execution(
            task,
            workspace=workspace,
            local_provider_override=local_provider_override,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )
        expected = confirmation_id(receipt, execution_binding)
        if not confirmation or not _constant_time_equal(confirmation, expected):
            raise AssistantBridgeError(
                "Execution confirmation does not match the current task, config, runtime, and workspace receipt."
            )
        if receipt.route == "blocked":
            return BridgeRunResult(
                status="blocked", code="route_blocked", receipt=receipt
            )

        if receipt.route == "premium":
            current = attest_workspace(workspace)
            bound_external = self._validate_external(external_evidence, task, current)
            diff = collect_git_evidence(
                workspace,
                self.config.capsule,
                include_excerpt=include_diff,
            )
            capsule = build_escalation_capsule(
                task,
                receipt,
                bound_external,
                self.config.capsule,
                failure_codes=(
                    "policy_selected_premium",
                    *(item.code for item in bound_external if not item.passed),
                ),
                diff_evidence=diff,
                workspace_fingerprint=current.fingerprint,
            )
            self._write_capsule(capsule, capsule_out)
            if not self._consume_budget(task, receipt):
                return BridgeRunResult(
                    status="blocked",
                    code="durable_premium_budget_exhausted",
                    receipt=receipt,
                    verification=bound_external,
                    capsule=capsule,
                )
            return self._run_premium(
                task,
                receipt,
                capsule,
                workspace=workspace,
                prior_commands=(),
                prior_evidence=bound_external,
                expected_initial_command_sha256=(
                    initial_commands[0].command_sha256 if initial_commands else None
                ),
            )

        local_prompt = build_local_prompt(task)
        with tempfile.TemporaryDirectory(prefix="mymoe-assistant-") as tmp:
            output_path = Path(tmp) / "local-final.txt"
            local_plan = build_codex_command_plan(
                self.config.local,
                prompt=local_prompt,
                workspace=workspace,
                demand=task.capability_demand,
                output_path=output_path,
                local_provider_override=local_provider_override,
                workspace_access=str(receipt.local_runtime["workspace_access"]),
            )
            if (
                not initial_commands
                or local_plan.command_sha256 != initial_commands[0].command_sha256
            ):
                raise AssistantBridgeError(
                    "Execution command no longer matches the confirmed plan."
                )
            with _isolated_codex_home(copy_auth=False) as codex_home:
                local_result = execute_codex_command(
                    local_plan,
                    prompt=local_prompt,
                    output_path=output_path,
                    timeout_seconds=self.config.local.timeout_seconds,
                    environment_overrides={
                        "CODEX_HOME": str(codex_home),
                        "HOME": str(codex_home),
                    },
                )
        current = attest_workspace(workspace)
        bound_external = self._validate_external(external_evidence, task, current)
        local_evidence = verify_command_result(
            local_result,
            self.config,
            task=task,
            workspace=current,
            external_evidence=bound_external,
            verifier_workspace=workspace,
        )
        if local_result.status == "blocked":
            return BridgeRunResult(
                status="blocked",
                code="local_launcher_unavailable",
                receipt=receipt,
                verification=local_evidence,
                commands=(local_result,),
                final_provider=self.config.local.id,
            )
        if _all_passed(local_evidence):
            return BridgeRunResult(
                status="completed",
                code="local_verification_passed",
                receipt=receipt,
                verification=local_evidence,
                commands=(local_result,),
                final_provider=self.config.local.id,
                final_output=local_result.output,
            )
        if receipt.route == "local":
            return BridgeRunResult(
                status="failed",
                code="local_verification_failed_remote_forbidden",
                receipt=receipt,
                verification=local_evidence,
                commands=(local_result,),
                final_provider=self.config.local.id,
                final_output=local_result.output,
            )

        diff = collect_git_evidence(
            workspace,
            self.config.capsule,
            include_excerpt=include_diff,
        )
        capsule = build_escalation_capsule(
            task,
            receipt,
            local_evidence,
            self.config.capsule,
            failure_codes=tuple(
                item.code for item in local_evidence if not item.passed
            ),
            diff_evidence=diff,
            workspace_fingerprint=current.fingerprint,
        )
        self._write_capsule(capsule, capsule_out)
        if not self._consume_budget(task, receipt):
            return BridgeRunResult(
                status="blocked",
                code="durable_premium_budget_exhausted",
                receipt=receipt,
                verification=local_evidence,
                commands=(local_result,),
                capsule=capsule,
                final_provider=self.config.local.id,
                final_output=local_result.output,
            )
        return self._run_premium(
            task,
            receipt,
            capsule,
            workspace=workspace,
            prior_commands=(local_result,),
            prior_evidence=local_evidence,
            expected_initial_command_sha256=None,
        )

    def _run_premium(
        self,
        task: AssistantTaskEnvelope,
        receipt: RouteDecisionReceipt,
        capsule: EscalationCapsule,
        *,
        workspace: str | Path,
        prior_commands: tuple[CommandResult, ...],
        prior_evidence: tuple[VerificationEvidence, ...],
        expected_initial_command_sha256: str | None,
    ) -> BridgeRunResult:
        premium_prompt = build_premium_prompt(capsule)
        with (
            _premium_workspace(
                self.config.premium,
                task,
                capsule,
                original_workspace=workspace,
            ) as premium_workspace,
            tempfile.TemporaryDirectory(prefix="mymoe-assistant-output-") as tmp,
        ):
            output_path = Path(tmp) / "premium-final.txt"
            workspace_access = _effective_workspace_access(
                self.config.premium,
                task.capability_demand,
                allow_remote_workspace=task.allow_remote_workspace,
            )
            premium_plan = build_codex_command_plan(
                self.config.premium,
                prompt=premium_prompt,
                workspace=premium_workspace,
                demand=task.capability_demand,
                output_path=output_path,
                workspace_access=workspace_access,
            )
            if (
                expected_initial_command_sha256 is not None
                and premium_plan.command_sha256 != expected_initial_command_sha256
            ):
                raise AssistantBridgeError(
                    "Execution command no longer matches the confirmed plan."
                )
            with _isolated_codex_home(copy_auth=True) as codex_home:
                premium_result = execute_codex_command(
                    premium_plan,
                    prompt=premium_prompt,
                    output_path=output_path,
                    timeout_seconds=self.config.premium.timeout_seconds,
                    environment_overrides={
                        "CODEX_HOME": str(codex_home),
                        "HOME": str(codex_home),
                    },
                )
            verification_workspace = (
                workspace if workspace_access == "read_write" else premium_workspace
            )
            current = attest_workspace(verification_workspace)
            premium_evidence = verify_command_result(
                premium_result,
                self.config,
                task=task,
                workspace=current,
                verifier_workspace=verification_workspace,
            )
        combined = (*prior_evidence, *premium_evidence)
        if premium_result.status == "blocked":
            status, code = "blocked", "premium_launcher_unavailable"
        elif _all_passed(premium_evidence):
            status, code = "completed", "premium_verification_passed"
        else:
            status, code = "failed", "premium_verification_failed"
        return BridgeRunResult(
            status=status,
            code=code,
            receipt=receipt,
            verification=combined,
            commands=(*prior_commands, premium_result),
            capsule=capsule,
            final_provider=self.config.premium.id,
            final_output=premium_result.output,
            premium_calls_used=1,
        )

    def _consume_budget(
        self,
        task: AssistantTaskEnvelope,
        receipt: RouteDecisionReceipt,
    ) -> bool:
        key = _sha256_text(
            _canonical_json(
                {
                    "task_fingerprint": task.task_fingerprint,
                    "config_sha256": receipt.config_sha256,
                }
            )
        )
        return self.budget_ledger.consume(key, receipt.premium_call_budget)

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
        path = Path(capsule_out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_symlink():
            raise AssistantBridgeError("Capsule output cannot target a symbolic link.")
        temp = path.with_suffix(path.suffix + ".tmp")
        try:
            temp.write_text(json.dumps(capsule.payload(), indent=2), encoding="utf-8")
            try:
                temp.chmod(0o600)
            except OSError:
                pass
            temp.replace(path)
        except OSError as exc:
            raise AssistantBridgeError("Could not persist escalation capsule.") from exc


def redact_and_bound(value: str, max_chars: int) -> tuple[str, int, bool]:
    redacted = value
    count = 0
    for pattern in _SECRET_PATTERNS:
        redacted, replacements = pattern.subn(_redaction_replacement, redacted)
        count += replacements
    truncated = len(redacted) > max_chars
    if truncated:
        redacted = redacted[: max(0, max_chars - 16)] + "...[truncated]"
    return redacted, count, truncated


def _redaction_replacement(match: re.Match[str]) -> str:
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}{match.group(2)}=[redacted]"
    prefix = match.group(1) if match.lastindex else ""
    return f"{prefix}[redacted]"


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
            "local_provider",
            "max_risk",
            "mode",
            "model",
            "network_access",
            "sandbox",
            "timeout_seconds",
            "tools",
            "version_args",
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
        extra_args=_string_tuple(raw.get("extra_args", []), "extra_args"),
        environment_allowlist=_string_tuple(
            raw.get("environment_allowlist", []),
            "environment_allowlist",
        ),
        version_args=_string_tuple(
            raw.get("version_args", ["--version"]),
            "version_args",
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
                "environment_allowlist",
                "execution_boundary",
                "network_policy",
                "purpose",
                "required_for_capabilities",
                "required_for_risks",
                "required_for_tools",
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
                purpose=_string_value(
                    value.get("purpose", "hygiene"),
                    f"command_verifiers[{index}].purpose",
                ),
                execution_boundary=_string_value(
                    value.get("execution_boundary", "disposable_workspace"),
                    f"command_verifiers[{index}].execution_boundary",
                ),
                network_policy=_string_value(
                    value.get("network_policy", "not_enforced"),
                    f"command_verifiers[{index}].network_policy",
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


def _run_command_verifier(
    spec: CommandVerifierSpec,
    *,
    task: AssistantTaskEnvelope,
    workspace: WorkspaceAttestation,
    verifier_workspace: str | Path,
) -> VerificationEvidence:
    root = str(Path(verifier_workspace).expanduser().resolve())
    argv = tuple(
        sys.executable
        if item == "{python}"
        else root
        if item == "{workspace}"
        else item
        for item in spec.argv
    )
    outcome = _execute_bounded_process(
        argv,
        stdin=b"",
        cwd=root,
        env=_sanitized_environment(()),
        timeout_seconds=spec.timeout_seconds,
    )
    passed = not outcome["launch_error"] and outcome["returncode"] == 0
    artifact = _sha256_text(
        _canonical_json(
            {
                "returncode": outcome["returncode"],
                "stdout_sha256": outcome["stdout_sha256"],
                "stdout_bytes": outcome["stdout_bytes"],
                "stderr_sha256": outcome["stderr_sha256"],
                "stderr_bytes": outcome["stderr_bytes"],
            }
        )
    )
    return VerificationEvidence(
        id=spec.id,
        verifier="command",
        kind="command",
        passed=passed,
        code="command_passed" if passed else _safe_code(str(outcome["code"])),
        artifact_sha256=artifact,
        observed_chars=int(outcome["stdout_bytes"]) + int(outcome["stderr_bytes"]),
        evidence_ref=f"command://{spec.id}",
        task_fingerprint=task.task_fingerprint,
        workspace_fingerprint=workspace.fingerprint,
        verifier_spec_sha256=spec.spec_sha256,
    )


def _execute_bounded_process(
    argv: Sequence[str],
    *,
    stdin: bytes,
    cwd: str | Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=dict(env),
            shell=False,
        )
    except FileNotFoundError:
        return _process_error_payload(started, "launcher_unavailable")
    except OSError:
        return _process_error_payload(started, "launcher_os_error")

    stdout_state = _StreamState()
    stderr_state = _StreamState()
    overflow = threading.Event()
    readers = (
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_state, overflow, process),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_state, overflow, process),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    try:
        if process.stdin is not None:
            try:
                process.stdin.write(stdin)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        try:
            returncode = process.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
    finally:
        for reader in readers:
            reader.join(timeout=2.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
    if timed_out:
        code = "launcher_timeout"
    elif overflow.is_set():
        code = "launcher_output_limit_exceeded"
    elif returncode != 0:
        code = "launcher_nonzero_exit"
    else:
        code = "launcher_completed"
    return {
        "launch_error": False,
        "code": code,
        "returncode": returncode,
        "duration_ms": _duration_ms(started),
        "stdout_sha256": stdout_state.hexdigest,
        "stdout_bytes": stdout_state.count,
        "stderr_sha256": stderr_state.hexdigest,
        "stderr_bytes": stderr_state.count,
    }


class _StreamState:
    def __init__(self) -> None:
        self.count = 0
        self._digest = hashlib.sha256()

    def update(self, chunk: bytes) -> None:
        self.count += len(chunk)
        self._digest.update(chunk)

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _drain_stream(
    stream: Any,
    state: _StreamState,
    overflow: threading.Event,
    process: subprocess.Popen[bytes],
) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                return
            state.update(chunk)
            if state.count > _MAX_STREAM_BYTES and not overflow.is_set():
                overflow.set()
                try:
                    process.kill()
                except OSError:
                    pass
    except OSError:
        return


def _process_error_payload(started: float, code: str) -> dict[str, object]:
    empty = _sha256_bytes(b"")
    return {
        "launch_error": True,
        "code": code,
        "returncode": None,
        "duration_ms": _duration_ms(started),
        "stdout_sha256": empty,
        "stdout_bytes": 0,
        "stderr_sha256": empty,
        "stderr_bytes": 0,
    }


def _sanitized_environment(
    allowlist: Sequence[str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
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


@contextmanager
def _isolated_codex_home(*, copy_auth: bool) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="mymoe-codex-home-") as tmp:
        target = Path(tmp)
        try:
            target.chmod(0o700)
        except OSError:
            pass
        if copy_auth:
            source_home = Path(
                os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
            ).expanduser()
            source = source_home / "auth.json"
            if source.is_file() and not source.is_symlink():
                try:
                    if source.stat().st_size > _MAX_JSON_BYTES:
                        raise AssistantBridgeError(
                            "Codex authentication artifact is too large."
                        )
                    destination = target / "auth.json"
                    shutil.copyfile(source, destination)
                    try:
                        destination.chmod(0o600)
                    except OSError:
                        pass
                except OSError as exc:
                    raise AssistantBridgeError(
                        "Could not stage isolated Codex authentication."
                    ) from exc
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


def _premium_preview_plan(
    provider: ProviderSpec,
    task: AssistantTaskEnvelope,
    prompt: str,
    receipt: RouteDecisionReceipt,
    *,
    workspace: str | Path,
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
    )


def _git_output(
    root: Path,
    args: Sequence[str],
    *,
    required: bool,
) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if required:
            raise AssistantBridgeError(
                "Could not produce complete Git evidence."
            ) from exc
        return None
    if completed.returncode != 0:
        if required:
            raise AssistantBridgeError("Could not produce complete Git evidence.")
        return None
    if len(completed.stdout) > _MAX_GIT_BYTES:
        raise AssistantBridgeError("Git evidence exceeds its complete-capture limit.")
    return completed.stdout


def _untracked_manifest(root: Path, raw_paths: bytes) -> bytes:
    records: list[dict[str, object]] = []
    for raw in sorted(item for item in raw_paths.split(b"\0") if item):
        relative = os.fsdecode(raw)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise AssistantBridgeError("Git reported an unsafe untracked path.")
        target = root / relative
        try:
            metadata = target.lstat()
        except OSError as exc:
            raise AssistantBridgeError("Could not attest an untracked path.") from exc
        if target.is_symlink():
            try:
                content = os.readlink(target).encode("utf-8", errors="surrogateescape")
            except OSError as exc:
                raise AssistantBridgeError(
                    "Could not attest an untracked symbolic link."
                ) from exc
            kind = "symlink"
            digest = _sha256_bytes(content)
            size = len(content)
        elif target.is_file():
            kind = "file"
            digest, size = _hash_file(target)
        else:
            kind = "other"
            digest = _sha256_text(kind)
            size = int(metadata.st_size)
        records.append(
            {
                "path": relative.replace(os.sep, "/"),
                "kind": kind,
                "size": size,
                "sha256": digest,
            }
        )
    return _canonical_json(records).encode("utf-8")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise AssistantBridgeError(
                        "An untracked artifact exceeds the complete-attestation limit."
                    )
                digest.update(chunk)
    except OSError as exc:
        raise AssistantBridgeError("Could not hash an untracked artifact.") from exc
    return digest.hexdigest(), size


def _validate_safe_extra_args(values: Sequence[str], *, provider_id: str) -> None:
    if values:
        raise AssistantBridgeError(
            f"Provider {provider_id} extra_args cannot be used inside the isolated authority boundary."
        )


def _launcher_artifact_digests(
    values: Sequence[str],
    *,
    workspace: str | Path,
) -> tuple[str, ...]:
    artifacts: list[str] = []
    base = Path(workspace)
    for value in values:
        candidate = Path(value).expanduser()
        path_shaped = candidate.is_absolute() or len(candidate.parts) > 1
        if not path_shaped:
            continue
        if not candidate.is_absolute():
            candidate = base / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise AssistantBridgeError(
                "A path-shaped launcher artifact cannot be attested."
            ) from None
        if not resolved.is_file() or resolved.is_symlink():
            raise AssistantBridgeError(
                "A launcher artifact must be a regular non-symlink file."
            )
        digest, size = _hash_file(resolved)
        artifacts.append(
            _sha256_text(
                _canonical_json(
                    {
                        "path_sha256": _sha256_text(str(resolved)),
                        "sha256": digest,
                        "size": size,
                    }
                )
            )
        )
    return tuple(artifacts)


def _normalize_ephemeral_paths(
    argv: tuple[str, ...],
    *,
    capsule_workspace: bool,
) -> tuple[str, ...]:
    normalized = list(argv)
    if capsule_workspace and "--cd" in normalized:
        index = normalized.index("--cd")
        if index + 1 < len(normalized):
            normalized[index + 1] = "<capsule-workspace>"
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
    }


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


def _duration_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _optional_returncode(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _constant_time_equal(first: str, second: str) -> bool:
    return (
        isinstance(first, str)
        and isinstance(second, str)
        and hmac.compare_digest(first.encode("utf-8"), second.encode("utf-8"))
    )


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
