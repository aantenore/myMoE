"""Resource-governance contracts for assistant-bridge command verifiers.

Filesystem and network isolation remain owned by
``assistant_bridge_verifier_isolation``.  This seam adds a separate, explicit
resource boundary and never upgrades an observed or post-run control into a
kernel guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Mapping, Sequence

from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    LauncherChainIdentity,
    resolve_executable,
    resolve_launcher_chain,
)


RESOURCE_STRENGTHS = frozenset(
    {"kernel_hard", "process_hard", "supervised", "post_run", "unsupported"}
)
RESOURCE_CONTROLS = (
    "cpu_time",
    "cpu_quota",
    "file_size",
    "open_files",
    "memory",
    "processes",
    "workspace_growth",
)
_STRENGTH_RANK = {
    "unsupported": 0,
    "post_run": 1,
    "supervised": 2,
    "process_hard": 3,
    "kernel_hard": 4,
}
_SCHEMA = "assistant-bridge-verifier-resources/v1"
_LINUX_SYSTEMD_RUN = Path("/usr/bin/systemd-run")
_SHA256 = frozenset("0123456789abcdef")
_DEFAULT_REQUIRED_STRENGTHS = {
    "cpu_time": "process_hard",
    "cpu_quota": "unsupported",
    "file_size": "process_hard",
    "open_files": "process_hard",
    "memory": "unsupported",
    "processes": "unsupported",
    "workspace_growth": "post_run",
}
_POSIX_RESOURCE_SUPERVISOR_ENV = "MYMOE_RESOURCE_SUPERVISOR"
_POSIX_RESOURCE_LAUNCH = (
    "exec(__import__('os').environ['MYMOE_RESOURCE_SUPERVISOR'])"
)


class VerifierResourceError(ValueError):
    """Raised when a resource contract is invalid or cannot be attested."""


@dataclass(frozen=True)
class VerifierResourcePolicy:
    """Provider-neutral resource ceilings and minimum accepted strengths."""

    required: bool = True
    cpu_time_seconds: int = 900
    cpu_quota_percent: int = 200
    file_size_bytes: int = 64 * 1024 * 1024
    open_files: int = 256
    memory_bytes: int = 2 * 1024 * 1024 * 1024
    processes: int = 256
    workspace_growth_bytes: int = 256 * 1024 * 1024
    required_strengths: Mapping[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_REQUIRED_STRENGTHS)
    )
    linux_backend: str = str(_LINUX_SYSTEMD_RUN)

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise VerifierResourceError("resources.required must be boolean.")
        for name, minimum, maximum in (
            ("cpu_time_seconds", 1, 86_400),
            ("cpu_quota_percent", 1, 10_000),
            ("file_size_bytes", 1024, 16 * 1024 * 1024 * 1024),
            ("open_files", 16, 1_048_576),
            ("memory_bytes", 16 * 1024 * 1024, 1024**5),
            ("processes", 1, 1_048_576),
            ("workspace_growth_bytes", 0, 1024**5),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not minimum <= value <= maximum:
                raise VerifierResourceError(
                    f"resources.{name} must be between {minimum} and {maximum}."
                )
        if Path(self.linux_backend) != _LINUX_SYSTEMD_RUN:
            raise VerifierResourceError(
                "The Linux resource backend must be /usr/bin/systemd-run."
            )
        strengths = dict(self.required_strengths)
        if set(strengths) != set(RESOURCE_CONTROLS):
            raise VerifierResourceError(
                "resources.required_strengths must declare every resource control."
            )
        if any(value not in RESOURCE_STRENGTHS for value in strengths.values()):
            raise VerifierResourceError(
                "resources.required_strengths contains an unsupported strength."
            )
        object.__setattr__(
            self,
            "required_strengths",
            MappingProxyType(dict(sorted(strengths.items()))),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "required": self.required,
            "limits": {
                "cpu_time_seconds": self.cpu_time_seconds,
                "cpu_quota_percent": self.cpu_quota_percent,
                "file_size_bytes": self.file_size_bytes,
                "open_files": self.open_files,
                "memory_bytes": self.memory_bytes,
                "processes": self.processes,
                "workspace_growth_bytes": self.workspace_growth_bytes,
            },
            "required_strengths": dict(self.required_strengths),
            "linux_backend_sha256": _sha256_text(self.linux_backend),
        }


@dataclass(frozen=True)
class ResourceControlCapability:
    """One resource control, including the exact strength it can support."""

    strength: str
    mechanism: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.strength not in RESOURCE_STRENGTHS:
            raise VerifierResourceError("Resource capability strength is invalid.")
        if not self.mechanism:
            raise VerifierResourceError("Resource capability mechanism is required.")

    def payload(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "mechanism": self.mechanism,
            "reason": self.reason or None,
        }


@dataclass(frozen=True)
class VerifierResourceCapabilities:
    """Attested host capabilities with explicit operational-strength limits."""

    platform: str
    backend: str
    supported: bool
    controls: Mapping[str, ResourceControlCapability]
    reason: str = ""
    executable: ExecutableIdentity | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        controls = dict(self.controls)
        if set(controls) != set(RESOURCE_CONTROLS):
            raise VerifierResourceError(
                "Resource capabilities must describe every resource control."
            )
        object.__setattr__(
            self, "controls", MappingProxyType(dict(sorted(controls.items())))
        )

    def strength(self, control: str) -> str:
        return self.controls[control].strength

    def missing_requirements(self, policy: VerifierResourcePolicy) -> tuple[str, ...]:
        return tuple(
            control
            for control in RESOURCE_CONTROLS
            if _STRENGTH_RANK[self.strength(control)]
            < _STRENGTH_RANK[policy.required_strengths[control]]
        )

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "platform": self.platform,
            "backend": self.backend,
            "supported": self.supported,
            "controls": {
                name: value.payload() for name, value in self.controls.items()
            },
            "reason": self.reason or None,
            "executable": (
                None
                if self.executable is None
                else self.executable.binding_payload()
            ),
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "platform": self.platform,
            "backend": self.backend,
            "supported": self.supported,
            "controls": {
                name: value.payload() for name, value in self.controls.items()
            },
            "reason": self.reason or None,
            "executable": (
                None if self.executable is None else self.executable.payload()
            ),
            "complete_resource_containment": False,
        }


@dataclass(frozen=True)
class VerifierResourcePlan:
    """Exact resource wrapper bound around an already-attested sandbox command."""

    policy: VerifierResourcePolicy
    capabilities: VerifierResourceCapabilities
    runnable: bool
    reason: str
    executable: ExecutableIdentity | None = field(default=None, repr=False)
    argv: tuple[str, ...] = field(default=(), repr=False)
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    launcher_chain: LauncherChainIdentity | None = field(default=None, repr=False)
    command_binding_sha256: str = ""
    argv_sha256: str = ""
    binding_sha256: str = ""
    workspace_before_bytes: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        if self.runnable and (
            self.executable is None or self.launcher_chain is None or not self.argv
        ):
            raise VerifierResourceError(
                "A runnable resource plan requires an attested launcher."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "policy": self.policy.payload(),
            "capabilities": self.capabilities.payload(),
            "runnable": self.runnable,
            "reason": self.reason or None,
            "command_binding_sha256": self.command_binding_sha256 or None,
            "resource_argv_sha256": self.argv_sha256 or None,
            "binding_sha256": self.binding_sha256,
            "launcher_chain": (
                None
                if self.launcher_chain is None
                else self.launcher_chain.payload()
            ),
            "complete_resource_containment": False,
        }


@dataclass(frozen=True)
class VerifierResourceEnforcementReport:
    """Post-execution evidence with output, cleanup, and resources separated."""

    controls: Mapping[str, Mapping[str, object]]
    output_capture: Mapping[str, object]
    tree_cleanup: Mapping[str, object]
    workspace_growth: Mapping[str, object]
    compliant: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controls",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(value))
                    for key, value in sorted(self.controls.items())
                }
            ),
        )
        object.__setattr__(
            self, "output_capture", MappingProxyType(dict(self.output_capture))
        )
        object.__setattr__(
            self, "tree_cleanup", MappingProxyType(dict(self.tree_cleanup))
        )
        object.__setattr__(
            self, "workspace_growth", MappingProxyType(dict(self.workspace_growth))
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "controls": {
                name: dict(value) for name, value in self.controls.items()
            },
            "output_capture": dict(self.output_capture),
            "tree_cleanup": dict(self.tree_cleanup),
            "workspace_growth": dict(self.workspace_growth),
            "compliant": self.compliant,
            "complete_resource_containment": False,
        }


def verifier_resource_capabilities(
    policy: VerifierResourcePolicy,
    *,
    platform_name: str | None = None,
) -> VerifierResourceCapabilities:
    """Return the strongest currently verifiable backend for this host."""

    selected = sys.platform if platform_name is None else platform_name
    if selected == "darwin":
        return _setrlimit_capabilities(policy, platform_name=selected)
    if selected.startswith("linux"):
        cgroup = _linux_cgroup_capabilities(policy, platform_name=selected)
        if cgroup is not None:
            return cgroup
        fallback = _setrlimit_capabilities(policy, platform_name=selected)
        return VerifierResourceCapabilities(
            platform=fallback.platform,
            backend=fallback.backend,
            supported=fallback.supported,
            controls=fallback.controls,
            reason=(
                "cgroup v2/systemd transient-unit enforcement was not verified; "
                "only inherited per-process limits are available"
            ),
            executable=fallback.executable,
        )
    if selected.startswith("win"):
        return _windows_job_capabilities(selected)
    return _unsupported_capabilities(
        selected, "no supported verifier resource backend for this platform"
    )


def build_verifier_resource_plan(
    policy: VerifierResourcePolicy,
    capabilities: VerifierResourceCapabilities,
    *,
    workspace: str | Path,
    command_executable: ExecutableIdentity | None,
    command_argv: Sequence[str],
    command_binding_sha256: str,
    environment: Mapping[str, str],
    sandbox_ready: bool,
    command_companions: Sequence[str | Path] = (),
    command_launcher_chain: LauncherChainIdentity | None = None,
) -> VerifierResourcePlan:
    """Bind resource enforcement around a sandbox; never replace the sandbox."""

    root = Path(workspace).expanduser().resolve(strict=True)
    _require_sha256(command_binding_sha256, "command binding")
    before_bytes = measure_workspace_bytes(root)
    missing = capabilities.missing_requirements(policy)
    blocked_reason = ""
    if not sandbox_ready or command_executable is None:
        blocked_reason = "filesystem_network_sandbox_unavailable"
    elif policy.required and missing:
        blocked_reason = "required_resource_strength_unavailable:" + ",".join(
            missing
        )
    elif capabilities.backend == "windows-job-object-contract":
        blocked_reason = "windows_job_object_execution_requires_sandbox_adapter"
    elif not capabilities.supported and policy.required:
        blocked_reason = "resource_backend_unavailable"

    if blocked_reason:
        binding = _plan_binding(
            policy,
            capabilities,
            command_binding_sha256,
            runnable=False,
            reason=blocked_reason,
            resource_argv_sha256="",
            launcher_chain=None,
        )
        return VerifierResourcePlan(
            policy=policy,
            capabilities=capabilities,
            runnable=False,
            reason=blocked_reason,
            command_binding_sha256=command_binding_sha256,
            binding_sha256=_sha256_json(binding),
            workspace_before_bytes=before_bytes,
            environment=environment,
        )

    executable: ExecutableIdentity
    argv: tuple[str, ...]
    semantic_argv: Mapping[str, object]
    companions = _deduplicate_companions(
        (command_executable.launch_path, *command_companions)
    )
    resource_environment = dict(environment)
    if capabilities.backend in {"darwin-setrlimit", "linux-setrlimit"}:
        if capabilities.executable is None:
            raise VerifierResourceError(
                "The setrlimit capability has no attested Python executable."
            )
        executable = capabilities.executable
        limits = {
            "cpu_time_seconds": policy.cpu_time_seconds,
            "file_size_bytes": policy.file_size_bytes,
            "open_files": policy.open_files,
        }
        argv = (
            "-I",
            "-c",
            _POSIX_RESOURCE_LAUNCH,
            _canonical_json(limits),
            command_executable.launch_path,
            *tuple(command_argv),
        )
        resource_environment[_POSIX_RESOURCE_SUPERVISOR_ENV] = (
            _POSIX_RESOURCE_SUPERVISOR
        )
        semantic_argv = {
            "wrapper": "fixed-python-setrlimit/v1",
            "supervisor_sha256": _sha256_text(_POSIX_RESOURCE_SUPERVISOR),
            "limits": limits,
            "command_binding_sha256": command_binding_sha256,
        }
    elif capabilities.backend == "linux-systemd-cgroup-v2":
        if capabilities.executable is None:
            raise VerifierResourceError(
                "The systemd capability has no attested executable."
            )
        executable = capabilities.executable
        properties = (
            "KillMode=control-group",
            f"CPUQuota={policy.cpu_quota_percent}%",
            f"LimitCPU={policy.cpu_time_seconds}",
            f"LimitFSIZE={policy.file_size_bytes}",
            f"LimitNOFILE={policy.open_files}",
            f"MemoryMax={policy.memory_bytes}",
            f"TasksMax={policy.processes}",
        )
        argv = (
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            *(f"--property={item}" for item in properties),
            "--",
            command_executable.launch_path,
            *tuple(command_argv),
        )
        semantic_argv = {
            "wrapper": "systemd-user-scope-cgroup-v2/v1",
            "properties": list(properties),
            "command_binding_sha256": command_binding_sha256,
        }
    else:
        if command_launcher_chain is None:
            raise VerifierResourceError(
                "Optional resource pass-through requires the sandbox launcher chain."
            )
        executable = command_executable
        argv = tuple(command_argv)
        semantic_argv = {
            "wrapper": "sandbox-pass-through/v1",
            "command_binding_sha256": command_binding_sha256,
        }

    try:
        launcher_chain = (
            command_launcher_chain
            if executable is command_executable
            else replace(
                resolve_launcher_chain(
                    executable,
                    ("bound-verifier-resource-governance",),
                    companions=companions,
                    cwd=root,
                    env=resource_environment,
                    strict=True,
                ),
                argv=argv,
            )
        )
    except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
        raise VerifierResourceError(
            "Resource launcher-chain attestation failed."
        ) from exc
    argv_sha256 = _sha256_json(semantic_argv)
    binding = _plan_binding(
        policy,
        capabilities,
        command_binding_sha256,
        runnable=True,
        reason="",
        resource_argv_sha256=argv_sha256,
        launcher_chain=launcher_chain,
    )
    return VerifierResourcePlan(
        policy=policy,
        capabilities=capabilities,
        runnable=True,
        reason="",
        executable=executable,
        argv=argv,
        environment=resource_environment,
        launcher_chain=launcher_chain,
        command_binding_sha256=command_binding_sha256,
        argv_sha256=argv_sha256,
        binding_sha256=_sha256_json(binding),
        workspace_before_bytes=before_bytes,
    )


def build_verifier_resource_enforcement_report(
    plan: VerifierResourcePlan,
    *,
    workspace: str | Path,
    stdout_bytes: int,
    stderr_bytes: int,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    stdout_truncated: bool,
    stderr_truncated: bool,
    cleanup: Mapping[str, object],
) -> VerifierResourceEnforcementReport:
    """Finalize bounded evidence after the verifier and its temp cleanup finish."""

    after_bytes = measure_workspace_bytes(workspace)
    signed_delta = after_bytes - plan.workspace_before_bytes
    growth = max(0, signed_delta)
    workspace_ok = growth <= plan.policy.workspace_growth_bytes
    output_ok = not stdout_truncated and not stderr_truncated
    cleanup_ok = cleanup.get("verified") is True
    controls = {
        name: {
            "strength": capability.strength,
            "mechanism": capability.mechanism,
            "required_strength": plan.policy.required_strengths[name],
            "available": capability.strength != "unsupported",
        }
        for name, capability in plan.capabilities.controls.items()
    }
    output_capture = {
        "strength": "supervised",
        "resource_control": False,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "stdout_limit_bytes": stdout_limit_bytes,
        "stderr_limit_bytes": stderr_limit_bytes,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "within_bound": output_ok,
    }
    tree_cleanup = {
        "strength": "supervised",
        "resource_control": False,
        "verified": cleanup_ok,
        "verification_scope": cleanup.get("verification_scope"),
        "hard_containment": cleanup.get("hard_containment") is True,
        "observed_descendants": cleanup.get("observed_descendants"),
    }
    workspace_growth = {
        "strength": "post_run",
        "resource_control": True,
        "quota_enforced_during_execution": False,
        "before_bytes": plan.workspace_before_bytes,
        "after_bytes": after_bytes,
        "signed_delta_bytes": signed_delta,
        "growth_bytes": growth,
        "limit_bytes": plan.policy.workspace_growth_bytes,
        "within_bound": workspace_ok,
    }
    return VerifierResourceEnforcementReport(
        controls=controls,
        output_capture=output_capture,
        tree_cleanup=tree_cleanup,
        workspace_growth=workspace_growth,
        compliant=bool(plan.runnable and workspace_ok and output_ok and cleanup_ok),
    )


def measure_workspace_bytes(workspace: str | Path) -> int:
    """Measure regular-file bytes without following workspace links."""

    root = Path(workspace).expanduser().resolve(strict=True)
    total = 0
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directories.sort()
        files.sort()
        for name in directories:
            metadata = current.joinpath(name).lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise VerifierResourceError(
                    "Workspace resource accounting rejects linked directories."
                )
        for name in files:
            metadata = current.joinpath(name).lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise VerifierResourceError(
                    "Workspace resource accounting accepts regular files only."
                )
            total += metadata.st_size
    return total


def _setrlimit_capabilities(
    policy: VerifierResourcePolicy,
    *,
    platform_name: str,
) -> VerifierResourceCapabilities:
    try:
        import resource

        for name in ("RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE"):
            if not hasattr(resource, name):
                raise OSError(f"{name} is unavailable")
        environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
        executable = resolve_executable(sys.executable, env=environment)
    except (AssistantBridgeRuntimeError, ImportError, OSError, ValueError):
        return _unsupported_capabilities(
            platform_name, "fixed Python setrlimit supervisor is unavailable"
        )
    backend = "darwin-setrlimit" if platform_name == "darwin" else "linux-setrlimit"
    controls = _control_map(
        cpu_time=("process_hard", "RLIMIT_CPU", "inherited per-process CPU limit"),
        cpu_quota=("unsupported", "none", "no tree-wide CPU quota"),
        file_size=("process_hard", "RLIMIT_FSIZE", "inherited per-process file limit"),
        open_files=("process_hard", "RLIMIT_NOFILE", "inherited descriptor limit"),
        memory=(
            "unsupported",
            "none",
            "hard process-tree memory accounting is unavailable",
        ),
        processes=(
            "unsupported",
            "none",
            "hard process-tree task accounting is unavailable",
        ),
        workspace_growth=(
            "post_run",
            "workspace byte accounting",
            "no filesystem quota is active during execution",
        ),
    )
    return VerifierResourceCapabilities(
        platform=platform_name,
        backend=backend,
        supported=True,
        controls=controls,
        reason=(
            "CPU, file-size, and descriptor limits are inherited per process; "
            "memory and process-count limits are not hard tree controls"
        ),
        executable=executable,
    )


def _linux_cgroup_capabilities(
    policy: VerifierResourcePolicy,
    *,
    platform_name: str,
) -> VerifierResourceCapabilities | None:
    if not _linux_cgroup_v2_available():
        return None
    executable = _attest_os_owned_executable(Path(policy.linux_backend))
    if executable is None or not _probe_systemd_user_scope(executable.launch_path):
        return None
    controls = _control_map(
        cpu_time=("process_hard", "systemd LimitCPU", "inherited CPU-time limit"),
        cpu_quota=("kernel_hard", "cgroup v2 cpu.max", "tree-wide CPU-rate quota"),
        file_size=("process_hard", "systemd LimitFSIZE", "inherited file-size limit"),
        open_files=("process_hard", "systemd LimitNOFILE", "inherited descriptor limit"),
        memory=("kernel_hard", "cgroup v2 memory.max", "tree-wide memory ceiling"),
        processes=("kernel_hard", "cgroup v2 pids.max", "tree-wide task ceiling"),
        workspace_growth=(
            "post_run",
            "workspace byte accounting",
            "no filesystem quota is active during execution",
        ),
    )
    return VerifierResourceCapabilities(
        platform=platform_name,
        backend="linux-systemd-cgroup-v2",
        supported=True,
        controls=controls,
        reason="cgroup v2 user-scope properties were verified by a transient probe",
        executable=executable,
    )


def _windows_job_capabilities(platform_name: str) -> VerifierResourceCapabilities:
    if not _windows_job_objects_available():
        return _unsupported_capabilities(
            platform_name, "Windows Job Object APIs are unavailable"
        )
    controls = _control_map(
        cpu_time=("kernel_hard", "Job Object time limit", "job-wide CPU-time limit"),
        cpu_quota=("kernel_hard", "Job Object CPU rate", "job-wide CPU-rate limit"),
        file_size=("unsupported", "none", "Job Objects do not cap file size"),
        open_files=("unsupported", "none", "Job Objects do not cap handles"),
        memory=("kernel_hard", "JobMemoryLimit", "job-wide memory ceiling"),
        processes=("kernel_hard", "ActiveProcessLimit", "job-wide process ceiling"),
        workspace_growth=(
            "post_run",
            "workspace byte accounting",
            "no filesystem quota is active during execution",
        ),
    )
    return VerifierResourceCapabilities(
        platform=platform_name,
        backend="windows-job-object-contract",
        supported=True,
        controls=controls,
        reason=(
            "Job Object controls are available, but command verifiers remain blocked "
            "until an independent filesystem and network sandbox is available"
        ),
    )


def _unsupported_capabilities(
    platform_name: str, reason: str
) -> VerifierResourceCapabilities:
    controls = _control_map(
        **{
            name: ("unsupported", "none", reason)
            for name in RESOURCE_CONTROLS
            if name != "workspace_growth"
        },
        workspace_growth=(
            "post_run",
            "workspace byte accounting",
            "no filesystem quota is active during execution",
        ),
    )
    return VerifierResourceCapabilities(
        platform=platform_name,
        backend="unsupported",
        supported=False,
        controls=controls,
        reason=reason,
    )


def _control_map(
    **values: tuple[str, str, str],
) -> dict[str, ResourceControlCapability]:
    if set(values) != set(RESOURCE_CONTROLS):
        raise VerifierResourceError("Internal resource capability map is incomplete.")
    return {
        name: ResourceControlCapability(
            strength=value[0], mechanism=value[1], reason=value[2]
        )
        for name, value in values.items()
    }


def _attest_os_owned_executable(path: Path) -> ExecutableIdentity | None:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
            raise OSError("backend is not an executable regular file")
        if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise OSError("backend is mutable by an unprivileged identity")
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
        identity = resolve_executable(str(path), env=environment)
        if Path(identity.resolved_path) != path.resolve(strict=True):
            raise OSError("backend resolved outside its fixed OS path")
        return identity
    except (AssistantBridgeRuntimeError, OSError, ValueError):
        return None


def _linux_cgroup_v2_available() -> bool:
    return (
        Path("/sys/fs/cgroup/cgroup.controllers").is_file()
        and Path("/run/systemd/system").is_dir()
        and bool(os.environ.get("XDG_RUNTIME_DIR"))
    )


def _probe_systemd_user_scope(executable: str) -> bool:
    true_path = Path("/usr/bin/true")
    if _attest_os_owned_executable(true_path) is None:
        return False
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"}
    }
    environment.update({"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            [
                executable,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--property=TasksMax=16",
                "--property=MemoryMax=67108864",
                str(true_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _windows_job_objects_available() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        return all(
            getattr(kernel32, name, None) is not None
            for name in (
                "AssignProcessToJobObject",
                "CreateJobObjectW",
                "SetInformationJobObject",
                "TerminateJobObject",
            )
        )
    except (AttributeError, OSError):
        return False


def _plan_binding(
    policy: VerifierResourcePolicy,
    capabilities: VerifierResourceCapabilities,
    command_binding_sha256: str,
    *,
    runnable: bool,
    reason: str,
    resource_argv_sha256: str,
    launcher_chain: LauncherChainIdentity | None,
) -> dict[str, object]:
    # The wrapper is rebuilt inside a disposable workspace at execution time.
    # Its exact cwd and environment therefore differ from the confirmed source
    # workspace.  The already-attested sandbox command binding owns those
    # semantic details; this binding owns only the resource wrapper contract.
    return {
        "schema_version": _SCHEMA,
        "policy": policy.payload(),
        "capabilities": capabilities.binding_payload(),
        "command_binding_sha256": command_binding_sha256,
        "runnable": runnable,
        "reason": reason or None,
        "resource_argv_sha256": resource_argv_sha256 or None,
        "launcher_chain_attested": launcher_chain is not None,
    }


def _deduplicate_companions(values: Sequence[str | Path]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = os.fspath(value)
        key = os.path.normcase(str(Path(raw).resolve(strict=True)))
        if key not in seen:
            seen.add(key)
            result.append(raw)
    return tuple(result)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(item not in _SHA256 for item in value):
        raise VerifierResourceError(f"{label} must be a lowercase SHA-256 digest.")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_POSIX_RESOURCE_SUPERVISOR = r"""
import json
import os
import resource
import sys

limits = json.loads(sys.argv[1])
if set(limits) != {"cpu_time_seconds", "file_size_bytes", "open_files"}:
    raise SystemExit(91)

def apply_limit(kind, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(92)
    _soft, hard = resource.getrlimit(kind)
    infinity = resource.RLIM_INFINITY
    effective = value if hard == infinity else min(value, hard)
    if effective < 1:
        raise SystemExit(93)
    resource.setrlimit(kind, (effective, effective))

apply_limit(resource.RLIMIT_CPU, limits["cpu_time_seconds"])
apply_limit(resource.RLIMIT_FSIZE, limits["file_size_bytes"])
apply_limit(resource.RLIMIT_NOFILE, limits["open_files"])
command = sys.argv[2:]
if not command or not os.path.isabs(command[0]) or "\x00" in command[0]:
    raise SystemExit(94)
os.execve(command[0], command, dict(os.environ))
""".strip()
