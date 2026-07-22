"""Shipping deterministic process-bound runtime supervisor contract fakes.

These fakes model process identity, socket ownership, readiness probes, binding
drift, and cleanup without opening a socket or starting a process.  Production
integration needs only an adapter implementing :class:`RuntimeSupervisorBackend`
and a driver implementing :class:`RuntimeSupervisorDriver`.

``FailClosedSupervisorOracle`` is deliberately a small test oracle, not a
production supervisor.  It makes the safety expectation executable while the
production lifecycle service is wired to the same backend seam.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Callable, Protocol, runtime_checkable


SCENARIO_NAMES = (
    "happy_path",
    "port_occupied",
    "restart_pid_reuse",
    "port_substitution",
    "unexpected_descendant",
    "binding_drift",
    "cleanup",
    "sticky_ambiguity",
)


def sha256_label(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    """Instance identity; a PID alone is intentionally insufficient."""

    pid: int
    start_token: str
    executable_sha256: str

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.pid, self.start_token, self.executable_sha256)

    def payload(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "start_token": self.start_token,
            "executable_sha256": self.executable_sha256,
        }


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def payload(self) -> dict[str, object]:
        return {"host": self.host, "port": self.port}


@dataclass(frozen=True)
class ResolvedRuntime:
    """Content-addressed launch material exposed to the supervisor adapter."""

    binding_sha256: str
    executable_sha256: str
    endpoint: Endpoint
    expected_model_id: str
    argv: tuple[str, ...]
    allowed_descendant_executable_sha256: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessTreeObservation:
    root: ProcessIdentity | None
    descendants: tuple[ProcessIdentity, ...]
    observation_token: str

    @property
    def identities(self) -> tuple[ProcessIdentity, ...]:
        if self.root is None:
            return self.descendants
        return (self.root, *self.descendants)


@dataclass(frozen=True)
class ListenerObservation:
    endpoint: Endpoint
    owner: ProcessIdentity | None
    listener_token: str | None

    @property
    def bound(self) -> bool:
        return self.owner is not None or self.listener_token is not None


@dataclass(frozen=True)
class ProbeObservation:
    healthy: bool
    model_id: str | None
    listener_token: str | None


@dataclass(frozen=True)
class CleanupObservation:
    complete: bool
    survivors: tuple[ProcessIdentity, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupervisorOutcome:
    status: str
    ready: bool
    reason_codes: tuple[str, ...]
    process_identity: ProcessIdentity | None
    process_started: bool
    cleanup_complete: bool | None
    sticky_ambiguity: bool

    def payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ready": self.ready,
            "reason_codes": list(self.reason_codes),
            "process_identity": (
                None
                if self.process_identity is None
                else self.process_identity.payload()
            ),
            "process_started": self.process_started,
            "cleanup_complete": self.cleanup_complete,
            "sticky_ambiguity": self.sticky_ambiguity,
        }


@runtime_checkable
class RuntimeSupervisorBackend(Protocol):
    """Provider-neutral seam required by the contract scenarios."""

    def resolve(self) -> ResolvedRuntime: ...

    def spawn(self, resolved: ResolvedRuntime) -> ProcessIdentity: ...

    def observe_tree(self, root: ProcessIdentity) -> ProcessTreeObservation: ...

    def observe_listener(self, resolved: ResolvedRuntime) -> ListenerObservation: ...

    def probe(self, resolved: ResolvedRuntime) -> ProbeObservation: ...

    def terminate(self, root: ProcessIdentity) -> CleanupObservation: ...

    def binding_digest(self, resolved: ResolvedRuntime) -> str: ...


@runtime_checkable
class RuntimeSupervisorDriver(Protocol):
    """Small integration surface a production supervisor adapter must expose."""

    def start(self, backend: RuntimeSupervisorBackend) -> SupervisorOutcome: ...

    def stop(self, backend: RuntimeSupervisorBackend) -> SupervisorOutcome: ...


Mutation = Callable[["FakeRuntimeSupervisorBackend"], None]


class FakeRuntimeSupervisorBackend:
    """In-memory process, listener, and binding world with TOCTOU hooks."""

    def __init__(self) -> None:
        executable_sha256 = sha256_label("runtime-executable-v1")
        binding_sha256 = sha256_label("resolved-runtime-binding-v1")
        self.resolved = ResolvedRuntime(
            binding_sha256=binding_sha256,
            executable_sha256=executable_sha256,
            endpoint=Endpoint("127.0.0.1", 8123),
            expected_model_id="synthetic-local-model",
            argv=("/fixture/runtime", "--host", "127.0.0.1", "--port", "8123"),
        )
        self.call_log: list[str] = []
        self.call_counts: dict[str, int] = defaultdict(int)
        self.spawn_count = 0
        self.auto_bind_on_spawn = True
        self._birth_sequence = 0
        self._listener_sequence = 0
        self._tree_sequence = 0
        self._binding_sha256 = binding_sha256
        self._managed_root: ProcessIdentity | None = None
        self._managed_alive = False
        self._observed_root: ProcessIdentity | None = None
        self._descendants: list[ProcessIdentity] = []
        self._listener = ListenerObservation(self.resolved.endpoint, None, None)
        self._probe = ProbeObservation(
            healthy=True,
            model_id=self.resolved.expected_model_id,
            listener_token=None,
        )
        self._cleanup_ambiguous = False
        self._failures: dict[str, list[BaseException]] = defaultdict(list)
        self._scheduled: dict[tuple[str, int], list[Mutation]] = defaultdict(list)

    @property
    def managed_root(self) -> ProcessIdentity | None:
        return self._managed_root

    @property
    def listener(self) -> ListenerObservation:
        return self._listener

    def inject_failure(self, stage: str, error: BaseException | None = None) -> None:
        """Raise once at ``stage`` before returning an observation."""

        self._failures[stage].append(
            error or RuntimeError(f"synthetic failure at {stage}")
        )

    def schedule_after(
        self,
        stage: str,
        occurrence: int,
        mutation: Mutation,
    ) -> None:
        """Apply ``mutation`` after one observation has been snapshotted."""

        if occurrence < 1:
            raise ValueError("occurrence must be positive")
        self._scheduled[(stage, occurrence)].append(mutation)

    def resolve(self) -> ResolvedRuntime:
        occurrence = self._begin("resolve")
        return self._finish("resolve", occurrence, self.resolved)

    def spawn(self, resolved: ResolvedRuntime) -> ProcessIdentity:
        occurrence = self._begin("spawn")
        if resolved != self.resolved:
            raise AssertionError("The fake received unrecognized launch material.")
        self.spawn_count += 1
        self._birth_sequence += 1
        root = ProcessIdentity(
            pid=4100,
            start_token=f"birth-{self._birth_sequence}",
            executable_sha256=resolved.executable_sha256,
        )
        self._managed_root = root
        self._managed_alive = True
        self._observed_root = root
        self._descendants.clear()
        if self.auto_bind_on_spawn:
            self.bind_managed_listener()
        return self._finish("spawn", occurrence, root)

    def observe_tree(self, root: ProcessIdentity) -> ProcessTreeObservation:
        occurrence = self._begin("observe_tree")
        self._tree_sequence += 1
        observed_root = (
            self._observed_root
            if self._observed_root is not None
            and self._observed_root.pid == root.pid
            else None
        )
        observation = ProcessTreeObservation(
            root=observed_root,
            descendants=tuple(sorted(self._descendants)),
            observation_token=f"tree-{self._tree_sequence}",
        )
        return self._finish("observe_tree", occurrence, observation)

    def observe_listener(self, resolved: ResolvedRuntime) -> ListenerObservation:
        occurrence = self._begin("observe_listener")
        if resolved.endpoint != self.resolved.endpoint:
            raise AssertionError("The fake received an unrecognized endpoint.")
        observation = self._listener
        return self._finish("observe_listener", occurrence, observation)

    def probe(self, resolved: ResolvedRuntime) -> ProbeObservation:
        occurrence = self._begin("probe")
        if resolved.endpoint != self.resolved.endpoint:
            raise AssertionError("The fake received an unrecognized endpoint.")
        observation = self._probe
        return self._finish("probe", occurrence, observation)

    def terminate(self, root: ProcessIdentity) -> CleanupObservation:
        occurrence = self._begin("terminate")
        if self._cleanup_ambiguous:
            survivors = tuple(
                item
                for item in (self._managed_root, *self._descendants)
                if item is not None
            )
            return self._finish(
                "terminate",
                occurrence,
                CleanupObservation(
                    complete=False,
                    survivors=survivors,
                    reason_codes=("cleanup_unverified",),
                ),
            )

        managed_members = tuple(
            item
            for item in (self._managed_root, *self._descendants)
            if item is not None
        )
        if self._managed_alive and root == self._managed_root:
            self._managed_alive = False
            self._observed_root = None
            self._descendants.clear()
            if self._listener.owner in managed_members:
                self._listener = ListenerObservation(
                    self.resolved.endpoint,
                    None,
                    None,
                )
                self._probe = ProbeObservation(False, None, None)
        return self._finish(
            "terminate",
            occurrence,
            CleanupObservation(complete=True),
        )

    def binding_digest(self, resolved: ResolvedRuntime) -> str:
        occurrence = self._begin("binding_digest")
        if resolved != self.resolved:
            raise AssertionError("The fake received unrecognized launch material.")
        digest = self._binding_sha256
        return self._finish("binding_digest", occurrence, digest)

    def occupy_port(self) -> None:
        self._set_foreign_listener("preexisting-port-owner")

    def bind_managed_listener(self) -> None:
        if self._managed_root is None:
            raise AssertionError("A managed process must exist before binding.")
        self._listener_sequence += 1
        token = f"listener-{self._listener_sequence}"
        self._listener = ListenerObservation(
            self.resolved.endpoint,
            self._managed_root,
            token,
        )
        self._probe = ProbeObservation(
            True,
            self.resolved.expected_model_id,
            token,
        )

    def restart_root_with_reused_pid(self) -> None:
        """Replace the root with a different process instance using the same PID."""

        if self._managed_root is None:
            raise AssertionError("A managed process must exist before restart.")
        self._birth_sequence += 1
        replacement = ProcessIdentity(
            pid=self._managed_root.pid,
            start_token=f"birth-{self._birth_sequence}",
            executable_sha256=self.resolved.executable_sha256,
        )
        self._managed_alive = False
        self._observed_root = replacement
        self._descendants.clear()
        self._set_listener(replacement)

    def substitute_listener(self) -> None:
        """Move the endpoint to a convincing but foreign process."""

        self._set_foreign_listener("substituted-port-owner")

    def add_unexpected_descendant(self) -> None:
        self._birth_sequence += 1
        self._descendants.append(
            ProcessIdentity(
                pid=4200 + self._birth_sequence,
                start_token=f"birth-{self._birth_sequence}",
                executable_sha256=sha256_label("unexpected-descendant"),
            )
        )

    def drift_binding(self) -> None:
        self._binding_sha256 = sha256_label("resolved-runtime-binding-drifted")

    def make_cleanup_ambiguous(self) -> None:
        self._cleanup_ambiguous = True

    def _set_foreign_listener(self, label: str) -> None:
        self._birth_sequence += 1
        foreign = ProcessIdentity(
            pid=9000 + self._birth_sequence,
            start_token=f"foreign-birth-{self._birth_sequence}",
            executable_sha256=sha256_label(label),
        )
        self._set_listener(foreign)

    def _set_listener(self, owner: ProcessIdentity) -> None:
        self._listener_sequence += 1
        token = f"listener-{self._listener_sequence}"
        self._listener = ListenerObservation(self.resolved.endpoint, owner, token)
        self._probe = ProbeObservation(
            True,
            self.resolved.expected_model_id,
            token,
        )

    def _begin(self, stage: str) -> int:
        self.call_log.append(stage)
        self.call_counts[stage] += 1
        if self._failures[stage]:
            raise self._failures[stage].pop(0)
        return self.call_counts[stage]

    def _finish(self, stage: str, occurrence: int, value):
        for mutation in self._scheduled.pop((stage, occurrence), ()):
            mutation(self)
        return value


class FailClosedSupervisorOracle:
    """Minimal executable oracle for the zero-false-ready contract."""

    def __init__(self) -> None:
        self._process: ProcessIdentity | None = None
        self._sticky_ambiguity = False
        self.state = "prepared"

    def start(self, backend: RuntimeSupervisorBackend) -> SupervisorOutcome:
        if self._sticky_ambiguity:
            return self._outcome(
                "unknown_blocking",
                ("sticky_ambiguity",),
                cleanup_complete=False,
            )

        process: ProcessIdentity | None = None
        try:
            resolved = backend.resolve()
            binding_before = backend.binding_digest(resolved)
            listener_before = backend.observe_listener(resolved)
            if listener_before.bound:
                self.state = "revoked"
                return self._outcome("revoked", ("port_occupied",))

            self.state = "starting"
            process = backend.spawn(resolved)
            self._process = process
            tree_before = backend.observe_tree(process)
            listener_started = backend.observe_listener(resolved)
            probe = backend.probe(resolved)
            binding_during = backend.binding_digest(resolved)
            tree_after = backend.observe_tree(process)
            listener_after = backend.observe_listener(resolved)
            binding_after = backend.binding_digest(resolved)
        except Exception:
            return self._fail_after_observation(backend, process, ("backend_failure",))

        reasons: set[str] = set()
        if binding_before != resolved.binding_sha256:
            reasons.add("binding_mismatch")
        if len({binding_before, binding_during, binding_after}) != 1:
            reasons.add("binding_drift")

        if tree_before.root != process or tree_after.root != process:
            reasons.add("process_identity_changed")
        if tuple(item.key for item in tree_before.identities) != tuple(
            item.key for item in tree_after.identities
        ):
            reasons.add("process_tree_changed")
        allowed_descendants = set(resolved.allowed_descendant_executable_sha256)
        if any(
            item.executable_sha256 not in allowed_descendants
            for tree in (tree_before, tree_after)
            for item in tree.descendants
        ):
            reasons.add("unexpected_descendant")

        stable_tree = set(tree_before.identities) & set(tree_after.identities)
        if not listener_started.bound or not listener_after.bound:
            reasons.add("listener_missing")
        if (
            listener_started.endpoint != resolved.endpoint
            or listener_after.endpoint != resolved.endpoint
        ):
            reasons.add("listener_endpoint_changed")
        if (
            listener_started.owner not in stable_tree
            or listener_after.owner not in stable_tree
        ):
            reasons.add("listener_owner_mismatch")
        if (
            listener_started.owner != listener_after.owner
            or listener_started.listener_token != listener_after.listener_token
        ):
            reasons.add("listener_substituted")

        if not probe.healthy:
            reasons.add("probe_unhealthy")
        if probe.model_id != resolved.expected_model_id:
            reasons.add("model_identity_mismatch")
        if probe.listener_token != listener_started.listener_token:
            reasons.add("probe_listener_mismatch")

        if reasons:
            return self._fail_after_observation(
                backend,
                process,
                tuple(sorted(reasons)),
            )

        self.state = "ready"
        return self._outcome("ready", (), ready=True, cleanup_complete=None)

    def stop(self, backend: RuntimeSupervisorBackend) -> SupervisorOutcome:
        if self._sticky_ambiguity:
            return self._outcome(
                "unknown_blocking",
                ("sticky_ambiguity",),
                cleanup_complete=False,
            )
        if self._process is None:
            self.state = "stopped"
            return self._outcome("stopped", (), cleanup_complete=True)

        process = self._process
        self.state = "stopping"
        try:
            cleanup = backend.terminate(process)
        except Exception:
            cleanup = CleanupObservation(
                complete=False,
                survivors=(self._process,),
                reason_codes=("cleanup_unverified",),
            )
        if not cleanup.complete:
            self._sticky_ambiguity = True
            self.state = "unknown_blocking"
            return self._outcome(
                "unknown_blocking",
                tuple(sorted(set(cleanup.reason_codes) | {"cleanup_unverified"})),
                cleanup_complete=False,
            )

        self._process = None
        self.state = "stopped"
        return self._outcome(
            "stopped",
            (),
            cleanup_complete=True,
            process_identity=process,
            process_started=True,
        )

    def _fail_after_observation(
        self,
        backend: RuntimeSupervisorBackend,
        process: ProcessIdentity | None,
        reasons: tuple[str, ...],
    ) -> SupervisorOutcome:
        if process is None:
            self.state = "revoked"
            return self._outcome("revoked", tuple(sorted(set(reasons))))
        try:
            cleanup = backend.terminate(process)
        except Exception:
            cleanup = CleanupObservation(
                complete=False,
                survivors=(process,),
                reason_codes=("cleanup_unverified",),
            )
        if not cleanup.complete:
            self._sticky_ambiguity = True
            self.state = "unknown_blocking"
            combined = set(reasons) | set(cleanup.reason_codes) | {
                "cleanup_unverified"
            }
            return self._outcome(
                "unknown_blocking",
                tuple(sorted(combined)),
                cleanup_complete=False,
            )
        self._process = None
        self.state = "revoked"
        return self._outcome(
            "revoked",
            tuple(sorted(set(reasons))),
            cleanup_complete=True,
            process_identity=process,
            process_started=True,
        )

    def _outcome(
        self,
        status: str,
        reason_codes: tuple[str, ...],
        *,
        ready: bool = False,
        cleanup_complete: bool | None = None,
        process_identity: ProcessIdentity | None = None,
        process_started: bool | None = None,
    ) -> SupervisorOutcome:
        identity = self._process if process_identity is None else process_identity
        return SupervisorOutcome(
            status=status,
            ready=ready,
            reason_codes=reason_codes,
            process_identity=identity,
            process_started=(identity is not None if process_started is None else process_started),
            cleanup_complete=cleanup_complete,
            sticky_ambiguity=self._sticky_ambiguity,
        )


@dataclass(frozen=True)
class ScenarioExecution:
    name: str
    outcomes: tuple[SupervisorOutcome, ...]
    call_log: tuple[str, ...]
    spawn_count: int
    false_ready: bool

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "outcomes": [item.payload() for item in self.outcomes],
            "call_log": list(self.call_log),
            "spawn_count": self.spawn_count,
            "false_ready": self.false_ready,
        }


def configure_scenario(backend: FakeRuntimeSupervisorBackend, name: str) -> None:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unsupported runtime supervisor scenario: {name}")
    if name == "port_occupied":
        backend.occupy_port()
    elif name == "restart_pid_reuse":
        backend.schedule_after(
            "observe_tree",
            1,
            lambda world: world.restart_root_with_reused_pid(),
        )
    elif name == "port_substitution":
        # Occurrence one is the pre-launch occupancy check.  Mutate after the
        # first post-launch listener ownership snapshot.
        backend.schedule_after(
            "observe_listener",
            2,
            lambda world: world.substitute_listener(),
        )
    elif name == "unexpected_descendant":
        backend.schedule_after(
            "observe_tree",
            1,
            lambda world: world.add_unexpected_descendant(),
        )
    elif name == "binding_drift":
        backend.schedule_after(
            "binding_digest",
            1,
            lambda world: world.drift_binding(),
        )


def run_scenario(
    driver: RuntimeSupervisorDriver,
    name: str,
    *,
    backend: FakeRuntimeSupervisorBackend | None = None,
) -> ScenarioExecution:
    """Exercise one complete scenario through an injected supervisor driver."""

    world = backend or FakeRuntimeSupervisorBackend()
    configure_scenario(world, name)
    outcomes: list[SupervisorOutcome] = [driver.start(world)]

    if name == "cleanup":
        outcomes.append(driver.stop(world))
    elif name == "sticky_ambiguity":
        world.make_cleanup_ambiguous()
        outcomes.append(driver.stop(world))
        outcomes.append(driver.start(world))

    if name in {"happy_path", "cleanup", "sticky_ambiguity"}:
        false_ready = name == "sticky_ambiguity" and outcomes[-1].ready
    else:
        false_ready = any(item.ready for item in outcomes)
    return ScenarioExecution(
        name=name,
        outcomes=tuple(outcomes),
        call_log=tuple(world.call_log),
        spawn_count=world.spawn_count,
        false_ready=false_ready,
    )


__all__ = [
    "CleanupObservation",
    "Endpoint",
    "FailClosedSupervisorOracle",
    "FakeRuntimeSupervisorBackend",
    "ListenerObservation",
    "ProbeObservation",
    "ProcessIdentity",
    "ProcessTreeObservation",
    "ResolvedRuntime",
    "RuntimeSupervisorBackend",
    "RuntimeSupervisorDriver",
    "SCENARIO_NAMES",
    "ScenarioExecution",
    "SupervisorOutcome",
    "configure_scenario",
    "run_scenario",
    "sha256_label",
]
