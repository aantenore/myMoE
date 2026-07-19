"""Claim-bound Assistant Bridge adapter for verified paired execution."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from .assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    AssistantTaskEnvelope,
    BridgeRunResult,
    VerificationEvidence,
    WorkspaceAttestation,
)
from .assistant_bridge_attestation import AttestationVerificationError
from .assistant_bridge_cas import ContentAddressedStore
from .assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from .assistant_bridge_two_phase_config import TwoPhaseTrustConfig
from .assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    CandidateBinding,
    IndependentEvaluationAttestation,
)
from .assistant_bridge_two_phase_ports import EvidenceStore
from .assistant_bridge_workspace import (
    WorkspaceChange,
    WorkspaceFile,
    WorkspaceSecurityError,
    WorkspaceSnapshot,
    snapshot_workspace,
)
from .paired_execution_contracts import PairedOutcomeBinding, PairedRunSlot
from .paired_evidence import (
    PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE,
    PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE,
    PAIRED_CHANGESET_MEDIA_TYPE,
    PAIRED_MANIFEST_MEDIA_TYPE,
    PAIRED_RESULT_METADATA_MEDIA_TYPE,
    PAIRED_SIGNALS_MEDIA_TYPE,
    PairedAttestationReceipt,
    paired_attestation_challenge_sha256,
    paired_artifact_max_bytes,
)
from .route_signals import TaskSignals
from .verified_routing_contracts import (
    canonical_json,
    require_sha256,
    sha256_json,
)


class PairedAttestationProducer(Protocol):
    """Untrusted producer port returning raw signed DSSE evidence only."""

    @property
    def configuration_sha256(self) -> str: ...

    @property
    def state_paths(self) -> tuple[Path, ...]: ...

    def attest(
        self,
        binding: CandidateBinding,
        workspace: Path,
        deadline: float,
    ) -> Sequence[bytes]: ...


@dataclass(frozen=True)
class PairedArmPlan:
    """A ticketed, content-free bridge plan for one paired execution slot."""

    slot: PairedRunSlot
    permit: PairedOutcomeBinding = field(repr=False)
    signals: TaskSignals = field(repr=False)
    operation_sha256: str
    confirmation: str = field(repr=False)
    bridge_plan: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.slot, PairedRunSlot):
            raise AssistantBridgeError("Paired arm plan requires a run slot.")
        if not isinstance(self.permit, PairedOutcomeBinding):
            raise AssistantBridgeError("Paired arm plan requires a claim permit.")
        if not isinstance(self.signals, TaskSignals):
            raise AssistantBridgeError("Paired arm plan requires frozen task signals.")
        if (
            self.signals.request_fingerprint != self.permit.task_fingerprint
            or self.signals.signals_sha256 != self.permit.signals_sha256
        ):
            raise AssistantBridgeError(
                "Paired arm signals do not match the durable claim."
            )
        if (
            self.permit.slot != self.slot.slot
            or self.permit.arm != self.slot.arm
            or self.permit.ordinal != self.slot.ordinal
            or self.permit.route != self.slot.route
        ):
            raise AssistantBridgeError(
                "Paired arm permit does not match its execution slot."
            )
        if self.operation_sha256 != paired_arm_operation_sha256(self.permit):
            raise AssistantBridgeError(
                "Paired arm operation is not derived from its durable claim."
            )
        if not isinstance(self.confirmation, str) or not self.confirmation:
            raise AssistantBridgeError(
                "Paired arm plan requires a one-shot confirmation ticket."
            )
        if not isinstance(self.bridge_plan, Mapping):
            raise AssistantBridgeError("Paired bridge plan must be an object.")
        object.__setattr__(
            self,
            "bridge_plan",
            MappingProxyType(dict(self.bridge_plan)),
        )

    def metadata_payload(self) -> dict[str, object]:
        """Return only public planning metadata; the ticket is deliberately absent."""

        allowed = {
            "mode",
            "execute",
            "guarded_baseline_route",
            "evaluation_route",
            "source_snapshot_sha256",
            "route_receipt",
            "generator_config_sha256",
            "paired_executor_config_sha256",
            "lifecycle_config_sha256",
            "operation_sha256",
            "authority",
            "privacy",
        }
        bridge_plan = _sanitize_metadata(
            {
                key: value
                for key, value in self.bridge_plan.items()
                if key in allowed
            }
        )
        return {
            "slot": self.slot.payload(),
            "binding_sha256": self.permit.binding_sha256,
            "signals_sha256": self.signals.signals_sha256,
            "operation_sha256": self.operation_sha256,
            "bridge_plan": bridge_plan,
            "confirmation_persisted": False,
        }


class PairedArmExecutor(Protocol):
    """Least-authority port used by the paired execution orchestrator."""

    @property
    def bridge_config_sha256(self) -> str: ...

    @property
    def configuration_sha256(self) -> str: ...

    @property
    def execution_harness_sha256(self) -> str: ...

    @property
    def state_paths(self) -> tuple[Path, ...]: ...

    def preflight(self, task: AssistantTaskEnvelope) -> None: ...

    def snapshot_source(self, workspace: str | Path) -> WorkspaceSnapshot: ...

    def plan_arm(
        self,
        task: AssistantTaskEnvelope,
        *,
        source_workspace: str | Path,
        source_snapshot: WorkspaceSnapshot,
        signals: TaskSignals,
        lifecycle_config_sha256: str,
        baseline_route: str,
        slot: PairedRunSlot,
        permit: PairedOutcomeBinding,
    ) -> PairedArmPlan: ...

    def run_arm(
        self,
        task: AssistantTaskEnvelope,
        *,
        source_workspace: str | Path,
        source_snapshot: WorkspaceSnapshot,
        signals: TaskSignals,
        lifecycle_config_sha256: str,
        baseline_route: str,
        plan: PairedArmPlan,
    ) -> BridgeRunResult: ...

    def assert_source_unchanged(self, snapshot: WorkspaceSnapshot) -> None: ...

    def verify_outcome(self, record: object, pricing: object) -> object: ...


class AssistantBridgePairedArmExecutor:
    """Provider-agnostic adapter over Assistant Bridge candidate generation.

    Provider selection and execution remain owned by the runner's immutable adapter
    registry.  This wrapper only narrows authority to an exact route in a disposable
    workspace and never exposes source-apply operations.
    """

    def __init__(
        self,
        runner: AssistantBridgeRunner,
        *,
        local_provider_override: str | None = None,
        attestation_producer: PairedAttestationProducer | None = None,
        trust_config: TwoPhaseTrustConfig | None = None,
        evidence_store: EvidenceStore | None = None,
        clock: Callable[[], float] = time.time,
        attestation_ttl_seconds: float = 300.0,
        attestation_timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(runner, AssistantBridgeRunner):
            raise AssistantBridgeError(
                "Paired execution requires an Assistant Bridge runner."
            )
        self._runner = runner
        self._generator = runner.candidate_generator()
        self._local_provider_override = local_provider_override
        self._attestation_producer = attestation_producer
        self._trust_config = trust_config
        self._evidence_store = evidence_store
        if not callable(clock):
            raise AssistantBridgeError("Paired attestation clock must be callable.")
        self._clock = clock
        _clock_configuration_sha256(clock)
        self._attestation_ttl_seconds = _bounded_seconds(
            attestation_ttl_seconds,
            "attestation TTL",
            maximum=7 * 24 * 60 * 60,
        )
        self._attestation_timeout_seconds = _bounded_seconds(
            attestation_timeout_seconds,
            "attestation timeout",
            maximum=3600.0,
        )
        if self._attestation_timeout_seconds > self._attestation_ttl_seconds:
            raise AssistantBridgeError(
                "Paired attestation timeout cannot exceed its TTL."
            )

    @property
    def bridge_config_sha256(self) -> str:
        return self._runner.config.source_sha256

    @property
    def configuration_sha256(self) -> str:
        producer_sha256 = (
            None
            if self._attestation_producer is None
            else _configuration_sha256(
                self._attestation_producer,
                "paired attestation producer",
            )
        )
        evidence_store_sha256 = (
            None
            if self._evidence_store is None
            else _configuration_sha256(
                self._evidence_store,
                "paired evidence store",
            )
        )
        trust_descriptor = (
            None
            if self._trust_config is None
            else self._trust_config.descriptor()
        )
        external_verifiers = [
            {
                "id": verifier_id,
                "verifier": spec.verifier,
                "spec_sha256": spec.spec_sha256,
            }
            for verifier_id, spec in sorted(
                self._runner.config.external_verifiers.items()
            )
        ]
        producer_state_paths = (
            ()
            if self._attestation_producer is None
            else _producer_state_paths(self._attestation_producer)
        )
        evidence_store_state_paths = (
            ()
            if self._evidence_store is None
            else _evidence_store_state_paths(self._evidence_store)
        )
        bridge_state_paths = self._bridge_state_paths()
        return sha256_json(
            {
                "contract": "mymoe-paired-attestation-executor/v1",
                "generator_config_sha256": self._generator.configuration_sha256,
                "producer_config_sha256": producer_sha256,
                "trust": trust_descriptor,
                "external_verifiers": external_verifiers,
                "attestation_ttl_seconds": self._attestation_ttl_seconds,
                "attestation_timeout_seconds": self._attestation_timeout_seconds,
                "clock_config_sha256": _clock_configuration_sha256(self._clock),
                "evidence_store_config_sha256": evidence_store_sha256,
                "producer_state_paths_sha256": [
                    sha256_bytes(str(path).encode("utf-8"))
                    for path in producer_state_paths
                ],
                "evidence_store_state_paths_sha256": [
                    sha256_bytes(str(path).encode("utf-8"))
                    for path in evidence_store_state_paths
                ],
                "bridge_state_paths_sha256": [
                    sha256_bytes(str(path).encode("utf-8"))
                    for path in bridge_state_paths
                ],
                "local_provider_override": self._local_provider_override,
            }
        )

    @property
    def execution_harness_sha256(self) -> str:
        """Freeze semantic execution behavior without machine-local state paths."""

        from .paired_execution import paired_runner_source_sha256

        producer_sha256 = (
            None
            if self._attestation_producer is None
            else _semantic_configuration_sha256(
                self._attestation_producer,
                "paired attestation producer",
            )
        )
        evidence_store_sha256 = (
            None
            if self._evidence_store is None
            else _semantic_configuration_sha256(
                self._evidence_store,
                "paired evidence store",
            )
        )
        trust_policy_sha256 = (
            None
            if self._trust_config is None
            else self._trust_config.policy.policy_sha256
        )
        external_verifiers = [
            {
                "id": verifier_id,
                "verifier": spec.verifier,
                "spec_sha256": spec.spec_sha256,
            }
            for verifier_id, spec in sorted(
                self._runner.config.external_verifiers.items()
            )
        ]
        return sha256_json(
            {
                "contract": "mymoe-paired-execution-harness/v1",
                "runner_source_sha256": paired_runner_source_sha256(),
                "bridge_config_sha256": self.bridge_config_sha256,
                "generator_config_sha256": self._generator.configuration_sha256,
                "producer_semantic_config_sha256": producer_sha256,
                "trust_policy_sha256": trust_policy_sha256,
                "external_verifiers": external_verifiers,
                "attestation_ttl_seconds": self._attestation_ttl_seconds,
                "attestation_timeout_seconds": self._attestation_timeout_seconds,
                "clock_config_sha256": _clock_configuration_sha256(self._clock),
                "evidence_store_semantic_config_sha256": evidence_store_sha256,
                "local_provider_override": self._local_provider_override,
            }
        )

    @property
    def state_paths(self) -> tuple[Path, ...]:
        producer_paths = (
            ()
            if self._attestation_producer is None
            else _producer_state_paths(self._attestation_producer)
        )
        evidence_paths = (
            ()
            if self._evidence_store is None
            else _evidence_store_state_paths(self._evidence_store)
        )
        return (*producer_paths, *evidence_paths, *self._bridge_state_paths())

    def preflight(self, task: AssistantTaskEnvelope) -> None:
        """Fail before claim/provider authority if signed verification is absent."""

        if not isinstance(task, AssistantTaskEnvelope):
            raise TypeError("task must be an AssistantTaskEnvelope.")
        producer = self._attestation_producer
        trust = self._trust_config
        store = self._evidence_store
        if (
            producer is None
            or trust is None
            or store is None
            or not callable(getattr(producer, "attest", None))
            or not callable(getattr(store, "put_bytes", None))
            or not callable(getattr(store, "get_bytes", None))
        ):
            raise AssistantBridgeError(
                "signed_verifier_required: paired execution requires a producer, "
                "trust policy, and immutable evidence store."
            )
        _configuration_sha256(producer, "paired attestation producer")
        _configuration_sha256(store, "paired evidence store")
        _clock_configuration_sha256(self._clock)
        _producer_state_paths(producer)
        if not _evidence_store_state_paths(store):
            raise AssistantBridgeError(
                "signed_verifier_configuration_invalid: evidence store state "
                "path is unavailable."
            )
        try:
            trust.build_trust_store()
        except (AttestationVerificationError, ValueError) as exc:
            raise AssistantBridgeError(
                "signed_verifier_configuration_invalid: trust store is incoherent."
            ) from exc
        external = self._runner.config.external_verifiers
        policy_ids = {
            requirement.verifier_id for requirement in trust.policy.verifiers
        }
        for requirement in trust.policy.verifiers:
            configured = external.get(requirement.verifier_id)
            if (
                configured is None
                or configured.spec_sha256 != requirement.spec_sha256
            ):
                raise AssistantBridgeError(
                    "signed_verifier_required: trust policy verifier is absent "
                    "from Assistant Bridge external verifier configuration."
                )
        required = set(task.required_verifier_ids)
        if not required.issubset(policy_ids) or not required.issubset(external):
            raise AssistantBridgeError(
                "signed_verifier_required: task-required verifiers are absent "
                "from the paired trust policy."
            )
        self._now()

    def _bridge_state_paths(self) -> tuple[Path, ...]:
        raw = getattr(self._runner.state_ledger, "path", None)
        if not isinstance(raw, (str, Path)):
            raise AssistantBridgeError(
                "Paired bridge ledger state path is unavailable."
            )
        declared = Path(raw).expanduser()
        if declared.is_symlink():
            raise AssistantBridgeError(
                "Paired bridge ledger state path cannot be a symbolic link."
            )
        try:
            ledger = declared.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise AssistantBridgeError(
                "Paired bridge ledger state path is unavailable."
            ) from exc
        lock = ledger.with_suffix(ledger.suffix + ".lock")
        if lock.is_symlink():
            raise AssistantBridgeError(
                "Paired bridge ledger lock path cannot be a symbolic link."
            )
        return ledger, lock

    def _assert_state_isolated_from_source(
        self,
        workspace: str | Path,
    ) -> Path:
        try:
            source_root = Path(workspace).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AssistantBridgeError(
                "Paired source workspace is unavailable."
            ) from exc
        for state_path in self.state_paths:
            if _paths_overlap(source_root, state_path):
                raise AssistantBridgeError(
                    "Paired verifier state must be isolated from the source "
                    "workspace."
                )
        return source_root

    def snapshot_source(self, workspace: str | Path) -> WorkspaceSnapshot:
        try:
            snapshot = snapshot_workspace(
                workspace,
                self._runner.config.workspace.scope,
            )
            self._assert_state_isolated_from_source(snapshot.root)
            return snapshot
        except WorkspaceSecurityError as exc:
            raise AssistantBridgeError(str(exc)) from None

    def plan_arm(
        self,
        task: AssistantTaskEnvelope,
        *,
        source_workspace: str | Path,
        source_snapshot: WorkspaceSnapshot,
        signals: TaskSignals,
        lifecycle_config_sha256: str,
        baseline_route: str,
        slot: PairedRunSlot,
        permit: PairedOutcomeBinding,
    ) -> PairedArmPlan:
        self.preflight(task)
        self._assert_state_isolated_from_source(source_workspace)
        if not isinstance(source_snapshot, WorkspaceSnapshot):
            raise AssistantBridgeError(
                "Paired execution requires a frozen source snapshot."
            )
        if not isinstance(signals, TaskSignals):
            raise AssistantBridgeError(
                "Paired execution requires frozen task signals."
            )
        if not isinstance(permit, PairedOutcomeBinding):
            raise AssistantBridgeError(
                "Paired execution requires a durable claim permit."
            )
        if (
            permit.slot != slot.slot
            or permit.arm != slot.arm
            or permit.ordinal != slot.ordinal
            or permit.route != slot.route
        ):
            raise AssistantBridgeError(
                "Paired execution permit does not match the requested slot."
            )
        operation_sha256 = paired_arm_operation_sha256(permit)
        executor_config_sha256 = self.configuration_sha256
        raw = self._generator._plan_paired_evidence_arm(
            task,
            workspace=source_workspace,
            expected_source_fingerprint=source_snapshot.fingerprint,
            expected_config_sha256=lifecycle_config_sha256,
            expected_baseline_route=baseline_route,
            route=slot.route,
            operation_sha256=operation_sha256,
            local_provider_override=self._local_provider_override,
            external_evidence=(),
            include_diff=False,
        )
        if self.configuration_sha256 != executor_config_sha256:
            raise AssistantBridgeError(
                "Paired attestation configuration changed during planning."
            )
        raw = dict(raw)
        raw["paired_executor_config_sha256"] = executor_config_sha256
        confirmation = raw.get("confirmation_id")
        if not isinstance(confirmation, str) or not confirmation:
            raise AssistantBridgeError(
                "Paired bridge planning did not issue an execution ticket."
            )
        return PairedArmPlan(
            slot=slot,
            permit=permit,
            signals=signals,
            operation_sha256=operation_sha256,
            confirmation=confirmation,
            bridge_plan=raw,
        )

    def run_arm(
        self,
        task: AssistantTaskEnvelope,
        *,
        source_workspace: str | Path,
        source_snapshot: WorkspaceSnapshot,
        signals: TaskSignals,
        lifecycle_config_sha256: str,
        baseline_route: str,
        plan: PairedArmPlan,
    ) -> BridgeRunResult:
        self.preflight(task)
        self._assert_state_isolated_from_source(source_workspace)
        if not isinstance(plan, PairedArmPlan):
            raise AssistantBridgeError("Paired execution plan has the wrong type.")
        if signals != plan.signals:
            raise AssistantBridgeError(
                "Paired execution signals changed after planning."
            )
        if plan.operation_sha256 != paired_arm_operation_sha256(plan.permit):
            raise AssistantBridgeError(
                "Paired execution plan is not bound to its durable claim."
            )
        planned_config_sha256 = plan.bridge_plan.get(
            "paired_executor_config_sha256"
        )
        if (
            not isinstance(planned_config_sha256, str)
            or planned_config_sha256 != self.configuration_sha256
        ):
            raise AssistantBridgeError(
                "Paired attestation configuration changed after planning."
            )
        receipt_descriptors: list[ArtifactDescriptor] = []
        receipt_created_at: list[float] = []

        def finalize_candidate(
            result: BridgeRunResult,
            verifier_workspace: Path,
            workspace_attestation: WorkspaceAttestation,
            candidate_snapshot: WorkspaceSnapshot,
            candidate_files: Sequence[WorkspaceFile],
            changes: Sequence[WorkspaceChange],
        ) -> tuple[VerificationEvidence, ...]:
            external, receipt_descriptor, observed_at = self._attest_candidate(
                task=task,
                result=result,
                verifier_workspace=verifier_workspace,
                workspace_attestation=workspace_attestation,
                candidate_snapshot=candidate_snapshot,
                candidate_files=candidate_files,
                changes=changes,
                source_snapshot=source_snapshot,
                lifecycle_config_sha256=lifecycle_config_sha256,
                plan=plan,
                expected_executor_config_sha256=planned_config_sha256,
            )
            if receipt_descriptors:
                raise AssistantBridgeError(
                    "Paired candidate finalizer produced more than one receipt."
                )
            receipt_descriptors.append(receipt_descriptor)
            receipt_created_at.append(observed_at)
            return (*result.verification, *external)

        result = self._generator._run_paired_evidence_arm(
            task,
            source_workspace=source_workspace,
            expected_source_fingerprint=source_snapshot.fingerprint,
            expected_config_sha256=lifecycle_config_sha256,
            expected_baseline_route=baseline_route,
            route=plan.slot.route,
            operation_sha256=plan.operation_sha256,
            confirmation=plan.confirmation,
            local_provider_override=self._local_provider_override,
            external_evidence=(),
            include_diff=False,
            candidate_finalizer=finalize_candidate,
        )
        if len(receipt_descriptors) != 1 or len(receipt_created_at) != 1:
            raise AssistantBridgeError(
                "Paired execution did not produce one attestation receipt."
            )
        result = replace(
            result,
            paired_evidence=receipt_descriptors[0].payload(),
            paired_evidence_created_at=receipt_created_at[0],
        )
        self.assert_source_unchanged(source_snapshot)
        return result

    def _attest_candidate(
        self,
        *,
        task: AssistantTaskEnvelope,
        result: BridgeRunResult,
        verifier_workspace: Path,
        workspace_attestation: WorkspaceAttestation,
        candidate_snapshot: WorkspaceSnapshot,
        candidate_files: Sequence[WorkspaceFile],
        changes: Sequence[WorkspaceChange],
        source_snapshot: WorkspaceSnapshot,
        lifecycle_config_sha256: str,
        plan: PairedArmPlan,
        expected_executor_config_sha256: str,
    ) -> tuple[tuple[VerificationEvidence, ...], ArtifactDescriptor, float]:
        """Verify signed pass/fail evidence and persist its complete CAS receipt."""

        self.preflight(task)
        if self.configuration_sha256 != expected_executor_config_sha256:
            raise AssistantBridgeError(
                "Paired attestation configuration changed before verification."
            )
        producer = self._attestation_producer
        trust_config = self._trust_config
        evidence_store = self._evidence_store
        assert producer is not None
        assert trust_config is not None
        assert evidence_store is not None

        issued_at = self._now()
        result_metadata = _put_verified_json(
            evidence_store,
            result.metadata_payload(),
            media_type=PAIRED_RESULT_METADATA_MEDIA_TYPE,
        )
        signals = _put_verified_json(
            evidence_store,
            plan.signals.payload(),
            media_type=PAIRED_SIGNALS_MEDIA_TYPE,
        )
        common = {
            "pairedOutcomeBindingSha256": plan.permit.binding_sha256,
            "operationSha256": plan.operation_sha256,
            "executorConfigSha256": expected_executor_config_sha256,
            "executionHarnessSha256": plan.permit.execution_harness_sha256,
            "lifecycleConfigSha256": lifecycle_config_sha256,
            "runnerSourceSha256": plan.permit.runner_source_sha256,
            "signalsSha256": plan.signals.signals_sha256,
            "resultMetadata": result_metadata.payload(),
            "signals": signals.payload(),
            "sourceFingerprint": source_snapshot.fingerprint,
            "observedAt": issued_at,
        }
        manifest = _put_verified_json(
            evidence_store,
            {
                "schemaVersion": "1.0",
                "contract": "PairedCandidateManifest",
                **common,
                "candidateWorkspaceFingerprint": candidate_snapshot.fingerprint,
                "candidateSnapshot": candidate_snapshot.payload(),
                "workspaceAttestationFingerprint": workspace_attestation.fingerprint,
                "files": [item.payload() for item in sorted(candidate_files)],
            },
            media_type=PAIRED_MANIFEST_MEDIA_TYPE,
        )
        changeset = _put_verified_json(
            evidence_store,
            {
                "schemaVersion": "1.0",
                "contract": "PairedCandidateChangeset",
                **common,
                "candidateWorkspaceFingerprint": candidate_snapshot.fingerprint,
                "changes": [
                    _change_payload(item)
                    for item in sorted(changes, key=lambda item: item.path)
                ],
            },
            media_type=PAIRED_CHANGESET_MEDIA_TYPE,
        )
        binding = CandidateBinding(
            workflow_id=f"paired-{plan.operation_sha256}",
            stage_idempotency_sha256=plan.operation_sha256,
            task_fingerprint=task.task_fingerprint,
            config_sha256=lifecycle_config_sha256,
            source_fingerprint=source_snapshot.fingerprint,
            challenge_sha256=paired_attestation_challenge_sha256(
                operation_sha256=plan.operation_sha256,
                paired_outcome_binding_sha256=plan.permit.binding_sha256,
                executor_config_sha256=expected_executor_config_sha256,
                lifecycle_config_sha256=lifecycle_config_sha256,
                signals_sha256=plan.signals.signals_sha256,
                result_metadata_sha256=result_metadata.sha256,
                run_instance_nonce=plan.permit.run_instance_nonce,
            ),
            manifest=manifest,
            changeset=changeset,
            verification_policy=trust_config.policy,
            created_at=issued_at,
            expires_at=issued_at + self._attestation_ttl_seconds,
        )
        deadline = issued_at + self._attestation_timeout_seconds
        raw_envelopes = producer.attest(binding, verifier_workspace, deadline)
        completed_at = self._now()
        if completed_at < issued_at or completed_at > deadline:
            raise AssistantBridgeError(
                "Paired attestation producer exceeded its trusted deadline."
            )
        if (
            not isinstance(raw_envelopes, SequenceABC)
            or isinstance(raw_envelopes, (bytes, bytearray, str))
        ):
            raise AssistantBridgeError(
                "Paired attestation producer violated its envelope protocol."
            )
        envelopes = tuple(raw_envelopes)
        if len(envelopes) > len(trust_config.policy.verifiers):
            raise AssistantBridgeError(
                "Paired attestation producer returned excess verifier evidence."
            )
        trust_store = trust_config.build_trust_store()
        verified: dict[str, IndependentEvaluationAttestation] = {}
        for envelope in envelopes:
            if not isinstance(envelope, bytes):
                raise AssistantBridgeError(
                    "Paired attestation producer returned non-byte evidence."
                )
            attestation = trust_store.verify_evaluation(
                binding,
                envelope,
                now=completed_at,
            )
            if envelope != attestation.envelope_bytes:
                raise AssistantBridgeError(
                    "Paired attestation envelope is not canonical."
                )
            if attestation.verifier_id in verified:
                raise AssistantBridgeError(
                    "Paired attestation repeats a verifier identity."
                )
            verified[attestation.verifier_id] = attestation
        if len(verified) < trust_config.policy.quorum:
            raise AssistantBridgeError(
                "Paired attestation did not satisfy the trusted verifier quorum."
            )
        missing_required = sorted(
            set(task.required_verifier_ids).difference(verified)
        )
        if missing_required:
            raise AssistantBridgeError(
                "Paired attestation omitted task-required verifier evidence: "
                + ", ".join(missing_required)
                + "."
            )

        external: list[VerificationEvidence] = []
        for verifier_id in sorted(verified):
            attestation = verified[verifier_id]
            if len(attestation.envelope_bytes) > paired_artifact_max_bytes(
                PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE
            ):
                raise AssistantBridgeError(
                    "Paired attestation envelope exceeds its qualification bound."
                )
            descriptor = evidence_store.put_bytes(
                attestation.envelope_bytes,
                media_type=PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE,
            )
            if (
                not isinstance(descriptor, ArtifactDescriptor)
                or descriptor.sha256 != attestation.evidence_sha256
                or descriptor.size_bytes != len(attestation.envelope_bytes)
                or descriptor.media_type != PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE
                or evidence_store.get_bytes(descriptor)
                != attestation.envelope_bytes
            ):
                raise AssistantBridgeError(
                    "Paired attestation evidence store failed integrity validation."
                )
            spec = self._runner.config.external_verifiers[verifier_id]
            external.append(
                VerificationEvidence(
                    id=verifier_id,
                    verifier=spec.verifier,
                    kind="external",
                    passed=attestation.passed,
                    code=(
                        "signed-attestation-passed"
                        if attestation.passed
                        else "signed-attestation-failed"
                    ),
                    artifact_sha256=attestation.evidence_sha256,
                    observed_chars=len(attestation.envelope_bytes),
                    evidence_ref=(
                        f"cas://sha256/{attestation.evidence_sha256}"
                    ),
                    task_fingerprint=task.task_fingerprint,
                    workspace_fingerprint=workspace_attestation.fingerprint,
                    verifier_spec_sha256=spec.spec_sha256,
                )
            )
        validated = self._runner._validate_external(
            external,
            task,
            workspace_attestation,
        )
        if self.configuration_sha256 != expected_executor_config_sha256:
            raise AssistantBridgeError(
                "Paired attestation configuration changed during verification."
            )
        receipt = PairedAttestationReceipt(
            binding=binding,
            paired_outcome_binding=plan.permit,
            result_metadata=result_metadata,
            signals=signals,
            manifest=manifest,
            changeset=changeset,
            envelopes=tuple(
                sorted(
                    (
                        ArtifactDescriptor(
                            media_type=PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE,
                            sha256=item.evidence_sha256,
                            size_bytes=len(item.envelope_bytes),
                        )
                        for item in verified.values()
                    ),
                    key=lambda item: item.sha256,
                )
            ),
            workspace_fingerprint=workspace_attestation.fingerprint,
        )
        receipt_descriptor = _put_verified_json(
            evidence_store,
            receipt.payload(),
            media_type=PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE,
        )
        return validated, receipt_descriptor, binding.created_at

    def _now(self) -> float:
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise AssistantBridgeError(
                "Paired attestation clock returned an invalid timestamp."
            )
        return float(value)

    def assert_source_unchanged(self, snapshot: WorkspaceSnapshot) -> None:
        current = self.snapshot_source(snapshot.root)
        if (
            current.fingerprint != snapshot.fingerprint
            or current.files != snapshot.files
            or current.head_sha != snapshot.head_sha
            or current.index_sha256 != snapshot.index_sha256
        ):
            raise AssistantBridgeError(
                "Source workspace changed during paired evidence execution."
            )

    def verify_outcome(self, record: object, pricing: object) -> object:
        """Reconstruct an outcome from this executor's own signed CAS receipt."""

        from .paired_evidence import PairedAttestationVerifier
        from .paired_execution_pricing import PricingContract
        from .route_outcomes import VerifiedOutcomeRecord

        if not isinstance(record, VerifiedOutcomeRecord):
            raise AssistantBridgeError("Paired outcome has the wrong type.")
        if not isinstance(pricing, PricingContract):
            raise AssistantBridgeError("Paired pricing has the wrong type.")
        if not isinstance(self._evidence_store, ContentAddressedStore):
            raise AssistantBridgeError(
                "Paired outcome reconstruction requires the concrete immutable CAS."
            )
        if self._trust_config is None:
            raise AssistantBridgeError(
                "Paired outcome reconstruction requires public trust configuration."
            )
        return PairedAttestationVerifier(
            trust_config=self._trust_config,
            evidence_store=self._evidence_store,
            bridge_config=self._runner.config,
        ).verify_record(record, pricing=pricing)


def paired_arm_operation_sha256(permit: PairedOutcomeBinding) -> str:
    """Derive the only accepted bridge operation from a durable claim."""

    if not isinstance(permit, PairedOutcomeBinding):
        raise AssistantBridgeError("Paired arm operation requires a claim permit.")
    return sha256_json(
        {
            "contract": "mymoe-paired-arm-operation/v1",
            "run_id": permit.run_id,
            "claim_sha256": permit.claim_sha256,
        }
    )


def _configuration_sha256(value: object, label: str) -> str:
    configured = getattr(value, "configuration_sha256", None)
    if callable(configured):
        configured = configured()
    if not isinstance(configured, str):
        raise AssistantBridgeError(f"{label} configuration digest is unavailable.")
    try:
        return require_sha256(configured, f"{label} configuration")
    except ValueError as exc:
        raise AssistantBridgeError(str(exc)) from None


def _semantic_configuration_sha256(value: object, label: str) -> str:
    configured = getattr(value, "semantic_configuration_sha256", None)
    if configured is None:
        configured = getattr(value, "configuration_sha256", None)
    if callable(configured):
        configured = configured()
    if not isinstance(configured, str):
        raise AssistantBridgeError(
            f"{label} semantic configuration digest is unavailable."
        )
    try:
        return require_sha256(configured, f"{label} semantic configuration")
    except ValueError as exc:
        raise AssistantBridgeError(str(exc)) from None


def _clock_configuration_sha256(clock: Callable[[], float]) -> str:
    if clock is time.time:
        return sha256_json(
            {
                "contract": "mymoe-system-wall-clock/v1",
                "implementation": "time.time",
            }
        )
    return _configuration_sha256(clock, "paired attestation clock")


def _producer_state_paths(producer: object) -> tuple[Path, ...]:
    return _component_state_paths(
        producer,
        "paired attestation producer",
        fallback_root=False,
    )


def _evidence_store_state_paths(store: object) -> tuple[Path, ...]:
    return _component_state_paths(
        store,
        "paired evidence store",
        fallback_root=True,
    )


def _component_state_paths(
    component: object,
    label: str,
    *,
    fallback_root: bool,
) -> tuple[Path, ...]:
    declared = getattr(component, "state_paths", None)
    if callable(declared):
        declared = declared()
    if declared is None and fallback_root:
        root = getattr(component, "root", None)
        declared = () if root is None else (root,)
    if (
        declared is None
        or isinstance(declared, (str, bytes, bytearray))
        or not isinstance(declared, SequenceABC)
        or any(not isinstance(item, (str, Path)) for item in declared)
    ):
        raise AssistantBridgeError(f"{label} state paths are invalid.")
    paths: list[Path] = []
    for raw in declared:
        declared_path = Path(raw).expanduser()
        if declared_path.is_symlink():
            raise AssistantBridgeError(f"{label} state path cannot be a symbolic link.")
        try:
            path = declared_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise AssistantBridgeError(f"{label} state path is unavailable.") from exc
        if path in paths:
            raise AssistantBridgeError(f"{label} repeats a state path.")
        paths.append(path)
    return tuple(paths)


def _bounded_seconds(value: float, label: str, *, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1.0 <= float(value) <= maximum
    ):
        raise AssistantBridgeError(f"Paired {label} is outside safe bounds.")
    return float(value)


def _put_verified_json(
    store: EvidenceStore,
    payload: Mapping[str, object],
    *,
    media_type: str,
) -> ArtifactDescriptor:
    value = canonical_json_bytes(dict(payload))
    if not 0 < len(value) <= paired_artifact_max_bytes(media_type):
        raise AssistantBridgeError(
            "Paired evidence artifact exceeds its qualification byte bound."
        )
    descriptor = store.put_bytes(value, media_type=media_type)
    if (
        not isinstance(descriptor, ArtifactDescriptor)
        or descriptor.media_type != media_type
        or descriptor.sha256 != sha256_bytes(value)
        or descriptor.size_bytes != len(value)
        or store.get_bytes(descriptor) != value
    ):
        raise AssistantBridgeError(
            "Paired evidence store failed canonical artifact validation."
        )
    return descriptor


def _change_payload(change: WorkspaceChange) -> dict[str, object]:
    return {
        "path": change.path,
        "before": None if change.before is None else change.before.payload(),
        "after": None if change.after is None else change.after.payload(),
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _sanitize_metadata(value: object) -> object:
    sensitive = {
        "api_key",
        "authorization",
        "confirmation",
        "confirmation_id",
        "credential",
        "password",
        "secret",
        "token",
    }
    if isinstance(value, Mapping):
        sanitized = {
            str(key): _sanitize_metadata(item)
            for key, item in value.items()
            if str(key).lower() not in sensitive
        }
        return json.loads(canonical_json(sanitized))
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata(item) for item in value]
    return value
