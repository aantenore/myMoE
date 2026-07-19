"""Authenticated CAS receipts for verified paired-routing outcomes.

The JSONL outcome is deliberately only an index.  Promotion authority is
derived by loading this receipt and reconstructing the outcome from signed,
content-addressed inputs rather than trusting fields copied into the JSONL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import re
from typing import Any, Mapping

from .assistant_bridge import AssistantBridgeConfig, VerificationEvidence
from .assistant_bridge_attestation import AttestationVerificationError
from .assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    _normalize_change,
    _normalize_file_record,
    _require_portable_unique_paths,
)
from .assistant_bridge_integrity import canonical_json_bytes
from .assistant_bridge_two_phase_config import TwoPhaseTrustConfig
from .assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    CandidateBinding,
    IndependentEvaluationAttestation,
    MAX_CANDIDATE_FILES,
    MAX_CANDIDATE_FILE_BYTES,
    MAX_CANDIDATE_TOTAL_BYTES,
    TwoPhaseContractError,
)
from .paired_execution_contracts import PairedOutcomeBinding
from .paired_execution_pricing import (
    IncompleteCostEvidenceError,
    PairedCostEvidence,
    PricingContract,
    build_cost_evidence,
)
from .route_outcomes import VerifiedOutcomeRecord, build_verified_outcome
from .route_signals import TaskSignals
from .verified_routing_contracts import (
    VerifiedRoutingError,
    require_sha256,
    sha256_json,
)


PAIRED_RESULT_METADATA_MEDIA_TYPE = (
    "application/vnd.mymoe.paired-result-metadata+json"
)
PAIRED_SIGNALS_MEDIA_TYPE = "application/vnd.mymoe.task-signals+json"
PAIRED_MANIFEST_MEDIA_TYPE = "application/vnd.mymoe.paired-manifest+json"
PAIRED_CHANGESET_MEDIA_TYPE = "application/vnd.mymoe.paired-changeset+json"
PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE = "application/vnd.dsse.envelope+json"
PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE = (
    "application/vnd.mymoe.paired-attestation-receipt+json"
)

_MAX_RECEIPT_BYTES = 512 * 1024
_MAX_RESULT_METADATA_BYTES = 2 * 1024 * 1024
_MAX_SIGNALS_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CHANGESET_BYTES = 8 * 1024 * 1024
_MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
_MAX_ENVELOPES = 64
_RECEIPT_FIELDS = {
    "schemaVersion",
    "contract",
    "binding",
    "pairedOutcomeBinding",
    "resultMetadata",
    "signals",
    "manifest",
    "changeset",
    "envelopes",
    "workspaceFingerprint",
}

_MEDIA_TYPE_MAX_BYTES = {
    PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE: _MAX_RECEIPT_BYTES,
    PAIRED_RESULT_METADATA_MEDIA_TYPE: _MAX_RESULT_METADATA_BYTES,
    PAIRED_SIGNALS_MEDIA_TYPE: _MAX_SIGNALS_BYTES,
    PAIRED_MANIFEST_MEDIA_TYPE: _MAX_MANIFEST_BYTES,
    PAIRED_CHANGESET_MEDIA_TYPE: _MAX_CHANGESET_BYTES,
    PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE: _MAX_ENVELOPE_BYTES,
}


def paired_artifact_max_bytes(media_type: str) -> int:
    try:
        return _MEDIA_TYPE_MAX_BYTES[media_type]
    except KeyError as exc:
        raise VerifiedRoutingError(
            "Paired evidence media type has no configured byte bound."
        ) from exc


@dataclass(frozen=True)
class PairedAttestationReceipt:
    """Complete immutable index needed to reproduce one paired outcome."""

    binding: CandidateBinding
    paired_outcome_binding: PairedOutcomeBinding
    result_metadata: ArtifactDescriptor
    signals: ArtifactDescriptor
    manifest: ArtifactDescriptor
    changeset: ArtifactDescriptor
    envelopes: tuple[ArtifactDescriptor, ...]
    workspace_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CandidateBinding):
            raise VerifiedRoutingError("Paired evidence binding is invalid.")
        if not isinstance(self.paired_outcome_binding, PairedOutcomeBinding):
            raise VerifiedRoutingError("Paired outcome binding is invalid.")
        for value, label in (
            (self.result_metadata, "result metadata"),
            (self.signals, "task signals"),
            (self.manifest, "manifest"),
            (self.changeset, "changeset"),
        ):
            if not isinstance(value, ArtifactDescriptor):
                raise VerifiedRoutingError(f"Paired evidence {label} is invalid.")
        envelopes = tuple(self.envelopes)
        if (
            not envelopes
            or len(envelopes) > _MAX_ENVELOPES
            or any(not isinstance(item, ArtifactDescriptor) for item in envelopes)
            or envelopes != tuple(sorted(envelopes, key=lambda item: item.sha256))
            or len({item.sha256 for item in envelopes}) != len(envelopes)
        ):
            raise VerifiedRoutingError(
                "Paired evidence envelopes must be non-empty, unique, and canonical."
            )
        object.__setattr__(self, "envelopes", envelopes)
        object.__setattr__(
            self,
            "workspace_fingerprint",
            require_sha256(self.workspace_fingerprint, "workspace_fingerprint"),
        )
        if self.binding.manifest != self.manifest or self.binding.changeset != self.changeset:
            raise VerifiedRoutingError(
                "Paired receipt artifact descriptors disagree with its signed binding."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": "1.0",
            "contract": "PairedAttestationReceipt",
            "binding": self.binding.payload(),
            "pairedOutcomeBinding": self.paired_outcome_binding.payload(),
            "resultMetadata": self.result_metadata.payload(),
            "signals": self.signals.payload(),
            "manifest": self.manifest.payload(),
            "changeset": self.changeset.payload(),
            "envelopes": [item.payload() for item in self.envelopes],
            "workspaceFingerprint": self.workspace_fingerprint,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "PairedAttestationReceipt":
        raw = _mapping(value, "paired attestation receipt")
        if set(raw) != _RECEIPT_FIELDS:
            raise VerifiedRoutingError(
                "Paired attestation receipt contains unknown or missing fields."
            )
        if (
            raw.get("schemaVersion") != "1.0"
            or raw.get("contract") != "PairedAttestationReceipt"
        ):
            raise VerifiedRoutingError("Paired attestation receipt is unsupported.")
        envelope_values = raw.get("envelopes")
        if not isinstance(envelope_values, list):
            raise VerifiedRoutingError("Paired receipt envelopes must be a list.")
        try:
            return cls(
                binding=CandidateBinding.from_payload(
                    _mapping(raw.get("binding"), "candidate binding")
                ),
                paired_outcome_binding=PairedOutcomeBinding.from_payload(
                    raw.get("pairedOutcomeBinding")
                ),
                result_metadata=ArtifactDescriptor.from_payload(
                    _mapping(raw.get("resultMetadata"), "result metadata descriptor")
                ),
                signals=ArtifactDescriptor.from_payload(
                    _mapping(raw.get("signals"), "signals descriptor")
                ),
                manifest=ArtifactDescriptor.from_payload(
                    _mapping(raw.get("manifest"), "manifest descriptor")
                ),
                changeset=ArtifactDescriptor.from_payload(
                    _mapping(raw.get("changeset"), "changeset descriptor")
                ),
                envelopes=tuple(
                    ArtifactDescriptor.from_payload(
                        _mapping(item, "attestation envelope descriptor")
                    )
                    for item in envelope_values
                ),
                workspace_fingerprint=str(raw.get("workspaceFingerprint", "")),
            )
        except TwoPhaseContractError as exc:
            raise VerifiedRoutingError(str(exc)) from exc


@dataclass(frozen=True)
class VerifiedPairedEvidence:
    """Qualification result plus lineage not duplicated in the outcome row."""

    record: VerifiedOutcomeRecord
    paired_outcome_binding: PairedOutcomeBinding
    receipt_descriptor: ArtifactDescriptor
    verifier_ids: tuple[str, ...]
    candidate_created_at: float
    latest_attestation_issued_at: float
    earliest_attestation_expires_at: float


class PairedAttestationVerifier:
    """Concrete fail-closed verifier used by qualification and signing."""

    __slots__ = (
        "_trust_config",
        "_evidence_root",
        "_bridge_config",
        "_runner_source_sha256",
        "_configuration_sha256",
        "_sealed",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PairedAttestationVerifier is final and cannot be subclassed.")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PairedAttestationVerifier is immutable.")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        trust_config: TwoPhaseTrustConfig,
        evidence_store: ContentAddressedStore,
        bridge_config: AssistantBridgeConfig,
    ) -> None:
        if not isinstance(trust_config, TwoPhaseTrustConfig):
            raise TypeError("trust_config must be TwoPhaseTrustConfig.")
        if type(evidence_store) is not ContentAddressedStore:
            raise TypeError("evidence_store must be ContentAddressedStore.")
        if not isinstance(bridge_config, AssistantBridgeConfig):
            raise TypeError("bridge_config must be AssistantBridgeConfig.")
        reopened_store = ContentAddressedStore(
            evidence_store.root,
            create_if_missing=False,
        )
        self._trust_config = trust_config
        self._evidence_root = reopened_store.root
        self._bridge_config = bridge_config
        from .paired_execution import paired_runner_source_sha256

        self._runner_source_sha256 = paired_runner_source_sha256()
        configured = bridge_config.external_verifiers
        for requirement in trust_config.policy.verifiers:
            spec = configured.get(requirement.verifier_id)
            if spec is None or spec.spec_sha256 != requirement.spec_sha256:
                raise VerifiedRoutingError(
                    "Paired verifier trust policy does not match Assistant Bridge."
                )
        self._configuration_sha256 = sha256_json(
            {
                "contract": "mymoe-paired-attestation-verifier/v2",
                "trust": trust_config.descriptor(),
                "evidence_store_semantic_sha256": (
                    reopened_store.semantic_configuration_sha256
                ),
                "bridge_config_sha256": bridge_config.source_sha256,
                "runner_source_sha256": self._runner_source_sha256,
            }
        )
        object.__setattr__(self, "_sealed", True)

    @property
    def trust_config(self) -> TwoPhaseTrustConfig:
        return self._trust_config

    @property
    def bridge_config(self) -> AssistantBridgeConfig:
        return self._bridge_config

    @property
    def runner_source_sha256(self) -> str:
        return self._runner_source_sha256

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    def verify_record(
        self,
        record: VerifiedOutcomeRecord,
        *,
        pricing: PricingContract,
    ) -> VerifiedPairedEvidence:
        if not isinstance(record, VerifiedOutcomeRecord):
            raise TypeError("record must be VerifiedOutcomeRecord.")
        if not isinstance(pricing, PricingContract):
            raise TypeError("pricing must be PricingContract.")
        raw_descriptor = record.paired_evidence
        if raw_descriptor is None:
            raise VerifiedRoutingError(
                "Outcome has no reconstructible paired attestation receipt."
            )
        try:
            descriptor = ArtifactDescriptor.from_payload(raw_descriptor)
        except TwoPhaseContractError as exc:
            raise VerifiedRoutingError(str(exc)) from exc
        _require_descriptor(
            descriptor,
            PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE,
            _MAX_RECEIPT_BYTES,
            "paired attestation receipt",
        )
        evidence_store = ContentAddressedStore(
            self._evidence_root,
            create_if_missing=False,
        )
        try:
            receipt_payload = ContentAddressedStore.get_json(
                evidence_store,
                descriptor,
                max_bytes=_MAX_RECEIPT_BYTES,
            )
        except ContentAddressedStoreError as exc:
            raise VerifiedRoutingError(str(exc)) from exc
        receipt = PairedAttestationReceipt.from_payload(receipt_payload)
        paired = receipt.paired_outcome_binding
        self._validate_binding(receipt)

        result_metadata = self._load_json(
            evidence_store,
            receipt.result_metadata,
            PAIRED_RESULT_METADATA_MEDIA_TYPE,
            _MAX_RESULT_METADATA_BYTES,
            "paired result metadata",
        )
        signals_payload = self._load_json(
            evidence_store,
            receipt.signals,
            PAIRED_SIGNALS_MEDIA_TYPE,
            _MAX_SIGNALS_BYTES,
            "paired task signals",
        )
        signals = TaskSignals.from_payload(signals_payload)
        manifest = self._load_json(
            evidence_store,
            receipt.manifest,
            PAIRED_MANIFEST_MEDIA_TYPE,
            _MAX_MANIFEST_BYTES,
            "paired manifest",
        )
        changeset = self._load_json(
            evidence_store,
            receipt.changeset,
            PAIRED_CHANGESET_MEDIA_TYPE,
            _MAX_CHANGESET_BYTES,
            "paired changeset",
        )
        self._validate_artifact_bindings(
            receipt,
            result_metadata=result_metadata,
            signals=signals,
            manifest=manifest,
            changeset=changeset,
        )

        verification = _mapping(
            result_metadata.get("verification"), "paired result verification"
        )
        if set(verification) != {"prior", "final"}:
            raise VerifiedRoutingError("Paired result verification shape is invalid.")
        raw_final = verification.get("final")
        if not isinstance(raw_final, list):
            raise VerifiedRoutingError("Paired final verification must be a list.")
        if any(
            isinstance(item, Mapping) and item.get("kind") == "external"
            for item in raw_final
        ):
            raise VerifiedRoutingError(
                "Pre-attestation result metadata already contains external evidence."
            )

        trust_store = self.trust_config.build_trust_store()
        verified: dict[str, IndependentEvaluationAttestation] = {}
        for envelope_descriptor in receipt.envelopes:
            _require_descriptor(
                envelope_descriptor,
                PAIRED_ATTESTATION_ENVELOPE_MEDIA_TYPE,
                _MAX_ENVELOPE_BYTES,
                "paired attestation envelope",
            )
            try:
                envelope = ContentAddressedStore.get_bytes(
                    evidence_store,
                    envelope_descriptor,
                )
            except ContentAddressedStoreError as exc:
                raise VerifiedRoutingError(str(exc)) from exc
            try:
                attestation = trust_store.verify_historical_evaluation(
                    receipt.binding,
                    envelope,
                )
            except AttestationVerificationError as exc:
                raise VerifiedRoutingError(str(exc)) from exc
            if (
                envelope != attestation.envelope_bytes
                or envelope_descriptor.sha256 != attestation.evidence_sha256
                or attestation.verifier_id in verified
            ):
                raise VerifiedRoutingError(
                    "Paired attestation envelope identity is invalid or repeated."
                )
            verified[attestation.verifier_id] = attestation
        if len(verified) < self.trust_config.policy.quorum:
            raise VerifiedRoutingError(
                "Paired attestation does not satisfy its signed verifier quorum."
            )

        route_receipt = _mapping(
            result_metadata.get("route_receipt"), "paired route receipt"
        )
        task = _mapping(route_receipt.get("task"), "paired route task")
        required_ids = task.get("required_verifier_ids")
        if not isinstance(required_ids, list) or any(
            not isinstance(item, str) for item in required_ids
        ):
            raise VerifiedRoutingError("Task-required verifier ids are invalid.")
        missing = sorted(set(required_ids).difference(verified))
        if missing:
            raise VerifiedRoutingError(
                "Paired attestation omits task-required verifier evidence."
            )

        external: list[VerificationEvidence] = []
        for verifier_id in sorted(verified):
            attestation = verified[verifier_id]
            spec = self.bridge_config.external_verifiers.get(verifier_id)
            requirement = self.trust_config.policy.requirement(verifier_id)
            if spec is None or spec.spec_sha256 != requirement.spec_sha256:
                raise VerifiedRoutingError(
                    "Signed verifier is absent from Assistant Bridge configuration."
                )
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
                    evidence_ref=f"cas://sha256/{attestation.evidence_sha256}",
                    task_fingerprint=paired.task_fingerprint,
                    workspace_fingerprint=receipt.workspace_fingerprint,
                    verifier_spec_sha256=spec.spec_sha256,
                )
            )
        reconstructed_metadata = dict(result_metadata)
        reconstructed_metadata["verification"] = {
            "prior": verification["prior"],
            "final": [*raw_final, *(item.payload() for item in external)],
        }
        cost = _recompute_cost(reconstructed_metadata, pricing)
        created_at = _signed_utc(receipt.binding.created_at)
        reconstructed = build_verified_outcome(
            reconstructed_metadata,
            signals,
            estimated_cost_usd=(
                None if cost is None else float(Decimal(cost.total_cost_usd))
            ),
            created_at=created_at,
            paired_run=paired.payload(),
            paired_cost=None if cost is None else cost.payload(),
            paired_evidence=descriptor.payload(),
        )
        if reconstructed.payload() != record.payload():
            raise VerifiedRoutingError(
                "Outcome row does not exactly reproduce from signed paired evidence."
            )
        return VerifiedPairedEvidence(
            record=reconstructed,
            paired_outcome_binding=paired,
            receipt_descriptor=descriptor,
            verifier_ids=tuple(sorted(verified)),
            candidate_created_at=receipt.binding.created_at,
            latest_attestation_issued_at=max(
                item.issued_at for item in verified.values()
            ),
            earliest_attestation_expires_at=min(
                item.expires_at for item in verified.values()
            ),
        )

    def _load_json(
        self,
        evidence_store: ContentAddressedStore,
        descriptor: ArtifactDescriptor,
        media_type: str,
        maximum: int,
        label: str,
    ) -> dict[str, Any]:
        _require_descriptor(descriptor, media_type, maximum, label)
        try:
            return ContentAddressedStore.get_json(
                evidence_store,
                descriptor,
                max_bytes=maximum,
            )
        except ContentAddressedStoreError as exc:
            raise VerifiedRoutingError(str(exc)) from exc

    def _validate_binding(self, receipt: PairedAttestationReceipt) -> None:
        binding = receipt.binding
        paired = receipt.paired_outcome_binding
        operation_sha256 = sha256_json(
            {
                "contract": "mymoe-paired-arm-operation/v1",
                "run_id": paired.run_id,
                "claim_sha256": paired.claim_sha256,
            }
        )
        lifecycle_config_sha256 = sha256_json(
            {
                "contract": "mymoe-paired-lifecycle/v1",
                "plan_sha256": paired.plan_sha256,
                "bridge_config_sha256": paired.bridge_config_sha256,
                "executor_config_sha256": paired.executor_config_sha256,
            }
        )
        if (
            binding.workflow_id != f"paired-{operation_sha256}"
            or binding.stage_idempotency_sha256 != operation_sha256
            or binding.task_fingerprint != paired.task_fingerprint
            or binding.config_sha256 != paired.lifecycle_config_sha256
            or paired.lifecycle_config_sha256 != lifecycle_config_sha256
            or binding.source_fingerprint != paired.source_snapshot_sha256
            or binding.verification_policy != self.trust_config.policy
            or paired.bridge_config_sha256 != self.bridge_config.source_sha256
        ):
            raise VerifiedRoutingError(
                "Signed candidate binding does not match paired-run lineage."
            )
        expected_challenge = paired_attestation_challenge_sha256(
            operation_sha256=operation_sha256,
            paired_outcome_binding_sha256=paired.binding_sha256,
            executor_config_sha256=paired.executor_config_sha256,
            lifecycle_config_sha256=paired.lifecycle_config_sha256,
            signals_sha256=paired.signals_sha256,
            result_metadata_sha256=receipt.result_metadata.sha256,
            run_instance_nonce=paired.run_instance_nonce,
        )
        if binding.challenge_sha256 != expected_challenge:
            raise VerifiedRoutingError("Paired attestation challenge is invalid.")

    def _validate_artifact_bindings(
        self,
        receipt: PairedAttestationReceipt,
        *,
        result_metadata: Mapping[str, object],
        signals: TaskSignals,
        manifest: Mapping[str, object],
        changeset: Mapping[str, object],
    ) -> None:
        paired = receipt.paired_outcome_binding
        from .paired_execution import paired_runner_sha256

        expected_runner_sha256 = paired_runner_sha256(
            executor_config_sha256=paired.executor_config_sha256,
            lifecycle_config_sha256=paired.lifecycle_config_sha256,
            signal_provider_config_sha256=signals.provider_config_sha256,
            runner_source_sha256=self.runner_source_sha256,
        )
        operation_sha256 = receipt.binding.stage_idempotency_sha256
        common = {
            "pairedOutcomeBindingSha256": paired.binding_sha256,
            "operationSha256": operation_sha256,
            "executorConfigSha256": paired.executor_config_sha256,
            "executionHarnessSha256": paired.execution_harness_sha256,
            "lifecycleConfigSha256": paired.lifecycle_config_sha256,
            "runnerSourceSha256": paired.runner_source_sha256,
            "signalsSha256": paired.signals_sha256,
            "resultMetadata": receipt.result_metadata.payload(),
            "signals": receipt.signals.payload(),
            "sourceFingerprint": paired.source_snapshot_sha256,
            "observedAt": receipt.binding.created_at,
        }
        manifest_required = {
            "schemaVersion",
            "contract",
            "candidateWorkspaceFingerprint",
            "candidateSnapshot",
            "workspaceAttestationFingerprint",
            "files",
            *common,
        }
        changeset_required = {
            "schemaVersion",
            "contract",
            "candidateWorkspaceFingerprint",
            "changes",
            *common,
        }
        if set(manifest) != manifest_required or set(changeset) != changeset_required:
            raise VerifiedRoutingError(
                "Paired manifest or changeset contains unknown or missing fields."
            )
        if (
            manifest.get("schemaVersion") != "1.0"
            or manifest.get("contract") != "PairedCandidateManifest"
            or changeset.get("schemaVersion") != "1.0"
            or changeset.get("contract") != "PairedCandidateChangeset"
            or any(manifest.get(key) != value for key, value in common.items())
            or any(changeset.get(key) != value for key, value in common.items())
            or manifest.get("candidateWorkspaceFingerprint")
            != changeset.get("candidateWorkspaceFingerprint")
            or manifest.get("workspaceAttestationFingerprint")
            != receipt.workspace_fingerprint
            or signals.signals_sha256 != paired.signals_sha256
            or signals.request_fingerprint != paired.task_fingerprint
            or paired.runner_sha256 != expected_runner_sha256
            or paired.runner_source_sha256 != self.runner_source_sha256
            or canonical_json_bytes(dict(result_metadata)).__len__()
            != receipt.result_metadata.size_bytes
        ):
            raise VerifiedRoutingError(
                "Paired manifest, signals, or result metadata binding is invalid."
            )
        if not isinstance(manifest.get("files"), list) or not isinstance(
            changeset.get("changes"), list
        ):
            raise VerifiedRoutingError("Paired artifact lists are invalid.")
        _validate_candidate_artifact_payloads(
            manifest,
            changeset,
            workspace_fingerprint=receipt.workspace_fingerprint,
        )


def _validate_candidate_artifact_payloads(
    manifest: Mapping[str, object],
    changeset: Mapping[str, object],
    *,
    workspace_fingerprint: str,
) -> None:
    raw_files = manifest.get("files")
    raw_changes = changeset.get("changes")
    raw_snapshot = manifest.get("candidateSnapshot")
    if (
        not isinstance(raw_files, list)
        or not isinstance(raw_changes, list)
        or not isinstance(raw_snapshot, Mapping)
    ):
        raise VerifiedRoutingError("Paired candidate artifact shape is invalid.")
    try:
        files = [
            _normalize_file_record(_mapping(item, "candidate file"))
            for item in raw_files
        ]
        changes = [
            _normalize_change(_mapping(item, "candidate change"))
            for item in raw_changes
        ]
        file_paths = [str(item["path"]) for item in files]
        change_paths = [str(item["path"]) for item in changes]
        _require_portable_unique_paths(file_paths, label="Paired candidate manifest")
        _require_portable_unique_paths(change_paths, label="Paired candidate changeset")
    except ContentAddressedStoreError as exc:
        raise VerifiedRoutingError(str(exc)) from exc
    if (
        files != raw_files
        or changes != raw_changes
        or file_paths != sorted(file_paths)
        or change_paths != sorted(change_paths)
        or len(file_paths) != len(set(file_paths))
        or len(change_paths) != len(set(change_paths))
    ):
        raise VerifiedRoutingError(
            "Paired candidate files or changes are not canonical and unique."
        )
    files_by_path = {str(item["path"]): item for item in files}
    for change in changes:
        after = change["after"]
        if after is not None and files_by_path.get(str(change["path"])) != after:
            raise VerifiedRoutingError(
                "Paired changeset after-state disagrees with candidate manifest."
            )
        if after is None and str(change["path"]) in files_by_path:
            raise VerifiedRoutingError(
                "Deleted paired changeset path remains in candidate manifest."
            )

    snapshot = _mapping(raw_snapshot, "candidate snapshot")
    snapshot_fields = {
        "root_sha256",
        "git_repository",
        "head_sha",
        "index_sha256",
        "status_sha256",
        "manifest_sha256",
        "fingerprint",
        "file_count",
        "tracked_file_count",
        "total_bytes",
        "scope",
    }
    if set(snapshot) != snapshot_fields:
        raise VerifiedRoutingError(
            "Paired candidate snapshot contains unknown or missing fields."
        )
    for name in (
        "root_sha256",
        "index_sha256",
        "status_sha256",
        "manifest_sha256",
        "fingerprint",
    ):
        require_sha256(snapshot.get(name), f"candidate snapshot {name}")
    git_repository = snapshot.get("git_repository")
    head_sha = snapshot.get("head_sha")
    file_count = snapshot.get("file_count")
    tracked_file_count = snapshot.get("tracked_file_count")
    total_bytes = snapshot.get("total_bytes")
    if (
        not isinstance(git_repository, bool)
        or (
            git_repository
            and (
                not isinstance(head_sha, str)
                or (
                    head_sha != "unborn"
                    and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head_sha)
                    is None
                )
            )
        )
        or (not git_repository and head_sha is not None)
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or isinstance(tracked_file_count, bool)
        or not isinstance(tracked_file_count, int)
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or not 0 <= tracked_file_count <= file_count
        or file_count > MAX_CANDIDATE_FILES
        or len(changes) > MAX_CANDIDATE_FILES
        or file_count != len(files)
        or any(int(item["size"]) > MAX_CANDIDATE_FILE_BYTES for item in files)
        or total_bytes > MAX_CANDIDATE_TOTAL_BYTES
        or total_bytes != sum(int(item["size"]) for item in files)
        or snapshot.get("scope")
        != "tracked_untracked_nonignored_plus_declared_ignored"
    ):
        raise VerifiedRoutingError("Paired candidate snapshot metadata is invalid.")
    manifest_sha256 = sha256_json(files)
    rendered_head = "" if head_sha is None else head_sha
    status_sha256 = sha256_json(
        {
            "derivation": "head-index-manifest/v1",
            "head_sha": rendered_head,
            "index_sha256": snapshot["index_sha256"],
            "manifest_sha256": manifest_sha256,
        }
    )
    candidate_fingerprint = sha256_json(
        {
            "root_sha256": snapshot["root_sha256"],
            "git_repository": git_repository,
            "head_sha": rendered_head,
            "index_sha256": snapshot["index_sha256"],
            "status_sha256": status_sha256,
            "manifest_sha256": manifest_sha256,
        }
    )
    attestation_fingerprint = sha256_json(
        {
            "git_repository": git_repository,
            "head_sha": rendered_head,
            "index_sha256": snapshot["index_sha256"],
            "status_sha256": status_sha256,
            "manifest_sha256": manifest_sha256,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "scope": snapshot["scope"],
        }
    )
    if (
        snapshot["manifest_sha256"] != manifest_sha256
        or snapshot["status_sha256"] != status_sha256
        or snapshot["fingerprint"] != candidate_fingerprint
        or manifest.get("candidateWorkspaceFingerprint") != candidate_fingerprint
        or manifest.get("workspaceAttestationFingerprint")
        != attestation_fingerprint
        or workspace_fingerprint != attestation_fingerprint
    ):
        raise VerifiedRoutingError(
            "Paired candidate snapshot fingerprint coherence is invalid."
        )


def paired_attestation_challenge_sha256(
    *,
    operation_sha256: str,
    paired_outcome_binding_sha256: str,
    executor_config_sha256: str,
    lifecycle_config_sha256: str,
    signals_sha256: str,
    result_metadata_sha256: str,
    run_instance_nonce: str,
) -> str:
    return sha256_json(
        {
            "contract": "mymoe-paired-attestation-challenge/v2",
            "operation_sha256": require_sha256(
                operation_sha256, "operation_sha256"
            ),
            "paired_outcome_binding_sha256": require_sha256(
                paired_outcome_binding_sha256,
                "paired_outcome_binding_sha256",
            ),
            "executor_config_sha256": require_sha256(
                executor_config_sha256, "executor_config_sha256"
            ),
            "lifecycle_config_sha256": require_sha256(
                lifecycle_config_sha256, "lifecycle_config_sha256"
            ),
            "signals_sha256": require_sha256(signals_sha256, "signals_sha256"),
            "result_metadata_sha256": require_sha256(
                result_metadata_sha256, "result_metadata_sha256"
            ),
            "run_instance_nonce": require_sha256(
                run_instance_nonce, "run_instance_nonce"
            ),
        }
    )


def _require_descriptor(
    descriptor: ArtifactDescriptor,
    media_type: str,
    maximum: int,
    label: str,
) -> None:
    if descriptor.media_type != media_type or not 0 < descriptor.size_bytes <= maximum:
        raise VerifiedRoutingError(f"{label} descriptor is invalid or oversized.")


def _recompute_cost(
    bridge_metadata: Mapping[str, object],
    pricing: PricingContract,
) -> PairedCostEvidence | None:
    receipt = _mapping(bridge_metadata.get("route_receipt"), "route receipt")
    runtimes: dict[str, Mapping[str, object]] = {}
    for provider_key, runtime_key in (
        ("local_provider", "local_runtime"),
        ("premium_provider", "premium_runtime"),
    ):
        provider = receipt.get(provider_key)
        if isinstance(provider, str):
            runtimes[provider] = _mapping(receipt.get(runtime_key), runtime_key)
    raw_commands = bridge_metadata.get("commands")
    if not isinstance(raw_commands, list):
        raise VerifiedRoutingError("Paired result commands must be a list.")
    commands: list[dict[str, object]] = []
    try:
        for raw in raw_commands:
            command = _mapping(raw, "paired result command")
            provider = command.get("provider_id")
            if not isinstance(provider, str) or provider not in runtimes:
                raise VerifiedRoutingError(
                    "Paired command provider is not bound by its route receipt."
                )
            usage = _mapping(command.get("usage"), "paired command usage")
            runtime = runtimes[provider]
            commands.append(
                {
                    "provider_id": provider,
                    "model": runtime.get("model"),
                    "provider_runtime_sha256": runtime.get("runtime_sha256"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )
        return build_cost_evidence(pricing, commands)
    except IncompleteCostEvidenceError:
        return None


def _signed_utc(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise VerifiedRoutingError(f"{label} must be an object.")
    return dict(value)
