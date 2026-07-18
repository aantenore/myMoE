from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Mapping, Sequence

from .assistant_bridge_integrity import canonical_json_bytes, canonical_sha256, sha256_bytes


TWO_PHASE_SCHEMA_VERSION = "1.0"
IN_TOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1"
INDEPENDENT_PREDICATE_V1 = (
    "https://github.com/aantenore/myMoE/tree/main/docs/spec/"
    "independent-candidate-attestation/v1"
)
DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json"
WORKFLOW_STATES = frozenset(
    {
        "staged",
        "attested",
        "ready",
        "applying",
        "applied",
        "conflicted",
        "expired",
        "failed",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]{0,95}$")


class TwoPhaseContractError(ValueError):
    """Raised when staged workflow data violates its public contract."""


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TwoPhaseContractError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def require_safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise TwoPhaseContractError(f"{label} must be a safe identifier.")
    return value


def _require_timestamp(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise TwoPhaseContractError(f"{label} must be a finite timestamp.")
    return float(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise TwoPhaseContractError(f"{label} contains unknown or missing fields.")


@dataclass(frozen=True)
class ArtifactDescriptor:
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.media_type, str) or _MEDIA_TYPE.fullmatch(
            self.media_type
        ) is None:
            raise TwoPhaseContractError("Artifact media_type is invalid.")
        require_sha256(self.sha256, "artifact sha256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or not 0 <= self.size_bytes <= 2**63 - 1
        ):
            raise TwoPhaseContractError("Artifact size_bytes is outside safe bounds.")

    def payload(self) -> dict[str, object]:
        return {
            "mediaType": self.media_type,
            "digest": {"sha256": self.sha256},
            "sizeBytes": self.size_bytes,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ArtifactDescriptor:
        if not isinstance(value, Mapping):
            raise TwoPhaseContractError("Artifact descriptor is invalid.")
        _require_exact_fields(
            value, {"mediaType", "digest", "sizeBytes"}, "Artifact descriptor"
        )
        digest = value.get("digest")
        if not isinstance(digest, Mapping):
            raise TwoPhaseContractError("Artifact digest descriptor is invalid.")
        _require_exact_fields(digest, {"sha256"}, "Artifact digest")
        media_type = value.get("mediaType")
        sha256 = digest.get("sha256")
        size_bytes = value.get("sizeBytes")
        if not isinstance(media_type, str) or not isinstance(sha256, str):
            raise TwoPhaseContractError("Artifact descriptor types are invalid.")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise TwoPhaseContractError("Artifact size_bytes is invalid.")
        return cls(media_type=media_type, sha256=sha256, size_bytes=size_bytes)


@dataclass(frozen=True, order=True)
class VerifierRequirement:
    verifier_id: str
    adapter_id: str
    key_id: str
    public_key_sha256: str
    spec_sha256: str

    def __post_init__(self) -> None:
        require_safe_id(self.verifier_id, "verifier_id")
        require_safe_id(self.adapter_id, "adapter_id")
        require_safe_id(self.key_id, "key_id")
        require_sha256(self.public_key_sha256, "public_key_sha256")
        require_sha256(self.spec_sha256, "spec_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "verifierId": self.verifier_id,
            "adapterId": self.adapter_id,
            "keyId": self.key_id,
            "publicKeySha256": self.public_key_sha256,
            "specSha256": self.spec_sha256,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> VerifierRequirement:
        _require_exact_fields(
            value,
            {
                "verifierId",
                "adapterId",
                "keyId",
                "publicKeySha256",
                "specSha256",
            },
            "Verifier requirement",
        )
        if not all(isinstance(item, str) for item in value.values()):
            raise TwoPhaseContractError("Verifier requirement types are invalid.")
        return cls(
            verifier_id=value["verifierId"],
            adapter_id=value["adapterId"],
            key_id=value["keyId"],
            public_key_sha256=value["publicKeySha256"],
            spec_sha256=value["specSha256"],
        )


@dataclass(frozen=True)
class VerificationPolicy:
    policy_id: str
    quorum: int
    verifiers: tuple[VerifierRequirement, ...]

    def __post_init__(self) -> None:
        require_safe_id(self.policy_id, "verification policy_id")
        ordered = tuple(sorted(self.verifiers, key=lambda item: item.verifier_id))
        if not ordered:
            raise TwoPhaseContractError(
                "Verification policy requires at least one verifier."
            )
        if len({item.verifier_id for item in ordered}) != len(ordered):
            raise TwoPhaseContractError("Verification policy repeats a verifier_id.")
        if (
            isinstance(self.quorum, bool)
            or not isinstance(self.quorum, int)
            or not 1 <= self.quorum <= len(ordered)
        ):
            raise TwoPhaseContractError("Verification policy quorum is invalid.")
        object.__setattr__(self, "verifiers", ordered)

    @property
    def policy_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "policyId": self.policy_id,
            "quorum": self.quorum,
            "verifiers": [item.payload() for item in self.verifiers],
        }

    def requirement(self, verifier_id: str) -> VerifierRequirement:
        for requirement in self.verifiers:
            if requirement.verifier_id == verifier_id:
                return requirement
        raise TwoPhaseContractError("Verifier is not trusted by this workflow.")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> VerificationPolicy:
        _require_exact_fields(
            value, {"policyId", "quorum", "verifiers"}, "Verification policy"
        )
        raw_verifiers = value.get("verifiers")
        if not isinstance(raw_verifiers, list) or not all(
            isinstance(item, Mapping) for item in raw_verifiers
        ):
            raise TwoPhaseContractError("Verification policy verifiers are invalid.")
        policy_id = value.get("policyId")
        quorum = value.get("quorum")
        if not isinstance(policy_id, str) or isinstance(quorum, bool) or not isinstance(
            quorum, int
        ):
            raise TwoPhaseContractError("Verification policy types are invalid.")
        return cls(
            policy_id=policy_id,
            quorum=quorum,
            verifiers=tuple(
                VerifierRequirement.from_payload(item) for item in raw_verifiers
            ),
        )


@dataclass(frozen=True)
class CandidateBinding:
    workflow_id: str
    stage_idempotency_sha256: str
    task_fingerprint: str
    config_sha256: str
    source_fingerprint: str
    challenge_sha256: str
    manifest: ArtifactDescriptor
    changeset: ArtifactDescriptor
    verification_policy: VerificationPolicy
    created_at: float
    expires_at: float

    def __post_init__(self) -> None:
        require_safe_id(self.workflow_id, "workflow_id")
        for label in (
            "stage_idempotency_sha256",
            "task_fingerprint",
            "config_sha256",
            "source_fingerprint",
            "challenge_sha256",
        ):
            require_sha256(getattr(self, label), label)
        created = _require_timestamp(self.created_at, "created_at")
        expires = _require_timestamp(self.expires_at, "expires_at")
        if created >= expires:
            raise TwoPhaseContractError("Candidate lifetime is invalid.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)

    @property
    def candidate_content_sha256(self) -> str:
        """Derive reusable artifact identity without workflow-specific freshness."""

        return canonical_sha256(
            {
                "derivation": "mymoe-candidate-content/v1",
                "manifest": self.manifest.payload(),
                "changeset": self.changeset.payload(),
            }
        )

    @property
    def candidate_fingerprint(self) -> str:
        """Bind artifact content to task, source, configuration, and trust policy."""

        return canonical_sha256(
            {
                "derivation": "mymoe-candidate-binding/v1",
                "taskFingerprint": self.task_fingerprint,
                "configSha256": self.config_sha256,
                "sourceFingerprint": self.source_fingerprint,
                "candidateContentSha256": self.candidate_content_sha256,
                "verificationPolicy": self.verification_policy.payload(),
                "verificationPolicySha256": self.verification_policy.policy_sha256,
            }
        )

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": TWO_PHASE_SCHEMA_VERSION,
            "workflowId": self.workflow_id,
            "stageIdempotencySha256": self.stage_idempotency_sha256,
            "taskFingerprint": self.task_fingerprint,
            "configSha256": self.config_sha256,
            "sourceFingerprint": self.source_fingerprint,
            "candidateContentSha256": self.candidate_content_sha256,
            "candidateFingerprint": self.candidate_fingerprint,
            "challengeSha256": self.challenge_sha256,
            "manifest": self.manifest.payload(),
            "changeset": self.changeset.payload(),
            "verificationPolicy": self.verification_policy.payload(),
            "verificationPolicySha256": self.verification_policy.policy_sha256,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> CandidateBinding:
        _require_exact_fields(
            value,
            {
                "schemaVersion",
                "workflowId",
                "stageIdempotencySha256",
                "taskFingerprint",
                "configSha256",
                "sourceFingerprint",
                "candidateContentSha256",
                "candidateFingerprint",
                "challengeSha256",
                "manifest",
                "changeset",
                "verificationPolicy",
                "verificationPolicySha256",
                "createdAt",
                "expiresAt",
            },
            "Candidate binding",
        )
        if value.get("schemaVersion") != TWO_PHASE_SCHEMA_VERSION:
            raise TwoPhaseContractError("Candidate binding schema is unsupported.")
        raw_manifest = value.get("manifest")
        raw_changeset = value.get("changeset")
        raw_policy = value.get("verificationPolicy")
        if not all(
            isinstance(item, Mapping)
            for item in (raw_manifest, raw_changeset, raw_policy)
        ):
            raise TwoPhaseContractError("Candidate binding descriptors are invalid.")
        strings = (
            "workflowId",
            "stageIdempotencySha256",
            "taskFingerprint",
            "configSha256",
            "sourceFingerprint",
            "candidateContentSha256",
            "candidateFingerprint",
            "challengeSha256",
            "verificationPolicySha256",
        )
        if not all(isinstance(value.get(name), str) for name in strings):
            raise TwoPhaseContractError("Candidate binding field types are invalid.")
        binding = cls(
            workflow_id=value["workflowId"],
            stage_idempotency_sha256=value["stageIdempotencySha256"],
            task_fingerprint=value["taskFingerprint"],
            config_sha256=value["configSha256"],
            source_fingerprint=value["sourceFingerprint"],
            challenge_sha256=value["challengeSha256"],
            manifest=ArtifactDescriptor.from_payload(raw_manifest),
            changeset=ArtifactDescriptor.from_payload(raw_changeset),
            verification_policy=VerificationPolicy.from_payload(raw_policy),
            created_at=value.get("createdAt"),
            expires_at=value.get("expiresAt"),
        )
        if value["candidateFingerprint"] != binding.candidate_fingerprint:
            raise TwoPhaseContractError("Candidate fingerprint derivation is invalid.")
        if value["candidateContentSha256"] != binding.candidate_content_sha256:
            raise TwoPhaseContractError("Candidate content digest derivation is invalid.")
        if value["verificationPolicySha256"] != binding.verification_policy.policy_sha256:
            raise TwoPhaseContractError("Verification policy digest is invalid.")
        return binding


@dataclass(frozen=True, order=True)
class AttestationCheck:
    check_id: str
    passed: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        require_safe_id(self.check_id, "attestation check_id")
        if not isinstance(self.passed, bool):
            raise TwoPhaseContractError("Attestation check result must be boolean.")
        require_sha256(self.evidence_sha256, "attestation check evidence_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "id": self.check_id,
            "passed": self.passed,
            "evidenceSha256": self.evidence_sha256,
        }


def build_attestation_statement(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    *,
    attestation_id: str,
    issued_at: float,
    expires_at: float,
    checks: Sequence[AttestationCheck],
) -> dict[str, object]:
    require_safe_id(attestation_id, "attestation_id")
    issued = _require_timestamp(issued_at, "attestation issued_at")
    expires = _require_timestamp(expires_at, "attestation expires_at")
    if issued >= expires or issued < binding.created_at or expires > binding.expires_at:
        raise TwoPhaseContractError("Attestation lifetime is outside the workflow.")
    ordered = tuple(sorted(checks, key=lambda item: item.check_id))
    if not ordered or len({item.check_id for item in ordered}) != len(ordered):
        raise TwoPhaseContractError("Attestation checks must be non-empty and unique.")
    return {
        "_type": IN_TOTO_STATEMENT_V1,
        "subject": [
            {
                "name": f"urn:mymoe:candidate:{binding.workflow_id}",
                "digest": {"sha256": binding.candidate_fingerprint},
            }
        ],
        "predicateType": INDEPENDENT_PREDICATE_V1,
        "predicate": {
            "schemaVersion": TWO_PHASE_SCHEMA_VERSION,
            "binding": binding.payload(),
            "bindingSha256": binding.binding_sha256,
            "attestation": {
                "attestationId": attestation_id,
                "verifierId": requirement.verifier_id,
                "adapterId": requirement.adapter_id,
                "keyId": requirement.key_id,
                "publicKeySha256": requirement.public_key_sha256,
                "specSha256": requirement.spec_sha256,
                "trustPolicySha256": binding.verification_policy.policy_sha256,
                "issuedAt": issued,
                "expiresAt": expires,
            },
            "outcome": {
                "passed": all(item.passed for item in ordered),
                "checks": [item.payload() for item in ordered],
            },
        },
    }


@dataclass(frozen=True)
class IndependentAttestation:
    """Immutable output of an adapter verification, never an authority input."""

    adapter_id: str
    verifier_id: str
    key_id: str
    attestation_id: str
    statement_bytes: bytes = field(repr=False)
    envelope_bytes: bytes = field(repr=False)
    evidence_sha256: str
    issued_at: float
    expires_at: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.adapter_id, "attestation adapter_id"),
            (self.verifier_id, "attestation verifier_id"),
            (self.key_id, "attestation key_id"),
            (self.attestation_id, "attestation_id"),
        ):
            require_safe_id(value, label)
        if not isinstance(self.statement_bytes, bytes) or not isinstance(
            self.envelope_bytes, bytes
        ):
            raise TwoPhaseContractError("Attestation evidence must be immutable bytes.")
        require_sha256(self.evidence_sha256, "attestation evidence_sha256")
        if sha256_bytes(self.envelope_bytes) != self.evidence_sha256:
            raise TwoPhaseContractError("Attestation evidence digest is invalid.")
        issued = _require_timestamp(self.issued_at, "attestation issued_at")
        expires = _require_timestamp(self.expires_at, "attestation expires_at")
        if issued >= expires:
            raise TwoPhaseContractError("Attestation lifetime is invalid.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)

    def statement(self) -> dict[str, Any]:
        value = json.loads(self.statement_bytes)
        if not isinstance(value, dict) or canonical_json_bytes(value) != self.statement_bytes:
            raise TwoPhaseContractError("Verified statement bytes are not canonical.")
        return value

    def metadata_payload(self) -> dict[str, object]:
        return {
            "adapterId": self.adapter_id,
            "verifierId": self.verifier_id,
            "keyId": self.key_id,
            "attestationId": self.attestation_id,
            "evidenceSha256": self.evidence_sha256,
            "statementSha256": canonical_sha256(self.statement()),
            "issuedAt": self.issued_at,
            "expiresAt": self.expires_at,
        }


@dataclass(frozen=True)
class StageReceipt:
    workflow_id: str
    status: str
    binding: CandidateBinding
    challenge: str = field(repr=False)
    idempotent_replay: bool = False

    def __post_init__(self) -> None:
        require_safe_id(self.workflow_id, "workflow_id")
        if self.status not in {"staged", "attested", "ready"}:
            raise TwoPhaseContractError("A stage receipt status is invalid.")
        if self.workflow_id != self.binding.workflow_id:
            raise TwoPhaseContractError("Stage receipt workflow binding is invalid.")
        if not isinstance(self.challenge, str) or not self.challenge or len(
            self.challenge
        ) > 1024:
            raise TwoPhaseContractError("Stage challenge is invalid.")

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": TWO_PHASE_SCHEMA_VERSION,
            "mode": "assistant_bridge_stage",
            "workflowId": self.workflow_id,
            "status": self.status,
            "binding": self.binding.payload(),
            "bindingSha256": self.binding.binding_sha256,
            "challenge": self.challenge,
            "idempotentReplay": self.idempotent_replay,
            "mutation": "none",
        }


@dataclass(frozen=True)
class ResumePlan:
    workflow_id: str
    plan_id: str
    confirmation_id: str = field(repr=False)
    confirmation_expires_at: float
    candidate_fingerprint: str
    source_fingerprint: str
    binding_sha256: str
    idempotent_replay: bool = False

    def __post_init__(self) -> None:
        require_safe_id(self.workflow_id, "workflow_id")
        require_sha256(self.plan_id, "resume plan_id")
        require_sha256(self.candidate_fingerprint, "candidate_fingerprint")
        require_sha256(self.source_fingerprint, "source_fingerprint")
        require_sha256(self.binding_sha256, "binding_sha256")
        expires = _require_timestamp(
            self.confirmation_expires_at, "confirmation_expires_at"
        )
        if not isinstance(self.confirmation_id, str) or not self.confirmation_id:
            raise TwoPhaseContractError("Resume confirmation is invalid.")
        object.__setattr__(self, "confirmation_expires_at", expires)

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": TWO_PHASE_SCHEMA_VERSION,
            "mode": "assistant_bridge_resume_plan",
            "workflowId": self.workflow_id,
            "planId": self.plan_id,
            "confirmationId": self.confirmation_id,
            "confirmationExpiresAt": self.confirmation_expires_at,
            "candidateFingerprint": self.candidate_fingerprint,
            "sourceFingerprint": self.source_fingerprint,
            "bindingSha256": self.binding_sha256,
            "idempotentReplay": self.idempotent_replay,
            "authority": "single_write_local_resume",
        }


@dataclass(frozen=True)
class ResumeResult:
    workflow_id: str
    status: str
    code: str
    candidate_fingerprint: str
    transaction_id: str | None = None
    result_sha256: str | None = None
    idempotent_replay: bool = False

    def __post_init__(self) -> None:
        require_safe_id(self.workflow_id, "workflow_id")
        if self.status not in WORKFLOW_STATES:
            raise TwoPhaseContractError("Resume result status is invalid.")
        require_safe_id(self.code, "resume result code")
        require_sha256(self.candidate_fingerprint, "candidate_fingerprint")
        if self.transaction_id is not None:
            require_sha256(self.transaction_id, "transaction_id")
        if self.result_sha256 is not None:
            require_sha256(self.result_sha256, "result_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": TWO_PHASE_SCHEMA_VERSION,
            "mode": "assistant_bridge_resume",
            "workflowId": self.workflow_id,
            "status": self.status,
            "code": self.code,
            "candidateFingerprint": self.candidate_fingerprint,
            "transactionId": self.transaction_id,
            "resultSha256": self.result_sha256,
            "idempotentReplay": self.idempotent_replay,
        }
