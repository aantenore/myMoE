from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from .assistant_bridge_two_phase_contracts import (
    AttestationCheck,
    CandidateBinding,
    DSSE_PAYLOAD_TYPE,
    INDEPENDENT_EVALUATION_PREDICATE_V2,
    INDEPENDENT_PREDICATE_V1,
    IN_TOTO_STATEMENT_V1,
    IndependentAttestation,
    IndependentEvaluationAttestation,
    TWO_PHASE_SCHEMA_VERSION,
    TwoPhaseContractError,
    VerifierRequirement,
    build_attestation_statement,
    build_evaluation_attestation_statement,
    require_safe_id,
    require_sha256,
)


ED25519_DSSE_ADAPTER_ID = "dsse-ed25519-v1"
_MAX_ENVELOPE_BYTES = 8 * 1024 * 1024


class AttestationVerificationError(ValueError):
    """Raised when independent evidence does not satisfy its signed contract."""


def ed25519_public_key_sha256(public_key: Ed25519PublicKey) -> str:
    value = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return sha256_bytes(value)


def load_ed25519_public_key_pem(value: bytes) -> Ed25519PublicKey:
    if not isinstance(value, bytes) or not value:
        raise AttestationVerificationError("Verifier public key is invalid.")
    try:
        key = serialization.load_pem_public_key(value)
    except (TypeError, ValueError) as exc:
        raise AttestationVerificationError("Verifier public key is invalid.") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AttestationVerificationError("Verifier key must be Ed25519.")
    return key


@dataclass(frozen=True)
class TrustedEd25519Verifier:
    """Application-owned public-key adapter; it never receives signing material."""

    requirement: VerifierRequirement
    _public_key: Ed25519PublicKey = field(repr=False)

    def __post_init__(self) -> None:
        if self.requirement.adapter_id != ED25519_DSSE_ADAPTER_ID:
            raise AttestationVerificationError("Verifier adapter id is unsupported.")
        if not isinstance(self._public_key, Ed25519PublicKey):
            raise AttestationVerificationError("Verifier key must be Ed25519.")
        if (
            ed25519_public_key_sha256(self._public_key)
            != self.requirement.public_key_sha256
        ):
            raise AttestationVerificationError(
                "Verifier public key does not match the trust policy."
            )

    @classmethod
    def from_pem(
        cls,
        requirement: VerifierRequirement,
        public_key_pem: bytes,
    ) -> TrustedEd25519Verifier:
        return cls(requirement, load_ed25519_public_key_pem(public_key_pem))

    def verify(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float,
    ) -> IndependentAttestation:
        current = _timestamp(now)
        canonical_envelope, statement_bytes, statement = _decode_envelope(
            envelope, expected_key_id=self.requirement.key_id
        )
        signature = _envelope_signature(canonical_envelope)
        try:
            self._public_key.verify(
                signature,
                _dsse_pae(DSSE_PAYLOAD_TYPE, statement_bytes),
            )
        except InvalidSignature as exc:
            raise AttestationVerificationError(
                "Independent attestation signature is invalid."
            ) from exc
        attestation_id, issued_at, expires_at = _validate_statement(
            statement,
            binding=binding,
            requirement=self.requirement,
            now=current,
        )
        return IndependentAttestation(
            adapter_id=self.requirement.adapter_id,
            verifier_id=self.requirement.verifier_id,
            key_id=self.requirement.key_id,
            attestation_id=attestation_id,
            statement_bytes=statement_bytes,
            envelope_bytes=canonical_envelope,
            evidence_sha256=sha256_bytes(canonical_envelope),
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def verify_evaluation(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float,
    ) -> IndependentEvaluationAttestation:
        """Verify a currently live v2 evaluation without granting apply authority."""

        return self._verify_evaluation(binding, envelope, now=_timestamp(now))

    def verify_historical_evaluation(
        self,
        binding: CandidateBinding,
        envelope: bytes,
    ) -> IndependentEvaluationAttestation:
        """Authenticate a v2 evaluation after expiry for offline analysis only."""

        return self._verify_evaluation(binding, envelope, now=None)

    def _verify_evaluation(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float | None,
    ) -> IndependentEvaluationAttestation:
        canonical_envelope, statement_bytes, statement = _decode_envelope(
            envelope,
            expected_key_id=self.requirement.key_id,
        )
        if canonical_envelope != envelope:
            raise AttestationVerificationError(
                "Evaluation attestation envelope must use canonical JSON."
            )
        signature = _envelope_signature(canonical_envelope)
        envelope_document = json.loads(canonical_envelope)
        if (
            envelope_document["payload"]
            != base64.b64encode(statement_bytes).decode("ascii")
            or envelope_document["signatures"][0]["sig"]
            != base64.b64encode(signature).decode("ascii")
        ):
            raise AttestationVerificationError(
                "Evaluation attestation must use canonical base64."
            )
        try:
            self._public_key.verify(
                signature,
                _dsse_pae(DSSE_PAYLOAD_TYPE, statement_bytes),
            )
        except InvalidSignature as exc:
            raise AttestationVerificationError(
                "Independent evaluation signature is invalid."
            ) from exc
        (
            attestation_id,
            issued_at,
            expires_at,
            passed,
            checks,
        ) = _validate_evaluation_statement(
            statement,
            binding=binding,
            requirement=self.requirement,
            now=now,
        )
        return IndependentEvaluationAttestation(
            adapter_id=self.requirement.adapter_id,
            verifier_id=self.requirement.verifier_id,
            key_id=self.requirement.key_id,
            attestation_id=attestation_id,
            statement_bytes=statement_bytes,
            envelope_bytes=canonical_envelope,
            evidence_sha256=sha256_bytes(canonical_envelope),
            issued_at=issued_at,
            expires_at=expires_at,
            passed=passed,
            checks=checks,
        )


class AttestationTrustStore:
    """Concrete registry of trusted verification-only adapters."""

    __slots__ = ("_verifiers",)

    def __init__(self, verifiers: Sequence[TrustedEd25519Verifier]) -> None:
        registry: dict[str, TrustedEd25519Verifier] = {}
        for verifier in verifiers:
            verifier_id = verifier.requirement.verifier_id
            if verifier_id in registry:
                raise AttestationVerificationError(
                    "Trust store repeats a verifier identity."
                )
            registry[verifier_id] = verifier
        if not registry:
            raise AttestationVerificationError("Trust store cannot be empty.")
        self._verifiers = MappingProxyType(registry)

    def verify(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float,
    ) -> IndependentAttestation:
        verifier_id = _peek_signed_verifier_id(envelope)
        try:
            requirement = binding.verification_policy.requirement(verifier_id)
            verifier = self._verifiers[verifier_id]
        except (KeyError, TwoPhaseContractError) as exc:
            raise AttestationVerificationError(
                "Attestation verifier is not trusted by this workflow."
            ) from exc
        if verifier.requirement != requirement:
            raise AttestationVerificationError(
                "Trust-store verifier does not match the signed workflow policy."
            )
        return verifier.verify(binding, envelope, now=now)

    def verify_evaluation(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float,
    ) -> IndependentEvaluationAttestation:
        verifier = self._evaluation_verifier(binding, envelope)
        return verifier.verify_evaluation(binding, envelope, now=now)

    def verify_historical_evaluation(
        self,
        binding: CandidateBinding,
        envelope: bytes,
    ) -> IndependentEvaluationAttestation:
        verifier = self._evaluation_verifier(binding, envelope)
        return verifier.verify_historical_evaluation(binding, envelope)

    def _evaluation_verifier(
        self,
        binding: CandidateBinding,
        envelope: bytes,
    ) -> TrustedEd25519Verifier:
        verifier_id = _peek_signed_verifier_id(envelope)
        try:
            requirement = binding.verification_policy.requirement(verifier_id)
            verifier = self._verifiers[verifier_id]
        except (KeyError, TwoPhaseContractError) as exc:
            raise AttestationVerificationError(
                "Evaluation verifier is not trusted by this workflow."
            ) from exc
        if verifier.requirement != requirement:
            raise AttestationVerificationError(
                "Trust-store verifier does not match the signed workflow policy."
            )
        return verifier


def create_ed25519_dsse_envelope(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    private_key: Ed25519PrivateKey,
    *,
    attestation_id: str,
    issued_at: float,
    expires_at: float,
    checks: Sequence[AttestationCheck],
) -> bytes:
    """Create evidence in an independent verifier process.

    The bridge authority path only accepts the returned raw envelope and verifies
    it with an application-owned public-key trust store.
    """

    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationVerificationError("Signing key must be Ed25519.")
    if ed25519_public_key_sha256(private_key.public_key()) != requirement.public_key_sha256:
        raise AttestationVerificationError(
            "Signing key does not match the workflow verifier requirement."
        )
    statement = build_attestation_statement(
        binding,
        requirement,
        attestation_id=attestation_id,
        issued_at=issued_at,
        expires_at=expires_at,
        checks=checks,
    )
    payload = canonical_json_bytes(statement)
    signature = private_key.sign(_dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    envelope = {
        "payloadType": DSSE_PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [
            {
                "keyid": requirement.key_id,
                "sig": base64.b64encode(signature).decode("ascii"),
            }
        ],
    }
    return canonical_json_bytes(envelope)


def create_ed25519_evaluation_dsse_envelope(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    private_key: Ed25519PrivateKey,
    *,
    attestation_id: str,
    issued_at: float,
    expires_at: float,
    checks: Sequence[AttestationCheck],
) -> bytes:
    """Create signed v2 pass-or-fail evidence in an independent evaluator."""

    if not isinstance(private_key, Ed25519PrivateKey):
        raise AttestationVerificationError("Signing key must be Ed25519.")
    if (
        ed25519_public_key_sha256(private_key.public_key())
        != requirement.public_key_sha256
    ):
        raise AttestationVerificationError(
            "Signing key does not match the workflow verifier requirement."
        )
    statement = build_evaluation_attestation_statement(
        binding,
        requirement,
        attestation_id=attestation_id,
        issued_at=issued_at,
        expires_at=expires_at,
        checks=checks,
    )
    payload = canonical_json_bytes(statement)
    signature = private_key.sign(_dsse_pae(DSSE_PAYLOAD_TYPE, payload))
    envelope = {
        "payloadType": DSSE_PAYLOAD_TYPE,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [
            {
                "keyid": requirement.key_id,
                "sig": base64.b64encode(signature).decode("ascii"),
            }
        ],
    }
    return canonical_json_bytes(envelope)


def _decode_envelope(
    envelope: bytes,
    *,
    expected_key_id: str,
) -> tuple[bytes, bytes, dict[str, Any]]:
    if (
        not isinstance(envelope, bytes)
        or not envelope
        or len(envelope) > _MAX_ENVELOPE_BYTES
    ):
        raise AttestationVerificationError("Attestation envelope size is invalid.")
    try:
        decoded = json.loads(envelope)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationVerificationError("Attestation envelope is not JSON.") from exc
    if not isinstance(decoded, dict) or set(decoded) != {
        "payloadType",
        "payload",
        "signatures",
    }:
        raise AttestationVerificationError("Attestation envelope shape is invalid.")
    if decoded.get("payloadType") != DSSE_PAYLOAD_TYPE:
        raise AttestationVerificationError("Attestation payload type is invalid.")
    signatures = decoded.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise AttestationVerificationError(
            "Attestation must contain exactly one verifier signature."
        )
    signature = signatures[0]
    if (
        not isinstance(signature, dict)
        or set(signature) != {"keyid", "sig"}
        or signature.get("keyid") != expected_key_id
        or not isinstance(signature.get("sig"), str)
    ):
        raise AttestationVerificationError("Attestation signature descriptor is invalid.")
    payload_text = decoded.get("payload")
    if not isinstance(payload_text, str):
        raise AttestationVerificationError("Attestation payload encoding is invalid.")
    payload = _decode_base64(payload_text, "payload")
    try:
        statement = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationVerificationError("Attestation statement is not JSON.") from exc
    if not isinstance(statement, dict) or canonical_json_bytes(statement) != payload:
        raise AttestationVerificationError(
            "Attestation statement must use RFC 8785 canonical JSON."
        )
    canonical_envelope = canonical_json_bytes(decoded)
    return canonical_envelope, payload, statement


def _envelope_signature(canonical_envelope: bytes) -> bytes:
    decoded = json.loads(canonical_envelope)
    return _decode_base64(decoded["signatures"][0]["sig"], "signature")


def _peek_signed_verifier_id(envelope: bytes) -> str:
    # This only selects a candidate public key. Full shape, signature, and binding
    # validation happens before the result can become authority.
    if not isinstance(envelope, bytes) or len(envelope) > _MAX_ENVELOPE_BYTES:
        raise AttestationVerificationError("Attestation envelope size is invalid.")
    try:
        decoded = json.loads(envelope)
        payload_text = decoded["payload"]
        payload = _decode_base64(payload_text, "payload")
        statement = json.loads(payload)
        verifier_id = statement["predicate"]["attestation"]["verifierId"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationVerificationError(
            "Attestation verifier identity is unavailable."
        ) from exc
    try:
        return require_safe_id(verifier_id, "attestation verifier_id")
    except TwoPhaseContractError as exc:
        raise AttestationVerificationError(str(exc)) from exc


def _validate_statement(
    statement: Mapping[str, Any],
    *,
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    now: float,
) -> tuple[str, float, float]:
    if set(statement) != {"_type", "subject", "predicateType", "predicate"}:
        raise AttestationVerificationError("Attestation statement shape is invalid.")
    expected_subject = [
        {
            "name": f"urn:mymoe:candidate:{binding.workflow_id}",
            "digest": {"sha256": binding.candidate_fingerprint},
        }
    ]
    if (
        statement.get("_type") != IN_TOTO_STATEMENT_V1
        or statement.get("predicateType") != INDEPENDENT_PREDICATE_V1
        or statement.get("subject") != expected_subject
    ):
        raise AttestationVerificationError("Attestation subject binding is invalid.")
    predicate = statement.get("predicate")
    if not isinstance(predicate, Mapping) or set(predicate) != {
        "schemaVersion",
        "binding",
        "bindingSha256",
        "attestation",
        "outcome",
    }:
        raise AttestationVerificationError("Attestation predicate shape is invalid.")
    if (
        predicate.get("schemaVersion") != TWO_PHASE_SCHEMA_VERSION
        or predicate.get("binding") != binding.payload()
        or predicate.get("bindingSha256") != binding.binding_sha256
    ):
        raise AttestationVerificationError("Attestation workflow binding is invalid.")
    metadata = predicate.get("attestation")
    expected_metadata_fields = {
        "attestationId",
        "verifierId",
        "adapterId",
        "keyId",
        "publicKeySha256",
        "specSha256",
        "trustPolicySha256",
        "issuedAt",
        "expiresAt",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_fields:
        raise AttestationVerificationError("Signed attestation metadata is invalid.")
    expected_identity = {
        "verifierId": requirement.verifier_id,
        "adapterId": requirement.adapter_id,
        "keyId": requirement.key_id,
        "publicKeySha256": requirement.public_key_sha256,
        "specSha256": requirement.spec_sha256,
        "trustPolicySha256": binding.verification_policy.policy_sha256,
    }
    if any(metadata.get(name) != value for name, value in expected_identity.items()):
        raise AttestationVerificationError(
            "Signed verifier identity or trust policy is invalid."
        )
    try:
        attestation_id = require_safe_id(
            metadata.get("attestationId"), "attestation_id"
        )
    except TwoPhaseContractError as exc:
        raise AttestationVerificationError(str(exc)) from exc
    issued_at = _timestamp(metadata.get("issuedAt"))
    expires_at = _timestamp(metadata.get("expiresAt"))
    if (
        issued_at >= expires_at
        or issued_at < binding.created_at
        or expires_at > binding.expires_at
        or not issued_at <= now <= expires_at
    ):
        raise AttestationVerificationError(
            "Independent attestation is not currently valid."
        )
    outcome = predicate.get("outcome")
    if not isinstance(outcome, Mapping) or set(outcome) != {"passed", "checks"}:
        raise AttestationVerificationError("Attestation outcome shape is invalid.")
    checks = outcome.get("checks")
    if outcome.get("passed") is not True or not isinstance(checks, list) or not checks:
        raise AttestationVerificationError("Attestation outcome did not pass.")
    check_ids: list[str] = []
    for raw in checks:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "passed",
            "evidenceSha256",
        }:
            raise AttestationVerificationError("Attestation check shape is invalid.")
        try:
            check_id = require_safe_id(raw.get("id"), "attestation check_id")
            require_sha256(
                raw.get("evidenceSha256"), "attestation check evidence_sha256"
            )
        except TwoPhaseContractError as exc:
            raise AttestationVerificationError(str(exc)) from exc
        if raw.get("passed") is not True:
            raise AttestationVerificationError("Attestation contains a failed check.")
        check_ids.append(check_id)
    if check_ids != sorted(check_ids) or len(check_ids) != len(set(check_ids)):
        raise AttestationVerificationError(
            "Attestation checks must be ordered and unique."
        )
    expected = build_attestation_statement(
        binding,
        requirement,
        attestation_id=attestation_id,
        issued_at=issued_at,
        expires_at=expires_at,
        checks=tuple(
            AttestationCheck(
                check_id=raw["id"],
                passed=raw["passed"],
                evidence_sha256=raw["evidenceSha256"],
            )
            for raw in checks
        ),
    )
    if statement != expected:
        raise AttestationVerificationError("Attestation statement is not canonical.")
    return attestation_id, issued_at, expires_at


def _validate_evaluation_statement(
    statement: Mapping[str, Any],
    *,
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    now: float | None,
) -> tuple[str, float, float, bool, tuple[AttestationCheck, ...]]:
    if set(statement) != {"_type", "subject", "predicateType", "predicate"}:
        raise AttestationVerificationError(
            "Evaluation attestation statement shape is invalid."
        )
    expected_subject = [
        {
            "name": f"urn:mymoe:candidate:{binding.workflow_id}",
            "digest": {"sha256": binding.candidate_fingerprint},
        }
    ]
    if (
        statement.get("_type") != IN_TOTO_STATEMENT_V1
        or statement.get("predicateType") != INDEPENDENT_EVALUATION_PREDICATE_V2
        or statement.get("subject") != expected_subject
    ):
        raise AttestationVerificationError(
            "Evaluation attestation subject binding is invalid."
        )
    predicate = statement.get("predicate")
    if not isinstance(predicate, Mapping) or set(predicate) != {
        "schemaVersion",
        "binding",
        "bindingSha256",
        "attestation",
        "outcome",
    }:
        raise AttestationVerificationError(
            "Evaluation attestation predicate shape is invalid."
        )
    if (
        predicate.get("schemaVersion") != TWO_PHASE_SCHEMA_VERSION
        or predicate.get("binding") != binding.payload()
        or predicate.get("bindingSha256") != binding.binding_sha256
    ):
        raise AttestationVerificationError(
            "Evaluation attestation workflow binding is invalid."
        )
    try:
        policy_requirement = binding.verification_policy.requirement(
            requirement.verifier_id
        )
    except TwoPhaseContractError as exc:
        raise AttestationVerificationError(
            "Evaluation verifier is absent from the signed workflow policy."
        ) from exc
    if policy_requirement != requirement:
        raise AttestationVerificationError(
            "Evaluation verifier does not match the signed workflow policy."
        )
    metadata = predicate.get("attestation")
    expected_metadata_fields = {
        "attestationId",
        "verifierId",
        "adapterId",
        "keyId",
        "publicKeySha256",
        "specSha256",
        "trustPolicySha256",
        "issuedAt",
        "expiresAt",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_fields:
        raise AttestationVerificationError(
            "Signed evaluation attestation metadata is invalid."
        )
    expected_identity = {
        "verifierId": requirement.verifier_id,
        "adapterId": requirement.adapter_id,
        "keyId": requirement.key_id,
        "publicKeySha256": requirement.public_key_sha256,
        "specSha256": requirement.spec_sha256,
        "trustPolicySha256": binding.verification_policy.policy_sha256,
    }
    if any(metadata.get(name) != value for name, value in expected_identity.items()):
        raise AttestationVerificationError(
            "Signed evaluation identity, spec, or trust policy is invalid."
        )
    try:
        attestation_id = require_safe_id(
            metadata.get("attestationId"),
            "attestation_id",
        )
    except TwoPhaseContractError as exc:
        raise AttestationVerificationError(str(exc)) from exc
    issued_at = _timestamp(metadata.get("issuedAt"))
    expires_at = _timestamp(metadata.get("expiresAt"))
    if (
        issued_at >= expires_at
        or issued_at < binding.created_at
        or expires_at > binding.expires_at
    ):
        raise AttestationVerificationError(
            "Independent evaluation lifetime is outside the workflow."
        )
    if now is not None and not issued_at <= now <= expires_at:
        raise AttestationVerificationError(
            "Independent evaluation is not currently valid."
        )
    outcome = predicate.get("outcome")
    if not isinstance(outcome, Mapping) or set(outcome) != {"passed", "checks"}:
        raise AttestationVerificationError(
            "Evaluation attestation outcome shape is invalid."
        )
    passed = outcome.get("passed")
    raw_checks = outcome.get("checks")
    if not isinstance(passed, bool) or not isinstance(raw_checks, list) or not raw_checks:
        raise AttestationVerificationError(
            "Evaluation attestation outcome is invalid."
        )
    checks: list[AttestationCheck] = []
    for raw in raw_checks:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "passed",
            "evidenceSha256",
        }:
            raise AttestationVerificationError(
                "Evaluation attestation check shape is invalid."
            )
        if not isinstance(raw.get("passed"), bool):
            raise AttestationVerificationError(
                "Evaluation attestation check result is invalid."
            )
        try:
            check = AttestationCheck(
                check_id=require_safe_id(
                    raw.get("id"),
                    "attestation check_id",
                ),
                passed=raw["passed"],
                evidence_sha256=require_sha256(
                    raw.get("evidenceSha256"),
                    "attestation check evidence_sha256",
                ),
            )
        except TwoPhaseContractError as exc:
            raise AttestationVerificationError(str(exc)) from exc
        checks.append(check)
    check_ids = tuple(item.check_id for item in checks)
    if check_ids != tuple(sorted(check_ids)) or len(check_ids) != len(
        set(check_ids)
    ):
        raise AttestationVerificationError(
            "Evaluation attestation checks must be ordered and unique."
        )
    if passed is not all(item.passed for item in checks):
        raise AttestationVerificationError(
            "Evaluation outcome does not match its signed checks."
        )
    expected = build_evaluation_attestation_statement(
        binding,
        requirement,
        attestation_id=attestation_id,
        issued_at=issued_at,
        expires_at=expires_at,
        checks=tuple(checks),
    )
    if statement != expected:
        raise AttestationVerificationError(
            "Evaluation attestation statement is not canonical."
        )
    return attestation_id, issued_at, expires_at, passed, tuple(checks)


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


def _decode_base64(value: str, label: str) -> bytes:
    if not isinstance(value, str):
        raise AttestationVerificationError(f"Attestation {label} is invalid.")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise AttestationVerificationError(
            f"Attestation {label} is not strict base64."
        ) from exc


def _timestamp(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise AttestationVerificationError("Attestation timestamp is invalid.")
    return float(value)
