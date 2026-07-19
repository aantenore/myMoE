from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import (
    ED25519_DSSE_ADAPTER_ID,
    AttestationTrustStore,
    AttestationVerificationError,
    TrustedEd25519Verifier,
    create_ed25519_dsse_envelope,
    create_ed25519_evaluation_dsse_envelope,
    ed25519_public_key_sha256,
)
from local_moe.assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from local_moe.assistant_bridge_two_phase_contracts import (
    DSSE_PAYLOAD_TYPE,
    INDEPENDENT_EVALUATION_PREDICATE_V2,
    ArtifactDescriptor,
    AttestationCheck,
    CandidateBinding,
    IndependentAttestation,
    IndependentEvaluationAttestation,
    TwoPhaseContractError,
    VerificationPolicy,
    VerifierRequirement,
)


class IndependentEvaluationAttestationTests(unittest.TestCase):
    def test_real_ed25519_live_verification_authenticates_pass_and_fail(self) -> None:
        binding, requirement, private_key, trust = _fixture()
        cases = (
            (
                True,
                (
                    AttestationCheck("format", True, "1" * 64),
                    AttestationCheck("tests", True, "2" * 64),
                ),
            ),
            (
                False,
                (
                    AttestationCheck("format", True, "1" * 64),
                    AttestationCheck("tests", False, "2" * 64),
                ),
            ),
        )
        for expected_passed, checks in cases:
            with self.subTest(passed=expected_passed):
                envelope = _evaluation_envelope(
                    binding,
                    requirement,
                    private_key,
                    checks=tuple(reversed(checks)),
                )

                verified = trust.verify_evaluation(binding, envelope, now=120)

                self.assertIsInstance(verified, IndependentEvaluationAttestation)
                self.assertNotIsInstance(verified, IndependentAttestation)
                self.assertIs(verified.passed, expected_passed)
                self.assertIsInstance(verified.checks, tuple)
                self.assertEqual(verified.checks, checks)
                self.assertEqual(verified.evidence_sha256, sha256_bytes(envelope))
                self.assertEqual(
                    verified.statement()["predicateType"],
                    INDEPENDENT_EVALUATION_PREDICATE_V2,
                )
                self.assertEqual(
                    verified.metadata_payload()["passed"],
                    expected_passed,
                )

    def test_live_and_historical_apis_are_distinct_and_v1_remains_pass_only(
        self,
    ) -> None:
        binding, requirement, private_key, trust = _fixture()
        failed = _evaluation_envelope(
            binding,
            requirement,
            private_key,
            checks=(AttestationCheck("tests", False, "3" * 64),),
            expires_at=121,
        )

        with self.assertRaisesRegex(
            AttestationVerificationError,
            "not currently valid",
        ):
            trust.verify_evaluation(binding, failed, now=122)
        historical = trust.verify_historical_evaluation(binding, failed)
        self.assertFalse(historical.passed)
        self.assertEqual(historical.expires_at, 121)

        with self.assertRaises(AttestationVerificationError):
            trust.verify(binding, failed, now=120)

        v1 = create_ed25519_dsse_envelope(
            binding,
            requirement,
            private_key,
            attestation_id="v1-pass",
            issued_at=110,
            expires_at=130,
            checks=(AttestationCheck("tests", True, "4" * 64),),
        )
        self.assertIsInstance(
            trust.verify(binding, v1, now=120),
            IndependentAttestation,
        )
        with self.assertRaises(AttestationVerificationError):
            trust.verify_evaluation(binding, v1, now=120)
        with self.assertRaises(AttestationVerificationError):
            trust.verify_historical_evaluation(binding, v1)

    def test_authentically_resigned_shape_binding_policy_order_and_lifetime_tampering_fails(
        self,
    ) -> None:
        binding, requirement, private_key, trust = _fixture()
        envelope = _evaluation_envelope(
            binding,
            requirement,
            private_key,
            checks=(
                AttestationCheck("alpha", True, "5" * 64),
                AttestationCheck("beta", False, "6" * 64),
            ),
        )
        mutations = (
            lambda value: value["predicate"].__setitem__("extra", True),
            lambda value: value["predicate"].__setitem__(
                "bindingSha256",
                "7" * 64,
            ),
            lambda value: value["predicate"]["attestation"].__setitem__(
                "specSha256",
                "8" * 64,
            ),
            lambda value: value["predicate"]["attestation"].__setitem__(
                "trustPolicySha256",
                "9" * 64,
            ),
            lambda value: value["predicate"]["outcome"]["checks"].reverse(),
            lambda value: value["predicate"]["outcome"].__setitem__(
                "passed",
                True,
            ),
            lambda value: value["predicate"]["attestation"].__setitem__(
                "issuedAt",
                99,
            ),
            lambda value: value.__setitem__(
                "predicateType",
                "https://example.invalid/evaluation/v1",
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = _resign(envelope, private_key, mutate)
                with self.assertRaises(AttestationVerificationError):
                    trust.verify_evaluation(binding, changed, now=120)
                with self.assertRaises(AttestationVerificationError):
                    trust.verify_historical_evaluation(binding, changed)

    def test_signature_canonical_envelope_and_exact_policy_are_required(self) -> None:
        binding, requirement, private_key, trust = _fixture()
        envelope = _evaluation_envelope(binding, requirement, private_key)
        decoded = json.loads(envelope)
        statement = json.loads(base64.b64decode(decoded["payload"]))
        statement["predicate"]["outcome"]["checks"][0]["passed"] = False
        decoded["payload"] = base64.b64encode(
            canonical_json_bytes(statement)
        ).decode("ascii")
        forged = canonical_json_bytes(decoded)
        with self.assertRaisesRegex(
            AttestationVerificationError,
            "signature is invalid",
        ):
            trust.verify_evaluation(binding, forged, now=120)

        pretty = json.dumps(json.loads(envelope), indent=2).encode("utf-8")
        with self.assertRaisesRegex(
            AttestationVerificationError,
            "canonical JSON",
        ):
            trust.verify_evaluation(binding, pretty, now=120)

        outsider = replace(
            requirement,
            verifier_id="outsider",
            key_id="outsider-key",
        )
        outsider_envelope = _evaluation_envelope(
            binding,
            outsider,
            private_key,
        )
        outsider_verifier = TrustedEd25519Verifier(
            outsider,
            private_key.public_key(),
        )
        with self.assertRaisesRegex(
            AttestationVerificationError,
            "absent from the signed workflow policy",
        ):
            outsider_verifier.verify_evaluation(
                binding,
                outsider_envelope,
                now=120,
            )

    def test_verified_evaluation_result_and_checks_are_immutable(self) -> None:
        binding, requirement, private_key, trust = _fixture()
        verified = trust.verify_evaluation(
            binding,
            _evaluation_envelope(binding, requirement, private_key),
            now=120,
        )

        with self.assertRaises(FrozenInstanceError):
            verified.passed = False  # type: ignore[misc]
        with self.assertRaises(TwoPhaseContractError):
            replace(verified, checks=list(verified.checks))  # type: ignore[arg-type]
        with self.assertRaises(TwoPhaseContractError):
            replace(verified, passed=not verified.passed)


def _fixture() -> tuple[
    CandidateBinding,
    VerifierRequirement,
    Ed25519PrivateKey,
    AttestationTrustStore,
]:
    private_key = Ed25519PrivateKey.generate()
    requirement = VerifierRequirement(
        verifier_id="evaluation-tests",
        adapter_id=ED25519_DSSE_ADAPTER_ID,
        key_id="evaluation-tests-key",
        public_key_sha256=ed25519_public_key_sha256(private_key.public_key()),
        spec_sha256="a" * 64,
    )
    policy = VerificationPolicy("evaluation-policy", 1, (requirement,))
    binding = CandidateBinding(
        workflow_id="evaluation-workflow",
        stage_idempotency_sha256="b" * 64,
        task_fingerprint="c" * 64,
        config_sha256="d" * 64,
        source_fingerprint="e" * 64,
        challenge_sha256="f" * 64,
        manifest=ArtifactDescriptor("application/json", "1" * 64, 10),
        changeset=ArtifactDescriptor("application/json", "2" * 64, 10),
        verification_policy=policy,
        created_at=100,
        expires_at=200,
    )
    trust = AttestationTrustStore(
        (TrustedEd25519Verifier(requirement, private_key.public_key()),)
    )
    return binding, requirement, private_key, trust


def _evaluation_envelope(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    private_key: Ed25519PrivateKey,
    *,
    checks: tuple[AttestationCheck, ...] = (
        AttestationCheck("tests", True, "3" * 64),
    ),
    expires_at: float = 130,
) -> bytes:
    return create_ed25519_evaluation_dsse_envelope(
        binding,
        requirement,
        private_key,
        attestation_id="evaluation-attestation",
        issued_at=110,
        expires_at=expires_at,
        checks=checks,
    )


def _resign(envelope: bytes, private_key: Ed25519PrivateKey, mutate) -> bytes:
    decoded = json.loads(envelope)
    statement = json.loads(base64.b64decode(decoded["payload"]))
    mutate(statement)
    payload = canonical_json_bytes(statement)
    encoded_type = DSSE_PAYLOAD_TYPE.encode("utf-8")
    pae = b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )
    decoded["payload"] = base64.b64encode(payload).decode("ascii")
    decoded["signatures"][0]["sig"] = base64.b64encode(
        private_key.sign(pae)
    ).decode("ascii")
    return canonical_json_bytes(decoded)


if __name__ == "__main__":
    unittest.main()
