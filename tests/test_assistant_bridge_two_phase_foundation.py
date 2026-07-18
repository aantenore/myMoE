from __future__ import annotations

import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import (
    AttestationTrustStore,
    AttestationVerificationError,
    ED25519_DSSE_ADAPTER_ID,
    TrustedEd25519Verifier,
    create_ed25519_dsse_envelope,
    ed25519_public_key_sha256,
)
from local_moe.assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from local_moe.assistant_bridge_integrity import (
    canonical_json_bytes,
    sha256_bytes,
)
from local_moe.assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    AttestationCheck,
    CandidateBinding,
    VerificationPolicy,
    VerifierRequirement,
)
from local_moe.assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowStoreError,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
STAGE_KEY = "stage-operation-0000000000000001"
RESUME_KEY = "resume-operation-000000000000001"


class TwoPhaseFoundationTests(unittest.TestCase):
    def test_cas_uses_rfc8785_and_detects_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ContentAddressedStore(Path(temporary) / "cas")
            descriptor = store.put_json(
                {"z": 1, "a": "value"},
                media_type="application/vnd.mymoe.test+json",
            )

            self.assertEqual(
                store.get_bytes(descriptor),
                canonical_json_bytes({"a": "value", "z": 1}),
            )
            object_path = (
                store.root
                / "objects"
                / "sha256"
                / descriptor.sha256[:2]
                / descriptor.sha256[2:]
            )
            object_path.write_bytes(b"changed")

            with self.assertRaisesRegex(
                ContentAddressedStoreError,
                "size binding|digest binding",
            ):
                store.get_bytes(descriptor)

    @unittest.skipIf(not hasattr(os, "symlink"), "symbolic links unavailable")
    def test_cas_rejects_linked_object_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ContentAddressedStore(root / "cas")
            descriptor = ArtifactDescriptor(
                media_type="application/octet-stream",
                sha256=sha256_bytes(b"candidate"),
                size_bytes=len(b"candidate"),
            )
            prefix = store.root / "objects" / "sha256" / descriptor.sha256[:2]
            prefix.mkdir(mode=0o700)
            target = root / "peer.bin"
            target.write_bytes(b"candidate")
            os.symlink(target, prefix / descriptor.sha256[2:])

            with self.assertRaisesRegex(ContentAddressedStoreError, "regular file"):
                store.get_bytes(descriptor)

    def test_store_candidate_binds_ordered_manifest_changeset_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            content = b"final candidate\n"
            (candidate / "src").mkdir()
            (candidate / "src" / "module.py").write_bytes(content)
            store = ContentAddressedStore(root / "cas")
            after = _file("src/module.py", content)

            manifest, changeset = store.store_candidate(
                candidate,
                (after,),
                ({"path": "src/module.py", "before": None, "after": after},),
                source_fingerprint=DIGEST_A,
            )

            manifest_payload = store.get_json(manifest)
            changeset_payload = store.get_json(changeset)
            self.assertNotIn("candidateFingerprint", manifest_payload)
            self.assertEqual(manifest_payload["changeset"], changeset.payload())
            self.assertEqual(changeset_payload["changes"][0]["after"], after)
            with store.materialize_candidate(manifest) as materialized:
                self.assertEqual(
                    (materialized / "src" / "module.py").read_bytes(),
                    content,
                )

    def test_cas_rejects_incoherent_size_mode_order_and_change_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "a.txt").write_text("a", encoding="utf-8")
            store = ContentAddressedStore(root / "cas")
            valid = _file("a.txt", b"a")

            for mutation in (
                {**valid, "size": True},
                {**valid, "mode": 0o1000},
                {**valid, "direction": "output_only"},
            ):
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ContentAddressedStoreError):
                        store.store_candidate(
                            candidate,
                            (mutation,),
                            (),
                            source_fingerprint=DIGEST_A,
                        )
            with self.assertRaisesRegex(ContentAddressedStoreError, "path binding"):
                store.store_candidate(
                    candidate,
                    (valid,),
                    (
                        {
                            "path": "other.txt",
                            "before": None,
                            "after": valid,
                        },
                    ),
                    source_fingerprint=DIGEST_A,
                )

    def test_candidate_content_and_subject_identity_exclude_fresh_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "state.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            first, _ = _binding(root / "first", store, policy, stage_key=STAGE_KEY)
            second = replace(
                first,
                workflow_id="wf-different",
                stage_idempotency_sha256=DIGEST_C,
                challenge_sha256=DIGEST_D,
            )

            self.assertEqual(
                first.candidate_content_sha256,
                second.candidate_content_sha256,
            )
            self.assertEqual(first.candidate_fingerprint, second.candidate_fingerprint)
            self.assertNotEqual(first.binding_sha256, second.binding_sha256)
            self.assertIsNotNone(private_key)

    def test_real_dsse_adapter_verifies_signed_complete_predicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "state.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, _ = _binding(root / "binding", store, policy)
            envelope = _envelope(binding, requirement, private_key)
            trust = AttestationTrustStore(
                (TrustedEd25519Verifier(requirement, private_key.public_key()),)
            )

            attestation = trust.verify(binding, envelope, now=110)

            self.assertEqual(attestation.evidence_sha256, sha256_bytes(envelope))
            self.assertEqual(
                attestation.statement()["predicate"]["bindingSha256"],
                binding.binding_sha256,
            )
            self.assertEqual(
                attestation.statement()["predicate"]["attestation"]["specSha256"],
                requirement.spec_sha256,
            )

    def test_dsse_adapter_rejects_resigned_spec_expiry_and_outcome_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "state.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, _ = _binding(root / "binding", store, policy)
            verifier = TrustedEd25519Verifier(requirement, private_key.public_key())
            envelope = _envelope(binding, requirement, private_key)

            mutations = (
                lambda statement: statement["predicate"]["attestation"].__setitem__(
                    "specSha256", DIGEST_E
                ),
                lambda statement: statement["predicate"]["attestation"].__setitem__(
                    "expiresAt", 201
                ),
                lambda statement: statement["predicate"]["outcome"].__setitem__(
                    "passed", False
                ),
            )
            for mutate in mutations:
                with self.subTest(mutate=mutate):
                    changed = _resign(envelope, private_key, mutate)
                    with self.assertRaises(AttestationVerificationError):
                        verifier.verify(binding, changed, now=110)

    def test_workflow_store_enforces_quorum_and_raw_envelope_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "state" / "workflows.sqlite3")
            first_key, first_requirement = _verifier("verifier-a")
            second_key, second_requirement = _verifier("verifier-b")
            policy = VerificationPolicy(
                "policy-v1", 2, (first_requirement, second_requirement)
            )
            binding, challenge = _binding(root / "binding", store, policy)
            staged, replay = store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=100,
            )
            trust = AttestationTrustStore(
                (
                    TrustedEd25519Verifier(
                        first_requirement, first_key.public_key()
                    ),
                    TrustedEd25519Verifier(
                        second_requirement, second_key.public_key()
                    ),
                )
            )

            partial, partial_replay = store.record_attestation_envelope(
                binding.workflow_id,
                _envelope(binding, first_requirement, first_key, attestation_id="att-a"),
                trust_store=trust,
                now=110,
            )
            ready, ready_replay = store.record_attestation_envelope(
                binding.workflow_id,
                _envelope(binding, second_requirement, second_key, attestation_id="att-b"),
                trust_store=trust,
                now=111,
            )

            self.assertEqual(staged.status, "staged")
            self.assertFalse(replay)
            self.assertEqual(partial.status, "attested")
            self.assertFalse(partial_replay)
            self.assertEqual(ready.status, "ready")
            self.assertFalse(ready_replay)
            self.assertTrue(ready.quorum_satisfied)

    def test_stage_idempotency_creation_time_and_database_identity_are_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "state" / "workflows.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, challenge = _binding(root / "binding", store, policy)
            first, first_replay = store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=100,
            )
            second, second_replay = store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=101,
            )

            self.assertFalse(first_replay)
            self.assertTrue(second_replay)
            self.assertEqual(first.binding, second.binding)
            with self.assertRaisesRegex(WorkflowStoreError, "not currently valid"):
                other_store = SQLiteWorkflowStore(root / "other.sqlite3")
                other_binding, other_challenge = _binding(
                    root / "other", other_store, policy, created_at=120
                )
                other_store.create_workflow(
                    other_binding,
                    challenge=other_challenge,
                    stage_idempotency_key=STAGE_KEY,
                    workspace_root_sha256=DIGEST_E,
                    now=119,
                )
            replacement = root / "replacement.sqlite3"
            replacement.write_bytes(store.path.read_bytes())
            replacement.chmod(0o600)
            store.path.unlink()
            replacement.rename(store.path)
            with self.assertRaisesRegex(WorkflowStoreError, "identity changed"):
                store.get_workflow(binding.workflow_id, now=102)
            self.assertIsNotNone(private_key)

    def test_resume_is_idempotent_and_persists_transaction_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "workflows.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, challenge = _binding(root / "binding", store, policy)
            store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=100,
            )
            trust = AttestationTrustStore(
                (TrustedEd25519Verifier(requirement, private_key.public_key()),)
            )
            envelope = _envelope(binding, requirement, private_key)
            store.record_attestation_envelope(
                binding.workflow_id, envelope, trust_store=trust, now=110
            )
            first_plan = store.issue_resume_plan(
                binding.workflow_id,
                idempotency_key=RESUME_KEY,
                ttl_seconds=30,
                now=111,
            )
            repeated_plan = store.issue_resume_plan(
                binding.workflow_id,
                idempotency_key=RESUME_KEY,
                ttl_seconds=30,
                now=112,
            )

            self.assertEqual(first_plan.confirmation_id, repeated_plan.confirmation_id)
            self.assertEqual(first_plan.plan_id, repeated_plan.plan_id)
            self.assertTrue(repeated_plan.idempotent_replay)
            applying, replay = store.consume_resume_confirmation(
                binding.workflow_id,
                plan_id=first_plan.plan_id,
                confirmation_id=first_plan.confirmation_id,
                binding_sha256=first_plan.binding_sha256,
                now=113,
            )
            retry, retry_replay = store.consume_resume_confirmation(
                binding.workflow_id,
                plan_id=first_plan.plan_id,
                confirmation_id=first_plan.confirmation_id,
                binding_sha256=first_plan.binding_sha256,
                now=114,
            )

            self.assertEqual(applying.status, "applying")
            self.assertFalse(replay)
            self.assertEqual(len(applying.apply_transaction_id), 64)
            self.assertEqual(retry.apply_transaction_id, applying.apply_transaction_id)
            self.assertTrue(retry_replay)
            applied, applied_replay = store.mark_applied(
                binding.workflow_id,
                transaction_id=applying.apply_transaction_id,
                result_sha256=DIGEST_C,
                now=115,
            )
            repeated, confirmation_replay = store.consume_resume_confirmation(
                binding.workflow_id,
                plan_id=first_plan.plan_id,
                confirmation_id=first_plan.confirmation_id,
                binding_sha256=first_plan.binding_sha256,
                now=116,
            )
            self.assertEqual(applied.status, "applied")
            self.assertFalse(applied_replay)
            self.assertEqual(repeated, applied)
            self.assertTrue(confirmation_replay)

    def test_resume_requires_current_attestation_quorum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "workflows.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, challenge = _binding(root / "binding", store, policy)
            store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=100,
            )
            trust = AttestationTrustStore(
                (TrustedEd25519Verifier(requirement, private_key.public_key()),)
            )
            envelope = _envelope(
                binding,
                requirement,
                private_key,
                expires_at=120,
            )
            store.record_attestation_envelope(
                binding.workflow_id, envelope, trust_store=trust, now=110
            )

            with self.assertRaisesRegex(WorkflowStoreError, "verified ready"):
                store.issue_resume_plan(
                    binding.workflow_id,
                    idempotency_key=RESUME_KEY,
                    ttl_seconds=30,
                    now=121,
                )

    def test_persisted_binding_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SQLiteWorkflowStore(root / "workflows.sqlite3")
            private_key, requirement = _verifier("verifier-a")
            policy = VerificationPolicy("policy-v1", 1, (requirement,))
            binding, challenge = _binding(root / "binding", store, policy)
            store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=STAGE_KEY,
                workspace_root_sha256=DIGEST_E,
                now=100,
            )
            with sqlite3.connect(store.path) as connection:
                connection.execute(
                    "UPDATE workflows SET binding_sha256 = ? WHERE workflow_id = ?",
                    (DIGEST_A, binding.workflow_id),
                )

            with self.assertRaisesRegex(WorkflowStoreError, "tampered"):
                store.get_workflow(binding.workflow_id, now=101)
            self.assertIsNotNone(private_key)


def _file(path: str, value: bytes) -> dict[str, object]:
    return {
        "path": path,
        "kind": "file",
        "sha256": sha256_bytes(value),
        "size": len(value),
        "mode": 0o644,
        "direction": "round_trip",
    }


def _verifier(name: str) -> tuple[Ed25519PrivateKey, VerifierRequirement]:
    private_key = Ed25519PrivateKey.generate()
    requirement = VerifierRequirement(
        verifier_id=name,
        adapter_id=ED25519_DSSE_ADAPTER_ID,
        key_id=f"{name}-key",
        public_key_sha256=ed25519_public_key_sha256(private_key.public_key()),
        spec_sha256=sha256_bytes(f"{name}-spec".encode("utf-8")),
    )
    return private_key, requirement


def _binding(
    root: Path,
    workflow_store: SQLiteWorkflowStore,
    policy: VerificationPolicy,
    *,
    stage_key: str = STAGE_KEY,
    created_at: float = 100,
    expires_at: float = 200,
) -> tuple[CandidateBinding, str]:
    root.mkdir(parents=True, exist_ok=True)
    cas = ContentAddressedStore(root / "cas")
    candidate = root / "candidate"
    candidate.mkdir()
    manifest, changeset = cas.store_candidate(
        candidate,
        (),
        (),
        source_fingerprint=DIGEST_D,
    )
    workflow_id, challenge, stage_sha256 = workflow_store.stage_identity(stage_key)
    return (
        CandidateBinding(
            workflow_id=workflow_id,
            stage_idempotency_sha256=stage_sha256,
            task_fingerprint=DIGEST_A,
            config_sha256=DIGEST_B,
            source_fingerprint=DIGEST_D,
            challenge_sha256=sha256_bytes(challenge.encode("utf-8")),
            manifest=manifest,
            changeset=changeset,
            verification_policy=policy,
            created_at=created_at,
            expires_at=expires_at,
        ),
        challenge,
    )


def _envelope(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    private_key: Ed25519PrivateKey,
    *,
    attestation_id: str = "attestation-1",
    issued_at: float = 105,
    expires_at: float = 150,
) -> bytes:
    return create_ed25519_dsse_envelope(
        binding,
        requirement,
        private_key,
        attestation_id=attestation_id,
        issued_at=issued_at,
        expires_at=expires_at,
        checks=(
            AttestationCheck(
                check_id="tests",
                passed=True,
                evidence_sha256=DIGEST_C,
            ),
        ),
    )


def _resign(
    envelope: bytes,
    private_key: Ed25519PrivateKey,
    mutate: object,
) -> bytes:
    decoded = json.loads(envelope)
    statement = json.loads(base64.b64decode(decoded["payload"], validate=True))
    mutate(statement)
    payload = canonical_json_bytes(statement)
    payload_type = decoded["payloadType"]
    encoded_type = payload_type.encode("utf-8")
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
