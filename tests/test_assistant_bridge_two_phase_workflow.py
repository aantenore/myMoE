from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import (
    AttestationTrustStore,
    ED25519_DSSE_ADAPTER_ID,
    TrustedEd25519Verifier,
    create_ed25519_dsse_envelope,
    ed25519_public_key_sha256,
)
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.assistant_bridge_integrity import sha256_bytes
from local_moe.assistant_bridge_two_phase import (
    TwoPhaseWorkflowConfig,
    TwoPhaseWorkflowError,
    TwoPhaseWorkflowService,
    candidate_workspace_snapshot_fingerprint,
)
from local_moe.assistant_bridge_two_phase_contracts import (
    AttestationCheck,
    VerificationPolicy,
    VerifierRequirement,
)
from local_moe.assistant_bridge_workflow_store import SQLiteWorkflowStore
from local_moe.assistant_bridge_workspace import (
    WorkspaceScopePolicy,
    apply_changeset as real_apply_changeset,
    snapshot_workspace,
)


TASK_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
CHECK_SHA256 = "c" * 64
STAGE_KEY = "stage-workflow-operation-00000001"
RESUME_KEY = "resume-workflow-operation-0000001"


class TwoPhaseWorkflowTests(unittest.TestCase):
    def test_stage_attest_plan_apply_and_replay_are_end_to_end_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)

            receipt = context.stage(now=100)
            envelope = context.envelope(receipt.binding)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(envelope,),
                now=110,
            )
            result = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=111,
            )
            repeated = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=112,
            )

            self.assertEqual(result.status, "applied")
            self.assertEqual(result.code, "applied")
            self.assertEqual((context.source / "app.txt").read_text(), "candidate\n")
            self.assertEqual(repeated.status, "applied")
            self.assertEqual(repeated.code, "already_applied")
            self.assertTrue(repeated.idempotent_replay)
            self.assertEqual(repeated.transaction_id, result.transaction_id)

    def test_stage_replay_is_stable_and_does_not_reread_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)

            first = context.stage(now=100)
            second = context.stage(now=100)
            self.assertEqual(first.binding, second.binding)
            self.assertTrue(second.idempotent_replay)

            (context.candidate / "app.txt").write_text("drifted\n")
            third = context.stage(now=100)

            self.assertEqual(first.binding, third.binding)
            self.assertTrue(third.idempotent_replay)

    def test_stage_rejects_candidate_drift_after_snapshot_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            candidate_snapshot_fingerprint = (
                candidate_workspace_snapshot_fingerprint(
                    context.candidate,
                    context.service.config.workspace_policy,
                )
            )
            (context.candidate / "app.txt").write_text("drifted\n")

            with self.assertRaisesRegex(
                TwoPhaseWorkflowError,
                "evaluated snapshot",
            ):
                context.stage(
                    now=100,
                    expected_candidate_snapshot_fingerprint=(
                        candidate_snapshot_fingerprint
                    ),
                )

    def test_source_drift_before_plan_is_terminal_and_never_issues_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            (context.source / "app.txt").write_text("external change\n")

            with self.assertRaisesRegex(TwoPhaseWorkflowError, "source drifted"):
                context.service.plan_resume(
                    receipt.workflow_id,
                    workspace=context.source,
                    expected_source_fingerprint=receipt.binding.source_fingerprint,
                    expected_config_sha256=CONFIG_SHA256,
                    idempotency_key=RESUME_KEY,
                    attestation_envelopes=(context.envelope(receipt.binding),),
                    now=110,
                )

            self.assertEqual(
                context.service.status(receipt.workflow_id, now=111).status,
                "conflicted",
            )

    def test_source_drift_after_plan_returns_conflict_without_consuming_as_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )
            (context.source / "app.txt").write_text("external change\n")

            result = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=111,
            )

            self.assertEqual(result.status, "conflicted")
            self.assertEqual(result.code, "source_drift")
            self.assertEqual((context.source / "app.txt").read_text(), "external change\n")

    def test_verified_no_change_still_consumes_fresh_confirmation_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=False)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            result = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=111,
            )

            self.assertEqual(result.status, "applied")
            self.assertEqual(result.code, "verified_no_change")
            self.assertIsNotNone(result.result_sha256)

    def test_crash_during_transaction_recovers_then_requires_new_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            def crash(**kwargs: object) -> object:
                return real_apply_changeset(**kwargs, _fault_after_mutation=0)

            with patch(
                "local_moe.assistant_bridge_two_phase.apply_changeset", crash
            ):
                with self.assertRaises(RuntimeError):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )
            applying = context.service.status(receipt.workflow_id, now=112)
            self.assertEqual(applying.status, "applying")
            transaction_id = applying.apply_transaction_id

            recovered = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=113,
            )

            self.assertEqual(recovered.status, "ready")
            self.assertEqual(recovered.code, "recovered_confirmation_required")
            self.assertEqual((context.source / "app.txt").read_text(), "source\n")
            self.assertFalse(
                (
                    context.transactions
                    / f"transaction-{transaction_id}"
                ).exists()
            )
            new_plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key="resume-workflow-operation-0000002",
                now=114,
            )
            applied = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=new_plan.plan_id,
                confirmation_id=new_plan.confirmation_id,
                now=115,
            )
            self.assertEqual(applied.status, "applied")

    def test_expired_applying_journal_rolls_back_without_new_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            def crash(**kwargs: object) -> object:
                return real_apply_changeset(**kwargs, _fault_after_mutation=0)

            with patch(
                "local_moe.assistant_bridge_two_phase.apply_changeset", crash
            ):
                with self.assertRaises(RuntimeError):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )

            applying = context.service.status(receipt.workflow_id, now=250)
            self.assertEqual(applying.status, "applying")
            recovered = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=250,
            )
            repeated = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=251,
            )

            self.assertEqual(recovered.status, "expired")
            self.assertEqual(recovered.code, "recovered_expired")
            self.assertTrue(recovered.idempotent_replay)
            self.assertEqual(repeated, recovered)
            self.assertEqual((context.source / "app.txt").read_text(), "source\n")

    def test_expired_applying_source_state_resets_without_reusing_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            with patch(
                "local_moe.assistant_bridge_two_phase.apply_changeset",
                side_effect=RuntimeError("simulated pre-journal crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "pre-journal"):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )

            recovered = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=250,
            )

            self.assertEqual(recovered.status, "expired")
            self.assertEqual(recovered.code, "recovered_expired")
            self.assertEqual((context.source / "app.txt").read_text(), "source\n")

    def test_consumed_confirmation_without_a_journal_requires_fresh_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            with patch(
                "local_moe.assistant_bridge_two_phase.apply_changeset",
                side_effect=RuntimeError("simulated pre-journal crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "pre-journal"):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )

            recovered = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=112,
            )
            repeated = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=113,
            )

            self.assertEqual(recovered.status, "ready")
            self.assertEqual(recovered.code, "recovered_confirmation_required")
            self.assertEqual(repeated, recovered)
            self.assertEqual((context.source / "app.txt").read_text(), "source\n")

    def test_crash_after_workspace_commit_is_reconciled_without_duplicate_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            with patch.object(
                context.service,
                "_finalize_applied",
                side_effect=RuntimeError("simulated post-commit crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "post-commit"):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )
            self.assertEqual((context.source / "app.txt").read_text(), "candidate\n")

            recovered = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=112,
            )

            self.assertEqual(recovered.status, "applied")
            self.assertEqual(recovered.code, "applied_recovered")
            self.assertTrue(recovered.idempotent_replay)

    def test_post_commit_crash_finalizes_and_replays_after_all_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )

            with patch.object(
                context.service,
                "_finalize_applied",
                side_effect=RuntimeError("simulated post-commit crash"),
            ):
                with self.assertRaisesRegex(RuntimeError, "post-commit"):
                    context.service.apply_resume(
                        receipt.workflow_id,
                        workspace=context.source,
                        expected_source_fingerprint=receipt.binding.source_fingerprint,
                        expected_config_sha256=CONFIG_SHA256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=111,
                    )
            self.assertEqual((context.source / "app.txt").read_text(), "candidate\n")
            self.assertEqual(
                context.service.status(receipt.workflow_id, now=250).status,
                "applying",
            )

            finalized = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=250,
            )
            replayed = context.service.apply_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=251,
            )

            self.assertEqual(finalized.status, "applied")
            self.assertEqual(finalized.code, "applied_recovered")
            self.assertTrue(finalized.idempotent_replay)
            self.assertEqual(replayed.status, "applied")
            self.assertEqual(replayed.code, "already_applied")
            self.assertEqual(replayed.transaction_id, finalized.transaction_id)

    def test_another_workspace_root_cannot_use_a_valid_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = _context(Path(temporary), changed=True)
            receipt = context.stage(now=100)
            plan = context.service.plan_resume(
                receipt.workflow_id,
                workspace=context.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=CONFIG_SHA256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(context.envelope(receipt.binding),),
                now=110,
            )
            other = Path(temporary) / "other"
            shutil.copytree(context.source, other)

            with self.assertRaisesRegex(TwoPhaseWorkflowError, "another workspace"):
                context.service.apply_resume(
                    receipt.workflow_id,
                    workspace=other,
                    expected_source_fingerprint=receipt.binding.source_fingerprint,
                    expected_config_sha256=CONFIG_SHA256,
                    plan_id=plan.plan_id,
                    confirmation_id=plan.confirmation_id,
                    now=111,
                )


class _Context:
    def __init__(self, root: Path, *, changed: bool) -> None:
        self.root = root
        self.source = root / "source"
        self.candidate = root / "candidate"
        self.transactions = root / "transactions"
        self.source.mkdir()
        (self.source / "app.txt").write_text("source\n")
        shutil.copytree(self.source, self.candidate)
        if changed:
            (self.candidate / "app.txt").write_text("candidate\n")
        self.private_key = Ed25519PrivateKey.generate()
        self.requirement = VerifierRequirement(
            verifier_id="independent-tests",
            adapter_id=ED25519_DSSE_ADAPTER_ID,
            key_id="independent-tests-key",
            public_key_sha256=ed25519_public_key_sha256(
                self.private_key.public_key()
            ),
            spec_sha256=sha256_bytes(b"independent-tests-spec-v1"),
        )
        self.policy = VerificationPolicy(
            policy_id="independent-policy-v1",
            quorum=1,
            verifiers=(self.requirement,),
        )
        store = SQLiteWorkflowStore(root / "state" / "workflows.sqlite3")
        cas = ContentAddressedStore(root / "state" / "cas")
        trust = AttestationTrustStore(
            (TrustedEd25519Verifier(self.requirement, self.private_key.public_key()),)
        )
        self.service = TwoPhaseWorkflowService(
            store=store,
            cas=cas,
            config=TwoPhaseWorkflowConfig(
                workspace_policy=WorkspaceScopePolicy(),
                transaction_state_dir=str(self.transactions),
                candidate_ttl_seconds=100,
                confirmation_ttl_seconds=20,
            ),
            trust_store=trust,
        )

    def stage(
        self,
        *,
        now: float,
        expected_candidate_snapshot_fingerprint: str | None = None,
    ) -> object:
        source_fingerprint = snapshot_workspace(
            self.source, self.service.config.workspace_policy
        ).fingerprint
        return self.service.stage_candidate(
            source_workspace=self.source,
            candidate_workspace=self.candidate,
            task_fingerprint=TASK_SHA256,
            expected_source_fingerprint=source_fingerprint,
            expected_config_sha256=CONFIG_SHA256,
            expected_candidate_snapshot_fingerprint=(
                expected_candidate_snapshot_fingerprint
            ),
            verification_policy=self.policy,
            idempotency_key=STAGE_KEY,
            now=now,
        )

    def envelope(self, binding: object) -> bytes:
        return create_ed25519_dsse_envelope(
            binding,
            self.requirement,
            self.private_key,
            attestation_id="independent-attestation-1",
            issued_at=105,
            expires_at=190,
            checks=(
                AttestationCheck(
                    check_id="project-tests",
                    passed=True,
                    evidence_sha256=CHECK_SHA256,
                ),
            ),
        )


def _context(root: Path, *, changed: bool) -> _Context:
    return _Context(root, changed=changed)


if __name__ == "__main__":
    unittest.main()
