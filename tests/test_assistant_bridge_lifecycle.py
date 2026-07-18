from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe import assistant_bridge_workflow_store as workflow_store
from local_moe.assistant_bridge_attestation import (
    ED25519_DSSE_ADAPTER_ID,
    create_ed25519_dsse_envelope,
)
from local_moe.assistant_bridge_integrity import sha256_bytes
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.assistant_bridge_lifecycle import (
    GeneratedCandidate,
    TwoPhaseLifecycleError,
    build_two_phase_lifecycle,
)
from local_moe.assistant_bridge_two_phase import (
    TwoPhaseWorkflowError,
    candidate_workspace_snapshot_fingerprint,
)
from local_moe.assistant_bridge_two_phase_config import (
    TwoPhaseConfigError,
    load_two_phase_lifecycle_config,
)
from local_moe.assistant_bridge_two_phase_contracts import AttestationCheck
from local_moe.assistant_bridge_two_phase_state import (
    load_two_phase_state_config,
)
from local_moe.assistant_bridge_two_phase_status import (
    TwoPhaseStatusError,
    build_two_phase_status_reader,
)
from local_moe.assistant_bridge_workspace import (
    WorkspaceScopePolicy,
    snapshot_workspace,
)


TASK_SHA256 = "a" * 64
GENERATOR_CONFIG_SHA256 = "b" * 64
STAGE_KEY = "stage-lifecycle-operation-00000001"
RESUME_KEY = "resume-lifecycle-operation-0000001"


def _filesystem_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        state = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        entries.append(
            (
                path.relative_to(root).as_posix(),
                state.st_mode,
                state.st_size,
                state.st_mtime_ns,
                digest,
            )
        )
    return tuple(entries)


def _cas_object_path(cas: ContentAddressedStore, digest: str) -> Path:
    return cas.root / "objects" / "sha256" / digest[:2] / digest[2:]


class TwoPhaseConfigurationTests(unittest.TestCase):
    def test_loads_only_public_verification_material_and_resolves_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))

            config = load_two_phase_lifecycle_config(fixture.config_path)

            self.assertEqual(
                config.state.database_path,
                fixture.root / "state" / "workflows.sqlite3",
            )
            self.assertEqual(config.state.cas_path, fixture.root / "state" / "cas")
            self.assertEqual(config.trust.policy.quorum, 1)
            self.assertEqual(
                config.trust.policy.verifiers[0].adapter_id,
                ED25519_DSSE_ADAPTER_ID,
            )
            self.assertEqual(len(config.effective_sha256), 64)
            self.assertIsNotNone(config.trust.build_trust_store())

    def test_state_only_load_does_not_open_public_trust_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            fixture.public_key_path.unlink()

            state = load_two_phase_state_config(fixture.config_path)

            self.assertEqual(state.database_path, root / "state" / "workflows.sqlite3")
            with self.assertRaisesRegex(TwoPhaseConfigError, "public key"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_status_import_path_does_not_load_trust_or_provider_modules(self) -> None:
        program = """
import sys
import local_moe.assistant_bridge_two_phase_status

blocked = {
    "local_moe.assistant_bridge_attestation",
    "local_moe.assistant_bridge_lifecycle",
    "local_moe.assistant_bridge_two_phase_config",
}
loaded = sorted(
    name
    for name in sys.modules
    if name in blocked or name == "cryptography" or name.startswith("cryptography.")
)
if loaded:
    raise SystemExit(",".join(loaded))
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_private_key_material_and_unknown_private_key_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            private_path = fixture.root / "private.pem"
            private_path.write_bytes(
                fixture.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            raw["trust"]["verifiers"][0]["public_key_file"] = "private.pem"
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(TwoPhaseConfigError, "public.*only"):
                load_two_phase_lifecycle_config(fixture.config_path)

            raw["trust"]["verifiers"][0]["public_key_file"] = "public.pem"
            raw["trust"]["verifiers"][0]["private_key_file"] = "private.pem"
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(TwoPhaseConfigError, "unknown fields"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_rejects_linked_configuration_and_public_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            linked_config = fixture.root / "linked-config.json"
            linked_key = fixture.root / "linked-public.pem"
            try:
                os.symlink(fixture.config_path, linked_config)
                os.symlink(fixture.public_key_path, linked_key)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")

            with self.assertRaisesRegex(TwoPhaseConfigError, "non-link"):
                load_two_phase_state_config(linked_config)

            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            raw["trust"]["verifiers"][0]["public_key_file"] = linked_key.name
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(TwoPhaseConfigError, "non-link"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_rejects_oversized_configuration_and_public_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.config_path.write_bytes(b"{" + b" " * (1024 * 1024))

            with self.assertRaisesRegex(TwoPhaseConfigError, "safe bounds"):
                load_two_phase_state_config(fixture.config_path)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.public_key_path.write_bytes(b"P" * (64 * 1024 + 1))

            with self.assertRaisesRegex(TwoPhaseConfigError, "safe bounds"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_rejects_duplicate_physical_public_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            duplicate = dict(raw["trust"]["verifiers"][0])
            duplicate["verifier_id"] = "second-independent-tests"
            duplicate["key_id"] = "second-independent-tests-key"
            raw["trust"]["quorum"] = 2
            raw["trust"]["verifiers"].append(duplicate)
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(TwoPhaseConfigError, "physical public key"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_rejects_unbounded_verifier_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            verifier = raw["trust"]["verifiers"][0]
            raw["trust"]["verifiers"] = [dict(verifier) for _ in range(65)]
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(TwoPhaseConfigError, "safe bounds"):
                load_two_phase_lifecycle_config(fixture.config_path)

    def test_rejects_unbounded_aggregate_public_key_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            verifiers = []
            for index in range(17):
                public_key = (
                    Ed25519PrivateKey.generate()
                    .public_key()
                    .public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )
                public_key += b"\n" * (64 * 1024 - len(public_key))
                key_name = f"public-{index}.pem"
                (fixture.root / key_name).write_bytes(public_key)
                verifiers.append(
                    {
                        "verifier_id": f"independent-tests-{index}",
                        "adapter_id": ED25519_DSSE_ADAPTER_ID,
                        "key_id": f"independent-tests-key-{index}",
                        "public_key_file": key_name,
                        "spec_sha256": sha256_bytes(b"test-spec-v1"),
                    }
                )
            raw["trust"]["verifiers"] = verifiers
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(TwoPhaseConfigError, "Aggregate public"):
                load_two_phase_lifecycle_config(fixture.config_path)


class TwoPhaseLifecycleTests(unittest.TestCase):
    def test_stage_operation_digest_binds_idempotency_namespace_and_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            common = {
                "source_workspace": fixture.source,
                "task_fingerprint": TASK_SHA256,
                "expected_source_fingerprint": fixture.source_fingerprint,
                "expected_config_sha256": lifecycle.effective_config_sha256,
            }

            default = lifecycle.stage_operation_sha256(
                **common,
                idempotency_key=STAGE_KEY,
            )
            another_key = lifecycle.stage_operation_sha256(
                **common,
                idempotency_key="stage-lifecycle-operation-00000002",
            )
            another_ttl = lifecycle.stage_operation_sha256(
                **common,
                idempotency_key=STAGE_KEY,
                ttl_seconds=60,
            )

        self.assertEqual(len(default), 64)
        self.assertEqual(len({default, another_key, another_ttl}), 3)

    def test_rejects_each_configured_state_path_inside_governed_workspace(
        self,
    ) -> None:
        cases = (
            ("database_path", "source/.durable/workflows.sqlite3"),
            ("cas_path", "source/.durable/cas"),
            ("transaction_state_dir", "source/.durable/transactions"),
        )
        for field, value in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _Fixture(Path(temporary))
                raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
                raw["state"][field] = value
                fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")
                generator = _CandidateGenerator(fixture.source)

                with self.assertRaisesRegex(
                    TwoPhaseLifecycleError,
                    "outside the governed workspace",
                ):
                    fixture.lifecycle(generator)

                self.assertEqual(generator.calls, 0)
                self.assertFalse((fixture.source / ".durable").exists())

    def test_state_path_overlap_uses_resolved_physical_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            alias = fixture.root / "workspace-alias"
            try:
                os.symlink(fixture.source, alias)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            raw = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            raw["state"]["database_path"] = "workspace-alias/.durable/workflows.sqlite3"
            fixture.config_path.write_text(json.dumps(raw), encoding="utf-8")
            generator = _CandidateGenerator(fixture.source)

            with self.assertRaisesRegex(
                TwoPhaseLifecycleError,
                "outside the governed workspace",
            ):
                fixture.lifecycle(generator)

            self.assertEqual(generator.calls, 0)
            self.assertFalse((fixture.source / ".durable").exists())

    def test_stage_replay_preflight_skips_candidate_generator_and_shares_cas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            generator = _CandidateGenerator(fixture.source)
            lifecycle = fixture.lifecycle(generator)
            source_fingerprint = fixture.source_fingerprint

            first = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            generator.configuration_sha256 = "c" * 64
            second = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=101,
            )

            self.assertEqual(generator.calls, 1)
            self.assertEqual(first.binding, second.binding)
            self.assertTrue(second.idempotent_replay)
            self.assertEqual((fixture.source / "app.txt").read_text(), "source\n")
            self.assertIs(
                lifecycle.workflow_service.cas,
                lifecycle.workflow_service.store.evidence_cas,
            )

    def test_expected_config_and_source_are_checked_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            generator = _CandidateGenerator(fixture.source)
            lifecycle = fixture.lifecycle(generator)

            with self.assertRaisesRegex(TwoPhaseLifecycleError, "configuration"):
                lifecycle.stage(
                    "change-app",
                    source_workspace=fixture.source,
                    task_fingerprint=TASK_SHA256,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256="f" * 64,
                    idempotency_key=STAGE_KEY,
                    now=100,
                )
            with self.assertRaisesRegex(TwoPhaseWorkflowError, "expected stage"):
                lifecycle.stage(
                    "change-app",
                    source_workspace=fixture.source,
                    task_fingerprint=TASK_SHA256,
                    expected_source_fingerprint="e" * 64,
                    expected_config_sha256=lifecycle.effective_config_sha256,
                    idempotency_key=STAGE_KEY,
                    now=100,
                )
            generator.configuration_sha256 = "c" * 64
            with self.assertRaisesRegex(TwoPhaseLifecycleError, "configuration"):
                lifecycle.stage(
                    "change-app",
                    source_workspace=fixture.source,
                    task_fingerprint=TASK_SHA256,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256=lifecycle.effective_config_sha256,
                    idempotency_key=STAGE_KEY,
                    now=100,
                )

            self.assertEqual(generator.calls, 0)

    def test_candidate_drift_after_evaluation_is_rejected_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            generator = _CandidateGenerator(
                fixture.source,
                drift_after_fingerprint=True,
            )
            lifecycle = fixture.lifecycle(generator)

            with self.assertRaisesRegex(
                TwoPhaseLifecycleError,
                "evaluated snapshot",
            ):
                lifecycle.stage(
                    "change-app",
                    source_workspace=fixture.source,
                    task_fingerprint=TASK_SHA256,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256=lifecycle.effective_config_sha256,
                    idempotency_key=STAGE_KEY,
                    now=100,
                )

            self.assertEqual(generator.calls, 1)
            self.assertEqual(lifecycle.workflow_service.store.list_workflows(), ())

    def test_plan_and_apply_require_the_staged_config_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            generator = _CandidateGenerator(fixture.source)
            lifecycle = fixture.lifecycle(generator)
            config_sha256 = lifecycle.effective_config_sha256
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            requirement = lifecycle.config.trust.policy.verifiers[0]
            envelope = create_ed25519_dsse_envelope(
                receipt.binding,
                requirement,
                fixture.private_key,
                attestation_id="lifecycle-attestation-1",
                issued_at=105,
                expires_at=250,
                checks=(
                    AttestationCheck(
                        check_id="project-tests",
                        passed=True,
                        evidence_sha256=sha256_bytes(b"passed"),
                    ),
                ),
            )

            with self.assertRaisesRegex(TwoPhaseLifecycleError, "configuration"):
                lifecycle.plan_resume(
                    receipt.workflow_id,
                    workspace=fixture.source,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256="f" * 64,
                    idempotency_key=RESUME_KEY,
                    attestation_envelopes=(envelope,),
                    now=110,
                )
            with self.assertRaisesRegex(TwoPhaseWorkflowError, "expected source"):
                lifecycle.plan_resume(
                    receipt.workflow_id,
                    workspace=fixture.source,
                    expected_source_fingerprint="e" * 64,
                    expected_config_sha256=config_sha256,
                    idempotency_key=RESUME_KEY,
                    attestation_envelopes=(envelope,),
                    now=110,
                )
            plan = lifecycle.plan_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=config_sha256,
                idempotency_key=RESUME_KEY,
                attestation_envelopes=(envelope,),
                now=110,
            )
            with self.assertRaisesRegex(TwoPhaseLifecycleError, "configuration"):
                lifecycle.apply_resume(
                    receipt.workflow_id,
                    workspace=fixture.source,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256="f" * 64,
                    plan_id=plan.plan_id,
                    confirmation_id=plan.confirmation_id,
                    now=111,
                )
            with self.assertRaisesRegex(TwoPhaseWorkflowError, "expected source"):
                lifecycle.apply_resume(
                    receipt.workflow_id,
                    workspace=fixture.source,
                    expected_source_fingerprint="e" * 64,
                    expected_config_sha256=config_sha256,
                    plan_id=plan.plan_id,
                    confirmation_id=plan.confirmation_id,
                    now=111,
                )
            result = lifecycle.apply_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=config_sha256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=111,
            )

            self.assertEqual(result.status, "applied")
            self.assertEqual((fixture.source / "app.txt").read_text(), "candidate\n")

    def test_status_reader_needs_neither_generator_nor_public_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            fixture.public_key_path.unlink()

            state = load_two_phase_state_config(fixture.config_path)
            reader = build_two_phase_status_reader(state)
            record = reader.status(receipt.workflow_id, now=101)

            self.assertEqual(record.status, "staged")
            self.assertEqual(record.binding, receipt.binding)

    def test_status_reader_requires_existing_state_without_creating_anything(
        self,
    ) -> None:
        for missing in ("cas", "database", "key", "schema"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _Fixture(Path(temporary))
                fixture.lifecycle(_CandidateGenerator(fixture.source))
                state = load_two_phase_state_config(fixture.config_path)
                key = state.database_path.with_suffix(".key")
                if missing == "cas":
                    shutil.rmtree(state.cas_path)
                elif missing == "database":
                    state.database_path.unlink()
                elif missing == "key":
                    key.unlink()
                else:
                    state.database_path.unlink()
                    sqlite3.connect(state.database_path).close()
                    if os.name == "posix":
                        state.database_path.chmod(0o600)
                before = _filesystem_snapshot(fixture.root)

                with self.assertRaises(TwoPhaseStatusError) as caught:
                    build_two_phase_status_reader(state)

                self.assertEqual(caught.exception.code, "state_uninitialized")
                self.assertNotIn(str(fixture.root), str(caught.exception))
                self.assertEqual(_filesystem_snapshot(fixture.root), before)

    def test_status_reader_reports_typed_redacted_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            reader = build_two_phase_status_reader(state)

            for workflow_id, now, code in (
                ("wf-missing", 101, "workflow_not_found"),
                (receipt.workflow_id, 99, "clock_conflict"),
            ):
                with self.subTest(code=code):
                    with self.assertRaises(TwoPhaseStatusError) as caught:
                        reader.status(workflow_id, now=now)
                    self.assertEqual(caught.exception.code, code)
                    self.assertNotIn(str(state.database_path), str(caught.exception))

            with mock.patch.object(
                reader.store,
                "read_workflow",
                side_effect=RuntimeError(f"internal path: {fixture.root}"),
            ):
                with self.assertRaises(TwoPhaseStatusError) as caught:
                    reader.status(receipt.workflow_id, now=101)
            self.assertEqual(caught.exception.code, "status_runtime_failed")
            self.assertNotIn(str(fixture.root), str(caught.exception))

    def test_status_reader_classifies_partial_schema_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.lifecycle(_CandidateGenerator(fixture.source))
            state = load_two_phase_state_config(fixture.config_path)
            with (
                closing(sqlite3.connect(state.database_path)) as connection,
                connection,
            ):
                connection.execute("DROP TABLE workflow_events")

            with self.assertRaises(TwoPhaseStatusError) as caught:
                build_two_phase_status_reader(state)

            self.assertEqual(caught.exception.code, "state_invalid")

    def test_status_reader_classifies_incoherent_applied_row_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            with (
                closing(sqlite3.connect(state.database_path)) as connection,
                connection,
            ):
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute(
                    "UPDATE workflows SET status = 'applied', "
                    "apply_transaction_id = NULL, result_sha256 = NULL "
                    "WHERE workflow_id = ?",
                    (receipt.workflow_id,),
                )

            reader = build_two_phase_status_reader(state)
            with self.assertRaises(TwoPhaseStatusError) as caught:
                reader.status(receipt.workflow_id, now=101)

            self.assertEqual(caught.exception.code, "state_invalid")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs unavailable")
    def test_status_reader_rejects_fifo_secret_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.lifecycle(_CandidateGenerator(fixture.source))
            state = load_two_phase_state_config(fixture.config_path)
            key = state.database_path.with_suffix(".key")
            key.unlink()
            os.mkfifo(key, mode=0o600)
            before = _filesystem_snapshot(fixture.root)

            with mock.patch(
                "local_moe.assistant_bridge_workflow_store.os.open",
                wraps=os.open,
            ) as opened:
                with self.assertRaises(TwoPhaseStatusError) as caught:
                    build_two_phase_status_reader(state)

            self.assertEqual(caught.exception.code, "state_invalid")
            opened.assert_not_called()
            self.assertEqual(_filesystem_snapshot(fixture.root), before)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs unavailable")
    def test_secret_open_is_nonblocking_if_file_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / "state.key"
            secret.write_bytes(b"s" * 32)
            fifo = root / "replacement"
            os.mkfifo(fifo, mode=0o600)
            fifo_descriptor = os.open(
                fifo,
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
            )

            with mock.patch.object(
                workflow_store.os,
                "open",
                return_value=fifo_descriptor,
            ) as opened:
                with self.assertRaisesRegex(
                    workflow_store.WorkflowStoreError,
                    "private",
                ):
                    workflow_store._load_existing_secret(secret)

            flags = int(opened.call_args.args[1])
            self.assertTrue(flags & getattr(os, "O_NONBLOCK", 0))
            if hasattr(os, "O_NOFOLLOW"):
                self.assertTrue(flags & os.O_NOFOLLOW)

    def test_status_reader_validates_complete_live_candidate_closure(self) -> None:
        for failure in ("missing-manifest", "corrupt-content"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _Fixture(Path(temporary))
                lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
                receipt = lifecycle.stage(
                    "change-app",
                    source_workspace=fixture.source,
                    task_fingerprint=TASK_SHA256,
                    expected_source_fingerprint=fixture.source_fingerprint,
                    expected_config_sha256=lifecycle.effective_config_sha256,
                    idempotency_key=STAGE_KEY,
                    now=100,
                )
                state = load_two_phase_state_config(fixture.config_path)
                cas = ContentAddressedStore(
                    state.cas_path,
                    create_if_missing=False,
                )
                if failure == "missing-manifest":
                    target = _cas_object_path(cas, receipt.binding.manifest.sha256)
                    target.unlink()
                else:
                    manifest = cas.get_json(receipt.binding.manifest)
                    file_record = next(
                        item for item in manifest["files"] if item["kind"] == "file"
                    )
                    content = file_record["content"]
                    target = _cas_object_path(cas, content["digest"]["sha256"])
                    target.write_bytes(b"\x00" * int(content["sizeBytes"]))
                before = _filesystem_snapshot(fixture.root / "state")
                reader = build_two_phase_status_reader(state)

                with mock.patch(
                    "local_moe.assistant_bridge_cas.tempfile.TemporaryDirectory"
                ) as materialized:
                    with self.assertRaises(TwoPhaseStatusError) as caught:
                        reader.status(receipt.workflow_id, now=101)

                self.assertEqual(caught.exception.code, "artifact_invalid")
                materialized.assert_not_called()
                self.assertEqual(
                    _filesystem_snapshot(fixture.root / "state"),
                    before,
                )

    def test_terminal_status_does_not_require_retained_candidate_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            cas = ContentAddressedStore(state.cas_path, create_if_missing=False)
            _cas_object_path(cas, receipt.binding.manifest.sha256).unlink()
            before = _filesystem_snapshot(fixture.root / "state")

            record = build_two_phase_status_reader(state).status(
                receipt.workflow_id,
                now=301,
            )

            self.assertEqual(record.status, "expired")
            self.assertEqual(_filesystem_snapshot(fixture.root / "state"), before)

    def test_status_reader_detects_database_ctime_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            reader = build_two_phase_status_reader(state)

            snapshot = workflow_store._database_read_snapshot(state.database_path)
            self.assertEqual(len(snapshot), 5)
            changed_ctime = (*snapshot[:-1], snapshot[-1] + 1)
            with mock.patch.object(
                workflow_store,
                "_database_read_snapshot",
                side_effect=(snapshot, changed_ctime),
            ):
                with self.assertRaises(TwoPhaseStatusError) as caught:
                    reader.status(receipt.workflow_id, now=101)

            self.assertEqual(caught.exception.code, "state_invalid")

    def test_database_snapshot_uses_comparable_windows_identity_time(self) -> None:
        path_state = mock.Mock(st_birthtime_ns=100, st_ctime_ns=200)
        descriptor_state = mock.Mock(st_birthtime_ns=100, st_ctime_ns=300)

        with mock.patch.object(workflow_store.os, "name", "nt"):
            path_time = workflow_store._database_identity_time_ns(path_state)
            descriptor_time = workflow_store._database_identity_time_ns(
                descriptor_state
            )

        self.assertEqual(path_time, descriptor_time)

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "mkfifo"),
        "POSIX descriptor aliases and FIFOs are required",
    )
    def test_status_reader_never_reopens_a_swapped_database_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            reader = build_two_phase_status_reader(state)
            parked = state.database_path.with_suffix(".parked")
            fifo = state.database_path.with_suffix(".fifo")
            os.mkfifo(fifo, mode=0o600)
            real_connect = sqlite3.connect
            targets: list[str] = []

            def swap_before_connect(target, *args, **kwargs):
                targets.append(str(target))
                state.database_path.rename(parked)
                fifo.rename(state.database_path)
                try:
                    return real_connect(target, *args, **kwargs)
                finally:
                    state.database_path.rename(fifo)
                    parked.rename(state.database_path)
                    state.database_path.chmod(0o400)
                    state.database_path.chmod(0o600)

            with mock.patch.object(
                workflow_store.sqlite3,
                "connect",
                side_effect=swap_before_connect,
            ):
                with self.assertRaises(TwoPhaseStatusError) as caught:
                    reader.status(receipt.workflow_id, now=101)

            self.assertEqual(caught.exception.code, "state_invalid")
            self.assertTrue(targets)
            self.assertTrue(
                all(target.startswith("file:/dev/fd/") for target in targets)
            )
            self.assertTrue(state.database_path.is_file())
            self.assertTrue(fifo.exists())

    def test_status_reader_fallback_snapshot_is_outside_configured_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            before = _filesystem_snapshot(fixture.root / "state")
            real_connect = sqlite3.connect
            targets: list[str] = []

            def capture_connect(target, *args, **kwargs):
                targets.append(str(target))
                return real_connect(target, *args, **kwargs)

            with (
                mock.patch.object(
                    workflow_store,
                    "_database_descriptor_uri",
                    return_value=None,
                ),
                mock.patch.object(
                    workflow_store.sqlite3,
                    "connect",
                    side_effect=capture_connect,
                ),
            ):
                reader = build_two_phase_status_reader(state)
                record = reader.status(receipt.workflow_id, now=101)

            self.assertEqual(record.status, "staged")
            self.assertTrue(targets)
            self.assertTrue(
                all("mymoe-workflow-status-" in target for target in targets)
            )
            self.assertTrue(
                all(str(state.database_path) not in target for target in targets)
            )
            self.assertEqual(_filesystem_snapshot(fixture.root / "state"), before)

    def test_status_reader_derives_expiry_without_mutating_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint=TASK_SHA256,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=100,
            )
            state = load_two_phase_state_config(fixture.config_path)
            before = _filesystem_snapshot(fixture.root / "state")

            reader = build_two_phase_status_reader(state)
            record = reader.status(receipt.workflow_id, now=301)

            self.assertEqual(record.status, "expired")
            self.assertEqual(_filesystem_snapshot(fixture.root / "state"), before)


class _CandidateGenerator:
    configuration_sha256 = GENERATOR_CONFIG_SHA256

    def __init__(
        self,
        source: Path,
        *,
        drift_after_fingerprint: bool = False,
    ) -> None:
        self.source = source
        self.drift_after_fingerprint = drift_after_fingerprint
        self.calls = 0

    @contextmanager
    def generate(
        self,
        request: str,
        *,
        source_workspace: str | Path,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        expected_operation_sha256: str,
    ):
        self.calls += 1
        if request != "change-app" or Path(source_workspace) != self.source:
            raise AssertionError("candidate request binding changed")
        if len(expected_config_sha256) != 64:
            raise AssertionError("candidate config binding changed")
        if len(expected_operation_sha256) != 64:
            raise AssertionError("candidate operation binding changed")
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "candidate"
            shutil.copytree(self.source, candidate)
            (candidate / "app.txt").write_text("candidate\n")
            candidate_snapshot_fingerprint = candidate_workspace_snapshot_fingerprint(
                candidate,
                WorkspaceScopePolicy(),
            )
            if self.drift_after_fingerprint:
                (candidate / "app.txt").write_text("drifted\n")
            yield GeneratedCandidate(
                workspace=candidate,
                source_fingerprint=expected_source_fingerprint,
                candidate_snapshot_fingerprint=candidate_snapshot_fingerprint,
            )


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source"
        self.source.mkdir()
        (self.source / "app.txt").write_text("source\n")
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key_path = root / "public.pem"
        self.public_key_path.write_bytes(
            self.private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        self.config_path = root / "two-phase.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "state": {
                        "database_path": "state/workflows.sqlite3",
                        "cas_path": "state/cas",
                        "transaction_state_dir": "state/transactions",
                        "candidate_ttl_seconds": 200,
                        "confirmation_ttl_seconds": 20,
                        "transaction_lock_ttl_seconds": 30,
                        "sqlite_timeout_seconds": 5,
                    },
                    "trust": {
                        "policy_id": "independent-policy-v1",
                        "quorum": 1,
                        "verifiers": [
                            {
                                "verifier_id": "independent-tests",
                                "adapter_id": ED25519_DSSE_ADAPTER_ID,
                                "key_id": "independent-tests-key",
                                "public_key_file": "public.pem",
                                "spec_sha256": sha256_bytes(b"test-spec-v1"),
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

    @property
    def source_fingerprint(self) -> str:
        return snapshot_workspace(self.source, WorkspaceScopePolicy()).fingerprint

    def lifecycle(self, generator: _CandidateGenerator):
        config = load_two_phase_lifecycle_config(self.config_path)
        return build_two_phase_lifecycle(
            config,
            governed_workspace=self.source,
            workspace_policy=WorkspaceScopePolicy(),
            candidate_generator=generator,
        )


if __name__ == "__main__":
    unittest.main()
