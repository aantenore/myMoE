from __future__ import annotations

import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge import load_assistant_bridge_config
from local_moe.assistant_bridge_attestation import ed25519_public_key_sha256
from local_moe.assistant_bridge_integrity import canonical_json_bytes
from local_moe.route_canary import (
    AUTHORIZATION_PAYLOAD_TYPE,
    VerifiedRoutingCanaryAuthorization,
    load_and_verify_canary_authorization,
    load_verified_routing_canary_manifest,
    load_verified_routing_runtime_config,
)
from local_moe.route_outcomes import VerifiedOutcomeRecord
from local_moe.route_policy import load_route_policy
from local_moe.route_promotion import (
    build_evidence_plan,
    evaluate_route_promotion,
)
from local_moe.route_scorecard import build_route_scorecard, load_route_scorecard
from local_moe.verified_routing_contracts import sha256_json
from tests.test_route_promotion import (
    _cases,
    _gate_policy,
    _holdout_records,
    _training_records,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "sign_verified_routing_canary.py"
FIXTURES = ROOT / "tests" / "fixtures"


class RouteCanarySignCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_dir = self.root / "configs"
        self.work_dir = self.root / "work"
        self.config_dir.mkdir()
        self.work_dir.mkdir()

        self.private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
        self.private_key_path = self.work_dir / "operator-private.pem"
        self.private_key_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        self.private_key_path.write_bytes(self.private_key_bytes)
        self.private_key_path.chmod(0o600)
        self.public_key_path = self.work_dir / "operator-public.pem"
        self.public_key_path.write_bytes(
            self.private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

        self.policy_path = FIXTURES / "verified-routing-policy.json"
        self.scorecard_path = self.work_dir / "scorecard.json"
        self.gate_policy_path = self.work_dir / "gate-policy.json"
        self.plan_path = self.work_dir / "evidence-plan.json"
        self.training_records_path = self.work_dir / "training-records.json"
        self.holdout_records_path = self.work_dir / "holdout-records.json"
        self.manifest_path = self.work_dir / "canary-manifest.json"
        self.authorization_path = self.work_dir / "canary-authorization.dsse.json"
        self.runtime_path = self.config_dir / "verified-routing-runtime.json"
        runtime = {
            "schema_version": "1.0",
            "mode": "canary",
            "route_policy_path": str(self.policy_path),
            "scorecard_path": str(self.scorecard_path),
            "manifest_path": str(self.manifest_path),
            "authorization_path": str(self.authorization_path),
            "operator_key_id": "operator-test",
            "operator_public_key_path": str(self.public_key_path),
            "operator_public_key_sha256": ed25519_public_key_sha256(
                self.private_key.public_key()
            ),
            "assignment_secret_env": "MYMOE_TEST_CANARY_SECRET",
            "chronology_path": str(self.work_dir / "canary-chronology.json"),
        }
        self.runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

        bridge = json.loads(
            (ROOT / "configs" / "assistant-bridge.json").read_text(encoding="utf-8")
        )
        bridge["verified_routing"] = {
            "enabled": True,
            "config_path": "configs/verified-routing-runtime.json",
        }
        self.bridge_path = self.config_dir / "assistant-bridge.json"
        self.bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
        bridge_config = load_assistant_bridge_config(self.bridge_path)
        self.bridge_config_sha256 = bridge_config.source_sha256

        policy = load_route_policy(self.policy_path)
        training = [
            self._with_config(record, bridge_config.source_sha256)
            for record in _training_records()
        ]
        holdout = [
            self._with_config(record, bridge_config.source_sha256)
            for record in _holdout_records(20)
        ]
        scorecard = build_route_scorecard(
            training,
            minimum_evidence_strength="independent",
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=86_400,
        )
        gate_policy = _gate_policy()
        cases = [
            replace(case, config_sha256=bridge_config.source_sha256)
            for case in _cases(20)
        ]
        plan = build_evidence_plan(
            cases,
            route_policy=policy,
            scorecard=scorecard,
            gate_policy=gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
        )
        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=gate_policy,
            route_policy=policy,
            scorecard=scorecard,
            training_records=training,
            holdout_records=holdout,
            evaluated_at="2026-07-19T02:00:00+00:00",
        )
        self.assertEqual(report.payload()["status"], "eligible")
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.scorecard_path.write_text(
            json.dumps(scorecard.payload()), encoding="utf-8"
        )
        self.gate_policy_path.write_text(
            json.dumps(gate_policy.payload()), encoding="utf-8"
        )
        self.plan_path.write_text(
            json.dumps(plan.payload()), encoding="utf-8"
        )
        self._write_records(self.training_records_path, training)
        self._write_records(self.holdout_records_path, holdout)
        self.manifest_path.write_text(
            json.dumps(manifest.payload()), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_dsse_round_trips_through_runtime_verifier(self) -> None:
        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self._assert_private_material_absent(completed)
        envelope_bytes = self.authorization_path.read_bytes()
        envelope = json.loads(envelope_bytes)
        self.assertEqual(envelope_bytes, canonical_json_bytes(envelope))
        self.assertEqual(envelope["payloadType"], AUTHORIZATION_PAYLOAD_TYPE)
        self.assertEqual(len(envelope["signatures"]), 1)

        payload_bytes = base64.b64decode(envelope["payload"], validate=True)
        payload = json.loads(payload_bytes)
        self.assertEqual(payload_bytes, canonical_json_bytes(payload))
        authorization = VerifiedRoutingCanaryAuthorization.from_payload(payload)

        bridge = load_assistant_bridge_config(self.bridge_path)
        runtime = load_verified_routing_runtime_config(
            self.runtime_path,
            expected_source_sha256=str(bridge.verified_routing.config_sha256),
        )
        verified = load_and_verify_canary_authorization(
            self.authorization_path,
            manifest=load_verified_routing_canary_manifest(self.manifest_path),
            runtime=runtime,
            bridge_config_sha256=bridge.source_sha256,
            now="2026-07-19T02:05:00+00:00",
        )

        self.assertEqual(verified.authorization, authorization)
        self.assertEqual(authorization.bridge_config_sha256, bridge.source_sha256)
        self.assertEqual(authorization.maximum_canary_basis_points, 500)
        authorization_metadata = self.authorization_path.stat()
        self.assertTrue(stat.S_ISREG(authorization_metadata.st_mode))
        self.assertGreater(authorization_metadata.st_size, 0)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(authorization_metadata.st_mode), 0o600)

    def test_refuses_to_overwrite_existing_authorization(self) -> None:
        self.authorization_path.write_bytes(b"existing-authorization")

        completed = self._run()

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("already exists", completed.stderr)
        self.assertEqual(
            self.authorization_path.read_bytes(),
            b"existing-authorization",
        )
        self._assert_private_material_absent(completed)

    def test_rejects_a_private_key_outside_the_runtime_trust_policy(self) -> None:
        wrong_key = Ed25519PrivateKey.from_private_bytes(bytes(range(33, 65)))
        wrong_key_path = self.work_dir / "wrong-private.pem"
        wrong_key_bytes = wrong_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        wrong_key_path.write_bytes(wrong_key_bytes)
        wrong_key_path.chmod(0o600)

        completed = self._run("--private-key", str(wrong_key_path))

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("does not match", completed.stderr)
        self.assertFalse(self.authorization_path.exists())
        self._assert_key_bytes_absent(wrong_key_bytes, completed)

    def test_rejects_windows_outside_manifest_non_utc_and_unsafe_caps(self) -> None:
        cases = (
            (
                ("--issued-at", "2026-07-17T23:59:00+00:00", "--not-before", "2026-07-18T00:00:00+00:00"),
                "inside the manifest window",
            ),
            (
                ("--expires-at", "2026-07-20T00:00:00+00:00"),
                "inside the manifest window",
            ),
            (
                ("--not-before", "2026-07-19T03:00:00+01:00"),
                "must use UTC",
            ),
            (
                ("--maximum-canary-basis-points", "501"),
                "cap is unsafe",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                completed = self._run(*arguments)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(self.authorization_path.exists())
                self._assert_private_material_absent(completed)

    def test_rejects_training_or_bridge_lineage_drift(self) -> None:
        original = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        cases = (
            ("training_source_digest", "training lineage"),
            ("config_sha256", "assistant bridge configuration"),
        )
        for field, message in cases:
            with self.subTest(field=field):
                manifest = json.loads(json.dumps(original))
                if field == "training_source_digest":
                    manifest["lineage"][field] = "f" * 64
                else:
                    manifest["enabled_cells"][0][field] = "f" * 64
                unsigned = dict(manifest)
                unsigned.pop("manifest_sha256")
                manifest["manifest_sha256"] = sha256_json(unsigned)
                self.manifest_path.write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )

                completed = self._run()

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(self.authorization_path.exists())
        self.manifest_path.write_text(json.dumps(original), encoding="utf-8")

    def test_rejects_manifest_not_reconstructed_from_paired_evidence(self) -> None:
        original = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        mutations = (
            ("plan_sha256", "f" * 64),
            ("report_sha256", "e" * 64),
            ("evaluator_sha256", "d" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                manifest = json.loads(json.dumps(original))
                manifest["lineage"][field] = value
                unsigned = dict(manifest)
                unsigned.pop("manifest_sha256")
                manifest["manifest_sha256"] = sha256_json(unsigned)
                self.manifest_path.write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                completed = self._run()

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("eligible paired evidence", completed.stderr)
                self.assertFalse(self.authorization_path.exists())

        manifest = json.loads(json.dumps(original))
        manifest["enabled_cells"][0]["paired_tasks"] += 1
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        manifest["manifest_sha256"] = sha256_json(unsigned)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        completed = self._run()

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("eligible paired evidence", completed.stderr)
        self.assertFalse(self.authorization_path.exists())
        self.manifest_path.write_text(json.dumps(original), encoding="utf-8")

    def test_rejects_when_supplied_holdout_is_no_longer_eligible(self) -> None:
        holdout = [
            self._with_config(record, self.bridge_config_sha256)
            for record in _holdout_records(20, candidate_latency_ms=1_200)
        ]
        self._write_records(self.holdout_records_path, holdout)

        completed = self._run()

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("eligible paired evidence", completed.stderr)
        self.assertFalse(self.authorization_path.exists())

    @staticmethod
    def _with_config(
        record: VerifiedOutcomeRecord,
        config_sha256: str,
    ) -> VerifiedOutcomeRecord:
        unsigned = record.payload()
        unsigned.pop("record_id")
        unsigned["config_sha256"] = config_sha256
        return VerifiedOutcomeRecord.from_payload(
            {
                "record_id": f"outcome-{sha256_json(unsigned)}",
                **unsigned,
            }
        )

    @staticmethod
    def _write_records(
        path: Path,
        records: list[VerifiedOutcomeRecord],
    ) -> None:
        path.write_text(
            json.dumps({"records": [record.payload() for record in records]}),
            encoding="utf-8",
        )

    def _run(self, *overrides: str) -> subprocess.CompletedProcess[str]:
        values = {
            "--manifest": str(self.manifest_path),
            "--plan": str(self.plan_path),
            "--gate-policy": str(self.gate_policy_path),
            "--training-records": str(self.training_records_path),
            "--holdout-records": str(self.holdout_records_path),
            "--assistant-bridge-config": str(self.bridge_path),
            "--runtime-config": str(self.runtime_path),
            "--private-key": str(self.private_key_path),
            "--activation-id": "activation-test",
            "--issued-at": "2026-07-19T01:59:00+00:00",
            "--not-before": "2026-07-19T02:00:00+00:00",
            "--expires-at": "2026-07-19T02:45:00+00:00",
            "--maximum-canary-basis-points": "500",
            "--out": str(self.authorization_path),
        }
        if len(overrides) % 2:
            raise AssertionError("CLI overrides must be option/value pairs.")
        values.update(zip(overrides[::2], overrides[1::2]))
        command = [sys.executable, str(SCRIPT)]
        for option, value in values.items():
            command.extend((option, value))
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(ROOT / "src"), existing_pythonpath) if item
        )
        return subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def _assert_private_material_absent(
        self,
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        self._assert_key_bytes_absent(self.private_key_bytes, completed)

    def _assert_key_bytes_absent(
        self,
        key_bytes: bytes,
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        private_pem = key_bytes.decode("ascii")
        for stream in (completed.stdout, completed.stderr):
            self.assertNotIn(private_pem, stream)
            for line in private_pem.splitlines():
                if line and not line.startswith("-----"):
                    self.assertNotIn(line, stream)


if __name__ == "__main__":
    unittest.main()
