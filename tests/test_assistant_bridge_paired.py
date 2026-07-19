from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from local_moe import assistant_bridge as assistant_bridge_module
from local_moe.assistant_bridge import (
    build_assistant_task,
    build_codex_command_plan,
    execute_codex_command,
)
from local_moe.paired_execution_bridge import (
    AssistantBridgePairedArmExecutor,
    PairedArmPlan,
    paired_arm_operation_sha256,
)
from local_moe.paired_execution_contracts import PairedRunRoot
from local_moe.paired_execution_store import PairedExecutionStore
from local_moe.route_signals import MetadataTaskSignalProvider
from tests.paired_attestation_fakes import build_signed_paired_executor
from tests.test_assistant_bridge import _fake_bridge, _fake_environment, _read_jsonl


LIFECYCLE_CONFIG_SHA256 = "d" * 64
SENTINEL = "PRIVATE-CONFIRMATION-SENTINEL"


class AssistantBridgePairedArmExecutorTests(unittest.TestCase):
    def test_exact_route_hooks_are_not_public_candidate_api(self) -> None:
        with _fake_bridge() as fixture:
            generator = fixture.runner.candidate_generator()

            self.assertFalse(hasattr(generator, "plan_paired_evidence_arm"))
            self.assertFalse(hasattr(generator, "run_paired_evidence_arm"))

    def test_candidate_first_plan_issues_only_its_own_confirmation(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture),
            tempfile.TemporaryDirectory() as temporary,
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="balanced",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            signed = build_signed_paired_executor(
                fixture,
                Path(temporary) / "evidence",
            )
            executor = signed.executor
            source = executor.snapshot_source(fixture.root)
            signals = MetadataTaskSignalProvider().signals_from_metadata(
                task.metadata_payload()
            )
            root = PairedRunRoot.build(
                plan_sha256="1" * 64,
                case_sha256="2" * 64,
                task_fingerprint=task.task_fingerprint,
                normalized_item_sha256="3" * 64,
                source_snapshot_sha256=source.fingerprint,
                bridge_config_sha256=executor.bridge_config_sha256,
                executor_config_sha256=executor.configuration_sha256,
                execution_harness_sha256=executor.execution_harness_sha256,
                lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                signals_sha256=signals.signals_sha256,
                runner_sha256="4" * 64,
                runner_source_sha256="6" * 64,
                pricing_sha256="5" * 64,
                run_instance_nonce="7" * 64,
                order="BA",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(root)
            binding = store.binding_for(store.claim(root.slots[0]))

            with patch.object(
                fixture.runner.state_ledger,
                "issue_confirmation",
                wraps=fixture.runner.state_ledger.issue_confirmation,
            ) as issue_confirmation:
                plan = executor.plan_arm(
                    task,
                    source_workspace=fixture.root,
                    source_snapshot=source,
                    signals=signals,
                    lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                    baseline_route="local_then_verify",
                    slot=root.slots[0],
                    permit=binding,
                )

            self.assertEqual(plan.slot.arm, "candidate")
            self.assertEqual(issue_confirmation.call_count, 1)
            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_codex_execution_maps_json_usage_without_persisting_stdout(self) -> None:
        stdout_sentinel = "PRIVATE-STDOUT-SENTINEL"
        stdout = (
            json.dumps(
                {"type": "item.completed", "message": stdout_sentinel},
                separators=(",", ":"),
            )
            + "\n"
            + json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 20_403,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 0,
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with _fake_bridge() as fixture:
            prompt = "bounded usage"
            output_path = fixture.root / "usage-output.txt"
            plan = build_codex_command_plan(
                fixture.config.local,
                prompt=prompt,
                workspace=fixture.root,
                output_path=output_path,
                runtime_policy=fixture.config.runtime,
            )

            def execute(*args, **kwargs):
                output_path.write_text("VERIFIED", encoding="utf-8")
                return SimpleNamespace(
                    ok=True,
                    code="completed",
                    returncode=0,
                    duration_ms=1,
                    stdout=stdout,
                    stderr=b"",
                    stdout_bytes=len(stdout),
                    stderr_bytes=0,
                    stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                    stderr_sha256=hashlib.sha256(b"").hexdigest(),
                    stdout_truncated=False,
                )

            with patch.object(
                assistant_bridge_module,
                "execute_process",
                side_effect=execute,
            ):
                result = execute_codex_command(
                    plan,
                    prompt=prompt,
                    output_path=output_path,
                    timeout_seconds=10,
                )

        self.assertEqual(result.prompt_tokens, 20_403)
        self.assertEqual(result.completion_tokens, 5)
        self.assertNotIn(stdout_sentinel, json.dumps(result.metadata_payload()))

    def test_claim_bound_plan_runs_without_source_apply_and_sanitizes_ticket(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED temporary candidate",
                write_relative="tracked.txt",
                write_content="temporary-only\n",
            ),
            tempfile.TemporaryDirectory() as temporary,
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="balanced",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            signed = build_signed_paired_executor(
                fixture,
                Path(temporary) / "evidence",
            )
            executor = signed.executor
            source = executor.snapshot_source(fixture.root)
            signals = MetadataTaskSignalProvider().signals_from_metadata(
                task.metadata_payload()
            )
            root = PairedRunRoot.build(
                plan_sha256="1" * 64,
                case_sha256="2" * 64,
                task_fingerprint=task.task_fingerprint,
                normalized_item_sha256="3" * 64,
                source_snapshot_sha256=source.fingerprint,
                bridge_config_sha256=executor.bridge_config_sha256,
                executor_config_sha256=executor.configuration_sha256,
                execution_harness_sha256=executor.execution_harness_sha256,
                lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                signals_sha256=signals.signals_sha256,
                runner_sha256="4" * 64,
                runner_source_sha256="6" * 64,
                pricing_sha256="5" * 64,
                run_instance_nonce="7" * 64,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(root)
            binding = store.binding_for(store.claim("A"))
            plan = executor.plan_arm(
                task,
                source_workspace=fixture.root,
                source_snapshot=source,
                signals=signals,
                lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                baseline_route="local_then_verify",
                slot=root.slots[0],
                permit=binding,
            )
            self.assertEqual(
                plan.operation_sha256,
                paired_arm_operation_sha256(binding),
            )
            poisoned = PairedArmPlan(
                slot=plan.slot,
                permit=plan.permit,
                signals=plan.signals,
                operation_sha256=plan.operation_sha256,
                confirmation=SENTINEL,
                bridge_plan={
                    **dict(plan.bridge_plan),
                    "confirmation_id": SENTINEL,
                    "confirmation": {"token": SENTINEL},
                    "authority": {
                        **dict(plan.bridge_plan["authority"]),  # type: ignore[arg-type]
                        "secret": SENTINEL,
                    },
                },
            )
            public = json.dumps(poisoned.metadata_payload())
            self.assertNotIn(SENTINEL, public)
            self.assertNotIn("confirmation_id", public)

            with patch.object(
                fixture.runner,
                "_apply_verified_candidate",
            ) as apply_candidate:
                result = executor.run_arm(
                    task,
                    source_workspace=fixture.root,
                    source_snapshot=source,
                    signals=signals,
                    lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                    baseline_route="local_then_verify",
                    plan=plan,
                )
                apply_candidate.assert_not_called()
            executor.assert_source_unchanged(source)
            invocations = _read_jsonl(fixture.log)
            persisted = b"".join(
                path.read_bytes()
                for path in sorted(
                    (signed.evidence_store.root / "objects" / "sha256").rglob("*")
                )
                if path.is_file()
            )

        self.assertEqual(result.receipt.route, "local_then_verify")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(signed.producer.calls), 1)
        self.assertNotEqual(
            signed.producer.calls[0][1].resolve(),
            fixture.root.resolve(),
        )
        self.assertEqual(
            [item.id for item in result.verification if item.kind == "external"],
            ["focused-tests"],
        )
        self.assertNotIn(b"temporary-only", persisted)
        self.assertNotIn(b"initial\n", persisted)
        self.assertEqual(
            [item["mode"] for item in invocations],
            ["local"],
        )

    def test_plan_rejects_a_permit_for_another_slot_before_invocation(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture),
            tempfile.TemporaryDirectory() as temporary,
        ):
            task = build_assistant_task(
                "Change one file.",
                profile="balanced",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            executor = build_signed_paired_executor(
                fixture,
                Path(temporary) / "evidence",
            ).executor
            source = executor.snapshot_source(fixture.root)
            signals = MetadataTaskSignalProvider().signals_from_metadata(
                task.metadata_payload()
            )
            root = PairedRunRoot.build(
                plan_sha256="1" * 64,
                case_sha256="2" * 64,
                task_fingerprint=task.task_fingerprint,
                normalized_item_sha256="3" * 64,
                source_snapshot_sha256=source.fingerprint,
                bridge_config_sha256=executor.bridge_config_sha256,
                executor_config_sha256=executor.configuration_sha256,
                execution_harness_sha256=executor.execution_harness_sha256,
                lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                signals_sha256=signals.signals_sha256,
                runner_sha256="4" * 64,
                runner_source_sha256="6" * 64,
                pricing_sha256="5" * 64,
                run_instance_nonce="7" * 64,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(root)
            binding = store.binding_for(store.claim("A"))

            with self.assertRaisesRegex(ValueError, "permit"):
                executor.plan_arm(
                    task,
                    source_workspace=fixture.root,
                    source_snapshot=source,
                    signals=signals,
                    lifecycle_config_sha256=LIFECYCLE_CONFIG_SHA256,
                    baseline_route="local_then_verify",
                    slot=root.slots[1],
                    permit=binding,
                )
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(invocations, [])


if __name__ == "__main__":
    unittest.main()
