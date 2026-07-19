from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from local_moe import paired_execution_cli as cli
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.paired_attestation_directory import (
    DirectoryPairedAttestationProducer,
)
from local_moe.paired_execution_store import PairedExecutionStore
from local_moe.verified_routing_contracts import VerifiedRoutingError
from tests.test_paired_execution_store import _root


class PairedExecutionCliTests(unittest.TestCase):
    def test_status_missing_is_known_and_does_not_load_provider_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            cli, "_load_task"
        ) as load_task, patch.object(cli, "_load_bridge_config") as load_bridge:
            code, stdout, stderr = _main(
                [
                    "status",
                    "--run-dir",
                    str(Path(temporary) / "missing"),
                    "--json",
                ]
            )

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_SUCCESS)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["status"], "missing")
        self.assertEqual(payload["run"]["state"], "missing")
        load_task.assert_not_called()
        load_bridge.assert_not_called()

    def test_status_reports_recovered_claim_as_indeterminate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            owner = PairedExecutionStore(run_dir)
            owner.prepare(_root())
            owner.claim("A")

            code, stdout, _ = _main(
                ["--json", "status", "--run-dir", str(run_dir)]
            )

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["status"], "indeterminate")
        self.assertEqual(payload["run"]["current_claim"]["slot"], "A")

    def test_run_selects_exact_case_and_uses_only_embedded_pricing(self) -> None:
        inputs = _run_inputs()
        result = SimpleNamespace(
            state="complete",
            metadata_payload=lambda: {
                "state": "complete",
                "root": {"run_id": "paired-run-" + "a" * 64},
            },
        )
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(cli, "_invoke_run", return_value=result) as invoke:
            code, stdout, stderr = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_SUCCESS)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["status"], "complete")
        kwargs = invoke.call_args.kwargs
        self.assertEqual(
            set(kwargs),
            {
                "task",
                "plan",
                "case",
                "source_workspace",
                "pricing",
                "run_store",
                "outcome_store",
                "executor",
            },
        )
        self.assertIs(kwargs["case"], inputs.case)
        self.assertIs(kwargs["pricing"], inputs.pricing)
        self.assertEqual(kwargs["source_workspace"], Path(temporary) / "source")
        self.assertEqual(kwargs["run_store"], Path(temporary) / "private" / "run")
        self.assertEqual(
            kwargs["outcome_store"],
            Path(temporary) / "private" / "holdout.jsonl",
        )

    def test_run_composes_public_trust_cas_and_directory_sidecar(self) -> None:
        inputs = _run_inputs()
        result = SimpleNamespace(
            state="complete",
            metadata_payload=lambda: {"state": "complete", "root": None},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _private_exchange(root / "exchange")
            cas = ContentAddressedStore(root / "cas")
            trust = object()
            workflow = SimpleNamespace(
                state=SimpleNamespace(cas_path=cas.root),
                trust=trust,
            )
            with _patched_inputs(inputs), patch.object(
                cli,
                "_load_workflow_config",
                return_value=workflow,
            ) as load_workflow, patch.object(
                cli,
                "_build_executor",
                return_value=inputs.executor,
            ) as build_executor, patch.object(
                cli,
                "_invoke_run",
                return_value=result,
            ):
                code, _, _ = _main(
                    _run_argv(
                        temporary,
                        workflow_config=root / "workflow.json",
                        exchange_dir=exchange,
                    )
                )

        self.assertEqual(code, cli.EXIT_SUCCESS)
        load_workflow.assert_called_once_with(str(root / "workflow.json"))
        components = build_executor.call_args.kwargs
        self.assertIsInstance(
            components["attestation_producer"],
            DirectoryPairedAttestationProducer,
        )
        self.assertEqual(
            components["attestation_producer"].state_paths,
            (exchange,),
        )
        self.assertIs(components["trust_config"], trust)
        self.assertEqual(components["evidence_store"].root, cas.root)

    def test_run_requires_workflow_and_exchange_as_one_composition(self) -> None:
        inputs = _run_inputs()
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(inputs):
            root = Path(temporary)
            for arguments in (
                _run_argv(temporary, workflow_config=root / "workflow.json"),
                _run_argv(
                    temporary,
                    exchange_dir=root / "attestation-exchange",
                ),
            ):
                with self.subTest(arguments=arguments):
                    code, stdout, _ = _main(arguments)
                    payload = json.loads(stdout)
                    self.assertEqual(code, cli.EXIT_CONTRACT)
                    self.assertEqual(
                        payload["error"]["code"],
                        "signed_verifier_configuration_invalid",
                    )

    def test_run_never_creates_missing_workflow_cas(self) -> None:
        inputs = _run_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _private_exchange(root / "exchange")
            missing_cas = root / "state" / "cas"
            workflow = SimpleNamespace(
                state=SimpleNamespace(cas_path=missing_cas),
                trust=object(),
            )
            with _patched_inputs(inputs), patch.object(
                cli,
                "_load_workflow_config",
                return_value=workflow,
            ), patch.object(cli, "_invoke_run") as invoke:
                code, stdout, _ = _main(
                    _run_argv(
                        temporary,
                        workflow_config=root / "workflow.json",
                        exchange_dir=exchange,
                    )
                )

            payload = json.loads(stdout)
            self.assertEqual(code, cli.EXIT_CONTRACT)
            self.assertEqual(payload["error"]["code"], "input_or_config_invalid")
            self.assertFalse(missing_cas.exists())
            self.assertFalse((root / "private" / "run").exists())
            self.assertFalse((root / "private" / "holdout.jsonl").exists())
            invoke.assert_not_called()

    def test_run_requires_signed_verifier_before_creating_state(self) -> None:
        inputs = _run_inputs()

        def reject_unsigned(task):
            raise RuntimeError("signed_verifier_required")

        inputs.executor = SimpleNamespace(preflight=reject_unsigned)
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(cli, "_invoke_run") as invoke:
            code, stdout, _ = _main(_run_argv(temporary))

            payload = json.loads(stdout)
            self.assertEqual(code, cli.EXIT_CONTRACT)
            self.assertEqual(payload["error"]["code"], "signed_verifier_required")
            self.assertFalse((Path(temporary) / "private" / "run").exists())
            self.assertFalse(
                (Path(temporary) / "private" / "holdout.jsonl").exists()
            )
            invoke.assert_not_called()

    def test_run_rejects_legacy_plan_without_embedded_pricing(self) -> None:
        inputs = _run_inputs()
        inputs.plan.pricing_contract = None
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(cli, "_build_executor") as build_executor:
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_CONTRACT)
        self.assertEqual(payload["error"]["code"], "embedded_pricing_required")
        build_executor.assert_not_called()

    def test_run_rejects_task_absent_from_frozen_plan(self) -> None:
        inputs = _run_inputs()
        inputs.case.task_fingerprint = "b" * 64
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(inputs):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_CONTRACT)
        self.assertEqual(payload["error"]["code"], "task_case_not_found")

    def test_missing_journal_after_entering_runner_is_not_retry_safe(self) -> None:
        inputs = _run_inputs()
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(cli, "_invoke_run", side_effect=OSError("unavailable")):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["error"]["code"], "run_state_unknown")

    def test_failure_after_uncheckpointed_claim_exits_one(self) -> None:
        inputs = _run_inputs()

        def fail_after_claim(**kwargs):
            store = PairedExecutionStore(kwargs["run_store"])
            store.run_dir.parent.mkdir(parents=True)
            store.prepare(_root())
            claim = store.claim("A")
            store.abandon(claim)
            raise RuntimeError("provider boundary crash")

        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(cli, "_invoke_run", side_effect=fail_after_claim):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["error"]["code"], "run_indeterminate")

    def test_unreadable_run_state_never_reports_retry_safe(self) -> None:
        inputs = _run_inputs()

        def fail_with_unreadable_state(**kwargs):
            run_dir = kwargs["run_store"]
            run_dir.parent.mkdir(parents=True)
            run_dir.write_text("not a paired-run directory", encoding="utf-8")
            raise OSError("operational failure")

        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(
            cli,
            "_invoke_run",
            side_effect=fail_with_unreadable_state,
        ):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["error"]["code"], "run_state_unknown")

    def test_started_provider_with_deleted_journal_is_not_retry_safe(self) -> None:
        inputs = _run_inputs()

        def fail_after_deleting_claim(**kwargs):
            store = PairedExecutionStore(kwargs["run_store"])
            store.run_dir.parent.mkdir(parents=True)
            store.prepare(_root())
            claim = store.claim("A")
            store.abandon(claim)
            shutil.rmtree(store.run_dir)
            raise RuntimeError("provider started and journal disappeared")

        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(
            cli,
            "_invoke_run",
            side_effect=fail_after_deleting_claim,
        ):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["error"]["code"], "run_state_unknown")

    def test_contract_error_after_runner_entry_without_journal_exits_one(self) -> None:
        inputs = _run_inputs()
        with tempfile.TemporaryDirectory() as temporary, _patched_inputs(
            inputs
        ), patch.object(
            cli,
            "_invoke_run",
            side_effect=VerifiedRoutingError("contract mismatch"),
        ):
            code, stdout, _ = _main(_run_argv(temporary))

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_INDETERMINATE)
        self.assertEqual(payload["error"]["code"], "run_state_unknown")

    def test_status_operational_failure_exits_three(self) -> None:
        with patch.object(cli, "_new_run_store", side_effect=OSError("unavailable")):
            code, stdout, _ = _main(
                ["status", "--run-dir", "/unavailable", "--json"]
            )

        payload = json.loads(stdout)
        self.assertEqual(code, cli.EXIT_OPERATIONAL)
        self.assertEqual(payload["error"]["code"], "status_operational_failure")

    def test_run_help_has_no_caller_controlled_digest_or_pricing_flags(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.build_parser().parse_args(["run", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--task", help_text)
        self.assertIn("--plan", help_text)
        self.assertIn("--outcome-store", help_text)
        self.assertIn("--workflow-config", help_text)
        self.assertIn("--attestation-exchange-dir", help_text)
        self.assertNotIn("--pricing", help_text)
        self.assertNotIn("--operation-digest", help_text)
        self.assertNotIn("--lifecycle-digest", help_text)


def _main(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _run_inputs() -> SimpleNamespace:
    fingerprint = "a" * 64
    task = SimpleNamespace(
        task_fingerprint=fingerprint,
        capability_demand=SimpleNamespace(risk_class="read_only"),
    )
    case = SimpleNamespace(
        task_fingerprint=fingerprint,
        baseline_route="local_then_verify",
        candidate_route="local",
    )
    pricing = object()
    plan = SimpleNamespace(cases=(case,), pricing_contract=pricing)
    app_config = SimpleNamespace(
        permissions=SimpleNamespace(
            assistant_bridge_execution_policy="hybrid_receipt_confirmation",
            default_write_policy="approval_required",
        )
    )
    return SimpleNamespace(
        task=task,
        case=case,
        plan=plan,
        pricing=pricing,
        app_config=app_config,
        bridge_config=object(),
        executor=SimpleNamespace(preflight=lambda task: None),
    )


def _patched_inputs(inputs: SimpleNamespace):
    context = ExitStack()
    context.enter_context(
        patch.object(cli, "_load_app_config", return_value=inputs.app_config)
    )
    context.enter_context(patch.object(cli, "_load_task", return_value=inputs.task))
    context.enter_context(patch.object(cli, "_load_plan", return_value=inputs.plan))
    context.enter_context(
        patch.object(cli, "_load_bridge_config", return_value=inputs.bridge_config)
    )
    context.enter_context(
        patch.object(cli, "_build_executor", return_value=inputs.executor)
    )
    return context


def _run_argv(
    temporary: str,
    *,
    workflow_config: Path | None = None,
    exchange_dir: Path | None = None,
) -> list[str]:
    root = Path(temporary)
    arguments = [
        "run",
        "--task",
        str(root / "task.json"),
        "--plan",
        str(root / "plan.json"),
        "--bridge-config",
        str(root / "bridge.json"),
        "--app-config",
        str(root / "app.json"),
        "--workspace",
        str(root / "source"),
        "--run-dir",
        str(root / "private" / "run"),
        "--outcome-store",
        str(root / "private" / "holdout.jsonl"),
    ]
    if workflow_config is not None:
        arguments.extend(("--workflow-config", str(workflow_config)))
    if exchange_dir is not None:
        arguments.extend(("--attestation-exchange-dir", str(exchange_dir)))
    arguments.append("--json")
    return arguments


def _private_exchange(path: Path) -> Path:
    for directory in (path, path / "requests", path / "responses"):
        directory.mkdir(parents=True, mode=0o700)
        if os.name == "posix":
            directory.chmod(0o700)
    return path.resolve(strict=True)


if __name__ == "__main__":
    unittest.main()
