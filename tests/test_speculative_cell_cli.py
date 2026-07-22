from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.benchmark_speculative_cell_qualifier import _plan, _trials
from local_moe.speculative_cell_cli import _starter_plan_payload, main
from local_moe.speculative_cell_contracts import speculative_plan_from_payload
from local_moe.verified_routing_contracts import canonical_json


class SpeculativeCellCliTests(unittest.TestCase):
    def test_checked_example_plan_is_valid_and_non_authorizing(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "configs" / "speculative-cell-plan.example.json").read_text(
                encoding="utf-8"
            )
        )
        plan = speculative_plan_from_payload(payload)

        self.assertEqual(payload, _starter_plan_payload())
        self.assertEqual(plan.candidate.speculation_mode, "ngram-simple")
        self.assertEqual(plan.expected_trial_count, 32)
        self.assertEqual(plan.authority, "advisory_only")

    def test_init_materializes_self_contained_no_clobber_template(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mymoe-speculative-init-") as temp:
            destination = Path(temp) / "plan.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["init", "--out", str(destination), "--json"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "created")
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload, _starter_plan_payload())
            self.assertEqual(
                speculative_plan_from_payload(payload).expected_trial_count, 32
            )
            if os.name != "nt":
                self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                repeated_exit = main(["init", "--out", str(destination), "--json"])
            self.assertEqual(repeated_exit, 1)
            self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_inspect_rejects_another_adapter_contract(self) -> None:
        payload = _starter_plan_payload()
        execution = dict(payload["execution"])
        execution["adapter_contract_sha256"] = "0" * 64
        execution["digest"] = ""
        payload["execution"] = execution
        payload["digest"] = ""
        with tempfile.TemporaryDirectory(prefix="mymoe-speculative-inspect-") as temp:
            plan_path = Path(temp) / "plan.json"
            plan_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(["inspect", "--plan", str(plan_path), "--json"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_plan_and_trials_reject_duplicate_or_nonfinite_json(self) -> None:
        plan = _plan()
        trial = _trials(plan)[0]
        with tempfile.TemporaryDirectory(prefix="mymoe-speculative-json-") as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            trials_path = root / "trials.jsonl"
            plan_path.write_text(
                '{"schema_version":"1.0","schema_version":"1.0"}\n',
                encoding="utf-8",
            )
            trials_path.write_text(
                canonical_json(trial.payload()).replace(
                    '"sequence_index":0', '"sequence_index":NaN'
                )
                + "\n",
                encoding="utf-8",
            )

            for argv in (
                ["inspect", "--plan", str(plan_path), "--json"],
                [
                    "qualify",
                    "--plan",
                    str(root / "valid-plan.json"),
                    "--trials",
                    str(trials_path),
                    "--json",
                ],
            ):
                if argv[0] == "qualify":
                    (root / "valid-plan.json").write_text(
                        canonical_json(plan.payload()) + "\n", encoding="utf-8"
                    )
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = main(argv)
                self.assertEqual(exit_code, 1)
                self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_inspect_and_qualify_emit_content_free_receipt(self) -> None:
        plan = _plan()
        trials = _trials(plan)
        with tempfile.TemporaryDirectory(prefix="mymoe-speculative-cli-") as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            trials_path = root / "trials.jsonl"
            receipt_path = root / "receipt.json"
            plan_path.write_text(
                canonical_json(plan.payload()) + "\n", encoding="utf-8"
            )
            trials_path.write_text(
                "".join(canonical_json(trial.payload()) + "\n" for trial in trials),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                inspect_exit = main(["inspect", "--plan", str(plan_path), "--json"])
            self.assertEqual(inspect_exit, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "valid")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                qualify_exit = main(
                    [
                        "qualify",
                        "--plan",
                        str(plan_path),
                        "--trials",
                        str(trials_path),
                        "--out",
                        str(receipt_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(qualify_exit, 0)
            self.assertEqual(payload["status"], "qualified")
            self.assertFalse(payload["receipt"]["activation_authorized"])
            self.assertEqual(json.loads(receipt_path.read_text()), payload)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                repeated_exit = main(
                    [
                        "qualify",
                        "--plan",
                        str(plan_path),
                        "--trials",
                        str(trials_path),
                        "--out",
                        str(receipt_path),
                        "--json",
                    ]
                )
            self.assertEqual(repeated_exit, 1)
            self.assertEqual(json.loads(stderr.getvalue())["status"], "error")

    def test_incomplete_evidence_has_distinct_gate_exit(self) -> None:
        plan = _plan()
        trials = _trials(plan)[:-1]
        with tempfile.TemporaryDirectory(prefix="mymoe-speculative-cli-") as temp:
            root = Path(temp)
            plan_path = root / "plan.json"
            trials_path = root / "trials.jsonl"
            plan_path.write_text(
                canonical_json(plan.payload()) + "\n", encoding="utf-8"
            )
            trials_path.write_text(
                "".join(canonical_json(trial.payload()) + "\n" for trial in trials),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "qualify",
                        "--plan",
                        str(plan_path),
                        "--trials",
                        str(trials_path),
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "abstained")


if __name__ == "__main__":
    unittest.main()
