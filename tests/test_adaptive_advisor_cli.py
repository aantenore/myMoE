from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from local_moe import adaptive_advisor_cli
from local_moe.adaptive_advisor_cli import (
    AdvisorCliError,
    MAX_EVALUATION_CONTRACT_BYTES,
    MAX_TASK_BYTES,
    _validate_windows_output_name,
    _write_output,
    _write_output_windows,
    main as advisor_main,
    render_human,
)
from local_moe.adaptive_advisor_service import (
    AdaptiveAdvisorReceipt,
    AdvisorServiceError,
)
from local_moe.adaptive_selector import (
    AdaptiveAdvice,
    CandidateAssessment,
    build_adaptive_request,
)
from local_moe.cli import main as mymoe_main
from local_moe.resource_snapshot import build_resource_snapshot


ROOT = Path(__file__).resolve().parents[1]
SHA_A, SHA_B, SHA_C = (character * 64 for character in "abc")
EVALUATED_AT = "2026-07-21T12:00:00+00:00"


def _arguments(
    contract: Path,
    *,
    catalog: Path | None = None,
    task_source: tuple[str, ...] = ("--task", "private task"),
) -> list[str]:
    return [
        "--catalog",
        str(catalog or ROOT / "configs" / "adaptive-cells.example.json"),
        *task_source,
        "--workload",
        "coding.edit",
        "--capability",
        "code",
        "--tool-surface",
        "workspace",
        "--risk-class",
        "low",
        "--context-tokens",
        "4096",
        "--evaluation-contract",
        str(contract),
        "--goal",
        "balanced",
    ]


def _snapshot():
    return build_resource_snapshot(
        system="Linux",
        os_release="test",
        machine="x86_64",
        cpu_count=8,
        cpu_identity_sha256=SHA_A,
        memory_topology="system",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        effective_memory_limit_bytes=16 * 1024**3,
        swap_used_bytes=0,
        accelerator_kind="none",
        accelerator_identity_sha256=None,
        runtime_environment_sha256=SHA_B,
        captured_at=EVALUATED_AT,
        source={"fixture": "cli"},
    )


def _receipt(
    task_text: str = "private task", *, reasons: tuple[str, ...] = ()
) -> AdaptiveAdvisorReceipt:
    request = build_adaptive_request(
        exact_request_fingerprint=hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
        intent_family_sha256=None,
        workload_id="coding.edit",
        required_capabilities=("code",),
        required_tool_surfaces=("workspace",),
        risk_class="low",
        required_context_tokens=4096,
        evaluation_contract_sha256=SHA_A,
        profile="balanced",
        evaluated_at=EVALUATED_AT,
    )
    if reasons:
        candidate = CandidateAssessment(
            cell_id="coder-local",
            passport_sha256=SHA_A,
            hard_eligible=False,
            pareto_eligible=False,
            rejection_codes=reasons,
            success_rate=None,
            p95_latency_ms=None,
            memory_pool=None,
            placement=None,
            effective_peak_host_memory_bytes=None,
            effective_peak_unified_memory_bytes=None,
            effective_peak_accelerator_memory_bytes=None,
            utility=None,
        )
        advice = AdaptiveAdvice(
            catalog_sha256=SHA_B,
            request_sha256=request.digest,
            resource_snapshot_sha256=_snapshot().digest,
            evaluated_at=EVALUATED_AT,
            profile="balanced",
            status="abstained",
            selected_cell_id=None,
            candidates=(candidate,),
            reason_codes=("advisory_only", "no_eligible_cell"),
        )
        display_state = (
            "not_available_now"
            if set(reasons).issubset({"model_unavailable", "capability_gap"})
            else "not_enough_evidence"
        )
    else:
        candidate = CandidateAssessment(
            cell_id="coder-local",
            passport_sha256=SHA_A,
            hard_eligible=True,
            pareto_eligible=True,
            rejection_codes=(),
            success_rate=0.95,
            p95_latency_ms=250,
            memory_pool="host",
            placement="cpu",
            effective_peak_host_memory_bytes=2 * 1024**3,
            effective_peak_unified_memory_bytes=None,
            effective_peak_accelerator_memory_bytes=None,
            utility=0.9,
        )
        advice = AdaptiveAdvice(
            catalog_sha256=SHA_B,
            request_sha256=request.digest,
            resource_snapshot_sha256=_snapshot().digest,
            evaluated_at=EVALUATED_AT,
            profile="balanced",
            status="recommended",
            selected_cell_id="coder-local",
            candidates=(candidate,),
            reason_codes=("advisory_only", "pareto_frontier_selected"),
        )
        display_state = "recommended_now"
    return AdaptiveAdvisorReceipt(
        request=request,
        advice=advice,
        task_chars=len(task_text),
        display_state=display_state,
    )


def _evaluate_from_task(**kwargs):
    return _receipt(kwargs["task_text"])


class AdaptiveAdvisorCliTests(unittest.TestCase):
    def test_windows_output_names_reject_ambiguous_or_reserved_forms(self) -> None:
        for name in (
            "receipt.json:stream",
            "receipt.json.",
            "receipt.json ",
            "CON",
            "nul.json",
            "COM1.txt",
            "LPT9",
            "a" * 256,
        ):
            with self.subTest(name=name), self.assertRaises(AdvisorCliError):
                _validate_windows_output_name(name)

        _validate_windows_output_name("advisor-receipt.json")

    @unittest.skipIf(os.name == "nt", "portable mocked Win32 race fixture")
    def test_windows_publish_rejects_same_bytes_from_a_replacement_identity(
        self,
    ) -> None:
        class FakeIdentity:
            def __init__(self, value: str) -> None:
                self.value = value
                self.is_reparse_point = False

            def same_file_as(self, other: object) -> bool:
                return isinstance(other, FakeIdentity) and self.value == other.value

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "receipt.json"
            staged = FakeIdentity("staged")
            replacement = FakeIdentity("replacement")

            def open_nofollow(path, *, directory, writable, share_delete):
                del writable, share_delete
                descriptor = os.open(path, os.O_RDONLY)
                identity = (
                    FakeIdentity(f"directory:{path}") if directory else replacement
                )
                return descriptor, identity

            def replace_before_move(source, destination):
                Path(source).unlink()
                Path(destination).write_bytes(b"safe receipt\n")

            with (
                patch(
                    "local_moe._win32_fs.identity_from_fd",
                    return_value=staged,
                ),
                patch(
                    "local_moe._win32_fs.open_nofollow_fd",
                    side_effect=open_nofollow,
                ),
                patch(
                    "local_moe._win32_fs.move_no_replace",
                    side_effect=replace_before_move,
                ),
                self.assertRaises(AdvisorCliError) as raised,
            ):
                _write_output_windows(target, b"safe receipt\n")

            self.assertEqual(raised.exception.code, "output_verification_failed")
            self.assertEqual(target.read_bytes(), b"safe receipt\n")

    @unittest.skipIf(os.name == "nt", "POSIX directory-fd publication only")
    def test_parent_swap_cannot_redirect_publish_or_cleanup_to_a_victim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_parent = root / "output"
            replacement = root / "replacement"
            moved_original = root / "moved-original"
            output_parent.mkdir()
            replacement.mkdir()
            victim = replacement / "receipt.json"
            victim.write_text("do-not-touch", encoding="utf-8")
            real_link = os.link

            def swap_parent_then_link(*args, **kwargs):
                output_parent.rename(moved_original)
                replacement.rename(output_parent)
                return real_link(*args, **kwargs)

            with (
                patch(
                    "local_moe.adaptive_advisor_cli.os.link",
                    side_effect=swap_parent_then_link,
                ),
                self.assertRaises(AdvisorCliError) as raised,
            ):
                _write_output(
                    output_parent / "receipt.json",
                    b"safe receipt\n",
                    protected_inputs=(),
                )

            self.assertEqual(raised.exception.code, "output_parent_changed")
            self.assertEqual(
                (output_parent / "receipt.json").read_text(encoding="utf-8"),
                "do-not-touch",
            )
            self.assertEqual(
                list(moved_original.iterdir()),
                [],
            )
            self.assertFalse(
                any(path.name.endswith(".tmp") for path in moved_original.iterdir())
            )

    @unittest.skipIf(os.name == "nt", "POSIX directory-fd publication only")
    def test_parent_moved_below_protected_root_after_link_is_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_parent = root / "output"
            protected_root = root / "protected"
            moved_output = protected_root / "captured-output"
            output_parent.mkdir()
            protected_root.mkdir()
            real_link = os.link

            def link_then_move_parent(*args, **kwargs):
                result = real_link(*args, **kwargs)
                output_parent.rename(moved_output)
                return result

            with (
                patch(
                    "local_moe.adaptive_advisor_cli.os.link",
                    side_effect=link_then_move_parent,
                ),
                self.assertRaises(AdvisorCliError) as raised,
            ):
                _write_output(
                    output_parent / "receipt.json",
                    b"safe receipt\n",
                    protected_inputs=(),
                    protected_roots=(protected_root,),
                )

            self.assertEqual(raised.exception.code, "output_parent_changed")
            self.assertEqual(list(moved_output.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "POSIX directory-fd publication only")
    def test_protected_root_alias_is_rejected_after_output_parent_is_pinned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_parent = root / "output"
            protected_root = root / "protected"
            moved_output = root / "moved-output"
            output_parent.mkdir()
            protected_root.mkdir()

            real_open_chain = adaptive_advisor_cli._open_posix_directory_chain

            def swap_before_output_pin(path: Path):
                if Path(path).name == output_parent.name:
                    output_parent.rename(moved_output)
                    protected_root.rename(output_parent)
                return real_open_chain(path)

            with (
                patch(
                    "local_moe.adaptive_advisor_cli._open_posix_directory_chain",
                    side_effect=swap_before_output_pin,
                ),
                self.assertRaises(AdvisorCliError) as raised,
            ):
                _write_output(
                    output_parent / "receipt.json",
                    b"safe receipt\n",
                    protected_inputs=(),
                    protected_roots=(protected_root,),
                )

            self.assertEqual(raised.exception.code, "output_path_conflict")
            self.assertFalse((output_parent / "receipt.json").exists())
            self.assertFalse((moved_output / "receipt.json").exists())

    @unittest.skipIf(os.name == "nt", "POSIX directory-fd publication only")
    def test_direct_output_below_protected_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protected_root = Path(temporary) / "protected"
            protected_root.mkdir()
            target = protected_root / "receipt.json"

            with self.assertRaises(AdvisorCliError) as raised:
                _write_output(
                    target,
                    b"safe receipt\n",
                    protected_inputs=(),
                    protected_roots=(protected_root,),
                )

            self.assertEqual(raised.exception.code, "output_path_conflict")
            self.assertFalse(target.exists())

    @unittest.skipIf(os.name == "nt", "POSIX long-name fixture only")
    def test_long_target_name_does_not_expand_the_staging_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ("r" * 220)
            _write_output(target, b"receipt\n", protected_inputs=())
            self.assertEqual(target.read_bytes(), b"receipt\n")

    def test_root_and_advisor_help_expose_read_only_bounded_inputs(self) -> None:
        root_stdout = StringIO()
        with patch("sys.argv", ["mymoe", "--help"]), redirect_stdout(root_stdout):
            with self.assertRaises(SystemExit) as root_exit:
                mymoe_main()
        self.assertEqual(root_exit.exception.code, 0)
        self.assertIn("advisor", root_stdout.getvalue())

        advisor_stdout = StringIO()
        with (
            patch("sys.argv", ["mymoe", "advisor", "--help"]),
            redirect_stdout(advisor_stdout),
        ):
            with self.assertRaises(SystemExit) as advisor_exit:
                mymoe_main()
        self.assertEqual(advisor_exit.exception.code, 0)
        help_text = advisor_stdout.getvalue()
        for option in (
            "--catalog",
            "--task",
            "--task-file",
            "--task-stdin",
            "--risk-class",
            "--goal",
        ):
            self.assertIn(option, help_text)
        self.assertNotIn("--apply", help_text)
        self.assertNotIn("--download", help_text)

    def test_json_receipt_is_metadata_only_private_and_deterministic(self) -> None:
        secret = "SECRET user task that must never enter a receipt"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "evaluation.json"
            contract.write_text('{"suite":"local"}\n', encoding="utf-8")
            out = root / "receipt.json"
            stdout, stderr = StringIO(), StringIO()
            with (
                patch(
                    "local_moe.adaptive_advisor_cli.evaluate_advisor",
                    side_effect=_evaluate_from_task,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                status = advisor_main(
                    [
                        *_arguments(contract, task_source=("--task", secret)),
                        "--json",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertNotIn(secret, stdout.getvalue())
            self.assertNotIn(secret, out.read_text(encoding="utf-8"))
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload, json.loads(out.read_text(encoding="utf-8")))
            self.assertEqual(payload["advice"]["status"], "recommended")
            self.assertFalse(payload["advice"]["applied"])
            self.assertFalse(payload["advice"]["network_used"])
            self.assertEqual(payload["advice"]["model_invocations"], 0)
            self.assertEqual(
                payload["request"]["exact_request_fingerprint"],
                hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(payload["task_chars"], len(secret))
            self.assertNotIn("task", payload["request"])
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)

    def test_task_file_and_stdin_are_bounded_mutually_exclusive_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "evaluation.json"
            contract.write_text("{}\n", encoding="utf-8")
            task_file = root / "task.txt"
            task_file.write_text("file secret", encoding="utf-8")

            with patch(
                "local_moe.adaptive_advisor_cli.evaluate_advisor",
                side_effect=_evaluate_from_task,
            ) as evaluate:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    file_status = advisor_main(
                        _arguments(
                            contract,
                            task_source=("--task-file", str(task_file)),
                        )
                    )
                self.assertEqual(file_status, 0)
                self.assertEqual(evaluate.call_args.kwargs["task_text"], "file secret")

                with (
                    patch("sys.stdin", StringIO("stdin secret")),
                    redirect_stdout(StringIO()),
                    redirect_stderr(StringIO()),
                ):
                    stdin_status = advisor_main(
                        _arguments(contract, task_source=("--task-stdin",))
                    )
                self.assertEqual(stdin_status, 0)
                self.assertEqual(evaluate.call_args.kwargs["task_text"], "stdin secret")

            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                duplicate_status = advisor_main(
                    _arguments(
                        contract,
                        task_source=(
                            "--task",
                            "hidden",
                            "--task-file",
                            str(task_file),
                        ),
                    )
                )
            self.assertEqual(duplicate_status, 2)
            self.assertEqual(
                json.loads(stderr.getvalue())["code"], "invocation_invalid"
            )
            self.assertNotIn("hidden", stderr.getvalue())

    def test_oversize_task_file_and_stdin_fail_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "evaluation.json"
            contract.write_text("{}\n", encoding="utf-8")
            oversized = root / "oversized.txt"
            with oversized.open("wb") as handle:
                handle.truncate(MAX_TASK_BYTES + 1)

            with patch("local_moe.adaptive_advisor_cli.evaluate_advisor") as evaluate:
                file_stderr = StringIO()
                with redirect_stdout(StringIO()), redirect_stderr(file_stderr):
                    file_status = advisor_main(
                        _arguments(
                            contract,
                            task_source=("--task-file", str(oversized)),
                        )
                    )
                self.assertEqual(file_status, 2)
                self.assertEqual(
                    json.loads(file_stderr.getvalue())["code"], "task_input_invalid"
                )

                stdin_stderr = StringIO()
                with (
                    patch("sys.stdin", StringIO("x" * (MAX_TASK_BYTES + 1))),
                    redirect_stdout(StringIO()),
                    redirect_stderr(stdin_stderr),
                ):
                    stdin_status = advisor_main(
                        _arguments(contract, task_source=("--task-stdin",))
                    )
                self.assertEqual(stdin_status, 2)
                self.assertEqual(
                    json.loads(stdin_stderr.getvalue())["code"], "task_too_large"
                )
                evaluate.assert_not_called()

    def test_evaluation_contract_links_and_oversize_files_fail_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "evaluation.json"
            target.write_text("{}\n", encoding="utf-8")
            linked = root / "evaluation-link.json"
            try:
                linked.symlink_to(target)
            except OSError:
                linked = None
            oversized = root / "evaluation-large.json"
            with oversized.open("wb") as handle:
                handle.truncate(MAX_EVALUATION_CONTRACT_BYTES + 1)

            candidates = [oversized]
            if linked is not None:
                candidates.append(linked)
            for contract in candidates:
                stderr = StringIO()
                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    status = advisor_main(_arguments(contract))
                self.assertEqual(status, 2)
                self.assertEqual(
                    json.loads(stderr.getvalue())["code"],
                    "evaluation_contract_invalid",
                )

    def test_output_is_no_clobber_rejects_links_and_never_aliases_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "evaluation.json"
            contract.write_text("{}\n", encoding="utf-8")
            existing = root / "existing.json"
            existing.write_text("keep", encoding="utf-8")
            symlink = root / "linked.json"
            try:
                symlink.symlink_to(existing)
            except OSError:
                symlink = None

            with patch(
                "local_moe.adaptive_advisor_cli.evaluate_advisor",
                side_effect=_evaluate_from_task,
            ):
                for target, expected_code in (
                    (existing, "output_exists"),
                    (contract, "output_aliases_input"),
                ):
                    stderr = StringIO()
                    with redirect_stdout(StringIO()), redirect_stderr(stderr):
                        status = advisor_main(
                            [*_arguments(contract), "--out", str(target)]
                        )
                    self.assertEqual(status, 2)
                    self.assertEqual(
                        json.loads(stderr.getvalue())["code"], expected_code
                    )
                if symlink is not None:
                    stderr = StringIO()
                    with redirect_stdout(StringIO()), redirect_stderr(stderr):
                        status = advisor_main(
                            [*_arguments(contract), "--out", str(symlink)]
                        )
                    self.assertEqual(status, 2)
                    self.assertEqual(
                        json.loads(stderr.getvalue())["code"], "output_exists"
                    )
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_errors_are_stable_sanitized_and_unknown_reasons_are_explicit(self) -> None:
        secret = "PRIVATE-CONTENT-MUST-NOT-LEAK"
        with tempfile.TemporaryDirectory() as temporary:
            contract = Path(temporary) / "evaluation.json"
            contract.write_text("{}\n", encoding="utf-8")
            stderr = StringIO()
            with (
                patch(
                    "local_moe.adaptive_advisor_cli.evaluate_advisor",
                    side_effect=AdvisorServiceError(
                        "catalog_invalid",
                        "Adaptive cell catalog could not be verified.",
                    ),
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(stderr),
            ):
                status = advisor_main(
                    _arguments(contract, task_source=("--task", secret))
                )
        self.assertEqual(status, 2)
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["error"], "advisor_error")
        self.assertEqual(error["code"], "catalog_invalid")
        self.assertNotIn(secret, stderr.getvalue())

        human = render_human(_receipt(reasons=("future_boundary",)))
        self.assertIn("unrecognized boundary (future_boundary)", human)
        for label in (
            "Receipt SHA-256:",
            "Advice SHA-256:",
            "Request SHA-256:",
            "Catalog SHA-256:",
            "Snapshot SHA-256:",
        ):
            self.assertIn(label, human)

    def test_example_catalog_abstains_without_treating_it_as_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = Path(temporary) / "evaluation.json"
            contract.write_text("{}\n", encoding="utf-8")
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = advisor_main(_arguments(contract))

        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(
            stdout.getvalue().startswith(
                ("Not enough verified evidence", "Not available now")
            )
        )
        self.assertIn("Boundary: this receipt is read-only advice", stdout.getvalue())
        self.assertIn(
            "No model invocation, network access, download", stdout.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
