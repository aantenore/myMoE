from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import stat
from io import StringIO
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.benchmark_runtime_binding import _Fixture
from local_moe import runtime_binding_cli
from local_moe.cli import main as mymoe_main
from local_moe.runtime_binding_cli import main as binding_main
from local_moe.runtime_binding_inspector import (
    RuntimeBindingInspectionError,
    inspect_cell_binding as real_inspect_cell_binding,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _bundle(*, status: str = "verified") -> SimpleNamespace:
    reasons = () if status == "verified" else ("component_digest_changed",)
    manifest = SimpleNamespace(cell_id="coder-local", digest=SHA_A)
    receipt = SimpleNamespace(
        status=status,
        reason_codes=reasons,
        component_count=3,
        observed_component_count=3,
        digest=SHA_B,
    )
    payload = {
        "schema_version": "1.0",
        "contract": "BoundCellInspector",
        "request_sha256": SHA_B,
        "binding_manifest": {
            "cell_id": "coder-local",
            "components": [],
            "digest": SHA_A,
        },
        "inspection_receipt": {
            "status": status,
            "reason_codes": list(reasons),
            "network_used": False,
            "model_invocations": 0,
            "process_mutations": False,
            "authorizes_execution": False,
            "digest": SHA_B,
        },
    }
    return SimpleNamespace(
        request_sha256=SHA_B,
        publication_protected_roots=(),
        manifest=manifest,
        receipt=receipt,
        payload=lambda: payload,
    )


class RuntimeBindingCliTests(unittest.TestCase):
    def test_verified_and_abstained_have_stable_exit_codes_and_json(self) -> None:
        for status, expected_exit in (("verified", 0), ("abstained", 1)):
            stdout, stderr = StringIO(), StringIO()
            bundle = _bundle(status=status)
            with (
                patch(
                    "local_moe.runtime_binding_cli.inspect_cell_binding",
                    return_value=bundle,
                ) as inspect,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                observed = binding_main(
                    ["inspect", "--request", "request.json", "--json"]
                )

            self.assertEqual(observed, expected_exit)
            self.assertEqual(json.loads(stdout.getvalue()), bundle.payload())
            self.assertEqual(stderr.getvalue(), "")
            inspect.assert_called_once_with(
                Path("request.json").absolute(),
                publication_path=None,
            )

    def test_human_output_is_bounded_and_explains_the_boundary(self) -> None:
        sensitive_marker = "request text that must not be rendered"
        for status in ("verified", "abstained"):
            stdout, stderr = StringIO(), StringIO()
            with (
                patch(
                    "local_moe.runtime_binding_cli.inspect_cell_binding",
                    return_value=_bundle(status=status),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                observed = binding_main(
                    ["inspect", "--request", f"/tmp/{sensitive_marker}.json"]
                )

            self.assertEqual(observed, 0 if status == "verified" else 1)
            if status == "verified":
                self.assertIn("Cell binding verified", stdout.getvalue())
            else:
                self.assertIn("Cell binding inspection abstained", stdout.getvalue())
            self.assertIn("No model was started or downloaded", stdout.getvalue())
            self.assertIn("no network", stdout.getvalue())
            self.assertNotIn(sensitive_marker, stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_output_is_utf8_owner_only_atomic_and_no_clobber(self) -> None:
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text("{}", encoding="utf-8")
            target = root / "inspection.json"
            arguments = [
                "inspect",
                "--request",
                str(request),
                "--out",
                str(target),
                "--json",
            ]
            with (
                patch(
                    "local_moe.runtime_binding_cli.inspect_cell_binding",
                    return_value=bundle,
                ),
                patch(
                    "local_moe.runtime_binding_cli._publication_inputs",
                    return_value=(request,),
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(binding_main(arguments), 0)

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), bundle.payload()
            )
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(
                [item for item in root.iterdir() if item.name.startswith(".mymoe-")],
                [],
            )

            target_bytes = target.read_bytes()
            stderr = StringIO()
            with (
                patch(
                    "local_moe.runtime_binding_cli.inspect_cell_binding",
                    return_value=bundle,
                ),
                patch(
                    "local_moe.runtime_binding_cli._publication_inputs",
                    return_value=(request,),
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(stderr),
            ):
                self.assertEqual(binding_main(arguments), 2)
            self.assertEqual(target.read_bytes(), target_bytes)
            failure = json.loads(stderr.getvalue())
            self.assertEqual(failure["code"], "output_exists")

    def test_request_mutation_before_publication_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            target = fixture.root / "inspection.json"

            def inspect_then_mutate(*args, **kwargs):
                bundle = real_inspect_cell_binding(*args, **kwargs)
                fixture.request_path.write_text("{}", encoding="utf-8")
                return bundle

            stderr = StringIO()
            with (
                patch(
                    "local_moe.runtime_binding_cli.inspect_cell_binding",
                    side_effect=inspect_then_mutate,
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(stderr),
            ):
                observed = binding_main(
                    [
                        "inspect",
                        "--request",
                        str(fixture.request_path),
                        "--out",
                        str(target),
                    ]
                )

            self.assertEqual(observed, 2)
            self.assertFalse(target.exists())
            self.assertEqual(
                json.loads(stderr.getvalue())["code"],
                "request_changed_during_inspection",
            )

    @unittest.skipUnless(os.name == "posix", "POSIX link and FIFO fixtures")
    def test_output_rejects_symlink_and_special_file_without_replacing_them(
        self,
    ) -> None:
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text("{}", encoding="utf-8")
            regular = root / "existing.json"
            regular.write_text("unchanged", encoding="utf-8")
            symlink = root / "linked.json"
            symlink.symlink_to(regular)
            fifo = root / "inspection.pipe"
            os.mkfifo(fifo)

            for target in (symlink, fifo):
                with self.subTest(target=target.name):
                    stderr = StringIO()
                    with (
                        patch(
                            "local_moe.runtime_binding_cli.inspect_cell_binding",
                            return_value=bundle,
                        ),
                        patch(
                            "local_moe.runtime_binding_cli._publication_inputs",
                            return_value=(request,),
                        ),
                        redirect_stdout(StringIO()),
                        redirect_stderr(stderr),
                    ):
                        observed = binding_main(
                            [
                                "inspect",
                                "--request",
                                str(request),
                                "--out",
                                str(target),
                            ]
                        )
                    self.assertEqual(observed, 2)
                    self.assertEqual(
                        json.loads(stderr.getvalue())["code"], "output_exists"
                    )

            self.assertTrue(symlink.is_symlink())
            self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
            self.assertEqual(regular.read_text(encoding="utf-8"), "unchanged")

    def test_output_inside_inspected_runtime_or_model_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            for target in (
                fixture.runtime_root / "inspection.json",
                fixture.model_root / "inspection.json",
            ):
                with self.subTest(target=target), redirect_stdout(StringIO()):
                    stderr = StringIO()
                    with redirect_stderr(stderr):
                        observed = binding_main(
                            [
                                "inspect",
                                "--request",
                                str(fixture.request_path),
                                "--out",
                                str(target),
                            ]
                        )

                self.assertEqual(observed, 2)
                self.assertFalse(target.exists())
                self.assertEqual(
                    json.loads(stderr.getvalue())["code"],
                    "output_path_conflict",
                )

    def test_case_alias_of_model_root_cannot_receive_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            alias = fixture.root / fixture.model_root.name.upper()
            try:
                aliases_model_root = os.path.samefile(alias, fixture.model_root)
            except (FileNotFoundError, OSError):
                aliases_model_root = False
            if not aliases_model_root:
                self.skipTest("The test filesystem is case-sensitive.")
            target = alias / "inspection.json"
            stdout, stderr = StringIO(), StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                observed = binding_main(
                    [
                        "inspect",
                        "--request",
                        str(fixture.request_path),
                        "--out",
                        str(target),
                    ]
                )

            self.assertEqual(observed, 2)
            self.assertFalse(target.exists())
            self.assertEqual(
                json.loads(stderr.getvalue())["code"],
                "output_path_conflict",
            )

    def test_inspected_runtime_identity_cannot_be_moved_under_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            output_parent = fixture.root / "publication"
            target = output_parent / "inspection.json"
            real_publication_inputs = runtime_binding_cli._publication_inputs

            def move_runtime_after_boundary_check(**kwargs):
                inputs = real_publication_inputs(**kwargs)
                fixture.runtime_root.rename(output_parent)
                fixture.runtime_root.mkdir()
                return inputs

            stderr = StringIO()
            with (
                patch(
                    "local_moe.runtime_binding_cli._publication_inputs",
                    side_effect=move_runtime_after_boundary_check,
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(stderr),
            ):
                observed = binding_main(
                    [
                        "inspect",
                        "--request",
                        str(fixture.request_path),
                        "--out",
                        str(target),
                    ]
                )

            self.assertEqual(observed, 2)
            self.assertFalse(target.exists())
            self.assertEqual(
                json.loads(stderr.getvalue())["code"],
                "output_path_conflict",
            )

    def test_relative_paths_keep_the_initial_working_directory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            fixture = _Fixture(first)
            (first / "out").mkdir()
            (second / "out").mkdir()

            def inspect_then_change_cwd(*args, **kwargs):
                bundle = real_inspect_cell_binding(*args, **kwargs)
                os.chdir(second)
                return bundle

            try:
                os.chdir(first)
                with (
                    patch(
                        "local_moe.runtime_binding_cli.inspect_cell_binding",
                        side_effect=inspect_then_change_cwd,
                    ),
                    redirect_stdout(StringIO()),
                    redirect_stderr(StringIO()),
                ):
                    observed = binding_main(
                        [
                            "inspect",
                            "--request",
                            fixture.request_path.name,
                            "--out",
                            "out/inspection.json",
                        ]
                    )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(observed, 1)
            self.assertTrue((first / "out" / "inspection.json").is_file())
            self.assertFalse((second / "out" / "inspection.json").exists())

    def test_invalid_and_operational_errors_are_json_and_redacted(self) -> None:
        sensitive_marker = "SENSITIVE-REQUEST-CONTENT"
        path = f"/tmp/{sensitive_marker}.json"
        stderr = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            self.assertEqual(
                binding_main(
                    ["inspect", "--request", path, "--unknown", sensitive_marker]
                ),
                2,
            )
        invocation = stderr.getvalue()
        self.assertEqual(json.loads(invocation)["code"], "invocation_invalid")
        self.assertNotIn(sensitive_marker, invocation)
        self.assertNotIn(path, invocation)

        stderr = StringIO()
        with (
            patch(
                "local_moe.runtime_binding_cli.inspect_cell_binding",
                side_effect=RuntimeError(f"failed for {path}: {sensitive_marker}"),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            self.assertEqual(binding_main(["inspect", "--request", path]), 2)
        operational = stderr.getvalue()
        self.assertEqual(json.loads(operational)["code"], "inspection_failed")
        self.assertNotIn(sensitive_marker, operational)
        self.assertNotIn(path, operational)

        stderr = StringIO()
        with (
            patch(
                "local_moe.runtime_binding_cli.inspect_cell_binding",
                side_effect=RuntimeBindingInspectionError(
                    "request_invalid", f"failed for {path}: {sensitive_marker}"
                ),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            self.assertEqual(binding_main(["inspect", "--request", path]), 2)
        contract_failure = stderr.getvalue()
        self.assertEqual(json.loads(contract_failure)["code"], "request_invalid")
        self.assertNotIn(sensitive_marker, contract_failure)
        self.assertNotIn(path, contract_failure)

    def test_root_and_command_help_expose_only_read_only_inspection(self) -> None:
        root_stdout = StringIO()
        with patch("sys.argv", ["mymoe", "--help"]), redirect_stdout(root_stdout):
            with self.assertRaises(SystemExit) as root_exit:
                mymoe_main()
        self.assertEqual(root_exit.exception.code, 0)
        self.assertIn("cell-bind", root_stdout.getvalue())

        command_stdout = StringIO()
        with (
            patch("sys.argv", ["mymoe", "cell-bind", "--help"]),
            redirect_stdout(command_stdout),
        ):
            with self.assertRaises(SystemExit) as command_exit:
                mymoe_main()
        self.assertEqual(command_exit.exception.code, 0)
        self.assertIn("inspect", command_stdout.getvalue())
        self.assertIn("does not start or download models", command_stdout.getvalue())

        help_stdout = StringIO()
        with (
            patch("sys.argv", ["mymoe", "cell-bind", "inspect", "--help"]),
            redirect_stdout(help_stdout),
        ):
            with self.assertRaises(SystemExit) as help_exit:
                mymoe_main()
        self.assertEqual(help_exit.exception.code, 0)
        help_text = help_stdout.getvalue()
        for expected in (
            "--request",
            "--json",
            "--out",
            "does not start or download models",
            "access the network",
            "grant authorization",
        ):
            self.assertIn(expected, help_text)

        for forbidden in ("apply", "start", "execute"):
            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                self.assertEqual(binding_main([forbidden]), 2)
            self.assertEqual(
                json.loads(stderr.getvalue())["code"], "invocation_invalid"
            )


if __name__ == "__main__":
    unittest.main()
