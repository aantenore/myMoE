from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_moe.cli as cli
from local_moe.adaptive_advisor_service import evaluate_advisor
from local_moe.adaptive_execution_gate import load_adaptive_execution_policy
from local_moe.advisor_setup import materialize_advisor_workspace
from local_moe.app_config import load_app_config
from local_moe.cell_passport import load_cell_catalog
from local_moe.chat_store import FileChatStore
from local_moe.extensions import load_extension_registry
from local_moe.package_defaults import (
    resolve_advisor_config_reference,
    resolve_app_config_reference,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "adaptive-cells.json",
    "adaptive-execution-policy.json",
    "adaptive-evaluation-contract.json",
    "app.json",
    "context-policy.json",
    "moe.json",
}


class AdvisorSetupTests(unittest.TestCase):
    def test_source_version_fallback_is_safe_without_distribution_metadata(
        self,
    ) -> None:
        with patch.object(
            cli.importlib_metadata,
            "version",
            side_effect=cli.importlib_metadata.PackageNotFoundError,
        ):
            self.assertEqual(cli._distribution_version(), "unknown")

    def test_materializes_private_self_contained_zero_claim_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "advisor workspace"
            result = materialize_advisor_workspace(destination)
            app_path = destination / "app.json"
            app = load_app_config(app_path)
            catalog_path = resolve_advisor_config_reference(
                app.advisor.catalog_path,
                app_path,
            )
            evaluation_path = resolve_advisor_config_reference(
                app.advisor.evaluation_contract_path,
                app_path,
            )
            catalog = load_cell_catalog(catalog_path)
            execution_policy = load_adaptive_execution_policy(
                destination / "adaptive-execution-policy.json"
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "created")
            self.assertEqual(set(result["files"]), EXPECTED_FILES)
            self.assertEqual(result["next"]["run_from"], "workspace")
            self.assertEqual(result["next"]["advisor_argv"][0:2], ["mymoe", "advisor"])
            self.assertEqual(
                result["next"]["advisor_argv"][-3:],
                ["--json", "--out", "./advisor-receipt.json"],
            )
            self.assertEqual(
                result["next"]["web_argv"],
                ["mymoe-web", "--app-config", "./app.json"],
            )
            self.assertEqual(
                {item.name for item in destination.iterdir()}, EXPECTED_FILES
            )
            self.assertNotIn(str(destination), json.dumps(result))
            self.assertTrue(app.advisor.enabled)
            self.assertEqual(
                resolve_app_config_reference(app.default_moe_config, app_path),
                (destination / "moe.json").resolve(),
            )
            self.assertEqual(
                resolve_app_config_reference(
                    app.runtime.context_policy_config,
                    app_path,
                ),
                (destination / "context-policy.json").resolve(),
            )
            self.assertEqual(evaluation["qualification"]["status"], "not_qualified")
            self.assertEqual(evaluation["qualification"]["claims"], [])
            self.assertEqual(execution_policy.mode, "dry_run")
            self.assertEqual(execution_policy.allowed_risk_classes, ("compute_only",))
            self.assertEqual(execution_policy.max_tool_surfaces, 0)
            self.assertTrue(
                all(cell.measured.sample_count == 0 for cell in catalog.cells)
            )
            self.assertTrue(
                all(
                    cell.declaration.risk_classes == ("compute_only", "write_local")
                    for cell in catalog.cells
                )
            )
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)
                self.assertTrue(
                    all(
                        stat.S_IMODE(path.stat().st_mode) == 0o600
                        for path in destination.iterdir()
                    )
                )

    def test_zero_claim_workspace_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "advisor"
            materialize_advisor_workspace(destination)

            receipt = evaluate_advisor(
                catalog_path=destination / "adaptive-cells.json",
                evaluation_contract_path=(
                    destination / "adaptive-evaluation-contract.json"
                ),
                task_text="Summarize this local note.",
                workload_id="local-summary",
                required_capabilities=("summarization",),
                required_tool_surfaces=(),
                risk_class="compute_only",
                context_tokens=4096,
                profile="balanced",
            )

        self.assertNotEqual(receipt.display_state, "recommended_now")
        self.assertEqual(receipt.advice.status, "abstained")
        self.assertIsNone(receipt.advice.selected_cell_id)

    def test_existing_workspace_is_unchanged_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "advisor"
            materialize_advisor_workspace(destination)
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
            }

            with self.assertRaises(FileExistsError):
                materialize_advisor_workspace(destination)

            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
            }
            self.assertEqual(after, before)

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows")
    def test_existing_symlink_workspace_is_never_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "marker.txt"
            marker.write_text("unchanged\n", encoding="utf-8")
            destination = root / "advisor"
            destination.symlink_to(outside, target_is_directory=True)

            with self.assertRaises(FileExistsError):
                materialize_advisor_workspace(destination)

            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged\n")
            self.assertEqual({path.name for path in outside.iterdir()}, {"marker.txt"})

    def test_failure_rolls_back_only_files_created_by_this_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "advisor"
            calls = 0
            from local_moe import advisor_setup

            real_write = advisor_setup._write_exclusive_bytes

            def fail_second_write(workspace, name, content):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected write failure")
                return real_write(workspace, name, content)

            with (
                patch(
                    "local_moe.advisor_setup._write_exclusive_bytes",
                    side_effect=fail_second_write,
                ),
                self.assertRaises(OSError),
            ):
                materialize_advisor_workspace(destination)

            if os.name == "nt":
                self.assertTrue(destination.is_dir())
                self.assertEqual(list(destination.iterdir()), [])
            else:
                self.assertFalse(destination.exists())

    @unittest.skipIf(os.name == "nt", "POSIX directory handles are tested here")
    def test_root_replacement_cannot_redirect_failure_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "advisor"
            moved = root / "moved-original"
            victim = root / "victim"
            victim.mkdir()
            marker = victim / "app.json"
            marker.write_text("victim stays untouched\n", encoding="utf-8")
            calls = 0
            from local_moe import advisor_setup

            real_write = advisor_setup._write_exclusive_bytes

            def replace_after_first_write(workspace, name, content):
                nonlocal calls
                calls += 1
                if calls == 1:
                    real_write(workspace, name, content)
                    destination.rename(moved)
                    destination.symlink_to(victim, target_is_directory=True)
                    return None
                raise OSError("injected write failure after root replacement")

            with (
                patch(
                    "local_moe.advisor_setup._write_exclusive_bytes",
                    side_effect=replace_after_first_write,
                ),
                self.assertRaises(OSError),
            ):
                materialize_advisor_workspace(destination)

            self.assertTrue(destination.is_symlink())
            self.assertEqual(
                marker.read_text(encoding="utf-8"), "victim stays untouched\n"
            )
            self.assertEqual({path.name for path in victim.iterdir()}, {"app.json"})
            self.assertTrue(moved.is_dir())
            self.assertEqual(list(moved.iterdir()), [])

    def test_parent_segments_are_rejected_before_creating_any_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = root / "ghost"
            destination = ghost / ".." / "advisor"

            with self.assertRaisesRegex(ValueError, "invalid"):
                materialize_advisor_workspace(destination)

            self.assertFalse(ghost.exists())
            self.assertFalse((root / "advisor").exists())

    def test_starter_is_isolated_from_a_hostile_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hostile = root / "hostile"
            ambient_skill = hostile / "skills" / "ambient-secret"
            ambient_skill.mkdir(parents=True)
            (ambient_skill / "SKILL.md").write_text(
                "---\nname: ambient-secret\ndescription: must not load\n---\n",
                encoding="utf-8",
            )
            hostile_configs = hostile / "configs"
            hostile_configs.mkdir()
            (hostile_configs / "tools.json").write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "ambient-tool",
                                "description": "must not load",
                                "risk_class": "compute_only",
                                "side_effects": "none",
                                "enabled": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "advisor"
            previous = Path.cwd()
            try:
                os.chdir(hostile)
                result = materialize_advisor_workspace(destination)
                app = load_app_config(destination / "app.json")
                registry = load_extension_registry(
                    plugins_dir=app.extensions.plugins_dir,
                    skills_dir=app.extensions.skills_dir,
                    tools_config=app.extensions.tools_config,
                    mcp_config=app.extensions.mcp_config,
                    cron_config=app.extensions.cron_config,
                )
                FileChatStore(Path(app.runtime.work_dir) / "chats.json").create_session(
                    title="workspace isolation"
                )
            finally:
                os.chdir(previous)

            owned_paths = (
                app.default_moe_config,
                app.runtime.work_dir,
                app.runtime.context_policy_config,
                app.runtime.profile_dir,
                app.runtime.evaluation_dir,
                app.extensions.plugins_dir,
                app.extensions.skills_dir,
                app.extensions.tools_config,
                app.extensions.mcp_config,
                app.extensions.cron_config,
                app.advisor.catalog_path,
                app.advisor.evaluation_contract_path,
            )
            self.assertTrue(
                all(
                    Path(path).is_relative_to(destination.resolve())
                    for path in owned_paths
                )
            )
            self.assertEqual(result["next"]["run_from"], "workspace")
            self.assertEqual(
                result["next"]["cell_execution_preview_argv"],
                [
                    "mymoe",
                    "cell-exec",
                    "preview",
                    "--receipt",
                    "./advisor-receipt.json",
                    "--task-stdin",
                    "--catalog",
                    "./adaptive-cells.json",
                    "--evaluation-contract",
                    "./adaptive-evaluation-contract.json",
                    "--policy",
                    "./adaptive-execution-policy.json",
                    "--json",
                ],
            )
            self.assertEqual(registry.skills, ())
            self.assertEqual(registry.tools, ())
            self.assertEqual(registry.plugins, ())
            self.assertTrue((destination / "work" / "runtime" / "chats.json").is_file())
            self.assertFalse((hostile / "work").exists())

    def test_cli_init_is_discoverable_sanitized_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "private-sentinel-workspace"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            command = [
                sys.executable,
                "-m",
                "local_moe.cli",
                "advisor-init",
                "--out",
                str(destination),
            ]
            completed = subprocess.run(
                command,
                cwd=tmp,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            repeated = subprocess.run(
                command,
                cwd=tmp,
                env=environment,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "created")
        self.assertNotIn(str(destination), completed.stdout)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(json.loads(repeated.stderr)["error"], "advisor_init_failed")
        self.assertNotIn(str(destination), repeated.stderr)


if __name__ == "__main__":
    unittest.main()
