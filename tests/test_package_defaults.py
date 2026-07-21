from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.adaptive_execution_gate import load_adaptive_execution_policy
from local_moe.cell_passport import load_cell_catalog
from local_moe.chat_store import FileChatStore
from local_moe.config import load_config
from local_moe.context_policy import load_context_policy
from local_moe.extensions import load_extension_registry
from local_moe.package_defaults import (
    packaged_default_path,
    resolve_advisor_config_reference,
    resolve_app_config_path,
    resolve_app_config_reference,
)


ROOT = Path(__file__).resolve().parents[1]


class PackageDefaultsTests(unittest.TestCase):
    def test_packaged_defaults_form_a_standalone_valid_configuration(self) -> None:
        app_path = packaged_default_path("app.json")
        app_config = load_app_config(app_path)
        moe_path = resolve_app_config_reference(
            app_config.default_moe_config,
            app_path,
        )
        context_path = resolve_app_config_reference(
            app_config.runtime.context_policy_config,
            app_path,
        )

        config = load_config(moe_path)
        context = load_context_policy(
            context_path,
            app_config.runtime.context_policy_profile,
        )
        catalog_path = resolve_advisor_config_reference(
            app_config.advisor.catalog_path,
            app_path,
        )
        evaluation_path = resolve_advisor_config_reference(
            app_config.advisor.evaluation_contract_path,
            app_path,
        )
        catalog = load_cell_catalog(catalog_path)
        execution_policy = load_adaptive_execution_policy(
            packaged_default_path("adaptive-execution-policy.json")
        )

        self.assertEqual(config.experts[0].id, "local")
        self.assertEqual(context.context_limit_tokens, 16384)
        self.assertTrue(app_config.advisor.enabled)
        self.assertTrue(evaluation_path.is_file())
        self.assertTrue(all(cell.measured.sample_count == 0 for cell in catalog.cells))
        self.assertEqual(execution_policy.mode, "dry_run")
        self.assertEqual(execution_policy.max_tool_surfaces, 0)
        self.assertEqual(
            packaged_default_path("adaptive-execution-policy.json").read_bytes(),
            (
                ROOT / "configs" / "adaptive-execution-policy.example.json"
            ).read_bytes(),
        )

    def test_packaged_app_ignores_ambient_extensions_and_keeps_defaults_read_only(self) -> None:
        packaged_names = (
            "app.json",
            "adaptive-cells.json",
            "adaptive-execution-policy.json",
            "adaptive-evaluation-contract.json",
            "moe.json",
            "context-policy.json",
        )
        defaults = {name: packaged_default_path(name) for name in packaged_names}
        before = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in defaults.items()
        }
        defaults_root = defaults["app.json"].parent.resolve()
        with tempfile.TemporaryDirectory() as tmp:
            hostile = Path(tmp) / "hostile"
            ambient_skill = hostile / "skills" / "ambient-secret"
            ambient_skill.mkdir(parents=True)
            (ambient_skill / "SKILL.md").write_text(
                "---\nname: ambient-secret\ndescription: must not load\n---\n",
                encoding="utf-8",
            )
            configs = hostile / "configs"
            configs.mkdir()
            (configs / "tools.json").write_text(
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
            previous = Path.cwd()
            try:
                os.chdir(hostile)
                app = load_app_config(defaults["app.json"])
                registry = load_extension_registry(
                    plugins_dir=app.extensions.plugins_dir,
                    skills_dir=app.extensions.skills_dir,
                    tools_config=app.extensions.tools_config,
                    mcp_config=app.extensions.mcp_config,
                    cron_config=app.extensions.cron_config,
                )
                FileChatStore(Path(app.runtime.work_dir) / "chats.json").create_session(
                    title="wheel runtime"
                )
            finally:
                os.chdir(previous)

            self.assertEqual(app.runtime.work_dir, "work/runtime")
            self.assertTrue((hostile / "work" / "runtime" / "chats.json").is_file())
            self.assertEqual(registry.skills, ())
            self.assertEqual(registry.tools, ())
            self.assertEqual(registry.plugins, ())
            owned_paths = (
                app.default_moe_config,
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
                all(Path(path).is_relative_to(defaults_root) for path in owned_paths)
            )

        after = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in defaults.items()
        }
        self.assertEqual(after, before)

    def test_explicit_app_config_is_never_replaced_by_a_packaged_default(self) -> None:
        requested = Path("missing") / "custom-app.json"

        self.assertEqual(resolve_app_config_path(requested), requested)

    def test_empty_working_directory_uses_packaged_app_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_app_config_path(working_directory=tmp)

        self.assertEqual(resolved, packaged_default_path("app.json"))

    def test_packaged_references_cannot_escape_the_defaults_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay inside defaults"):
            resolve_app_config_reference(
                "../outside.json",
                packaged_default_path("app.json"),
            )

    def test_explicit_dot_slash_reference_is_relative_to_owning_app(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_path = Path(tmp) / "workspace" / "app.json"
            app_path.parent.mkdir()
            app_path.write_text("{}", encoding="utf-8")

            resolved = resolve_app_config_reference("./moe.json", app_path)
            advisor = resolve_advisor_config_reference("catalog.json", app_path)

        self.assertEqual(resolved, (app_path.parent / "moe.json").resolve())
        self.assertEqual(advisor, (app_path.parent / "catalog.json").resolve())

    def test_advisor_reference_cannot_escape_owning_app_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_path = Path(tmp) / "app.json"
            app_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay beside"):
                resolve_advisor_config_reference("../outside.json", app_path)
