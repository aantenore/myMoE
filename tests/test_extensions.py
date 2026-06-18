from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan
from local_moe.config import load_config
from local_moe.extensions import (
    ExtensionError,
    create_plugin_scaffold,
    load_extension_registry,
)
from local_moe.scheduler import due_jobs


class ExtensionTests(unittest.TestCase):
    def test_loads_app_and_extension_registry(self) -> None:
        app_config = load_app_config("configs/app.json")
        registry = load_extension_registry(
            plugins_dir=app_config.extensions.plugins_dir,
            skills_dir=app_config.extensions.skills_dir,
            tools_config=app_config.extensions.tools_config,
            mcp_config=app_config.extensions.mcp_config,
            cron_config=app_config.extensions.cron_config,
        )

        self.assertEqual(app_config.mode, "local_model_required")
        self.assertTrue(registry.tools)
        self.assertTrue(registry.skills)
        self.assertTrue(registry.plugins)
        self.assertTrue(registry.cron_jobs)

    def test_creates_plugin_scaffold_with_valid_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = create_plugin_scaffold("sample-plugin", root=tmp)

            self.assertTrue((path / "plugin.json").exists())
            self.assertTrue((path / "SKILL.md").exists())

    def test_rejects_invalid_plugin_id(self) -> None:
        with self.assertRaises(ExtensionError):
            create_plugin_scaffold("Bad_Plugin", root=Path("/tmp"))

    def test_scheduler_returns_due_interval_jobs(self) -> None:
        registry = load_extension_registry()
        due = due_jobs(registry.cron_jobs, {"memory-maintenance": 0}, now_epoch=90000)

        self.assertIn("memory-maintenance", {job.id for job in due})

    def test_builds_runtime_plan_for_live_config(self) -> None:
        app_config = load_app_config("configs/app.json")
        moe_config = load_config(app_config.default_moe_config)
        plan = build_runtime_plan(moe_config, app_config.runtime.preferred_backends)

        self.assertIn(plan.backend, {"mlx_lm", "ollama", "llama_cpp"})
        self.assertTrue(plan.install_commands)


if __name__ == "__main__":
    unittest.main()
