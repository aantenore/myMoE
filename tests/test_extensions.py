from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan
from local_moe.config import load_config
from local_moe.extensions import (
    ExtensionError,
    audit_extension_registry,
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

    def test_extension_audit_validates_plugin_references(self) -> None:
        registry = load_extension_registry()
        audit = audit_extension_registry(registry)

        self.assertTrue(audit["checked"])
        self.assertEqual(audit["issue_count"], 0)

    def test_creates_plugin_scaffold_with_valid_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = create_plugin_scaffold(
                "sample-plugin",
                root=tmp,
                name='Sample Plugin: "quoted"\nName',
                description="Adds sample behavior.\nWith wrapped text.",
                risk_class="compute_only",
            )
            registry = load_extension_registry(plugins_dir=tmp)
            audit = audit_extension_registry(registry)
            skill_text = (path / "SKILL.md").read_text(encoding="utf-8")

            self.assertTrue((path / "plugin.json").exists())
            self.assertTrue((path / "SKILL.md").exists())
            self.assertIn('description: "Use this skill when working with the Sample Plugin: \\"quoted\\" Name plugin."', skill_text)
            self.assertIn("sample-plugin", {skill.name for skill in registry.skills})
            self.assertEqual(audit["issue_count"], 0)
            self.assertEqual(registry.plugins[0].permissions["risk_class"], "compute_only")

    def test_rejects_invalid_plugin_id(self) -> None:
        with self.assertRaises(ExtensionError):
            create_plugin_scaffold("Bad_Plugin", root=Path("/tmp"))

    def test_rejects_invalid_mcp_env_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "mcp.json"
            config.write_text(
                '{"servers":[{"name":"bad","command":"python","env":[],"risk_class":"read_only"}]}',
                encoding="utf-8",
            )

            with self.assertRaises(ExtensionError):
                load_extension_registry(mcp_config=config)

    def test_rejects_invalid_mcp_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "mcp.json"
            config.write_text(
                '{"servers":[{"name":"bad","command":"python","timeout_seconds":0,"risk_class":"read_only"}]}',
                encoding="utf-8",
            )

            with self.assertRaises(ExtensionError):
                load_extension_registry(mcp_config=config)

    def test_scheduler_returns_due_interval_jobs(self) -> None:
        registry = load_extension_registry()
        due = due_jobs(registry.cron_jobs, {"memory-maintenance": 0}, now_epoch=90000)

        self.assertIn("memory-maintenance", {job.id for job in due})

    def test_builds_runtime_plan_for_live_config(self) -> None:
        app_config = load_app_config("configs/app.json")
        moe_config = load_config(app_config.default_moe_config)
        plan = build_runtime_plan(moe_config, app_config.runtime.preferred_backends)

        self.assertIn(plan.backend, {"mlx_lm", "ollama", "llama_cpp", "mixed"})
        self.assertTrue(plan.install_commands)

    def test_builds_runtime_plan_for_gemma_pinned_mlx_config(self) -> None:
        moe_config = load_config("configs/moe.live.gemma-e4b-mlx.example.json")
        plan = build_runtime_plan(moe_config, {"darwin_arm64": "mlx_lm", "fallback": "mlx_lm"})
        flattened = [" ".join(command) for command in plan.model_commands]

        self.assertTrue(any("mlx_lm.server" in command for command in flattened))
        self.assertTrue(any("gemma-4-e4b-it-4bit" in command for command in flattened))
        self.assertTrue(any(".[mlx]" in command for command in (" ".join(item) for item in plan.install_commands)))

    def test_builds_runtime_plan_for_huggingface_gguf_config(self) -> None:
        moe_config = load_config("configs/moe.live.gemma-12b-coder-gguf.example.json")
        plan = build_runtime_plan(moe_config, {"darwin_arm64": "mlx_lm", "fallback": "mlx_lm"})
        flattened = [" ".join(command) for command in plan.model_commands]

        self.assertEqual(plan.backend, "llama_cpp")
        self.assertTrue(any("llama-server" in command for command in flattened))
        self.assertTrue(any("-hf yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M" in command for command in flattened))
        self.assertTrue(any(".[gguf]" in command for command in (" ".join(item) for item in plan.install_commands)))

    def test_builds_runtime_plan_for_huggingface_agentic_gguf_config(self) -> None:
        moe_config = load_config("configs/moe.live.gemma-12b-agentic-gguf.example.json")
        plan = build_runtime_plan(moe_config, {"darwin_arm64": "mlx_lm", "fallback": "mlx_lm"})
        flattened = [" ".join(command) for command in plan.model_commands]

        self.assertEqual(plan.backend, "llama_cpp")
        self.assertTrue(any("llama-server" in command for command in flattened))
        self.assertTrue(any("-hf yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M" in command for command in flattened))
        self.assertTrue(any(".[gguf]" in command for command in (" ".join(item) for item in plan.install_commands)))


if __name__ == "__main__":
    unittest.main()
