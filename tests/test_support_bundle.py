from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config
from local_moe.extensions import load_extension_registry
from local_moe.support_bundle import build_support_bundle


class SupportBundleTests(unittest.TestCase):
    def test_builds_privacy_safe_support_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_gate = root / "quality-gate.json"
            hardware = root / "hardware.json"
            quality_gate.write_text('{"passed": true}', encoding="utf-8")
            hardware.write_text('{"machine": "test"}', encoding="utf-8")
            app_config = load_app_config("configs/app.json")
            config = load_config("tests/fixtures/moe.synthetic.json")
            registry = load_extension_registry(
                plugins_dir=app_config.extensions.plugins_dir,
                skills_dir=app_config.extensions.skills_dir,
                tools_config=app_config.extensions.tools_config,
                mcp_config=app_config.extensions.mcp_config,
                cron_config=app_config.extensions.cron_config,
            )

            bundle = build_support_bundle(
                config_path="tests/fixtures/moe.synthetic.json",
                config=config,
                app_config=app_config,
                registry=registry,
                quality_gate_path=quality_gate,
                hardware_profile_path=hardware,
            )

        self.assertEqual(bundle["schema_version"], "1.0")
        self.assertEqual(bundle["doctor"]["status"], "ready")
        self.assertEqual(bundle["environment"]["schema_version"], "1.0")
        self.assertEqual(bundle["environment"]["paths"]["moe_config"], "tests/fixtures/moe.synthetic.json")
        self.assertEqual(bundle["environment"]["storage"]["schema_version"], "1.0")
        self.assertEqual(bundle["model_inventory"]["schema_version"], "1.0")
        self.assertEqual(bundle["model_inventory"]["summary"]["asset_count"], 0)
        self.assertEqual(bundle["quality_gate"]["data"]["passed"], True)
        self.assertEqual(bundle["hardware_profile"]["data"]["machine"], "test")
        self.assertIn("chat transcripts", bundle["privacy"]["excludes"])
        self.assertIn("memory records", bundle["privacy"]["excludes"])
        self.assertIn("generation run log contents", bundle["privacy"]["excludes"])
        self.assertIn("benchmark prompt response excerpts", bundle["privacy"]["excludes"])
        self.assertEqual(bundle["performance"]["schema_version"], "1.0")
        self.assertNotIn("content_excerpt", str(bundle["performance"]))
        self.assertEqual(bundle["runtime_optimizer"]["schema_version"], "1.0")
        self.assertEqual(bundle["runtime_optimizer"]["mode"], "read_only")
        self.assertIn("runtime optimizer summary", " ".join(bundle["privacy"]["includes"]))
        self.assertIn("storage capacity summary", " ".join(bundle["privacy"]["includes"]))
        self.assertIn("model asset inventory", " ".join(bundle["privacy"]["includes"]))
        self.assertIn("chat_store", bundle["runtime_files"])
        self.assertIn("run_log", bundle["runtime_files"])


if __name__ == "__main__":
    unittest.main()
