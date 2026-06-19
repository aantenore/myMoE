from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.config_profiles import discover_config_profiles


class ConfigProfileTests(unittest.TestCase):
    def test_discovers_runnable_profiles_with_setup_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            profile_path = config_dir / "moe.test.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "routing": {"top_k": 1, "fallback_order": ["general"]},
                        "experts": [
                            {
                                "id": "general",
                                "provider": "synthetic",
                                "model": "synthetic-general",
                                "role": "general",
                            }
                        ],
                        "rules": [
                            {
                                "expert_id": "general",
                                "keywords": ["summarize"],
                                "weight": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "model-candidates.json").write_text("{}", encoding="utf-8")
            app_config = load_app_config("configs/app.json")
            app_config = replace(app_config, default_moe_config=str(profile_path))

            payload = discover_config_profiles(
                active_config_path=str(profile_path),
                app_config=app_config,
                config_dir=config_dir,
            )

        self.assertEqual(payload["count"], 1)
        profile = payload["profiles"][0]
        self.assertTrue(profile["active"])
        self.assertTrue(profile["default"])
        self.assertEqual(profile["status"], "valid")
        self.assertEqual(profile["setup"]["status"], "ready")
        self.assertEqual(profile["expert_count"], 1)
        self.assertEqual(profile["experts"][0]["model"], "synthetic-general")

    def test_includes_active_profile_outside_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            active_path = root / "active.json"
            active_path.write_text(Path("tests/fixtures/moe.synthetic.json").read_text(encoding="utf-8"))
            app_config = load_app_config("configs/app.json")

            payload = discover_config_profiles(
                active_config_path=str(active_path),
                app_config=app_config,
                config_dir=config_dir,
            )

        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["profiles"][0]["active"])
        self.assertEqual(payload["profiles"][0]["status"], "valid")


if __name__ == "__main__":
    unittest.main()
