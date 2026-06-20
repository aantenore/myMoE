from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.profile_activation import (
    activate_config_profile,
    activate_recommended_config_profile,
)


class ProfileActivationTests(unittest.TestCase):
    def test_requires_confirmation_before_writing_default_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, default_config="configs/moe.live.fast-mlx.example.json")
            app_config = load_app_config(app_config_path)

            result = activate_config_profile(
                "tests/fixtures/moe.synthetic.json",
                active_config_path="configs/moe.live.fast-mlx.example.json",
                app_config=app_config,
                app_config_path=str(app_config_path),
            )
            raw = json.loads(app_config_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "confirmation_required")
        self.assertFalse(result["activated"])
        self.assertEqual(raw["default_moe_config"], "configs/moe.live.fast-mlx.example.json")

    def test_confirmed_activation_updates_default_profile_and_reports_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, default_config="configs/moe.live.fast-mlx.example.json")
            app_config = load_app_config(app_config_path)

            result = activate_config_profile(
                "tests/fixtures/moe.synthetic.json",
                active_config_path="configs/moe.live.fast-mlx.example.json",
                app_config=app_config,
                app_config_path=str(app_config_path),
                confirm=True,
            )
            raw = json.loads(app_config_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["activated"])
        self.assertTrue(result["restart_required"])
        self.assertFalse(result["current_process_changed"])
        self.assertEqual(raw["default_moe_config"], "tests/fixtures/moe.synthetic.json")
        self.assertIn("--config", result["restart_command"])
        self.assertIn("tests/fixtures/moe.synthetic.json", result["restart_command"])

    def test_can_activate_current_recommended_profile_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            app_config_path = _write_temp_app_config(root, default_config="tests/fixtures/moe.synthetic.json")
            app_config = load_app_config(app_config_path)

            result = activate_recommended_config_profile(
                active_config_path="tests/fixtures/moe.synthetic.json",
                app_config=app_config,
                app_config_path=str(app_config_path),
                config_dir=config_dir,
                confirm=True,
            )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["restart_required"])
        self.assertEqual(result["recommendation"]["profile_path"], "tests/fixtures/moe.synthetic.json")


def _write_temp_app_config(root: Path, *, default_config: str) -> Path:
    raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = default_config
    raw["runtime"]["work_dir"] = str(root / "runtime")
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
