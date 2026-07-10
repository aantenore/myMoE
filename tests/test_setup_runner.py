from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.setup_runner import run_runtime_setup, setup_run_payload


class SetupRunnerTests(unittest.TestCase):
    def test_preview_reports_plan_without_side_effects(self) -> None:
        result = run_runtime_setup(config_path="tests/fixtures/moe.synthetic.json")

        payload = setup_run_payload(result)
        self.assertEqual(payload["status"], "planned")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["execute"])
        self.assertFalse(payload["download_models"])
        self.assertEqual(payload["steps"], [])
        self.assertEqual(payload["setup_before"]["status"], "ready")

    def test_side_effects_require_confirmation(self) -> None:
        result = run_runtime_setup(
            config_path="tests/fixtures/moe.synthetic.json",
            execute=True,
            confirm=False,
        )

        payload = setup_run_payload(result)
        self.assertEqual(payload["status"], "confirmation_required")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["steps"][0]["status"], "confirmation_required")

    def test_execute_uses_injected_command_runner(self) -> None:
        commands: list[tuple[str, ...]] = []

        with patch("local_moe.bootstrap.detect_platform_key", return_value="darwin_arm64"):
            result = run_runtime_setup(
                config_path="tests/fixtures/moe.synthetic.json",
                execute=True,
                confirm=True,
                command_runner=commands.append,
            )

        payload = setup_run_payload(result)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["ok"])
        self.assertTrue(any(command[:3] == ("uv", "pip", "install") for command in commands))
        self.assertTrue(any(step["phase"] == "install" for step in payload["steps"]))

    def test_download_models_validates_local_files_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "tiny.Q4_K_M.gguf"
            model_path.write_bytes(b"gguf")
            config_path = _write_local_file_config(root, model_path)
            app_config_path = _write_app_config(root, config_path)

            result = run_runtime_setup(
                config_path=str(config_path),
                app_config_path=str(app_config_path),
                download_models=True,
                confirm=True,
            )

        payload = setup_run_payload(result)
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["steps"][0]["phase"], "download")
        self.assertEqual(payload["steps"][0]["status"], "ok")
        self.assertIn("local model file", payload["steps"][0]["message"])


def _write_local_file_config(root: Path, model_path: Path) -> Path:
    config_path = root / "moe.json"
    config_path.write_text(
        json.dumps(
            {
                "routing": {"top_k": 1, "fallback_order": ["general"], "aggregation": "best"},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": str(model_path),
                        "role": "general",
                        "base_url": "http://127.0.0.1:8101/v1",
                        "params": {"runtime_backend": "llama_cpp"},
                    }
                ],
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_app_config(root: Path, config_path: Path) -> Path:
    raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = str(config_path)
    raw["runtime"]["work_dir"] = str(root / "runtime")
    raw["runtime"]["model_cache_dir"] = str(root / "cache")
    app_config_path = root / "app.json"
    app_config_path.write_text(json.dumps(raw), encoding="utf-8")
    return app_config_path


if __name__ == "__main__":
    unittest.main()
