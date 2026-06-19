from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.app_config import load_app_config
from local_moe.config import parse_config
from local_moe.setup_status import inspect_setup_status, setup_status_payload


class SetupStatusTests(unittest.TestCase):
    def test_reports_cached_huggingface_snapshot_with_allow_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub_cache = root / "hub"
            snapshot = hub_cache / "models--owner--model" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "model.Q4_K_M.gguf").write_text("stub", encoding="utf-8")
            config = _config(
                model="owner/model:Q4_K_M",
                backend="llama_cpp",
            )
            app_config = _app_config(root / "cache")

            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub_cache)}):
                payload = setup_status_payload(
                    inspect_setup_status(
                        "configs/test.json",
                        config,
                        app_config,
                        app_config_path="configs/app.test.json",
                    )
                )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["models"][0]["status"], "cached")
        self.assertEqual(payload["models"][0]["allow_patterns"], ["*Q4_K_M*.gguf"])
        self.assertIn("configs/app.test.json", payload["download_command_display"])

    def test_reports_missing_local_file_as_setup_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_model = root / "missing.Q4_K_M.gguf"
            config = _config(model=str(missing_model), backend="llama_cpp")
            app_config = _app_config(root / "cache")

            payload = setup_status_payload(
                inspect_setup_status("configs/test.json", config, app_config)
            )

        self.assertEqual(payload["status"], "needs_setup")
        self.assertEqual(payload["models"][0]["status"], "missing")
        self.assertIn("Local model file is missing", payload["models"][0]["detail"])

    def test_reports_synthetic_fixture_as_ready_without_model_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = parse_config(
                {
                    "routing": {"top_k": 1},
                    "experts": [
                        {
                            "id": "synthetic",
                            "provider": "synthetic",
                            "model": "synthetic-model",
                            "role": "general",
                        }
                    ],
                    "rules": [],
                }
            )
            app_config = _app_config(Path(tmp) / "cache")

            payload = setup_status_payload(
                inspect_setup_status("tests/fixtures/moe.synthetic.json", config, app_config)
            )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["models"], [])

    def test_reports_ollama_pull_command_without_python_env_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(model="qwen3:4b", backend="ollama")
            app_config = _app_config(Path(tmp) / "cache")

            with patch("local_moe.setup_status.shutil.which", return_value="/usr/local/bin/ollama"):
                payload = setup_status_payload(
                    inspect_setup_status("configs/ollama.json", config, app_config)
                )

        self.assertEqual(payload["status"], "needs_setup")
        self.assertEqual(payload["models"][0]["status"], "pull_required")
        self.assertEqual(payload["models"][0]["command_display"], "ollama pull qwen3:4b")
        self.assertNotIn("PYTHONPATH", payload["models"][0]["command_display"])


def _config(model: str, backend: str):
    return parse_config(
        {
            "routing": {"top_k": 1},
            "experts": [
                {
                    "id": "local",
                    "provider": "openai_compatible",
                    "model": model,
                    "role": "local",
                    "params": {"runtime_backend": backend},
                }
            ],
            "rules": [],
        }
    )


def _app_config(cache_dir: Path):
    app_config = load_app_config("configs/app.json")
    runtime = replace(app_config.runtime, model_cache_dir=str(cache_dir))
    return replace(app_config, runtime=runtime)


if __name__ == "__main__":
    unittest.main()
