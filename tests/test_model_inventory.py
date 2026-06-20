from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.app_config import load_app_config
from local_moe.config import parse_config
from local_moe.model_inventory import build_model_asset_inventory


class ModelInventoryTests(unittest.TestCase):
    def test_reports_cached_huggingface_snapshot_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub_cache = root / "hub"
            snapshot = hub_cache / "models--owner--model" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "model.Q4_K_M.gguf").write_bytes(b"x" * 2048)
            config = _config(model="owner/model:Q4_K_M", backend="llama_cpp")
            app_config = _app_config(root / "cache")

            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub_cache)}):
                inventory = build_model_asset_inventory(
                    config_path="configs/test.json",
                    config=config,
                    app_config=app_config,
                )

        self.assertEqual(inventory["schema_version"], "1.0")
        self.assertEqual(inventory["status"], "ready")
        self.assertEqual(inventory["summary"]["asset_count"], 1)
        self.assertEqual(inventory["summary"]["configured_size_bytes"], 2048)
        self.assertEqual(inventory["assets"][0]["status"], "cached")
        self.assertEqual(inventory["assets"][0]["matched_file_count"], 1)

    def test_reports_missing_local_file_as_attention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.Q4_K_M.gguf"
            inventory = build_model_asset_inventory(
                config_path="configs/test.json",
                config=_config(model=str(missing), backend="llama_cpp"),
                app_config=_app_config(root / "cache"),
            )

        self.assertEqual(inventory["status"], "attention")
        self.assertEqual(inventory["summary"]["attention"], 1)
        self.assertEqual(inventory["assets"][0]["status"], "missing")
        self.assertTrue(inventory["recommendations"])

    def test_deduplicates_repeated_model_requests(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "one",
                        "provider": "openai_compatible",
                        "model": "owner/model",
                        "role": "one",
                        "params": {"runtime_backend": "mlx_lm"},
                    },
                    {
                        "id": "two",
                        "provider": "openai_compatible",
                        "model": "owner/model",
                        "role": "two",
                        "params": {"runtime_backend": "mlx_lm"},
                    },
                ],
                "rules": [],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            inventory = build_model_asset_inventory(
                config_path="configs/test.json",
                config=config,
                app_config=_app_config(Path(tmp) / "cache"),
            )

        self.assertEqual(inventory["summary"]["asset_count"], 1)


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
