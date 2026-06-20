from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.app_config import load_app_config
from local_moe.config_profiles import discover_config_profiles, recommend_config_profile
from local_moe.hardware import HardwareProfile


TEST_HARDWARE = HardwareProfile(
    machine="arm64",
    cpu_brand="Apple Test",
    memory_bytes=24 * 1024**3,
    memory_gib=24.0,
    recommended_strategy="general_purpose_moe_single_resident",
    rationale=("Use one strong resident general expert plus a small fallback.",),
)


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
                hardware_profile=TEST_HARDWARE,
                candidate_paths=(),
            )

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["hardware"]["memory_gib"], 24.0)
        self.assertEqual(payload["hardware"]["recommended_strategy"], "general_purpose_moe_single_resident")
        profile = payload["profiles"][0]
        self.assertEqual(payload["recommendation"]["status"], "ready")
        self.assertEqual(payload["recommendation"]["profile_path"], profile["path"])
        self.assertTrue(profile["recommended"])
        self.assertTrue(profile["active"])
        self.assertTrue(profile["default"])
        self.assertEqual(profile["status"], "valid")
        self.assertEqual(profile["hardware_fit"]["status"], "compatible")
        self.assertEqual(profile["hardware_fit"]["estimated_memory_gb"], 0.0)
        self.assertEqual(profile["setup"]["status"], "ready")
        self.assertEqual(profile["expert_count"], 1)
        self.assertEqual(profile["experts"][0]["model"], "synthetic-general")
        command_ids = {command["id"] for command in profile["launch_commands"]}
        self.assertIn("inspect_setup", command_ids)
        self.assertIn("prepare_runtime", command_ids)
        self.assertIn("start_models", command_ids)
        self.assertIn("start_ui", command_ids)
        prepare = next(command for command in profile["launch_commands"] if command["id"] == "prepare_runtime")
        self.assertTrue(prepare["requires_confirmation"])
        self.assertEqual(prepare["env"]["PYTHONPATH"], "src")
        self.assertIn("--config", prepare["argv"])
        self.assertIn(profile["path"], prepare["argv"])
        self.assertIn("PYTHONPATH=src", prepare["display"])

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
                hardware_profile=TEST_HARDWARE,
                candidate_paths=(),
            )

        self.assertEqual(payload["count"], 1)
        self.assertTrue(payload["profiles"][0]["active"])
        self.assertEqual(payload["profiles"][0]["status"], "valid")
        self.assertTrue(payload["profiles"][0]["launch_commands"])
        self.assertIn("hardware_fit", payload["profiles"][0])

    def test_scores_profile_hardware_fit_from_candidate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            profile_path = config_dir / "moe.live.test.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "routing": {"top_k": 1, "fallback_order": ["fast_fallback"]},
                        "experts": [
                            {
                                "id": "general",
                                "provider": "openai_compatible",
                                "base_url": "http://127.0.0.1:8101/v1",
                                "model": "example/Qwen3-30B-A3B-4bit",
                                "role": "primary-general-purpose",
                                "params": {"runtime_backend": "mlx_lm"},
                            },
                            {
                                "id": "fast_fallback",
                                "provider": "openai_compatible",
                                "base_url": "http://127.0.0.1:8102/v1",
                                "model": "example/Gemma-E4B-4bit",
                                "role": "fast-summary-and-fallback",
                                "params": {"runtime_backend": "mlx_lm"},
                            },
                        ],
                        "rules": [
                            {
                                "expert_id": "fast_fallback",
                                "keywords": ["summarize"],
                                "weight": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "qwen-general",
                                "repo": "example/Qwen3-30B-A3B-4bit",
                                "role": "primary_general",
                                "estimated_memory_gb": 18.5,
                            },
                            {
                                "id": "gemma-fallback",
                                "repo": "example/Gemma-E4B-4bit",
                                "role": "fast_compaction_or_fallback",
                                "estimated_memory_gb": 5.5,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            app_config = load_app_config("configs/app.json")
            app_config = replace(app_config, default_moe_config=str(profile_path))

            payload = discover_config_profiles(
                active_config_path=str(profile_path),
                app_config=app_config,
                config_dir=config_dir,
                hardware_profile=TEST_HARDWARE,
                candidate_paths=(candidates,),
            )

        fit = payload["profiles"][0]["hardware_fit"]
        self.assertEqual(fit["status"], "recommended")
        self.assertEqual(fit["estimated_memory_gb"], 24.0)
        self.assertEqual(fit["headroom_gb"], 0.0)
        self.assertEqual(fit["resident_large_experts"], 1)
        self.assertEqual({item["candidate_id"] for item in fit["matched_models"]}, {"qwen-general", "gemma-fallback"})

    def test_recommends_hardware_fit_ready_general_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            fast_profile = config_dir / "moe.fast.json"
            fast_profile.write_text(
                json.dumps(
                    {
                        "routing": {"top_k": 1, "fallback_order": []},
                        "experts": [
                            {
                                "id": "general",
                                "provider": "openai_compatible",
                                "base_url": "http://127.0.0.1:8101/v1",
                                "model": "example/Fast-4B-4bit",
                                "role": "fast-local-general-purpose",
                                "params": {"runtime_backend": "mlx_lm"},
                            }
                        ],
                        "rules": [],
                    }
                ),
                encoding="utf-8",
            )
            general_profile = config_dir / "moe.general.json"
            general_profile.write_text(
                json.dumps(
                    {
                        "routing": {
                            "top_k": 1,
                            "fallback_order": ["fast_fallback"],
                            "semantic": {"enabled": True, "examples": []},
                            "distilled": {
                                "enabled": True,
                                "artifact_path": "outputs/router-distilled-live-general.json",
                            },
                        },
                        "experts": [
                            {
                                "id": "general",
                                "provider": "openai_compatible",
                                "base_url": "http://127.0.0.1:8101/v1",
                                "model": "example/Qwen3-30B-A3B-4bit",
                                "role": "primary-general-purpose",
                                "params": {"runtime_backend": "mlx_lm"},
                            },
                            {
                                "id": "fast_fallback",
                                "provider": "openai_compatible",
                                "base_url": "http://127.0.0.1:8102/v1",
                                "model": "example/Gemma-E4B-4bit",
                                "role": "fast-summary-and-fallback",
                                "params": {"runtime_backend": "mlx_lm"},
                            },
                        ],
                        "rules": [],
                    }
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "fast-general",
                                "repo": "example/Fast-4B-4bit",
                                "role": "fast_general",
                                "estimated_memory_gb": 4.0,
                            },
                            {
                                "id": "qwen-general",
                                "repo": "example/Qwen3-30B-A3B-4bit",
                                "role": "primary_general",
                                "estimated_memory_gb": 18.5,
                            },
                            {
                                "id": "gemma-fallback",
                                "repo": "example/Gemma-E4B-4bit",
                                "role": "fast_compaction_or_fallback",
                                "estimated_memory_gb": 5.5,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            hub_cache = root / "hf-cache"
            _write_hf_cache(hub_cache, "example/Fast-4B-4bit")
            _write_hf_cache(hub_cache, "example/Qwen3-30B-A3B-4bit")
            _write_hf_cache(hub_cache, "example/Gemma-E4B-4bit")
            app_config = load_app_config("configs/app.json")
            app_config = replace(app_config, default_moe_config=str(fast_profile))

            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub_cache)}):
                payload = discover_config_profiles(
                    active_config_path=str(fast_profile),
                    app_config=app_config,
                    config_dir=config_dir,
                    hardware_profile=TEST_HARDWARE,
                    candidate_paths=(candidates,),
                )
                recommendation = recommend_config_profile(
                    active_config_path=str(fast_profile),
                    app_config=app_config,
                    config_dir=config_dir,
                    hardware_profile=TEST_HARDWARE,
                    candidate_paths=(candidates,),
                )

        self.assertEqual(payload["recommendation"]["status"], "ready")
        self.assertEqual(payload["recommendation"]["profile_path"], general_profile.as_posix())
        self.assertEqual(recommendation["recommendation"]["profile_path"], general_profile.as_posix())
        self.assertGreater(payload["recommendation"]["score"], 0)
        selected = next(item for item in payload["profiles"] if item["path"] == general_profile.as_posix())
        self.assertTrue(selected["recommended"])
        self.assertFalse(next(item for item in payload["profiles"] if item["path"] == fast_profile.as_posix())["recommended"])
        self.assertIn("start_models", {command["id"] for command in payload["recommendation"]["next_actions"]})


def _write_hf_cache(root: Path, repo_id: str) -> None:
    snapshot = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "test"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_text("cached", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
