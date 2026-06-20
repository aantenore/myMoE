from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config
from local_moe.doctor import build_doctor_report, render_doctor_report_markdown
from local_moe.extensions import load_extension_registry
from local_moe.hardware import HardwareProfile


TEST_HARDWARE = HardwareProfile(
    machine="arm64",
    cpu_brand="Apple Test",
    memory_bytes=24 * 1024**3,
    memory_gib=24.0,
    recommended_strategy="general_purpose_moe_single_resident",
    rationale=("Use one strong resident general expert plus a small fallback.",),
)


class DoctorTests(unittest.TestCase):
    def test_builds_ready_report_for_synthetic_profile(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")
        registry = load_extension_registry(
            plugins_dir=app_config.extensions.plugins_dir,
            skills_dir=app_config.extensions.skills_dir,
            tools_config=app_config.extensions.tools_config,
            mcp_config=app_config.extensions.mcp_config,
            cron_config=app_config.extensions.cron_config,
        )

        report = build_doctor_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
            registry=registry,
        )

        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertTrue(report["recommendations"])
        self.assertEqual(checks["setup"]["status"], "pass")
        self.assertEqual(checks["health"]["status"], "pass")
        self.assertEqual(checks["extensions"]["status"], "pass")
        self.assertEqual(checks["hardware_fit"]["status"], "pass")
        self.assertEqual(report["hardware_fit"]["status"], "compatible")
        self.assertIn("extension_audit", report)
        self.assertTrue(report["extensions"]["tools"])

    def test_reports_too_large_active_profile_as_required_failure(self) -> None:
        app_config = load_app_config("configs/app.json")
        registry = load_extension_registry(
            plugins_dir=app_config.extensions.plugins_dir,
            skills_dir=app_config.extensions.skills_dir,
            tools_config=app_config.extensions.tools_config,
            mcp_config=app_config.extensions.mcp_config,
            cron_config=app_config.extensions.cron_config,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "moe.too-large.json"
            config_path.write_text(
                json.dumps(
                    {
                        "routing": {"top_k": 1, "fallback_order": []},
                        "experts": [
                            {
                                "id": "general",
                                "provider": "openai_compatible",
                                "base_url": "not-a-url",
                                "model": "example/giant-local-model",
                                "role": "primary-general-purpose",
                                "params": {"runtime_backend": "llama_cpp"},
                            }
                        ],
                        "rules": [],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path = root / "candidates.json"
            candidate_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "giant",
                                "repo": "example/giant-local-model",
                                "role": "not_viable_on_24gb",
                                "minimum_memory_gb": 30,
                                "recommended_memory_gb": 45,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)

            report = build_doctor_report(
                config_path=str(config_path),
                config=config,
                app_config=app_config,
                registry=registry,
                hardware_profile=TEST_HARDWARE,
                candidate_paths=(str(candidate_path),),
            )

        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(checks["hardware_fit"]["status"], "fail")
        self.assertEqual(checks["hardware_fit"]["severity"], "required")
        self.assertEqual(report["hardware_fit"]["status"], "too_large")
        self.assertIn("Switch to a smaller runtime profile", " ".join(report["recommendations"]))

    def test_renders_metadata_only_markdown_report(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")
        registry = load_extension_registry(
            plugins_dir=app_config.extensions.plugins_dir,
            skills_dir=app_config.extensions.skills_dir,
            tools_config=app_config.extensions.tools_config,
            mcp_config=app_config.extensions.mcp_config,
            cron_config=app_config.extensions.cron_config,
        )
        report = build_doctor_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
            registry=registry,
        )

        markdown = render_doctor_report_markdown(report)

        self.assertIn("# myMoE System Doctor Report", markdown)
        self.assertIn("## Checks", markdown)
        self.assertIn("`hardware_fit`", markdown)
        self.assertIn("## Privacy", markdown)
        self.assertNotIn("content_excerpt", markdown)
        self.assertNotIn("api_key", markdown.lower())


if __name__ == "__main__":
    unittest.main()
