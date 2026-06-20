from __future__ import annotations

import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config, parse_config
from local_moe.environment import build_environment_report, render_environment_report_markdown


class EnvironmentReportTests(unittest.TestCase):
    def test_builds_metadata_only_environment_snapshot(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")

        report = build_environment_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
        )

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["app"]["mode"], "local_model_required")
        self.assertEqual(report["paths"]["moe_config"], "tests/fixtures/moe.synthetic.json")
        self.assertIn("python", report)
        self.assertIn("packages", report)
        self.assertIn("git", report)
        self.assertIn("hardware", report)
        self.assertIn("storage", report)
        self.assertEqual(report["storage"]["schema_version"], "1.0")
        self.assertEqual(report["runtime"]["expert_count"], 3)
        self.assertIn("chat transcripts", report["privacy"]["excludes"])
        serialized = str(report).lower()
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("content_excerpt", serialized)

    def test_redacts_secret_like_nested_params(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": "local-model",
                        "role": "general",
                        "base_url": "http://127.0.0.1:9999/v1",
                        "params": {
                            "temperature": 0.1,
                            "api_key": "secret-value",
                            "headers": {"Authorization": "Bearer secret-value"},
                        },
                    }
                ],
                "rules": [{"expert_id": "general", "keywords": ["test"], "weight": 1.0}],
            }
        )

        report = build_environment_report(
            config_path="inline",
            config=config,
            app_config=app_config,
        )

        serialized = str(report)
        self.assertIn("temperature", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("secret-value", serialized)
        self.assertIn("[redacted]", serialized)

    def test_renders_markdown_environment_snapshot(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")
        report = build_environment_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
        )

        markdown = render_environment_report_markdown(report)

        self.assertIn("# myMoE Environment Snapshot", markdown)
        self.assertIn("## Experts", markdown)
        self.assertIn("## Storage", markdown)
        self.assertIn("`synthetic-general`", markdown)
        self.assertIn("## Privacy", markdown)
        self.assertNotIn("api_key", markdown.lower())


if __name__ == "__main__":
    unittest.main()
