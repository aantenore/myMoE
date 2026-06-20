from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_moe.config import load_config
from local_moe.smoke import build_generation_smoke_report


class GenerationSmokeTests(unittest.TestCase):
    def test_passes_when_selected_expert_returns_visible_content(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")

        report = build_generation_smoke_report(config, prompt="Summarize this in one sentence.")

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["status"], "pass")
        self.assertGreater(report["content_chars"], 0)
        self.assertEqual(report["route"]["selected"][0]["expert_id"], "general")
        self.assertTrue(report["results"])

    def test_fails_when_provider_returns_only_blank_content(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")
        empty_response = SimpleNamespace(
            content="  ",
            correlation_id="smoke-empty",
            route=SimpleNamespace(
                selected=(SimpleNamespace(expert_id="general", score=1.0, matched_keywords=("test",)),),
                fallback_order=(),
            ),
            results=(
                SimpleNamespace(
                    expert_id="general",
                    model="empty-model",
                    content="  ",
                    prompt_tokens=3,
                    completion_tokens=0,
                    predicted_tokens_per_second=None,
                ),
            ),
            errors=(),
            disagreement=None,
        )

        with patch("local_moe.smoke.LocalMoE") as moe_class:
            moe_class.return_value.generate.return_value = empty_response
            report = build_generation_smoke_report(config, prompt="Return nothing.")

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["content_chars"], 0)
        self.assertIn("no visible content", " ".join(report["recommendations"]).lower())


if __name__ == "__main__":
    unittest.main()
