from __future__ import annotations

import unittest

from local_moe.compaction import LocalCompactionProvider
from local_moe.config import load_config
from local_moe.context import ConversationTurn


class CompactionTests(unittest.TestCase):
    def test_selects_live_fallback_expert_for_compaction(self) -> None:
        provider = LocalCompactionProvider(load_config("configs/moe.live.general-mlx.example.json"))

        self.assertEqual(provider.expert_id, "fast_fallback")

    def test_compacts_with_configured_synthetic_fallback(self) -> None:
        provider = LocalCompactionProvider(load_config("tests/fixtures/moe.synthetic.json"))

        result = provider.compact(
            turns=(
                ConversationTurn(role="user", content="Remember src/local_moe/compaction.py"),
                ConversationTurn(role="assistant", content="Ran tests successfully."),
            ),
            existing_summary="Existing summary",
            correlation_id="compact-1",
        )

        self.assertEqual(result.expert_id, "general")
        self.assertEqual(result.correlation_id, "compact-1")
        self.assertIn("[general:synthetic-general]", result.summary)


if __name__ == "__main__":
    unittest.main()
