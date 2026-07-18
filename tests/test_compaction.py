from __future__ import annotations

import unittest

from local_moe.compaction import LocalCompactionProvider
from local_moe.config import load_config, parse_config
from local_moe.context import ConversationTurn
from local_moe.execution_scope import ScopePolicyError


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

    def test_blocks_out_of_policy_compaction_before_provider_invocation(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "remote",
                        "provider": "openai_compatible",
                        "model": "remote-model",
                        "role": "summary",
                        "base_url": "https://models.example.test/v1",
                        "execution": {
                            "scope": "paid_remote",
                            "transport": "gateway",
                        },
                    }
                ],
                "rules": [],
            }
        )
        provider = LocalCompactionProvider(config)

        with self.assertRaisesRegex(ScopePolicyError, "scope_blocked"):
            provider.compact(
                turns=(ConversationTurn(role="user", content="Summarize this."),),
                correlation_id="compact-blocked",
            )


if __name__ == "__main__":
    unittest.main()
