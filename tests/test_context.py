from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.context import (
    ContextPolicy,
    ContextSection,
    ConversationTurn,
    MemorySnippet,
    build_compaction_prompt,
    build_context_bundle,
)
from local_moe.context_policy import load_context_policy


class ContextTests(unittest.TestCase):
    def test_builds_cache_friendly_context_order(self) -> None:
        bundle = build_context_bundle(
            system_prompt="Stable system prompt",
            memories=[MemorySnippet(id="m1", text="Antonio prefers local models.")],
            summary="Prior decision: use one strong expert first.",
            turns=[ConversationTurn(role="user", content="Earlier request")],
            current_prompt="Write code",
            policy=ContextPolicy(context_limit_tokens=2048),
        )

        self.assertEqual(
            [part.section for part in bundle.parts],
            [
                ContextSection.SYSTEM,
                ContextSection.MEMORY,
                ContextSection.SUMMARY,
                ContextSection.RECENT_TURNS,
                ContextSection.CURRENT_PROMPT,
            ],
        )
        self.assertIn("Antonio prefers local models", bundle.as_prompt())
        self.assertFalse(bundle.compaction_needed)

    def test_drops_old_turns_when_budget_is_tight(self) -> None:
        turns = [
            ConversationTurn(role="user", content=f"old turn {index} " * 20)
            for index in range(20)
        ]

        bundle = build_context_bundle(
            system_prompt="System",
            current_prompt="Current",
            turns=turns,
            policy=ContextPolicy(
                context_limit_tokens=90,
                reserved_output_tokens=20,
                max_recent_turns=20,
            ),
        )

        self.assertGreater(bundle.dropped_turns, 0)
        self.assertTrue(bundle.compaction_needed)

    def test_memory_items_are_ranked_and_limited(self) -> None:
        memories = [
            MemorySnippet(id="low", text="low", score=0.1),
            MemorySnippet(id="high", text="high", score=0.9),
            MemorySnippet(id="mid", text="mid", score=0.5),
        ]

        bundle = build_context_bundle(
            system_prompt="System",
            current_prompt="Current",
            memories=memories,
            policy=ContextPolicy(context_limit_tokens=2048, max_memory_items=2),
        )

        prompt = bundle.as_prompt()
        self.assertLess(prompt.find("[high]"), prompt.find("[mid]"))
        self.assertNotIn("[low]", prompt)

    def test_compaction_prompt_preserves_artifact_requirements(self) -> None:
        prompt = build_compaction_prompt(
            existing_summary="Touched src/local_moe/context.py",
            turns=[ConversationTurn(role="assistant", content="Ran ./scripts/run_all_checks.sh")],
        )

        self.assertIn("exact file paths", prompt)
        self.assertIn("src/local_moe/context.py", prompt)
        self.assertIn("./scripts/run_all_checks.sh", prompt)

    def test_loads_context_policy_profile_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context-policy.json"
            path.write_text(
                json.dumps(
                    {
                        "default": {
                            "context_limit_tokens": 2048,
                            "reserved_output_tokens": 256,
                        },
                        "small": {
                            "context_limit_tokens": 512,
                            "reserved_output_tokens": 128,
                            "compaction_trigger_ratio": 0.5,
                            "max_recent_turns": 4,
                            "max_memory_items": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            policy = load_context_policy(path, "small")

        self.assertEqual(policy.context_limit_tokens, 512)
        self.assertEqual(policy.reserved_output_tokens, 128)
        self.assertEqual(policy.compaction_trigger_ratio, 0.5)
        self.assertEqual(policy.max_recent_turns, 4)
        self.assertEqual(policy.max_memory_items, 2)


if __name__ == "__main__":
    unittest.main()
