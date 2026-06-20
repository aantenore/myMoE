from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.run_log import RunLogStore, run_log_payload, run_log_prune_payload


class RunLogTests(unittest.TestCase):
    def test_records_generation_metadata_without_prompt_or_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunLogStore(Path(tmp) / "runs.jsonl")
            store.record_generation(
                mode="generate",
                prompt="Private prompt text",
                session_id="session-1",
                latency_ms=42,
                context_payload={
                    "token_estimate": 123,
                    "budget_tokens": 1000,
                    "compaction_needed": False,
                    "dropped_turns": 0,
                    "sections": {"current_prompt": 9},
                    "memory_ids": ["mem-1"],
                },
                response_payload={
                    "content": "Private answer text",
                    "correlation_id": "corr-1",
                    "route": {
                        "selected": [{"expert_id": "general", "score": 1.0}],
                        "fallback_order": ["fast_fallback"],
                    },
                    "results": [
                        {
                            "expert_id": "general",
                            "model": "local/model",
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "predicted_tokens_per_second": 12.5,
                        }
                    ],
                    "errors": [],
                    "disagreement": None,
                },
            )

            payload = run_log_payload(store.list_records(), path=store.path)

        rendered = str(payload)
        self.assertEqual(payload["count"], 1)
        record = payload["records"][0]
        self.assertEqual(record["mode"], "generate")
        self.assertEqual(record["session_id"], "session-1")
        self.assertEqual(record["selected_experts"], ["general"])
        self.assertEqual(record["context"]["memory_ids"], ["mem-1"])
        self.assertEqual(record["latency_ms"], 42)
        self.assertEqual(record["prompt_tokens"], 10)
        self.assertEqual(record["completion_tokens"], 5)
        self.assertNotIn("Private prompt text", rendered)
        self.assertNotIn("Private answer text", rendered)
        self.assertEqual(len(record["prompt_sha256"]), 64)

    def test_prunes_to_requested_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunLogStore(Path(tmp) / "runs.jsonl")
            for index in range(3):
                store.record_generation(
                    mode="generate",
                    prompt=f"Prompt {index}",
                    response_payload={
                        "correlation_id": f"corr-{index}",
                        "route": {"selected": [], "fallback_order": []},
                        "results": [],
                        "errors": [],
                    },
                )

            report = store.prune(keep=2)
            payload = run_log_prune_payload(report)
            remaining = run_log_payload(store.list_records(limit=10), path=store.path)

        self.assertEqual(payload["before_count"], 3)
        self.assertEqual(payload["after_count"], 2)
        self.assertEqual(payload["removed_count"], 1)
        self.assertEqual(remaining["count"], 2)
        self.assertEqual([item["correlation_id"] for item in remaining["records"]], ["corr-2", "corr-1"])


if __name__ == "__main__":
    unittest.main()
