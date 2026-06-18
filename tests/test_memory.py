from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.memory import FileMemoryStore


class MemoryTests(unittest.TestCase):
    def test_adds_and_lists_memory_records_by_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            first = store.add("Antonio prefers modular local AI apps.", scope="antonio")
            store.add("Other user prefers hosted APIs.", scope="other")

            records = store.list(scope="antonio")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, first.id)
        self.assertEqual(records[0].text, "Antonio prefers modular local AI apps.")

    def test_searches_records_with_simple_keyword_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            store.add("Use Qwen3 Coder for local coding tasks.", scope="project")
            store.add("Summaries preserve file paths and decisions.", scope="project")

            results = store.search("local coding qwen3", scope="project")

        self.assertEqual(results[0][0].text, "Use Qwen3 Coder for local coding tasks.")
        self.assertGreater(results[0][1], 0)

    def test_filters_expired_temporal_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            store.add(
                "Old model recommendation.",
                scope="project",
                valid_until="2026-01-01T00:00:00+00:00",
            )
            store.add("Current model recommendation.", scope="project")

            results = store.search(
                "model recommendation",
                scope="project",
                now="2026-06-18T00:00:00+00:00",
            )

        self.assertEqual([item[0].text for item in results], ["Current model recommendation."])


if __name__ == "__main__":
    unittest.main()
