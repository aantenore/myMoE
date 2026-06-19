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

    def test_ingests_knowledge_document_as_chunked_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            report = store.ingest_document(
                "Alpha routing note.\n\nBeta context note.\n\nGamma memory note.",
                title="Architecture Notes",
                scope="project",
                chunk_chars=200,
                metadata={"source": "test"},
            )
            results = store.search("gamma memory", scope="project")

        self.assertEqual(report.title, "Architecture Notes")
        self.assertEqual(report.scope, "project")
        self.assertEqual(report.chunk_count, 1)
        self.assertEqual(len(report.record_ids), 1)
        self.assertEqual(results[0][0].kind, "knowledge")
        self.assertEqual(results[0][0].metadata["document_id"], report.document_id)
        self.assertEqual(results[0][0].metadata["title"], "Architecture Notes")
        self.assertIn("Gamma memory note.", results[0][0].text)

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

    def test_maintenance_distinguishes_pending_and_expired_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            store.add("Active fact.")
            store.add("Future fact.", valid_from="2026-07-01T00:00:00+00:00")
            store.add("Expired fact.", valid_until="2026-01-01T00:00:00+00:00")

            report = store.maintenance_report(now="2026-06-20T00:00:00+00:00")

        self.assertEqual(report.total_records, 3)
        self.assertEqual(report.active_records, 1)
        self.assertEqual(report.pending_records, 1)
        self.assertEqual(report.expired_records, 1)

    def test_prunes_only_expired_temporal_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            active = store.add("Active fact.")
            pending = store.add("Future fact.", valid_from="2026-07-01T00:00:00+00:00")
            expired = store.add("Expired fact.", valid_until="2026-01-01T00:00:00+00:00")

            report = store.prune_expired(now="2026-06-20T00:00:00+00:00")
            remaining = store.list()

        self.assertEqual(report.before_count, 3)
        self.assertEqual(report.removed_count, 1)
        self.assertEqual(report.removed_ids, (expired.id,))
        self.assertEqual({record.id for record in remaining}, {active.id, pending.id})

    def test_forgets_single_record_and_document_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileMemoryStore(Path(tmp) / "memory.jsonl")
            first = store.add("Temporary memory.", scope="project")
            report = store.ingest_document(
                "Alpha note.\n\nBeta note.",
                title="Temporary Document",
                scope="project",
                chunk_chars=200,
            )
            removed_record = store.forget_record(first.id)
            removed_document = store.forget_document(report.document_id)
            records = store.list(scope="project")

        self.assertEqual(removed_record.removed_count, 1)
        self.assertEqual(removed_record.removed_ids, (first.id,))
        self.assertEqual(removed_document.removed_count, 1)
        self.assertEqual(removed_document.removed_ids, report.record_ids)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
