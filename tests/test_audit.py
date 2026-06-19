from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.audit import AuditLogStore, audit_event_payload, audit_log_payload, audit_prune_payload


class AuditLogTests(unittest.TestCase):
    def test_records_lists_and_filters_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditLogStore(Path(tmp) / "audit.jsonl")
            first = store.record(
                "data.export",
                "confirmation_required",
                risk_class="read_only",
                metadata={"note": "contains private data"},
            )
            second = store.record(
                "data.export",
                "ok",
                risk_class="read_only",
                metadata={"chat_sessions": 2},
            )

            all_events = store.list_events(limit=10)
            ok_events = store.list_events(status="ok")
            export_events = store.list_events(action="data.export")

        self.assertEqual(audit_event_payload(second)["status"], "ok")
        self.assertEqual(audit_log_payload(ok_events)["count"], 1)
        self.assertEqual(all_events[0].id, second.id)
        self.assertEqual(all_events[1].id, first.id)
        self.assertEqual(len(export_events), 2)

    def test_metadata_is_truncated_and_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditLogStore(Path(tmp) / "audit.jsonl")
            event = store.record(
                "tool.run",
                "tool_error",
                subject="data.import",
                metadata={
                    "message": "x" * 300,
                    "nested": {"enabled": True},
                    "items": list(range(30)),
                },
            )

        self.assertLessEqual(len(event.metadata["message"]), 240)
        self.assertEqual(event.metadata["nested"]["enabled"], True)
        self.assertEqual(len(event.metadata["items"]), 20)

    def test_prunes_to_latest_events_and_records_prune_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditLogStore(Path(tmp) / "audit.jsonl")
            for index in range(4):
                store.record("data.export", "ok", metadata={"index": index})

            report = store.prune(keep=3)
            events = store.list_events(limit=10)

        payload = audit_prune_payload(report)
        self.assertEqual(payload["before_count"], 4)
        self.assertEqual(payload["after_count"], 3)
        self.assertEqual(payload["removed_count"], 2)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].action, "audit.prune")
        self.assertEqual(events[1].metadata["index"], 3)
        self.assertEqual(events[2].metadata["index"], 2)


if __name__ == "__main__":
    unittest.main()
