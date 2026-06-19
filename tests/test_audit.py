from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.audit import AuditLogStore, audit_event_payload, audit_log_payload


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


if __name__ == "__main__":
    unittest.main()
