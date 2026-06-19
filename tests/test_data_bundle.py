from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.chat_store import FileChatStore
from local_moe.data_bundle import build_local_data_bundle, restore_local_data_bundle
from local_moe.memory import FileMemoryStore


class LocalDataBundleTests(unittest.TestCase):
    def test_exports_and_restores_chats_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_chats = FileChatStore(root / "source" / "chats.json")
            source_memory = FileMemoryStore(root / "source" / "memory.jsonl")
            session = source_chats.append_exchange(
                session_id=None,
                user_content="Remember the local backup plan.",
                assistant_content="Export chats and memory as JSON.",
            )
            memory = source_memory.add("Backup bundles contain local memory.", scope="default")

            bundle = build_local_data_bundle(chat_store=source_chats, memory_store=source_memory)
            target_chats = FileChatStore(root / "target" / "chats.json")
            target_memory = FileMemoryStore(root / "target" / "memory.jsonl")
            report = restore_local_data_bundle(
                bundle,
                chat_store=target_chats,
                memory_store=target_memory,
            )
            restored_session = target_chats.get_session(session.id)
            restored_memory = target_memory.search("backup bundles", scope="default")

        self.assertEqual(bundle["schema_version"], "mymoe.local-data.v1")
        self.assertTrue(bundle["privacy"]["contains_user_content"])
        self.assertEqual(bundle["counts"]["chat_sessions"], 1)
        self.assertEqual(bundle["counts"]["chat_messages"], 2)
        self.assertEqual(bundle["counts"]["memory_records"], 1)
        self.assertEqual(report.chats["imported_count"], 1)
        self.assertEqual(report.memory["imported_count"], 1)
        self.assertIsNotNone(restored_session)
        assert restored_session is not None
        self.assertEqual(restored_session.messages[0].content, "Remember the local backup plan.")
        self.assertEqual(restored_memory[0][0].id, memory.id)

    def test_replace_mode_removes_existing_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_chats = FileChatStore(root / "source" / "chats.json")
            source_memory = FileMemoryStore(root / "source" / "memory.jsonl")
            source_chats.create_session(title="Restored")
            source_memory.add("Restored memory.", scope="default")
            bundle = build_local_data_bundle(chat_store=source_chats, memory_store=source_memory)

            target_chats = FileChatStore(root / "target" / "chats.json")
            target_memory = FileMemoryStore(root / "target" / "memory.jsonl")
            old_session = target_chats.create_session(title="Old")
            target_memory.add("Old memory.", scope="default")

            report = restore_local_data_bundle(
                bundle,
                chat_store=target_chats,
                memory_store=target_memory,
                mode="replace",
            )
            old_session_removed = target_chats.get_session(old_session.id) is None
            restored_sessions = target_chats.list_sessions()
            restored_records = target_memory.list(scope="default")

        self.assertEqual(report.mode, "replace")
        self.assertTrue(old_session_removed)
        self.assertEqual(len(restored_sessions), 1)
        self.assertEqual(len(restored_records), 1)
        self.assertEqual(restored_records[0].text, "Restored memory.")


if __name__ == "__main__":
    unittest.main()
