from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.chat_store import FileChatStore, chat_session_payload, chat_summary_payload


class ChatStoreTests(unittest.TestCase):
    def test_appends_and_reloads_chat_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chats.json"
            store = FileChatStore(path)

            session = store.append_exchange(
                session_id=None,
                user_content="Explain the local routing strategy for multilingual prompts.",
                assistant_content="Use a cheap local router first.",
                assistant_meta={"route": {"selected": [{"expert_id": "general"}]}},
            )
            continued = store.append_exchange(
                session_id=session.id,
                user_content="Now summarize it.",
                assistant_content="Route cheaply, answer locally.",
            )
            reloaded = FileChatStore(path).get_session(session.id)

        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(continued.id, session.id)
        self.assertEqual(len(reloaded.messages), 4)
        self.assertEqual(reloaded.messages[0].role, "user")
        self.assertEqual(reloaded.messages[1].meta["route"]["selected"][0]["expert_id"], "general")
        self.assertTrue(reloaded.title.startswith("Explain the local routing strategy"))

    def test_creates_lists_and_deletes_blank_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileChatStore(Path(tmp) / "chats.json")
            session = store.create_session(title="Planning")
            summaries = store.list_sessions()
            deleted = store.delete_session(session.id)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(chat_summary_payload(summaries[0])["title"], "Planning")
        self.assertEqual(chat_session_payload(session)["message_count"], 0)
        self.assertTrue(deleted)

    def test_searches_renames_and_exports_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileChatStore(Path(tmp) / "chats.json")
            session = store.append_exchange(
                session_id=None,
                user_content="Design a plugin lifecycle for local tools.",
                assistant_content="Use allowlists and explicit confirmations.",
                assistant_meta={"route": {"selected": [{"expert_id": "general"}]}},
            )
            renamed = store.rename_session(session.id, "Plugin Lifecycle")
            search_by_title = store.search_sessions("plugin lifecycle")
            search_by_message = store.search_sessions("allowlists confirmations")
            markdown = store.export_markdown(session.id)

        self.assertEqual(renamed.title, "Plugin Lifecycle")
        self.assertEqual(search_by_title[0].id, session.id)
        self.assertEqual(search_by_message[0].id, session.id)
        self.assertIn("# Plugin Lifecycle", markdown)
        self.assertIn("## User", markdown)
        self.assertIn("## Assistant", markdown)
        self.assertIn("Routed to", markdown)

    def test_rejects_unknown_session_on_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileChatStore(Path(tmp) / "chats.json")

            with self.assertRaises(KeyError):
                store.append_exchange(
                    session_id="missing",
                    user_content="Hello",
                    assistant_content="Hi",
                )


if __name__ == "__main__":
    unittest.main()
