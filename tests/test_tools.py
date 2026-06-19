from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.config import load_config
from local_moe.extensions import load_extension_registry
from local_moe.memory import FileMemoryStore
from local_moe.tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload


class ToolRunnerTests(unittest.TestCase):
    def test_memory_search_returns_scoped_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FileMemoryStore(memory_path)
            store.add("Local model routing should stay configurable.", scope="project")
            store.add("Hosted model note.", scope="other")
            runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)

            result = runner.run(
                "memory.search",
                {"query": "local routing configurable", "scope": "project", "limit": 5},
            )

        payload = tool_result_payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["count"], 1)
        self.assertEqual(
            payload["payload"]["records"][0]["text"],
            "Local model routing should stay configurable.",
        )

    def test_context_compact_can_use_configured_model(self) -> None:
        runner = LocalToolRunner(
            load_extension_registry(),
            moe_config=load_config("tests/fixtures/moe.synthetic.json"),
        )

        result = runner.run(
            "context.compact",
            {
                "turns": [
                    {"role": "user", "content": "We chose a local-first app."},
                    {"role": "assistant", "content": "The router remains lightweight."},
                ],
                "existing_summary": "Existing decision: no cloud inference.",
                "use_model": True,
            },
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["turn_count"], 2)
        self.assertIn("Summarize this local-agent session", result.payload["prompt"])
        self.assertIn("synthetic-general", result.payload["summary"])

    def test_plugin_create_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = LocalToolRunner(load_extension_registry(), plugins_dir=tmp)

            with self.assertRaises(ToolExecutionError):
                runner.run("plugin.create", {"plugin_id": "demo-plugin"})

            result = runner.run(
                "plugin.create",
                {"plugin_id": "demo-plugin", "confirm": True},
            )

            created_path = Path(result.payload["path"])
            manifest_exists = (created_path / "plugin.json").exists()
            skill_exists = (created_path / "SKILL.md").exists()

        self.assertEqual(created_path.name, "demo-plugin")
        self.assertEqual(result.status, "ok")
        self.assertTrue(manifest_exists)
        self.assertTrue(skill_exists)

    def test_mcp_search_capabilities_returns_declared_servers(self) -> None:
        runner = LocalToolRunner(load_extension_registry())

        result = runner.run("mcp.search_capabilities", {"query": "filesystem"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["count"], 1)
        self.assertEqual(result.payload["servers"][0]["name"], "filesystem")
        self.assertFalse(result.payload["servers"][0]["enabled"])


if __name__ == "__main__":
    unittest.main()
