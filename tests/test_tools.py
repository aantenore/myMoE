from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from local_moe.config import load_config
from local_moe.extensions import (
    ExtensionRegistry,
    McpServerDefinition,
    ToolDefinition,
    load_extension_registry,
)
from local_moe.memory import FileMemoryStore
from local_moe.tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload
from tests.mcp_test_utils import write_fake_mcp_server


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
        self.assertIn("allowed_tools", result.payload["servers"][0])

    def test_mcp_list_tools_requires_confirmation_and_lists_enabled_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            registry = ExtensionRegistry(
                tools=(
                    ToolDefinition(
                        name="mcp.list_tools",
                        description="List MCP tools",
                        risk_class="process_execution",
                        side_effects="starts_process",
                        enabled=True,
                    ),
                ),
                skills=(),
                mcp_servers=(
                    McpServerDefinition(
                        name="fake",
                        description="Fake MCP server",
                        command=sys.executable,
                        args=(str(script),),
                        enabled=True,
                        risk_class="read_only",
                        capabilities=("tools",),
                        allowed_tools=("echo",),
                    ),
                ),
                cron_jobs=(),
                plugins=(),
            )
            blocked_runner = LocalToolRunner(registry)
            with self.assertRaises(ToolExecutionError):
                blocked_runner.run(
                    "mcp.list_tools",
                    {"server": "fake", "confirm_process_execution": True},
                )

            runner = LocalToolRunner(registry, allow_process_execution=True)
            with self.assertRaises(ToolExecutionError):
                runner.run("mcp.list_tools", {"server": "fake"})

            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.list_tools",
                    {
                        "server": "fake",
                        "confirm_process_execution": True,
                        "timeout_seconds": "slow",
                    },
                )

            result = runner.run(
                "mcp.list_tools",
                {"server": "fake", "confirm_process_execution": True, "timeout_seconds": 3},
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["server"], "fake")
        self.assertEqual(result.payload["tools"][0]["name"], "echo")

    def test_mcp_call_tool_requires_allowlist_and_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            registry = ExtensionRegistry(
                tools=(
                    ToolDefinition(
                        name="mcp.call_tool",
                        description="Call MCP tool",
                        risk_class="process_execution",
                        side_effects="starts_process_and_calls_tool",
                        enabled=True,
                    ),
                ),
                skills=(),
                mcp_servers=(
                    McpServerDefinition(
                        name="fake",
                        description="Fake MCP server",
                        command=sys.executable,
                        args=(str(script),),
                        enabled=True,
                        risk_class="read_only",
                        capabilities=("tools",),
                        allowed_tools=("echo",),
                    ),
                ),
                cron_jobs=(),
                plugins=(),
            )
            blocked_runner = LocalToolRunner(registry)
            with self.assertRaises(ToolExecutionError):
                blocked_runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )

            runner = LocalToolRunner(registry, allow_process_execution=True)
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "confirm_process_execution": True,
                    },
                )
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "not-allowed",
                        "arguments": {},
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": [],
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )

            result = runner.run(
                "mcp.call_tool",
                {
                    "server": "fake",
                    "tool_name": "echo",
                    "arguments": {"text": "hello"},
                    "confirm_process_execution": True,
                    "confirm_tool_call": True,
                    "timeout_seconds": 3,
                },
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["server"], "fake")
        self.assertEqual(result.payload["tool_name"], "echo")
        self.assertEqual(result.payload["content"][0]["text"], "echo:hello")


if __name__ == "__main__":
    unittest.main()
