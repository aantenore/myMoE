from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from local_moe.extensions import McpServerDefinition
from local_moe.mcp_client import McpClientError, StdioMcpClient, mcp_tool_list_payload
from tests.mcp_test_utils import write_fake_mcp_server


class McpClientTests(unittest.TestCase):
    def test_lists_tools_from_enabled_stdio_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            server = McpServerDefinition(
                name="fake",
                description="Fake MCP server",
                command=sys.executable,
                args=(str(script),),
                enabled=True,
                risk_class="read_only",
                capabilities=("tools",),
            )

            result = StdioMcpClient(server, timeout_seconds=3).list_tools()

        payload = mcp_tool_list_payload(result)
        self.assertEqual(payload["server"], "fake")
        self.assertEqual(payload["protocol_version"], "2025-11-25")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["tools"][0]["name"], "echo")
        self.assertEqual(payload["tools"][0]["input_schema"]["type"], "object")

    def test_rejects_disabled_stdio_server(self) -> None:
        server = McpServerDefinition(
            name="disabled",
            description="Disabled MCP server",
            command=sys.executable,
            args=("-c", "print('should not run')"),
            enabled=False,
            risk_class="read_only",
            capabilities=("tools",),
        )

        with self.assertRaises(McpClientError):
            StdioMcpClient(server, timeout_seconds=1).list_tools()

if __name__ == "__main__":
    unittest.main()
