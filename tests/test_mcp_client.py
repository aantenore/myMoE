from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from local_moe.extensions import McpServerDefinition
from local_moe.mcp_client import (
    McpClientError,
    StdioMcpClient,
    mcp_tool_call_payload,
    mcp_tool_list_payload,
)
from tests.mcp_test_utils import (
    write_descendant_fake_mcp_server,
    write_fake_mcp_server,
    write_oversized_fake_mcp_server,
    write_stateful_fake_mcp_server,
)


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
                allowed_tools=("echo",),
            )

            result = StdioMcpClient(server, timeout_seconds=3).list_tools()

        payload = mcp_tool_list_payload(result)
        self.assertEqual(payload["server"], "fake")
        self.assertEqual(payload["protocol_version"], "2025-11-25")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["tools"][0]["name"], "echo")
        self.assertEqual(payload["tools"][0]["input_schema"]["type"], "object")

    def test_calls_tool_from_enabled_stdio_server(self) -> None:
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
                allowed_tools=("echo",),
            )

            result = StdioMcpClient(server, timeout_seconds=3).call_tool("echo", {"text": "hello"})

        payload = mcp_tool_call_payload(result)
        self.assertEqual(payload["server"], "fake")
        self.assertEqual(payload["tool_name"], "echo")
        self.assertFalse(payload["is_error"])
        self.assertEqual(payload["content"][0]["text"], "echo:hello")

    def test_stdio_protocol_is_utf8_on_every_platform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            server = McpServerDefinition(
                name="unicode",
                description="Unicode MCP server",
                command=sys.executable,
                args=(str(script),),
                enabled=True,
                risk_class="read_only",
                capabilities=("tools",),
                allowed_tools=("echo",),
            )

            result = StdioMcpClient(server, timeout_seconds=3).call_tool(
                "echo",
                {"text": "caffè 🧠 日本語"},
            )

        self.assertEqual(result.content[0]["text"], "echo:caffè 🧠 日本語")

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

    def test_persistent_session_preserves_server_state_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_stateful_fake_mcp_server(Path(tmp) / "stateful_mcp.py")
            server = McpServerDefinition(
                name="stateful",
                description="Stateful fake MCP server",
                command=sys.executable,
                args=(str(script),),
                enabled=True,
                risk_class="read_only",
                capabilities=("tools",),
                allowed_tools=("increment",),
            )

            with StdioMcpClient(server, timeout_seconds=3).session() as session:
                first = session.call_tool("increment", {})
                second = session.call_tool("increment", {})

            self.assertEqual(first.content[0]["text"], "1")
            self.assertEqual(second.content[0]["text"], "2")
            self.assertFalse(session.active)

    def test_oversized_stdio_message_is_rejected_before_json_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_oversized_fake_mcp_server(Path(tmp) / "oversized_mcp.py")
            server = McpServerDefinition(
                name="oversized",
                description="Oversized fake MCP server",
                command=sys.executable,
                args=(str(script),),
                enabled=True,
                risk_class="read_only",
                capabilities=("tools",),
                allowed_tools=("oversized",),
            )

            session = StdioMcpClient(server, timeout_seconds=3).session()
            session.start()
            with self.assertRaisesRegex(McpClientError, "bounded message size"):
                session.list_tools()
            self.assertFalse(session.active)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object integration test")
    def test_windows_job_closes_the_full_mcp_process_tree(self) -> None:
        import ctypes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "descendant.pid"
            script = write_descendant_fake_mcp_server(
                root / "descendant_mcp.py", pid_file
            )
            server = McpServerDefinition(
                name="descendant",
                description="MCP server with one child process",
                command=sys.executable,
                args=(str(script),),
                enabled=True,
                risk_class="read_only",
                capabilities=("tools",),
            )
            session = StdioMcpClient(server, timeout_seconds=5).session()
            session.start()
            self.assertTrue(pid_file.is_file())
            self.assertEqual(session._process.args[1:3], ["-I", "-S"])

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel32.OpenProcess(0x00100000, False, int(pid_file.read_text()))
            self.assertTrue(handle)
            try:
                session.close()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if kernel32.WaitForSingleObject(handle, 0) == 0:
                        break
                    time.sleep(0.05)
                self.assertEqual(kernel32.WaitForSingleObject(handle, 0), 0)
            finally:
                kernel32.CloseHandle(handle)

if __name__ == "__main__":
    unittest.main()
