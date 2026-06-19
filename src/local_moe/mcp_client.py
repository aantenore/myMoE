from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Any

from .extensions import McpServerDefinition


class McpClientError(RuntimeError):
    """Raised when an MCP stdio server cannot be inspected."""


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolList:
    server: str
    tools: tuple[McpTool, ...]
    protocol_version: str


class StdioMcpClient:
    """Small MCP stdio client for capability discovery only."""

    def __init__(
        self,
        server: McpServerDefinition,
        *,
        timeout_seconds: float = 8.0,
        protocol_version: str = "2025-11-25",
    ):
        self._server = server
        self._timeout_seconds = timeout_seconds
        self._protocol_version = protocol_version

    def list_tools(self) -> McpToolList:
        if not self._server.enabled:
            raise McpClientError(f"MCP server is disabled: {self._server.name}")
        if self._server.transport != "stdio":
            raise McpClientError(f"Unsupported MCP transport for {self._server.name}: {self._server.transport}")

        env = dict(os.environ)
        env.update(self._server.env)
        try:
            process = subprocess.Popen(
                [self._server.command, *self._server.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(Path(self._server.cwd).expanduser()) if self._server.cwd else str(Path.cwd()),
                env=env,
            )
        except OSError as exc:
            raise McpClientError(f"Failed to start MCP server {self._server.name}: {exc}") from exc
        stdout_queue: Queue[str] = Queue()
        reader = threading.Thread(
            target=_read_lines,
            args=(process.stdout, stdout_queue),
            daemon=True,
        )
        reader.start()

        try:
            initialize = self._request(
                process,
                stdout_queue,
                1,
                "initialize",
                {
                    "protocolVersion": self._protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "myMoE",
                        "version": "0.1.0",
                    },
                },
            )
            self._notify(process, "notifications/initialized", {})
            tools_result = self._request(process, stdout_queue, 2, "tools/list", {})
        finally:
            _close_process(process)

        negotiated = str(
            initialize.get("protocolVersion")
            or initialize.get("serverInfo", {}).get("protocolVersion")
            or self._protocol_version
        )
        tools = tuple(_parse_tool(item) for item in tools_result.get("tools", []))
        return McpToolList(server=self._server.name, tools=tools, protocol_version=negotiated)

    def _request(
        self,
        process: subprocess.Popen[str],
        stdout_queue: Queue[str],
        request_id: int,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        _write_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        response = _read_response(process, stdout_queue, request_id, self._timeout_seconds)
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise McpClientError(str(error.get("message", "MCP request failed.")))
            raise McpClientError("MCP request failed.")
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise McpClientError(f"MCP method returned invalid result: {method}")
        return result

    def _notify(
        self,
        process: subprocess.Popen[str],
        method: str,
        params: dict[str, Any],
    ) -> None:
        _write_message(
            process,
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            },
        )


def mcp_tool_list_payload(result: McpToolList) -> dict[str, Any]:
    return {
        "server": result.server,
        "protocol_version": result.protocol_version,
        "count": len(result.tools),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "annotations": tool.annotations,
            }
            for tool in result.tools
        ],
    }


def _parse_tool(raw: object) -> McpTool:
    if not isinstance(raw, dict):
        raise McpClientError("MCP tools/list returned a non-object tool.")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise McpClientError("MCP tools/list returned a tool without a name.")
    input_schema = raw.get("inputSchema", {})
    annotations = raw.get("annotations", {})
    return McpTool(
        name=name,
        description=str(raw.get("description", "")),
        input_schema=input_schema if isinstance(input_schema, dict) else {},
        annotations=annotations if isinstance(annotations, dict) else {},
    )


def _write_message(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise McpClientError("MCP process stdin is unavailable.")
    try:
        process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        process.stdin.flush()
    except BrokenPipeError as exc:
        raise McpClientError("MCP process closed stdin.") from exc


def _read_response(
    process: subprocess.Popen[str],
    stdout_queue: Queue[str],
    request_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise McpClientError(f"MCP process exited with code {process.returncode}.")
        try:
            line = stdout_queue.get(timeout=max(0.05, min(0.25, deadline - time.monotonic())))
        except Empty:
            continue
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        if message.get("id") != request_id:
            continue
        return message
    raise McpClientError(f"Timed out waiting for MCP response id {request_id}.")


def _read_lines(stream: object, output: Queue[str]) -> None:
    if stream is None:
        return
    for line in stream:
        output.put(line)


def _close_process(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()
