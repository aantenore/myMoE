from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from .extensions import McpServerDefinition


class McpClientError(RuntimeError):
    """Raised when an MCP stdio server cannot be inspected."""


_MAX_MCP_MESSAGE_CHARS = 2_000_000
_MAX_MCP_QUEUED_MESSAGES = 64


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


@dataclass(frozen=True)
class McpToolCallResult:
    server: str
    tool_name: str
    content: tuple[dict[str, Any], ...]
    is_error: bool
    structured_content: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class StdioMcpClient:
    """Small MCP stdio client for guarded tool discovery and calls."""

    def __init__(
        self,
        server: McpServerDefinition,
        *,
        timeout_seconds: float = 8.0,
        protocol_version: str = "2025-11-25",
        base_environment: Mapping[str, str] | None = None,
    ):
        self._server = server
        self._timeout_seconds = timeout_seconds
        self._protocol_version = protocol_version
        self._base_environment = base_environment

    def session(self) -> StdioMcpSession:
        """Create one stateful MCP session; callers own its lifetime."""

        return StdioMcpSession(
            self._server,
            timeout_seconds=self._timeout_seconds,
            protocol_version=self._protocol_version,
            base_environment=self._base_environment,
        )

    def list_tools(self) -> McpToolList:
        with self.session() as session:
            return session.list_tools()

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        with self.session() as session:
            return session.call_tool(tool_name, arguments)


class StdioMcpSession:
    """Persistent MCP stdio session with deterministic teardown."""

    def __init__(
        self,
        server: McpServerDefinition,
        *,
        timeout_seconds: float = 8.0,
        protocol_version: str = "2025-11-25",
        base_environment: Mapping[str, str] | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._server = server
        self._timeout_seconds = timeout_seconds
        self._protocol_version = protocol_version
        self._base_environment = base_environment
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: Queue[object] | None = None
        self._reader: threading.Thread | None = None
        self._windows_job: object | None = None
        self._initialize: dict[str, Any] = {}
        self._next_request_id = 2
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def protocol_version(self) -> str:
        return str(
            self._initialize.get("protocolVersion")
            or self._initialize.get("serverInfo", {}).get("protocolVersion")
            or self._protocol_version
        )

    def __enter__(self) -> StdioMcpSession:
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def start(self) -> StdioMcpSession:
        if self.active:
            return self
        if self._process is not None or self._reader is not None or self._windows_job is not None:
            self.close()
        if not self._server.enabled:
            raise McpClientError(f"MCP server is disabled: {self._server.name}")
        if self._server.transport != "stdio":
            raise McpClientError(
                f"Unsupported MCP transport for {self._server.name}: {self._server.transport}"
            )

        env = (
            dict(os.environ)
            if self._base_environment is None
            else {str(key): str(value) for key, value in self._base_environment.items()}
        )
        env.update(self._server.env)
        popen_kwargs: dict[str, Any] = {}
        command = [self._server.command, *self._server.args]
        windows_job: object | None = None
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            try:
                from ._win32_job import WindowsKillJob

                windows_job = WindowsKillJob()
            except Exception as exc:
                raise McpClientError(
                    "Failed to create the Windows MCP process containment job."
                ) from exc
            command = [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).with_name("_windows_stdio_launcher.py").resolve()),
                *command,
            ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="strict",
                cwd=(
                    str(Path(self._server.cwd).expanduser())
                    if self._server.cwd
                    else str(Path.cwd())
                ),
                env=env,
                **popen_kwargs,
            )
        except OSError as exc:
            close_job = getattr(windows_job, "close", None)
            if close_job is not None:
                close_job()
            raise McpClientError(
                f"Failed to start MCP server {self._server.name}: {exc}"
            ) from exc
        if windows_job is not None:
            try:
                process_handle = int(getattr(process, "_handle"))
                getattr(windows_job, "assign")(process_handle)
                if process.stdin is None:
                    raise McpClientError("Windows MCP launcher stdin is unavailable.")
                process.stdin.write("\0")
                process.stdin.flush()
            except Exception as exc:
                _close_process(process, windows_job=windows_job)
                raise McpClientError(
                    "Failed to contain the Windows MCP process before launch."
                ) from exc
        stdout_queue: Queue[object] = Queue(maxsize=_MAX_MCP_QUEUED_MESSAGES)
        reader = threading.Thread(
            target=_read_lines,
            args=(process.stdout, stdout_queue),
            daemon=True,
        )
        reader.start()
        self._process = process
        self._stdout_queue = stdout_queue
        self._reader = reader
        self._windows_job = windows_job
        try:
            self._initialize = self._request(
                1,
                "initialize",
                {
                    "protocolVersion": self._protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "myMoE", "version": "0.1.0"},
                },
            )
            self._notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        process = self._process
        reader = self._reader
        windows_job = self._windows_job
        self._process = None
        self._stdout_queue = None
        self._reader = None
        self._windows_job = None
        self._initialize = {}
        if process is not None:
            _close_process(process, windows_job=windows_job)
        else:
            close_job = getattr(windows_job, "close", None)
            if close_job is not None:
                close_job()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)

    def list_tools(self, *, timeout_seconds: float | None = None) -> McpToolList:
        result = self._request(
            self._request_id(),
            "tools/list",
            {},
            timeout_seconds=timeout_seconds,
        )
        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raise McpClientError("MCP tools/list returned an invalid tools field.")
        tools = tuple(_parse_tool(item) for item in raw_tools)
        return McpToolList(
            server=self._server.name,
            tools=tools,
            protocol_version=self.protocol_version,
        )

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        result = self._request(
            self._request_id(),
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_seconds=timeout_seconds,
        )
        content = result.get("content", [])
        structured_content = result.get("structuredContent", {})
        meta = result.get("_meta", {})
        return McpToolCallResult(
            server=self._server.name,
            tool_name=tool_name,
            content=tuple(item for item in content if isinstance(item, dict)) if isinstance(content, list) else (),
            is_error=bool(result.get("isError", False)),
            structured_content=structured_content if isinstance(structured_content, dict) else {},
            meta=meta if isinstance(meta, dict) else {},
        )

    def _request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _request(
        self,
        request_id: int,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        process = self._process
        stdout_queue = self._stdout_queue
        if process is None or stdout_queue is None or process.poll() is not None:
            raise McpClientError("MCP session is not active.")
        with self._lock:
            _write_message(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
            )
            try:
                response = _read_response(
                    process,
                    stdout_queue,
                    request_id,
                    self._timeout_seconds
                    if timeout_seconds is None
                    else min(self._timeout_seconds, max(0.05, timeout_seconds)),
                )
            except Exception:
                self.close()
                raise
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
        method: str,
        params: dict[str, Any],
    ) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            raise McpClientError("MCP session is not active.")
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


def mcp_tool_call_payload(result: McpToolCallResult) -> dict[str, Any]:
    return {
        "server": result.server,
        "tool_name": result.tool_name,
        "is_error": result.is_error,
        "content": list(result.content),
        "structured_content": result.structured_content,
        "meta": result.meta,
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
    stdout_queue: Queue[object],
    request_id: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise McpClientError(f"MCP process exited with code {process.returncode}.")
        try:
            event = stdout_queue.get(timeout=max(0.05, min(0.25, deadline - time.monotonic())))
        except Empty:
            continue
        if isinstance(event, McpClientError):
            raise event
        if not isinstance(event, str):
            raise McpClientError("MCP stdout reader returned an invalid event.")
        line = event
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(message, dict):
            continue
        if message.get("id") != request_id:
            continue
        return message
    raise McpClientError(f"Timed out waiting for MCP response id {request_id}.")


def _read_lines(stream: object, output: Queue[object]) -> None:
    if stream is None:
        return
    try:
        while True:
            line = stream.readline(_MAX_MCP_MESSAGE_CHARS + 1)
            if not line:
                return
            if len(line) > _MAX_MCP_MESSAGE_CHARS:
                _queue_reader_event(
                    output,
                    McpClientError("MCP response exceeded the bounded message size."),
                )
                return
            try:
                output.put(line, timeout=0.5)
            except Full:
                return
    except (OSError, UnicodeError):
        _queue_reader_event(output, McpClientError("MCP stdout was not valid UTF-8."))


def _queue_reader_event(output: Queue[object], event: object) -> None:
    try:
        output.put(event, timeout=0.1)
    except Full:
        return


def _close_process(
    process: subprocess.Popen[str],
    *,
    windows_job: object | None = None,
) -> None:
    try:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    elif windows_job is None and process.poll() is None:
        process.terminate()
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            elif windows_job is None:
                process.kill()
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    close_job = getattr(windows_job, "close", None)
    if close_job is not None:
        close_job()
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name != "posix":
                process.kill()
                process.wait(timeout=2)
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()
