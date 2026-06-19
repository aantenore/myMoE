from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .compaction import LocalCompactionProvider
from .context import ConversationTurn, build_compaction_prompt, estimate_tokens
from .extensions import (
    ExtensionRegistry,
    McpServerDefinition,
    ToolDefinition,
    audit_extension_registry,
    configure_extension_entry,
    create_plugin_scaffold,
    load_extension_registry,
    registry_payload,
)
from .mcp_client import (
    McpClientError,
    StdioMcpClient,
    mcp_tool_call_payload,
    mcp_tool_list_payload,
)
from .memory import FileMemoryStore


class ToolExecutionError(ValueError):
    """Raised when an allowlisted local tool cannot be executed."""


@dataclass(frozen=True)
class ToolRunResult:
    name: str
    status: str
    risk_class: str
    side_effects: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


class LocalToolRunner:
    """Execute configured local tools without exposing arbitrary process execution."""

    _SUPPORTED = {
        "memory.search",
        "memory.forget",
        "knowledge.ingest",
        "context.compact",
        "extension.audit",
        "extension.configure",
        "plugin.create",
        "mcp.search_capabilities",
        "mcp.list_tools",
        "mcp.call_tool",
    }

    def __init__(
        self,
        registry: ExtensionRegistry,
        *,
        app_config: object | None = None,
        moe_config: object | None = None,
        memory_path: str | Path | None = None,
        plugins_dir: str | Path | None = None,
        allow_process_execution: bool | None = None,
    ):
        self._registry = registry
        self._app_config = app_config
        self._moe_config = moe_config
        self._allow_process_execution = (
            bool(allow_process_execution)
            if allow_process_execution is not None
            else bool(getattr(getattr(app_config, "permissions", None), "allow_process_execution", False))
        )
        self._memory_path = (
            Path(memory_path)
            if memory_path is not None
            else _runtime_path(app_config, "memory.jsonl")
        )
        self._plugins_dir = Path(plugins_dir) if plugins_dir is not None else _plugins_dir(app_config)

    def run(self, name: str, payload: dict[str, Any] | None = None) -> ToolRunResult:
        tool = self._tool(name)
        tool_payload = payload or {}
        if not isinstance(tool_payload, dict):
            raise ToolExecutionError("Tool input must be a JSON object.")

        if tool.name == "memory.search":
            return self._memory_search(tool, tool_payload)
        if tool.name == "memory.forget":
            return self._memory_forget(tool, tool_payload)
        if tool.name == "knowledge.ingest":
            return self._knowledge_ingest(tool, tool_payload)
        if tool.name == "context.compact":
            return self._context_compact(tool, tool_payload)
        if tool.name == "extension.audit":
            return self._extension_audit(tool, tool_payload)
        if tool.name == "extension.configure":
            return self._extension_configure(tool, tool_payload)
        if tool.name == "plugin.create":
            return self._plugin_create(tool, tool_payload)
        if tool.name == "mcp.search_capabilities":
            return self._mcp_search_capabilities(tool, tool_payload)
        if tool.name == "mcp.list_tools":
            return self._mcp_list_tools(tool, tool_payload)
        if tool.name == "mcp.call_tool":
            return self._mcp_call_tool(tool, tool_payload)

        raise ToolExecutionError(f"Unsupported tool: {tool.name}")

    def _tool(self, name: str) -> ToolDefinition:
        requested = str(name).strip()
        if not requested:
            raise ToolExecutionError("Tool name is required.")
        if requested not in self._SUPPORTED:
            raise ToolExecutionError(f"Unsupported tool: {requested}")

        for tool in self._registry.tools:
            if tool.name == requested:
                if not tool.enabled:
                    raise ToolExecutionError(f"Tool is disabled: {requested}")
                return tool
        raise ToolExecutionError(f"Tool is not configured: {requested}")

    def _memory_search(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        query = _required_text(payload, "query")
        scope = _optional_text(payload, "scope")
        limit = _int_in_range(payload.get("limit", 8), "limit", minimum=1, maximum=50)
        results = FileMemoryStore(self._memory_path).search(query, scope=scope, limit=limit)
        return _ok(
            tool,
            "Memory search completed.",
            {
                "query": query,
                "scope": scope,
                "memory_path": str(self._memory_path),
                "count": len(results),
                "records": [
                    {
                        "id": record.id,
                        "scope": record.scope,
                        "kind": record.kind,
                        "text": record.text,
                        "metadata": record.metadata,
                        "created_at": record.created_at,
                        "score": score,
                    }
                    for record, score in results
                ],
            },
        )

    def _memory_forget(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if payload.get("confirm") is not True:
            raise ToolExecutionError("memory.forget requires confirm=true because it deletes local records.")
        record_id = _optional_text(payload, "record_id")
        document_id = _optional_text(payload, "document_id")
        if bool(record_id) == bool(document_id):
            raise ToolExecutionError("memory.forget requires exactly one of record_id or document_id.")
        store = FileMemoryStore(self._memory_path)
        try:
            report = store.forget_record(record_id) if record_id else store.forget_document(document_id or "")
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return _ok(
            tool,
            "Local memory records removed.",
            {
                "target": report.target,
                "removed_count": report.removed_count,
                "remaining_count": report.remaining_count,
                "removed_ids": list(report.removed_ids),
                "memory_path": str(self._memory_path),
            },
        )

    def _knowledge_ingest(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if payload.get("confirm") is not True:
            raise ToolExecutionError("knowledge.ingest requires confirm=true because it writes local memory records.")
        title = _required_text(payload, "title")
        content = _required_text(payload, "content")
        scope = _optional_text(payload, "scope") or "default"
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ToolExecutionError("metadata must be a JSON object.")
        chunk_chars = _int_in_range(payload.get("chunk_chars", 1200), "chunk_chars", minimum=200, maximum=8000)
        try:
            report = FileMemoryStore(self._memory_path).ingest_document(
                content,
                title=title,
                scope=scope,
                chunk_chars=chunk_chars,
                metadata=metadata,
            )
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return _ok(
            tool,
            "Knowledge document ingested into local memory.",
            {
                "document_id": report.document_id,
                "title": report.title,
                "scope": report.scope,
                "chunk_count": report.chunk_count,
                "record_ids": list(report.record_ids),
                "memory_path": str(self._memory_path),
            },
        )

    def _context_compact(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        turns = _turns(payload.get("turns", []))
        existing_summary = _optional_text(payload, "existing_summary") or ""
        prompt = build_compaction_prompt(turns=turns, existing_summary=existing_summary)
        result_payload: dict[str, Any] = {
            "turn_count": len(turns),
            "existing_summary_present": bool(existing_summary.strip()),
            "prompt": prompt,
            "prompt_token_estimate": estimate_tokens(prompt),
        }

        if bool(payload.get("use_model", True)):
            if self._moe_config is None:
                raise ToolExecutionError("context.compact with use_model=true requires a MoE config.")
            compactor = LocalCompactionProvider(
                self._moe_config,
                expert_id=_optional_text(payload, "expert_id"),
            )
            compacted = compactor.compact(
                turns=turns,
                existing_summary=existing_summary,
                correlation_id=_optional_text(payload, "correlation_id") or str(uuid4()),
            )
            result_payload.update(
                {
                    "summary": compacted.summary,
                    "expert_id": compacted.expert_id,
                    "model": compacted.model,
                    "correlation_id": compacted.correlation_id,
                }
            )
            return _ok(tool, "Context compaction completed with the configured local model.", result_payload)

        return _ok(tool, "Context compaction prompt prepared.", result_payload)

    def _extension_audit(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        return _ok(
            tool,
            "Extension registry audit completed.",
            audit_extension_registry(self._registry),
        )

    def _extension_configure(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if payload.get("confirm") is not True:
            raise ToolExecutionError(
                "extension.configure requires confirm=true because it writes local registry files."
            )
        if self._app_config is None:
            raise ToolExecutionError("extension.configure requires an app config.")
        extensions = getattr(self._app_config, "extensions", None)
        if extensions is None:
            raise ToolExecutionError("extension.configure requires configured extension paths.")
        definition = payload.get("definition", {})
        if not isinstance(definition, dict):
            raise ToolExecutionError("definition must be a JSON object.")

        result = configure_extension_entry(
            _required_text(payload, "surface"),
            definition,
            mode=_optional_text(payload, "mode") or "upsert",
            mcp_config=getattr(extensions, "mcp_config"),
            cron_config=getattr(extensions, "cron_config"),
        )
        registry = load_extension_registry(
            plugins_dir=getattr(extensions, "plugins_dir"),
            skills_dir=getattr(extensions, "skills_dir"),
            tools_config=getattr(extensions, "tools_config"),
            mcp_config=getattr(extensions, "mcp_config"),
            cron_config=getattr(extensions, "cron_config"),
        )
        result.update(
            {
                "audit": audit_extension_registry(registry),
                "extensions": registry_payload(registry),
            }
        )
        return _ok(tool, "Extension registry configuration updated.", result)

    def _plugin_create(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if payload.get("confirm") is not True:
            raise ToolExecutionError("plugin.create requires confirm=true because it writes local files.")
        plugin_id = _required_text(payload, "plugin_id")
        path = create_plugin_scaffold(
            plugin_id,
            root=self._plugins_dir,
            name=_optional_text(payload, "name"),
            description=_optional_text(payload, "description"),
            risk_class=_optional_text(payload, "risk_class") or "read_only",
        )
        return _ok(
            tool,
            "Plugin scaffold created.",
            {
                "plugin_id": plugin_id,
                "path": str(path),
                "manifest": str(path / "plugin.json"),
                "skill": str(path / "SKILL.md"),
            },
        )

    def _mcp_search_capabilities(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        query = (_optional_text(payload, "query") or "").lower()
        servers = []
        for server in self._registry.mcp_servers:
            haystack = " ".join(
                [
                    server.name,
                    server.description,
                    server.risk_class,
                    *server.capabilities,
                ]
            ).lower()
            if query and query not in haystack:
                continue
            servers.append(
                {
                    "name": server.name,
                    "description": server.description,
                    "enabled": server.enabled,
                    "risk_class": server.risk_class,
                    "capabilities": list(server.capabilities),
                    "transport": server.transport,
                    "cwd": server.cwd,
                    "timeout_seconds": server.timeout_seconds,
                    "allowed_tools": list(server.allowed_tools),
                    "command": server.command,
                    "args": list(server.args),
                }
            )

        return _ok(
            tool,
            "MCP capabilities returned.",
            {
                "query": query or None,
                "count": len(servers),
                "servers": servers,
            },
        )

    def _mcp_list_tools(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if not self._allow_process_execution:
            raise ToolExecutionError(
                "mcp.list_tools is disabled by app permissions; set allow_process_execution=true in the app config."
            )
        if payload.get("confirm_process_execution") is not True:
            raise ToolExecutionError(
                "mcp.list_tools requires confirm_process_execution=true because it starts an MCP server process."
            )
        server = self._mcp_server(_required_text(payload, "server"))
        try:
            timeout = float(payload.get("timeout_seconds", server.timeout_seconds))
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("timeout_seconds must be a number.") from exc
        if timeout <= 0 or timeout > 30:
            raise ToolExecutionError("timeout_seconds must be greater than 0 and at most 30.")
        try:
            result = StdioMcpClient(server, timeout_seconds=timeout).list_tools()
        except McpClientError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return _ok(tool, "MCP tools listed.", mcp_tool_list_payload(result))

    def _mcp_call_tool(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if not self._allow_process_execution:
            raise ToolExecutionError(
                "mcp.call_tool is disabled by app permissions; set allow_process_execution=true in the app config."
            )
        if payload.get("confirm_process_execution") is not True:
            raise ToolExecutionError(
                "mcp.call_tool requires confirm_process_execution=true because it starts an MCP server process."
            )
        if payload.get("confirm_tool_call") is not True:
            raise ToolExecutionError("mcp.call_tool requires confirm_tool_call=true.")

        server = self._mcp_server(_required_text(payload, "server"))
        tool_name = _required_text(payload, "tool_name")
        if tool_name not in server.allowed_tools:
            raise ToolExecutionError(f"MCP tool is not allowlisted for server {server.name}: {tool_name}")
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ToolExecutionError("arguments must be a JSON object.")
        try:
            timeout = float(payload.get("timeout_seconds", server.timeout_seconds))
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("timeout_seconds must be a number.") from exc
        if timeout <= 0 or timeout > 30:
            raise ToolExecutionError("timeout_seconds must be greater than 0 and at most 30.")
        try:
            result = StdioMcpClient(server, timeout_seconds=timeout).call_tool(tool_name, arguments)
        except McpClientError as exc:
            raise ToolExecutionError(str(exc)) from exc
        message = "MCP tool returned an error." if result.is_error else "MCP tool called."
        return _ok(tool, message, mcp_tool_call_payload(result))

    def _mcp_server(self, name: str) -> McpServerDefinition:
        for server in self._registry.mcp_servers:
            if server.name == name:
                return server
        raise ToolExecutionError(f"MCP server is not configured: {name}")


def tool_result_payload(result: ToolRunResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "status": result.status,
        "risk_class": result.risk_class,
        "side_effects": result.side_effects,
        "message": result.message,
        "payload": result.payload,
    }


def _ok(tool: ToolDefinition, message: str, payload: dict[str, Any]) -> ToolRunResult:
    return ToolRunResult(
        name=tool.name,
        status="ok",
        risk_class=tool.risk_class,
        side_effects=tool.side_effects,
        message=message,
        payload=payload,
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ToolExecutionError(f"{key} is required.")
    return value


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    raw = payload.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _int_in_range(raw: object, key: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(f"{key} must be an integer.") from exc
    if value < minimum or value > maximum:
        raise ToolExecutionError(f"{key} must be between {minimum} and {maximum}.")
    return value


def _turns(raw: object) -> tuple[ConversationTurn, ...]:
    if not isinstance(raw, list):
        raise ToolExecutionError("turns must be a list of {role, content} objects.")
    turns = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ToolExecutionError(f"turns[{index}] must be an object.")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if not role or not content:
            raise ToolExecutionError(f"turns[{index}] requires role and content.")
        turns.append(ConversationTurn(role=role, content=content))
    return tuple(turns)


def _runtime_path(app_config: object | None, filename: str) -> Path:
    work_dir = getattr(getattr(app_config, "runtime", None), "work_dir", "work/runtime")
    return Path(str(work_dir)) / filename


def _plugins_dir(app_config: object | None) -> Path:
    plugins_dir = getattr(getattr(app_config, "extensions", None), "plugins_dir", "plugins")
    return Path(str(plugins_dir))
