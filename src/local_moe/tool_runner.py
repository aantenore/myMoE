from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .compaction import LocalCompactionProvider
from .context import ConversationTurn, build_compaction_prompt, estimate_tokens
from .extensions import (
    ExtensionRegistry,
    ToolDefinition,
    create_plugin_scaffold,
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
        "context.compact",
        "plugin.create",
        "mcp.search_capabilities",
    }

    def __init__(
        self,
        registry: ExtensionRegistry,
        *,
        app_config: object | None = None,
        moe_config: object | None = None,
        memory_path: str | Path | None = None,
        plugins_dir: str | Path | None = None,
    ):
        self._registry = registry
        self._moe_config = moe_config
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
        if tool.name == "context.compact":
            return self._context_compact(tool, tool_payload)
        if tool.name == "plugin.create":
            return self._plugin_create(tool, tool_payload)
        if tool.name == "mcp.search_capabilities":
            return self._mcp_search_capabilities(tool, tool_payload)

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

    def _plugin_create(self, tool: ToolDefinition, payload: dict[str, Any]) -> ToolRunResult:
        if payload.get("confirm") is not True:
            raise ToolExecutionError("plugin.create requires confirm=true because it writes local files.")
        plugin_id = _required_text(payload, "plugin_id")
        path = create_plugin_scaffold(plugin_id, root=self._plugins_dir)
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
