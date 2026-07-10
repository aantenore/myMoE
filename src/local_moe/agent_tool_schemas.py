from __future__ import annotations

from typing import Any, Mapping, Sequence


def _object_schema(
    properties: Mapping[str, Any] | None = None,
    *,
    required: Sequence[str] = (),
    one_of: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": False,
    }
    if one_of:
        schema["oneOf"] = list(one_of)
    return schema


_TEXT = {"type": "string", "minLength": 1}
_OPTIONAL_TEXT = {"type": "string"}
_OPAQUE_OBJECT = {"type": "object", "additionalProperties": True}
_RISK_ENUM = [
    "read_only",
    "search_only",
    "compute_only",
    "draft_only",
    "write_local",
    "write_internal",
    "write_external",
    "financial",
    "communication",
    "identity_access",
    "security_sensitive",
    "process_execution",
    "network_open_world",
    "destructive",
    "privileged_admin",
]

LOCAL_TOOL_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "memory.search": _object_schema(
        {
            "query": _TEXT,
            "scope": _OPTIONAL_TEXT,
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        required=("query",),
    ),
    "memory.maintenance": _object_schema({"now": _OPTIONAL_TEXT}),
    "memory.prune_expired": _object_schema({"now": _OPTIONAL_TEXT}),
    "memory.forget": _object_schema(
        {"record_id": _TEXT, "document_id": _TEXT},
        one_of=({"required": ["record_id"]}, {"required": ["document_id"]}),
    ),
    "knowledge.ingest": _object_schema(
        {
            "title": _TEXT,
            "content": _TEXT,
            "scope": _OPTIONAL_TEXT,
            "metadata": _OPAQUE_OBJECT,
            "chunk_chars": {"type": "integer", "minimum": 200, "maximum": 8000},
        },
        required=("title", "content"),
    ),
    "data.export": _object_schema(),
    "data.import": _object_schema(
        {
            "bundle": _OPAQUE_OBJECT,
            "mode": {"type": "string", "enum": ["merge", "replace"]},
        },
        required=("bundle",),
    ),
    "context.compact": _object_schema(
        {
            "turns": {
                "type": "array",
                "items": _object_schema(
                    {"role": _TEXT, "content": _TEXT},
                    required=("role", "content"),
                ),
                "maxItems": 200,
            },
            "existing_summary": _OPTIONAL_TEXT,
            "use_model": {"type": "boolean"},
            "expert_id": _OPTIONAL_TEXT,
            "correlation_id": _OPTIONAL_TEXT,
        }
    ),
    "extension.audit": _object_schema(),
    "extension.configure": _object_schema(
        {
            "surface": {"type": "string", "enum": ["mcp_server", "cron_job"]},
            "definition": _OPAQUE_OBJECT,
            "mode": {"type": "string", "enum": ["upsert", "remove"]},
        },
        required=("surface", "definition"),
    ),
    "profile.activate": _object_schema(
        {"profile_path": _TEXT, "recommended": {"type": "boolean", "enum": [True]}},
        one_of=({"required": ["profile_path"]}, {"required": ["recommended"]}),
    ),
    "storage.inspect": _object_schema(
        {"min_free_gib": {"type": "number", "minimum": 0, "maximum": 1_000_000_000}}
    ),
    "models.inventory": _object_schema(
        {"max_files": {"type": "integer", "minimum": 1, "maximum": 200_000}}
    ),
    "security.audit": _object_schema(),
    "plugin.create": _object_schema(
        {
            "plugin_id": {
                "type": "string",
                "pattern": r"[a-z][a-z0-9-]{1,63}",
            },
            "name": _OPTIONAL_TEXT,
            "description": _OPTIONAL_TEXT,
            "risk_class": {"type": "string", "enum": _RISK_ENUM},
        },
        required=("plugin_id",),
    ),
    "mcp.search_capabilities": _object_schema({"query": _OPTIONAL_TEXT}),
    "mcp.list_tools": _object_schema(
        {
            "server": _TEXT,
            "timeout_seconds": {"type": "number", "minimum": 0.001, "maximum": 30},
        },
        required=("server",),
    ),
    "mcp.call_tool": _object_schema(
        {
            "server": _TEXT,
            "tool_name": _TEXT,
            "arguments": _OPAQUE_OBJECT,
            "timeout_seconds": {"type": "number", "minimum": 0.001, "maximum": 30},
        },
        required=("server", "tool_name"),
    ),
}


LOCAL_TOOL_CONFIRMATIONS: dict[str, dict[str, bool]] = {
    "memory.prune_expired": {"confirm": True},
    "memory.forget": {"confirm": True},
    "knowledge.ingest": {"confirm": True},
    "data.export": {"confirm": True},
    "data.import": {"confirm": True},
    "extension.configure": {"confirm": True},
    "profile.activate": {"confirm": True},
    "plugin.create": {"confirm": True},
    "mcp.list_tools": {"confirm_process_execution": True},
    "mcp.call_tool": {
        "confirm_process_execution": True,
        "confirm_tool_call": True,
    },
}
