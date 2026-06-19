from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


class ExtensionError(ValueError):
    """Raised when extension metadata is invalid."""


VALID_ID = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
RISK_CLASSES = {
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
}

CRON_ACTION_PRESETS = {
    "extension.audit": {
        "description": "Validate configured extension surfaces.",
        "risk_class": "compute_only",
    },
    "memory.maintenance": {
        "description": "Report local memory store health.",
        "risk_class": "compute_only",
    },
    "memory.prune_expired": {
        "description": "Delete expired local memory records after explicit confirmation.",
        "risk_class": "write_local",
    },
    "router.distill": {
        "description": "Refresh the local distilled router artifact from curated labels.",
        "risk_class": "write_local",
    },
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    risk_class: str
    side_effects: str
    enabled: bool


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    path: str
    enabled: bool = True


@dataclass(frozen=True)
class McpServerDefinition:
    name: str
    description: str
    command: str
    args: tuple[str, ...]
    enabled: bool
    risk_class: str
    capabilities: tuple[str, ...]
    transport: str = "stdio"
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 8.0
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class CronJobDefinition:
    id: str
    description: str
    enabled: bool
    schedule: dict[str, Any]
    command: tuple[str, ...]
    risk_class: str


@dataclass(frozen=True)
class PluginManifest:
    id: str
    name: str
    version: str
    description: str
    path: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    mcp_servers: tuple[str, ...]
    cron_jobs: tuple[str, ...]
    permissions: dict[str, Any]


@dataclass(frozen=True)
class ExtensionRegistry:
    tools: tuple[ToolDefinition, ...]
    skills: tuple[SkillDefinition, ...]
    mcp_servers: tuple[McpServerDefinition, ...]
    cron_jobs: tuple[CronJobDefinition, ...]
    plugins: tuple[PluginManifest, ...]


@dataclass(frozen=True)
class ExtensionAuditIssue:
    plugin_id: str
    surface: str
    reference: str
    message: str


def load_extension_registry(
    *,
    plugins_dir: str | Path = "plugins",
    skills_dir: str | Path = "skills",
    tools_config: str | Path = "configs/tools.json",
    mcp_config: str | Path = "configs/mcp.json",
    cron_config: str | Path = "configs/cron.json",
) -> ExtensionRegistry:
    plugins = tuple(load_plugins(plugins_dir))
    return ExtensionRegistry(
        tools=tuple(load_tools(tools_config)),
        skills=tuple(load_skills(skills_dir)) + tuple(load_plugin_skills(plugins)),
        mcp_servers=tuple(load_mcp_servers(mcp_config)),
        cron_jobs=tuple(load_cron_jobs(cron_config)),
        plugins=plugins,
    )


def load_tools(path: str | Path) -> list[ToolDefinition]:
    raw = _read_json_if_exists(path, {"tools": []})
    return [_parse_tool(item) for item in raw.get("tools", [])]


def load_mcp_servers(path: str | Path) -> list[McpServerDefinition]:
    raw = _read_json_if_exists(path, {"servers": []})
    return [_parse_mcp_server(item) for item in raw.get("servers", [])]


def load_cron_jobs(path: str | Path) -> list[CronJobDefinition]:
    raw = _read_json_if_exists(path, {"jobs": []})
    return [_parse_cron_job(item) for item in raw.get("jobs", [])]


def load_skills(root: str | Path) -> list[SkillDefinition]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    skills = []
    for skill_file in sorted(root_path.glob("*/SKILL.md")):
        metadata = _read_frontmatter(skill_file)
        name = str(metadata.get("name") or skill_file.parent.name)
        _validate_id(name)
        skills.append(
            SkillDefinition(
                name=name,
                description=str(metadata.get("description", "")),
                path=str(skill_file),
            )
        )
    return skills


def load_plugin_skills(plugins: tuple[PluginManifest, ...] | list[PluginManifest]) -> list[SkillDefinition]:
    skills = []
    for plugin in plugins:
        skill_file = Path(plugin.path) / "SKILL.md"
        if not skill_file.exists():
            continue
        metadata = _read_frontmatter(skill_file)
        name = str(metadata.get("name") or plugin.id)
        _validate_id(name)
        skills.append(
            SkillDefinition(
                name=name,
                description=str(metadata.get("description", "")),
                path=str(skill_file),
            )
        )
    return skills


def load_plugins(root: str | Path) -> list[PluginManifest]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    plugins = []
    for manifest_path in sorted(root_path.glob("*/plugin.json")):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        plugin_id = str(raw["id"])
        _validate_id(plugin_id)
        plugins.append(
            PluginManifest(
                id=plugin_id,
                name=str(raw.get("name", plugin_id)),
                version=str(raw.get("version", "0.0.0")),
                description=str(raw.get("description", "")),
                path=str(manifest_path.parent),
                skills=tuple(str(item) for item in raw.get("skills", [])),
                tools=tuple(str(item) for item in raw.get("tools", [])),
                mcp_servers=tuple(str(item) for item in raw.get("mcp_servers", [])),
                cron_jobs=tuple(str(item) for item in raw.get("cron_jobs", [])),
                permissions=dict(raw.get("permissions", {})),
            )
        )
    return plugins


def create_plugin_scaffold(
    plugin_id: str,
    *,
    root: str | Path = "plugins",
    name: str | None = None,
    description: str | None = None,
    risk_class: str = "read_only",
) -> Path:
    _validate_id(plugin_id)
    _validate_risk(risk_class)
    plugin_dir = Path(root) / plugin_id
    if plugin_dir.exists():
        raise ExtensionError(f"Plugin already exists: {plugin_id}")
    plugin_dir.mkdir(parents=True)
    plugin_name = _clean_generated_text(name or plugin_id.replace("-", " ").title())
    plugin_description = _clean_generated_text(description or "Local myMoE plugin scaffold.")
    manifest = {
        "id": plugin_id,
        "name": plugin_name,
        "version": "0.1.0",
        "description": plugin_description,
        "skills": [plugin_id],
        "tools": [],
        "mcp_servers": [],
        "cron_jobs": [],
        "permissions": {"risk_class": risk_class},
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (plugin_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {_frontmatter_scalar(plugin_id)}\n"
        f"description: {_frontmatter_scalar(f'Use this skill when working with the {plugin_name} plugin.')}\n"
        "---\n\n"
        f"# {plugin_name}\n\n"
        f"{plugin_description}\n\n"
        "Describe plugin behavior, inputs, validation, and gotchas here.\n",
        encoding="utf-8",
    )
    return plugin_dir


def configure_extension_entry(
    surface: str,
    definition: dict[str, Any],
    *,
    mode: str = "upsert",
    mcp_config: str | Path = "configs/mcp.json",
    cron_config: str | Path = "configs/cron.json",
) -> dict[str, Any]:
    """Add, update, or remove extension entries in the configured registry files."""

    if not isinstance(definition, dict):
        raise ExtensionError("definition must be an object.")
    normalized_surface = str(surface).strip()
    normalized_mode = str(mode).strip() or "upsert"
    if normalized_mode not in {"upsert", "remove"}:
        raise ExtensionError("mode must be upsert or remove.")

    if normalized_surface == "mcp_server":
        path = Path(mcp_config)
        list_key = "servers"
        identity_key = "name"
        parser = _parse_mcp_server
    elif normalized_surface == "cron_job":
        path = Path(cron_config)
        list_key = "jobs"
        identity_key = "id"
        parser = _parse_cron_job
    else:
        raise ExtensionError("surface must be mcp_server or cron_job.")

    raw = _read_json_if_exists(path, {list_key: []})
    entries = list(raw.get(list_key, []))
    if not isinstance(entries, list):
        raise ExtensionError(f"{path} field {list_key} must be a list.")

    identity = str(definition.get(identity_key, "")).strip()
    _validate_id(identity)

    before_count = len(entries)
    existing_index = next(
        (index for index, item in enumerate(entries) if str(item.get(identity_key, "")).strip() == identity),
        None,
    )

    if normalized_mode == "remove":
        if existing_index is None:
            action = "missing"
        else:
            del entries[existing_index]
            action = "removed"
    else:
        parsed = parser(definition)
        entry = _entry_payload(parsed)
        if existing_index is None:
            entries.append(entry)
            action = "created"
        else:
            entries[existing_index] = entry
            action = "updated"

    raw[list_key] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

    return {
        "surface": normalized_surface,
        "mode": normalized_mode,
        "action": action,
        "id": identity,
        "path": str(path),
        "before_count": before_count,
        "after_count": len(entries),
    }


def registry_payload(registry: ExtensionRegistry) -> dict[str, Any]:
    return {
        "tools": [item.__dict__ for item in registry.tools],
        "skills": [item.__dict__ for item in registry.skills],
        "mcp_servers": [item.__dict__ for item in registry.mcp_servers],
        "cron_jobs": [item.__dict__ for item in registry.cron_jobs],
        "plugins": [item.__dict__ for item in registry.plugins],
    }


def audit_extension_registry(registry: ExtensionRegistry) -> dict[str, Any]:
    tool_names = {tool.name for tool in registry.tools}
    skill_refs = _skill_reference_set(registry.skills)
    mcp_names = {server.name for server in registry.mcp_servers}
    cron_ids = {job.id for job in registry.cron_jobs}
    issues: list[ExtensionAuditIssue] = []

    for plugin in registry.plugins:
        issues.extend(_missing_refs(plugin, "tool", plugin.tools, tool_names))
        issues.extend(_missing_refs(plugin, "skill", plugin.skills, skill_refs))
        issues.extend(_missing_refs(plugin, "mcp_server", plugin.mcp_servers, mcp_names))
        issues.extend(_missing_refs(plugin, "cron_job", plugin.cron_jobs, cron_ids))
        risk_class = str(plugin.permissions.get("risk_class", "read_only"))
        if risk_class not in RISK_CLASSES:
            issues.append(
                ExtensionAuditIssue(
                    plugin_id=plugin.id,
                    surface="permission",
                    reference=risk_class,
                    message=f"Unknown plugin risk class: {risk_class}",
                )
            )

    return {
        "checked": True,
        "plugin_count": len(registry.plugins),
        "issue_count": len(issues),
        "issues": [issue.__dict__ for issue in issues],
    }


def extension_configuration_templates() -> dict[str, Any]:
    """Return safe starter templates for guided extension configuration."""

    return {
        "risk_classes": sorted(RISK_CLASSES),
        "surfaces": [
            {
                "id": "mcp_server",
                "label": "MCP Server",
                "identity_key": "name",
                "description": "Register a guarded stdio MCP server and its allowlisted tools.",
            },
            {
                "id": "cron_job",
                "label": "Cron Job",
                "identity_key": "id",
                "description": "Register a local scheduled job from the allowlisted cron actions.",
            },
        ],
        "presets": {
            "mcp_server": [
                {
                    "id": "filesystem-docs",
                    "label": "Local Docs Filesystem",
                    "description": "Disabled read/write-local filesystem MCP bridge scoped to docs by default.",
                    "definition": {
                        "name": "local-docs",
                        "description": "Read docs through allowlisted MCP.",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "docs"],
                        "enabled": False,
                        "risk_class": "write_local",
                        "capabilities": ["resources", "tools"],
                        "transport": "stdio",
                        "cwd": ".",
                        "env": {},
                        "timeout_seconds": 8,
                        "allowed_tools": [
                            "list_allowed_directories",
                            "list_directory",
                            "directory_tree",
                            "get_file_info",
                            "search_files",
                            "read_text_file",
                        ],
                    },
                },
                {
                    "id": "custom-stdio",
                    "label": "Custom Stdio MCP",
                    "description": "Blank stdio MCP server starter with process execution disabled by default.",
                    "definition": {
                        "name": "custom-mcp",
                        "description": "Custom local stdio MCP server.",
                        "command": "python",
                        "args": ["server.py"],
                        "enabled": False,
                        "risk_class": "process_execution",
                        "capabilities": ["tools"],
                        "transport": "stdio",
                        "cwd": ".",
                        "env": {},
                        "timeout_seconds": 8,
                        "allowed_tools": [],
                    },
                },
            ],
            "cron_job": [
                {
                    "id": "startup-extension-audit",
                    "label": "Startup Extension Audit",
                    "description": "Run a read-only extension registry audit once when the app starts.",
                    "definition": {
                        "id": "extension-audit",
                        "description": "Validate configured extension surfaces at startup.",
                        "enabled": True,
                        "schedule": {"type": "startup"},
                        "command": ["extension.audit"],
                        "risk_class": "compute_only",
                    },
                },
                {
                    "id": "daily-memory-maintenance",
                    "label": "Daily Memory Maintenance",
                    "description": "Report memory health and expired-record counts once per day.",
                    "definition": {
                        "id": "memory-maintenance",
                        "description": "Report local memory store health and expired-record counts.",
                        "enabled": True,
                        "schedule": {"type": "interval", "seconds": 86400},
                        "command": ["memory.maintenance", "--memory-path", "work/runtime/memory.jsonl"],
                        "risk_class": "compute_only",
                    },
                },
                {
                    "id": "weekly-router-distillation",
                    "label": "Weekly Router Distillation",
                    "description": "Refresh the distilled router artifact from the curated live eval labels.",
                    "definition": {
                        "id": "router-distillation-refresh",
                        "description": "Regenerate local route-label data and distilled router artifact.",
                        "enabled": True,
                        "schedule": {"type": "interval", "seconds": 604800},
                        "command": [
                            "router.distill",
                            "--eval",
                            "experiments/eval_set_live_general.jsonl",
                            "--labels",
                            "experiments/route_labels_live_general.jsonl",
                            "--artifact",
                            "outputs/router-distilled-live-general.json",
                            "--teacher-source",
                            "curated_live_eval",
                        ],
                        "risk_class": "write_local",
                    },
                },
            ],
        },
        "cron_actions": [
            {"id": name, **details}
            for name, details in sorted(CRON_ACTION_PRESETS.items())
        ],
    }


def _parse_tool(item: dict[str, Any]) -> ToolDefinition:
    name = str(item["name"])
    if "." not in name:
        _validate_id(name)
    risk_class = str(item.get("risk_class", "read_only"))
    _validate_risk(risk_class)
    return ToolDefinition(
        name=name,
        description=str(item.get("description", "")),
        risk_class=risk_class,
        side_effects=str(item.get("side_effects", "none")),
        enabled=bool(item.get("enabled", True)),
    )


def _parse_mcp_server(item: dict[str, Any]) -> McpServerDefinition:
    _validate_id(str(item["name"]))
    risk_class = str(item.get("risk_class", "read_only"))
    _validate_risk(risk_class)
    env_raw = item.get("env", {})
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise ExtensionError(f"MCP server {item['name']} env must be an object.")
    try:
        timeout_seconds = float(item.get("timeout_seconds", 8.0))
    except (TypeError, ValueError) as exc:
        raise ExtensionError(f"MCP server {item['name']} timeout_seconds must be numeric.") from exc
    if timeout_seconds <= 0:
        raise ExtensionError(f"MCP server {item['name']} timeout_seconds must be positive.")
    command = str(item.get("command", "")).strip()
    if not command:
        raise ExtensionError(f"MCP server {item['name']} command is required.")
    return McpServerDefinition(
        name=str(item["name"]),
        description=str(item.get("description", "")),
        command=command,
        args=tuple(str(arg) for arg in item.get("args", [])),
        enabled=bool(item.get("enabled", False)),
        risk_class=risk_class,
        capabilities=tuple(str(capability) for capability in item.get("capabilities", [])),
        transport=str(item.get("transport", "stdio")),
        cwd=str(item["cwd"]) if item.get("cwd") is not None else None,
        env={str(key): str(value) for key, value in env_raw.items()},
        timeout_seconds=timeout_seconds,
        allowed_tools=tuple(str(name) for name in item.get("allowed_tools", [])),
    )


def _parse_cron_job(item: dict[str, Any]) -> CronJobDefinition:
    _validate_id(str(item["id"]))
    risk_class = str(item.get("risk_class", "compute_only"))
    _validate_risk(risk_class)
    schedule = item.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ExtensionError(f"Cron job {item['id']} schedule must be an object.")
    command = tuple(str(arg) for arg in item.get("command", []))
    if not command:
        raise ExtensionError(f"Cron job {item['id']} command is required.")
    return CronJobDefinition(
        id=str(item["id"]),
        description=str(item.get("description", "")),
        enabled=bool(item.get("enabled", False)),
        schedule=dict(schedule),
        command=command,
        risk_class=risk_class,
    )


def _entry_payload(entry: McpServerDefinition | CronJobDefinition) -> dict[str, Any]:
    payload = entry.__dict__.copy()
    for key, value in tuple(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def _skill_reference_set(skills: tuple[SkillDefinition, ...]) -> set[str]:
    refs: set[str] = set()
    for skill in skills:
        path = Path(skill.path)
        refs.add(skill.name)
        refs.add(str(path))
        refs.add(str(path.parent))
        refs.add(path.parent.name)
        refs.add(f"{path.parent.name}/SKILL.md")
    return refs


def _missing_refs(
    plugin: PluginManifest,
    surface: str,
    references: tuple[str, ...],
    available: set[str],
) -> list[ExtensionAuditIssue]:
    issues = []
    for reference in references:
        if reference in available:
            continue
        issues.append(
            ExtensionAuditIssue(
                plugin_id=plugin.id,
                surface=surface,
                reference=reference,
                message=f"Plugin references unknown {surface}: {reference}",
            )
        )
    return issues


def _read_json_if_exists(path: str | Path, default: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _frontmatter_scalar(value: str) -> str:
    return json.dumps(_clean_generated_text(value))


def _clean_generated_text(value: str) -> str:
    return " ".join(str(value).split())


def _validate_id(value: str) -> None:
    if not VALID_ID.match(value):
        raise ExtensionError(f"Invalid extension id: {value}")


def _validate_risk(value: str) -> None:
    if value not in RISK_CLASSES:
        raise ExtensionError(f"Invalid risk class: {value}")
