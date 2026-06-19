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
    servers = []
    for item in raw.get("servers", []):
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
        servers.append(
            McpServerDefinition(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                command=str(item["command"]),
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
        )
    return servers


def load_cron_jobs(path: str | Path) -> list[CronJobDefinition]:
    raw = _read_json_if_exists(path, {"jobs": []})
    jobs = []
    for item in raw.get("jobs", []):
        _validate_id(str(item["id"]))
        risk_class = str(item.get("risk_class", "compute_only"))
        _validate_risk(risk_class)
        jobs.append(
            CronJobDefinition(
                id=str(item["id"]),
                description=str(item.get("description", "")),
                enabled=bool(item.get("enabled", False)),
                schedule=dict(item.get("schedule", {})),
                command=tuple(str(arg) for arg in item.get("command", [])),
                risk_class=risk_class,
            )
        )
    return jobs


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
