from __future__ import annotations

from importlib import resources
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

from .desktop_capability import (
    CUA_DRIVER_VERSION,
    DesktopCapabilityConfig,
    _resolve_executable,
    _resolve_process_identity,
    _sha256_file,
)
from .extensions import load_mcp_servers
from .tool_runner import ToolExecutionError


_OUTPUT_FILES = (
    "app.desktop.json",
    "mcp.cua-desktop.json",
    "moe.json",
    "context-policy.json",
    "tools.json",
    "cron.json",
)


def materialize_desktop_workspace(
    output_dir: str | Path,
    *,
    target_id: str,
    target_pid: int,
    window_id: int,
    provider_binary: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a desktop workspace to one current process instance and window."""

    if re.fullmatch(r"[a-z][a-z0-9-]{1,63}", target_id) is None:
        raise ValueError("Desktop target id is invalid.")
    if type(target_pid) is not int or not 1 <= target_pid <= 2_147_483_647:
        raise ValueError("Desktop target PID is invalid.")
    if type(window_id) is not int or not 0 <= window_id <= 18_446_744_073_709_551_615:
        raise ValueError("Desktop window id is invalid.")
    requested_root = Path(output_dir).expanduser().absolute()
    if os.path.lexists(requested_root) and requested_root.is_symlink():
        raise ValueError("Desktop workspace root must not be a symbolic link.")
    root = requested_root.resolve()
    binary = _provider_binary(provider_binary)
    version = _provider_version(binary)
    if version != CUA_DRIVER_VERSION:
        raise ToolExecutionError(
            f"Desktop init requires cua-driver {CUA_DRIVER_VERSION}; found {version}."
        )
    identity = _resolve_process_identity(target_pid)

    app = _read_packaged_json("templates/desktop/app.desktop.example.json")
    mcp = _read_packaged_json("templates/desktop/mcp.cua-desktop.example.json")
    moe = _read_packaged_json("defaults/moe.json")
    context_policy = _read_packaged_json("defaults/context-policy.json")
    servers = mcp.get("servers")
    if (
        not isinstance(servers, list)
        or len(servers) != 1
        or not isinstance(servers[0], dict)
    ):
        raise ValueError("Packaged desktop MCP template must contain exactly one server.")
    server = servers[0]
    server_name = str(server.get("name", "")).strip()
    capability = server.get("desktop_capability")
    if not server_name or not isinstance(capability, dict):
        raise ValueError("Packaged desktop MCP template is missing provider metadata.")
    server["command"] = str(binary)
    capability["provider_executable_sha256"] = _sha256_file(binary)
    capability["target"] = {
        "id": target_id,
        "pid": target_pid,
        "window_id": window_id,
        "process_name": identity["name"],
        "process_started_at": identity["started_at"],
        "process_executable_sha256": identity["executable_sha256"],
    }

    app["default_moe_config"] = str(root / "moe.json")
    app["runtime"].update(
        {
            "work_dir": str(root / "work" / "runtime"),
            "context_policy_config": str(root / "context-policy.json"),
            "profile_dir": str(root),
            "evaluation_dir": str(root / "experiments"),
        }
    )
    app["extensions"].update(
        {
            "plugins_dir": str(root / "plugins"),
            "skills_dir": str(root / "skills"),
            "tools_config": str(root / "tools.json"),
            "mcp_config": str(root / "mcp.cua-desktop.json"),
            "cron_config": str(root / "cron.json"),
        }
    )

    payloads = {
        "app.desktop.json": app,
        "mcp.cua-desktop.json": mcp,
        "moe.json": moe,
        "context-policy.json": context_policy,
        "tools.json": {"tools": []},
        "cron.json": {"jobs": []},
    }
    rendered = {
        name: json.dumps(payloads[name], indent=2, ensure_ascii=False) + "\n"
        for name in _OUTPUT_FILES
    }
    existing = [name for name in _OUTPUT_FILES if os.path.lexists(root / name)]
    if existing:
        raise FileExistsError(
            f"Desktop workspace files already exist and were not changed: {existing}"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_plain_directory(root)
    if os.name == "posix":
        root.chmod(0o700)
    for directory in (root / "plugins", root / "skills"):
        directory.mkdir(exist_ok=True, mode=0o700)
        _require_plain_directory(directory)
        if os.name == "posix":
            directory.chmod(0o700)
    _disable_provider_telemetry(binary)
    created: list[Path] = []
    try:
        for name in _OUTPUT_FILES:
            destination = root / name
            _write_exclusive_text(destination, rendered[name])
            created.append(destination)
        configured = load_mcp_servers(root / "mcp.cua-desktop.json")
        if len(configured) != 1:
            raise ValueError("Materialized desktop workspace has an invalid server count.")
        DesktopCapabilityConfig.from_server(configured[0])
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise

    canary_argv = [
        "mymoe",
        "--app-config",
        str(root / "app.desktop.json"),
        "--desktop-canary",
        server_name,
        "--desktop-canary-confirm",
    ]
    agent_argv = [
        "mymoe",
        "--app-config",
        str(root / "app.desktop.json"),
        "--agent-prompt",
        "Describe the configured desktop window and identify visible problems.",
        "--agent-desktop-server",
        server_name,
        "--agent-interactive-approvals",
        "--json",
    ]
    return {
        "schema_version": "1.0",
        "status": "created",
        "workspace": str(root),
        "app_config": str(root / "app.desktop.json"),
        "mcp_config": str(root / "mcp.cua-desktop.json"),
        "files": [str(root / name) for name in _OUTPUT_FILES],
        "target": {
            "id": target_id,
            "process_name": identity["name"],
            "process_executable_sha256": identity["executable_sha256"],
        },
        "provider": {
            "name": "cua_driver",
            "version": version,
            "executable_sha256": _sha256_file(binary),
            "telemetry_enabled": False,
        },
        "next": {
            "offline_canary_argv": canary_argv,
            "offline_agent_argv": agent_argv,
        },
    }


def _provider_binary(requested: str | Path | None) -> Path:
    if requested is None:
        try:
            from cua_driver import get_binary_path  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ToolExecutionError(
                "The desktop extra is required; install local-moe-orchestrator[desktop]."
            ) from exc
        requested = str(get_binary_path())
    return _resolve_executable(str(requested))


def _provider_version(binary: Path) -> str:
    completed = _run_provider(binary, ["--version"])
    if completed.returncode != 0:
        raise ToolExecutionError("Desktop provider version check failed.")
    prefix = "cua-driver "
    line = next(
        (item.strip() for item in completed.stdout.splitlines() if item.startswith(prefix)),
        "",
    )
    return line[len(prefix) :].strip()


def _disable_provider_telemetry(binary: Path) -> None:
    disabled = _run_provider(binary, ["telemetry", "disable"])
    if disabled.returncode != 0:
        raise ToolExecutionError("Desktop provider telemetry could not be disabled.")
    reset = _run_provider(binary, ["telemetry", "reset-id"])
    if reset.returncode != 0:
        raise ToolExecutionError("Desktop provider telemetry identity could not be erased.")
    status = _run_provider(
        binary,
        ["telemetry", "status", "--json"],
        telemetry_override=False,
    )
    try:
        payload = json.loads(status.stdout)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("Desktop provider telemetry status is invalid.") from exc
    if (
        status.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("enabled") is not False
        or payload.get("source") != "persisted"
        or payload.get("installation_id_present") is not False
    ):
        raise ToolExecutionError("Desktop provider telemetry did not fail closed.")


def _run_provider(
    binary: Path,
    arguments: list[str],
    *,
    telemetry_override: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "APPDATA",
            "HOME",
            "LOCALAPPDATA",
            "PATH",
            "SystemRoot",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USERPROFILE",
        }
    }
    environment["CUA_DRIVER_RS_UPDATE_CHECK"] = "false"
    if telemetry_override:
        environment["CUA_DRIVER_RS_TELEMETRY_ENABLED"] = "false"
    try:
        return subprocess.run(
            [str(binary), *arguments],
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise ToolExecutionError("Desktop provider setup failed safely.") from exc


def _require_plain_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("Desktop workspace directory is unavailable.") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Desktop workspace paths must be plain directories.")


def _write_exclusive_text(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise FileExistsError(
            f"Desktop workspace file already exists and was not changed: {path.name}"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_packaged_json(relative_path: str) -> dict[str, Any]:
    node = resources.files("local_moe")
    for part in relative_path.split("/"):
        node = node.joinpath(part)
    raw = json.loads(node.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Packaged desktop resource must contain an object: {relative_path}")
    return raw
