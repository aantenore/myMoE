from __future__ import annotations

from importlib import resources
import json
import os
from pathlib import Path
import stat
from typing import Any


_OUTPUT_FILES = (
    "app.browser.json",
    "mcp.playwright-browser.json",
    "moe.json",
    "context-policy.json",
    "tools.json",
    "cron.json",
)


def materialize_browser_workspace(output_dir: str | Path) -> dict[str, Any]:
    """Create a self-contained browser workspace from packaged resources."""

    requested_root = Path(output_dir).expanduser().absolute()
    if os.path.lexists(requested_root) and requested_root.is_symlink():
        raise ValueError("Browser workspace root must not be a symbolic link.")
    root = requested_root.resolve()
    app = _read_packaged_json("templates/browser/app.browser.example.json")
    mcp = _read_packaged_json("templates/browser/mcp.playwright-browser.example.json")
    moe = _read_packaged_json("defaults/moe.json")
    context_policy = _read_packaged_json("defaults/context-policy.json")

    servers = mcp.get("servers")
    if (
        not isinstance(servers, list)
        or len(servers) != 1
        or not isinstance(servers[0], dict)
    ):
        raise ValueError("Packaged browser MCP template must contain exactly one server.")
    server_name = str(servers[0].get("name", "")).strip()
    capability = servers[0].get("browser_capability")
    if not server_name or not isinstance(capability, dict):
        raise ValueError("Packaged browser MCP template is missing provider metadata.")
    package = str(capability.get("package", "")).strip()
    version = str(capability.get("version", "")).strip()
    if not package or not version:
        raise ValueError("Packaged browser MCP template is missing its package pin.")

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
            "mcp_config": str(root / "mcp.playwright-browser.json"),
            "cron_config": str(root / "cron.json"),
        }
    )

    payloads = {
        "app.browser.json": app,
        "mcp.playwright-browser.json": mcp,
        "moe.json": moe,
        "context-policy.json": context_policy,
        "tools.json": {"tools": []},
        "cron.json": {"jobs": []},
    }
    rendered_payloads = {
        name: json.dumps(payloads[name], indent=2, ensure_ascii=False) + "\n"
        for name in _OUTPUT_FILES
    }
    existing = [name for name in _OUTPUT_FILES if os.path.lexists(root / name)]
    if existing:
        raise FileExistsError(
            f"Browser workspace files already exist and were not changed: {existing}"
        )
    root.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(root)
    for directory in (root / "plugins", root / "skills"):
        directory.mkdir(exist_ok=True)
        _require_plain_directory(directory)

    created: list[Path] = []
    try:
        for name in _OUTPUT_FILES:
            destination = root / name
            _write_exclusive_text(destination, rendered_payloads[name])
            created.append(destination)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                continue
        raise

    prefetch_argv = [
        "mymoe",
        "browser-prefetch",
        "--mcp-config",
        str(root / "mcp.playwright-browser.json"),
        "--server",
        server_name,
    ]
    canary_argv = [
        "mymoe",
        "--app-config",
        str(root / "app.browser.json"),
        "--browser-canary",
        server_name,
        "--browser-canary-confirm",
    ]
    return {
        "schema_version": "1.0",
        "status": "created",
        "workspace": str(root),
        "app_config": str(root / "app.browser.json"),
        "mcp_config": str(root / "mcp.playwright-browser.json"),
        "files": [str(root / name) for name in _OUTPUT_FILES],
        "next": {
            "prefetch_provider_while_online_argv": prefetch_argv,
            "offline_canary_argv": canary_argv,
        },
        "provider": {"package": package, "version": version},
    }


def _require_plain_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("Browser workspace directory is unavailable.") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Browser workspace paths must be plain directories.")


def _write_exclusive_text(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise FileExistsError(
            f"Browser workspace file already exists and was not changed: {path.name}"
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
        raise ValueError(f"Packaged browser resource must contain an object: {relative_path}")
    return raw
