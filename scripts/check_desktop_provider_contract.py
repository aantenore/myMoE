from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


EXPECTED_PROVIDER_TOOL_COUNT = 49


def main() -> None:
    try:
        from cua_driver import get_binary_path  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install the desktop extra before checking the provider contract."
        ) from exc

    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "mcp.cua-desktop.example.json").read_text(
            encoding="utf-8"
        )
    )
    capability = config["servers"][0]["desktop_capability"]
    expected_version = capability["version"]
    expected_schema = capability["tool_schema_sha256"]["get_window_state"]
    binary = Path(get_binary_path()).resolve(strict=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "APPDATA",
            "HOME",
            "LOCALAPPDATA",
            "PATH",
            "PATHEXT",
            "SystemRoot",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USERPROFILE",
        }
    }
    environment.update(
        {
            "CUA_DRIVER_RS_TELEMETRY_ENABLED": "false",
            "CUA_DRIVER_RS_UPDATE_CHECK": "false",
        }
    )
    version = _run([str(binary), "--version"], environment)
    if version.stdout.strip() != f"cua-driver {expected_version}":
        raise SystemExit("Pinned desktop provider version did not match.")
    docs = _run(
        [str(binary), "dump-docs", "--type", "mcp"],
        environment,
    )
    payload = json.loads(docs.stdout)
    tools = payload.get("tools", [])
    if not isinstance(tools, list) or len(tools) != EXPECTED_PROVIDER_TOOL_COUNT:
        raise SystemExit(
            "Pinned desktop provider tool catalog size did not match the admitted contract."
        )
    tool_names = [
        item.get("name")
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    if len(tool_names) != len(tools) or len(set(tool_names)) != len(tool_names):
        raise SystemExit("Pinned desktop provider tool catalog is malformed.")
    tool = next(
        (
            item
            for item in tools
            if isinstance(item, dict) and item.get("name") == "get_window_state"
        ),
        None,
    )
    if tool is None:
        raise SystemExit("Pinned desktop provider omitted get_window_state.")
    actual_schema = _sha256_json(tool.get("input_schema"))
    if actual_schema != expected_schema:
        raise SystemExit("Pinned desktop provider schema digest did not match.")
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "passed",
                "provider": "cua_driver",
                "version": expected_version,
                "observed_provider_executable_sha256": _sha256_file(binary),
                "get_window_state_schema_sha256": actual_schema,
                "provider_tool_count": EXPECTED_PROVIDER_TOOL_COUNT,
                "model_visible_tool_count": 1,
                "telemetry_enabled_for_check": False,
                "update_check_enabled_for_check": False,
            },
            indent=2,
        )
    )


def _run(
    argv: list[str],
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            env=environment,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SystemExit("Desktop provider contract command failed safely.") from exc


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
