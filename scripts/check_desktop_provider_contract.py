from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from local_moe.desktop_provider_contract import (
    CUA_DRIVER_DISABLE_A11Y_ADVERTISE_ENV,
    CUA_DRIVER_DISABLE_A11Y_ADVERTISE_VALUE,
    validate_cua_provider_document,
)


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
            CUA_DRIVER_DISABLE_A11Y_ADVERTISE_ENV: (
                CUA_DRIVER_DISABLE_A11Y_ADVERTISE_VALUE
            ),
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
    if not isinstance(payload, dict):
        raise SystemExit("Pinned desktop provider documentation is malformed.")
    try:
        contract = validate_cua_provider_document(
            payload,
            version=expected_version,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Pinned desktop provider platform contract did not match: {exc}"
        ) from exc
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "passed",
                "provider": "cua_driver",
                "version": expected_version,
                "platform_system": contract.platform_system,
                "observed_provider_executable_sha256": _sha256_file(binary),
                "provider_catalog_names_sha256": contract.catalog_names_sha256,
                "get_window_state_schema_sha256": contract.observe_schema_sha256,
                "provider_tool_count": contract.tool_count,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
