#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


ENTRYPOINT = "mymoe-local-cascade-mcp"
PROJECT_ROOT_ENV = "MYMOE_PROJECT_ROOT"


def resolve_launch() -> tuple[str, list[str]]:
    raw_root = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if raw_root:
        root = Path(raw_root)
        uv = shutil.which("uv")
        if (
            not root.is_absolute()
            or not (root / "pyproject.toml").is_file()
            or uv is None
        ):
            raise RuntimeError(
                "MYMOE_PROJECT_ROOT must be an absolute checkout with locked "
                "dependencies and uv available."
            )
        return (
            "project_root_offline",
            [
                uv,
                "run",
                "--offline",
                "--locked",
                "--extra",
                "local-cascade",
                "--project",
                str(root),
                ENTRYPOINT,
            ],
        )

    console = shutil.which(ENTRYPOINT)
    if console:
        return "installed_console", [console]

    try:
        module_spec = importlib.util.find_spec("local_moe.local_cascade_mcp")
    except (ImportError, ModuleNotFoundError):
        module_spec = None
    if module_spec is not None:
        return "installed_module", [sys.executable, "-m", "local_moe.local_cascade_mcp"]

    raise RuntimeError(
        "Install myMoE Local Cascade or set MYMOE_PROJECT_ROOT to an "
        "absolute checkout with locked dependencies and uv available."
    )


def main() -> int:
    if sys.argv[1:] not in ([], ["--dry-run"]):
        print("Unsupported launcher arguments.", file=sys.stderr)
        return 2
    try:
        mode, command = resolve_launch()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if sys.argv[1:] == ["--dry-run"]:
        print(json.dumps({"status": "ready", "mode": mode}, separators=(",", ":")))
        return 0
    os.execv(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
