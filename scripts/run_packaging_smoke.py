from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib import error as urlerror
from urllib import request

MIN_PYTHON = (3, 10)
WEB_STARTUP_TIMEOUT_SECONDS = 45
BUILD_REQUIREMENTS = (
    "pip==25.2",
    "setuptools==80.9.0",
    "wheel==0.45.1",
)
RUNTIME_REQUIREMENTS = (
    "filelock==3.29.7",
    "platformdirs==4.10.1",
)
MINIMUM_SUPPORTED_MCP_VERSION = "1.27.2"
LOCAL_CASCADE_RUNTIME_REQUIREMENTS = (f"mcp=={MINIMUM_SUPPORTED_MCP_VERSION}",)
EXPECTED_LOCAL_CASCADE_TOOLS = (
    "delegate_plan",
    "delegate_run",
    "machine_inspect",
    "receipt_inspect",
)
LOCAL_CASCADE_MCP_CLIENT_SMOKE = """\
import anyio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def smoke() -> None:
    server = StdioServerParameters(command=sys.argv[1], args=[])
    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.list_tools()
            print(json.dumps(sorted(tool.name for tool in result.tools)))


anyio.run(smoke)
"""
REQUIRED_SDIST_ARTIFACTS = (
    ".agents/plugins/marketplace.json",
    "configs/cell-binding-request.example.json",
    "configs/local-cascade.example.json",
    "configs/local-cascade-moe.example.json",
    "configs/speculative-cell-plan.example.json",
    "docs/cell-runtime-binding.md",
    "docs/bound-cell-run.md",
    "docs/cooperative-resource-lease.md",
    "docs/local-cascade.md",
    "docs/speculative-cell-qualifier.md",
    "experiments/benchmark_runtime_binding.py",
    "experiments/benchmark_bound_cell_run.py",
    "experiments/benchmark_cooperative_resource_lease.py",
    "experiments/benchmark_local_cascade.py",
    "experiments/benchmark_speculative_cell_qualifier.py",
    "outputs/runtime-binding-contract.json",
    "outputs/bound-cell-run-contract.json",
    "outputs/cooperative-resource-lease-contract.json",
    "outputs/local-cascade-contract-benchmark.json",
    "outputs/speculative-cell-qualifier-contract.json",
    "scripts/run_packaging_smoke.py",
    "plugins/mymoe-local-cascade/.codex-plugin/plugin.json",
    "plugins/mymoe-local-cascade/.mcp.json",
    "plugins/mymoe-local-cascade/scripts/launch_mcp.py",
    "plugins/mymoe-local-cascade/skills/mymoe-local-cascade/SKILL.md",
    "plugins/mymoe-local-cascade/skills/mymoe-local-cascade/agents/openai.yaml",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    packaging_python = _select_packaging_python(root)
    with tempfile.TemporaryDirectory(prefix="mymoe-package-") as tmp:
        temporary = Path(tmp)
        venv_dir = temporary / "venv"
        dist_dir = temporary / "dist"
        runtime_dir = temporary / "Runtime Ω Space"
        dist_dir.mkdir()
        runtime_dir.mkdir()
        subprocess.run([str(packaging_python), "-m", "venv", str(venv_dir)], check=True)
        python = _venv_python(venv_dir)
        scripts_dir = _venv_scripts_dir(venv_dir)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *BUILD_REQUIREMENTS,
            ],
            cwd=runtime_dir,
            check=True,
        )
        sdist = _build_and_verify_sdist(
            python,
            root=root,
            dist_dir=dist_dir,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(dist_dir),
                str(sdist),
            ],
            cwd=runtime_dir,
            check=True,
        )
        wheels = sorted(dist_dir.glob("local_moe_orchestrator-*.whl"))
        if len(wheels) != 1:
            raise SystemExit("Packaging smoke did not produce exactly one myMoE wheel.")
        wheel = wheels[0]
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *RUNTIME_REQUIREMENTS,
            ],
            cwd=runtime_dir,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *LOCAL_CASCADE_RUNTIME_REQUIREMENTS,
            ],
            cwd=runtime_dir,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel),
            ],
            cwd=runtime_dir,
            check=True,
        )
        runtime_environment = dict(os.environ)
        runtime_environment.pop("PYTHONPATH", None)
        location_result = subprocess.run(
            [
                str(python),
                "-c",
                "from pathlib import Path; import local_moe; "
                "from local_moe import ("
                "_win32_fs, artifact_tree, bound_cell_run, bound_cell_run_contracts, "
                "bound_cell_run_envelope, cooperative_resource_lease, "
                "cooperative_resource_lease_contracts, "
                "llama_cpp_speculative_adapter, runtime_binding_cli, "
                "runtime_binding_contracts, runtime_binding_inspector, "
                "speculative_cell_cli, speculative_cell_contracts, "
                "speculative_cell_qualifier); "
                "print(Path(local_moe.__file__).resolve()); "
                "print(Path(_win32_fs.__file__).resolve()); "
                "print(Path(artifact_tree.__file__).resolve()); "
                "print(Path(bound_cell_run.__file__).resolve()); "
                "print(Path(bound_cell_run_contracts.__file__).resolve()); "
                "print(Path(bound_cell_run_envelope.__file__).resolve()); "
                "print(Path(cooperative_resource_lease.__file__).resolve()); "
                "print(Path(cooperative_resource_lease_contracts.__file__).resolve()); "
                "print(Path(llama_cpp_speculative_adapter.__file__).resolve()); "
                "print(Path(runtime_binding_cli.__file__).resolve()); "
                "print(Path(runtime_binding_contracts.__file__).resolve()); "
                "print(Path(runtime_binding_inspector.__file__).resolve()); "
                "print(Path(speculative_cell_cli.__file__).resolve()); "
                "print(Path(speculative_cell_contracts.__file__).resolve()); "
                "print(Path(speculative_cell_qualifier.__file__).resolve())",
            ],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        package_locations = tuple(
            Path(line) for line in location_result.stdout.splitlines() if line
        )
        if len(package_locations) != 15 or any(
            not location.is_relative_to(venv_dir.resolve())
            for location in package_locations
        ):
            raise SystemExit(
                "Packaging smoke imported local_moe outside the wheel environment."
            )

        mymoe = _console_script(scripts_dir, "mymoe")
        mymoe_local_cascade_mcp = _console_script(
            scripts_dir, "mymoe-local-cascade-mcp"
        )
        mymoe_paired = _console_script(scripts_dir, "mymoe-paired")
        mymoe_speculative = _console_script(scripts_dir, "mymoe-speculative")
        mymoe_web = _console_script(scripts_dir, "mymoe-web")
        _run_installed_local_cascade_mcp_smoke(
            python,
            mymoe_local_cascade_mcp,
            runtime_dir,
            environment=runtime_environment,
        )
        help_result = subprocess.run(
            [str(mymoe), "--help"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if "Local MoE orchestrator" not in help_result.stdout:
            raise SystemExit(
                "mymoe console script did not expose the expected help output."
            )
        if "advisor-init" not in help_result.stdout:
            raise SystemExit("mymoe console script did not expose advisor-init.")
        if "cell-exec" not in help_result.stdout:
            raise SystemExit("mymoe console script did not expose cell-exec.")
        if "cell-bind" not in help_result.stdout:
            raise SystemExit("mymoe console script did not expose cell-bind.")
        _run_installed_cell_binding_help_smoke(
            mymoe,
            runtime_dir,
            environment=runtime_environment,
        )
        _run_installed_cell_execution_help_smoke(
            mymoe,
            runtime_dir,
            environment=runtime_environment,
        )
        run_help_result = subprocess.run(
            [str(mymoe), "cell-exec", "run", "--help"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        for required in (
            "--binding-request",
            "--receipt-out",
            "--confirm",
            "--timeout-seconds",
            "--max-output-tokens",
        ):
            if required not in run_help_result.stdout:
                raise SystemExit(f"Installed cell-exec run help omitted {required}.")
        version_result = subprocess.run(
            [str(mymoe), "--version"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        metadata_version = subprocess.run(
            [
                str(python),
                "-c",
                "from importlib.metadata import version; "
                "print(version('local-moe-orchestrator'))",
            ],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if version_result.stdout.strip() != f"mymoe {metadata_version}":
            raise SystemExit("Installed mymoe version does not match wheel metadata.")

        default_probe_before = _installed_default_probe(
            python,
            runtime_dir,
            environment=runtime_environment,
        )

        _run_installed_advisor_smoke(
            mymoe,
            mymoe_web,
            temporary / "Advisor & Workspace",
            environment=runtime_environment,
        )

        browser_workspace = temporary / "Browser & Workspace"
        browser_init_result = subprocess.run(
            [str(mymoe), "browser-init", "--out", str(browser_workspace)],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        browser_init = json.loads(browser_init_result.stdout)
        expected_browser_files = {
            "app.browser.json",
            "mcp.playwright-browser.json",
            "moe.json",
            "context-policy.json",
            "tools.json",
            "cron.json",
        }
        actual_browser_files = {
            path.name for path in browser_workspace.iterdir() if path.is_file()
        }
        if (
            browser_init.get("status") != "created"
            or actual_browser_files != expected_browser_files
            or not isinstance(
                browser_init.get("next", {}).get("offline_canary_argv"), list
            )
        ):
            raise SystemExit(
                "Installed wheel did not materialize the packaged browser workspace."
            )

        _run_installed_desktop_smoke(
            python,
            mymoe,
            temporary / "Desktop & Workspace",
            runtime_dir,
            environment=runtime_environment,
        )

        paired_help_result = subprocess.run(
            [str(mymoe_paired), "--help"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if "claim-bound AB/BA evidence case" not in paired_help_result.stdout:
            raise SystemExit(
                "mymoe-paired console script did not expose the expected help output."
            )

        speculative_help_result = subprocess.run(
            [str(mymoe_speculative), "--help"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if "exact llama.cpp speculative-decoding cell" not in (
            speculative_help_result.stdout
        ):
            raise SystemExit(
                "mymoe-speculative console script omitted its qualification boundary."
            )
        speculative_plan = runtime_dir / "speculative-plan.json"
        subprocess.run(
            [
                str(mymoe_speculative),
                "init",
                "--out",
                str(speculative_plan),
                "--json",
            ],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        speculative_inspect = subprocess.run(
            [
                str(mymoe_speculative),
                "inspect",
                "--plan",
                str(speculative_plan),
                "--json",
            ],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if json.loads(speculative_inspect.stdout).get("status") != "valid":
            raise SystemExit("Installed mymoe-speculative template did not validate.")
        speculative_plan.unlink()

        _run_installed_web_smoke(
            mymoe_web,
            runtime_dir,
            environment=runtime_environment,
        )
        default_probe_after = _installed_default_probe(
            python,
            runtime_dir,
            environment=runtime_environment,
        )
        if default_probe_after != default_probe_before:
            raise SystemExit("Installed packaged defaults changed during smoke tests.")
        if any(runtime_dir.iterdir()):
            raise SystemExit(
                "Installed console scripts unexpectedly wrote into the empty runtime directory."
            )

        print(
            json.dumps(
                {
                    "status": "passed",
                    "build_requirements": list(BUILD_REQUIREMENTS),
                    "runtime_requirements": list(RUNTIME_REQUIREMENTS),
                    "local_cascade_runtime_requirements": list(
                        LOCAL_CASCADE_RUNTIME_REQUIREMENTS
                    ),
                    "local_cascade_tools": list(EXPECTED_LOCAL_CASCADE_TOOLS),
                    "packaging_python": str(packaging_python),
                    "python": str(python),
                    "sdist": sdist.name,
                    "wheel": wheel.name,
                    "scripts": [
                        str(mymoe),
                        str(mymoe_local_cascade_mcp),
                        str(mymoe_paired),
                        str(mymoe_speculative),
                        str(mymoe_web),
                    ],
                },
                indent=2,
            )
        )


def _build_and_verify_sdist(
    python: Path,
    *,
    root: Path,
    dist_dir: Path,
) -> Path:
    build_environment = dict(os.environ)
    build_environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            str(python),
            "-c",
            "from setuptools.build_meta import build_sdist; "
            "import sys; build_sdist(sys.argv[1])",
            str(dist_dir),
        ],
        cwd=root,
        env=build_environment,
        check=True,
    )
    sdists = sorted(dist_dir.glob("local_moe_orchestrator-*.tar.gz"))
    if len(sdists) != 1:
        raise SystemExit("Packaging smoke did not produce exactly one myMoE sdist.")
    sdist = sdists[0]
    _verify_sdist(sdist, root=root)
    return sdist


def _verify_sdist(sdist: Path, *, root: Path) -> None:
    suffix = ".tar.gz"
    if not sdist.name.endswith(suffix):
        raise SystemExit("Packaging smoke produced an unexpected sdist format.")
    archive_root = sdist.name[: -len(suffix)]
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise SystemExit("Packaging smoke produced an empty sdist.")

        by_name: dict[str, tarfile.TarInfo] = {}
        casefolded_names: set[str] = set()
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.as_posix() != name
                or not path.parts
                or path.parts[0] != archive_root
            ):
                raise SystemExit(
                    f"Packaging smoke rejected unsafe sdist member: {name!r}."
                )
            folded_name = name.casefold()
            if name in by_name or folded_name in casefolded_names:
                raise SystemExit(
                    f"Packaging smoke rejected duplicate sdist member: {name!r}."
                )
            if not (member.isfile() or member.isdir()):
                raise SystemExit(
                    f"Packaging smoke rejected non-regular sdist member: {name!r}."
                )
            by_name[name] = member
            casefolded_names.add(folded_name)

        for relative_name in REQUIRED_SDIST_ARTIFACTS:
            source = root / relative_name
            expected_name = f"{archive_root}/{relative_name}"
            member = by_name.get(expected_name)
            if not source.is_file() or member is None or not member.isfile():
                raise SystemExit(
                    f"Packaging smoke sdist omitted required artifact: {relative_name}."
                )
            archived = archive.extractfile(member)
            if archived is None or archived.read() != source.read_bytes():
                raise SystemExit(
                    "Packaging smoke sdist changed required artifact bytes: "
                    f"{relative_name}."
                )


def _run_installed_cell_binding_help_smoke(
    mymoe: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        [str(mymoe), "cell-bind", "inspect", "--help"],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    normalized = " ".join(completed.stdout.split())
    expected = (
        "--request PATH",
        "--json",
        "--out PATH",
        "does not start or download models",
        "access the network",
        "grant authorization",
    )
    if any(marker not in normalized for marker in expected):
        raise SystemExit(
            "Installed mymoe console script omitted the read-only cell-bind "
            "inspection contract."
        )


def _run_installed_local_cascade_mcp_smoke(
    python: Path,
    executable: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> None:
    installed_version = subprocess.run(
        [
            str(python),
            "-c",
            "from importlib.metadata import version; print(version('mcp'))",
        ],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if installed_version != MINIMUM_SUPPORTED_MCP_VERSION:
        raise SystemExit(
            "Packaging smoke did not install the minimum supported MCP SDK: "
            f"expected {MINIMUM_SUPPORTED_MCP_VERSION}, got {installed_version}."
        )

    completed = subprocess.run(
        [
            str(python),
            "-c",
            LOCAL_CASCADE_MCP_CLIENT_SMOKE,
            str(executable),
        ],
        cwd=runtime_dir,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Installed mymoe-local-cascade-mcp failed its stdio protocol smoke.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    try:
        tool_names = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "Installed mymoe-local-cascade-mcp client emitted non-JSON stdout."
        ) from exc
    if tool_names != list(EXPECTED_LOCAL_CASCADE_TOOLS):
        raise SystemExit(
            "Installed mymoe-local-cascade-mcp did not expose exactly the four "
            f"expected tools: {tool_names!r}."
        )


def _run_installed_cell_execution_help_smoke(
    mymoe: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        [str(mymoe), "cell-exec", "--help"],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    usage = completed.stdout.partition("\n")[0]
    choices_start = usage.find("{")
    choices_end = usage.find("}", choices_start + 1)
    if choices_start < 0 or choices_end < 0:
        raise SystemExit("Installed cell-exec help omitted its command choices.")
    choices = set(usage[choices_start + 1 : choices_end].split(","))
    forbidden_lifecycle_commands = {"start", "stop", "load", "unload"}
    if choices != {"preview", "run"} or choices & forbidden_lifecycle_commands:
        raise SystemExit(
            "Installed cell-exec unexpectedly exposed model lifecycle commands."
        )


def _installed_default_probe(
    python: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> dict[str, object]:
    probe = "; ".join(
        (
            "import hashlib,json",
            "from pathlib import Path",
            "from local_moe.app_config import load_app_config",
            "from local_moe.package_defaults import packaged_default_path",
            "names=('app.json','adaptive-cells.json','adaptive-execution-policy.json','adaptive-evaluation-contract.json','moe.json','context-policy.json')",
            "paths={name:packaged_default_path(name) for name in names}",
            "app=load_app_config(paths['app.json'])",
            "root=paths['app.json'].parent.resolve()",
            "owned=(app.default_moe_config,app.runtime.context_policy_config,app.runtime.profile_dir,app.runtime.evaluation_dir,app.extensions.plugins_dir,app.extensions.skills_dir,app.extensions.tools_config,app.extensions.mcp_config,app.extensions.cron_config,app.advisor.catalog_path,app.advisor.evaluation_contract_path)",
            "print(json.dumps({'work_dir':app.runtime.work_dir,'owned':all(Path(value).is_relative_to(root) for value in owned),'hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}}))",
        )
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    if payload.get("work_dir") != "work/runtime" or payload.get("owned") is not True:
        raise SystemExit(
            "Installed packaged app did not keep writable state outside its defaults."
        )
    return payload


def _run_installed_desktop_smoke(
    python: Path,
    mymoe: Path,
    desktop_workspace: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> None:
    help_result = subprocess.run(
        [str(mymoe), "desktop-init", "--help"],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    if not all(
        marker in help_result.stdout
        for marker in ("--target-id", "--target-pid", "--window-id")
    ):
        raise SystemExit(
            "Installed mymoe console script omitted the desktop-init interface."
        )

    probe = "\n".join(
        (
            "import sys",
            "from contextlib import ExitStack",
            "from pathlib import Path",
            "from unittest.mock import patch",
            "from local_moe import cli",
            "from local_moe.desktop_provider_contract import admitted_cua_provider_contract",
            "workspace = sys.argv[1]",
            "binary = Path(sys.executable).resolve()",
            "provider_contract = admitted_cua_provider_contract()",
            "identity = {",
            "    'pid': 4242,",
            "    'name': 'Installed Wheel Editor',",
            "    'started_at': '1753084800.000000',",
            "    'executable_sha256': 'b' * 64,",
            "}",
            "with ExitStack() as stack:",
            "    stack.enter_context(patch('local_moe.desktop_setup._provider_binary', return_value=binary))",
            "    stack.enter_context(patch('local_moe.desktop_setup._provider_version', return_value='0.10.0'))",
            "    stack.enter_context(patch('local_moe.desktop_setup._provider_contract', return_value=provider_contract))",
            "    stack.enter_context(patch('local_moe.desktop_setup._disable_provider_telemetry'))",
            "    stack.enter_context(patch('local_moe.desktop_setup._resolve_process_identity', return_value=identity))",
            "    sys.argv = [",
            "        'mymoe', 'desktop-init', '--out', workspace,",
            "        '--target-id', 'installed-editor',",
            "        '--target-pid', '4242', '--window-id', '17',",
            "    ]",
            "    cli.main()",
        )
    )
    completed = subprocess.run(
        [str(python), "-c", probe, str(desktop_workspace)],
        cwd=runtime_dir,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    expected_files = {
        "app.desktop.json",
        "mcp.cua-desktop.json",
        "moe.json",
        "context-policy.json",
        "tools.json",
        "cron.json",
    }
    actual_files = {path.name for path in desktop_workspace.iterdir() if path.is_file()}
    mcp = json.loads(
        (desktop_workspace / "mcp.cua-desktop.json").read_text(encoding="utf-8")
    )
    server = mcp.get("servers", [{}])[0]
    capability = server.get("desktop_capability", {})
    target = capability.get("target", {})
    if (
        payload.get("status") != "created"
        or actual_files != expected_files
        or server.get("command") != str(python.resolve())
        or target.get("pid") != 4242
        or target.get("window_id") != 17
        or capability.get("tool_schema_sha256", {}).get("get_window_state")
        != payload.get("provider", {}).get("observe_schema_sha256")
        or not isinstance(payload.get("next", {}).get("offline_canary_argv"), list)
    ):
        raise SystemExit(
            "Installed wheel did not materialize the packaged desktop workspace."
        )


def _run_installed_advisor_smoke(
    mymoe: Path,
    mymoe_web: Path,
    advisor_workspace: Path,
    *,
    environment: dict[str, str],
) -> None:
    hostile_runtime = advisor_workspace.parent / "Hostile Advisor Runtime"
    ambient_skill = hostile_runtime / "skills" / "ambient-secret"
    ambient_skill.mkdir(parents=True)
    (ambient_skill / "SKILL.md").write_text(
        "---\nname: ambient-secret\ndescription: must not load\n---\n",
        encoding="utf-8",
    )
    hostile_configs = hostile_runtime / "configs"
    hostile_configs.mkdir()
    (hostile_configs / "tools.json").write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "ambient-tool",
                        "description": "must not load",
                        "risk_class": "compute_only",
                        "side_effects": "none",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    initialized = subprocess.run(
        [str(mymoe), "advisor-init", "--out", str(advisor_workspace)],
        cwd=hostile_runtime,
        env=environment,
        text=True,
        capture_output=True,
    )
    if initialized.returncode != 0:
        raise SystemExit(
            "Installed advisor-init failed.\n"
            f"stdout:\n{initialized.stdout}\nstderr:\n{initialized.stderr}"
        )
    payload = json.loads(initialized.stdout)
    expected_files = {
        "adaptive-cells.json",
        "adaptive-execution-policy.json",
        "adaptive-evaluation-contract.json",
        "app.json",
        "context-policy.json",
        "moe.json",
    }
    actual_files = {path.name for path in advisor_workspace.iterdir() if path.is_file()}
    app = json.loads((advisor_workspace / "app.json").read_text(encoding="utf-8"))
    next_commands = payload.get("next")
    if not isinstance(next_commands, dict):
        raise SystemExit("Installed advisor-init omitted its next-command contract.")
    advisor_argv = next_commands.get("advisor_argv")
    execution_preview_argv = next_commands.get("cell_execution_preview_argv")
    web_argv = next_commands.get("web_argv")
    if (
        payload.get("status") != "created"
        or set(payload.get("files", [])) != expected_files
        or actual_files != expected_files
        or str(advisor_workspace) in initialized.stdout
        or app.get("default_moe_config") != "./moe.json"
        or app.get("advisor", {}).get("catalog_path") != "./adaptive-cells.json"
        or app.get("advisor", {}).get("evaluation_contract_path")
        != "./adaptive-evaluation-contract.json"
        or app.get("runtime", {}).get("work_dir") != "./work/runtime"
        or app.get("runtime", {}).get("profile_dir") != "./profiles"
        or app.get("runtime", {}).get("evaluation_dir") != "./experiments"
        or app.get("extensions", {}).get("skills_dir") != "./skills"
        or app.get("extensions", {}).get("tools_config") != "./tools.json"
        or next_commands.get("run_from") != "workspace"
        or not isinstance(advisor_argv, list)
        or advisor_argv[:2] != ["mymoe", "advisor"]
        or "--out" not in advisor_argv
        or "./advisor-receipt.json" not in advisor_argv
        or not isinstance(execution_preview_argv, list)
        or execution_preview_argv[:3] != ["mymoe", "cell-exec", "preview"]
        or "./adaptive-execution-policy.json" not in execution_preview_argv
        or not isinstance(web_argv, list)
        or web_argv != ["mymoe-web", "--app-config", "./app.json"]
    ):
        raise SystemExit(
            "Installed wheel did not materialize a self-contained Advisor workspace."
        )

    repeated = subprocess.run(
        [str(mymoe), "advisor-init", "--out", str(advisor_workspace)],
        cwd=hostile_runtime,
        env=environment,
        text=True,
        capture_output=True,
    )
    if (
        repeated.returncode != 2
        or json.loads(repeated.stderr).get("error") != "advisor_init_failed"
        or str(advisor_workspace) in repeated.stderr
    ):
        raise SystemExit("Installed advisor-init did not fail closed on repeat.")

    exact_advisor_argv = [str(mymoe), *advisor_argv[1:]]
    stdin_task = "private installed-wheel stdin task"
    stdin_result = subprocess.run(
        exact_advisor_argv,
        cwd=advisor_workspace,
        env=environment,
        input=stdin_task,
        check=True,
        text=True,
        capture_output=True,
    )
    _assert_advisor_abstention(stdin_result.stdout, stdin_task)
    _assert_task_not_persisted(advisor_workspace, stdin_task)

    receipt_path = advisor_workspace / "advisor-receipt.json"
    if (
        not receipt_path.is_file()
        or receipt_path.read_text(encoding="utf-8") != stdin_result.stdout
    ):
        raise SystemExit(
            "Installed advisor command did not publish its generated receipt."
        )
    exact_execution_preview_argv = [
        str(mymoe),
        *execution_preview_argv[1:],
    ]
    preview_result = subprocess.run(
        exact_execution_preview_argv,
        cwd=advisor_workspace,
        env=environment,
        input=stdin_task,
        text=True,
        capture_output=True,
    )
    preview = json.loads(preview_result.stdout)
    if (
        preview_result.returncode != 1
        or preview.get("status") != "admission_blocked"
        or "source_receipt_not_recommended" not in preview.get("reason_codes", [])
        or preview.get("applied") is not False
        or preview.get("authorizes_execution") is not False
        or preview.get("network_used") is not False
        or preview.get("model_invocations") != 0
    ):
        raise SystemExit(
            "Installed cell-exec did not preserve the dry-run blocked contract."
        )
    _assert_task_not_persisted(advisor_workspace, stdin_task)

    preview_task_path = advisor_workspace / "preview-task.txt"
    preview_task_path.write_text(stdin_task, encoding="utf-8")
    file_preview_argv = list(exact_execution_preview_argv)
    try:
        preview_stdin_index = file_preview_argv.index("--task-stdin")
    except ValueError as exc:
        raise SystemExit("Installed execution preview omitted --task-stdin.") from exc
    file_preview_argv[preview_stdin_index : preview_stdin_index + 1] = [
        "--task-file",
        str(preview_task_path),
    ]
    try:
        file_preview_result = subprocess.run(
            file_preview_argv,
            cwd=advisor_workspace,
            env=environment,
            text=True,
            capture_output=True,
        )
    finally:
        preview_task_path.unlink()
    file_preview = json.loads(file_preview_result.stdout)
    if (
        file_preview_result.returncode != 1
        or file_preview.get("status") != "admission_blocked"
        or "task_fingerprint_mismatch" in file_preview.get("reason_codes", [])
    ):
        raise SystemExit(
            "Installed cell-exec task-file mode did not preserve exact task bytes."
        )

    task_file = advisor_workspace / "task.txt"
    task_text = "private installed-wheel file task"
    task_file.write_text(task_text, encoding="utf-8")
    file_argv = list(exact_advisor_argv)
    try:
        task_stdin_index = file_argv.index("--task-stdin")
    except ValueError as exc:
        raise SystemExit("Installed advisor command omitted --task-stdin.") from exc
    file_argv[task_stdin_index : task_stdin_index + 1] = [
        "--task-file",
        str(task_file),
    ]
    file_receipt = advisor_workspace / "advisor-receipt-file.json"
    try:
        output_index = file_argv.index("--out")
    except ValueError as exc:
        raise SystemExit("Installed advisor command omitted --out.") from exc
    file_argv[output_index + 1] = str(file_receipt)
    try:
        file_result = subprocess.run(
            file_argv,
            cwd=advisor_workspace,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
    finally:
        task_file.unlink()
        file_receipt.unlink(missing_ok=True)
    _assert_advisor_abstention(file_result.stdout, task_text)
    _assert_task_not_persisted(advisor_workspace, task_text)

    _run_installed_advisor_web_smoke(
        mymoe_web,
        advisor_workspace,
        web_argv,
        environment=environment,
    )
    if (hostile_runtime / "work").exists():
        raise SystemExit("Installed Advisor wrote runtime state into its hostile CWD.")


def _assert_advisor_abstention(rendered: str, task_text: str) -> None:
    payload = json.loads(rendered)
    if (
        payload.get("advice", {}).get("status") != "abstained"
        or payload.get("display_state") == "recommended_now"
        or payload.get("advice", {}).get("selected_cell_id") is not None
        or task_text in rendered
    ):
        raise SystemExit("Installed zero-claim Advisor did not abstain safely.")


def _assert_task_not_persisted(workspace: Path, task_text: str) -> None:
    needle = task_text.encode("utf-8")
    for path in workspace.rglob("*"):
        if path.is_file() and needle in path.read_bytes():
            raise SystemExit(
                f"Installed Advisor persisted raw task text in {path.relative_to(workspace)}."
            )


def _run_installed_advisor_web_smoke(
    executable: Path,
    advisor_workspace: Path,
    web_argv: list[str],
    *,
    environment: dict[str, str],
) -> None:
    port = _available_loopback_port()
    process = subprocess.Popen(
        [
            str(executable),
            *web_argv[1:],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=advisor_workspace,
        env={**environment, "PYTHONUNBUFFERED": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_web_ready(process, port)
        with request.urlopen(
            f"http://127.0.0.1:{port}/api/advisor/config",
            timeout=5,
        ) as response:
            public = json.loads(response.read().decode("utf-8"))
        with request.urlopen(
            f"http://127.0.0.1:{port}/api/extensions",
            timeout=5,
        ) as response:
            extensions = response.read().decode("utf-8")
        task_text = "private installed-wheel browser task"
        advisor_request = request.Request(
            f"http://127.0.0.1:{port}/api/advisor",
            data=json.dumps(
                {
                    "task": task_text,
                    "profile": "balanced",
                    "context_tokens": 4096,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(advisor_request, timeout=5) as response:
            rendered = response.read().decode("utf-8")
            presentation = json.loads(rendered)
        if (
            public.get("enabled") is not True
            or "ambient-secret" in extensions
            or "ambient-tool" in extensions
            or "catalog_path" in json.dumps(public)
            or "evaluation_contract_path" in json.dumps(public)
            or presentation.get("display_state") == "recommended_now"
            or presentation.get("receipt", {}).get("advice", {}).get("status")
            != "abstained"
            or task_text in rendered
        ):
            raise SystemExit("Installed Advisor web endpoint failed its safe smoke.")
        _assert_task_not_persisted(advisor_workspace, task_text)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def _run_installed_web_smoke(
    executable: Path,
    runtime_dir: Path,
    *,
    environment: dict[str, str],
) -> None:
    if any(runtime_dir.iterdir()):
        raise SystemExit("Packaging smoke runtime directory must start empty.")
    port = _available_loopback_port()
    process = subprocess.Popen(
        [
            str(executable),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=runtime_dir,
        env={**environment, "PYTHONUNBUFFERED": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_web_ready(process, port)
        with request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            body = response.read().decode("utf-8")
            if response.status != 200 or "<title>myMoE</title>" not in body:
                raise SystemExit("installed web root returned an unexpected response")
        with request.urlopen(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
            model_ids = {item.get("id") for item in payload.get("data", [])}
            if (
                response.status != 200
                or payload.get("object") != "list"
                or "mymoe" not in model_ids
                or "mymoe/local" not in model_ids
            ):
                raise SystemExit(
                    "installed gateway returned an unexpected model catalog"
                )
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def _wait_for_web_ready(process: subprocess.Popen[str], port: int) -> None:
    # Hosted macOS runners can remain CPU-constrained after the full test suite.
    # Keep the product request timeout strict, but give the isolated interpreter
    # enough time to import the installed wheel before declaring startup broken.
    deadline = time.monotonic() + WEB_STARTUP_TIMEOUT_SECONDS
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise SystemExit(
                "Installed mymoe-web exited before startup.\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/", timeout=1):
                return
        except (OSError, urlerror.URLError) as exc:
            last_error = str(exc)
            time.sleep(0.1)
    raise SystemExit(f"Installed mymoe-web did not become ready: {last_error}")


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _select_packaging_python(root: Path) -> Path:
    candidates = []
    if os.environ.get("MYMOE_PACKAGING_PYTHON"):
        candidates.append(Path(os.environ["MYMOE_PACKAGING_PYTHON"]))
    candidates.append(Path(sys.executable))
    if os.name == "nt":
        candidates.append(root / ".venv" / "Scripts" / "python.exe")
    else:
        candidates.append(root / ".venv" / "bin" / "python")
        candidates.extend(
            Path(name) for name in ("python3.12", "python3.11", "python3.10")
        )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _python_is_compatible(candidate):
            return candidate
    raise SystemExit(
        "Packaging smoke requires Python >= 3.10. Set MYMOE_PACKAGING_PYTHON to a compatible interpreter."
    )


def _python_is_compatible(python: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_scripts_dir(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts"
    return venv_dir / "bin"


def _console_script(scripts_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return scripts_dir / f"{name}.exe"
    return scripts_dir / name


if __name__ == "__main__":
    main()
