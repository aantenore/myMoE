from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib import error as urlerror
from urllib import request

MIN_PYTHON = (3, 10)
BUILD_REQUIREMENTS = (
    "pip==25.2",
    "setuptools==80.9.0",
    "wheel==0.45.1",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    packaging_python = _select_packaging_python(root)
    with tempfile.TemporaryDirectory(prefix="mymoe-package-") as tmp:
        temporary = Path(tmp)
        venv_dir = temporary / "venv"
        dist_dir = temporary / "dist"
        runtime_dir = temporary / "runtime"
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
                str(root),
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
                "from local_moe import _win32_fs; "
                "print(Path(local_moe.__file__).resolve()); "
                "print(Path(_win32_fs.__file__).resolve())",
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
        if len(package_locations) != 2 or any(
            not location.is_relative_to(venv_dir.resolve())
            for location in package_locations
        ):
            raise SystemExit(
                "Packaging smoke imported local_moe outside the wheel environment."
            )

        mymoe = _console_script(scripts_dir, "mymoe")
        mymoe_paired = _console_script(scripts_dir, "mymoe-paired")
        mymoe_web = _console_script(scripts_dir, "mymoe-web")
        help_result = subprocess.run(
            [str(mymoe), "--help"],
            cwd=runtime_dir,
            env=runtime_environment,
            check=True,
            text=True,
            capture_output=True,
        )
        if "Local MoE orchestrator" not in help_result.stdout:
            raise SystemExit("mymoe console script did not expose the expected help output.")

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

        _run_installed_web_smoke(
            mymoe_web,
            runtime_dir,
            environment=runtime_environment,
        )
        if any(runtime_dir.iterdir()):
            raise SystemExit(
                "Installed console scripts unexpectedly wrote into the empty runtime directory."
            )

        print(
            json.dumps(
                {
                    "status": "passed",
                    "build_requirements": list(BUILD_REQUIREMENTS),
                    "packaging_python": str(packaging_python),
                    "python": str(python),
                    "wheel": wheel.name,
                    "scripts": [str(mymoe), str(mymoe_paired), str(mymoe_web)],
                },
                indent=2,
            )
        )


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
    deadline = time.monotonic() + 15
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
        candidates.extend(Path(name) for name in ("python3.12", "python3.11", "python3.10"))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _python_is_compatible(candidate):
            return candidate
    raise SystemExit("Packaging smoke requires Python >= 3.10. Set MYMOE_PACKAGING_PYTHON to a compatible interpreter.")


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
