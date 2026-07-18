from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MIN_PYTHON = (3, 10)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    packaging_python = _select_packaging_python(root)
    with tempfile.TemporaryDirectory(prefix="mymoe-package-") as tmp:
        temporary = Path(tmp)
        venv_dir = temporary / "venv"
        dist_dir = temporary / "dist"
        dist_dir.mkdir()
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
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            cwd=root,
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
            cwd=temporary,
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
            cwd=temporary,
            check=True,
        )
        location_result = subprocess.run(
            [
                str(python),
                "-c",
                "from pathlib import Path; import local_moe; "
                "print(Path(local_moe.__file__).resolve())",
            ],
            cwd=temporary,
            check=True,
            text=True,
            capture_output=True,
        )
        package_location = Path(location_result.stdout.strip())
        if not package_location.is_relative_to(venv_dir.resolve()):
            raise SystemExit("Packaging smoke imported local_moe outside the wheel environment.")

        mymoe = _console_script(scripts_dir, "mymoe")
        mymoe_web = _console_script(scripts_dir, "mymoe-web")
        prompt = "Packaging smoke: summarize local MoE in one sentence."
        completed = subprocess.run(
            [
                str(mymoe),
                "--config",
                str(root / "tests" / "fixtures" / "moe.synthetic.json"),
                "--prompt",
                prompt,
                "--json",
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout)
        if "content" not in payload or "synthetic-" not in str(payload["content"]):
            raise SystemExit("mymoe console script did not return the expected synthetic payload.")

        help_result = subprocess.run(
            [str(mymoe_web), "--help"],
            cwd=temporary,
            check=True,
            text=True,
            capture_output=True,
        )
        if "myMoE local web UI" not in help_result.stdout:
            raise SystemExit("mymoe-web console script did not expose the expected help output.")

        web_result = subprocess.run(
            [
                str(python),
                "-c",
                _WEB_ROOT_SMOKE,
                str(root / "tests" / "fixtures" / "moe.synthetic.json"),
                str(root / "configs" / "app.json"),
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        if web_result.stdout.strip() != "web-root-ok":
            raise SystemExit("Installed mymoe-web did not serve its packaged UI asset.")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "packaging_python": str(packaging_python),
                    "python": str(python),
                    "wheel": wheel.name,
                    "scripts": [str(mymoe), str(mymoe_web)],
                },
                indent=2,
            )
        )


_WEB_ROOT_SMOKE = """
import sys
from threading import Thread
from urllib.request import urlopen

from local_moe.web import build_server

server = build_server(sys.argv[1], "127.0.0.1", 0, app_config_path=sys.argv[2])
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
        body = response.read().decode("utf-8")
        if response.status != 200 or "<title>myMoE</title>" not in body:
            raise SystemExit("installed web root returned an unexpected response")
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
print("web-root-ok")
"""


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
