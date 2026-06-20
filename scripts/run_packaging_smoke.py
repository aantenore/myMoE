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
        venv_dir = Path(tmp) / "venv"
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
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(root),
            ],
            cwd=root,
            check=True,
        )

        mymoe = _console_script(scripts_dir, "mymoe")
        mymoe_web = _console_script(scripts_dir, "mymoe-web")
        prompt = "Packaging smoke: summarize local MoE in one sentence."
        completed = subprocess.run(
            [
                str(mymoe),
                "--config",
                "tests/fixtures/moe.synthetic.json",
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
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        if "myMoE local web UI" not in help_result.stdout:
            raise SystemExit("mymoe-web console script did not expose the expected help output.")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "packaging_python": str(packaging_python),
                    "python": str(python),
                    "scripts": [str(mymoe), str(mymoe_web)],
                },
                indent=2,
            )
        )


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
