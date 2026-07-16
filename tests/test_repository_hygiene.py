from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOKENS = (
    "".join(("re", "ply")).encode(),
    "".join(("users", ".invalid")).encode(),
)


class RepositoryHygieneTests(unittest.TestCase):
    def test_tracked_tree_excludes_forbidden_identity_tokens(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        paths = [
            Path(raw.decode("utf-8"))
            for raw in result.stdout.split(b"\0")
            if raw
        ]
        violations: list[str] = []
        for relative_path in paths:
            path = ROOT / relative_path
            lowered_path = relative_path.as_posix().lower().encode()
            lowered_content = path.read_bytes().lower()
            if any(
                token in lowered_path or token in lowered_content
                for token in FORBIDDEN_TOKENS
            ):
                violations.append(relative_path.as_posix())

        self.assertEqual(violations, [], f"Forbidden identity tokens: {violations}")


if __name__ == "__main__":
    unittest.main()
