from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_desktop_semantic import run_benchmark


class DesktopSemanticBenchmarkTests(unittest.TestCase):
    def test_canonical_result_is_reproducible_and_excludes_wall_clock_data(self) -> None:
        first = run_benchmark(iterations=5)
        second = run_benchmark(iterations=5)

        self.assertEqual(first, second)
        self.assertTrue(first["release_ready"])
        self.assertFalse(
            any(name.endswith("_ms") for name in first["measurements"])
        )

    def test_checked_in_artifact_is_exact_canonical_40_iteration_output(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = (
            json.dumps(
                run_benchmark(iterations=40),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

        self.assertEqual(
            (root / "outputs" / "desktop-semantic-benchmark.json").read_text(
                encoding="utf-8"
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
