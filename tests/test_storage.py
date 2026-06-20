from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from local_moe.storage import build_storage_report


@dataclass(frozen=True)
class _Runtime:
    model_cache_dir: str
    work_dir: str


@dataclass(frozen=True)
class _AppConfig:
    runtime: _Runtime


class StorageTests(unittest.TestCase):
    def test_reports_runtime_storage_without_creating_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "missing-cache"
            work = root / "work"
            app_config = _AppConfig(runtime=_Runtime(model_cache_dir=str(cache), work_dir=str(work)))

            report = build_storage_report(app_config, min_free_gib=0)
            cache_exists = cache.exists()
            work_exists = work.exists()

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["path_count"], 2)
        self.assertEqual({item["label"] for item in report["paths"]}, {"model_cache_dir", "work_dir"})
        self.assertTrue(all(item["free_gib"] is not None for item in report["paths"]))
        self.assertFalse(cache_exists)
        self.assertFalse(work_exists)

    def test_warns_when_free_space_is_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _AppConfig(runtime=_Runtime(model_cache_dir=str(root), work_dir=str(root)))

            report = build_storage_report(app_config, min_free_gib=1_000_000_000)

        self.assertEqual(report["status"], "attention")
        self.assertEqual(report["summary"]["attention"], 2)
        self.assertTrue(report["recommendations"])


if __name__ == "__main__":
    unittest.main()
