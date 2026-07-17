from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.path_security import PathBoundaryError, read_text_file, resolve_existing_file


class PathSecurityTests(unittest.TestCase):
    def test_resolves_bare_and_prefixed_paths_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            target = root / "data.json"
            target.write_text("{}", encoding="utf-8")

            bare = resolve_existing_file("data.json", allowed_roots=(root,))
            prefixed = resolve_existing_file(target, allowed_roots=(root,))

        self.assertEqual(bare, target.resolve())
        self.assertEqual(prefixed, target.resolve())

    def test_rejects_traversal_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "allowed"
            root.mkdir()
            outside = workspace / "outside.json"
            outside.write_text("{}", encoding="utf-8")

            with self.assertRaises(PathBoundaryError):
                resolve_existing_file("../outside.json", allowed_roots=(root,))

            link = root / "linked.json"
            try:
                link.symlink_to(outside)
            except OSError:
                return
            with self.assertRaises(PathBoundaryError):
                resolve_existing_file(link, allowed_roots=(root,))

    def test_enforces_file_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "large.txt"
            target.write_text("abcd", encoding="utf-8")

            with self.assertRaisesRegex(PathBoundaryError, "byte limit"):
                read_text_file(target, allowed_roots=(root,), max_bytes=3)


if __name__ == "__main__":
    unittest.main()
