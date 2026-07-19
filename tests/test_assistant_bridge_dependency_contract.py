from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "check_assistant_bridge_dependencies",
        ROOT / "scripts" / "check_assistant_bridge_dependencies.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the assistant-bridge dependency contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssistantBridgeDependencyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = _load_contract()
        self.versions = {
            "cryptography": "48.0.1",
            "detect-secrets": "1.5.0",
            "filelock": "3.29.7",
            "platformdirs": "4.10.1",
            "psutil": "7.2.2",
            "rfc8785": "0.1.4",
        }

    def test_accepts_first_patched_cryptography_release(self) -> None:
        with (
            patch.object(
                self.contract.metadata,
                "version",
                side_effect=self.versions.__getitem__,
            ),
            patch.object(self.contract, "import_module", return_value=object()),
        ):
            observed = self.contract.validate_optional_dependencies()

        self.assertEqual(observed["cryptography"], "48.0.1")

    def test_rejects_cryptography_release_before_security_floor(self) -> None:
        for vulnerable_release in ("47.0.0", "48.0.0"):
            with self.subTest(vulnerable_release=vulnerable_release):
                self.versions["cryptography"] = vulnerable_release
                with (
                    patch.object(
                        self.contract.metadata,
                        "version",
                        side_effect=self.versions.__getitem__,
                    ),
                    patch.object(
                        self.contract,
                        "import_module",
                        return_value=object(),
                    ),
                    self.assertRaisesRegex(SystemExit, r">=48\.0\.1,<49"),
                ):
                    self.contract.validate_optional_dependencies()

    def test_metadata_requires_patched_cryptography_floor(self) -> None:
        supported = 'cryptography<49,>=48.0.1; extra == "assistant-bridge"'
        vulnerable = 'cryptography<49,>=46; extra == "assistant-bridge"'

        self.assertTrue(
            self.contract._metadata_range_is_supported(
                "cryptography",
                supported,
            )
        )
        self.assertFalse(
            self.contract._metadata_range_is_supported(
                "cryptography",
                vulnerable,
            )
        )

    def test_release_parser_rejects_prerelease_lookalike(self) -> None:
        self.assertEqual(
            self.contract._release_triplet("48.0.1.post1+vendor"),
            (48, 0, 1),
        )
        with self.assertRaisesRegex(SystemExit, "Unsupported dependency version"):
            self.contract._release_triplet("48.0.1rc1")


if __name__ == "__main__":
    unittest.main()
