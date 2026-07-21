from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

import local_moe.desktop_provider_contract as provider_contract
from local_moe.desktop_provider_contract import (
    CuaProviderContract,
    admitted_cua_provider_contract,
    validate_cua_provider_document,
)


class DesktopProviderContractTests(unittest.TestCase):
    def test_admits_exact_platform_specific_cua_surfaces(self) -> None:
        expected = {
            "Darwin": (
                49,
                "a39bbb495c25d8c24f388e06ecd10f4aec96b8486b832750ef65daac89f4bd69",
                "a1685e0da284cf8445e9d2e11bdbd7249e20b72ee9109800aba798ee7ff322c3",
            ),
            "Linux": (
                53,
                "c8d63c8a14b49781d64c6f739b4dc484789f2738ed559fd3d15fcbed15271a85",
                "7c039adde1f1f403e9350e9e0c67005f0403b0898b677a2543702b838e8a167a",
            ),
            "Windows": (
                50,
                "ee77ceaf809bc8eef3f2a85ebdc19f66c798df73252e388a95203bc0d9421e81",
                "442ffd78dbf3212af5101d2f82ae988cff582d1cd07f3cb65a0f807e1169998d",
            ),
        }
        for system, values in expected.items():
            with self.subTest(system=system):
                contract = admitted_cua_provider_contract(platform_system=system)
                self.assertEqual(
                    (
                        contract.tool_count,
                        contract.catalog_names_sha256,
                        contract.observe_schema_sha256,
                    ),
                    values,
                )

    def test_unknown_platform_or_version_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            admitted_cua_provider_contract(platform_system="Plan9")
        with self.assertRaises(ValueError):
            admitted_cua_provider_contract(
                platform_system="Darwin",
                version="0.10.1",
            )

    def test_validates_complete_catalog_and_exact_observe_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"pid": {"type": "integer"}},
            "required": ["pid"],
            "additionalProperties": False,
        }
        names = ["get_window_state", "platform_probe"]
        contract = CuaProviderContract(
            platform_system="TestOS",
            version="0.10.0",
            tool_count=len(names),
            catalog_names_sha256=_digest(sorted(names)),
            observe_schema_sha256=_digest(schema),
        )
        payload = {
            "version": "0.10.0",
            "tools": [
                {"name": "platform_probe", "input_schema": {}},
                {"name": "get_window_state", "input_schema": schema},
            ],
        }
        with patch.dict(
            provider_contract._CUA_PROVIDER_CONTRACTS,
            {("0.10.0", "TestOS"): contract},
        ):
            self.assertIs(
                validate_cua_provider_document(
                    payload,
                    platform_system="TestOS",
                ),
                contract,
            )

            invalid_payloads = (
                {**payload, "version": "0.10.1"},
                {**payload, "tools": payload["tools"][:1]},
                {
                    **payload,
                    "tools": [
                        payload["tools"][0],
                        {"name": "platform_probe", "input_schema": {}},
                    ],
                },
                {
                    **payload,
                    "tools": [
                        payload["tools"][0],
                        {
                            "name": "get_window_state",
                            "input_schema": {"type": "string"},
                        },
                    ],
                },
            )
            for invalid in invalid_payloads:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    validate_cua_provider_document(
                        invalid,
                        platform_system="TestOS",
                    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
