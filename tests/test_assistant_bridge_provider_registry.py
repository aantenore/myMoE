from __future__ import annotations

from dataclasses import dataclass
import unittest

from local_moe.assistant_bridge_provider_registry import (
    ProviderAdapterRegistry,
    ProviderAdapterRegistryError,
)


@dataclass(frozen=True)
class _Adapter:
    adapter_id: str


class ProviderAdapterRegistryTests(unittest.TestCase):
    def test_registry_is_immutable_and_has_deterministic_ids(self) -> None:
        first = _Adapter("z-adapter")
        second = _Adapter("a-adapter")
        registry = ProviderAdapterRegistry((first, second))

        self.assertEqual(registry.ids, ("a-adapter", "z-adapter"))
        self.assertEqual(len(registry), 2)
        self.assertIn("a-adapter", registry)
        self.assertIs(registry.require("a-adapter"), second)
        with self.assertRaises(TypeError):
            registry._adapters["new"] = _Adapter("new")  # type: ignore[index]

    def test_registry_rejects_empty_duplicate_and_invalid_composition(self) -> None:
        cases = (
            (),
            (_Adapter("same"), _Adapter("same")),
            (_Adapter("../invalid"),),
        )
        for adapters in cases:
            with self.subTest(adapters=adapters):
                with self.assertRaises(ProviderAdapterRegistryError):
                    ProviderAdapterRegistry(adapters)

    def test_unknown_adapter_lookup_is_explicit(self) -> None:
        registry = ProviderAdapterRegistry((_Adapter("known"),))

        with self.assertRaisesRegex(
            ProviderAdapterRegistryError,
            "not registered",
        ):
            registry.require("missing")


if __name__ == "__main__":
    unittest.main()
