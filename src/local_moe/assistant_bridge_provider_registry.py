from __future__ import annotations

from types import MappingProxyType
import re
from typing import Generic, Protocol, Sequence, TypeVar


_SAFE_ADAPTER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class ProviderAdapterRegistryError(ValueError):
    """Raised when provider adapter composition is incomplete or ambiguous."""


class IdentifiedProviderAdapter(Protocol):
    adapter_id: str


AdapterT = TypeVar("AdapterT", bound=IdentifiedProviderAdapter)


class ProviderAdapterRegistry(Generic[AdapterT]):
    """Immutable provider-adapter lookup assembled by the application."""

    __slots__ = ("_adapters",)

    def __init__(self, adapters: Sequence[AdapterT]) -> None:
        registry: dict[str, AdapterT] = {}
        for adapter in adapters:
            adapter_id = getattr(adapter, "adapter_id", None)
            if (
                not isinstance(adapter_id, str)
                or _SAFE_ADAPTER_ID.fullmatch(adapter_id) is None
            ):
                raise ProviderAdapterRegistryError(
                    "Provider adapter id must contain safe identifier characters."
                )
            if adapter_id in registry:
                raise ProviderAdapterRegistryError(
                    f"Provider adapter {adapter_id!r} is registered more than once."
                )
            registry[adapter_id] = adapter
        if not registry:
            raise ProviderAdapterRegistryError(
                "Provider adapter registry cannot be empty."
            )
        self._adapters = MappingProxyType(registry)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def require(self, adapter_id: str) -> AdapterT:
        try:
            return self._adapters[adapter_id]
        except (KeyError, TypeError):
            raise ProviderAdapterRegistryError(
                f"Provider adapter {adapter_id!r} is not registered."
            ) from None

    def __contains__(self, adapter_id: object) -> bool:
        return adapter_id in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)
