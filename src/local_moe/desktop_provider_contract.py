from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from typing import Any, Mapping


CUA_DRIVER_CONTRACT_VERSION = "0.10.0"
CUA_DRIVER_OBSERVE_TOOL = "get_window_state"
CUA_DRIVER_DISABLE_A11Y_ADVERTISE_ENV = (
    "CUA_DRIVER_RS_DISABLE_A11Y_ADVERTISE"
)
CUA_DRIVER_DISABLE_A11Y_ADVERTISE_VALUE = "1"


@dataclass(frozen=True)
class CuaProviderContract:
    """Reviewed native Cua Driver surface for one operating system."""

    platform_system: str
    version: str
    tool_count: int
    catalog_names_sha256: str
    observe_schema_sha256: str


_CUA_PROVIDER_CONTRACTS = {
    (CUA_DRIVER_CONTRACT_VERSION, "Darwin"): CuaProviderContract(
        platform_system="Darwin",
        version=CUA_DRIVER_CONTRACT_VERSION,
        tool_count=49,
        catalog_names_sha256=(
            "a39bbb495c25d8c24f388e06ecd10f4aec96b8486b832750ef65daac89f4bd69"
        ),
        observe_schema_sha256=(
            "a1685e0da284cf8445e9d2e11bdbd7249e20b72ee9109800aba798ee7ff322c3"
        ),
    ),
    (CUA_DRIVER_CONTRACT_VERSION, "Linux"): CuaProviderContract(
        platform_system="Linux",
        version=CUA_DRIVER_CONTRACT_VERSION,
        tool_count=53,
        catalog_names_sha256=(
            "c8d63c8a14b49781d64c6f739b4dc484789f2738ed559fd3d15fcbed15271a85"
        ),
        observe_schema_sha256=(
            "7c039adde1f1f403e9350e9e0c67005f0403b0898b677a2543702b838e8a167a"
        ),
    ),
    (CUA_DRIVER_CONTRACT_VERSION, "Windows"): CuaProviderContract(
        platform_system="Windows",
        version=CUA_DRIVER_CONTRACT_VERSION,
        tool_count=50,
        catalog_names_sha256=(
            "ee77ceaf809bc8eef3f2a85ebdc19f66c798df73252e388a95203bc0d9421e81"
        ),
        observe_schema_sha256=(
            "442ffd78dbf3212af5101d2f82ae988cff582d1cd07f3cb65a0f807e1169998d"
        ),
    ),
}


def admitted_cua_provider_contract(
    *,
    version: str = CUA_DRIVER_CONTRACT_VERSION,
    platform_system: str | None = None,
) -> CuaProviderContract:
    """Return the exact reviewed contract or fail closed on an unknown target."""

    system = platform.system() if platform_system is None else platform_system
    contract = _CUA_PROVIDER_CONTRACTS.get((version, system))
    if contract is None:
        raise ValueError("No admitted Cua Driver contract exists for this platform.")
    return contract


def validate_cua_provider_document(
    payload: Mapping[str, Any],
    *,
    version: str = CUA_DRIVER_CONTRACT_VERSION,
    platform_system: str | None = None,
) -> CuaProviderContract:
    """Validate version, complete platform catalog, and the admitted observe schema."""

    contract = admitted_cua_provider_contract(
        version=version,
        platform_system=platform_system,
    )
    if payload.get("version") != contract.version:
        raise ValueError(
            "Cua Driver documentation version did not match: "
            f"expected {contract.version!r}, observed {payload.get('version')!r}."
        )
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) != contract.tool_count:
        observed_count = len(tools) if isinstance(tools, list) else "malformed"
        raise ValueError(
            "Cua Driver platform catalog size did not match: "
            f"expected {contract.tool_count}, observed {observed_count}."
        )
    names: list[str] = []
    observe_schema: object | None = None
    for item in tools:
        if not isinstance(item, dict):
            raise ValueError("Cua Driver platform catalog is malformed.")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Cua Driver platform catalog is malformed.")
        names.append(name)
        if name == CUA_DRIVER_OBSERVE_TOOL:
            observe_schema = item.get("input_schema")
    if len(set(names)) != len(names):
        raise ValueError("Cua Driver platform catalog contains duplicate tools.")
    observed_catalog_digest = _sha256_json(sorted(names))
    if observed_catalog_digest != contract.catalog_names_sha256:
        raise ValueError(
            "Cua Driver platform catalog-name digest did not match: "
            f"expected {contract.catalog_names_sha256}, "
            f"observed {observed_catalog_digest}."
        )
    if not isinstance(observe_schema, dict):
        raise ValueError("Cua Driver omitted the admitted observe schema.")
    observed_schema_digest = _sha256_json(observe_schema)
    if observed_schema_digest != contract.observe_schema_sha256:
        raise ValueError(
            "Cua Driver observe schema digest did not match: "
            f"expected {contract.observe_schema_sha256}, "
            f"observed {observed_schema_digest}."
        )
    return contract


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
