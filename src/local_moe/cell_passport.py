from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Mapping, TypeVar

from .cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellContractError,
    CellDeclaration,
    CellEstimate,
    CellMeasurement,
    CellObservation,
    CellPassport,
    MAX_CELLS,
    MAX_PROFILES,
)
from .secure_files import read_bounded_regular_file


MAX_CATALOG_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_EVIDENCE_BYTES = 64 * 1024 * 1024
T = TypeVar("T")


def build_cell_passport(
    declaration: CellDeclaration,
    *,
    observed: CellObservation | None = None,
    estimated: CellEstimate | None = None,
    measured: CellMeasurement | None = None,
) -> CellPassport:
    if not isinstance(declaration, CellDeclaration):
        raise CellContractError("declaration must be a CellDeclaration.")
    return CellPassport(
        declaration=declaration,
        observed=observed
        or CellObservation(
            cell_id=declaration.cell_id, declaration_sha256=declaration.digest
        ),
        estimated=estimated
        or CellEstimate(
            cell_id=declaration.cell_id, declaration_sha256=declaration.digest
        ),
        measured=measured
        or CellMeasurement(
            cell_id=declaration.cell_id, declaration_sha256=declaration.digest
        ),
    )


def load_cell_catalog(path: str | Path) -> AdaptiveCellCatalog:
    try:
        source = Path(path)
        root = source.parent.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError(
            "Catalog parent directory does not exist or is inaccessible."
        ) from exc
    content = read_bounded_regular_file(
        root / source.name,
        root=root,
        maximum_bytes=MAX_CATALOG_BYTES,
        label="catalog",
    )
    try:
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: _raise_non_finite(value, str(source)),
        )
    except CellContractError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        ValueError,
    ) as exc:
        raise CellContractError(
            f"Invalid adaptive cell catalog JSON: {source}."
        ) from exc
    if not isinstance(raw, dict):
        raise CellContractError("Adaptive cell catalog must be a JSON object.")
    catalog = _trusted_cell_catalog_from_payload(raw)
    _verify_catalog_sources(catalog, root)
    return catalog


def _trusted_cell_catalog_from_payload(
    raw: Mapping[str, object],
) -> AdaptiveCellCatalog:
    data = _strict(raw, _field_names(AdaptiveCellCatalog), "adaptive cell catalog")
    cells_raw, profiles_raw = data["cells"], data["profiles"]
    if not isinstance(cells_raw, list) or not isinstance(profiles_raw, dict):
        raise CellContractError("Catalog cells and profiles have invalid types.")
    if any(not isinstance(key, str) for key in profiles_raw):
        raise CellContractError("Catalog profile identifiers must be strings.")
    if len(cells_raw) > MAX_CELLS:
        raise CellContractError(f"Catalog cannot contain more than {MAX_CELLS} cells.")
    if len(profiles_raw) > MAX_PROFILES:
        raise CellContractError(
            f"Catalog cannot contain more than {MAX_PROFILES} profiles."
        )
    cells = tuple(_cell_passport_from_payload(item) for item in cells_raw)
    profiles = {
        str(key): _construct(AdvisorProfile, value, "advisor profile")
        for key, value in profiles_raw.items()
    }
    return AdaptiveCellCatalog(
        catalog_id=data["catalog_id"],
        cells=cells,
        profiles=profiles,
        digest=data["digest"],
        mode=data["mode"],
        schema_version=data["schema_version"],
    )  # type: ignore[arg-type]


def _cell_passport_from_payload(raw: object) -> CellPassport:
    data = _strict(raw, _field_names(CellPassport), "cell passport")
    return CellPassport(
        declaration=_construct(
            CellDeclaration, data["declaration"], "cell declaration"
        ),
        observed=_construct(CellObservation, data["observed"], "cell observation"),
        estimated=_construct(CellEstimate, data["estimated"], "cell estimate"),
        measured=_construct(CellMeasurement, data["measured"], "cell measurement"),
        digest=data["digest"],
        schema_version=data["schema_version"],
    )  # type: ignore[arg-type]


def _construct(cls: type[T], raw: object, label: str) -> T:
    data = _strict(raw, _field_names(cls), label)
    try:
        return cls(**data)  # type: ignore[arg-type]
    except CellContractError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CellContractError(f"Invalid {label}: {exc}") from exc


def _verify_catalog_sources(catalog: AdaptiveCellCatalog, root: Path) -> None:
    cache: dict[str, tuple[str, int]] = {}
    total_evidence_bytes = 0
    for cell in catalog.cells:
        for label, evidence in (
            ("observed", cell.observed),
            ("estimated", cell.estimated),
            ("measured", cell.measured),
        ):
            source_path = evidence.source_path
            source_sha256 = evidence.source_sha256
            if source_path is None and source_sha256 is None:
                continue
            if source_path is None or source_sha256 is None:
                raise CellContractError(
                    f"{label} source path and digest must be paired."
                )
            cached = cache.get(source_path)
            if cached is None:
                content = read_bounded_regular_file(
                    root / source_path,
                    root=root,
                    maximum_bytes=MAX_EVIDENCE_BYTES,
                    label=f"{label} evidence",
                )
                actual = hashlib.sha256(content).hexdigest()
                size = len(content)
                total_evidence_bytes += size
                if total_evidence_bytes > MAX_TOTAL_EVIDENCE_BYTES:
                    raise CellContractError(
                        "Catalog evidence exceeds the cumulative bounded size limit."
                    )
                cache[source_path] = (actual, size)
            else:
                actual, _ = cached
            if actual != source_sha256:
                raise CellContractError(
                    f"{label} evidence source_sha256 does not match its file."
                )


def _field_names(cls: type[object]) -> set[str]:
    return {item.name for item in fields(cls)}


def _strict(raw: object, allowed: set[str], label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise CellContractError(f"{label} must be an object.")
    data = dict(raw)
    if any(not isinstance(key, str) for key in data):
        raise CellContractError(f"{label} field names must be strings.")
    unknown, missing = sorted(set(data) - allowed), sorted(allowed - set(data))
    if unknown:
        raise CellContractError(f"Unknown {label} fields: {', '.join(unknown)}.")
    if missing:
        raise CellContractError(f"Missing {label} fields: {', '.join(missing)}.")
    return data


def _raise_non_finite(value: str, label: str) -> object:
    raise CellContractError(f"Non-finite number {value} is not allowed in {label}.")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise CellContractError(
                f"Duplicate JSON object key is not allowed: {key}."
            )
        payload[key] = value
    return payload


__all__ = [
    "MAX_CATALOG_BYTES",
    "MAX_EVIDENCE_BYTES",
    "MAX_TOTAL_EVIDENCE_BYTES",
    "build_cell_passport",
    "load_cell_catalog",
]
