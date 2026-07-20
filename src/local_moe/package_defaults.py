from __future__ import annotations

from importlib import resources
from pathlib import Path


_DEFAULTS = frozenset({"app.json", "context-policy.json", "moe.json"})
_SOURCE_APP_CONFIG = Path("configs/app.json")


def packaged_default_path(name: str) -> Path:
    """Return a filesystem path for a default shipped inside the installed package."""

    if name not in _DEFAULTS:
        raise ValueError(f"Unknown packaged default: {name}")
    resource = resources.files("local_moe").joinpath("defaults", name)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged default is missing: {name}")
    try:
        return Path(resource)
    except TypeError as exc:  # pragma: no cover - pip installs wheels unpacked
        raise RuntimeError(
            "myMoE packaged defaults require an unpacked installation."
        ) from exc


def resolve_app_config_path(
    explicit: str | Path | None = None,
    *,
    working_directory: str | Path | None = None,
) -> Path:
    """Resolve an explicit app config, a source checkout config, or the wheel default."""

    if explicit is not None:
        return Path(explicit).expanduser()
    source_root = (
        Path.cwd()
        if working_directory is None
        else Path(working_directory).expanduser()
    )
    source_config = source_root / _SOURCE_APP_CONFIG
    if source_config.is_file():
        return (
            _SOURCE_APP_CONFIG
            if working_directory is None
            else source_config
        )
    return packaged_default_path("app.json")


def resolve_app_config_reference(value: str | Path, app_config_path: str | Path) -> Path:
    """Resolve package-relative references without changing normal CWD semantics."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    app_path = Path(app_config_path).expanduser()
    if _is_packaged_default(app_path):
        defaults_dir = app_path.parent.resolve()
        packaged_candidate = (defaults_dir / candidate).resolve()
        if not packaged_candidate.is_relative_to(defaults_dir):
            raise ValueError("Packaged config references must stay inside defaults.")
        return packaged_candidate
    return candidate


def _is_packaged_default(path: Path) -> bool:
    defaults_dir = packaged_default_path("app.json").parent.resolve()
    try:
        return path.resolve().parent == defaults_dir
    except OSError:
        return False
