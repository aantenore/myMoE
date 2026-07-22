from __future__ import annotations

from importlib import import_module, metadata
import json
import os
import re
import sys

from local_moe.assistant_bridge_runtime import runtime_capabilities


REQUIRED_MINOR_SERIES = {
    "detect-secrets": (1, 5),
    "psutil": (7, 2),
    "rfc8785": (0, 1),
}
REQUIRED_MAJOR_RANGES = {
    "cryptography": (48, 49),
    "filelock": (3, 4),
    "platformdirs": (4, 5),
}
REQUIRED_MINIMUM_RELEASES = {
    "cryptography": (48, 0, 1),
    "filelock": (3, 29, 7),
}
REQUIRED_RANGE_LABELS = {
    "cryptography": ">=48.0.1,<49",
    "filelock": ">=3.29.7,<4",
}


def main() -> None:
    versions = validate_optional_dependencies()
    validate_installed_project_metadata()
    capabilities = runtime_capabilities()
    if not capabilities.psutil_available:
        raise SystemExit("The assistant-bridge runtime did not load psutil.")
    if os.name == "nt" and not capabilities.strict_tree_supported:
        raise SystemExit(
            "The assistant-bridge runtime lacks strict process-tree support on Windows."
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": sys.platform,
                "dependencies": versions,
                "psutil_available": capabilities.psutil_available,
                "strict_tree_supported": capabilities.strict_tree_supported,
            },
            sort_keys=True,
        )
    )


def validate_optional_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, required_series in REQUIRED_MINOR_SERIES.items():
        try:
            value = metadata.version(distribution)
            import_module(distribution.replace("-", "_"))
        except (ImportError, metadata.PackageNotFoundError) as exc:
            raise SystemExit(
                f"The assistant-bridge extra is incomplete: {distribution} is missing."
            ) from exc
        if _major_minor(value) != required_series:
            required = ".".join(str(part) for part in required_series)
            raise SystemExit(
                f"The assistant-bridge extra requires {distribution} {required}.x; "
                f"the installed version is {value}."
            )
        versions[distribution] = value
    for distribution, (minimum, maximum) in REQUIRED_MAJOR_RANGES.items():
        try:
            value = metadata.version(distribution)
            import_module(distribution.replace("-", "_"))
        except (ImportError, metadata.PackageNotFoundError) as exc:
            raise SystemExit(
                f"The assistant-bridge extra is incomplete: {distribution} is missing."
            ) from exc
        major = _major_minor(value)[0]
        minimum_release = REQUIRED_MINIMUM_RELEASES.get(distribution)
        release_is_supported = (
            minimum_release is None or _release_triplet(value) >= minimum_release
        )
        if not minimum <= major < maximum or not release_is_supported:
            required = REQUIRED_RANGE_LABELS.get(
                distribution,
                f">={minimum},<{maximum}",
            )
            raise SystemExit(
                f"The assistant-bridge extra requires {distribution} "
                f"{required}; the installed version is {value}."
            )
        versions[distribution] = value
    return versions


def validate_installed_project_metadata() -> None:
    requirements = metadata.requires("local-moe-orchestrator") or []
    optional_expected = (
        "cryptography",
        "detect-secrets",
        "psutil",
        "rfc8785",
    )
    for dependency in optional_expected:
        matches = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith(dependency)
            and 'extra == "assistant-bridge"' in requirement
        ]
        if len(matches) != 1 or not _metadata_range_is_supported(
            dependency, matches[0]
        ):
            raise SystemExit(
                f"Installed metadata does not expose the expected {dependency} "
                "assistant-bridge requirement."
            )

    base_expected = ("filelock", "platformdirs")
    base_requirements = {
        dependency: [
            requirement
            for requirement in requirements
            if requirement.lower().startswith(dependency)
            and "extra ==" not in requirement
        ]
        for dependency in base_expected
    }
    for dependency, matches in base_requirements.items():
        if len(matches) != 1 or not _metadata_range_is_supported(
            dependency, matches[0]
        ):
            raise SystemExit(
                f"Installed metadata does not expose the expected {dependency} "
                "base requirement."
            )
    unexpected_base = [
        requirement
        for requirement in requirements
        if "extra ==" not in requirement
        and not any(
            requirement.lower().startswith(dependency)
            for dependency in base_expected
        )
    ]
    if unexpected_base:
        raise SystemExit("The base distribution exposes an unexpected dependency.")


def _metadata_range_is_supported(dependency: str, requirement: str) -> bool:
    if dependency == "detect-secrets":
        return ">=1.5" in requirement and "<1.6" in requirement
    if dependency == "psutil":
        return ">=7.2" in requirement and any(
            upper_bound in requirement for upper_bound in ("<7.3", "<8")
        )
    if dependency == "cryptography":
        return ">=48.0.1" in requirement and "<49" in requirement
    if dependency == "filelock":
        return ">=3.29.7" in requirement and "<4" in requirement
    if dependency == "platformdirs":
        return ">=4.3" in requirement and "<5" in requirement
    if dependency == "rfc8785":
        return ">=0.1.4" in requirement and "<0.2" in requirement
    return False


def _major_minor(value: str) -> tuple[int, int]:
    numeric = value.split("+", 1)[0].split(".")
    try:
        return int(numeric[0]), int(numeric[1])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"Unsupported dependency version format: {value}") from exc


def _release_triplet(value: str) -> tuple[int, int, int]:
    normalized = value.split("+", 1)[0]
    match = re.fullmatch(
        r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.post\d+)?",
        normalized,
    )
    if match is None:
        raise SystemExit(f"Unsupported dependency version format: {value}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


if __name__ == "__main__":
    main()
