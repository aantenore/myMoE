from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .app_config import AdvisorPolicy, AppConfig, load_app_config
from .adaptive_execution_gate import load_adaptive_execution_policy
from .cell_passport import load_cell_catalog
from .config import load_config
from .context_policy import load_context_policy
from .package_defaults import (
    packaged_default_path,
    resolve_advisor_config_reference,
    resolve_app_config_reference,
)
from .secure_files import read_bounded_regular_file


_OUTPUT_FILES = (
    "app.json",
    "adaptive-cells.json",
    "adaptive-execution-policy.json",
    "adaptive-evaluation-contract.json",
    "moe.json",
    "context-policy.json",
)
_MAX_PACKAGED_RESOURCE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class _WorkspaceHandle:
    root: Path
    root_fd: int
    root_identity: object
    parent_fd: int | None
    pinned_fds: tuple[int, ...]


def materialize_advisor_workspace(output_dir: str | Path) -> dict[str, Any]:
    """Create one self-contained Advisor starter without overwriting."""

    payloads, app = _validated_packaged_payloads()
    payloads = _starter_payloads(payloads)
    root = _new_workspace_path(output_dir)
    workspace: _WorkspaceHandle | None = None
    created: list[str] = []
    try:
        workspace = _create_workspace(root)
        for name in _OUTPUT_FILES:
            _write_exclusive_bytes(workspace, name, payloads[name])
            created.append(name)
        _sync_workspace(workspace)
        _require_current_workspace(workspace)
    except Exception:
        if workspace is not None:
            _cleanup_workspace(workspace, created)
        raise
    finally:
        if workspace is not None:
            _close_workspace(workspace)

    return {
        "schema_version": "1.0",
        "status": "created",
        "files": list(_OUTPUT_FILES),
        "advisor": {
            "enabled": app.advisor.enabled,
            "default_profile": app.advisor.default_profile,
            "evidence_status": "not_qualified",
            "evaluation_side_effects": "none",
        },
        "next": {
            "advisor_argv": _advisor_argv(app.advisor),
            "cell_execution_preview_argv": _cell_execution_preview_argv(),
            "web_argv": ["mymoe-web", "--app-config", "./app.json"],
            "run_from": "workspace",
        },
    }


def _advisor_argv(policy: AdvisorPolicy) -> list[str]:
    command = [
        "mymoe",
        "advisor",
        "--catalog",
        "./adaptive-cells.json",
        "--task-stdin",
        "--workload",
        policy.workload_id,
    ]
    for capability in policy.capabilities:
        command.extend(("--capability", capability))
    for tool_surface in policy.tool_surfaces:
        command.extend(("--tool-surface", tool_surface))
    command.extend(
        (
            "--risk-class",
            policy.risk_class,
            "--context-tokens",
            str(policy.context_tokens),
            "--evaluation-contract",
            "./adaptive-evaluation-contract.json",
            "--goal",
            policy.default_profile,
            "--json",
            "--out",
            "./advisor-receipt.json",
        )
    )
    return command


def _cell_execution_preview_argv() -> list[str]:
    return [
        "mymoe",
        "cell-exec",
        "preview",
        "--receipt",
        "./advisor-receipt.json",
        "--task-stdin",
        "--catalog",
        "./adaptive-cells.json",
        "--evaluation-contract",
        "./adaptive-evaluation-contract.json",
        "--policy",
        "./adaptive-execution-policy.json",
        "--json",
    ]


def _validated_packaged_payloads() -> tuple[dict[str, bytes], AppConfig]:
    payloads: dict[str, bytes] = {}
    defaults_root = packaged_default_path("app.json").parent.resolve()
    for name in _OUTPUT_FILES:
        resource_path = packaged_default_path(name)
        if resource_path.parent.resolve() != defaults_root:
            raise ValueError("Packaged Advisor resources must share one directory.")
        content = read_bounded_regular_file(
            name,
            root=defaults_root,
            maximum_bytes=_MAX_PACKAGED_RESOURCE_BYTES,
            label=f"packaged Advisor resource {name}",
        )
        if not content:
            raise ValueError(f"Packaged Advisor resource has an invalid size: {name}")
        _decode_json_object(content, label=f"packaged Advisor resource {name}")
        payloads[name] = content
    app = _validate_workspace(defaults_root)
    for name, expected in payloads.items():
        current = read_bounded_regular_file(
            name,
            root=defaults_root,
            maximum_bytes=_MAX_PACKAGED_RESOURCE_BYTES,
            label=f"packaged Advisor resource {name}",
        )
        if current != expected:
            raise ValueError(
                f"Packaged Advisor resource changed during validation: {name}"
            )
    return payloads, app


def _starter_payloads(payloads: Mapping[str, bytes]) -> dict[str, bytes]:
    rendered = dict(payloads)
    app = _decode_json_object(rendered["app.json"], label="packaged app config")
    runtime = app.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Packaged app runtime policy must be an object.")
    app["runtime"] = {
        **runtime,
        "work_dir": "./work/runtime",
        "profile_dir": "./profiles",
        "evaluation_dir": "./experiments",
    }
    rendered["app.json"] = (
        json.dumps(app, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _decode_json_object(rendered["app.json"], label="rendered starter app config")
    return rendered


def _decode_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate key in {label}.")
            payload[key] = value
        return payload

    def reject_constant(_value: str) -> None:
        raise ValueError(f"Non-finite number in {label}.")

    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be an object.")
    return decoded


def _new_workspace_path(output_dir: str | Path) -> Path:
    try:
        expanded = Path(output_dir).expanduser()
        if any(part == ".." for part in expanded.parts):
            raise ValueError("Advisor workspace path must not contain '..'.")
        requested = Path(os.path.abspath(os.fspath(expanded)))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("Advisor workspace path is invalid.") from exc
    if requested.name in {"", ".", ".."}:
        raise ValueError("Advisor workspace path is invalid.")
    if os.path.lexists(requested):
        raise FileExistsError("Advisor workspace already exists and was not changed.")
    parent = requested.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = parent.resolve(strict=True)
    _require_plain_directory(parent, "Advisor workspace parent")
    root = parent / requested.name
    if os.path.lexists(root):
        raise FileExistsError("Advisor workspace already exists and was not changed.")
    return root


def _validate_workspace(root: Path) -> AppConfig:
    app_path = root / "app.json"
    app = load_app_config(app_path)
    if not app.advisor.enabled:
        raise ValueError("Packaged Advisor policy must be enabled.")
    execution_policy = load_adaptive_execution_policy(
        root / "adaptive-execution-policy.json"
    )
    if execution_policy.mode != "dry_run":
        raise ValueError("Packaged execution policy must remain dry-run only.")
    load_config(resolve_app_config_reference(app.default_moe_config, app_path))
    load_context_policy(
        resolve_app_config_reference(
            app.runtime.context_policy_config,
            app_path,
        ),
        app.runtime.context_policy_profile,
    )
    catalog = load_cell_catalog(
        resolve_advisor_config_reference(app.advisor.catalog_path, app_path)
    )
    if any(profile not in catalog.profiles for profile in app.advisor.allowed_profiles):
        raise ValueError("Packaged Advisor profiles must exist in its catalog.")
    declared_fit = any(
        set(app.advisor.capabilities).issubset(cell.declaration.capabilities)
        and set(app.advisor.tool_surfaces).issubset(cell.declaration.tool_surfaces)
        and app.advisor.risk_class in cell.declaration.risk_classes
        and app.advisor.context_tokens <= cell.declaration.max_context_tokens
        for cell in catalog.cells
    )
    if not declared_fit:
        raise ValueError("Packaged Advisor workload has no declared candidate.")
    if any(
        cell.measured.sample_count != 0
        or cell.observed.model_status != "unknown"
        or cell.observed.runtime_status != "unknown"
        or cell.observed.harness_status != "unknown"
        or cell.observed.tool_contract_status != "unknown"
        or cell.observed.residency_status != "unknown"
        or cell.estimated.memory_pool is not None
        or cell.estimated.source_path is not None
        for cell in catalog.cells
    ):
        raise ValueError("Packaged Advisor catalog must not claim runtime evidence.")
    evaluation_path = resolve_advisor_config_reference(
        app.advisor.evaluation_contract_path,
        app_path,
    )
    evaluation = _decode_json_object(
        read_bounded_regular_file(
            evaluation_path,
            root=root.resolve(),
            maximum_bytes=_MAX_PACKAGED_RESOURCE_BYTES,
            label="Advisor evaluation contract",
        ),
        label="Advisor evaluation contract",
    )
    workload = evaluation.get("workload")
    if not isinstance(workload, Mapping) or workload != {
        "id": app.advisor.workload_id,
        "capabilities": list(app.advisor.capabilities),
        "tool_surfaces": list(app.advisor.tool_surfaces),
        "risk_class": app.advisor.risk_class,
        "context_tokens": app.advisor.context_tokens,
    }:
        raise ValueError("Packaged evaluation workload must match Advisor policy.")
    qualification = evaluation.get("qualification")
    if (
        not isinstance(qualification, Mapping)
        or qualification.get("status") != "not_qualified"
        or qualification.get("claims") != []
    ):
        raise ValueError("Packaged evaluation contract must contain zero claims.")
    return app


def _require_plain_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable.") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a plain directory.")


def _create_workspace(root: Path) -> _WorkspaceHandle:
    return (
        _create_windows_workspace(root)
        if os.name == "nt"
        else _create_posix_workspace(root)
    )


def _create_posix_workspace(root: Path) -> _WorkspaceHandle:
    required = (os.mkdir, os.open, os.rmdir, os.unlink)
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or any(function not in os.supports_dir_fd for function in required)
    ):
        raise ValueError("Secure Advisor workspace creation is unavailable.")
    parent_fd = _open_posix_directory(root.parent)
    created = False
    root_fd: int | None = None
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        created = True
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY")
            | getattr(os, "O_NOFOLLOW")
        )
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        metadata = os.fstat(root_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Advisor workspace must be a real directory.")
        os.fchmod(root_fd, 0o700)
        return _WorkspaceHandle(
            root=root,
            root_fd=root_fd,
            root_identity=(metadata.st_dev, metadata.st_ino),
            parent_fd=parent_fd,
            pinned_fds=(),
        )
    except FileExistsError:
        raise FileExistsError(
            "Advisor workspace already exists and was not changed."
        ) from None
    except Exception:
        if root_fd is not None:
            os.close(root_fd)
            root_fd = None
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if root_fd is None:
            os.close(parent_fd)


def _open_posix_directory(path: Path) -> int:
    resolved = path.resolve(strict=True)
    if not resolved.is_absolute() or not resolved.anchor:
        raise ValueError("Advisor workspace parent must be absolute.")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY")
        | getattr(os, "O_NOFOLLOW")
    )
    current = os.open(resolved.anchor, flags)
    try:
        for component in resolved.parts[1:]:
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise ValueError("Advisor workspace parent must be a real directory.")
        return current
    except Exception:
        os.close(current)
        raise


def _create_windows_workspace(root: Path) -> _WorkspaceHandle:
    from . import _win32_fs

    pinned: list[int] = []
    created = False
    root_fd: int | None = None
    try:
        for prefix in _windows_directory_prefixes(root.parent):
            descriptor, _identity = _win32_fs.open_nofollow_fd(
                prefix,
                directory=True,
                writable=False,
                share_delete=False,
            )
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError("Advisor workspace parent must be a real directory.")
            pinned.append(descriptor)
        root.mkdir(mode=0o700)
        created = True
        root_fd, identity = _win32_fs.open_nofollow_fd(
            root,
            directory=True,
            writable=False,
            share_delete=False,
        )
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError("Advisor workspace must be a real directory.")
        return _WorkspaceHandle(
            root=root,
            root_fd=root_fd,
            root_identity=identity,
            parent_fd=None,
            pinned_fds=tuple(pinned),
        )
    except FileExistsError:
        raise FileExistsError(
            "Advisor workspace already exists and was not changed."
        ) from None
    except Exception:
        if root_fd is not None:
            os.close(root_fd)
            root_fd = None
        # The root is intentionally left in place after a Windows failure. A
        # pathname-based rmdir after releasing the no-delete-sharing handle
        # would reintroduce a replacement race.
        if not created:
            pass
        raise
    finally:
        if root_fd is None:
            for descriptor in reversed(pinned):
                os.close(descriptor)


def _windows_directory_prefixes(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute() or not path.anchor:
        raise ValueError("Advisor workspace parent must be absolute.")
    current = Path(path.anchor)
    prefixes = [current]
    for component in path.parts[1:]:
        current = current / component
        prefixes.append(current)
    return tuple(prefixes)


def _write_exclusive_bytes(
    workspace: _WorkspaceHandle,
    name: str,
    content: bytes,
) -> None:
    if Path(name).name != name or name not in _OUTPUT_FILES:
        raise ValueError("Advisor workspace output name is invalid.")
    _require_current_workspace(workspace)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = (
            os.open(workspace.root / name, flags, 0o600)
            if os.name == "nt"
            else os.open(name, flags, 0o600, dir_fd=workspace.root_fd)
        )
    except FileExistsError:
        raise FileExistsError(
            f"Advisor workspace file already exists and was not changed: {name}"
        ) from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            written = handle.write(content)
            if written != len(content):
                raise OSError("Advisor workspace file write was incomplete.")
            handle.flush()
            if os.name == "posix":
                os.fchmod(handle.fileno(), 0o600)
            os.fsync(handle.fileno())
            if os.fstat(handle.fileno()).st_size != len(content):
                raise OSError("Advisor workspace file size is inconsistent.")
    except Exception:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        except (OSError, ValueError):
            pass
        try:
            _unlink_workspace_file(workspace, name)
        except (OSError, ValueError):
            pass
        raise


def _unlink_workspace_file(workspace: _WorkspaceHandle, name: str) -> None:
    if os.name == "nt":
        _require_current_workspace(workspace)
        (workspace.root / name).unlink()
    else:
        os.unlink(name, dir_fd=workspace.root_fd)


def _require_current_workspace(workspace: _WorkspaceHandle) -> None:
    if os.name == "nt":
        from . import _win32_fs

        descriptor, identity = _win32_fs.open_nofollow_fd(
            workspace.root,
            directory=True,
            writable=False,
            share_delete=False,
        )
        try:
            same_file_as = getattr(workspace.root_identity, "same_file_as", None)
            if same_file_as is None or not same_file_as(identity):
                raise ValueError("Advisor workspace identity changed.")
        finally:
            os.close(descriptor)
        return
    assert workspace.parent_fd is not None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY")
        | getattr(os, "O_NOFOLLOW")
    )
    try:
        descriptor = os.open(workspace.root.name, flags, dir_fd=workspace.parent_fd)
    except OSError as exc:
        raise ValueError("Advisor workspace identity changed.") from exc
    try:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != workspace.root_identity:
            raise ValueError("Advisor workspace identity changed.")
    finally:
        os.close(descriptor)


def _sync_workspace(workspace: _WorkspaceHandle) -> None:
    if os.name == "posix":
        os.fsync(workspace.root_fd)


def _cleanup_workspace(workspace: _WorkspaceHandle, created: list[str]) -> None:
    for name in reversed(created):
        try:
            _unlink_workspace_file(workspace, name)
        except (OSError, ValueError):
            pass
    try:
        _sync_workspace(workspace)
    except OSError:
        pass
    if os.name == "posix" and workspace.parent_fd is not None:
        try:
            _require_current_workspace(workspace)
            os.rmdir(workspace.root.name, dir_fd=workspace.parent_fd)
        except (OSError, ValueError):
            pass


def _close_workspace(workspace: _WorkspaceHandle) -> None:
    descriptors = [workspace.root_fd]
    if workspace.parent_fd is not None:
        descriptors.append(workspace.parent_fd)
    descriptors.extend(workspace.pinned_fds)
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


__all__ = ["materialize_advisor_workspace"]
