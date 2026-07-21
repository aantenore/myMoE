from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import sys
import threading
import time
from typing import Mapping

from .cell_contracts import (
    CellContractError,
    _digest,
    _enum,
    _integer,
    _optional_sha,
    _positive,
    _safe,
    _schema,
    _sha,
    _timestamp,
    normalize_machine,
)
from .verified_routing_contracts import CONTRACT_VERSION, now_utc, sha256_json


MEMORY_TOPOLOGIES = frozenset({"unified", "system", "dedicated", "unknown"})
ACCELERATOR_KINDS = frozenset({"integrated", "discrete", "none", "unknown"})
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
MAX_PROC_TEXT_BYTES = 4 * 1024 * 1024
MAX_CGROUP_TEXT_BYTES = 256 * 1024
MAX_MOUNTINFO_TEXT_BYTES = 2 * 1024 * 1024
PROBE_TIMEOUT_SECONDS = 2.0
_FIELDS = {
    "schema_version",
    "system",
    "os_release",
    "machine",
    "cpu_count",
    "cpu_identity_sha256",
    "memory_topology",
    "total_memory_bytes",
    "available_memory_bytes",
    "effective_memory_limit_bytes",
    "swap_used_bytes",
    "accelerator_kind",
    "accelerator_identity_sha256",
    "accelerator_memory_total_bytes",
    "accelerator_memory_available_bytes",
    "runtime_environment_sha256",
    "captured_at",
    "source_sha256",
    "resource_class_sha256",
    "digest",
}


def _optional_non_negative(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


@dataclass(frozen=True)
class ResourceSnapshot:
    system: str
    os_release: str
    machine: str
    cpu_count: int | None
    cpu_identity_sha256: str | None
    memory_topology: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    effective_memory_limit_bytes: int | None
    swap_used_bytes: int | None
    accelerator_kind: str
    accelerator_identity_sha256: str | None
    accelerator_memory_total_bytes: int | None
    accelerator_memory_available_bytes: int | None
    runtime_environment_sha256: str | None
    captured_at: str
    source_sha256: str
    resource_class_sha256: str = ""
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "resource snapshot")
        object.__setattr__(self, "system", _safe(self.system, "system"))
        object.__setattr__(self, "os_release", _safe(self.os_release, "os_release"))
        object.__setattr__(self, "machine", normalize_machine(self.machine))
        cpu_count = (
            None if self.cpu_count is None else _positive(self.cpu_count, "cpu_count")
        )
        object.__setattr__(self, "cpu_count", cpu_count)
        object.__setattr__(
            self,
            "cpu_identity_sha256",
            _optional_sha(self.cpu_identity_sha256, "cpu_identity_sha256"),
        )
        object.__setattr__(
            self,
            "memory_topology",
            _enum(self.memory_topology, MEMORY_TOPOLOGIES, "memory_topology"),
        )
        object.__setattr__(
            self,
            "accelerator_kind",
            _enum(self.accelerator_kind, ACCELERATOR_KINDS, "accelerator_kind"),
        )
        for name in (
            "total_memory_bytes",
            "available_memory_bytes",
            "effective_memory_limit_bytes",
            "swap_used_bytes",
            "accelerator_memory_total_bytes",
            "accelerator_memory_available_bytes",
        ):
            object.__setattr__(
                self, name, _optional_non_negative(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "accelerator_identity_sha256",
            _optional_sha(
                self.accelerator_identity_sha256, "accelerator_identity_sha256"
            ),
        )
        object.__setattr__(
            self,
            "runtime_environment_sha256",
            _optional_sha(
                self.runtime_environment_sha256, "runtime_environment_sha256"
            ),
        )
        if self.available_memory_bytes is not None:
            if (
                self.total_memory_bytes is None
                or self.available_memory_bytes > self.total_memory_bytes
            ):
                raise CellContractError(
                    "Available memory requires a total and cannot exceed it."
                )
        if (
            self.effective_memory_limit_bytes is not None
            and self.total_memory_bytes is not None
        ):
            if self.effective_memory_limit_bytes > self.total_memory_bytes:
                raise CellContractError(
                    "Effective memory limit cannot exceed physical memory."
                )
        if self.accelerator_memory_available_bytes is not None:
            if (
                self.accelerator_memory_total_bytes is None
                or self.accelerator_memory_available_bytes
                > self.accelerator_memory_total_bytes
            ):
                raise CellContractError(
                    "Accelerator available memory requires a total and cannot exceed it."
                )
        if self.accelerator_kind == "none" and any(
            value is not None
            for value in (
                self.accelerator_identity_sha256,
                self.accelerator_memory_total_bytes,
                self.accelerator_memory_available_bytes,
            )
        ):
            raise CellContractError(
                "accelerator_kind=none cannot claim identity or VRAM."
            )
        if self.memory_topology == "dedicated" and (
            self.accelerator_kind != "discrete"
            or self.accelerator_identity_sha256 is None
            or self.accelerator_memory_total_bytes is None
        ):
            raise CellContractError(
                "Dedicated topology requires a discrete identified accelerator and VRAM total."
            )
        if self.memory_topology == "unified" and (
            self.accelerator_kind != "integrated"
            or self.accelerator_identity_sha256 is None
        ):
            raise CellContractError(
                "Unified topology requires an identified integrated accelerator."
            )
        if self.memory_topology == "system" and (
            self.accelerator_kind != "none"
            or self.accelerator_identity_sha256 is not None
            or self.accelerator_memory_total_bytes is not None
            or self.accelerator_memory_available_bytes is not None
        ):
            raise CellContractError("System topology cannot claim an accelerator pool.")
        if self.memory_topology == "unknown" and (
            self.accelerator_kind != "unknown"
            or self.accelerator_identity_sha256 is not None
            or self.accelerator_memory_total_bytes is not None
            or self.accelerator_memory_available_bytes is not None
        ):
            raise CellContractError(
                "Unknown topology cannot claim accelerator identity or VRAM."
            )
        apple = self.system == "Darwin" and self.machine == "arm64"
        if apple and self.memory_topology not in {"unified", "unknown"}:
            raise CellContractError(
                "Apple Silicon memory_topology must be unified or unknown."
            )
        if apple and self.accelerator_kind not in {"integrated", "unknown"}:
            raise CellContractError(
                "Apple Silicon cannot declare a discrete accelerator pool."
            )
        if self.memory_topology == "unified" and any(
            value is not None
            for value in (
                self.accelerator_memory_total_bytes,
                self.accelerator_memory_available_bytes,
            )
        ):
            raise CellContractError(
                "Unified memory must not be represented as accelerator VRAM."
            )
        object.__setattr__(
            self, "captured_at", _timestamp(self.captured_at, "captured_at")
        )
        object.__setattr__(
            self, "source_sha256", _sha(self.source_sha256, "source_sha256")
        )
        class_digest = _digest(
            self.resource_class_sha256,
            self.resource_class_payload(),
            "resource_class_sha256",
        )
        object.__setattr__(self, "resource_class_sha256", class_digest)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "resource snapshot digest"),
        )

    def resource_class_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "system": self.system,
            "os_release": self.os_release,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "cpu_identity_sha256": self.cpu_identity_sha256,
            "memory_topology": self.memory_topology,
            "total_memory_bytes": self.total_memory_bytes,
            "effective_memory_limit_bytes": self.effective_memory_limit_bytes,
            "accelerator_kind": self.accelerator_kind,
            "accelerator_identity_sha256": self.accelerator_identity_sha256,
            "accelerator_memory_total_bytes": self.accelerator_memory_total_bytes,
            "runtime_environment_sha256": self.runtime_environment_sha256,
        }

    def content_payload(self) -> dict[str, object]:
        return {
            **self.resource_class_payload(),
            "available_memory_bytes": self.available_memory_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "accelerator_memory_available_bytes": self.accelerator_memory_available_bytes,
            "captured_at": self.captured_at,
            "source_sha256": self.source_sha256,
            "resource_class_sha256": self.resource_class_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def build_resource_snapshot(
    *,
    system: str,
    os_release: str,
    machine: str,
    cpu_count: int | None,
    cpu_identity_sha256: str | None,
    memory_topology: str,
    total_memory_bytes: int | None,
    available_memory_bytes: int | None,
    effective_memory_limit_bytes: int | None,
    swap_used_bytes: int | None,
    accelerator_kind: str,
    accelerator_identity_sha256: str | None,
    accelerator_memory_total_bytes: int | None = None,
    accelerator_memory_available_bytes: int | None = None,
    runtime_environment_sha256: str | None,
    captured_at: str,
    source: Mapping[str, object],
) -> ResourceSnapshot:
    try:
        source_sha256 = sha256_json(dict(source))
        return ResourceSnapshot(
            system=system,
            os_release=os_release,
            machine=machine,
            cpu_count=cpu_count,
            cpu_identity_sha256=cpu_identity_sha256,
            memory_topology=memory_topology,
            total_memory_bytes=total_memory_bytes,
            available_memory_bytes=available_memory_bytes,
            effective_memory_limit_bytes=effective_memory_limit_bytes,
            swap_used_bytes=swap_used_bytes,
            accelerator_kind=accelerator_kind,
            accelerator_identity_sha256=accelerator_identity_sha256,
            accelerator_memory_total_bytes=accelerator_memory_total_bytes,
            accelerator_memory_available_bytes=accelerator_memory_available_bytes,
            runtime_environment_sha256=runtime_environment_sha256,
            captured_at=captured_at,
            source_sha256=source_sha256,
        )
    except CellContractError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CellContractError("Invalid resource snapshot input.") from exc


def resource_snapshot_from_payload(raw: Mapping[str, object]) -> ResourceSnapshot:
    data = _strict(raw, _FIELDS, "resource snapshot")
    try:
        return ResourceSnapshot(**data)  # type: ignore[arg-type]
    except CellContractError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CellContractError("Invalid resource snapshot payload.") from exc


def collect_resource_snapshot(*, captured_at: str | None = None) -> ResourceSnapshot:
    system = platform.system() or "unknown"
    machine = normalize_machine(platform.machine() or "unknown")
    os_release = _safe(
        (platform.release() or "unknown").replace(" ", "_"), "os_release"
    )
    memory = _read_memory(system)
    cpu_identity = _cpu_identity(system)
    apple = system == "Darwin" and machine == "arm64"
    accelerator_identity = (
        sha256_json({"kind": "integrated", "soc": cpu_identity})
        if apple and cpu_identity is not None
        else None
    )
    runtime_environment = sha256_json(
        {
            "implementation": sys.implementation.name,
            "python": platform.python_version(),
            "system": system,
            "machine": machine,
            "cgroup": memory.get("cgroup_mode"),
        }
    )
    source = {
        "system": system,
        "os_release": os_release,
        "machine": machine,
        "cpu_count": os.cpu_count(),
        "cpu_identity_sha256": cpu_identity,
        "memory": memory,
        "runtime_environment_sha256": runtime_environment,
        "accelerator_identity_sha256": accelerator_identity,
    }
    qualified_apple = apple and accelerator_identity is not None
    return build_resource_snapshot(
        system=system,
        os_release=os_release,
        machine=machine,
        cpu_count=os.cpu_count(),
        cpu_identity_sha256=cpu_identity,
        memory_topology="unified" if qualified_apple else "unknown",
        total_memory_bytes=memory.get("total_memory_bytes"),
        available_memory_bytes=memory.get("available_memory_bytes"),
        effective_memory_limit_bytes=memory.get("effective_memory_limit_bytes"),
        swap_used_bytes=memory.get("swap_used_bytes"),
        accelerator_kind="integrated" if qualified_apple else "unknown",
        accelerator_identity_sha256=accelerator_identity if qualified_apple else None,
        runtime_environment_sha256=runtime_environment,
        captured_at=captured_at or now_utc(),
        source=source,
    )


def _read_memory(system: str) -> dict[str, object]:
    if system == "Linux":
        return _read_linux_memory()
    if system == "Darwin":
        return _read_darwin_memory()
    if system == "Windows":
        return _read_windows_memory()
    return _unknown_memory()


def _read_linux_memory() -> dict[str, object]:
    text = _read_bounded_text(Path("/proc/meminfo"), maximum_bytes=256 * 1024)
    if text is None:
        return _unknown_memory()
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, raw = line.partition(":")
        match = re.match(r"\s*(\d+)\s+kB", raw) if separator else None
        if match:
            parsed = _parse_non_negative_decimal(match.group(1))
            if parsed is None:
                return _unknown_memory()
            values[name] = parsed * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total, swap_free = values.get("SwapTotal"), values.get("SwapFree")
    limit, cgroup_available, mode = _linux_cgroup_memory()
    if limit is not None and total is not None:
        limit = min(limit, total)
    if cgroup_available is not None:
        available = (
            cgroup_available if available is None else min(available, cgroup_available)
        )
    return {
        "total_memory_bytes": total,
        "available_memory_bytes": available,
        "effective_memory_limit_bytes": (
            total if mode in {"v1_unbounded", "v2_unbounded"} else limit
        ),
        "swap_used_bytes": None
        if swap_total is None or swap_free is None
        else max(0, swap_total - swap_free),
        "cgroup_mode": mode,
    }


def _linux_cgroup_memory() -> tuple[int | None, int | None, str]:
    memberships = _read_bounded_text(
        Path("/proc/self/cgroup"),
        maximum_bytes=MAX_CGROUP_TEXT_BYTES,
    )
    mounts = _read_bounded_text(
        Path("/proc/self/mountinfo"),
        maximum_bytes=MAX_MOUNTINFO_TEXT_BYTES,
    )
    if memberships is None or mounts is None:
        return None, None, "unknown"
    resolved = _resolve_cgroup_memory_files(memberships, mounts)
    if resolved is None:
        return None, None, "unknown"
    mode, leaf, boundary = resolved
    finite_limits: list[int] = []
    finite_headrooms: list[int] = []
    levels = _cgroup_levels(leaf, boundary)
    if levels is None:
        return None, None, "unknown"
    for level in levels:
        if mode == "v2":
            limit_path, current_path = level / "memory.max", level / "memory.current"
        else:
            limit_path = level / "memory.limit_in_bytes"
            current_path = level / "memory.usage_in_bytes"
        limit_raw = _read_bounded_text(limit_path, maximum_bytes=256)
        current_raw = _read_bounded_text(current_path, maximum_bytes=256)
        if limit_raw is None or current_raw is None:
            return None, None, "unknown"
        current = _parse_non_negative_decimal(current_raw.strip())
        if current is None:
            return None, None, "unknown"
        limit_text = limit_raw.strip()
        if mode == "v2" and limit_text == "max":
            continue
        limit = _parse_non_negative_decimal(limit_text)
        if limit is None:
            return None, None, "unknown"
        if mode == "v1" and limit >= (1 << 60):
            continue
        finite_limits.append(limit)
        finite_headrooms.append(max(0, limit - current))
    if not finite_limits:
        return None, None, f"{mode}_unbounded"
    return min(finite_limits), min(finite_headrooms), mode


def _resolve_cgroup_memory_files(
    cgroup_text: str,
    mountinfo_text: str,
) -> tuple[str, Path, Path] | None:
    memberships = _parse_cgroup_memberships(cgroup_text)
    mounts = _parse_cgroup_mounts(mountinfo_text)
    if memberships is None or mounts is None:
        return None
    candidates: list[tuple[str, Path, Path]] = []
    for mode, membership_path in memberships:
        for mount_mode, mount_root, mount_point in mounts:
            if mount_mode != mode:
                continue
            mapped = _map_cgroup_membership(
                membership_path,
                mount_root=mount_root,
                mount_point=mount_point,
            )
            if mapped is None:
                continue
            candidates.append((mode, mapped, Path(mount_point.as_posix())))
    return candidates[0] if len(candidates) == 1 else None


def _cgroup_levels(leaf: Path, boundary: Path) -> tuple[Path, ...] | None:
    try:
        leaf.relative_to(boundary)
    except ValueError:
        return None
    levels: list[Path] = []
    current = leaf
    while True:
        levels.append(current)
        if current == boundary:
            return tuple(levels)
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _parse_cgroup_memberships(text: str) -> list[tuple[str, PurePosixPath]] | None:
    results: list[tuple[str, PurePosixPath]] = []
    seen_modes: set[str] = set()
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            return None
        hierarchy, raw_controllers, raw_path = parts
        path = _absolute_kernel_path(raw_path)
        if path is None:
            return None
        controllers = raw_controllers.split(",") if raw_controllers else []
        if any(
            not controller or re.fullmatch(r"[A-Za-z0-9_.=-]+", controller) is None
            for controller in controllers
        ):
            return None
        mode: str | None = None
        if hierarchy == "0" and not controllers:
            mode = "v2"
        elif "memory" in controllers:
            mode = "v1"
        if mode is not None:
            if mode in seen_modes:
                return None
            seen_modes.add(mode)
            results.append((mode, path))
    return results


def _parse_cgroup_mounts(
    text: str,
) -> list[tuple[str, PurePosixPath, PurePosixPath]] | None:
    results: list[tuple[str, PurePosixPath, PurePosixPath]] = []
    for line in text.splitlines():
        if not line:
            continue
        sections = line.split(" - ")
        if len(sections) != 2:
            return None
        left, right = sections[0].split(), sections[1].split()
        if len(left) < 6 or len(right) < 3:
            return None
        filesystem = right[0]
        if filesystem not in {"cgroup", "cgroup2"}:
            continue
        raw_root = _decode_mountinfo_field(left[3])
        raw_mount = _decode_mountinfo_field(left[4])
        if raw_root is None or raw_mount is None:
            return None
        mount_root = _absolute_kernel_path(raw_root)
        mount_point = _absolute_kernel_path(raw_mount)
        if mount_root is None or mount_point is None:
            return None
        if filesystem == "cgroup2":
            results.append(("v2", mount_root, mount_point))
            continue
        super_options = set(right[2].split(","))
        if "memory" in super_options:
            results.append(("v1", mount_root, mount_point))
    return results


def _decode_mountinfo_field(value: str) -> str | None:
    replacements = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            output.append(value[index])
            index += 1
            continue
        escaped = value[index + 1 : index + 4]
        replacement = replacements.get(escaped)
        if replacement is None:
            return None
        output.append(replacement)
        index += 4
    return "".join(output)


def _absolute_kernel_path(value: str) -> PurePosixPath | None:
    if not value.startswith("/") or "\x00" in value:
        return None
    if value != "/" and any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        return None
    path = PurePosixPath(value)
    return path


def _map_cgroup_membership(
    membership: PurePosixPath,
    *,
    mount_root: PurePosixPath,
    mount_point: PurePosixPath,
) -> Path | None:
    try:
        relative = membership.relative_to(mount_root)
    except ValueError:
        return None
    return Path(mount_point.as_posix()).joinpath(*relative.parts)


def _parse_non_negative_decimal(value: str) -> int | None:
    if not value.isdigit():
        return None
    try:
        return int(value)
    except (OverflowError, ValueError):
        return None


def _read_darwin_memory() -> dict[str, object]:
    total_raw = _run_readonly(("/usr/sbin/sysctl", "-n", "hw.memsize"))
    total = int(total_raw) if total_raw.isdigit() else None
    vm_stat = _run_readonly(("/usr/bin/vm_stat",))
    page_match = re.search(r"page size of (\d+) bytes", vm_stat)
    page_size = int(page_match.group(1)) if page_match else None
    pages: dict[str, int] = {}
    for line in vm_stat.splitlines():
        name, separator, raw = line.partition(":")
        value = raw.strip().rstrip(".")
        if separator and value.isdigit():
            pages[name] = int(value)
    available = (
        None
        if page_size is None
        else page_size
        * sum(
            pages.get(name, 0)
            for name in ("Pages free", "Pages inactive", "Pages speculative")
        )
    )
    swap = _run_readonly(("/usr/sbin/sysctl", "-n", "vm.swapusage"))
    used = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", swap)
    return {
        "total_memory_bytes": total,
        "available_memory_bytes": available,
        "effective_memory_limit_bytes": total,
        "swap_used_bytes": None
        if used is None
        else _scaled_bytes(float(used.group(1)), used.group(2)),
        "cgroup_mode": "none",
    }


def _read_windows_memory() -> dict[str, object]:
    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    try:
        populated = bool(
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        )
    except (AttributeError, OSError):
        populated = False
    if not populated:
        return _unknown_memory()
    return {
        "total_memory_bytes": int(status.total_physical),
        "available_memory_bytes": int(status.available_physical),
        "effective_memory_limit_bytes": int(status.total_physical),
        # Commit charge is not swap usage. Unknown is safer than a false signal.
        "swap_used_bytes": None,
        "cgroup_mode": "none",
    }


def _unknown_memory() -> dict[str, object]:
    return {
        "total_memory_bytes": None,
        "available_memory_bytes": None,
        "effective_memory_limit_bytes": None,
        "swap_used_bytes": None,
        "cgroup_mode": "unknown",
    }


def _cpu_identity(system: str) -> str | None:
    if system == "Darwin":
        rendered = "|".join(
            filter(
                None,
                (
                    _run_readonly(
                        ("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
                    ),
                    _run_readonly(("/usr/sbin/sysctl", "-n", "machdep.cpu.features")),
                    _run_readonly(
                        ("/usr/sbin/sysctl", "-n", "machdep.cpu.leaf7_features")
                    ),
                ),
            )
        )
    elif system == "Linux":
        raw = _read_bounded_text(
            Path("/proc/cpuinfo"),
            maximum_bytes=MAX_PROC_TEXT_BYTES,
            errors="ignore",
        )
        lines = [] if raw is None else raw.splitlines()
        rendered = "|".join(
            sorted(
                {
                    line.strip()
                    for line in lines
                    if line.lower().startswith(
                        ("model name", "vendor_id", "hardware", "flags", "features")
                    )
                }
            )
        )
    else:
        rendered = platform.processor().strip()
    return sha256_json({"cpu": rendered}) if rendered else None


def _run_readonly(command: tuple[str, ...]) -> str:
    if not command or not Path(command[0]).is_absolute():
        raise CellContractError(
            "Read-only probe commands must use absolute executable paths."
        )
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LC_ALL": "C", "LANG": "C"},
            bufsize=0,
        )
    except OSError:
        return ""
    if process.stdout is None:
        _terminate_probe(process)
        return ""
    chunks: list[bytes] = []
    state: dict[str, object] = {"overflow": False, "error": None}

    def _drain() -> None:
        total = 0
        try:
            while True:
                chunk = os.read(
                    process.stdout.fileno(),
                    min(16 * 1024, MAX_COMMAND_OUTPUT_BYTES - total + 1),
                )
                if not chunk:
                    return
                total += len(chunk)
                if total > MAX_COMMAND_OUTPUT_BYTES:
                    state["overflow"] = True
                    return
                chunks.append(chunk)
        except OSError as exc:
            state["error"] = exc

    reader = threading.Thread(target=_drain, daemon=True)
    started = time.monotonic()
    reader.start()
    reader.join(PROBE_TIMEOUT_SECONDS)
    timed_out = reader.is_alive()
    if timed_out or state["overflow"]:
        _terminate_probe(process)
    else:
        remaining = max(0.0, PROBE_TIMEOUT_SECONDS - (time.monotonic() - started))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_probe(process)
    reader.join(0.5)
    try:
        process.stdout.close()
    except OSError:
        pass
    if (
        timed_out
        or reader.is_alive()
        or state["overflow"]
        or state["error"] is not None
        or process.returncode != 0
    ):
        return ""
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def _terminate_probe(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.25)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.25)
    except (OSError, subprocess.TimeoutExpired):
        return


def _read_bounded_text(
    path: Path,
    *,
    maximum_bytes: int,
    errors: str = "strict",
) -> str | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= getattr(os, "O_NOFOLLOW")
    try:
        descriptor = os.open(path, flags)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None
        while True:
            chunk = os.read(descriptor, min(64 * 1024, maximum_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                return None
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_mode")
        ):
            return None
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        return b"".join(chunks).decode("utf-8", errors=errors)
    except UnicodeDecodeError:
        return None


def _scaled_bytes(value: float, unit: str) -> int:
    return int(value * (1024 ** {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit]))


def _strict(
    raw: Mapping[str, object], fields: set[str], label: str
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise CellContractError(f"{label} must be an object.")
    try:
        data = dict(raw)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CellContractError(f"{label} must be an object.") from exc
    if any(not isinstance(key, str) for key in data):
        raise CellContractError(f"{label} field names must be strings.")
    unknown, missing = sorted(set(data) - fields), sorted(fields - set(data))
    if unknown:
        raise CellContractError(f"Unknown {label} fields: {', '.join(unknown)}.")
    if missing:
        raise CellContractError(f"Missing {label} fields: {', '.join(missing)}.")
    return data


__all__ = [
    "ACCELERATOR_KINDS",
    "MAX_COMMAND_OUTPUT_BYTES",
    "MAX_PROC_TEXT_BYTES",
    "MEMORY_TOPOLOGIES",
    "ResourceSnapshot",
    "build_resource_snapshot",
    "collect_resource_snapshot",
    "resource_snapshot_from_payload",
]
