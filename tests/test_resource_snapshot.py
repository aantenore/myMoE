from unittest.mock import patch
import os
import subprocess
import sys
import time
import unittest

from local_moe.cell_contracts import CellContractError
from local_moe.resource_snapshot import (
    MAX_COMMAND_OUTPUT_BYTES,
    ResourceSnapshot,
    _read_darwin_memory,
    _read_windows_memory,
    _read_linux_memory,
    _linux_cgroup_memory,
    _run_readonly,
    build_resource_snapshot,
    resource_snapshot_from_payload,
)
from local_moe.verified_routing_contracts import sha256_json


GIB = 1024**3
CPU_SHA = sha256_json({"cpu": "fixture"})
ACCELERATOR_SHA = sha256_json({"accelerator": "fixture"})
RUNTIME_SHA = sha256_json({"runtime": "fixture"})


def apple_snapshot(**overrides: object) -> ResourceSnapshot:
    values: dict[str, object] = {
        "system": "Darwin",
        "os_release": "25.0.0",
        "machine": "aarch64",
        "cpu_count": 12,
        "cpu_identity_sha256": CPU_SHA,
        "memory_topology": "unified",
        "total_memory_bytes": 24 * GIB,
        "available_memory_bytes": 16 * GIB,
        "effective_memory_limit_bytes": 24 * GIB,
        "swap_used_bytes": 0,
        "accelerator_kind": "integrated",
        "accelerator_identity_sha256": ACCELERATOR_SHA,
        "runtime_environment_sha256": RUNTIME_SHA,
        "captured_at": "2026-07-21T10:00:00+00:00",
        "source": {"fixture": "apple"},
    }
    values.update(overrides)
    return build_resource_snapshot(**values)  # type: ignore[arg-type]


class ResourceSnapshotTests(unittest.TestCase):
    def test_apple_uses_unified_memory_without_synthetic_vram(self) -> None:
        snapshot = apple_snapshot()
        self.assertEqual(snapshot.machine, "arm64")
        self.assertIsNone(snapshot.accelerator_memory_total_bytes)
        with self.assertRaisesRegex(CellContractError, "accelerator VRAM"):
            apple_snapshot(accelerator_memory_total_bytes=8 * GIB)

    def test_resource_class_binds_stable_os_cpu_accelerator_limit_and_runtime(
        self,
    ) -> None:
        first = apple_snapshot()
        second = apple_snapshot(
            available_memory_bytes=10 * GIB,
            captured_at="2026-07-21T10:01:00+00:00",
        )
        self.assertEqual(first.resource_class_sha256, second.resource_class_sha256)
        self.assertNotEqual(first.digest, second.digest)
        payload = first.resource_class_payload()
        self.assertEqual(payload["os_release"], "25.0.0")
        self.assertEqual(payload["cpu_identity_sha256"], CPU_SHA)
        self.assertEqual(payload["accelerator_identity_sha256"], ACCELERATOR_SHA)
        self.assertEqual(payload["runtime_environment_sha256"], RUNTIME_SHA)

    def test_topology_kind_identity_and_vram_contradictions_are_rejected(self) -> None:
        with self.assertRaises(CellContractError):
            apple_snapshot(accelerator_kind="none")
        with self.assertRaises(CellContractError):
            apple_snapshot(accelerator_identity_sha256=None)
        with self.assertRaises(CellContractError):
            apple_snapshot(memory_topology="dedicated", accelerator_kind="discrete")
        with self.assertRaises(CellContractError):
            build_resource_snapshot(
                system="Linux",
                os_release="6.12",
                machine="x86_64",
                cpu_count=8,
                cpu_identity_sha256=CPU_SHA,
                memory_topology="system",
                total_memory_bytes=16 * GIB,
                available_memory_bytes=8 * GIB,
                effective_memory_limit_bytes=16 * GIB,
                swap_used_bytes=0,
                accelerator_kind="none",
                accelerator_identity_sha256=ACCELERATOR_SHA,
                runtime_environment_sha256=RUNTIME_SHA,
                captured_at="2026-07-21T10:00:00+00:00",
                source={"fixture": True},
            )

    def test_windows_does_not_mislabel_commit_charge_as_swap(self) -> None:
        with patch("local_moe.resource_snapshot.ctypes.windll", create=True) as windll:
            windll.kernel32.GlobalMemoryStatusEx.return_value = True
            observed = _read_windows_memory()
        self.assertIsNone(observed["swap_used_bytes"])

    @unittest.skipIf(os.name == "nt", "macOS probes use POSIX pipe semantics")
    def test_macos_probe_requires_absolute_path_minimal_env_and_bounded_output(
        self,
    ) -> None:
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('fixture\\n')",
        )
        with patch(
            "local_moe.resource_snapshot.subprocess.Popen",
            wraps=subprocess.Popen,
        ) as popen:
            self.assertEqual(_run_readonly(command), "fixture")
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["env"], {"LC_ALL": "C", "LANG": "C"})
        self.assertEqual(popen.call_args.args[0][0], sys.executable)
        with self.assertRaises(CellContractError):
            _run_readonly(("sysctl", "-n", "hw.memsize"))
        started = time.monotonic()
        overflow = _run_readonly(
            (
                sys.executable,
                "-c",
                (
                    "import sys,time;"
                    f"sys.stdout.write('x'*{MAX_COMMAND_OUTPUT_BYTES + 1});"
                    "sys.stdout.flush();time.sleep(10)"
                ),
            )
        )
        self.assertEqual(overflow, "")
        self.assertLess(time.monotonic() - started, 3.0)
        failed = _run_readonly(
            (
                sys.executable,
                "-c",
                "import sys;sys.stdout.write('999');sys.exit(7)",
            )
        )
        self.assertEqual(failed, "")

    def test_macos_available_memory_excludes_purgeable_pages(self) -> None:
        vm_stat = (
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
            "Pages free: 1.\n"
            "Pages inactive: 2.\n"
            "Pages speculative: 3.\n"
            "Pages purgeable: 100.\n"
        )
        with patch(
            "local_moe.resource_snapshot._run_readonly",
            side_effect=("1048576", vm_stat, "total = 0.00M used = 0.00M free = 0.00M"),
        ):
            observed = _read_darwin_memory()
        self.assertEqual(observed["available_memory_bytes"], 6 * 4096)

    def test_cgroup_v2_resolves_subgroup_mount_offset_and_parent_bottleneck(
        self,
    ) -> None:
        cgroup = "0::/tenant/job\n"
        mountinfo = (
            "29 23 0:26 /tenant /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n"
        )
        values = {
            "/proc/self/cgroup": cgroup,
            "/proc/self/mountinfo": mountinfo,
            "/sys/fs/cgroup/job/memory.max": str(8 * GIB),
            "/sys/fs/cgroup/job/memory.current": str(2 * GIB),
            "/sys/fs/cgroup/memory.max": str(6 * GIB),
            "/sys/fs/cgroup/memory.current": str(5 * GIB),
        }

        def read(path, **_: object):
            return values.get(str(path))

        with patch("local_moe.resource_snapshot._read_bounded_text", side_effect=read):
            limit, available, mode = _linux_cgroup_memory()
        self.assertEqual((limit, available, mode), (6 * GIB, GIB, "v2"))

    def test_cgroup_v1_process_membership_and_unbounded_parent(self) -> None:
        cgroup = "5:memory:/tenant/job\n4:cpu:/tenant/job\n"
        mountinfo = (
            "30 23 0:27 /tenant /sys/fs/cgroup/memory rw,nosuid "
            "- cgroup cgroup rw,memory\n"
        )
        values = {
            "/proc/self/cgroup": cgroup,
            "/proc/self/mountinfo": mountinfo,
            "/sys/fs/cgroup/memory/job/memory.limit_in_bytes": str(4 * GIB),
            "/sys/fs/cgroup/memory/job/memory.usage_in_bytes": str(GIB),
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(1 << 62),
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(2 * GIB),
        }

        def read(path, **_: object):
            return values.get(str(path))

        with patch("local_moe.resource_snapshot._read_bounded_text", side_effect=read):
            self.assertEqual(_linux_cgroup_memory(), (4 * GIB, 3 * GIB, "v1"))

    def test_cgroup_malformed_missing_controller_or_ambiguous_is_unknown(self) -> None:
        cases = (
            (
                "malformed\n",
                "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            ),
            (
                "5:cpu:/job\n",
                "30 23 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
            ),
            (
                "0::/job\n",
                (
                    "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
                    "31 23 0:28 / /sys/fs/cgroup-alt rw - cgroup2 cgroup rw\n"
                ),
            ),
            (
                "0::/job\n",
                "29 23 0:26 relative /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
            ),
        )
        for cgroup, mountinfo in cases:
            with self.subTest(cgroup=cgroup, mountinfo=mountinfo):

                def read(path, **_: object):
                    return {
                        "/proc/self/cgroup": cgroup,
                        "/proc/self/mountinfo": mountinfo,
                    }.get(str(path))

                with patch(
                    "local_moe.resource_snapshot._read_bounded_text",
                    side_effect=read,
                ):
                    self.assertEqual(
                        _linux_cgroup_memory(),
                        (None, None, "unknown"),
                    )

    def test_cgroup_zero_is_preserved_and_probe_failure_is_unknown(self) -> None:
        meminfo = "MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
        with (
            patch(
                "local_moe.resource_snapshot._read_bounded_text", return_value=meminfo
            ),
            patch(
                "local_moe.resource_snapshot._linux_cgroup_memory",
                return_value=(0, 0, "v2"),
            ),
        ):
            zero = _read_linux_memory()
        self.assertEqual(zero["effective_memory_limit_bytes"], 0)
        self.assertEqual(zero["available_memory_bytes"], 0)
        with (
            patch(
                "local_moe.resource_snapshot._read_bounded_text", return_value=meminfo
            ),
            patch(
                "local_moe.resource_snapshot._linux_cgroup_memory",
                return_value=(None, None, "unknown"),
            ),
        ):
            unknown = _read_linux_memory()
        self.assertIsNone(unknown["effective_memory_limit_bytes"])

    def test_cgroup_v2_missing_falls_back_to_v1_or_unknown(self) -> None:
        with patch("local_moe.resource_snapshot._read_bounded_text", return_value=None):
            self.assertEqual(_linux_cgroup_memory(), (None, None, "unknown"))

    def test_payload_boundary_is_strict_and_tamper_evident(self) -> None:
        snapshot = apple_snapshot()
        self.assertEqual(resource_snapshot_from_payload(snapshot.payload()), snapshot)
        tampered = snapshot.payload()
        tampered["os_release"] = "changed"
        with self.assertRaises(CellContractError):
            resource_snapshot_from_payload(tampered)
        unknown = snapshot.payload()
        unknown["future"] = True
        with self.assertRaises(CellContractError):
            resource_snapshot_from_payload(unknown)


if __name__ == "__main__":
    unittest.main()
