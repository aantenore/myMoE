from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from local_moe.runtime_process_observer import (
    PsutilRuntimeProcessObserver,
    RuntimeProcessObservationError,
    normalize_numeric_loopback_host,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class _Address:
    ip: str
    port: int


@dataclass
class _Connection:
    laddr: _Address
    status: str
    pid: int | None


class _Process:
    def __init__(
        self,
        pid: int,
        executable: str,
        create_time: float,
        *,
        children: tuple[object, ...] = (),
        connections: tuple[_Connection, ...] = (),
    ) -> None:
        self.pid = pid
        self._executable = executable
        self._create_time = create_time
        self._children = children
        self._connections = connections
        self.children_calls = 0

    def create_time(self) -> float:
        return self._create_time

    def exe(self) -> str:
        return self._executable

    def children(self, recursive: bool = False):
        self.children_calls += 1
        if recursive is not True:
            raise AssertionError("observer must inspect the recursive tree")
        return list(self._children)

    def net_connections(self, *, kind: str):
        if kind != "tcp":
            raise AssertionError("observer must request TCP listeners only")
        return list(self._connections)


class _PsutilBackend:
    CONN_LISTEN = "LISTEN"

    class AccessDenied(Exception):
        pass

    def __init__(
        self,
        process: _Process,
        connections: list[_Connection] | None = None,
        deny_global: bool = False,
    ) -> None:
        self.process = process
        self.connections = connections or []
        self.process_calls: list[int] = []
        self.connection_calls = 0
        self.deny_global = deny_global

    def Process(self, pid: int) -> _Process:  # noqa: N802 - psutil API shape
        self.process_calls.append(pid)
        return self.process

    def net_connections(self, *, kind: str):
        self.connection_calls += 1
        if self.deny_global:
            raise self.AccessDenied("global listing denied")
        if kind != "tcp":
            raise AssertionError("observer must request TCP listeners only")
        return list(self.connections)


class RuntimeProcessObserverTests(unittest.TestCase):
    def test_psutil_is_lazy_and_process_evidence_is_root_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "llama-server"
            executable.write_bytes(b"pinned llama.cpp binary")
            process = _Process(4312, str(executable), 1_725_000_000.125)
            backend = _PsutilBackend(process)
            loads = 0

            def load_backend() -> object:
                nonlocal loads
                loads += 1
                return backend

            observer = PsutilRuntimeProcessObserver(psutil_loader=load_backend)
            self.assertEqual(loads, 0)

            evidence = observer.observe_process_tree(4312)

        self.assertEqual(loads, 1)
        self.assertEqual(evidence.root_pid, 4312)
        self.assertEqual(evidence.process_count, 1)
        self.assertTrue(evidence.root_only)
        self.assertEqual(evidence.pids_digest, _sha256_json({"pids": [4312]}))
        self.assertEqual(
            evidence.root_executable_sha256,
            hashlib.sha256(b"pinned llama.cpp binary").hexdigest(),
        )
        self.assertEqual(evidence.payload()["digest"], evidence.digest)
        self.assertNotIn(str(executable), json.dumps(evidence.payload()))
        self.assertEqual(process.children_calls, 2)

    def test_unexpected_descendant_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "llama-server"
            executable.write_bytes(b"pinned llama.cpp binary")
            child = type("Child", (), {"pid": 9991})()
            process = _Process(
                4312,
                str(executable),
                1_725_000_000.125,
                children=(child,),
            )
            observer = PsutilRuntimeProcessObserver(
                psutil_loader=lambda: _PsutilBackend(process)
            )

            with self.assertRaises(RuntimeProcessObservationError) as raised:
                observer.observe_process_tree(4312)

        self.assertEqual(raised.exception.code, "unexpected_descendant")

    def test_listener_is_owned_only_by_the_expected_root_pid(self) -> None:
        process = _Process(4312, "/unused/llama-server", 100.0)
        backend = _PsutilBackend(
            process,
            [
                _Connection(_Address("127.0.0.1", 8123), "LISTEN", 4312),
                _Connection(_Address("127.0.0.1", 8123), "ESTABLISHED", 9999),
                _Connection(_Address("127.0.0.1", 9000), "LISTEN", 9999),
            ],
        )
        observer = PsutilRuntimeProcessObserver(psutil_loader=lambda: backend)

        evidence = observer.observe_endpoint_ownership(
            host="127.0.0.1",
            port=8123,
            root_pid=4312,
        )

        self.assertEqual(evidence.listener_pids, (4312,))
        self.assertEqual(
            evidence.listener_pids_digest,
            _sha256_json({"pids": [4312]}),
        )
        self.assertTrue(evidence.owned_by_root)
        self.assertFalse(evidence.ambiguous)
        self.assertEqual(evidence.payload()["digest"], evidence.digest)

    def test_unknown_or_multiple_listener_owners_are_ambiguous(self) -> None:
        process = _Process(4312, "/unused/llama-server", 100.0)
        backend = _PsutilBackend(
            process,
            [
                _Connection(_Address("::1", 8123), "LISTEN", 4312),
                _Connection(_Address("::1", 8123), "LISTEN", 9911),
                _Connection(_Address("::1", 8123), "LISTEN", None),
            ],
        )
        observer = PsutilRuntimeProcessObserver(psutil_loader=lambda: backend)

        evidence = observer.observe_endpoint_ownership(
            host="::1",
            port=8123,
            root_pid=4312,
        )

        self.assertEqual(evidence.listener_pids, (4312, 9911))
        self.assertFalse(evidence.owned_by_root)
        self.assertTrue(evidence.ambiguous)

    def test_prelaunch_observation_reports_a_vacant_endpoint(self) -> None:
        backend = _PsutilBackend(_Process(4312, "/unused", 100.0))
        observer = PsutilRuntimeProcessObserver(psutil_loader=lambda: backend)

        evidence = observer.observe_endpoint_ownership(
            host="127.0.0.1",
            port=8123,
            root_pid=None,
        )

        self.assertEqual(evidence.listener_pids, ())
        self.assertFalse(evidence.owned_by_root)
        self.assertFalse(evidence.ambiguous)

    def test_access_denied_falls_back_to_bind_probe_and_owned_process_view(self) -> None:
        class _SocketProbe:
            def __init__(self) -> None:
                self.bound = None
                self.closed = False

            def bind(self, address) -> None:
                self.bound = address

            def close(self) -> None:
                self.closed = True

        probe = _SocketProbe()
        process = _Process(
            4312,
            "/unused/llama-server",
            100.0,
            connections=(
                _Connection(_Address("127.0.0.1", 8123), "LISTEN", None),
            ),
        )
        backend = _PsutilBackend(process, deny_global=True)
        observer = PsutilRuntimeProcessObserver(
            psutil_loader=lambda: backend,
            socket_factory=lambda *_args: probe,
            global_listener_reader=(
                lambda _host, _port: ()
                if not probe.closed
                else (4312,)
            ),
        )

        vacant = observer.observe_endpoint_ownership(
            host="127.0.0.1", port=8123, root_pid=None
        )
        owned = observer.observe_endpoint_ownership(
            host="127.0.0.1", port=8123, root_pid=4312
        )

        self.assertEqual(probe.bound, ("127.0.0.1", 8123))
        self.assertTrue(probe.closed)
        self.assertFalse(vacant.ambiguous)
        self.assertEqual(owned.listener_pids, (4312,))
        self.assertTrue(owned.owned_by_root)

    def test_access_denied_bind_failure_is_ambiguous_not_vacant(self) -> None:
        class _OccupiedProbe:
            def bind(self, _address) -> None:
                raise OSError("address unavailable")

            def close(self) -> None:
                return None

        backend = _PsutilBackend(
            _Process(4312, "/unused", 100.0), deny_global=True
        )
        observer = PsutilRuntimeProcessObserver(
            psutil_loader=lambda: backend,
            socket_factory=lambda *_args: _OccupiedProbe(),
            global_listener_reader=lambda _host, _port: (),
        )

        evidence = observer.observe_endpoint_ownership(
            host="127.0.0.1", port=8123, root_pid=None
        )

        self.assertTrue(evidence.ambiguous)
        self.assertEqual(evidence.listener_pids, ())

    def test_access_denied_global_fallback_detects_multiple_listener_pids(self) -> None:
        backend = _PsutilBackend(
            _Process(4312, "/unused", 100.0), deny_global=True
        )
        observer = PsutilRuntimeProcessObserver(
            psutil_loader=lambda: backend,
            global_listener_reader=lambda _host, _port: (4312, 9911),
        )

        evidence = observer.observe_endpoint_ownership(
            host="127.0.0.1", port=8123, root_pid=4312
        )

        self.assertEqual(evidence.listener_pids, (4312, 9911))
        self.assertFalse(evidence.owned_by_root)
        self.assertTrue(evidence.ambiguous)

    def test_hostnames_and_non_loopback_addresses_are_rejected(self) -> None:
        for host in ("localhost", "0.0.0.0", "192.168.1.20", "example.test"):
            with self.subTest(host=host), self.assertRaisesRegex(
                ValueError, "numeric loopback"
            ):
                normalize_numeric_loopback_host(host)

    def test_missing_psutil_is_a_stable_fail_closed_error(self) -> None:
        def unavailable() -> object:
            raise ImportError("not installed")

        observer = PsutilRuntimeProcessObserver(psutil_loader=unavailable)

        with self.assertRaises(RuntimeProcessObservationError) as raised:
            observer.observe_endpoint_ownership(
                host="127.0.0.1",
                port=8123,
                root_pid=None,
            )

        self.assertEqual(raised.exception.code, "psutil_unavailable")


if __name__ == "__main__":
    unittest.main()
