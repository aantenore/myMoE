from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from local_moe.llama_cpp_runtime_supervisor import (
    LlamaCppApplicationEvidence,
    LlamaCppRuntimeEvidence,
    LlamaCppRuntimeSupervisorError,
)
from local_moe.runtime_binding_inspector import ResolvedCellRuntimeLaunch
from local_moe.runtime_process_observer import (
    EndpointOwnershipEvidence,
    ProcessTreeEvidence,
)
from local_moe.runtime_supervisor import (
    RuntimeSupervisorError,
    build_llama_cpp_runtime_spec,
    build_runtime_supervisor,
)
from local_moe.runtime_supervisor_store import SQLiteRuntimeSupervisorLeaseStore


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _runtime_evidence() -> LlamaCppRuntimeEvidence:
    process = ProcessTreeEvidence(
        root_pid=4321,
        create_time_ns=1_725_000_000_000_000_000,
        process_count=1,
        pids_digest=_sha({"pids": [4321]}),
        root_executable_sha256="a" * 64,
        root_only=True,
    )
    endpoint = EndpointOwnershipEvidence(
        host="127.0.0.1",
        port=8188,
        listener_pids=(4321,),
        listener_pids_digest=_sha({"pids": [4321]}),
        owned_by_root=True,
        ambiguous=False,
    )
    application = LlamaCppApplicationEvidence(
        ready=True,
        model_ids=("local_coder",),
        models_digest=_sha({"model_ids": ["local_coder"]}),
        props_digest="b" * 64,
        reason_codes=(),
    )
    return LlamaCppRuntimeEvidence(
        process=process,
        endpoint=endpoint,
        application=application,
    )


class _FakeAdapter:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        cleanup_unknown: bool = False,
    ) -> None:
        self.evidence = _runtime_evidence()
        self.start_error = start_error
        self._cleanup_unknown = cleanup_unknown
        self.owns_process = False
        self.calls: list[str] = []

    @property
    def cleanup_unknown(self) -> bool:
        return self._cleanup_unknown

    def start(self, *, on_process_launched=None, on_process_observed=None):
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        self.owns_process = True
        if on_process_launched is not None:
            on_process_launched()
        if on_process_observed is not None:
            on_process_observed(self.evidence.process)
        return self.evidence

    def inspect(self):
        self.calls.append("inspect")
        if not self.owns_process:
            raise LlamaCppRuntimeSupervisorError("process_exited", "exited")
        return self.evidence

    def stop(self) -> None:
        self.calls.append("stop")
        if self._cleanup_unknown:
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_unverified", "cleanup unknown"
            )
        self.owns_process = False


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class RuntimeSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.request = self.root / "binding.json"
        self.request.write_text("{}", encoding="utf-8")
        self.executable = self.root / "runtime" / "bin" / "llama-server"
        self.model = self.root / "models" / "local-coder.gguf"
        self.executable.parent.mkdir(parents=True)
        self.model.parent.mkdir(parents=True)
        self.executable.write_bytes(b"runtime")
        self.model.write_bytes(b"GGUF-model")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def resolved(self, *, manifest_digest: str = "2" * 64):
        manifest = SimpleNamespace(
            config_source_sha256="3" * 64,
            runtime_config_sha256="4" * 64,
            runtime_identity_sha256="5" * 64,
        )
        bundle = SimpleNamespace(manifest=manifest)
        argv = (
            "runtime/bin/llama-server",
            "-m",
            "models/local-coder.gguf",
            "--alias",
            "local_coder",
            "--host",
            "127.0.0.1",
            "--port",
            "8188",
            "--offline",
            "--no-ui",
            "--no-ui-mcp-proxy",
            "--no-agent",
            "--no-slots",
            "--fit",
            "off",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
        )
        return ResolvedCellRuntimeLaunch(
            request_path=self.request,
            working_directory=self.root,
            runtime_executable_path=self.executable,
            model_artifact_path=self.model,
            argv=argv,
            backend="llama_cpp",
            runtime_security_profile="process_bound_v1",
            expert_id="local_coder",
            endpoint_base_url="http://127.0.0.1:8188/v1",
            endpoint_host="127.0.0.1",
            endpoint_port=8188,
            expected_model_id="local_coder",
            request_sha256="1" * 64,
            binding_manifest_sha256=manifest_digest,
            inspection_receipt_sha256="6" * 64,
            launch_plan_sha256="7" * 64,
            endpoint_authority_sha256="8" * 64,
            runtime_executable_sha256="a" * 64,
            model_identity_sha256="9" * 64,
            bundle=bundle,
        )

    def store(self) -> SQLiteRuntimeSupervisorLeaseStore:
        return SQLiteRuntimeSupervisorLeaseStore(
            self.root / "state" / "leases.sqlite3",
            sentinel_root=self.root / "state" / "owners",
        )

    def test_spec_builder_reproduces_only_the_exact_bound_argv(self) -> None:
        resolved = self.resolved()
        spec = build_llama_cpp_runtime_spec(resolved)
        normalized = list(resolved.argv)
        normalized[0] = str(self.executable)
        normalized[2] = str(self.model)

        self.assertEqual(spec.argv(), tuple(normalized))
        with self.assertRaises(RuntimeSupervisorError):
            build_llama_cpp_runtime_spec(
                replace(resolved, argv=resolved.argv + ("--metrics",))
            )

    def test_happy_lifecycle_binds_application_evidence_and_two_mutations(self) -> None:
        resolved = self.resolved()
        adapter = _FakeAdapter()
        store = self.store()
        session = build_runtime_supervisor(
            self.request,
            lease_store=store,
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )

        started = session.start()
        inspected = session.inspect()
        final = session.stop()

        self.assertEqual(started.state, "ready")
        self.assertEqual(inspected.state, "ready")
        self.assertEqual(final.state, "stopped")
        self.assertEqual(final.lifecycle_operations, 2)
        self.assertTrue(final.process_mutations)
        self.assertEqual(final.model_invocations, 0)
        self.assertFalse(final.supervisor_remote_egress)
        self.assertEqual(final.runtime_egress_attestation, "not_observed")
        self.assertTrue(final.offline_launch_profile)
        self.assertFalse(final.authorizes_inference)
        self.assertEqual(
            final.application_evidence_sha256,
            adapter.evidence.application.digest,
        )
        self.assertEqual(
            [receipt.state for receipt in session._lease_chain],  # noqa: SLF001
            ["prepared", "starting", "ready", "stopping", "stopped"],
        )
        self.assertEqual(store.list_active(), ())
        self.assertFalse(session.cleanup_unknown)

    def test_start_reobserves_runtime_after_the_post_ready_full_hash(self) -> None:
        resolved = self.resolved()
        adapter = _FakeAdapter()
        original_inspect = adapter.inspect

        def changed_inspect():
            evidence = original_inspect()
            adapter.evidence = replace(
                evidence,
                process=replace(
                    evidence.process,
                    create_time_ns=evidence.process.create_time_ns + 1,
                ),
            )
            return adapter.evidence

        adapter.inspect = changed_inspect  # type: ignore[method-assign]
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )

        with self.assertRaises(RuntimeSupervisorError) as raised:
            session.start()
        final = session.stop()

        self.assertEqual(raised.exception.code, "process_identity_changed")
        self.assertEqual(final.state, "stopped")
        self.assertEqual(adapter.calls, ["start", "inspect", "stop"])

    def test_post_ready_binding_drift_revokes_then_stops_owned_process(self) -> None:
        first = self.resolved()
        drifted = self.resolved(manifest_digest="f" * 64)
        resolutions = iter((first, first, drifted))
        adapter = _FakeAdapter()
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=lambda _path: next(resolutions),
            adapter_factory=lambda _spec: adapter,
        )

        with self.assertRaises(RuntimeSupervisorError) as raised:
            session.start()
        final = session.stop()

        self.assertEqual(raised.exception.code, "binding_changed")
        self.assertEqual(final.state, "stopped")
        self.assertIn("binding_changed", final.reason_codes)
        self.assertEqual(final.lifecycle_operations, 2)
        self.assertEqual(adapter.calls, ["start", "stop"])

    def test_binding_drift_discovered_during_stop_is_recorded_after_cleanup(self) -> None:
        first = self.resolved()
        drifted = self.resolved(manifest_digest="f" * 64)
        resolutions = iter((first, first, first, drifted))
        adapter = _FakeAdapter()
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=lambda _path: next(resolutions),
            adapter_factory=lambda _spec: adapter,
        )

        session.start()
        final = session.stop()

        self.assertEqual(final.state, "stopped")
        self.assertIn("binding_changed", final.reason_codes)
        self.assertFalse(session.cleanup_unknown)
        self.assertEqual(final.lifecycle_operations, 2)
        self.assertEqual(adapter.calls, ["start", "inspect", "stop"])

    def test_periodic_inspection_uses_stat_then_bounded_full_recheck(self) -> None:
        resolved = self.resolved()
        calls = 0

        def resolver(_path):
            nonlocal calls
            calls += 1
            return resolved

        clock = _Clock()
        adapter = _FakeAdapter()
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=resolver,
            adapter_factory=lambda _spec: adapter,
            static_recheck_interval_seconds=300,
            monotonic=clock,
        )
        session.start()
        self.assertEqual(calls, 3)

        clock.value = 10
        session.inspect()
        self.assertEqual(calls, 3)
        clock.value = 301
        session.inspect()
        self.assertEqual(calls, 4)
        session.stop()

    def test_model_stat_drift_forces_full_recheck_and_revocation(self) -> None:
        resolved = self.resolved()
        calls = 0

        def resolver(_path):
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise RuntimeSupervisorError("binding_changed")
            return resolved

        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=resolver,
            adapter_factory=lambda _spec: _FakeAdapter(),
        )
        session.start()
        self.model.write_bytes(b"GGUF-drifted-model")

        with self.assertRaises(RuntimeSupervisorError) as raised:
            session.inspect()
        final = session.stop()

        self.assertEqual(raised.exception.code, "binding_changed")
        self.assertEqual(final.state, "stopped")
        self.assertIn("binding_changed", final.reason_codes)

    def test_periodic_full_hash_is_followed_by_fresh_runtime_evidence(self) -> None:
        resolved = self.resolved()
        clock = _Clock()
        adapter = _FakeAdapter()
        original_inspect = adapter.inspect
        inspections = 0

        def changing_inspect():
            nonlocal inspections
            inspections += 1
            evidence = original_inspect()
            if inspections == 3:
                adapter.evidence = replace(
                    evidence,
                    process=replace(
                        evidence.process,
                        create_time_ns=evidence.process.create_time_ns + 1,
                    ),
                )
            return adapter.evidence

        adapter.inspect = changing_inspect  # type: ignore[method-assign]
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
            static_recheck_interval_seconds=300,
            monotonic=clock,
        )
        session.start()
        clock.value = 301

        with self.assertRaises(RuntimeSupervisorError) as raised:
            session.inspect()
        session.stop()

        self.assertEqual(raised.exception.code, "process_identity_changed")
        self.assertEqual(inspections, 3)

    def test_prelaunch_collision_releases_lease_without_process_mutation(self) -> None:
        resolved = self.resolved()
        adapter = _FakeAdapter(
            start_error=LlamaCppRuntimeSupervisorError(
                "endpoint_in_use", "occupied"
            )
        )
        store = self.store()
        session = build_runtime_supervisor(
            self.request,
            lease_store=store,
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )

        with self.assertRaises(RuntimeSupervisorError) as raised:
            session.start()
        final = session.stop()

        self.assertEqual(raised.exception.code, "endpoint_in_use")
        self.assertEqual(final.state, "stopped")
        self.assertEqual(final.lifecycle_operations, 0)
        self.assertFalse(final.process_mutations)
        self.assertIn("endpoint_already_occupied", final.reason_codes)
        self.assertEqual(store.list_active(), ())

    def test_cleanup_ambiguity_is_sticky_and_never_reports_stopped(self) -> None:
        resolved = self.resolved()
        adapter = _FakeAdapter(
            start_error=LlamaCppRuntimeSupervisorError(
                "cleanup_unverified", "unknown"
            ),
            cleanup_unknown=True,
        )
        store = self.store()
        session = build_runtime_supervisor(
            self.request,
            lease_store=store,
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )

        with self.assertRaises(RuntimeSupervisorError):
            session.start()
        first = session.stop()
        second = session.stop()

        self.assertEqual(first, second)
        self.assertEqual(first.state, "unknown_blocking")
        self.assertIn("cleanup_unverified", first.reason_codes)
        self.assertTrue(session.cleanup_unknown)
        self.assertEqual(store.list_active()[0].state, "unknown_blocking")

    def test_ledger_failure_never_skips_owned_process_teardown(self) -> None:
        resolved = self.resolved()
        adapter = _FakeAdapter()
        store = self.store()
        session = build_runtime_supervisor(
            self.request,
            lease_store=store,
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )
        session.start()
        original_transition = store.transition

        def failing_transition(handle, target_state, **kwargs):
            if target_state == "stopping":
                raise RuntimeError("synthetic ledger failure")
            return original_transition(handle, target_state, **kwargs)

        with mock.patch.object(store, "transition", side_effect=failing_transition):
            final = session.stop()

        self.assertIn("stop", adapter.calls)
        self.assertFalse(adapter.owns_process)
        self.assertEqual(final.state, "unknown_blocking")
        self.assertTrue(session.cleanup_unknown)

    def test_spawn_before_first_observation_is_receipted_as_two_mutations(self) -> None:
        class _LaunchThenFailAdapter(_FakeAdapter):
            def start(self, *, on_process_launched=None, on_process_observed=None):
                del on_process_observed
                self.calls.append("start")
                self.owns_process = True
                if on_process_launched is not None:
                    on_process_launched()
                self.owns_process = False
                raise LlamaCppRuntimeSupervisorError(
                    "process_observation_failed", "synthetic observation failure"
                )

        resolved = self.resolved()
        adapter = _LaunchThenFailAdapter()
        session = build_runtime_supervisor(
            self.request,
            lease_store=self.store(),
            resolver=lambda _path: resolved,
            adapter_factory=lambda _spec: adapter,
        )

        with self.assertRaises(RuntimeSupervisorError):
            session.start()
        final = session.stop()

        self.assertEqual(final.lifecycle_operations, 2)
        self.assertTrue(final.process_mutations)

    def test_adapter_factory_failure_releases_prepared_lease(self) -> None:
        store = self.store()
        resolved = self.resolved()

        with self.assertRaises(RuntimeSupervisorError):
            build_runtime_supervisor(
                self.request,
                lease_store=store,
                resolver=lambda _path: resolved,
                adapter_factory=lambda _spec: (_ for _ in ()).throw(
                    ValueError("factory failed")
                ),
            )

        self.assertEqual(store.list_active(), ())


if __name__ == "__main__":
    unittest.main()
