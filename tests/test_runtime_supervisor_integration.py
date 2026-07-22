"""Integration contract for a process-bound runtime supervisor.

The suite currently executes against the shipping deterministic oracle and
backend from ``experiments.runtime_supervisor_fakes``. A production adapter can be substituted directly
in ``run_scenario`` once its service API is available; no scenario depends on a
concrete process launcher, socket library, or persistence implementation.
"""

from __future__ import annotations

import unittest

from experiments.runtime_supervisor_fakes import (
    FailClosedSupervisorOracle,
    FakeRuntimeSupervisorBackend,
    RuntimeSupervisorBackend,
    RuntimeSupervisorDriver,
    SCENARIO_NAMES,
    run_scenario,
)


class ProcessBoundRuntimeIntegrationContractTests(unittest.TestCase):
    def test_fake_backend_and_oracle_implement_the_adapter_protocols(self) -> None:
        self.assertIsInstance(FakeRuntimeSupervisorBackend(), RuntimeSupervisorBackend)
        self.assertIsInstance(FailClosedSupervisorOracle(), RuntimeSupervisorDriver)

    def test_happy_path_requires_stable_process_listener_probe_and_binding(self) -> None:
        execution = run_scenario(FailClosedSupervisorOracle(), "happy_path")

        self.assertFalse(execution.false_ready)
        self.assertEqual(len(execution.outcomes), 1)
        self.assertTrue(execution.outcomes[0].ready)
        self.assertEqual(execution.outcomes[0].status, "ready")
        self.assertEqual(execution.outcomes[0].reason_codes, ())
        self.assertEqual(execution.spawn_count, 1)
        self.assertEqual(execution.call_log.count("observe_tree"), 2)
        self.assertEqual(execution.call_log.count("observe_listener"), 3)
        self.assertEqual(execution.call_log.count("binding_digest"), 3)

    def test_preexisting_port_is_never_adopted_or_spawned_over(self) -> None:
        execution = run_scenario(FailClosedSupervisorOracle(), "port_occupied")
        outcome = execution.outcomes[0]

        self.assertFalse(outcome.ready)
        self.assertEqual(outcome.status, "revoked")
        self.assertIn("port_occupied", outcome.reason_codes)
        self.assertEqual(execution.spawn_count, 0)
        self.assertNotIn("spawn", execution.call_log)
        self.assertNotIn("probe", execution.call_log)

    def test_pid_reuse_is_not_process_identity_reuse(self) -> None:
        world = FakeRuntimeSupervisorBackend()
        execution = run_scenario(
            FailClosedSupervisorOracle(),
            "restart_pid_reuse",
            backend=world,
        )
        outcome = execution.outcomes[0]

        self.assertFalse(outcome.ready)
        self.assertFalse(execution.false_ready)
        self.assertIn("process_identity_changed", outcome.reason_codes)
        self.assertIn("listener_owner_mismatch", outcome.reason_codes)
        self.assertEqual(outcome.status, "revoked")
        # Cleanup targets the original instance and must not kill a same-PID
        # replacement that has a different birth token.
        self.assertTrue(world.listener.bound)
        self.assertNotEqual(
            world.listener.owner,
            world.managed_root,
        )

    def test_listener_substitution_between_observations_never_becomes_ready(self) -> None:
        execution = run_scenario(
            FailClosedSupervisorOracle(),
            "port_substitution",
        )
        outcome = execution.outcomes[0]

        self.assertFalse(outcome.ready)
        self.assertFalse(execution.false_ready)
        self.assertIn("listener_owner_mismatch", outcome.reason_codes)
        self.assertIn("listener_substituted", outcome.reason_codes)
        self.assertEqual(outcome.cleanup_complete, True)

    def test_unexpected_descendant_invalidates_the_observed_tree(self) -> None:
        execution = run_scenario(
            FailClosedSupervisorOracle(),
            "unexpected_descendant",
        )
        outcome = execution.outcomes[0]

        self.assertFalse(outcome.ready)
        self.assertIn("unexpected_descendant", outcome.reason_codes)
        self.assertIn("process_tree_changed", outcome.reason_codes)
        self.assertEqual(outcome.status, "revoked")

    def test_binding_drift_between_resolution_and_readiness_is_fail_closed(self) -> None:
        execution = run_scenario(FailClosedSupervisorOracle(), "binding_drift")
        outcome = execution.outcomes[0]

        self.assertFalse(outcome.ready)
        self.assertFalse(execution.false_ready)
        self.assertIn("binding_drift", outcome.reason_codes)
        self.assertEqual(outcome.cleanup_complete, True)

    def test_cleanup_removes_the_managed_listener_and_reaches_stopped(self) -> None:
        world = FakeRuntimeSupervisorBackend()
        execution = run_scenario(
            FailClosedSupervisorOracle(),
            "cleanup",
            backend=world,
        )

        self.assertEqual(
            [item.status for item in execution.outcomes],
            ["ready", "stopped"],
        )
        self.assertTrue(execution.outcomes[0].ready)
        self.assertFalse(execution.outcomes[1].ready)
        self.assertEqual(execution.outcomes[1].cleanup_complete, True)
        self.assertFalse(world.listener.bound)
        self.assertEqual(execution.call_log.count("terminate"), 1)
        self.assertFalse(execution.false_ready)

    def test_cleanup_ambiguity_is_sticky_and_blocks_reentry_without_observation(self) -> None:
        world = FakeRuntimeSupervisorBackend()
        driver = FailClosedSupervisorOracle()

        ready = driver.start(world)
        world.make_cleanup_ambiguous()
        ambiguous = driver.stop(world)
        calls_after_ambiguity = tuple(world.call_log)
        second_start = driver.start(world)

        self.assertTrue(ready.ready)
        self.assertEqual(ambiguous.status, "unknown_blocking")
        self.assertTrue(ambiguous.sticky_ambiguity)
        self.assertIn("cleanup_unverified", ambiguous.reason_codes)
        self.assertEqual(second_start.status, "unknown_blocking")
        self.assertFalse(second_start.ready)
        self.assertIn("sticky_ambiguity", second_start.reason_codes)
        self.assertEqual(tuple(world.call_log), calls_after_ambiguity)

    def test_backend_failures_at_each_readiness_stage_have_zero_false_ready(self) -> None:
        stages = (
            "resolve",
            "binding_digest",
            "observe_listener",
            "spawn",
            "observe_tree",
            "probe",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                world = FakeRuntimeSupervisorBackend()
                world.inject_failure(stage)

                outcome = FailClosedSupervisorOracle().start(world)

                self.assertFalse(outcome.ready)
                self.assertIn("backend_failure", outcome.reason_codes)
                self.assertIn(outcome.status, {"revoked", "unknown_blocking"})

    def test_cleanup_failure_is_unknown_blocking_not_revoked(self) -> None:
        world = FakeRuntimeSupervisorBackend()
        world.schedule_after(
            "binding_digest",
            1,
            lambda backend: backend.drift_binding(),
        )
        world.inject_failure("terminate")

        outcome = FailClosedSupervisorOracle().start(world)

        self.assertFalse(outcome.ready)
        self.assertEqual(outcome.status, "unknown_blocking")
        self.assertTrue(outcome.sticky_ambiguity)
        self.assertIn("cleanup_unverified", outcome.reason_codes)

    def test_complete_contract_matrix_has_zero_false_ready(self) -> None:
        executions = {
            name: run_scenario(FailClosedSupervisorOracle(), name)
            for name in SCENARIO_NAMES
        }

        self.assertEqual(set(executions), set(SCENARIO_NAMES))
        self.assertEqual(
            [name for name, item in executions.items() if item.false_ready],
            [],
        )
        for name in (
            "port_occupied",
            "restart_pid_reuse",
            "port_substitution",
            "unexpected_descendant",
            "binding_drift",
        ):
            self.assertFalse(executions[name].outcomes[0].ready)


if __name__ == "__main__":
    unittest.main()
