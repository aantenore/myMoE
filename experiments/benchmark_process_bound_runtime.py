"""Deterministic zero-false-ready benchmark for runtime supervision.

The default driver is the test oracle, so this benchmark is a contract baseline
rather than production evidence.  Pass another ``driver_factory`` to
``run_benchmark`` to exercise a production adapter through the same scenarios.
No scenario starts a process, opens a socket, invokes a model, or writes a file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Callable, Iterator
from unittest import mock
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "outputs" / "process-bound-runtime-contract.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime_supervisor_fakes import (  # noqa: E402
    FailClosedSupervisorOracle,
    RuntimeSupervisorDriver,
    SCENARIO_NAMES,
    ScenarioExecution,
    run_scenario,
)


DriverFactory = Callable[[], RuntimeSupervisorDriver]
ADVERSARIAL_SCENARIOS = (
    "port_occupied",
    "restart_pid_reuse",
    "port_substitution",
    "unexpected_descendant",
    "binding_drift",
)
GUARDED_SIDE_EFFECTS = (
    "network_socket",
    "process_spawn",
    "subprocess_run",
    "url_fetch",
)


@contextmanager
def _side_effect_guard() -> Iterator[None]:
    blocked = AssertionError(
        "The process-bound runtime benchmark attempted a real side effect."
    )
    with (
        mock.patch.object(socket, "socket", side_effect=blocked),
        mock.patch.object(socket, "create_connection", side_effect=blocked),
        mock.patch.object(subprocess, "Popen", side_effect=blocked),
        mock.patch.object(subprocess, "run", side_effect=blocked),
        mock.patch.object(urllib.request, "urlopen", side_effect=blocked),
    ):
        yield


def _run_matrix(driver_factory: DriverFactory) -> dict[str, ScenarioExecution]:
    with _side_effect_guard():
        return {
            name: run_scenario(driver_factory(), name)
            for name in SCENARIO_NAMES
        }


def _checks(scenarios: dict[str, ScenarioExecution]) -> dict[str, bool]:
    happy = scenarios["happy_path"]
    occupied = scenarios["port_occupied"]
    pid_reuse = scenarios["restart_pid_reuse"]
    substituted = scenarios["port_substitution"]
    descendant = scenarios["unexpected_descendant"]
    binding = scenarios["binding_drift"]
    cleanup = scenarios["cleanup"]
    sticky = scenarios["sticky_ambiguity"]

    return {
        "happy_path_ready": (
            len(happy.outcomes) == 1
            and happy.outcomes[0].ready
            and happy.outcomes[0].status == "ready"
        ),
        "port_occupied_never_spawned": (
            occupied.spawn_count == 0
            and not occupied.outcomes[0].ready
            and "port_occupied" in occupied.outcomes[0].reason_codes
        ),
        "pid_reuse_blocked": (
            not pid_reuse.outcomes[0].ready
            and "process_identity_changed" in pid_reuse.outcomes[0].reason_codes
        ),
        "port_substitution_blocked": (
            not substituted.outcomes[0].ready
            and "listener_substituted" in substituted.outcomes[0].reason_codes
        ),
        "unexpected_descendant_blocked": (
            not descendant.outcomes[0].ready
            and "unexpected_descendant" in descendant.outcomes[0].reason_codes
        ),
        "binding_drift_blocked": (
            not binding.outcomes[0].ready
            and "binding_drift" in binding.outcomes[0].reason_codes
        ),
        "cleanup_verified": (
            [item.status for item in cleanup.outcomes] == ["ready", "stopped"]
            and cleanup.outcomes[-1].cleanup_complete is True
        ),
        "cleanup_ambiguity_is_sticky": (
            [item.status for item in sticky.outcomes]
            == ["ready", "unknown_blocking", "unknown_blocking"]
            and not sticky.outcomes[-1].ready
            and "sticky_ambiguity" in sticky.outcomes[-1].reason_codes
        ),
        "zero_false_ready": all(
            not scenario.false_ready for scenario in scenarios.values()
        ),
    }


def run_benchmark(
    driver_factory: DriverFactory = FailClosedSupervisorOracle,
) -> dict[str, object]:
    """Run the deterministic matrix with an injected supervisor driver."""

    scenarios = _run_matrix(driver_factory)
    checks = _checks(scenarios)
    false_ready = [
        name for name, scenario in scenarios.items() if scenario.false_ready
    ]
    return {
        "schema_version": "process-bound-runtime-benchmark/v1",
        "benchmark": "process_bound_runtime_zero_false_ready",
        "driver": getattr(driver_factory, "__name__", type(driver_factory).__name__),
        "production_evidence": False,
        "side_effects_performed": False,
        "guarded_side_effects": list(GUARDED_SIDE_EFFECTS),
        "integration_seam": [
            "resolve",
            "spawn",
            "observe_tree",
            "observe_listener",
            "probe",
            "terminate",
            "binding_digest",
        ],
        "scenario_count": len(scenarios),
        "adversarial_scenario_count": len(ADVERSARIAL_SCENARIOS),
        "false_ready_count": len(false_ready),
        "false_ready_scenarios": false_ready,
        "checks": checks,
        "contract_checks_passed": all(checks.values()),
        "scenarios": {
            name: scenarios[name].payload() for name in SCENARIO_NAMES
        },
        "residual_limits": [
            "Observed process trees are not a kernel containment boundary.",
            "Loopback ownership does not prove that the runtime cannot proxy outward.",
            "Process and listener identity do not attest semantic model behavior.",
        ],
    }


def render_report(report: dict[str, object]) -> str:
    return json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the regenerated report byte-for-byte against the artifact",
    )
    parser.add_argument("--out", help="Optional report path to write or verify.")
    args = parser.parse_args(argv)
    report = run_benchmark()
    rendered = render_report(report)
    destination = Path(args.out) if args.out else DEFAULT_ARTIFACT
    if args.check:
        try:
            current = destination.read_bytes()
        except OSError as exc:
            raise SystemExit("Unable to read process-bound runtime artifact.") from exc
        if current != rendered.encode("utf-8"):
            raise SystemExit("Process-bound runtime benchmark artifact is out of date.")
    elif args.out:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered.encode("utf-8"))
    else:
        print(rendered, end="")
    if not report["contract_checks_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
