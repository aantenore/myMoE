from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from typing import Any, Iterator
from unittest import mock
import urllib.request

from local_moe import secure_files
from local_moe.cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellDeclaration,
)
from local_moe.cell_passport import build_cell_passport
from local_moe.runtime_binding_inspector import (
    ADAPTER_ID,
    CellBindingInspectionBundle,
    inspect_cell_binding,
)


CAPTURED_AT = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
FRESH_CAPTURED_AT = CAPTURED_AT + timedelta(seconds=5)
STREAM_READ_BOUND_BYTES = 64 * 1024
MODEL_BYTES = (b"GGUF-bound-cell-streaming-fixture-v1\n" * 4_096) + b"end\n"
RUNTIME_BYTES = b"runtime-executable-v1\n"
DRIVER_BYTES = b"runtime-driver-v1\n"
HARNESS_BYTES = b"cell-harness-v1\n"


def _write_json(path: Path, payload: object) -> None:
    """Write fixture JSON as platform-neutral UTF-8 bytes."""

    path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.request_path = root / "inspect.json"
        self.config_path = root / "runtime.json"
        self.catalog_path = root / "catalog.json"
        self.runtime_root = root / "runtime"
        self.model_root = root / "models"
        self.runtime_executable = self.runtime_root / "bin" / "runtime"
        self.driver = self.runtime_root / "lib" / "driver.py"
        self.harness = self.runtime_root / "lib" / "harness.py"
        self.model = self.model_root / "coder.gguf"

        self.runtime_executable.parent.mkdir(parents=True)
        self.driver.parent.mkdir(parents=True)
        self.model_root.mkdir()
        self.runtime_executable.write_bytes(RUNTIME_BYTES)
        self.driver.write_bytes(DRIVER_BYTES)
        self.harness.write_bytes(HARNESS_BYTES)
        self.model.write_bytes(MODEL_BYTES)

        self.config: dict[str, object] = {
            "execution": {
                "max_scope": "device_only",
                "allowed_scopes": ["device_only"],
                "allow_scope_widening": False,
            },
            "routing": {
                "top_k": 1,
                "fallback_order": ["coder"],
                "aggregation": "best",
            },
            "experts": [self._expert("coder", "models/coder.gguf", 8123)],
            "rules": [],
        }
        self.request: dict[str, object] = {
            "schema_version": "1.0",
            "contract": "CellBindingInspectRequest",
            "cell_id": "coder-local",
            "expert_id": "coder",
            "adapter_id": ADAPTER_ID,
            "catalog_path": "catalog.json",
            "runtime_config_path": "runtime.json",
            "runtime_root": "runtime",
            "model_artifact_root": "models",
            "runtime_components": [
                {"role": "harness", "path": "lib/harness.py"},
                {"role": "runtime_executable", "path": "bin/runtime"},
                {"role": "driver", "path": "lib/driver.py"},
            ],
            "observation_ttl_seconds": 60,
            "hash_limits": {
                "max_files": 32,
                "max_total_bytes": 2 * 1024 * 1024,
                "max_depth": 8,
                "max_file_bytes": 1024 * 1024,
            },
        }
        self.write_config()
        _write_json(self.request_path, self.request)
        self.write_catalog()

    @staticmethod
    def _expert(expert_id: str, model: str, port: int) -> dict[str, object]:
        return {
            "id": expert_id,
            "provider": "openai_compatible",
            "model": model,
            "role": "coding",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "params": {
                "runtime_backend": "llama_cpp",
                "runtime_model_source": "local",
                "runtime_executable": "runtime/bin/runtime",
            },
            "execution": {
                "scope": "device_only",
                "transport": "direct_local",
            },
        }

    def write_config(self) -> None:
        _write_json(self.config_path, self.config)

    def write_catalog(
        self,
        expected: dict[str, str | None] | None = None,
    ) -> None:
        expected = expected or {}
        declaration = CellDeclaration(
            cell_id="coder-local",
            model="synthetic-coder",
            quantization="fixture",
            runtime="llama_cpp",
            harness="mymoe",
            capabilities=("coding",),
            tool_surfaces=(),
            risk_classes=("compute_only",),
            supported_systems=("darwin", "linux", "windows"),
            supported_machines=("arm64", "x86_64"),
            max_context_tokens=4096,
            offline_capable=True,
            expected_model_sha256=expected.get("expected_model_sha256"),
            expected_runtime_sha256=expected.get("expected_runtime_sha256"),
            expected_harness_sha256=expected.get("expected_harness_sha256"),
            expected_tool_contract_sha256=expected.get("expected_tool_contract_sha256"),
        )
        profile = AdvisorProfile(
            quality_weight=1,
            latency_weight=1,
            memory_weight=1,
            min_success_rate=0.8,
            min_samples=1,
            reserve_memory_bytes=0,
            latency_reference_ms=1000,
            memory_reference_bytes=1,
            max_snapshot_age_seconds=120,
            max_swap_used_bytes=0,
        )
        catalog = AdaptiveCellCatalog(
            catalog_id="runtime-binding-contract-fixture",
            cells=(build_cell_passport(declaration),),
            profiles={"balanced": profile},
        )
        _write_json(self.catalog_path, catalog.payload())

    def add_second_expert(self) -> None:
        experts = self.config["experts"]
        if not isinstance(experts, list):
            raise AssertionError("Fixture experts must remain a list.")
        experts.append(self._expert("other", "models/uninspected.gguf", 8124))
        self.write_config()

    def reverse_experts(self) -> None:
        experts = self.config["experts"]
        if not isinstance(experts, list):
            raise AssertionError("Fixture experts must remain a list.")
        experts.reverse()
        self.write_config()


@contextmanager
def _inspection_guard(
    read_requests: list[int],
    model_read_requests: list[int],
    *,
    model_path: Path,
) -> Iterator[None]:
    original_read = os.read
    model_stat = model_path.stat()
    model_identity = (int(model_stat.st_dev), int(model_stat.st_ino))

    def record_read(descriptor: int, size: int) -> bytes:
        read_requests.append(size)
        observed = os.fstat(descriptor)
        if (int(observed.st_dev), int(observed.st_ino)) == model_identity:
            model_read_requests.append(size)
        return original_read(descriptor, size)

    blocked = AssertionError("The static binding benchmark attempted a side effect.")
    with (
        mock.patch("local_moe.bootstrap.detect_platform_key", return_value="linux"),
        mock.patch.object(secure_files.os, "read", side_effect=record_read),
        mock.patch.object(Path, "read_bytes", side_effect=blocked),
        mock.patch.object(socket, "socket", side_effect=blocked),
        mock.patch.object(socket, "create_connection", side_effect=blocked),
        mock.patch.object(subprocess, "Popen", side_effect=blocked),
        mock.patch.object(urllib.request, "urlopen", side_effect=blocked),
        mock.patch(
            "local_moe.model_servers.ModelServerManager.start",
            side_effect=blocked,
        ),
    ):
        yield


def _inspect(
    fixture: _Fixture,
    read_requests: list[int],
    model_read_requests: list[int],
    *,
    now: datetime,
) -> CellBindingInspectionBundle:
    with _inspection_guard(
        read_requests,
        model_read_requests,
        model_path=fixture.model,
    ):
        return inspect_cell_binding(fixture.request_path, now=now)


def _observed_identities(bundle: CellBindingInspectionBundle) -> dict[str, str]:
    return {
        "expected_model_sha256": bundle.manifest.model_identity_sha256,
        "expected_runtime_sha256": bundle.manifest.runtime_identity_sha256,
        "expected_harness_sha256": bundle.manifest.harness_identity_sha256,
        "expected_tool_contract_sha256": (
            bundle.manifest.tool_contract_identity_sha256
        ),
    }


def _scenario(bundle: CellBindingInspectionBundle) -> dict[str, Any]:
    receipt = bundle.receipt
    manifest = bundle.manifest
    return {
        "status": receipt.status,
        "reason_codes": list(receipt.reason_codes),
        "cell_id": manifest.cell_id,
        "expert_id": manifest.expert_id,
        "adapter_id": manifest.adapter_id,
        "execution_scope": manifest.execution_scope,
        "transport": manifest.transport,
        "launch_plan_sha256": manifest.launch_plan_sha256,
        "expert_config_sha256": manifest.expert_config_sha256,
        "observed_identities": {
            "model_sha256": manifest.model_identity_sha256,
            "runtime_sha256": manifest.runtime_identity_sha256,
            "harness_sha256": manifest.harness_identity_sha256,
            "tool_contract_sha256": manifest.tool_contract_identity_sha256,
        },
        "component_count": receipt.component_count,
        "observed_component_count": receipt.observed_component_count,
        "captured_at": receipt.captured_at,
        "expires_at": receipt.expires_at,
        "residency_status": receipt.residency_status,
        "applied": receipt.applied,
        "network_used": receipt.network_used,
        "processes_started": receipt.processes_started,
        "model_invocations": receipt.model_invocations,
        "process_mutations": receipt.process_mutations,
        "authorizes_execution": receipt.authorizes_execution,
    }


def _is_static_and_non_authorizing(bundle: CellBindingInspectionBundle) -> bool:
    receipt = bundle.receipt
    return (
        not receipt.applied
        and not receipt.network_used
        and receipt.processes_started == 0
        and receipt.model_invocations == 0
        and not receipt.process_mutations
        and not receipt.authorizes_execution
        and receipt.residency_status == "unknown"
    )


def run_benchmark() -> dict[str, Any]:
    """Exercise deterministic runtime-binding contracts without execution."""

    read_requests: list[int] = []
    model_read_requests: list[int] = []
    with tempfile.TemporaryDirectory() as temp:
        fixture = _Fixture(Path(temp))

        first_run = _inspect(
            fixture,
            read_requests,
            model_read_requests,
            now=CAPTURED_AT,
        )
        expected = _observed_identities(first_run)
        fixture.write_catalog(expected)
        verified = _inspect(
            fixture, read_requests, model_read_requests, now=CAPTURED_AT
        )
        fresh_receipt = _inspect(
            fixture, read_requests, model_read_requests, now=FRESH_CAPTURED_AT
        )

        fixture.model.write_bytes(MODEL_BYTES + b"drift\n")
        model_drift = _inspect(
            fixture, read_requests, model_read_requests, now=CAPTURED_AT
        )
        fixture.model.write_bytes(MODEL_BYTES)

        fixture.runtime_executable.write_bytes(RUNTIME_BYTES + b"drift\n")
        runtime_drift = _inspect(
            fixture, read_requests, model_read_requests, now=CAPTURED_AT
        )
        fixture.runtime_executable.write_bytes(RUNTIME_BYTES)

        fixture.add_second_expert()
        before_reorder = _inspect(
            fixture, read_requests, model_read_requests, now=CAPTURED_AT
        )
        fixture.reverse_experts()
        after_reorder = _inspect(
            fixture, read_requests, model_read_requests, now=CAPTURED_AT
        )

    bundles = (
        first_run,
        verified,
        fresh_receipt,
        model_drift,
        runtime_drift,
        before_reorder,
        after_reorder,
    )
    unknown_reasons = {
        "harness_identity_unknown",
        "model_identity_unknown",
        "runtime_identity_unknown",
        "tool_contract_identity_unknown",
    }
    identity_values = tuple(expected.values())
    max_read_request = max(read_requests, default=0)
    max_model_read_request = max(model_read_requests, default=0)
    criteria = {
        "first_run_abstains_with_observed_identities": (
            first_run.receipt.status == "abstained"
            and set(first_run.receipt.reason_codes) == unknown_reasons
            and len(identity_values) == 4
            and all(len(value) == 64 for value in identity_values)
        ),
        "pinned_expected_identities_verify": (
            verified.receipt.status == "verified"
            and verified.receipt.reason_codes == ()
        ),
        "model_content_drift_abstains": (
            model_drift.receipt.status == "abstained"
            and model_drift.receipt.reason_codes == ("model_identity_mismatch",)
        ),
        "runtime_content_drift_abstains": (
            runtime_drift.receipt.status == "abstained"
            and runtime_drift.receipt.reason_codes == ("runtime_identity_mismatch",)
        ),
        "expert_reorder_preserves_selected_binding": (
            before_reorder.receipt.status == "verified"
            and after_reorder.receipt.status == "verified"
            and before_reorder.manifest.expert_id
            == after_reorder.manifest.expert_id
            == "coder"
            and before_reorder.manifest.expert_config_sha256
            == after_reorder.manifest.expert_config_sha256
            and before_reorder.manifest.launch_plan_sha256
            == after_reorder.manifest.launch_plan_sha256
        ),
        "manifest_is_clock_stable_and_receipt_is_fresh": (
            verified.manifest == fresh_receipt.manifest
            and verified.receipt.digest != fresh_receipt.receipt.digest
            and fresh_receipt.receipt.captured_at == FRESH_CAPTURED_AT.isoformat()
        ),
        "every_receipt_reports_zero_network_process_and_model_use": all(
            not bundle.receipt.network_used
            and bundle.receipt.processes_started == 0
            and bundle.receipt.model_invocations == 0
            for bundle in bundles
        ),
        "every_inspection_is_static_and_non_authorizing": all(
            _is_static_and_non_authorizing(bundle) for bundle in bundles
        ),
        "artifact_hashing_uses_bounded_streaming_reads": (
            len(MODEL_BYTES) > STREAM_READ_BOUND_BYTES
            and len(model_read_requests) > 1
            and max_model_read_request <= STREAM_READ_BOUND_BYTES
        ),
    }
    return {
        "schema_version": "1.0",
        "benchmark": "bound_cell_attestor_contract",
        "fixture": {
            "kind": "deterministic_synthetic_contract_fixture",
            "backend": "llama_cpp",
            "platform_key": "linux",
            "model_artifact_bytes": len(MODEL_BYTES),
            "stream_read_bound_bytes": STREAM_READ_BOUND_BYTES,
            "observed_max_read_request_bytes": max_read_request,
            "observed_model_read_request_count": len(model_read_requests),
            "observed_model_max_read_request_bytes": max_model_read_request,
            "whole_file_path_reads_blocked": True,
            "guarded_side_effect_surfaces": [
                "model_server_start",
                "network_socket",
                "process_spawn",
                "url_fetch",
            ],
        },
        "scenarios": {
            "first_run_unpinned": _scenario(first_run),
            "verified_after_expected_pinning": _scenario(verified),
            "model_content_drift": _scenario(model_drift),
            "runtime_content_drift": _scenario(runtime_drift),
            "expert_reorder": {
                "before": _scenario(before_reorder),
                "after": _scenario(after_reorder),
                "selected_binding_preserved": (
                    before_reorder.manifest.launch_plan_sha256
                    == after_reorder.manifest.launch_plan_sha256
                    and before_reorder.manifest.expert_config_sha256
                    == after_reorder.manifest.expert_config_sha256
                ),
            },
            "fresh_receipt": {
                "manifest_clock_stable": verified.manifest == fresh_receipt.manifest,
                "receipt_changed": (
                    verified.receipt.digest != fresh_receipt.receipt.digest
                ),
                "first_captured_at": verified.receipt.captured_at,
                "second_captured_at": fresh_receipt.receipt.captured_at,
                "second_expires_at": fresh_receipt.receipt.expires_at,
            },
        },
        "criteria": criteria,
        "contract_checks_passed": all(criteria.values()),
        "limits": [
            "synthetic_contract_fixture_only",
            "does_not_measure_model_quality",
            "does_not_measure_latency_throughput_or_memory",
            "does_not_benchmark_hash_performance",
            "does_not_start_stop_or_call_a_model",
            "does_not_open_network_connections",
            "does_not_run_tools",
            "does_not_authorize_or_apply_execution",
            "does_not_perform_operator_review_or_trust_decisions",
            "identity_digests_are_not_signatures_or_authenticated_provenance",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Bound Cell Attestor contract benchmark."
    )
    parser.add_argument(
        "--out",
        help="Optional path for the machine-readable JSON report.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    rendered = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not report["contract_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
