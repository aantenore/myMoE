from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.request

from local_moe import runtime_binding_inspector
from local_moe.cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellContractError,
    CellDeclaration,
)
from local_moe.cell_passport import build_cell_passport
from local_moe.runtime_binding_inspector import (
    ADAPTER_ID,
    PRODUCER_TRUST_BOUNDARY,
    RuntimeBindingInspectionError,
    inspect_cell_binding,
)


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
GENERIC_BACKEND = "llama_cpp" if os.name == "nt" else "mlx_lm"


class _InspectionFixture:
    def __init__(self, root: Path, backend: str) -> None:
        self.root = root
        self.backend = backend
        self.request_path = root / "inspect.json"
        self.config_path = root / "runtime.json"
        self.catalog_path = root / "catalog.json"
        self.runtime_root = root / "runtime"
        self.model_root = root / "models"
        self.runtime_root.mkdir()
        self.model_root.mkdir()
        (self.runtime_root / "bin").mkdir()
        (self.runtime_root / "lib").mkdir()
        (self.runtime_root / "bin" / "runtime").write_bytes(b"runtime-v1")
        (self.runtime_root / "lib" / "driver.py").write_bytes(b"driver-v1")
        (self.runtime_root / "lib" / "harness.py").write_bytes(b"harness-v1")
        if backend == "llama_cpp":
            self.model_reference = "models/private-coder.gguf"
            (root / self.model_reference).write_bytes(b"GGUF-local-model")
        else:
            self.model_reference = "models/private-coder"
            model_dir = root / self.model_reference
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"local"}', encoding="utf-8"
            )
            (model_dir / "private-coder.safetensors").write_bytes(b"weights-v1")
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
            "experts": [
                {
                    "id": "coder",
                    "provider": "openai_compatible",
                    "model": self.model_reference,
                    "role": "coding",
                    "base_url": "http://127.0.0.1:8123/v1",
                    "params": {
                        "runtime_backend": backend,
                        "runtime_model_source": "local",
                        "runtime_executable": "runtime/bin/runtime",
                        "opaque_parameter": "private-marker-value",
                    },
                    "execution": {
                        "scope": "device_only",
                        "transport": "direct_local",
                    },
                }
            ],
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
                "max_files": 100,
                "max_total_bytes": 2 * 1024 * 1024,
                "max_depth": 8,
                "max_file_bytes": 1024 * 1024,
            },
        }
        self.declaration_overrides: dict[str, object] = {}
        self.write_config()
        self.write_request()
        self.write_catalog()

    def write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(self.config, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_request(self) -> None:
        self.request_path.write_text(
            json.dumps(self.request, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_catalog(
        self,
        expected: dict[str, str | None] | None = None,
        **overrides: object,
    ) -> None:
        values: dict[str, object] = {
            "cell_id": "coder-local",
            "model": "coder-model",
            "quantization": "local",
            "runtime": self.backend,
            "harness": "mymoe",
            "capabilities": ("coding",),
            "tool_surfaces": (),
            "risk_classes": ("compute_only",),
            "supported_systems": ("darwin", "linux", "windows"),
            "supported_machines": ("arm64", "x86_64"),
            "max_context_tokens": 4096,
            "offline_capable": True,
            "expected_model_sha256": None,
            "expected_runtime_sha256": None,
            "expected_harness_sha256": None,
            "expected_tool_contract_sha256": None,
        }
        values.update(self.declaration_overrides)
        values.update(overrides)
        if expected is not None:
            values.update(expected)
        declaration = CellDeclaration(**values)  # type: ignore[arg-type]
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
            catalog_id="inspection-catalog",
            cells=(build_cell_passport(declaration),),
            profiles={"balanced": profile},
        )
        self.catalog_path.write_text(
            json.dumps(catalog.payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def observed_identities(self) -> dict[str, str | None]:
        bundle = inspect_cell_binding(self.request_path, now=NOW)
        return {
            "expected_model_sha256": bundle.manifest.model_identity_sha256,
            "expected_runtime_sha256": bundle.manifest.runtime_identity_sha256,
            "expected_harness_sha256": bundle.manifest.harness_identity_sha256,
            "expected_tool_contract_sha256": (
                bundle.manifest.tool_contract_identity_sha256
            ),
        }

    def make_verified(self) -> None:
        self.write_catalog(self.observed_identities())


class RuntimeBindingInspectorTests(unittest.TestCase):
    def _assert_verified_without_disclosure(self, backend: str) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), backend)
            fixture.make_verified()

            bundle = inspect_cell_binding(fixture.request_path, now=NOW)
            rendered = json.dumps(bundle.payload(), sort_keys=True)

            self.assertEqual(bundle.receipt.status, "verified")
            self.assertEqual(bundle.receipt.reason_codes, ())
            self.assertFalse(bundle.receipt.authorizes_execution)
            self.assertFalse(bundle.receipt.network_used)
            self.assertFalse(bundle.receipt.process_mutations)
            self.assertEqual(bundle.receipt.model_invocations, 0)
            self.assertEqual(bundle.contract, "BoundCellInspector")
            self.assertNotIn(str(fixture.root), rendered)
            self.assertNotIn(fixture.model_reference, rendered)
            self.assertNotIn("private-coder", rendered)
            self.assertNotIn("http://127.0.0.1:8123/v1", rendered)
            self.assertNotIn("private-marker-value", rendered)
            self.assertNotIn("--model", rendered)
            self.assertNotIn("-m", rendered)

    def test_verifies_gguf_file_without_disclosing_runtime_data(self) -> None:
        self._assert_verified_without_disclosure("llama_cpp")

    @unittest.skipIf(
        os.name == "nt",
        "v1 MLX directory identity requires secure POSIX traversal",
    )
    def test_verifies_mlx_directories_without_disclosing_runtime_data(self) -> None:
        for backend in ("mlx_lm", "mlx_vlm"):
            with self.subTest(backend=backend):
                self._assert_verified_without_disclosure(backend)

    def test_unknown_and_mismatched_expected_identities_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            unknown = inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(unknown.receipt.status, "abstained")
            self.assertEqual(
                set(unknown.receipt.reason_codes),
                {
                    "harness_identity_unknown",
                    "model_identity_unknown",
                    "runtime_identity_unknown",
                    "tool_contract_identity_unknown",
                },
            )

            fixture.write_catalog(
                {
                    "expected_model_sha256": "a" * 64,
                    "expected_runtime_sha256": "b" * 64,
                    "expected_harness_sha256": "c" * 64,
                    "expected_tool_contract_sha256": "d" * 64,
                }
            )
            mismatch = inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(mismatch.receipt.status, "abstained")
            self.assertEqual(
                set(mismatch.receipt.reason_codes),
                {
                    "harness_identity_mismatch",
                    "model_identity_mismatch",
                    "runtime_identity_mismatch",
                    "tool_contract_identity_mismatch",
                },
            )

    def test_manifest_is_clock_independent_but_receipt_records_fresh_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            fixture.make_verified()

            first = inspect_cell_binding(fixture.request_path, now=NOW)
            second = inspect_cell_binding(
                fixture.request_path, now=NOW + timedelta(seconds=5)
            )

            self.assertEqual(first.manifest, second.manifest)
            self.assertNotEqual(first.receipt.digest, second.receipt.digest)
            self.assertEqual(
                second.receipt.expires_at,
                (NOW + timedelta(seconds=65)).isoformat(),
            )

    def test_expert_reordering_preserves_selected_launch_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            other = {
                "id": "other",
                "provider": "openai_compatible",
                "model": "models/unused",
                "role": "general",
                "base_url": "http://127.0.0.1:8124/v1",
                "params": {
                    "runtime_backend": GENERIC_BACKEND,
                    "runtime_model_source": "local",
                    "runtime_executable": "runtime/bin/runtime",
                },
                "execution": {
                    "scope": "device_only",
                    "transport": "direct_local",
                },
            }
            experts = fixture.config["experts"]
            self.assertIsInstance(experts, list)
            experts.append(other)  # type: ignore[union-attr]
            fixture.write_config()
            first = inspect_cell_binding(fixture.request_path, now=NOW)

            experts.reverse()  # type: ignore[union-attr]
            fixture.write_config()
            second = inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(
                first.manifest.launch_plan_sha256,
                second.manifest.launch_plan_sha256,
            )
            self.assertEqual(
                first.manifest.expert_config_sha256,
                second.manifest.expert_config_sha256,
            )
            self.assertEqual(first.manifest.expert_id, "coder")
            self.assertEqual(second.manifest.expert_id, "coder")

    def test_rejects_remote_or_credentialed_endpoint_and_unsafe_runtime_modes(
        self,
    ) -> None:
        cases = (
            ("provider", "synthetic", "provider_unsupported"),
            ("backend", "ollama", "runtime_backend_unsupported"),
            (
                "model_source",
                "huggingface",
                "runtime_model_source_unsupported",
            ),
            ("endpoint", "https://example.com:8123/v1", "endpoint_unsupported"),
            (
                "endpoint",
                "https://127.0.0.1:8123/v1",
                "endpoint_unsupported",
            ),
            (
                "endpoint",
                "http://user:opaque@127.0.0.1:8123/v1",
                "endpoint_unsupported",
            ),
            (
                "endpoint",
                "http://127.0.0.1:8123/v1?marker=opaque",
                "endpoint_unsupported",
            ),
        )
        for kind, value, code in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
                expert = fixture.config["experts"][0]  # type: ignore[index]
                if kind == "provider":
                    expert["provider"] = value  # type: ignore[index]
                elif kind == "backend":
                    expert["params"]["runtime_backend"] = value  # type: ignore[index]
                elif kind == "model_source":
                    expert["params"]["runtime_model_source"] = value  # type: ignore[index]
                else:
                    expert["base_url"] = value  # type: ignore[index]
                fixture.write_config()
                with self.assertRaises(RuntimeBindingInspectionError) as raised:
                    inspect_cell_binding(fixture.request_path, now=NOW)
                self.assertEqual(raised.exception.code, code)

    def test_rejects_scope_tools_and_non_offline_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            execution = fixture.config["execution"]
            execution["max_scope"] = "private_mesh"  # type: ignore[index]
            execution["allowed_scopes"] = ["device_only", "private_mesh"]  # type: ignore[index]
            fixture.write_config()
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "execution_policy_unsupported")

        for overrides in (
            {"offline_capable": False},
            {"risk_classes": ("compute_only", "write_local")},
            {"tool_surfaces": ("filesystem",)},
        ):
            with (
                self.subTest(overrides=overrides),
                tempfile.TemporaryDirectory() as temp,
            ):
                fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
                fixture.write_catalog(**overrides)
                with self.assertRaises(RuntimeBindingInspectionError) as raised:
                    inspect_cell_binding(fixture.request_path, now=NOW)
                self.assertEqual(raised.exception.code, "cell_contract_unsupported")

    def test_requires_exact_runtime_executable_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            expert = fixture.config["experts"][0]  # type: ignore[index]
            expert["params"]["runtime_executable"] = "bin/other"  # type: ignore[index]
            fixture.write_config()

            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(raised.exception.code, "runtime_executable_mismatch")

    def test_request_is_strict_duplicate_safe_bounded_and_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            request = deepcopy(fixture.request)
            request["command"] = ["sh", "-c", "anything"]
            fixture.request_path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "request_invalid")

            fixture.request_path.write_text(
                '{"schema_version":"1.0","schema_version":"1.0"}',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "json_duplicate_key")

            request = deepcopy(fixture.request)
            request["catalog_path"] = "../catalog.json"
            fixture.request_path.write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "path_escape")

    def test_combined_hash_budget_is_applied_before_the_next_artifact(self) -> None:
        for limit_name in ("max_files", "max_total_bytes"):
            with (
                self.subTest(limit_name=limit_name),
                tempfile.TemporaryDirectory() as temp,
            ):
                fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
                limits = fixture.request["hash_limits"]
                self.assertIsInstance(limits, dict)
                if limit_name == "max_files":
                    limits["max_files"] = 3  # type: ignore[index]
                else:
                    limits["max_total_bytes"] = sum(  # type: ignore[index]
                        path.stat().st_size
                        for path in (
                            fixture.runtime_root / "bin" / "runtime",
                            fixture.runtime_root / "lib" / "driver.py",
                            fixture.runtime_root / "lib" / "harness.py",
                        )
                    )
                fixture.write_request()
                original = runtime_binding_inspector.hash_artifact_tree

                with mock.patch.object(
                    runtime_binding_inspector,
                    "hash_artifact_tree",
                    wraps=original,
                ) as observed_hashes:
                    with self.assertRaises(RuntimeBindingInspectionError) as raised:
                        inspect_cell_binding(fixture.request_path, now=NOW)

                self.assertEqual(raised.exception.code, "artifact_limits_exceeded")
                self.assertEqual(observed_hashes.call_count, 3)

    def test_publication_path_cannot_alias_inputs_or_enter_artifact_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            for output in (
                fixture.request_path,
                fixture.catalog_path,
                fixture.config_path,
                fixture.runtime_root / "receipt.json",
                fixture.model_root / "receipt.json",
            ):
                with self.subTest(output=output):
                    with self.assertRaises(RuntimeBindingInspectionError) as raised:
                        inspect_cell_binding(
                            fixture.request_path,
                            now=NOW,
                            publication_path=output,
                        )
                    self.assertEqual(raised.exception.code, "output_path_conflict")

    def test_producer_identity_covers_binding_decision_dependencies(self) -> None:
        expected = {
            "_win32_fs.py",
            "artifact_tree.py",
            "bootstrap.py",
            "cell_contracts.py",
            "cell_passport.py",
            "config.py",
            "execution_scope.py",
            "http_boundary.py",
            "model_servers.py",
            "path_security.py",
            "runtime_binding_contracts.py",
            "runtime_binding_inspector.py",
            "secure_files.py",
            "verified_routing_contracts.py",
        }

        self.assertTrue(expected.issubset(PRODUCER_TRUST_BOUNDARY))

    def test_rejects_duplicate_config_keys_and_reported_tree_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            fixture.config_path.write_text(
                '{"experts":[],"experts":[]}', encoding="utf-8"
            )
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "json_duplicate_key")

        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            with mock.patch(
                "local_moe.runtime_binding_inspector.hash_artifact_tree",
                side_effect=CellContractError("artifact changed during inspection"),
            ):
                with self.assertRaises(RuntimeBindingInspectionError) as raised:
                    inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "runtime_component_invalid")

    def test_rejects_linked_runtime_component(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            driver = fixture.runtime_root / "lib" / "driver.py"
            driver.unlink()
            try:
                driver.symlink_to(fixture.runtime_root / "lib" / "harness.py")
            except OSError:
                self.skipTest("Symlinks are unavailable on this platform.")
            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)
            self.assertEqual(raised.exception.code, "runtime_component_invalid")

    def test_rejects_hardlink_aliases_across_runtime_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            harness = fixture.runtime_root / "lib" / "harness.py"
            harness.unlink()
            try:
                os.link(fixture.runtime_root / "lib" / "driver.py", harness)
            except OSError:
                self.skipTest("Hard links are unavailable on this platform.")

            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(raised.exception.code, "runtime_component_invalid")

    def test_rejects_case_aliases_across_runtime_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            alias = fixture.runtime_root / "LIB" / "driver.py"
            original = fixture.runtime_root / "lib" / "driver.py"
            try:
                aliases_driver = os.path.samefile(alias, original)
            except (FileNotFoundError, OSError):
                aliases_driver = False
            if not aliases_driver:
                self.skipTest("The test filesystem is case-sensitive.")
            components = fixture.request["runtime_components"]
            self.assertIsInstance(components, list)
            components[0]["path"] = "LIB/driver.py"  # type: ignore[index]
            fixture.write_request()

            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(raised.exception.code, "runtime_component_invalid")

    @unittest.skipIf(
        os.name == "nt",
        "directory-link fixtures require secure POSIX directory traversal",
    )
    def test_rejects_linked_runtime_and_model_roots(self) -> None:
        for linked_root in ("runtime", "models"):
            with (
                self.subTest(linked_root=linked_root),
                tempfile.TemporaryDirectory() as temp,
            ):
                fixture = _InspectionFixture(Path(temp), "mlx_lm")
                target = fixture.root / f"{linked_root}-target"
                source = fixture.root / linked_root
                source.rename(target)
                try:
                    source.symlink_to(target, target_is_directory=True)
                except OSError:
                    self.skipTest(
                        "Directory symlinks are unavailable on this platform."
                    )
                with self.assertRaises(RuntimeBindingInspectionError) as raised:
                    inspect_cell_binding(fixture.request_path, now=NOW)
                self.assertEqual(raised.exception.code, "artifact_root_invalid")

    def test_model_artifact_root_must_be_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            fixture.request["model_artifact_root"] = "models/coder.gguf"
            fixture.write_request()

            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(raised.exception.code, "artifact_root_invalid")

    @unittest.skipIf(os.name == "nt", "MLX directory traversal is POSIX-only in v1")
    def test_model_relative_path_rename_changes_redacted_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), "mlx_lm")
            first = inspect_cell_binding(fixture.request_path, now=NOW)
            model_directory = fixture.root / fixture.model_reference
            (model_directory / "config.json").rename(model_directory / "settings.json")

            second = inspect_cell_binding(fixture.request_path, now=NOW)
            rendered = json.dumps(second.payload(), sort_keys=True)

            self.assertNotEqual(
                first.manifest.model_identity_sha256,
                second.manifest.model_identity_sha256,
            )
            self.assertNotIn("config.json", rendered)
            self.assertNotIn("settings.json", rendered)

    @unittest.skipIf(
        os.name == "nt",
        "directory-link fixtures require secure POSIX directory traversal",
    )
    def test_rejects_catalog_below_a_linked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            real_catalog_dir = fixture.root / "catalog-target"
            real_catalog_dir.mkdir()
            fixture.catalog_path.rename(real_catalog_dir / "catalog.json")
            linked_catalog_dir = fixture.root / "catalog-link"
            try:
                linked_catalog_dir.symlink_to(
                    real_catalog_dir,
                    target_is_directory=True,
                )
            except OSError:
                self.skipTest("Directory symlinks are unavailable on this platform.")
            fixture.request["catalog_path"] = "catalog-link/catalog.json"
            fixture.write_request()

            with self.assertRaises(RuntimeBindingInspectionError) as raised:
                inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(raised.exception.code, "catalog_invalid")

    def test_does_not_open_sockets_spawn_processes_fetch_or_start_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            fixture = _InspectionFixture(Path(temp), GENERIC_BACKEND)
            fixture.make_verified()
            with (
                mock.patch.object(socket, "socket", side_effect=AssertionError),
                mock.patch.object(
                    socket, "create_connection", side_effect=AssertionError
                ),
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError),
                mock.patch.object(
                    urllib.request, "urlopen", side_effect=AssertionError
                ),
                mock.patch(
                    "local_moe.model_servers.ModelServerManager.start",
                    side_effect=AssertionError,
                ),
            ):
                bundle = inspect_cell_binding(fixture.request_path, now=NOW)

            self.assertEqual(bundle.receipt.status, "verified")


if __name__ == "__main__":
    unittest.main()
