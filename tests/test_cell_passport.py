import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import local_moe.cell_passport as passport_module

from local_moe.cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellContractError,
    CellDeclaration,
    CellEstimate,
    CellMeasurement,
    CellObservation,
    MAX_CELLS,
    MAX_PROFILES,
    WorkloadDemand,
)
from local_moe.cell_passport import (
    MAX_CATALOG_BYTES,
    _trusted_cell_catalog_from_payload,
    build_cell_passport,
    load_cell_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
SHA_A, SHA_B, SHA_C, SHA_D = (character * 64 for character in "abcd")
SECURE_LOADER_SUPPORTED = os.name == "nt" or (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
)


def profile() -> AdvisorProfile:
    return AdvisorProfile(
        quality_weight=0.5,
        latency_weight=0.3,
        memory_weight=0.2,
        min_success_rate=0.8,
        min_samples=2,
        reserve_memory_bytes=GIB,
        latency_reference_ms=1000,
        memory_reference_bytes=16 * GIB,
        max_snapshot_age_seconds=60,
        max_swap_used_bytes=0,
    )


def passport(source_path: str | None = None, source_sha256: str | None = None):
    declared = CellDeclaration(
        cell_id="cell-a",
        model="local/model",
        quantization="int4",
        runtime="runtime-a",
        harness="harness-a",
        capabilities=("code",),
        tool_surfaces=("workspace",),
        risk_classes=("low",),
        supported_systems=("Darwin",),
        supported_machines=("arm64",),
        max_context_tokens=8192,
        offline_capable=True,
        expected_model_sha256=SHA_A,
        expected_runtime_sha256=SHA_B,
        expected_harness_sha256=SHA_C,
        expected_tool_contract_sha256=SHA_D,
    )
    if source_path is None:
        return build_cell_passport(declared)
    demand = WorkloadDemand(
        workload_id="coding.edit",
        capabilities=("code",),
        tool_surfaces=("workspace",),
        risk_class="low",
        context_tokens=4096,
    )
    observed = CellObservation(
        cell_id=declared.cell_id,
        declaration_sha256=declared.digest,
        model_status="available",
        runtime_status="available",
        harness_status="available",
        tool_contract_status="available",
        residency_status="not_resident",
        observed_model_sha256=SHA_A,
        observed_runtime_sha256=SHA_B,
        observed_harness_sha256=SHA_C,
        observed_tool_contract_sha256=SHA_D,
        captured_at="2026-07-21T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path=source_path,
        source_sha256=source_sha256,
    )
    estimated = CellEstimate(
        cell_id=declared.cell_id,
        declaration_sha256=declared.digest,
        memory_pool="unified",
        placement="integrated_accelerator",
        peak_unified_memory_bytes=4 * GIB,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    measured = CellMeasurement(
        cell_id=declared.cell_id,
        declaration_sha256=declared.digest,
        sample_count=4,
        success_rate=0.9,
        p95_latency_ms=100,
        memory_pool="unified",
        placement="integrated_accelerator",
        peak_unified_memory_bytes=4 * GIB,
        resource_class_sha256=SHA_A,
        demand_sha256=demand.digest,
        evaluation_contract_sha256=SHA_B,
        measured_at="2026-07-20T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path=source_path,
        source_sha256=source_sha256,
    )
    return build_cell_passport(
        declared, observed=observed, estimated=estimated, measured=measured
    )


def catalog(item) -> AdaptiveCellCatalog:
    return AdaptiveCellCatalog(
        catalog_id="fixture", cells=(item,), profiles={"balanced": profile()}
    )


class CellPassportTests(unittest.TestCase):
    def test_loader_rejects_duplicate_json_keys_at_every_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = CellDeclaration(
                cell_id="duplicate-key-cell",
                model="fixture/model",
                quantization="int4",
                runtime="fixture-runtime",
                harness="fixture-harness",
                capabilities=("summarization",),
                tool_surfaces=(),
                risk_classes=("compute_only",),
                supported_systems=("Linux",),
                supported_machines=("x86_64",),
                max_context_tokens=4096,
                offline_capable=True,
            )
            catalog = AdaptiveCellCatalog(
                catalog_id="duplicate-key-catalog",
                cells=(build_cell_passport(declaration),),
                profiles={"balanced": profile()},
            )
            rendered = json.dumps(catalog.payload(), separators=(",", ":"))
            duplicate_root = rendered.replace(
                '"catalog_id":"duplicate-key-catalog"',
                '"catalog_id":"decoy","catalog_id":"duplicate-key-catalog"',
                1,
            )
            duplicate_nested = rendered.replace(
                '"cell_id":"duplicate-key-cell"',
                '"cell_id":"decoy","cell_id":"duplicate-key-cell"',
                1,
            )
            for name, content in (
                ("root.json", duplicate_root),
                ("nested.json", duplicate_nested),
            ):
                with self.subTest(name=name):
                    path = root / name
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(CellContractError, "Duplicate JSON"):
                        load_cell_catalog(path)

    def test_unknown_builder_keeps_four_evidence_classes_separate(self) -> None:
        item = passport()
        self.assertEqual(item.observed.model_status, "unknown")
        self.assertIsNone(item.estimated.source_path)
        self.assertEqual(item.measured.sample_count, 0)
        self.assertEqual(
            set(item.payload()),
            {
                "schema_version",
                "declaration",
                "observed",
                "estimated",
                "measured",
                "digest",
            },
        )

    def test_programmatic_parser_is_trusted_but_loader_verifies_real_source_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence" / "source.json"
            evidence.parent.mkdir()
            content = b'{"evidence":"fixture"}\n'
            evidence.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            payload = catalog(passport("evidence/source.json", digest)).payload()
            configured = _trusted_cell_catalog_from_payload(payload)
            self.assertEqual(
                configured.cells[0].observed.source_path, "evidence/source.json"
            )
            config = root / "catalog.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            if not SECURE_LOADER_SUPPORTED:
                with self.assertRaisesRegex(CellContractError, "no-follow"):
                    load_cell_catalog(config)
                return
            self.assertEqual(load_cell_catalog(config).digest, configured.digest)
            evidence.write_bytes(b"tampered")
            with self.assertRaisesRegex(CellContractError, "source_sha256"):
                load_cell_catalog(config)

    def test_evidence_is_read_once_and_charged_to_a_total_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence.json"
            content = b'{"evidence":"shared"}\n'
            evidence.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            config = root / "catalog.json"
            config.write_text(
                json.dumps(catalog(passport("evidence.json", digest)).payload()),
                encoding="utf-8",
            )
            real_reader = passport_module.read_bounded_regular_file
            with patch.object(
                passport_module,
                "read_bounded_regular_file",
                wraps=real_reader,
            ) as reader:
                load_cell_catalog(config)
            evidence_reads = [
                call
                for call in reader.call_args_list
                if str(call.kwargs.get("label", "")).endswith("evidence")
            ]
            self.assertEqual(len(evidence_reads), 1)
            with patch.object(
                passport_module,
                "MAX_TOTAL_EVIDENCE_BYTES",
                len(content) - 1,
            ):
                with self.assertRaisesRegex(CellContractError, "cumulative"):
                    load_cell_catalog(config)

    def test_catalog_count_limits_are_enforced_before_item_construction(self) -> None:
        payload = catalog(passport()).payload()
        too_many_cells = dict(payload)
        too_many_cells["cells"] = [{} for _ in range(MAX_CELLS + 1)]
        with self.assertRaisesRegex(CellContractError, str(MAX_CELLS)):
            _trusted_cell_catalog_from_payload(too_many_cells)
        too_many_profiles = dict(payload)
        profile_payload = profile().payload()
        too_many_profiles["profiles"] = {
            f"profile-{index}": profile_payload for index in range(MAX_PROFILES + 1)
        }
        with self.assertRaisesRegex(CellContractError, str(MAX_PROFILES)):
            _trusted_cell_catalog_from_payload(too_many_profiles)

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows CI")
    def test_loader_rejects_symlinked_catalog_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_dir = root / "evidence"
            evidence_dir.mkdir()
            real = root / "real.json"
            real.write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            linked = evidence_dir / "source.json"
            linked.symlink_to(real)
            config = root / "catalog.json"
            config.write_text(
                json.dumps(catalog(passport("evidence/source.json", digest)).payload()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CellContractError, "securely|symlink"):
                load_cell_catalog(config)
            linked.unlink()
            linked.write_bytes(b"fixture")
            config_link = root / "catalog-link.json"
            config_link.symlink_to(config)
            with self.assertRaisesRegex(CellContractError, "securely|symlink"):
                load_cell_catalog(config_link)

            real_catalog_dir = root / "real-catalog"
            real_catalog_dir.mkdir()
            nested_catalog = real_catalog_dir / "catalog.json"
            nested_catalog.write_text(
                json.dumps(catalog(passport()).payload()),
                encoding="utf-8",
            )
            linked_catalog_dir = root / "linked-catalog"
            linked_catalog_dir.symlink_to(real_catalog_dir, target_is_directory=True)
            with self.assertRaisesRegex(CellContractError, "securely|symlink"):
                load_cell_catalog(
                    linked_catalog_dir / "catalog.json",
                    confinement_root=root,
                )

    def test_loader_is_bounded_and_relative_paths_cannot_escape(self) -> None:
        with self.assertRaises(CellContractError):
            passport("../outside.json", SHA_A)
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "catalog.json"
            oversized.write_bytes(b" " * (MAX_CATALOG_BYTES + 1))
            expected = "bounded" if SECURE_LOADER_SUPPORTED else "no-follow"
            with self.assertRaisesRegex(CellContractError, expected):
                load_cell_catalog(oversized)
        with self.assertRaises(CellContractError):
            load_cell_catalog("/definitely/missing/catalog.json")
        with self.assertRaises(CellContractError):
            load_cell_catalog("\0")
        with self.assertRaises(CellContractError):
            load_cell_catalog("x" * 10000)

    def test_only_loader_is_public_provenance_boundary(self) -> None:
        self.assertFalse(hasattr(passport_module, "cell_catalog_from_payload"))
        self.assertFalse(hasattr(passport_module, "cell_passport_from_payload"))
        self.assertNotIn("_trusted_cell_catalog_from_payload", passport_module.__all__)

    def test_example_catalog_has_no_synthetic_provenance(self) -> None:
        path = ROOT / "configs" / "adaptive-cells.example.json"
        if SECURE_LOADER_SUPPORTED:
            example = load_cell_catalog(path)
        else:
            with self.assertRaisesRegex(CellContractError, "no-follow"):
                load_cell_catalog(path)
            example = _trusted_cell_catalog_from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        for item in example.cells:
            self.assertIsNone(item.observed.source_sha256)
            self.assertIsNone(item.estimated.source_sha256)
            self.assertIsNone(item.measured.source_sha256)
            self.assertIsNone(item.declaration.expected_model_sha256)


if __name__ == "__main__":
    unittest.main()
