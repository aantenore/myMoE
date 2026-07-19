from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
from unittest.mock import patch

from local_moe import route_outcomes as route_outcomes_module
from local_moe.route_canary import CanaryRouteDecision
from local_moe.route_outcomes import (
    OutcomeStore,
    VerifiedOutcomeRecord,
    build_verified_outcome,
    runtime_plan_sha256,
)
from local_moe.route_scorecard import build_route_scorecard
from local_moe.route_signals import MetadataTaskSignalProvider, TaskSignals
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
    sha256_json,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64
_DIGEST_E = "e" * 64
_DIGEST_F = "f" * 64
_RUNTIME_PLAN_SHA256 = (
    "5c3dfe530447050655c4b563df4baf261d32f1ef5564668800a0b3887fd558cd"
)


class RouteOutcomeTests(unittest.TestCase):
    def test_builds_metadata_only_record_and_aggregates_metrics(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("check-a", passed=True)],
            commands=[_command(11, 5, 3), _command(17, 7, 4)],
            capsule=_capsule(321),
            premium_calls=1,
        )

        signals = _signals()
        record = build_verified_outcome(
            metadata,
            signals,
            estimated_cost_usd=0.0125,
            created_at="2026-07-19T03:00:00+00:00",
        )
        rendered = canonical_json(record.payload())

        self.assertEqual(record.outcome, "passed")
        self.assertEqual(record.evidence_strength, "deterministic")
        self.assertEqual(record.failure_class, "none")
        self.assertEqual(record.latency_ms, 28)
        self.assertEqual(record.prompt_tokens, 12)
        self.assertEqual(record.completion_tokens, 7)
        self.assertEqual(record.premium_calls, 1)
        self.assertEqual(record.remote_payload_chars, 321)
        self.assertEqual(record.model, "local/model-a")
        self.assertEqual(record.provider_runtime_sha256, metadata["route_receipt"]["local_runtime"]["runtime_sha256"])
        self.assertEqual(
            record.signal_provider_config_sha256,
            signals.provider_config_sha256,
        )
        self.assertEqual(
            record.runtime_plan_sha256,
            runtime_plan_sha256(metadata["route_receipt"]),
        )
        self.assertEqual(record.runtime_plan_sha256, _RUNTIME_PLAN_SHA256)
        self.assertNotIn("reasoning", rendered.lower())
        self.assertNotIn("private-result-body", rendered)
        self.assertNotIn("content", record.payload())

    def test_provider_config_and_complete_runtime_plan_are_content_bound(self) -> None:
        metadata = _bridge_metadata(evidence=[_evidence("check-a", passed=True)])
        first = build_verified_outcome(
            metadata,
            _signals(provider_config_sha256=_DIGEST_E),
            created_at="2026-07-19T03:00:00+00:00",
        )

        changed_runtime = _bridge_metadata(
            evidence=[_evidence("check-a", passed=True)]
        )
        premium_runtime = changed_runtime["route_receipt"]["premium_runtime"]
        premium_runtime["model"] = "premium/model-b"
        premium_runtime["runtime_sha256"] = sha256_json(
            {
                key: value
                for key, value in premium_runtime.items()
                if key != "runtime_sha256"
            }
        )
        second = build_verified_outcome(
            changed_runtime,
            _signals(provider_config_sha256=_DIGEST_F),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertNotEqual(
            first.signal_provider_config_sha256,
            second.signal_provider_config_sha256,
        )
        self.assertNotEqual(first.runtime_plan_sha256, second.runtime_plan_sha256)
        self.assertEqual(
            first.provider_runtime_sha256,
            second.provider_runtime_sha256,
        )

    def test_canary_lineage_is_persisted_without_breaking_legacy_records(self) -> None:
        metadata = _bridge_metadata(evidence=[_evidence("check-a", passed=True)])
        receipt = metadata["route_receipt"]
        receipt["expected_flow"] = [
            "local",
            "verify",
            "stop_or_capsule",
            "premium",
            "verify",
        ]
        unsigned_baseline = dict(receipt)
        unsigned_baseline.pop("receipt_id")
        receipt["receipt_id"] = f"route-{sha256_json(unsigned_baseline)[:32]}"
        baseline_receipt_id = receipt["receipt_id"]
        baseline_receipt_sha256 = sha256_json(receipt)
        canary_signals = MetadataTaskSignalProvider().signals_from_metadata(
            receipt["task"]
        )
        decision = CanaryRouteDecision(
            task_fingerprint=_DIGEST_A,
            profile="balanced",
            capabilities=("code", "tests"),
            difficulty=canary_signals.difficulty,
            baseline_route="local_then_verify",
            effective_route="local_then_verify",
            shadow_recommended_route="local",
            applied=False,
            abstained=True,
            reason_codes=("outside_canary_cohort",),
            route_receipt_id=baseline_receipt_id,
            route_receipt_sha256=baseline_receipt_sha256,
            runtime_plan_sha256=_RUNTIME_PLAN_SHA256,
            signal_provider_config_sha256=(
                canary_signals.provider_config_sha256
            ),
            shadow_decision_sha256=_DIGEST_E,
            policy_digest=_DIGEST_A,
            scorecard_digest=_DIGEST_B,
            bridge_config_sha256=_DIGEST_C,
            manifest_sha256=_DIGEST_D,
            authorization_sha256=_DIGEST_E,
            operator_key_id="operator-a",
            assignment_bucket=700,
            canary_basis_points=500,
        )
        receipt["rationale_codes"].append(
            "verified_route_canary_baseline_retained"
        )
        receipt["route_canary"] = decision.payload()
        unsigned_current = dict(receipt)
        unsigned_current.pop("receipt_id")
        receipt["receipt_id"] = f"route-{sha256_json(unsigned_current)[:32]}"

        record = build_verified_outcome(
            metadata,
            canary_signals,
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(
            record.route_canary["decision_sha256"],
            decision.decision_sha256,
        )
        self.assertEqual(VerifiedOutcomeRecord.from_payload(record.payload()), record)

        transplanted = record.payload()
        transplanted["task_fingerprint"] = _DIGEST_F
        unsigned_transplanted = dict(transplanted)
        unsigned_transplanted.pop("record_id")
        transplanted["record_id"] = (
            f"outcome-{sha256_json(unsigned_transplanted)}"
        )
        with self.assertRaisesRegex(
            VerifiedRoutingError,
            "route canary binding",
        ):
            VerifiedOutcomeRecord.from_payload(transplanted)

        self.assertNotIn("route_canary", _bridge_metadata(evidence=[])["route_receipt"])
        with self.assertRaises(TypeError):
            record.route_canary["reason_codes"][0] = "tampered"  # type: ignore[index]

    def test_failed_evidence_wins_and_external_evidence_is_independent(self) -> None:
        metadata = _bridge_metadata(
            evidence=[
                _evidence("check-a", passed=True),
                _evidence("contract-failed", passed=False, kind="external"),
            ]
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "contract-failed")
        self.assertEqual(record.evidence_strength, "independent")

    def test_missing_final_evidence_is_inconclusive(self) -> None:
        record = build_verified_outcome(
            _bridge_metadata(evidence=[], required_verifier_ids=[]),
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "inconclusive")
        self.assertEqual(record.evidence_strength, "implicit")
        self.assertEqual(record.failure_class, "verification_missing")

    def test_required_verifier_must_pass_in_final_phase(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("unrelated-check", passed=True)],
            prior_evidence=[_evidence("check-a", passed=True)],
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "required_verifier_missing")

    def test_failed_bridge_cannot_be_rescued_by_unrelated_passing_evidence(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("unrelated-check", passed=True)],
            status="failed",
            code="premium-runtime-failed",
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "premium-runtime-failed")

    def test_prior_failure_does_not_override_verified_final_recovery(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("check-a", passed=True)],
            prior_evidence=[
                _evidence("local-check", passed=False, kind="external")
            ],
            code="premium_verification_passed",
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "passed")
        self.assertEqual(record.failure_class, "none")
        self.assertEqual(record.evidence_strength, "deterministic")
        scorecard = build_route_scorecard(
            [record],
            minimum_evidence_strength="deterministic",
            generated_at="2026-07-19T03:01:00+00:00",
        )
        self.assertEqual(scorecard.entries[0].verified_samples, 1)
        self.assertEqual(scorecard.entries[0].success_rate, 1.0)

    def test_capability_comparison_is_order_independent(self) -> None:
        metadata = _bridge_metadata(evidence=[])
        metadata["route_receipt"]["task"]["capability_demand"]["required"] = [
            "tests",
            "code",
        ]

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.capabilities, ("code", "tests"))

    def test_store_is_idempotent_and_fails_closed_on_corruption(self) -> None:
        record = build_verified_outcome(
            _bridge_metadata(evidence=[_evidence("check-a", passed=True)]),
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            store = OutcomeStore(path)
            self.assertTrue(store.append(record))
            self.assertFalse(store.append(record))
            self.assertEqual(store.list_records(), (record,))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

            with path.open("a", encoding="utf-8") as handle:
                handle.write("{broken\n")
            with self.assertRaisesRegex(VerifiedRoutingError, "corrupt"):
                store.list_records()
            with self.assertRaises(VerifiedRoutingError):
                store.append(record)

    def test_store_serializes_concurrent_process_appends(self) -> None:
        records = [
            build_verified_outcome(
                _bridge_metadata(evidence=[_evidence("check-a", passed=True)]),
                _signals(),
                created_at=f"2026-07-19T03:00:0{index}+00:00",
            )
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            appended = _run_concurrent_appends(
                path,
                [record.payload() for record in records],
            )

            self.assertEqual(appended, [True, True, True])
            stored = OutcomeStore(path).list_records()
            self.assertEqual(
                {record.record_id for record in stored},
                {record.record_id for record in records},
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_store_same_record_is_idempotent_across_processes(self) -> None:
        record = build_verified_outcome(
            _bridge_metadata(evidence=[_evidence("check-a", passed=True)]),
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            appended = _run_concurrent_appends(
                path,
                [record.payload(), record.payload(), record.payload()],
            )

            self.assertEqual(sorted(appended), [False, False, True])
            self.assertEqual(OutcomeStore(path).list_records(), (record,))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    @unittest.skipIf(
        os.name == "nt",
        "Win32 payload descriptors are covered by test_win32_fs",
    )
    def test_store_opens_payload_in_binary_mode_when_available(self) -> None:
        synthetic_binary_flag = 1 << 29
        native_binary_flag = getattr(os, "O_BINARY", 0)
        real_open = os.open
        observed_flags: list[int] = []

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"

            def open_with_synthetic_binary_flag(
                candidate: str | os.PathLike[str],
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if Path(candidate) == path:
                    observed_flags.append(flags)
                portable_flags = flags & ~synthetic_binary_flag
                if flags & synthetic_binary_flag:
                    portable_flags |= native_binary_flag
                if dir_fd is None:
                    return real_open(candidate, portable_flags, mode)
                return real_open(
                    candidate,
                    portable_flags,
                    mode,
                    dir_fd=dir_fd,
                )

            with patch.object(
                route_outcomes_module.os,
                "O_BINARY",
                synthetic_binary_flag,
                create=True,
            ), patch.object(
                route_outcomes_module.os,
                "open",
                side_effect=open_with_synthetic_binary_flag,
            ):
                route_outcomes_module._append_secure_outcome_file(path, b"{}\n")
                self.assertEqual(
                    route_outcomes_module._read_secure_outcome_file(path),
                    b"{}\n",
                )

        self.assertEqual(len(observed_flags), 2)
        self.assertTrue(
            all(flags & synthetic_binary_flag for flags in observed_flags)
        )

    def test_windows_payload_io_uses_one_pinned_nofollow_file_id(self) -> None:
        from local_moe import _win32_fs

        identity = _outcome_windows_identity(9, attributes=0)
        archived_identity = _outcome_windows_identity(9, attributes=0x20)
        real_open = os.open
        real_read = os.read
        real_write = os.write
        opened: list[tuple[int, bool, bool]] = []
        read_descriptors: list[int] = []
        write_descriptors: list[int] = []

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_bytes(b"first\n")
            path.chmod(0o600)

            def open_nofollow(
                candidate: str | os.PathLike[str],
                *,
                directory: bool,
                writable: bool,
                share_delete: bool,
            ) -> tuple[int, object]:
                self.assertEqual(Path(candidate), path)
                self.assertFalse(directory)
                descriptor = real_open(
                    candidate,
                    os.O_RDWR if writable else os.O_RDONLY,
                )
                opened.append((descriptor, writable, share_delete))
                return descriptor, identity

            def tracked_read(descriptor: int, size: int) -> bytes:
                read_descriptors.append(descriptor)
                return real_read(descriptor, size)

            def tracked_write(descriptor: int, value: object) -> int:
                write_descriptors.append(descriptor)
                return real_write(descriptor, value)  # type: ignore[arg-type]

            with (
                mock.patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_nofollow,
                ) as nofollow,
                mock.patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    return_value=archived_identity,
                ),
                mock.patch.object(
                    route_outcomes_module.os,
                    "read",
                    side_effect=tracked_read,
                ),
                mock.patch.object(
                    route_outcomes_module.os,
                    "write",
                    side_effect=tracked_write,
                ),
            ):
                route_outcomes_module._append_secure_outcome_file_windows(
                    path,
                    b"second\n",
                )
                observed = (
                    route_outcomes_module._read_secure_outcome_file_windows(path)
                )

            self.assertEqual(observed, b"first\nsecond\n")
            self.assertEqual(path.read_bytes(), observed)

        self.assertEqual(nofollow.call_count, 6)
        self.assertTrue(opened[0][1])
        self.assertTrue(all(not share_delete for _, _, share_delete in opened))
        self.assertEqual(write_descriptors, [opened[0][0]])
        self.assertTrue(read_descriptors)
        self.assertEqual(set(read_descriptors), {opened[3][0]})

    def test_windows_identity_validation_is_separate_from_continuity(self) -> None:
        regular = _outcome_windows_identity(13, attributes=0x20)
        reparse = _outcome_windows_identity(13, attributes=0x400)
        directory = _outcome_windows_identity(13, attributes=0x10)

        self.assertTrue(regular.same_file_as(reparse))
        self.assertTrue(regular.same_file_as(directory))
        with self.assertRaisesRegex(VerifiedRoutingError, "reparse"):
            route_outcomes_module._validate_windows_outcome_identity(
                reparse,
                directory=False,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "regular file"):
            route_outcomes_module._validate_windows_outcome_identity(
                directory,
                directory=False,
            )

    def test_windows_payload_file_id_swap_fails_before_write(self) -> None:
        from local_moe import _win32_fs

        original_identity = _outcome_windows_identity(10, attributes=0x20)
        replacement_identity = _outcome_windows_identity(11, attributes=0x20)
        real_open = os.open
        identities = iter((original_identity, replacement_identity))
        identity_by_descriptor: dict[int, object] = {}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_bytes(b"stable\n")
            path.chmod(0o600)

            def open_nofollow(
                candidate: str | os.PathLike[str],
                *,
                directory: bool,
                writable: bool,
                share_delete: bool,
            ) -> tuple[int, object]:
                self.assertFalse(directory)
                self.assertFalse(share_delete)
                descriptor = real_open(
                    candidate,
                    os.O_RDWR if writable else os.O_RDONLY,
                )
                identity = next(identities)
                identity_by_descriptor[descriptor] = identity
                return descriptor, identity

            with (
                mock.patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_nofollow,
                ),
                mock.patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    side_effect=lambda descriptor: identity_by_descriptor[descriptor],
                ),
                mock.patch.object(route_outcomes_module.os, "write") as write,
                self.assertRaisesRegex(
                    VerifiedRoutingError,
                    "pathname no longer names",
                ),
            ):
                route_outcomes_module._append_secure_outcome_file_windows(
                    path,
                    b"unsafe\n",
                )

            self.assertEqual(path.read_bytes(), b"stable\n")
            write.assert_not_called()

    def test_windows_payload_reparse_attribute_is_rejected_before_open(
        self,
    ) -> None:
        from local_moe import _win32_fs

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_bytes(b"stable\n")
            observed = path.lstat()
            reparse = types.SimpleNamespace(
                st_mode=observed.st_mode,
                st_nlink=observed.st_nlink,
                st_size=observed.st_size,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_mtime_ns=observed.st_mtime_ns,
                st_file_attributes=0x00000400,
            )

            with (
                mock.patch.object(Path, "lstat", return_value=reparse),
                mock.patch.object(_win32_fs, "open_nofollow_fd") as nofollow,
                self.assertRaisesRegex(VerifiedRoutingError, "regular non-link"),
            ):
                route_outcomes_module._read_secure_outcome_file_windows(path)

            nofollow.assert_not_called()

    def test_windows_post_append_recheck_accepts_deferred_mtime(self) -> None:
        from local_moe import _win32_fs

        identity = _outcome_windows_identity(12, attributes=0x20)
        real_open = os.open

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_bytes(b"complete\n")
            path.chmod(0o600)
            observed = path.lstat()
            writer_view = types.SimpleNamespace(
                st_mode=observed.st_mode,
                st_nlink=observed.st_nlink,
                st_size=observed.st_size,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_mtime_ns=observed.st_mtime_ns + 1,
                st_file_attributes=0,
            )

            def open_nofollow(
                candidate: str | os.PathLike[str],
                *,
                directory: bool,
                writable: bool,
                share_delete: bool,
            ) -> tuple[int, object]:
                self.assertFalse(directory)
                self.assertFalse(writable)
                self.assertFalse(share_delete)
                return real_open(candidate, os.O_RDONLY), identity

            with (
                mock.patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_nofollow,
                ),
                mock.patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    return_value=identity,
                ),
            ):
                route_outcomes_module._recheck_windows_outcome_path(
                    path,
                    expected_identity=identity,
                    expected_metadata=writer_view,
                    label="outcome store",
                    maximum_bytes=1024,
                    compare_mtime=False,
                )

    def test_windows_lock_pins_parent_and_lock_by_handle_identity(self) -> None:
        from local_moe import _win32_fs

        parent = _outcome_windows_identity(1, attributes=0x10)
        lock = _outcome_windows_identity(2, attributes=0x20)
        fake_msvcrt = _outcome_fake_msvcrt()
        with (
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[
                    (71, parent),
                    (72, lock),
                    (73, parent),
                    (74, lock),
                ],
            ) as open_nofollow,
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[parent, lock, parent, lock, parent, lock],
            ),
            mock.patch.object(route_outcomes_module.os, "lseek"),
            mock.patch.object(route_outcomes_module.os, "close") as close,
        ):
            with route_outcomes_module._locked_windows_outcome_paths(
                Path("outcomes.jsonl.lock"),
                Path("store"),
                timeout_seconds=1.0,
            ):
                self.assertEqual(
                    close.call_args_list,
                    [mock.call(74), mock.call(73)],
                )

        self.assertEqual(
            open_nofollow.call_args_list,
            [
                mock.call(
                    Path("store"),
                    directory=True,
                    writable=False,
                    share_delete=False,
                ),
                mock.call(
                    Path("outcomes.jsonl.lock"),
                    directory=False,
                    writable=True,
                    share_delete=False,
                ),
                mock.call(
                    Path("store"),
                    directory=True,
                    writable=False,
                    share_delete=False,
                ),
                mock.call(
                    Path("outcomes.jsonl.lock"),
                    directory=False,
                    writable=True,
                    share_delete=False,
                ),
            ],
        )
        self.assertEqual(
            close.call_args_list,
            [mock.call(74), mock.call(73), mock.call(72), mock.call(71)],
        )
        self.assertEqual(
            fake_msvcrt.locking.call_args_list,
            [mock.call(72, 1, 1), mock.call(72, 2, 1)],
        )

    def test_windows_lock_rejects_split_path_identity(self) -> None:
        from local_moe import _win32_fs

        parent = _outcome_windows_identity(1, attributes=0x10)
        lock = _outcome_windows_identity(2, attributes=0x20)
        replacement = _outcome_windows_identity(3, attributes=0x20)
        fake_msvcrt = _outcome_fake_msvcrt()
        with (
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[
                    (71, parent),
                    (72, lock),
                    (73, parent),
                    (74, replacement),
                ],
            ),
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[
                    parent,
                    lock,
                    parent,
                    lock,
                    parent,
                    replacement,
                ],
            ),
            mock.patch.object(route_outcomes_module.os, "lseek"),
            mock.patch.object(route_outcomes_module.os, "close") as close,
            self.assertRaisesRegex(
                VerifiedRoutingError,
                "changed during acquisition",
            ),
        ):
            with route_outcomes_module._locked_windows_outcome_paths(
                Path("outcomes.jsonl.lock"),
                Path("store"),
                timeout_seconds=1.0,
            ):
                self.fail("a replaced lock path must not yield")

        self.assertEqual(
            close.call_args_list,
            [mock.call(74), mock.call(73), mock.call(72), mock.call(71)],
        )
        self.assertEqual(
            fake_msvcrt.locking.call_args_list,
            [mock.call(72, 1, 1), mock.call(72, 2, 1)],
        )

    def test_store_lock_timeout_fails_closed(self) -> None:
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            store = OutcomeStore(path, lock_timeout_seconds=0.05)
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_hold_lock,
                args=(str(store.lock_path), ready, release),
            )
            process.start()
            try:
                self.assertTrue(ready.wait(10.0))
                with self.assertRaisesRegex(
                    VerifiedRoutingError,
                    "lock acquisition timed out",
                ):
                    store.list_records()
            finally:
                release.set()
                process.join(10.0)
                if process.is_alive():
                    process.terminate()
                process.join(5.0)
            self.assertEqual(process.exitcode, 0)

    @unittest.skipIf(os.name == "nt", "POSIX link semantics")
    def test_store_rejects_symlink_and_hardlink_outcome_paths(self) -> None:
        record = _store_record()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.jsonl"
            target.write_text("target-must-not-change\n", encoding="utf-8")
            os.chmod(target, 0o600)
            symlink = root / "symlink.jsonl"
            symlink.symlink_to(target)

            with self.assertRaisesRegex(VerifiedRoutingError, "non-link"):
                OutcomeStore(symlink).append(record)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "target-must-not-change\n",
            )

            source = root / "source.jsonl"
            source.write_text(
                canonical_json(record.payload()) + "\n",
                encoding="utf-8",
            )
            os.chmod(source, 0o600)
            hardlink = root / "hardlink.jsonl"
            os.link(source, hardlink)
            with self.assertRaisesRegex(VerifiedRoutingError, "one hard link"):
                OutcomeStore(hardlink).list_records()

    @unittest.skipUnless(os.name == "nt", "native Windows reparse semantics")
    def test_native_windows_store_rejects_reparse_outcome_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.jsonl"
            target.write_bytes(b"target-must-not-change\n")
            link = root / "link.jsonl"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"Windows symlink unavailable: {exc}")

            with self.assertRaisesRegex(VerifiedRoutingError, "non-link|reparse"):
                OutcomeStore(link).list_records()
            self.assertEqual(target.read_bytes(), b"target-must-not-change\n")

    @unittest.skipIf(os.name == "nt", "POSIX symlink semantics")
    def test_store_pins_an_ancestor_symlink_without_following_the_target_file(self) -> None:
        record = _store_record()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "private"
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
            alias = root / "alias"
            alias.symlink_to(private, target_is_directory=True)

            store = OutcomeStore(alias / "outcomes.jsonl")
            self.assertEqual(store.path, private.resolve() / "outcomes.jsonl")
            self.assertTrue(store.append(record))
            self.assertEqual(store.list_records(), (record,))

    @unittest.skipIf(os.name == "nt", "POSIX link semantics")
    def test_store_rejects_symlink_and_hardlink_lock_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = OutcomeStore(root / "outcomes.jsonl")
            target = root / "lock-target"
            target.write_text("lock-must-not-change", encoding="utf-8")
            os.chmod(target, 0o600)
            store.lock_path.symlink_to(target)

            with self.assertRaisesRegex(VerifiedRoutingError, "non-link"):
                store.list_records()
            self.assertEqual(target.read_text(encoding="utf-8"), "lock-must-not-change")

            store.lock_path.unlink()
            source = root / "lock-source"
            source.write_text("", encoding="utf-8")
            os.chmod(source, 0o600)
            os.link(source, store.lock_path)
            with self.assertRaisesRegex(VerifiedRoutingError, "one hard link"):
                store.list_records()

    @unittest.skipIf(os.name == "nt", "POSIX FIFO semantics")
    def test_store_rejects_directory_and_fifo_without_opening_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "directory.jsonl"
            directory.mkdir(mode=0o700)
            with self.assertRaisesRegex(VerifiedRoutingError, "regular non-link"):
                OutcomeStore(directory).list_records()

            fifo = root / "fifo.jsonl"
            os.mkfifo(fifo, mode=0o600)
            with self.assertRaisesRegex(VerifiedRoutingError, "regular non-link"):
                OutcomeStore(fifo).list_records()

    @unittest.skipIf(os.name == "nt", "POSIX permission semantics")
    def test_store_rejects_permissive_parent_file_and_lock_modes(self) -> None:
        record = _store_record()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            permissive_parent = root / "permissive-parent"
            permissive_parent.mkdir(mode=0o755)
            os.chmod(permissive_parent, 0o755)
            with self.assertRaisesRegex(VerifiedRoutingError, "0700"):
                OutcomeStore(permissive_parent / "outcomes.jsonl").list_records()

            outcome = root / "permissive-outcomes.jsonl"
            outcome.write_text(
                canonical_json(record.payload()) + "\n",
                encoding="utf-8",
            )
            os.chmod(outcome, 0o644)
            with self.assertRaisesRegex(VerifiedRoutingError, "0600"):
                OutcomeStore(outcome).list_records()

            lock_store = OutcomeStore(root / "lock-outcomes.jsonl")
            lock_store.lock_path.write_text("", encoding="utf-8")
            os.chmod(lock_store.lock_path, 0o644)
            with self.assertRaisesRegex(VerifiedRoutingError, "0600"):
                lock_store.list_records()

    def test_store_rejects_oversized_file_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.jsonl"
            with path.open("wb") as handle:
                handle.truncate(route_outcomes_module._MAX_OUTCOME_STORE_BYTES + 1)
            if os.name != "nt":
                os.chmod(path, 0o600)

            with self.assertRaisesRegex(VerifiedRoutingError, "size limit"):
                OutcomeStore(path).list_records()

    @unittest.skipIf(os.name == "nt", "POSIX symlink replacement semantics")
    def test_store_rejects_replacement_race_before_target_write(self) -> None:
        record = _store_record()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "outcomes.jsonl"
            path.touch(mode=0o600)
            os.chmod(path, 0o600)
            store = OutcomeStore(path)
            original = root / "original.jsonl"
            target = root / "replacement-target.jsonl"
            target.write_text("target-must-not-change\n", encoding="utf-8")
            os.chmod(target, 0o600)
            real_open = os.open
            swapped = False

            def replace_before_append(
                candidate: object,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal swapped
                if (
                    not swapped
                    and Path(candidate) == store.path
                    and flags & os.O_APPEND
                ):
                    path.rename(original)
                    path.symlink_to(target)
                    swapped = True
                if dir_fd is None:
                    return real_open(candidate, flags, mode)
                return real_open(candidate, flags, mode, dir_fd=dir_fd)

            with patch.object(
                route_outcomes_module.os,
                "open",
                side_effect=replace_before_append,
            ), self.assertRaises(VerifiedRoutingError):
                store.append(record)

            self.assertTrue(swapped)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "target-must-not-change\n",
            )

    def test_strict_parsing_rejects_leaks_tampering_and_non_finite_cost(self) -> None:
        metadata = _bridge_metadata(evidence=[])
        metadata["raw_output"] = "private-result-body"
        with self.assertRaises(VerifiedRoutingError):
            build_verified_outcome(metadata, _signals())

        clean = _bridge_metadata(evidence=[])
        with self.assertRaisesRegex(VerifiedRoutingError, "finite"):
            build_verified_outcome(clean, _signals(), estimated_cost_usd=float("nan"))

        invalid_plan = _bridge_metadata(evidence=[])
        invalid_plan["route_receipt"]["premium_runtime"]["model"] = "tampered"
        with self.assertRaisesRegex(VerifiedRoutingError, "premium runtime digest"):
            build_verified_outcome(invalid_plan, _signals())

        record = build_verified_outcome(
            clean,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )
        tampered = record.payload()
        tampered["latency_ms"] = 999
        with self.assertRaisesRegex(VerifiedRoutingError, "record_id"):
            VerifiedOutcomeRecord.from_payload(tampered)

        serialized = canonical_json(record.payload()).replace(
            '"confidence":0.9', '"confidence":NaN'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_text(serialized + "\n", encoding="utf-8")
            if os.name != "nt":
                os.chmod(path, 0o600)
            with self.assertRaisesRegex(VerifiedRoutingError, "Non-finite"):
                OutcomeStore(path).list_records()


def _store_record() -> VerifiedOutcomeRecord:
    return build_verified_outcome(
        _bridge_metadata(evidence=[_evidence("check-a", passed=True)]),
        _signals(),
        created_at="2026-07-19T03:00:00+00:00",
    )


def _signals(*, provider_config_sha256: str = _DIGEST_E) -> TaskSignals:
    return TaskSignals(
        request_fingerprint=_DIGEST_A,
        capabilities=("tests", "code"),
        difficulty="complex",
        confidence=0.9,
        abstained=False,
        source="task-metadata-v1",
        objective_chars=1200,
        context_tokens=400,
        constraint_count=2,
        tool_count=2,
        provider_config_sha256=provider_config_sha256,
    )


def _bridge_metadata(
    *,
    evidence: list[dict[str, object]],
    prior_evidence: list[dict[str, object]] | None = None,
    commands: list[dict[str, object]] | None = None,
    capsule: dict[str, object] | None = None,
    premium_calls: int = 0,
    required_verifier_ids: list[str] | None = None,
    status: str = "completed",
    code: str = "completed",
) -> dict[str, object]:
    local_runtime = {
        "provider_id": "local-a",
        "model": "local/model-a",
        "execution_scope": "device_only",
    }
    local_runtime["runtime_sha256"] = sha256_json(local_runtime)
    premium_runtime = {
        "provider_id": "premium-a",
        "model": "premium/model-a",
        "execution_scope": "paid_remote",
    }
    premium_runtime["runtime_sha256"] = sha256_json(premium_runtime)
    return {
        "schema_version": "2.0",
        "mode": "assistant_bridge",
        "status": status,
        "code": code,
        "route_receipt": {
            "schema_version": "2.0",
            "contract": "RouteDecisionReceipt",
            "receipt_id": "route-1234",
            "task": {
                "task_id": "task-a",
                "objective_sha256": _DIGEST_B,
                "task_fingerprint": _DIGEST_A,
                "objective_chars": 1200,
                "profile": "balanced",
                "capability_demand": {
                    "required": ["code", "tests"],
                    "tools": ["filesystem", "shell"],
                    "risk_class": "write_local",
                },
                "constraint_count": 2,
                "no_change_expected": False,
                "required_verifier_ids": (
                    ["check-a"]
                    if required_verifier_ids is None
                    else required_verifier_ids
                ),
                "allow_remote": True,
                "allow_remote_workspace": False,
                "max_premium_calls": 1,
            },
            "route": "local_then_verify",
            "local_provider": "local-a",
            "premium_provider": "premium-a",
            "local_gaps": [],
            "premium_gaps": [],
            "remote_allowed": True,
            "premium_call_budget": 1,
            "rationale_codes": ["profile_balanced"],
            "expected_flow": ["local", "verify"],
            "config_sha256": _DIGEST_C,
            "workspace": {"fingerprint": _DIGEST_D},
            "local_runtime": local_runtime,
            "premium_runtime": premium_runtime,
        },
        "verification": {"prior": prior_evidence or [], "final": evidence},
        "commands": commands or [],
        "capsule": capsule,
        "final_provider": "local-a",
        "premium_calls_used": premium_calls,
        "privacy": "metadata_only",
    }


def _evidence(
    code: str, *, passed: bool, kind: str = "command"
) -> dict[str, object]:
    return {
        "id": code,
        "verifier": "verifier-a",
        "kind": kind,
        "passed": passed,
        "code": code,
        "artifact_sha256": _DIGEST_B,
        "observed_chars": 0,
        "evidence_ref": None,
        "task_fingerprint": _DIGEST_A,
        "workspace_fingerprint": _DIGEST_D,
        "verifier_spec_sha256": _DIGEST_C,
    }


def _command(duration_ms: int, prompt_tokens: int, completion_tokens: int) -> dict[str, object]:
    return {
        "provider_id": "local-a",
        "status": "completed",
        "code": "completed",
        "returncode": 0,
        "duration_ms": duration_ms,
        "output_sha256": _DIGEST_A,
        "output_chars": 10,
        "stdout_sha256": _DIGEST_B,
        "stdout_bytes": 10,
        "stderr_sha256": None,
        "stderr_bytes": 0,
        "command_sha256": _DIGEST_C,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": None,
            "cost_status": "not_computed_without_pricing_contract",
        },
    }


def _capsule(characters: int) -> dict[str, object]:
    return {
        "capsule_id": "capsule-a",
        "sha256": _DIGEST_A,
        "characters": characters,
        "objective_sha256": _DIGEST_B,
        "constraint_count": 2,
        "verification_count": 1,
        "failure_codes": ["check-failed"],
        "diff_sha256": None,
        "redaction_count": 0,
        "residual_assured": True,
        "residual_detector": "detector-a",
        "truncated": False,
        "content_in_metadata": False,
    }


def _append_record(
    path: str,
    payload: dict[str, object],
    ready: object,
    start: object,
    results: object,
) -> None:
    try:
        ready.put(True)  # type: ignore[attr-defined]
        if not start.wait(15.0):  # type: ignore[attr-defined]
            raise RuntimeError("concurrent append start gate timed out")
        appended = OutcomeStore(path).append(
            VerifiedOutcomeRecord.from_payload(payload)
        )
        results.put((True, appended))  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - surfaced in the parent process
        results.put((False, f"{type(exc).__name__}: {exc}"))  # type: ignore[attr-defined]


def _run_concurrent_appends(
    path: Path,
    payloads: list[dict[str, object]],
) -> list[bool]:
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_append_record,
            args=(str(path), payload, ready, start, results),
        )
        for payload in payloads
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            if ready.get(timeout=15.0) is not True:
                raise AssertionError("concurrent append worker did not become ready")
        start.set()
        observed = [results.get(timeout=20.0) for _ in processes]
        failures = [value for success, value in observed if not success]
        if failures:
            raise AssertionError(f"concurrent append worker failed: {failures[0]}")
        return [bool(value) for success, value in observed if success]
    finally:
        start.set()
        for process in processes:
            process.join(10.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
        ready.close()
        results.close()
        ready.join_thread()
        results.join_thread()


def _hold_lock(lock_path: str, ready: object, release: object) -> None:
    from filelock import FileLock

    with FileLock(lock_path, timeout=10.0, mode=0o600):
        ready.set()  # type: ignore[attr-defined]
        if not release.wait(15.0):  # type: ignore[attr-defined]
            raise RuntimeError("lock release gate timed out")


def _outcome_windows_identity(
    value: int,
    *,
    attributes: int,
):
    from local_moe._win32_fs import Win32FileIdentity

    return Win32FileIdentity(
        volume_serial=7,
        file_id=bytes([value]) * 16,
        attributes=attributes,
        reparse_tag=0,
    )


def _outcome_fake_msvcrt() -> types.ModuleType:
    module = types.ModuleType("msvcrt")
    module.LK_NBLCK = 1  # type: ignore[attr-defined]
    module.LK_UNLCK = 2  # type: ignore[attr-defined]
    module.locking = mock.Mock()  # type: ignore[attr-defined]
    return module


if __name__ == "__main__":
    unittest.main()
