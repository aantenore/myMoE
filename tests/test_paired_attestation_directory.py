from __future__ import annotations

import base64
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import (
    ED25519_DSSE_ADAPTER_ID,
    TrustedEd25519Verifier,
    create_ed25519_dsse_envelope,
    ed25519_public_key_sha256,
)
from local_moe.assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from local_moe.assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    AttestationCheck,
    CandidateBinding,
    VerificationPolicy,
    VerifierRequirement,
)
from local_moe.paired_attestation_directory import (
    DIRECTORY_PAIRED_ATTESTATION_REQUEST_CONTRACT,
    DIRECTORY_PAIRED_ATTESTATION_RESPONSE_CONTRACT,
    DirectoryPairedAttestationError,
    DirectoryPairedAttestationProducer,
    DirectoryPairedAttestationTimeout,
    _atomic_no_clobber,
)
from local_moe import _win32_fs
from local_moe import paired_attestation_directory as directory_adapter


class DirectoryPairedAttestationProducerTests(unittest.TestCase):
    def test_watcher_returns_real_signed_dsse_for_exact_canonical_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "verifier-workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.005,
                maximum_wait_seconds=5.0,
            )
            observed: list[dict[str, object]] = []

            def respond(request: dict[str, object]) -> None:
                observed.append(request)
                request_binding = CandidateBinding.from_payload(request["binding"])
                envelope = _envelope(
                    request_binding,
                    requirement,
                    private_key,
                )
                _write_response(exchange, request, (envelope,))

            watcher, errors = _watch_once(exchange, respond)
            try:
                envelopes = producer.attest(
                    binding,
                    workspace,
                    time.time() + 2.0,
                )
            finally:
                _finish_watcher(watcher, errors)

            self.assertEqual(len(envelopes), 1)
            verified = TrustedEd25519Verifier(
                requirement,
                private_key.public_key(),
            ).verify(binding, envelopes[0], now=time.time())
            self.assertEqual(verified.verifier_id, requirement.verifier_id)
            request = observed[0]
            self.assertEqual(
                request["contract"],
                DIRECTORY_PAIRED_ATTESTATION_REQUEST_CONTRACT,
            )
            self.assertEqual(request["binding"], binding.payload())
            self.assertEqual(request["bindingSha256"], binding.binding_sha256)
            self.assertEqual(request["verifierWorkspacePath"], str(workspace))
            self.assertRegex(str(request["requestId"]), r"^[0-9a-f]{64}$")
            request_path = next((exchange / "requests").iterdir())
            self.assertEqual(
                request_path.read_bytes(),
                canonical_json_bytes(request),
            )
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(request_path.stat().st_mode), 0o600)
            self.assertEqual(request_path.stat().st_nlink, 1)
            self.assertNotIn("PRIVATE KEY", request_path.read_text(encoding="utf-8"))
            self.assertEqual(producer.state_paths, (exchange,))
            self.assertEqual(len(producer.configuration_sha256), 64)

    @unittest.skipUnless(os.name == "posix", "POSIX link publication semantics")
    def test_reader_waits_for_atomic_response_link_to_settle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "verifier-workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.005,
                maximum_wait_seconds=1.0,
            )
            publication_observed = threading.Event()
            real_unlink = os.unlink
            real_pending = directory_adapter._posix_atomic_publication_pending

            def delayed_response_unlink(path, *args, **kwargs) -> None:
                candidate = Path(path)
                if (
                    candidate.parent == exchange / "responses"
                    and candidate.name.startswith(".response-")
                    and candidate.name.endswith(".tmp")
                ):
                    if not publication_observed.wait(timeout=1.0):
                        raise AssertionError(
                            "reader did not observe the linked publication state"
                        )
                real_unlink(path, *args, **kwargs)

            def observe_pending(path, metadata, *, maximum_bytes):
                pending = real_pending(
                    path,
                    metadata,
                    maximum_bytes=maximum_bytes,
                )
                if pending:
                    publication_observed.set()
                return pending

            def respond(request: dict[str, object]) -> None:
                envelope = _envelope(binding, requirement, private_key)
                _write_response(exchange, request, (envelope,))

            watcher, errors = _watch_once(exchange, respond)
            with (
                patch.object(
                    directory_adapter.os,
                    "unlink",
                    side_effect=delayed_response_unlink,
                ),
                patch.object(
                    directory_adapter,
                    "_posix_atomic_publication_pending",
                    side_effect=observe_pending,
                ),
            ):
                envelopes = producer.attest(
                    binding,
                    workspace,
                    time.time() + 0.5,
                )
            _finish_watcher(watcher, errors)

            self.assertTrue(publication_observed.is_set())
            self.assertEqual(len(envelopes), 1)
            response_path = next((exchange / "responses").iterdir())
            self.assertEqual(response_path.stat().st_nlink, 1)

    @unittest.skipUnless(os.name == "posix", "POSIX link publication semantics")
    def test_matching_temporary_hardlink_times_out_without_being_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "verifier-workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.005,
                maximum_wait_seconds=1.0,
            )
            response_reads: list[Path] = []
            real_read = directory_adapter._read_secure_file

            def respond(request: dict[str, object]) -> None:
                envelope = _envelope(binding, requirement, private_key)
                response = _response(request, (envelope,))
                target = (
                    exchange
                    / "responses"
                    / f"response-{request['requestId']}.json"
                )
                source = target.parent / f".{target.name}.{'a' * 32}.tmp"
                source.write_bytes(response)
                source.chmod(0o600)
                os.link(source, target, follow_symlinks=False)

            def record_read(path: Path, *, maximum_bytes: int, label: str) -> bytes:
                if path.parent == exchange / "responses":
                    response_reads.append(path)
                return real_read(
                    path,
                    maximum_bytes=maximum_bytes,
                    label=label,
                )

            watcher, errors = _watch_once(exchange, respond)
            with (
                patch.object(
                    directory_adapter,
                    "_read_secure_file",
                    side_effect=record_read,
                ),
                self.assertRaises(DirectoryPairedAttestationTimeout),
            ):
                producer.attest(
                    binding,
                    workspace,
                    time.time() + 0.1,
                )
            _finish_watcher(watcher, errors)

            self.assertEqual(response_reads, [])

    def test_replayed_response_for_another_request_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.005,
                maximum_wait_seconds=5.0,
            )
            captured: list[bytes] = []
            real_publish = directory_adapter._atomic_no_clobber

            def publish_first_response(path: Path, value: bytes) -> None:
                real_publish(path, value)
                if path.parent == exchange / "requests":
                    request = json.loads(value)
                    response = _response(
                        request,
                        (_envelope(binding, requirement, private_key),),
                    )
                    captured.append(response)
                    response_path = exchange / "responses" / (
                        f"response-{request['requestId']}.json"
                    )
                    real_publish(response_path, response)

            with patch.object(
                directory_adapter,
                "_atomic_no_clobber",
                side_effect=publish_first_response,
            ):
                producer.attest(binding, workspace, time.time() + 2.0)

            def publish_replayed_response(path: Path, value: bytes) -> None:
                real_publish(path, value)
                if path.parent == exchange / "requests":
                    request = json.loads(value)
                    response_path = exchange / "responses" / (
                        f"response-{request['requestId']}.json"
                    )
                    real_publish(response_path, captured[0])

            with (
                patch.object(
                    directory_adapter,
                    "_atomic_no_clobber",
                    side_effect=publish_replayed_response,
                ),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "exact request binding",
                ),
            ):
                producer.attest(binding, workspace, time.time() + 2.0)

    def test_response_symlink_hardlink_and_permissive_mode_fail_closed(self) -> None:
        modes = (
            ("hardlink",)
            if os.name == "nt"
            else ("symlink", "hardlink", "permissive")
        )
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                exchange = _exchange(root / "exchange")
                workspace = _private_directory(root / "workspace")
                binding, requirement, private_key = _binding()
                producer = _producer(exchange)

                def respond(request: dict[str, object]) -> None:
                    response = _response(
                        request,
                        (_envelope(binding, requirement, private_key),),
                    )
                    _publish_response(exchange, request, response, mode=mode)

                watcher, errors = _watch_once(exchange, respond)
                with self.assertRaises(DirectoryPairedAttestationError):
                    producer.attest(binding, workspace, time.time() + 0.5)
                _finish_watcher(watcher, errors)

    def test_windows_publication_is_single_link_and_first_writer_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            observed_links: list[int] = []
            winners: list[bytes] = []
            failures: list[BaseException] = []
            barrier = threading.Barrier(3)
            move_lock = threading.Lock()
            real_rename = os.rename

            def windows_move(source, destination, *, write_through=True) -> None:
                self.assertTrue(write_through)
                with move_lock:
                    if os.path.exists(destination):
                        raise FileExistsError(os.fspath(destination))
                    real_rename(source, destination)
                    observed_links.append(os.lstat(destination).st_nlink)

            def publish(value: bytes) -> None:
                barrier.wait()
                try:
                    _atomic_no_clobber(target, value)
                    winners.append(value)
                except BaseException as exc:
                    failures.append(exc)

            with (
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(
                    _win32_fs,
                    "move_no_replace",
                    side_effect=windows_move,
                ),
                patch.object(
                    directory_adapter.os,
                    "link",
                    side_effect=AssertionError("Windows publication used a hard link"),
                ),
                patch.object(
                    directory_adapter,
                    "_read_secure_file",
                    side_effect=lambda path, **_: path.read_bytes(),
                ),
            ):
                threads = (
                    threading.Thread(target=publish, args=(b"first",)),
                    threading.Thread(target=publish, args=(b"second",)),
                )
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())

            self.assertEqual(observed_links, [1])
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], DirectoryPairedAttestationError)
            self.assertIn("overwrite is forbidden", str(failures[0]))
            self.assertEqual(target.read_bytes(), winners[0])

    @unittest.skipUnless(os.name == "nt", "native Windows move semantics")
    def test_native_windows_publication_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            winners: list[bytes] = []
            failures: list[BaseException] = []
            barrier = threading.Barrier(3)

            def publish(value: bytes) -> None:
                barrier.wait()
                try:
                    _atomic_no_clobber(target, value)
                    winners.append(value)
                except BaseException as exc:
                    failures.append(exc)

            threads = (
                threading.Thread(target=publish, args=(b"first",)),
                threading.Thread(target=publish, args=(b"second",)),
            )
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

            self.assertEqual(len(winners), 1)
            self.assertEqual(len(failures), 1)
            self.assertIsInstance(failures[0], DirectoryPairedAttestationError)
            self.assertIn("overwrite is forbidden", str(failures[0]))
            self.assertEqual(target.read_bytes(), winners[0])
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertEqual(
                tuple(target.parent.glob(f".{target.name}.*.tmp")),
                (),
            )

    def test_windows_reparse_response_is_rejected_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            target.write_bytes(b"x")
            observed = target.lstat()
            reparse = SimpleNamespace(
                st_mode=observed.st_mode,
                st_nlink=1,
                st_size=observed.st_size,
                st_uid=getattr(observed, "st_uid", 0),
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_mtime_ns=observed.st_mtime_ns,
                st_file_attributes=getattr(
                    stat,
                    "FILE_ATTRIBUTE_REPARSE_POINT",
                    0x400,
                ),
            )

            with (
                patch.object(Path, "lstat", return_value=reparse),
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(_win32_fs, "open_nofollow_fd") as open_file,
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "bounded single-link regular file",
                ),
            ):
                directory_adapter._read_secure_file(
                    target,
                    maximum_bytes=1024,
                    label="attestation response",
                )
            open_file.assert_not_called()

    def test_windows_response_path_reparse_after_read_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            target.write_bytes(b"response")
            observed = target.lstat()
            reparse = _metadata_with(
                observed,
                st_file_attributes=getattr(
                    stat,
                    "FILE_ATTRIBUTE_REPARSE_POINT",
                    0x400,
                ),
            )
            identity = _win32_identity(1)
            descriptors: list[int] = []

            def open_file(path, **kwargs):
                del kwargs
                descriptor = os.open(path, os.O_RDONLY)
                descriptors.append(descriptor)
                return descriptor, identity

            with (
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(
                    Path,
                    "lstat",
                    side_effect=(observed, observed, reparse),
                ),
                patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_file,
                ),
                patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    return_value=identity,
                ),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "pathname became a reparse point",
                ),
            ):
                directory_adapter._read_secure_file(
                    target,
                    maximum_bytes=1024,
                    label="attestation response",
                )

            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_windows_response_archive_bit_change_preserves_file_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            target.write_bytes(b"response")
            original = _win32_identity(2)
            archived = _win32_identity(2, attributes=0x20)

            def open_file(path, **kwargs):
                del kwargs
                return os.open(path, os.O_RDONLY), original

            with (
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_file,
                ),
                patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    return_value=archived,
                ),
            ):
                observed = directory_adapter._read_secure_file(
                    target,
                    maximum_bytes=1024,
                    label="attestation response",
                )

            self.assertEqual(observed, b"response")

    def test_windows_response_file_id_swap_on_reopen_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "response.json"
            target.write_bytes(b"response")
            identities = (_win32_identity(3), _win32_identity(4))
            descriptors: list[int] = []
            identity_by_descriptor: dict[int, _win32_fs.Win32FileIdentity] = {}

            def open_file(path, **kwargs):
                del kwargs
                descriptor = os.open(path, os.O_RDONLY)
                identity = identities[len(descriptors)]
                descriptors.append(descriptor)
                identity_by_descriptor[descriptor] = identity
                return descriptor, identity

            with (
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_file,
                ),
                patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    side_effect=lambda descriptor: identity_by_descriptor[descriptor],
                ),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "pathname no longer resolves to the pinned file ID",
                ),
            ):
                directory_adapter._read_secure_file(
                    target,
                    maximum_bytes=1024,
                    label="attestation response",
                )

            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_windows_workspace_swap_between_lstat_and_open_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve(strict=True)
            descriptor_source = workspace / "descriptor-source"
            descriptor_source.touch()
            observed = workspace.lstat()
            swapped = _metadata_with(observed, st_ino=observed.st_ino + 1)
            identity = _win32_identity(2)
            descriptors: list[int] = []

            def open_directory(path, **kwargs):
                del path, kwargs
                descriptor = os.open(descriptor_source, os.O_RDONLY)
                descriptors.append(descriptor)
                return descriptor, identity

            with (
                ExitStack() as held_paths,
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(Path, "resolve", return_value=workspace),
                patch.object(Path, "lstat", return_value=swapped),
                patch.object(
                    directory_adapter,
                    "_same_windows_path",
                    return_value=True,
                ),
                patch.object(
                    _win32_fs,
                    "open_nofollow_fd",
                    side_effect=open_directory,
                ),
                patch.object(
                    _win32_fs,
                    "identity_from_fd",
                    return_value=identity,
                ),
                patch.object(directory_adapter.os, "fstat", return_value=observed),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "identity changed during no-follow open",
                ),
            ):
                directory_adapter._pin_verifier_workspace(
                    workspace,
                    held_paths=held_paths,
                )

            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_windows_workspace_resolved_reparse_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            resolved_elsewhere = Path(temporary) / "junction-target"
            resolved_elsewhere.mkdir()
            observed = workspace.lstat()

            with (
                patch.object(directory_adapter, "_OS_NAME", "nt"),
                patch.object(Path, "lstat", return_value=observed),
                patch.object(Path, "resolve", return_value=resolved_elsewhere),
                patch.object(_win32_fs, "open_nofollow_fd") as open_file,
                self.assertRaisesRegex(
                    DirectoryPairedAttestationError,
                    "cannot traverse a reparse point",
                ),
            ):
                directory_adapter._pin_verifier_workspace(workspace)

            open_file.assert_not_called()

    def test_attest_holds_all_pinned_handles_until_exchange_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, _, _ = _binding()
            producer = _producer(exchange)
            markers = tuple(root / f"held-{index}" for index in range(4))
            for marker in markers:
                marker.write_bytes(b"pin")
            descriptors: list[int] = []

            def retain(marker: Path, held_paths: ExitStack) -> None:
                descriptor = os.open(marker, os.O_RDONLY)
                descriptors.append(descriptor)
                held_paths.callback(os.close, descriptor)

            def pin_workspace(value: Path, *, held_paths: ExitStack) -> Path:
                self.assertEqual(value, workspace)
                retain(markers[0], held_paths)
                return workspace

            def hold_directories(held_paths: ExitStack) -> None:
                for marker in markers[1:]:
                    retain(marker, held_paths)

            def attest_pinned(*args) -> tuple[bytes, ...]:
                del args
                self.assertEqual(len(descriptors), 4)
                for descriptor in descriptors:
                    os.fstat(descriptor)
                return (b"signed",)

            with (
                patch.object(
                    directory_adapter,
                    "_pin_verifier_workspace",
                    side_effect=pin_workspace,
                ),
                patch.object(
                    producer,
                    "_hold_directories",
                    side_effect=hold_directories,
                ),
                patch.object(
                    producer,
                    "_attest_pinned",
                    side_effect=attest_pinned,
                ),
            ):
                result = producer.attest(
                    binding,
                    workspace,
                    time.time() + 0.5,
                )

            self.assertEqual(result, (b"signed",))
            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_response_replaced_by_fifo_between_lstat_and_open_never_blocks(
        self,
    ) -> None:
        if (
            os.name != "posix"
            or not hasattr(os, "mkfifo")
            or not getattr(os, "O_NONBLOCK", 0)
        ):
            self.skipTest("POSIX nonblocking FIFO hardening contract")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.05,
                maximum_wait_seconds=1.0,
            )
            response_target: list[Path] = []
            matching_lstats = 0
            swapped = False
            nonblocking_opened = False
            real_lstat = Path.lstat
            real_open = os.open

            def publish_without_adapter_read(request: dict[str, object]) -> None:
                target = exchange / "responses" / (
                    f"response-{request['requestId']}.json"
                )
                response_target.append(target)
                staging = target.with_suffix(".staging")
                staging.write_bytes(
                    _response(
                        request,
                        (_envelope(binding, requirement, private_key),),
                    )
                )
                staging.chmod(0o600)
                os.replace(staging, target)

            def swap_after_second_lstat(path: Path, *args, **kwargs):
                nonlocal matching_lstats, swapped
                details = real_lstat(path, *args, **kwargs)
                if response_target and path == response_target[0] and not swapped:
                    matching_lstats += 1
                    if matching_lstats == 2:
                        path.unlink()
                        os.mkfifo(path, 0o600)
                        swapped = True
                return details

            def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
                nonlocal nonblocking_opened
                if (
                    response_target
                    and Path(path) == response_target[0]
                    and swapped
                ):
                    if not flags & os.O_NONBLOCK:
                        raise AssertionError(
                            "response reads must use O_NONBLOCK after lstat"
                        )
                    nonblocking_opened = True
                options = {} if dir_fd is None else {"dir_fd": dir_fd}
                return real_open(path, flags, mode, **options)

            watcher, errors = _watch_once(exchange, publish_without_adapter_read)
            with patch.object(
                Path,
                "lstat",
                new=swap_after_second_lstat,
            ), patch.object(directory_adapter.os, "open", new=guarded_open):
                with self.assertRaises(DirectoryPairedAttestationError):
                    producer.attest(binding, workspace, time.time() + 0.5)
            _finish_watcher(watcher, errors)
            self.assertTrue(swapped)
            self.assertTrue(nonblocking_opened)

    def test_wrong_binding_and_noncanonical_response_are_rejected(self) -> None:
        for mode in ("wrong-binding", "noncanonical"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                exchange = _exchange(root / "exchange")
                workspace = _private_directory(root / "workspace")
                binding, requirement, private_key = _binding()
                producer = _producer(exchange)

                def respond(request: dict[str, object]) -> None:
                    envelope = _envelope(binding, requirement, private_key)
                    raw = {
                        "schemaVersion": "1.0",
                        "contract": DIRECTORY_PAIRED_ATTESTATION_RESPONSE_CONTRACT,
                        "requestId": request["requestId"],
                        "bindingSha256": (
                            "f" * 64
                            if mode == "wrong-binding"
                            else request["bindingSha256"]
                        ),
                        "envelopes": [base64.b64encode(envelope).decode("ascii")],
                    }
                    value = (
                        json.dumps(raw, indent=2).encode("utf-8")
                        if mode == "noncanonical"
                        else canonical_json_bytes(raw)
                    )
                    _publish_response(exchange, request, value)

                watcher, errors = _watch_once(exchange, respond)
                with self.assertRaises(DirectoryPairedAttestationError):
                    producer.attest(binding, workspace, time.time() + 0.5)
                _finish_watcher(watcher, errors)

    def test_timeout_leaves_request_and_never_invents_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, _, _ = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.002,
                maximum_wait_seconds=0.1,
            )

            with self.assertRaises(DirectoryPairedAttestationTimeout):
                producer.attest(binding, workspace, time.time() + 0.03)

            self.assertEqual(len(tuple((exchange / "requests").iterdir())), 1)
            self.assertEqual(len(tuple((exchange / "responses").iterdir())), 0)

    def test_path_pinning_cannot_reset_the_absolute_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, _, _ = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                maximum_wait_seconds=1.0,
            )
            real_pin = directory_adapter._pin_verifier_workspace

            def delayed_pin(value: Path, *, held_paths: ExitStack) -> Path:
                time.sleep(0.03)
                return real_pin(value, held_paths=held_paths)

            with (
                patch.object(
                    directory_adapter,
                    "_pin_verifier_workspace",
                    side_effect=delayed_pin,
                ),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationTimeout,
                    "before publishing",
                ),
            ):
                producer.attest(binding, workspace, time.time() + 0.01)

            self.assertEqual(tuple((exchange / "requests").iterdir()), ())

    def test_response_read_after_deadline_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, requirement, private_key = _binding()
            producer = DirectoryPairedAttestationProducer(
                exchange,
                poll_interval_seconds=0.002,
                maximum_wait_seconds=5.0,
            )
            envelope = _envelope(binding, requirement, private_key)

            real_publish = directory_adapter._atomic_no_clobber
            real_read = directory_adapter._read_secure_file
            monotonic_value = 100.0

            def controlled_monotonic() -> float:
                return monotonic_value

            def publish_request_and_response(path: Path, value: bytes) -> None:
                real_publish(path, value)
                if path.parent == exchange / "requests":
                    request = json.loads(value)
                    response_path = (
                        exchange
                        / "responses"
                        / f"response-{request['requestId']}.json"
                    )
                    real_publish(
                        response_path,
                        _response(request, (envelope,)),
                    )

            def delayed_read(path: Path, *, maximum_bytes: int, label: str) -> bytes:
                nonlocal monotonic_value
                if label == "attestation response":
                    monotonic_value = 110.0
                return real_read(
                    path,
                    maximum_bytes=maximum_bytes,
                    label=label,
                )

            with (
                patch.object(
                    directory_adapter,
                    "_atomic_no_clobber",
                    side_effect=publish_request_and_response,
                ),
                patch.object(
                    directory_adapter,
                    "_read_secure_file",
                    side_effect=delayed_read,
                ),
                patch.object(
                    directory_adapter.time,
                    "monotonic",
                    side_effect=controlled_monotonic,
                ),
                patch.object(
                    directory_adapter.time,
                    "time",
                    return_value=1_000.0,
                ),
                self.assertRaisesRegex(
                    DirectoryPairedAttestationTimeout,
                    "after its deadline",
                ),
            ):
                producer.attest(binding, workspace, 1_004.0)

    def test_response_byte_and_envelope_bounds_are_enforced(self) -> None:
        for mode in ("bytes", "envelopes"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                exchange = _exchange(root / "exchange")
                workspace = _private_directory(root / "workspace")
                binding, requirement, private_key = _binding()
                producer = DirectoryPairedAttestationProducer(
                    exchange,
                    poll_interval_seconds=0.005,
                    maximum_wait_seconds=1.0,
                    maximum_response_bytes=(1024 if mode == "bytes" else 64 * 1024),
                    maximum_envelopes=1,
                )

                def respond(request: dict[str, object]) -> None:
                    envelope = _envelope(binding, requirement, private_key)
                    if mode == "bytes":
                        value = b"{" + b" " * 2048 + b"}"
                    else:
                        value = _response(request, (envelope, envelope))
                    _publish_response(exchange, request, value)

                watcher, errors = _watch_once(exchange, respond)
                with self.assertRaises(DirectoryPairedAttestationError):
                    producer.attest(binding, workspace, time.time() + 0.5)
                _finish_watcher(watcher, errors)

    def test_exchange_configuration_rejects_links_and_unsafe_modes(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX link and permission contract")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = _exchange(root / "real")
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(DirectoryPairedAttestationError):
                DirectoryPairedAttestationProducer(linked)

            real.chmod(0o755)
            with self.assertRaisesRegex(
                DirectoryPairedAttestationError,
                "owner-only",
            ):
                DirectoryPairedAttestationProducer(real)

    def test_configuration_digest_binds_path_poll_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _exchange(root / "first")
            second = _exchange(root / "second")
            baseline = DirectoryPairedAttestationProducer(first)
            variants = (
                DirectoryPairedAttestationProducer(second),
                DirectoryPairedAttestationProducer(
                    first,
                    poll_interval_seconds=0.1,
                ),
                DirectoryPairedAttestationProducer(
                    first,
                    maximum_response_bytes=1024 * 1024,
                ),
            )
            for variant in variants:
                self.assertNotEqual(
                    baseline.configuration_sha256,
                    variant.configuration_sha256,
                )


def _metadata_with(observed, **overrides):
    values = {
        "st_mode": observed.st_mode,
        "st_nlink": observed.st_nlink,
        "st_size": observed.st_size,
        "st_uid": getattr(observed, "st_uid", 0),
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_mtime_ns": observed.st_mtime_ns,
        "st_file_attributes": int(
            getattr(observed, "st_file_attributes", 0)
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _win32_identity(
    marker: int,
    *,
    attributes: int = 0,
    reparse_tag: int = 0,
) -> _win32_fs.Win32FileIdentity:
    return _win32_fs.Win32FileIdentity(
        volume_serial=marker,
        file_id=bytes([marker]) * 16,
        attributes=attributes,
        reparse_tag=reparse_tag,
    )


def _producer(exchange: Path) -> DirectoryPairedAttestationProducer:
    return DirectoryPairedAttestationProducer(
        exchange,
        poll_interval_seconds=0.005,
        maximum_wait_seconds=1.0,
    )


def _binding() -> tuple[
    CandidateBinding,
    VerifierRequirement,
    Ed25519PrivateKey,
]:
    private_key = Ed25519PrivateKey.generate()
    requirement = VerifierRequirement(
        verifier_id="sidecar-tests",
        adapter_id=ED25519_DSSE_ADAPTER_ID,
        key_id="sidecar-tests-key",
        public_key_sha256=ed25519_public_key_sha256(private_key.public_key()),
        spec_sha256="e" * 64,
    )
    policy = VerificationPolicy("sidecar-policy", 1, (requirement,))
    now = time.time()
    binding = CandidateBinding(
        workflow_id="paired-sidecar-tests",
        stage_idempotency_sha256="a" * 64,
        task_fingerprint="b" * 64,
        config_sha256="c" * 64,
        source_fingerprint="d" * 64,
        challenge_sha256="f" * 64,
        manifest=ArtifactDescriptor("application/json", "1" * 64, 10),
        changeset=ArtifactDescriptor("application/json", "2" * 64, 10),
        verification_policy=policy,
        created_at=now - 1,
        expires_at=now + 30,
    )
    return binding, requirement, private_key


def _envelope(
    binding: CandidateBinding,
    requirement: VerifierRequirement,
    private_key: Ed25519PrivateKey,
) -> bytes:
    now = time.time()
    return create_ed25519_dsse_envelope(
        binding,
        requirement,
        private_key,
        attestation_id=f"sidecar-{secrets_token()}",
        issued_at=now,
        expires_at=min(binding.expires_at, now + 20),
        checks=(
            AttestationCheck(
                "sidecar-tests",
                True,
                sha256_bytes(b"sidecar-tests-passed"),
            ),
        ),
    )


def secrets_token() -> str:
    return sha256_bytes(str(time.time_ns()).encode("ascii"))[:24]


def _exchange(path: Path) -> Path:
    root = _private_directory(path)
    _private_directory(root / "requests")
    _private_directory(root / "responses")
    return root


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)
    return path.resolve(strict=True)


def _watch_once(exchange: Path, callback):
    errors: list[BaseException] = []

    def watch() -> None:
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                requests = sorted((exchange / "requests").glob("request-*.json"))
                pending = [
                    request
                    for request in requests
                    if not (
                        exchange
                        / "responses"
                        / request.name.replace("request-", "response-")
                    ).exists()
                ]
                if pending:
                    value = pending[-1].read_bytes()
                    request = json.loads(value)
                    if canonical_json_bytes(request) != value:
                        raise AssertionError("request is not canonical")
                    callback(request)
                    return
                time.sleep(0.002)
            raise AssertionError("sidecar request did not arrive")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread, errors


def _finish_watcher(thread: threading.Thread, errors: list[BaseException]) -> None:
    thread.join(timeout=6)
    if thread.is_alive():
        raise AssertionError("sidecar watcher did not terminate")
    if errors:
        raise errors[0]


def _response(
    request: dict[str, object],
    envelopes: tuple[bytes, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "schemaVersion": "1.0",
            "contract": DIRECTORY_PAIRED_ATTESTATION_RESPONSE_CONTRACT,
            "requestId": request["requestId"],
            "bindingSha256": request["bindingSha256"],
            "envelopes": [
                base64.b64encode(envelope).decode("ascii")
                for envelope in envelopes
            ],
        }
    )


def _write_response(
    exchange: Path,
    request: dict[str, object],
    envelopes: tuple[bytes, ...],
) -> None:
    _publish_response(exchange, request, _response(request, envelopes))


def _publish_response(
    exchange: Path,
    request: dict[str, object],
    value: bytes,
    *,
    mode: str = "normal",
) -> None:
    target = exchange / "responses" / f"response-{request['requestId']}.json"
    if mode == "normal":
        _atomic_no_clobber(target, value)
        return
    source = exchange / "responses" / f"source-{request['requestId']}.json"
    source.write_bytes(value)
    source.chmod(0o600 if mode != "permissive" else 0o644)
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "permissive":
        os.replace(source, target)
    else:
        os.link(source, target, follow_symlinks=False)


if __name__ == "__main__":
    unittest.main()
