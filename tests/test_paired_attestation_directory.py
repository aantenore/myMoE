from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
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
                maximum_wait_seconds=1.0,
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
            envelopes = producer.attest(
                binding,
                workspace,
                time.time() + 0.5,
            )
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

    def test_replayed_response_for_another_request_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = _exchange(root / "exchange")
            workspace = _private_directory(root / "workspace")
            binding, requirement, private_key = _binding()
            producer = _producer(exchange)
            captured: list[bytes] = []

            def first(request: dict[str, object]) -> None:
                envelope = _envelope(binding, requirement, private_key)
                response = _response(request, (envelope,))
                captured.append(response)
                _publish_response(exchange, request, response)

            watcher, errors = _watch_once(exchange, first)
            producer.attest(binding, workspace, time.time() + 0.5)
            _finish_watcher(watcher, errors)

            def replay(request: dict[str, object]) -> None:
                _publish_response(exchange, request, captured[0])

            watcher, errors = _watch_once(exchange, replay)
            with self.assertRaisesRegex(
                DirectoryPairedAttestationError,
                "exact request binding",
            ):
                producer.attest(binding, workspace, time.time() + 0.5)
            _finish_watcher(watcher, errors)

    def test_response_symlink_hardlink_and_permissive_mode_fail_closed(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX link and permission contract")
        for mode in ("symlink", "hardlink", "permissive"):
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
            deadline = time.monotonic() + 2
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
    thread.join(timeout=3)
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
