from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_moe.adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from local_moe.bound_cell_run import (
    BoundCellRunTransportError,
    ModelIdentityProbe,
    OpenAICompatibleLoopbackTransport,
    run_bound_cell,
)
from tests.bound_cell_run_lease_fakes import (
    FakeLeaseStore,
    claim_for,
    preview_evaluation,
    resource_snapshot,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str,
        content_type: str = "application/json",
        content_encoding: str = "identity",
    ) -> None:
        self._payload = payload
        self._url = url
        self.headers = {
            "Content-Type": content_type,
            "Content-Encoding": content_encoding,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, maximum: int) -> bytes:
        return self._payload[:maximum]


class BoundCellRunTransportTests(unittest.TestCase):
    def test_transport_is_bounded_strict_and_omits_tool_fields(self) -> None:
        requests = []

        def opener(req, *, timeout):
            requests.append((req, timeout))
            if req.get_method() == "GET":
                body = b'{"data":[{"id":"model-a"}]}'
            else:
                body = (
                    b'{"model":"model-a","choices":[{"message":{"content":"answer"}}]}'
                )
            return _Response(body, url=req.full_url)

        transport = OpenAICompatibleLoopbackTransport(opener=opener)
        probe = transport.probe_models(
            base_url="http://127.0.0.1:8000/v1",
            timeout_seconds=1,
            maximum_bytes=1024,
            maximum_models=4,
        )
        output = transport.invoke(
            base_url="http://127.0.0.1:8000/v1",
            model="model-a",
            task_text="task-secret",
            timeout_seconds=1,
            maximum_bytes=1024,
            max_output_tokens=32,
        )

        self.assertEqual(probe.model_ids, ("model-a",))
        self.assertEqual(output, "answer")
        sent = json.loads(requests[1][0].data)
        self.assertNotIn("tools", sent)
        self.assertNotIn("tool_choice", sent)
        self.assertFalse(sent["stream"])

    def test_transport_rejects_redirected_final_url(self) -> None:
        def opener(req, *, timeout):
            return _Response(
                b'{"data":[{"id":"model-a"}]}',
                url="http://127.0.0.1:8000/v1/other",
            )

        with self.assertRaises(BoundCellRunTransportError):
            OpenAICompatibleLoopbackTransport(opener=opener).probe_models(
                base_url="http://127.0.0.1:8000/v1",
                timeout_seconds=1,
                maximum_bytes=1024,
                maximum_models=4,
            )

    def test_transport_rejects_duplicate_json_and_function_calls(self) -> None:
        payloads = iter(
            [
                b'{"data":[],"data":[]}',
                b'{"model":"model-a","choices":[{"message":{"content":"x","function_call":{}}}]}',
            ]
        )

        def opener(req, *, timeout):
            return _Response(next(payloads), url=req.full_url)

        transport = OpenAICompatibleLoopbackTransport(opener=opener)
        with self.assertRaises(BoundCellRunTransportError):
            transport.probe_models(
                base_url="http://127.0.0.1:8000/v1",
                timeout_seconds=1,
                maximum_bytes=1024,
                maximum_models=4,
            )
        with self.assertRaises(BoundCellRunTransportError) as raised:
            transport.invoke(
                base_url="http://127.0.0.1:8000/v1",
                model="model-a",
                task_text="task",
                timeout_seconds=1,
                maximum_bytes=1024,
                max_output_tokens=32,
            )
        self.assertTrue(raised.exception.response_received)

    def test_transport_rejects_surrogates_tool_finish_reason_and_ambiguous_framing(
        self,
    ) -> None:
        payloads = iter(
            [
                b'{"choices":[{"message":{"content":"\\ud800"}}]}',
                b'{"choices":[{"finish_reason":"tool_calls","message":{"content":"x"}}]}',
                b'{"choices":[{"message":{"content":"x"}}]}',
            ]
        )

        def opener(req, *, timeout):
            response = _Response(next(payloads), url=req.full_url)
            if (
                response._payload.endswith(b'"x"}}]}')
                and b"finish_reason" not in response._payload
            ):
                response.headers["Transfer-Encoding"] = "chunked"
                response.headers["Content-Length"] = str(len(response._payload))
            return response

        transport = OpenAICompatibleLoopbackTransport(opener=opener)
        for _ in range(3):
            with self.assertRaises(BoundCellRunTransportError) as raised:
                transport.invoke(
                    base_url="http://127.0.0.1:8000/v1",
                    model="model-a",
                    task_text="task",
                    timeout_seconds=1,
                    maximum_bytes=1024,
                    max_output_tokens=32,
                )
            self.assertTrue(raised.exception.response_received)

    def test_transport_requires_an_explicit_numeric_loopback_authority(self) -> None:
        transport = OpenAICompatibleLoopbackTransport(
            opener=lambda *_args, **_kwargs: self.fail("opener must not be called")
        )
        for endpoint in (
            "http://localhost:8000/v1",
            "https://127.0.0.1:8000/v1",
            "http://127.0.0.1/v1",
            "http://127.0.0.1:8000/other",
            "http://127.0.0.1:8000/v1?x=1",
        ):
            with (
                self.subTest(endpoint=endpoint),
                self.assertRaises(BoundCellRunTransportError),
            ):
                transport.probe_models(
                    base_url=endpoint,
                    timeout_seconds=1,
                    maximum_bytes=1024,
                    maximum_models=4,
                )


class BoundCellRunServiceTests(unittest.TestCase):
    def _preview(self) -> AdaptiveCellExecutionPreviewReceipt:
        return AdaptiveCellExecutionPreviewReceipt(
            source_advisor_receipt_sha256=SHA_A,
            source_request_sha256=SHA_A,
            fresh_advisor_receipt_sha256=SHA_A,
            fresh_request_sha256=SHA_A,
            policy_sha256=SHA_A,
            evaluated_at="2026-07-22T12:00:00+00:00",
            source_selected_cell_id="cell-a",
            fresh_selected_cell_id="cell-a",
            source_passport_sha256=SHA_B,
            fresh_passport_sha256=SHA_B,
            fresh_resource_snapshot_sha256=SHA_A,
            status="admission_passed",
            reason_codes=(),
            task_chars=4,
        )

    def test_preview_is_last_gate_and_invalidated_output_remains_available(
        self,
    ) -> None:
        events: list[str] = []
        target = SimpleNamespace(
            request=SimpleNamespace(
                digest=SHA_B,
                catalog_path="catalog.json",
                runtime_config_path="runtime.json",
            ),
            passport=SimpleNamespace(
                digest=SHA_B,
                declaration=SimpleNamespace(digest=SHA_D),
            ),
            expert=SimpleNamespace(
                id="expert-a",
                model="model-a",
                base_url="http://127.0.0.1:8000/v1",
            ),
            config_source_sha256=SHA_C,
            runtime_config_sha256=SHA_C,
        )
        bundles = iter(
            [
                SimpleNamespace(
                    manifest=SimpleNamespace(digest=SHA_A),
                    request_sha256=SHA_B,
                    publication_protected_roots=(),
                ),
                SimpleNamespace(
                    manifest=SimpleNamespace(digest=SHA_A),
                    request_sha256=SHA_B,
                    publication_protected_roots=(),
                ),
            ]
        )

        def resolver(_path):
            events.append("resolve")
            return target

        def inspector(_path, *, now, publication_path=None):
            del now, publication_path
            events.append("inspect")
            return next(bundles)

        snapshot = resource_snapshot()
        preview = self._preview()
        claim = claim_for(preview, snapshot, passport_sha256=target.passport.digest)
        lease_store = FakeLeaseStore(events=events)

        def previewer(*_args, resource_snapshot):
            self.assertIs(resource_snapshot, snapshot)
            events.append("preview")
            return preview_evaluation(preview, snapshot)

        class Transport:
            probes = 0

            def probe_models(self, **_kwargs):
                events.append("probe")
                self.probes += 1
                models = ["model-a"] if self.probes == 1 else ["model-a", "model-b"]
                return ModelIdentityProbe.from_ids(models, maximum=4)

            def invoke(self, **_kwargs):
                events.append("invoke")
                return "response-secret"

        def record(state, _bundle, *, prefix):
            state[f"{prefix}_binding_bundle_sha256"] = SHA_A
            state[f"{prefix}_binding_request_sha256"] = SHA_B
            state[f"{prefix}_binding_manifest_sha256"] = SHA_A
            state[f"{prefix}_inspection_receipt_sha256"] = SHA_A

        def clock():
            return datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

        with (
            patch("local_moe.bound_cell_run._record_binding", side_effect=record),
            patch("local_moe.bound_cell_run._validate_binding"),
            patch("local_moe.bound_cell_run._validate_preview_binding"),
            patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ),
        ):
            result = run_bound_cell(
                "advisor.json",
                "task",
                "catalog.json",
                "evaluation.json",
                "policy.json",
                "binding.json",
                confirmed=True,
                resolver=resolver,
                inspector=inspector,
                previewer=previewer,
                snapshot_collector=lambda: snapshot,
                lease_store=lease_store,
                transport=Transport(),
                clock=clock,
            )

        self.assertLess(events.index("inspect"), events.index("preview"))
        self.assertEqual(events[events.index("preview") + 1], "probe")
        self.assertLess(events.index("lease_arm"), events.index("invoke"))
        self.assertLess(
            events.index("invoke"),
            events.index("lease_release:response_received"),
        )
        self.assertLess(
            events.index("lease_release:response_received"),
            len(events) - 1 - events[::-1].index("probe"),
        )
        self.assertEqual(result.receipt.status, "invalidated")
        self.assertEqual(result.receipt.delivery_status, "response_received")
        self.assertEqual(result.envelope.lease_release_receipt.status, "released")
        self.assertEqual(result.response_text, "response-secret")
        serialized = json.dumps(result.receipt.payload())
        self.assertNotIn('task"', serialized)
        self.assertNotIn("response-secret", serialized)

    def test_interruption_is_receipted_after_both_post_checks(self) -> None:
        events: list[str] = []
        target = SimpleNamespace(
            request=SimpleNamespace(
                digest=SHA_B,
                catalog_path="catalog.json",
                runtime_config_path="runtime.json",
            ),
            passport=SimpleNamespace(
                digest=SHA_B,
                declaration=SimpleNamespace(digest=SHA_D),
            ),
            expert=SimpleNamespace(
                id="expert-a",
                model="model-a",
                base_url="http://127.0.0.1:8000/v1",
            ),
            config_source_sha256=SHA_C,
            runtime_config_sha256=SHA_C,
        )
        bundle = SimpleNamespace(
            manifest=SimpleNamespace(digest=SHA_A),
            request_sha256=SHA_B,
            publication_protected_roots=(),
        )

        def resolver(_path):
            events.append("resolve")
            return target

        def inspector(_path, *, now, publication_path=None):
            del now, publication_path
            events.append("inspect")
            return bundle

        snapshot = resource_snapshot()
        preview = self._preview()
        claim = claim_for(preview, snapshot, passport_sha256=target.passport.digest)
        lease_store = FakeLeaseStore(events=events)

        def previewer(*_args, resource_snapshot):
            self.assertIs(resource_snapshot, snapshot)
            events.append("preview")
            return preview_evaluation(preview, snapshot)

        class Transport:
            def probe_models(self, **_kwargs):
                events.append("probe")
                return ModelIdentityProbe.from_ids(["model-a"], maximum=4)

            def invoke(self, **_kwargs):
                events.append("invoke")
                raise KeyboardInterrupt()

        def record(state, _bundle, *, prefix):
            state[f"{prefix}_binding_bundle_sha256"] = SHA_A
            state[f"{prefix}_binding_request_sha256"] = SHA_B
            state[f"{prefix}_binding_manifest_sha256"] = SHA_A
            state[f"{prefix}_inspection_receipt_sha256"] = SHA_A

        def clock():
            return datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

        with (
            patch("local_moe.bound_cell_run._record_binding", side_effect=record),
            patch("local_moe.bound_cell_run._validate_binding"),
            patch("local_moe.bound_cell_run._validate_preview_binding"),
            patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ),
        ):
            result = run_bound_cell(
                "advisor.json",
                "task",
                "catalog.json",
                "evaluation.json",
                "policy.json",
                "binding.json",
                confirmed=True,
                resolver=resolver,
                inspector=inspector,
                previewer=previewer,
                snapshot_collector=lambda: snapshot,
                lease_store=lease_store,
                transport=Transport(),
                clock=clock,
            )

        self.assertEqual(result.receipt.status, "failed")
        self.assertEqual(result.receipt.reason_codes, ("execution_interrupted",))
        self.assertEqual(result.receipt.delivery_status, "attempted_unknown")
        self.assertEqual(
            result.envelope.lease_release_receipt.status, "unknown_blocking"
        )
        self.assertIsInstance(result.interruption, KeyboardInterrupt)
        self.assertEqual(events.count("invoke"), 1)
        self.assertEqual(events.count("probe"), 2)
        self.assertEqual(events.count("inspect"), 2)
        self.assertLess(events.index("lease_arm"), events.index("invoke"))
        self.assertLess(
            events.index("invoke"),
            events.index("lease_release:attempted_unknown"),
        )
        self.assertLess(
            events.index("lease_release:attempted_unknown"),
            len(events) - 1 - events[::-1].index("probe"),
        )

    def test_denied_lease_causes_zero_endpoint_traffic(self) -> None:
        events: list[str] = []
        target = SimpleNamespace(
            request=SimpleNamespace(
                digest=SHA_B,
                catalog_path="catalog.json",
                runtime_config_path="runtime.json",
            ),
            passport=SimpleNamespace(
                digest=SHA_B,
                declaration=SimpleNamespace(digest=SHA_D),
            ),
            expert=SimpleNamespace(
                id="expert-a",
                model="model-a",
                base_url="http://127.0.0.1:8000/v1",
            ),
            config_source_sha256=SHA_C,
            runtime_config_sha256=SHA_C,
        )
        bundle = SimpleNamespace(
            manifest=SimpleNamespace(digest=SHA_A),
            request_sha256=SHA_B,
            publication_protected_roots=(),
        )
        preview = self._preview()
        snapshot = resource_snapshot()
        claim = claim_for(preview, snapshot, passport_sha256=target.passport.digest)
        lease_store = FakeLeaseStore(admission_status="denied", events=events)

        class NoTrafficTransport:
            def probe_models(self, **_kwargs):
                self.fail("lease denial must precede GET")

            def invoke(self, **_kwargs):
                self.fail("lease denial must precede POST")

        transport = NoTrafficTransport()
        transport.fail = self.fail

        def record(state, _bundle, *, prefix):
            state[f"{prefix}_binding_bundle_sha256"] = SHA_A
            state[f"{prefix}_binding_request_sha256"] = SHA_B
            state[f"{prefix}_binding_manifest_sha256"] = SHA_A
            state[f"{prefix}_inspection_receipt_sha256"] = SHA_A

        with (
            patch("local_moe.bound_cell_run._record_binding", side_effect=record),
            patch("local_moe.bound_cell_run._validate_binding"),
            patch("local_moe.bound_cell_run._validate_preview_binding"),
            patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ),
        ):
            result = run_bound_cell(
                "advisor.json",
                "task",
                "catalog.json",
                "evaluation.json",
                "policy.json",
                "binding.json",
                confirmed=True,
                resolver=lambda _path: target,
                inspector=lambda *_args, **_kwargs: bundle,
                previewer=lambda *_args, **_kwargs: preview_evaluation(
                    preview, snapshot
                ),
                snapshot_collector=lambda: snapshot,
                lease_store=lease_store,
                transport=transport,
                clock=lambda: datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(result.receipt.status, "blocked")
        self.assertEqual(result.receipt.endpoint_probe_requests, 0)
        self.assertEqual(result.receipt.invocation_attempts, 0)
        self.assertEqual(result.envelope.lease_admission_receipt.status, "denied")
        self.assertIsNone(result.envelope.lease_transition_receipt)
        self.assertIsNone(result.envelope.lease_release_receipt)

    def test_failed_delivery_transition_releases_without_post(self) -> None:
        events: list[str] = []
        target = SimpleNamespace(
            request=SimpleNamespace(
                digest=SHA_B,
                catalog_path="catalog.json",
                runtime_config_path="runtime.json",
            ),
            passport=SimpleNamespace(
                digest=SHA_B,
                declaration=SimpleNamespace(digest=SHA_D),
            ),
            expert=SimpleNamespace(
                id="expert-a",
                model="model-a",
                base_url="http://127.0.0.1:8000/v1",
            ),
            config_source_sha256=SHA_C,
            runtime_config_sha256=SHA_C,
        )
        bundle = SimpleNamespace(
            manifest=SimpleNamespace(digest=SHA_A),
            request_sha256=SHA_B,
            publication_protected_roots=(),
        )
        preview = self._preview()
        snapshot = resource_snapshot()
        claim = claim_for(preview, snapshot, passport_sha256=target.passport.digest)
        lease_store = FakeLeaseStore(transition_applied=False, events=events)

        class ProbeOnlyTransport:
            def probe_models(self, **_kwargs):
                events.append("probe")
                return ModelIdentityProbe.from_ids(["model-a"], maximum=4)

            def invoke(self, **_kwargs):
                self.fail("a failed transition must fence the POST")

        transport = ProbeOnlyTransport()
        transport.fail = self.fail

        def record(state, _bundle, *, prefix):
            state[f"{prefix}_binding_bundle_sha256"] = SHA_A
            state[f"{prefix}_binding_request_sha256"] = SHA_B
            state[f"{prefix}_binding_manifest_sha256"] = SHA_A
            state[f"{prefix}_inspection_receipt_sha256"] = SHA_A

        with (
            patch("local_moe.bound_cell_run._record_binding", side_effect=record),
            patch("local_moe.bound_cell_run._validate_binding"),
            patch("local_moe.bound_cell_run._validate_preview_binding"),
            patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ),
        ):
            result = run_bound_cell(
                "advisor.json",
                "task",
                "catalog.json",
                "evaluation.json",
                "policy.json",
                "binding.json",
                confirmed=True,
                resolver=lambda _path: target,
                inspector=lambda *_args, **_kwargs: bundle,
                previewer=lambda *_args, **_kwargs: preview_evaluation(
                    preview, snapshot
                ),
                snapshot_collector=lambda: snapshot,
                lease_store=lease_store,
                transport=transport,
                clock=lambda: datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(result.receipt.status, "blocked")
        self.assertEqual(result.receipt.invocation_attempts, 0)
        self.assertEqual(result.receipt.endpoint_probe_requests, 1)
        self.assertEqual(result.envelope.lease_error_code, "lease_transition_failed")
        self.assertFalse(result.envelope.lease_transition_receipt.transition_applied)
        self.assertEqual(
            result.envelope.lease_release_receipt.delivery_status, "not_attempted"
        )
        self.assertEqual(
            events,
            ["lease_evaluate", "probe", "lease_arm", "lease_release:not_attempted"],
        )


if __name__ == "__main__":
    unittest.main()
