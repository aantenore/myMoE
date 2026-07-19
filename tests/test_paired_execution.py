from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import inspect
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch

from local_moe import assistant_bridge as assistant_bridge_module
from local_moe import paired_execution as paired_execution_module
from local_moe.assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    ExternalVerifierSpec,
    build_assistant_task,
)
from local_moe.assistant_bridge_attestation import (
    create_ed25519_evaluation_dsse_envelope,
)
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.assistant_bridge_integrity import (
    canonical_json_bytes,
    sha256_bytes,
)
from local_moe.assistant_bridge_two_phase_contracts import AttestationCheck
from local_moe.paired_execution_bridge import AssistantBridgePairedArmExecutor
from local_moe.paired_evidence import PairedAttestationVerifier
from local_moe.paired_execution import (
    paired_runner_sha256,
    paired_execution_harness_sha256,
    paired_runner_source_sha256,
    run_paired_case,
)
from local_moe.paired_execution_pricing import (
    PairedCostEvidence,
    PricingContract,
    PricingItem,
)
from local_moe.paired_execution_store import (
    PairedExecutionStore,
    PairedRunIndeterminateError,
)
from local_moe.route_outcomes import (
    OutcomeStore,
    VerifiedOutcomeRecord,
    runtime_plan_sha256,
)
from local_moe.route_promotion import PromotionCase, VerifiedRoutingEvidencePlan
from local_moe.route_signals import MetadataTaskSignalProvider
from local_moe.verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    sha256_json,
)
from tests.paired_attestation_fakes import (
    SigningPairedAttestationProducer,
    build_signed_paired_executor,
)
from tests.test_assistant_bridge import _fake_bridge, _fake_environment, _read_jsonl


LIFECYCLE_CONFIG_SHA256 = "d" * 64
CREATED_AT = "2026-07-19T10:00:00+00:00"
OUTPUT_SENTINEL = "RAW-OUTPUT-MUST-NOT-BE-PERSISTED"


class _StopAfterFirstCheckpoint(RuntimeError):
    pass


class _StopAfterFirstStore(PairedExecutionStore):
    def status(self):
        status = super().status()
        if status.state == "partial":
            raise _StopAfterFirstCheckpoint("simulate process stop")
        return status


class _RecordingExecutor:
    def __init__(
        self,
        delegate: AssistantBridgePairedArmExecutor,
        *,
        complete_usage: bool = True,
        fail_run_on: int | None = None,
    ) -> None:
        self.delegate = delegate
        self.complete_usage = complete_usage
        self.fail_run_on = fail_run_on
        self.plan_routes: list[str] = []
        self.run_routes: list[str] = []

    @property
    def bridge_config_sha256(self):
        return self.delegate.bridge_config_sha256

    @property
    def configuration_sha256(self):
        return self.delegate.configuration_sha256

    @property
    def execution_harness_sha256(self):
        return self.delegate.execution_harness_sha256

    @property
    def state_paths(self):
        return self.delegate.state_paths

    def snapshot_source(self, workspace):
        return self.delegate.snapshot_source(workspace)

    def preflight(self, task):
        return self.delegate.preflight(task)

    def assert_source_unchanged(self, snapshot):
        return self.delegate.assert_source_unchanged(snapshot)

    def verify_outcome(self, record, pricing):
        return self.delegate.verify_outcome(record, pricing)

    def plan_arm(self, task, **kwargs):
        os.environ["FAKE_CODEX_USAGE"] = "1" if self.complete_usage else "0"
        self.plan_routes.append(kwargs["slot"].route)
        return self.delegate.plan_arm(task, **kwargs)

    def run_arm(self, task, **kwargs):
        route = kwargs["plan"].slot.route
        self.run_routes.append(route)
        if self.fail_run_on == len(self.run_routes):
            raise RuntimeError("simulated provider boundary crash")
        # Usage must be present in the pre-attestation result.  Post-signature
        # mutation would correctly fail immediate receipt reconstruction.
        os.environ["FAKE_CODEX_USAGE"] = "1" if self.complete_usage else "0"
        return self.delegate.run_arm(task, **kwargs)


class _AdversarialProducer:
    def __init__(
        self,
        delegate: SigningPairedAttestationProducer,
        mode: str,
    ) -> None:
        self.delegate = delegate
        self.mode = mode
        self.configuration_sha256 = sha256_bytes(
            f"adversarial-producer/{mode}/v1".encode("utf-8")
        )
        self.semantic_configuration_sha256 = delegate.configuration_sha256
        self._replay: bytes | None = None

    @property
    def calls(self):
        return self.delegate.calls

    @property
    def state_paths(self):
        return self.delegate.state_paths

    def attest(self, binding, workspace, deadline):
        if self.mode in {"protocol", "quorum"}:
            self.delegate.calls.append((binding, workspace, deadline))
            return (object(),) if self.mode == "protocol" else ()
        signed_binding = binding
        if self.mode == "wrong-task":
            signed_binding = replace(binding, task_fingerprint="a" * 64)
        elif self.mode == "wrong-workspace":
            signed_binding = replace(binding, source_fingerprint="b" * 64)
        if self.mode == "wrong-spec":
            self.delegate.calls.append((binding, workspace, deadline))
            requirement = replace(
                self.delegate.requirements[0],
                spec_sha256="c" * 64,
            )
            envelopes = (
                create_ed25519_evaluation_dsse_envelope(
                    binding,
                    requirement,
                    self.delegate.private_keys[0],
                    attestation_id=(
                        "paired-wrong-spec-"
                        f"{binding.stage_idempotency_sha256[:24]}"
                    ),
                    issued_at=binding.created_at,
                    expires_at=binding.expires_at,
                    checks=(
                        AttestationCheck(
                            "signed-check",
                            True,
                            sha256_bytes(b"signed-check-passed"),
                        ),
                    ),
                ),
            )
        else:
            envelopes = self.delegate.attest(
                signed_binding,
                workspace,
                deadline,
            )
        if self.mode == "omit-required":
            return envelopes[1:]
        envelope = envelopes[0]
        if self.mode == "wrong-signature":
            raw = json.loads(envelope)
            signature = raw["signatures"][0]["sig"]
            raw["signatures"][0]["sig"] = (
                ("A" if signature[0] != "A" else "B") + signature[1:]
            )
            envelope = canonical_json_bytes(raw)
        elif self.mode == "replay":
            if self._replay is None:
                self._replay = envelope
            else:
                envelope = self._replay
        elif self.mode == "mutation":
            (workspace / "attestation-mutation.txt").write_text(
                "mutation\n",
                encoding="utf-8",
            )
        elif self.mode == "config-drift":
            self.configuration_sha256 = "d" * 64
        return (envelope,)


class _FailingEvidenceStore:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.configuration_sha256 = sha256_bytes(b"failing-evidence-store/v1")
        self.semantic_configuration_sha256 = (
            delegate.semantic_configuration_sha256
        )
        self.state_paths = (delegate.root,)

    def put_bytes(self, value, *, media_type):
        raise RuntimeError("simulated CAS persistence failure")

    def get_bytes(self, descriptor):
        return self.delegate.get_bytes(descriptor)


class PairedExecutionTests(unittest.TestCase):
    def test_verifier_lineage_is_portable_across_cas_replica_paths(self) -> None:
        with _fake_bridge() as fixture, tempfile.TemporaryDirectory() as temporary:
            signed = build_signed_paired_executor(
                fixture,
                Path(temporary) / "primary-evidence",
            )
            replica = ContentAddressedStore(
                Path(temporary) / "replica-evidence"
            )
            primary_verifier = PairedAttestationVerifier(
                trust_config=signed.trust_config,
                evidence_store=signed.evidence_store,
                bridge_config=fixture.runner.config,
            )
            replica_verifier = PairedAttestationVerifier(
                trust_config=signed.trust_config,
                evidence_store=replica,
                bridge_config=fixture.runner.config,
            )

            self.assertNotEqual(
                signed.evidence_store.configuration_sha256,
                replica.configuration_sha256,
            )
            self.assertEqual(
                primary_verifier.configuration_sha256,
                replica_verifier.configuration_sha256,
            )
            with self.assertRaisesRegex(TypeError, "final"):
                class _ForgedStore(ContentAddressedStore):
                    pass
            with self.assertRaisesRegex(AttributeError, "immutable"):
                signed.evidence_store.get_json = lambda *args: {}  # type: ignore[method-assign]
            with self.assertRaisesRegex(AttributeError, "immutable"):
                primary_verifier._evidence_root = Path("forged")

    def test_runner_digest_binds_every_package_module(self) -> None:
        self.assertNotIn(
            "lifecycle_config_sha256",
            inspect.signature(run_paired_case).parameters,
        )
        arguments = {
            "executor_config_sha256": "1" * 64,
            "lifecycle_config_sha256": "2" * 64,
            "signal_provider_config_sha256": "3" * 64,
        }
        baseline = paired_runner_sha256(**arguments)
        real_read = paired_execution_module._read_runner_source_file

        def drifted(path: Path) -> bytes:
            content = real_read(path)
            if path.name == "assistant_bridge_provider_registry.py":
                return content + b"\n# semantic drift\n"
            return content

        with patch.object(
            paired_execution_module,
            "_read_runner_source_file",
            side_effect=drifted,
        ):
            changed = paired_runner_sha256(**arguments)

        self.assertNotEqual(baseline, changed)

    def test_codex_json_usage_is_strict_bounded_and_cache_honest(self) -> None:
        parse = assistant_bridge_module._parse_codex_json_usage
        valid = _codex_usage_stdout()

        self.assertEqual(parse(valid), (20_403, 5))
        self.assertEqual(
            parse(
                b'{"type":"turn.started"}\n'
                + valid
                + b'{"type":"item.completed","item":{}}\n'
            ),
            (20_403, 5),
        )

        invalid = (
            b"",
            b'{"type":"turn.started"}\n',
            valid + valid,
            b"not-json\n",
            _codex_usage_stdout(input_tokens=-1),
            _codex_usage_stdout(input_tokens=True),
            _codex_usage_stdout(output_tokens="5"),
            _codex_usage_stdout(remove="reasoning_output_tokens"),
            _codex_usage_stdout(extra=1),
            _codex_usage_stdout(cached_input_tokens=1),
            _codex_usage_stdout(cache_write_input_tokens=1),
            b"{" + b"x" * (1024 * 1024) + b"}\n",
        )
        for payload in invalid:
            with self.subTest(payload_size=len(payload)):
                self.assertIsNone(parse(payload))

    def test_ab_executes_declared_routes_on_one_snapshot_without_apply(
        self,
    ) -> None:
        # BA is exercised through the partial-checkpoint resume lifecycle below.
        for order, expected_routes in (
            ("AB", ["local_then_verify", "local"]),
        ):
            with self.subTest(order=order), _fake_bridge() as fixture, _fake_environment(
                fixture,
                local_output=f"VERIFIED {OUTPUT_SENTINEL}",
                write_relative="tracked.txt",
                write_content="candidate-only\n",
            ), tempfile.TemporaryDirectory() as temporary:
                task = _balanced_task()
                delegate, plan, case, pricing = _planned_case(
                    fixture,
                    task,
                    order=order,
                    baseline_route="local_then_verify",
                    candidate_route="local",
                )
                executor = _RecordingExecutor(delegate)
                before = delegate.snapshot_source(fixture.root)
                run_dir = Path(temporary) / "run"
                outcomes_path = Path(temporary) / "outcomes.jsonl"

                with patch.object(
                    fixture.runner,
                    "_apply_verified_candidate",
                ) as apply_candidate:
                    result = run_paired_case(
                        task=task,
                        plan=plan,
                        case=case,
                        source_workspace=fixture.root,
                        pricing=pricing,
                        run_store=run_dir,
                        outcome_store=outcomes_path,
                        executor=executor,
                        created_at=CREATED_AT,
                    )
                    apply_candidate.assert_not_called()
                replay = run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=executor,
                    created_at=CREATED_AT,
                )

                after = delegate.snapshot_source(fixture.root)
                persisted = outcomes_path.read_text(encoding="utf-8")
                persisted += (run_dir / "run.json").read_text(encoding="utf-8")
                persisted += "".join(
                    item.read_text(encoding="utf-8")
                    for item in sorted((run_dir / "events").iterdir())
                )
                invocations = _read_jsonl(fixture.log)

                self.assertEqual(executor.plan_routes, expected_routes)
                self.assertEqual(executor.run_routes, expected_routes)
                self.assertEqual(before, after)
                self.assertEqual(result.state, "complete")
                self.assertTrue(result.cost_complete)
                self.assertEqual(len(result.records), 2)
                self.assertEqual(
                    [record.record_id for record in replay.records],
                    [record.record_id for record in result.records],
                )
                self.assertEqual(
                    [item["mode"] for item in invocations],
                    ["local", "local"],
                )
                self.assertNotIn(OUTPUT_SENTINEL, persisted)
                for record in result.records:
                    proof = delegate.verify_outcome(record, pricing)
                    signed_created_at = datetime.fromtimestamp(
                        proof.candidate_created_at,
                        tz=timezone.utc,
                    ).replace(microsecond=0).isoformat()
                    self.assertEqual(record.created_at, signed_created_at)
                    self.assertNotEqual(record.created_at, CREATED_AT)
                tampered_payload = result.records[0].payload()
                tampered_payload.pop("record_id")
                tampered_payload["latency_ms"] = int(
                    tampered_payload["latency_ms"]
                ) + 1
                tampered = VerifiedOutcomeRecord.from_payload(
                    {
                        "record_id": (
                            "outcome-" + sha256_json(tampered_payload)
                        ),
                        **tampered_payload,
                    }
                )
                with self.assertRaisesRegex(ValueError, "exactly reproduce"):
                    delegate.verify_outcome(tampered, pricing)
                for raw in result.cost_evidence_payloads:
                    self.assertIsNotNone(raw)
                    evidence = PairedCostEvidence.from_payload(
                        raw,  # type: ignore[arg-type]
                        pricing=pricing,
                    )
                    self.assertEqual(evidence.total_cost_usd, "0.00012")

    def test_signed_negative_evidence_is_durable_but_never_a_pass(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED negative evidence",
            write_relative="tracked.txt",
            write_content="candidate-only\n",
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            fixture.planned_signed_executor.producer.passed = False
            result = run_paired_case(
                task=task,
                plan=plan,
                case=case,
                source_workspace=fixture.root,
                pricing=pricing,
                run_store=Path(temporary) / "run",
                outcome_store=Path(temporary) / "outcomes.jsonl",
                executor=_RecordingExecutor(delegate),
                created_at=CREATED_AT,
            )

            status = PairedExecutionStore(Path(temporary) / "run").status()
            self.assertEqual(status.state, "complete")
            self.assertEqual(len(status.checkpoints), 2)
            self.assertEqual(
                [record.outcome for record in result.records],
                ["failed", "failed"],
            )
            self.assertTrue(
                all(record.paired_evidence is not None for record in result.records)
            )
            for record in result.records:
                self.assertEqual(
                    delegate.verify_outcome(record, pricing).record,
                    record,
                )
            self.assertEqual(
                [record.failure_class for record in result.records],
                ["signed-attestation-failed", "signed-attestation-failed"],
            )

    def test_partial_checkpoint_resumes_only_the_second_declared_slot(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED resumable",
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="BA",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate)
            run_dir = Path(temporary) / "run"
            outcomes = OutcomeStore(Path(temporary) / "outcomes.jsonl")

            with self.assertRaises(_StopAfterFirstCheckpoint):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=_StopAfterFirstStore(run_dir),
                    outcome_store=outcomes,
                    executor=executor,
                    created_at=CREATED_AT,
                )
            partial = PairedExecutionStore(run_dir).status()
            self.assertEqual(partial.state, "partial")
            assert partial.root is not None
            run_instance_nonce = partial.root.run_instance_nonce
            self.assertEqual(executor.run_routes, ["local"])

            resumed = run_paired_case(
                task=task,
                plan=plan,
                case=case,
                source_workspace=fixture.root,
                pricing=pricing,
                run_store=PairedExecutionStore(run_dir),
                outcome_store=outcomes,
                executor=executor,
                created_at=CREATED_AT,
            )

            self.assertEqual(executor.run_routes, ["local", "local_then_verify"])
            self.assertEqual(resumed.state, "complete")
            self.assertEqual(resumed.root.run_instance_nonce, run_instance_nonce)
            self.assertEqual(len(outcomes.list_records()), 2)

    def test_fresh_run_directories_receive_distinct_instance_nonces(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )

            class StopBeforePlanning(_RecordingExecutor):
                def plan_arm(self, task, **kwargs):
                    raise RuntimeError("stop after durable claim")

            nonces = ("a" * 64, "b" * 64)
            roots = []
            with patch.object(
                paired_execution_module,
                "_new_run_instance_nonce",
                side_effect=nonces,
            ):
                for index in range(2):
                    run_dir = Path(temporary) / f"run-{index}"
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "stop after durable claim",
                    ):
                        run_paired_case(
                            task=task,
                            plan=plan,
                            case=case,
                            source_workspace=fixture.root,
                            pricing=pricing,
                            run_store=run_dir,
                            outcome_store=(
                                Path(temporary) / f"outcomes-{index}.jsonl"
                            ),
                            executor=StopBeforePlanning(delegate),
                            created_at=CREATED_AT,
                        )
                    status = PairedExecutionStore(run_dir).status()
                    assert status.root is not None
                    roots.append(status.root)

            self.assertEqual(
                [root.run_instance_nonce for root in roots],
                list(nonces),
            )
            self.assertNotEqual(roots[0].run_id, roots[1].run_id)
            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_runner_source_drift_between_arms_blocks_the_second_claim(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED first arm",
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate)
            before = executor.snapshot_source(fixture.root)
            drift = {"enabled": False}

            class DriftAfterFirstCheckpointStore(PairedExecutionStore):
                def complete(self, *args, **kwargs):
                    checkpoint = super().complete(*args, **kwargs)
                    drift["enabled"] = True
                    return checkpoint

            store = DriftAfterFirstCheckpointStore(Path(temporary) / "run")
            real_read = paired_execution_module._read_runner_source_file

            def drifted(path: Path) -> bytes:
                content = real_read(path)
                if (
                    drift["enabled"]
                    and path.name == "assistant_bridge_provider_registry.py"
                ):
                    return content + b"\n# concurrent semantic drift\n"
                return content

            with patch.object(
                paired_execution_module,
                "_read_runner_source_file",
                side_effect=drifted,
            ), self.assertRaisesRegex(
                ValueError,
                "runner implementation changed",
            ):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=store,
                    outcome_store=Path(temporary) / "outcomes.jsonl",
                    executor=executor,
                    created_at=CREATED_AT,
                )

            self.assertEqual(store.status().state, "partial")
            self.assertEqual(len(executor.run_routes), 1)
            self.assertEqual(len(_read_jsonl(fixture.log)), 2)
            executor.assert_source_unchanged(before)

    def test_crash_after_claim_is_indeterminate_and_never_retried(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            crashing = _RecordingExecutor(delegate, fail_run_on=1)
            run_dir = Path(temporary) / "run"
            outcomes_path = Path(temporary) / "outcomes.jsonl"

            with self.assertRaisesRegex(RuntimeError, "provider boundary crash"):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=crashing,
                    created_at=CREATED_AT,
                )
            self.assertEqual(
                PairedExecutionStore(run_dir).status().state,
                "indeterminate",
            )
            healthy = _RecordingExecutor(delegate)
            with self.assertRaises(PairedRunIndeterminateError):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=healthy,
                    created_at=CREATED_AT,
                )
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(crashing.run_routes, ["local_then_verify"])
        self.assertEqual(healthy.run_routes, [])
        self.assertEqual(invocations, [])

    def test_quorum_cannot_substitute_for_a_task_required_verifier(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED signed candidate",
        ), tempfile.TemporaryDirectory() as temporary:
            secondary = ExternalVerifierSpec(
                id="secondary-tests",
                verifier="secondary-test-runner",
                spec_sha256="e" * 64,
            )
            external_verifiers = dict(fixture.runner.config.external_verifiers)
            external_verifiers[secondary.id] = secondary
            extended_config = replace(
                fixture.runner.config,
                external_verifiers=external_verifiers,
                source_sha256=sha256_json(
                    {
                        "prior": fixture.runner.config.source_sha256,
                        "secondary": secondary.spec_sha256,
                    }
                ),
            )
            fixture.config = extended_config
            fixture.runner = AssistantBridgeRunner(extended_config)
            task = build_assistant_task(
                "Change the tracked file.",
                profile="balanced",
                required_capabilities=("code",),
                required_verifier_ids=("focused-tests",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            _, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            signed = build_signed_paired_executor(
                fixture,
                Path(temporary) / "evidence",
                verifier_ids=("focused-tests", "secondary-tests"),
                quorum=1,
            )
            adversarial = _AdversarialProducer(
                signed.producer,
                "omit-required",
            )
            executor = _RecordingExecutor(
                AssistantBridgePairedArmExecutor(
                    fixture.runner,
                    attestation_producer=adversarial,
                    trust_config=signed.trust_config,
                    evidence_store=signed.evidence_store,
                )
            )
            plan = _plan_for_executor(
                plan,
                executor,
                attestation_policy_sha256=(
                    signed.trust_config.policy.policy_sha256
                ),
            )
            before = executor.snapshot_source(fixture.root)
            run_dir = Path(temporary) / "run"
            outcomes_path = Path(temporary) / "outcomes.jsonl"

            with self.assertRaisesRegex(
                AssistantBridgeError,
                "omitted task-required verifier evidence",
            ):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=executor,
                    created_at=CREATED_AT,
                )
            provider_calls = len(_read_jsonl(fixture.log))
            producer_calls = len(adversarial.calls)
            with self.assertRaises(PairedRunIndeterminateError):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=executor,
                    created_at=CREATED_AT,
                )

            self.assertEqual(
                PairedExecutionStore(run_dir).status().state,
                "indeterminate",
            )
            self.assertEqual(len(adversarial.calls), producer_calls)
            self.assertEqual(len(_read_jsonl(fixture.log)), provider_calls)
            executor.assert_source_unchanged(before)

    def test_candidate_first_requires_live_baseline_authority_before_invocation(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = build_assistant_task(
                "Change the tracked file.",
                profile="quality",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="BA",
                baseline_route="premium",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate)
            run_dir = Path(temporary) / "run"
            (fixture.root / "runtime" / "codex-home" / "auth.json").unlink()

            with self.assertRaisesRegex(ValueError, "baseline route"):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=Path(temporary) / "outcomes.jsonl",
                    executor=executor,
                    created_at=CREATED_AT,
                )
            invocations = _read_jsonl(fixture.log)
            state = PairedExecutionStore(run_dir).status().state

        self.assertEqual(executor.plan_routes, ["local"])
        self.assertEqual(executor.run_routes, [])
        self.assertEqual(invocations, [])
        self.assertEqual(state, "indeterminate")

    def test_store_preflight_rejects_source_symlinks_and_store_overlap(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate)
            temporary_root = Path(temporary)
            outcome_link = temporary_root / "outcome-link.jsonl"
            outcome_link.symlink_to(fixture.root / "hidden-outcomes.jsonl")
            scenarios = (
                (
                    "run-inside-source",
                    PairedExecutionStore(fixture.root / "paired-run"),
                    OutcomeStore(temporary_root / "outside-outcomes.jsonl"),
                ),
                (
                    "outcome-symlink-inside-source",
                    PairedExecutionStore(temporary_root / "outside-run"),
                    OutcomeStore(outcome_link),
                ),
                (
                    "stores-alias",
                    PairedExecutionStore(temporary_root / "aliased-store"),
                    OutcomeStore(temporary_root / "aliased-store"),
                ),
                (
                    "stores-overlap",
                    PairedExecutionStore(temporary_root / "overlapping-run"),
                    OutcomeStore(
                        temporary_root / "overlapping-run" / "outcomes.jsonl"
                    ),
                ),
            )

            for name, store, outcomes in scenarios:
                with self.subTest(name=name), patch.object(
                    executor,
                    "snapshot_source",
                    wraps=executor.snapshot_source,
                ) as snapshot_source, patch.object(
                    store,
                    "claim",
                    wraps=store.claim,
                ) as claim:
                    with self.assertRaisesRegex(
                        ValueError,
                        "physically isolated|must not alias or overlap",
                    ):
                        run_paired_case(
                            task=task,
                            plan=plan,
                            case=case,
                            source_workspace=fixture.root,
                            pricing=pricing,
                            run_store=store,
                            outcome_store=outcomes,
                            executor=executor,
                            created_at=CREATED_AT,
                        )
                    snapshot_source.assert_not_called()
                    claim.assert_not_called()
                    self.assertFalse(store.root_path.exists())

            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_windows_resolved_path_rejects_final_reparse_before_resolve(
        self,
    ) -> None:
        path = Path("outcomes.jsonl")
        metadata = Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_file_attributes=0x00000400,
        )
        with (
            patch.object(paired_execution_module, "_OS_NAME", "nt"),
            patch.object(Path, "lstat", return_value=metadata),
            patch.object(Path, "resolve") as resolve,
            self.assertRaisesRegex(VerifiedRoutingError, "physically isolated"),
        ):
            paired_execution_module._resolved_path(path, "outcome store")

        resolve.assert_not_called()

    def test_signed_preflight_is_required_before_store_or_provider_authority(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            _, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(
                AssistantBridgePairedArmExecutor(fixture.runner)
            )
            run_dir = Path(temporary) / "run"

            with self.assertRaisesRegex(
                AssistantBridgeError,
                "signed_verifier_required",
            ):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=Path(temporary) / "outcomes.jsonl",
                    executor=executor,
                    created_at=CREATED_AT,
                )

            self.assertFalse(run_dir.exists())
            self.assertEqual(_read_jsonl(fixture.log), [])
            self.assertEqual(executor.plan_routes, [])
            self.assertEqual(executor.run_routes, [])

    def test_verifier_state_is_isolated_from_source_run_outcome_and_itself(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            _, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            temporary_root = Path(temporary)

            signed_source = build_signed_paired_executor(
                fixture,
                fixture.root / "source-evidence",
            )
            signed_run = build_signed_paired_executor(
                fixture,
                temporary_root / "run-evidence",
            )
            signed_outcome = build_signed_paired_executor(
                fixture,
                temporary_root / "outcome-evidence",
            )
            signed_overlap = build_signed_paired_executor(
                fixture,
                temporary_root / "overlap-evidence",
            )
            producer_state = signed_overlap.evidence_store.root / "producer-state"
            producer_state.mkdir(mode=0o700)
            signed_overlap.producer._state_paths = (producer_state,)
            scenarios = (
                (
                    "source",
                    signed_source.executor,
                    temporary_root / "source-run",
                    temporary_root / "source-outcomes.jsonl",
                ),
                (
                    "run",
                    signed_run.executor,
                    signed_run.evidence_store.root / "run",
                    temporary_root / "run-outcomes.jsonl",
                ),
                (
                    "outcome",
                    signed_outcome.executor,
                    temporary_root / "outcome-run",
                    signed_outcome.evidence_store.root / "outcomes.jsonl",
                ),
                (
                    "component-overlap",
                    signed_overlap.executor,
                    temporary_root / "overlap-run",
                    temporary_root / "overlap-outcomes.jsonl",
                ),
            )

            for name, delegate, run_dir, outcomes_path in scenarios:
                executor = _RecordingExecutor(delegate)
                scenario_plan = _plan_for_executor(
                    plan,
                    executor,
                    attestation_policy_sha256=(
                        delegate._trust_config.policy.policy_sha256
                    ),
                )
                store = PairedExecutionStore(run_dir)
                with self.subTest(name=name), patch.object(
                    executor,
                    "snapshot_source",
                    wraps=executor.snapshot_source,
                ) as snapshot_source, patch.object(
                    store,
                    "claim",
                    wraps=store.claim,
                ) as claim:
                    with self.assertRaisesRegex(
                        ValueError,
                        "physically isolated|evidence state|state paths",
                    ):
                        run_paired_case(
                            task=task,
                            plan=scenario_plan,
                            case=case,
                            source_workspace=fixture.root,
                            pricing=pricing,
                            run_store=store,
                            outcome_store=outcomes_path,
                            executor=executor,
                            created_at=CREATED_AT,
                        )
                    snapshot_source.assert_not_called()
                    claim.assert_not_called()

            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_bridge_ledger_inside_source_fails_before_claim_or_write(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            original_ledger = fixture.runner.state_ledger
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            fixture.runner.state_ledger = original_ledger
            original_ledger.path.parent.chmod(0o700)
            lock_path = original_ledger.path.with_suffix(
                original_ledger.path.suffix + ".lock"
            )
            self.assertFalse(original_ledger.path.exists())
            self.assertFalse(lock_path.exists())
            executor = _RecordingExecutor(delegate)
            store = PairedExecutionStore(Path(temporary) / "run")

            with patch.object(
                executor,
                "snapshot_source",
                wraps=executor.snapshot_source,
            ) as snapshot_source, patch.object(
                store,
                "claim",
                wraps=store.claim,
            ) as claim, self.assertRaisesRegex(
                ValueError,
                "physically isolated",
            ):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=store,
                    outcome_store=Path(temporary) / "outcomes.jsonl",
                    executor=executor,
                    created_at=CREATED_AT,
                )

            snapshot_source.assert_not_called()
            claim.assert_not_called()
            self.assertFalse(original_ledger.path.exists())
            self.assertFalse(lock_path.exists())
            self.assertFalse(store.root_path.exists())
            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_workspace_tampering_is_indeterminate_and_not_retried(self) -> None:
        # Signature, exact binding, policy and replay rejection are covered at
        # their DSSE/directory boundaries; quorum has a dedicated lifecycle test.
        # Keep a real workspace mutation through the complete orchestration path.
        modes = ("mutation",)
        for mode in modes:
            with self.subTest(mode=mode), _fake_bridge() as fixture, _fake_environment(
                fixture,
                local_output="VERIFIED adversarial candidate",
            ), tempfile.TemporaryDirectory() as temporary:
                task = _balanced_task()
                _, plan, case, pricing = _planned_case(
                    fixture,
                    task,
                    order="AB",
                    baseline_route="local_then_verify",
                    candidate_route="local",
                )
                signed = build_signed_paired_executor(
                    fixture,
                    Path(temporary) / "evidence",
                )
                producer = _AdversarialProducer(signed.producer, mode)
                executor = _RecordingExecutor(
                    AssistantBridgePairedArmExecutor(
                        fixture.runner,
                        attestation_producer=producer,
                        trust_config=signed.trust_config,
                        evidence_store=signed.evidence_store,
                    )
                )
                plan = _plan_for_executor(
                    plan,
                    executor,
                    attestation_policy_sha256=(
                        signed.trust_config.policy.policy_sha256
                    ),
                )
                before = executor.snapshot_source(fixture.root)
                run_dir = Path(temporary) / "run"
                outcomes_path = Path(temporary) / "outcomes.jsonl"

                with self.assertRaises(
                    (AssistantBridgeError, ValueError, RuntimeError)
                ):
                    run_paired_case(
                        task=task,
                        plan=plan,
                        case=case,
                        source_workspace=fixture.root,
                        pricing=pricing,
                        run_store=run_dir,
                        outcome_store=outcomes_path,
                        executor=executor,
                        created_at=CREATED_AT,
                    )
                self.assertEqual(
                    PairedExecutionStore(run_dir).status().state,
                    "indeterminate",
                )
                provider_calls = len(_read_jsonl(fixture.log))
                producer_calls = len(producer.calls)
                with self.assertRaises(PairedRunIndeterminateError):
                    run_paired_case(
                        task=task,
                        plan=plan,
                        case=case,
                        source_workspace=fixture.root,
                        pricing=pricing,
                        run_store=run_dir,
                        outcome_store=outcomes_path,
                        executor=executor,
                        created_at=CREATED_AT,
                    )
                self.assertEqual(len(producer.calls), producer_calls)
                self.assertEqual(len(_read_jsonl(fixture.log)), provider_calls)
                executor.assert_source_unchanged(before)

    def test_evidence_store_failure_is_indeterminate_and_not_retried(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED candidate",
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            _, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            signed = build_signed_paired_executor(
                fixture,
                Path(temporary) / "evidence",
            )
            failing_store = _FailingEvidenceStore(signed.evidence_store)
            executor = _RecordingExecutor(
                AssistantBridgePairedArmExecutor(
                    fixture.runner,
                    attestation_producer=signed.producer,
                    trust_config=signed.trust_config,
                    evidence_store=failing_store,
                )
            )
            plan = _plan_for_executor(
                plan,
                executor,
                attestation_policy_sha256=(
                    signed.trust_config.policy.policy_sha256
                ),
            )
            before = executor.snapshot_source(fixture.root)
            run_dir = Path(temporary) / "run"
            outcomes_path = Path(temporary) / "outcomes.jsonl"

            with self.assertRaisesRegex(RuntimeError, "persistence failure"):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=executor,
                    created_at=CREATED_AT,
                )
            provider_calls = len(_read_jsonl(fixture.log))
            producer_calls = len(signed.producer.calls)
            with self.assertRaises(PairedRunIndeterminateError):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=outcomes_path,
                    executor=executor,
                    created_at=CREATED_AT,
                )

            self.assertEqual(
                PairedExecutionStore(run_dir).status().state,
                "indeterminate",
            )
            self.assertEqual(len(signed.producer.calls), producer_calls)
            self.assertEqual(len(_read_jsonl(fixture.log)), provider_calls)
            executor.assert_source_unchanged(before)

    def test_schema_one_rejects_custom_or_reconfigured_signal_providers(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate)

            class CustomSignalProvider:
                def signals_from_metadata(
                    self,
                    task_metadata,
                    *,
                    context_tokens=None,
                ):
                    self.last_metadata = task_metadata
                    self.last_context_tokens = context_tokens
                    return MetadataTaskSignalProvider().signals_from_metadata(
                        task_metadata,
                        context_tokens=context_tokens,
                    )

            provider = CustomSignalProvider()
            with patch.object(
                executor,
                "snapshot_source",
                wraps=executor.snapshot_source,
            ) as snapshot_source:
                with self.assertRaisesRegex(
                    TypeError,
                    "concrete MetadataTaskSignalProvider",
                ):
                    run_paired_case(
                        task=task,
                        plan=plan,
                        case=case,
                        source_workspace=fixture.root,
                        pricing=pricing,
                        run_store=Path(temporary) / "run",
                        outcome_store=Path(temporary) / "outcomes.jsonl",
                        executor=executor,
                        signal_provider=provider,
                        created_at=CREATED_AT,
                    )
                snapshot_source.assert_not_called()

            reconfigured = MetadataTaskSignalProvider(source="custom-signals-v1")
            with self.assertRaisesRegex(
                VerifiedRoutingError,
                "default MetadataTaskSignalProvider config",
            ):
                run_paired_case(
                    task=task,
                    plan=plan,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=Path(temporary) / "invalid-run",
                    outcome_store=Path(temporary) / "invalid-outcomes.jsonl",
                    executor=executor,
                    signal_provider=reconfigured,
                    created_at=CREATED_AT,
                )
            self.assertFalse(hasattr(provider, "last_metadata"))
            self.assertEqual(_read_jsonl(fixture.log), [])

    def test_incomplete_usage_persists_nonqualifying_outcomes_without_cost(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
            local_output="VERIFIED no usage",
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            executor = _RecordingExecutor(delegate, complete_usage=False)
            result = run_paired_case(
                task=task,
                plan=plan,
                case=case,
                source_workspace=fixture.root,
                pricing=pricing,
                run_store=Path(temporary) / "run",
                outcome_store=Path(temporary) / "outcomes.jsonl",
                executor=executor,
                created_at=CREATED_AT,
            )

        self.assertFalse(result.cost_complete)
        self.assertEqual(result.cost_evidence_payloads, (None, None))
        for record in result.records:
            self.assertIsNone(record.estimated_cost_usd)
            self.assertNotIn("paired_cost", record.payload())

    def test_legacy_plan_without_embedded_pricing_fails_before_claim(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(
            fixture,
        ), tempfile.TemporaryDirectory() as temporary:
            task = _balanced_task()
            delegate, plan, case, pricing = _planned_case(
                fixture,
                task,
                order="AB",
                baseline_route="local_then_verify",
                candidate_route="local",
            )
            content = plan.content_payload()
            content.pop("pricing_contract")
            fields = dict(content)
            fields.pop("cases")
            legacy = VerifiedRoutingEvidencePlan(
                **fields,
                cases=plan.cases,
                plan_sha256=sha256_json(content),
            )
            executor = _RecordingExecutor(delegate)
            run_dir = Path(temporary) / "run"

            with self.assertRaisesRegex(ValueError, "embedded pricing"):
                run_paired_case(
                    task=task,
                    plan=legacy,
                    case=case,
                    source_workspace=fixture.root,
                    pricing=pricing,
                    run_store=run_dir,
                    outcome_store=Path(temporary) / "outcomes.jsonl",
                    executor=executor,
                    created_at=CREATED_AT,
                )
            invocations = _read_jsonl(fixture.log)
            run_created = run_dir.exists()

        self.assertEqual(invocations, [])
        self.assertFalse(run_created)


def _balanced_task():
    return build_assistant_task(
        "Change the tracked file.",
        profile="balanced",
        required_capabilities=("code",),
        risk_class="write_local",
        allow_remote=True,
        allow_remote_workspace=True,
    )


def _codex_usage_stdout(
    *,
    remove: str | None = None,
    extra: object | None = None,
    **overrides: object,
) -> bytes:
    usage: dict[str, object] = {
        "input_tokens": 20_403,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 5,
        "reasoning_output_tokens": 0,
    }
    usage.update(overrides)
    if remove is not None:
        usage.pop(remove)
    event: dict[str, object] = {"type": "turn.completed", "usage": usage}
    if extra is not None:
        event["extra"] = extra
    return (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")


def _planned_case(
    fixture,
    task,
    *,
    order: str,
    baseline_route: str,
    candidate_route: str,
):
    signed_executor = build_signed_paired_executor(
        fixture,
        fixture.root.parent / "paired-evidence",
    )
    fixture.planned_signed_executor = signed_executor
    executor = signed_executor.executor
    receipt_object = fixture.runner.candidate_generator().inspect_candidate(
        task,
        workspace=fixture.root,
        expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
    )
    if receipt_object.route != baseline_route:
        raise AssertionError("fixture task does not produce the requested baseline")
    receipt = receipt_object.payload()
    signals = MetadataTaskSignalProvider().signals_from_metadata(
        task.metadata_payload()
    )
    case = PromotionCase(
        task_fingerprint=task.task_fingerprint,
        normalized_item_sha256=sha256_json(
            {"objective_sha256": task.objective_sha256}
        ),
        profile=task.profile,
        capabilities=tuple(sorted(task.capability_demand.required)),
        difficulty=signals.difficulty,
        baseline_route=baseline_route,
        candidate_route=candidate_route,
        order=order,
        config_sha256=receipt["config_sha256"],
        signal_provider_config_sha256=signals.provider_config_sha256,
        runtime_plan_sha256=runtime_plan_sha256(receipt),
    )
    pricing_items = []
    for provider_key, runtime_key, prompt_rate, completion_rate in (
        ("local_provider", "local_runtime", "1", "2"),
        ("premium_provider", "premium_runtime", "10", "20"),
    ):
        provider = receipt[provider_key]
        runtime = receipt[runtime_key]
        if provider is not None:
            pricing_items.append(
                PricingItem(
                    provider_id=provider,
                    model=runtime["model"],
                    prompt_usd_per_million=prompt_rate,
                    completion_usd_per_million=completion_rate,
                )
            )
    pricing = PricingContract.build(pricing_items)
    cases_payload = [case.payload()]
    content = {
        "schema_version": CONTRACT_VERSION,
        "contract": "VerifiedRoutingEvidencePlan",
        "created_at": CREATED_AT,
        "route_policy_digest": "1" * 64,
        "scorecard_digest": "2" * 64,
        "training_source_digest": "3" * 64,
        "gate_policy_digest": "4" * 64,
        "evaluator_sha256": "5" * 64,
        "split_sha256": sha256_json({"cases": cases_payload}),
        "canary_basis_points": 100,
        "manifest_ttl_seconds": 3600,
        "assignment_salt_sha256": "6" * 64,
        "attestation_policy_sha256": (
            signed_executor.trust_config.policy.policy_sha256
        ),
        "execution_harness_sha256": paired_execution_harness_sha256(
            executor_harness_sha256=executor.execution_harness_sha256,
            signal_provider_config_sha256=signals.provider_config_sha256,
        ),
        "runner_source_sha256": paired_runner_source_sha256(),
        "pricing_contract": pricing.payload(),
        "pricing_sha256": pricing.pricing_sha256,
        "cases": cases_payload,
    }
    plan_fields = dict(content)
    plan_fields.pop("cases")
    plan = VerifiedRoutingEvidencePlan(
        **plan_fields,
        cases=(case,),
        plan_sha256=sha256_json(content),
    )
    return executor, plan, case, pricing


def _plan_for_executor(
    plan: VerifiedRoutingEvidencePlan,
    executor: object,
    *,
    attestation_policy_sha256: str,
) -> VerifiedRoutingEvidencePlan:
    """Re-preregister a test plan for the exact executor under test."""

    content = plan.content_payload()
    content["attestation_policy_sha256"] = attestation_policy_sha256
    content["execution_harness_sha256"] = paired_execution_harness_sha256(
        executor_harness_sha256=executor.execution_harness_sha256,
        signal_provider_config_sha256=(
            plan.cases[0].signal_provider_config_sha256
        ),
    )
    content["runner_source_sha256"] = paired_runner_source_sha256()
    plan_fields = dict(content)
    plan_fields.pop("cases")
    return VerifiedRoutingEvidencePlan(
        **plan_fields,
        cases=plan.cases,
        plan_sha256=sha256_json(content),
    )


if __name__ == "__main__":
    unittest.main()
