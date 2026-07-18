from __future__ import annotations

import unittest
from unittest.mock import patch

from local_moe.assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    ProviderAdapterRegistry,
    build_assistant_task,
    default_provider_adapter_registry,
)
from local_moe.assistant_bridge_workspace import snapshot_workspace
from tests.test_assistant_bridge import (
    _fake_bridge,
    _fake_environment,
    _read_jsonl,
    _RecordingProviderAdapter,
)

LIFECYCLE_CONFIG_SHA256 = "d" * 64


class AssistantBridgeCandidateGeneratorTests(unittest.TestCase):
    def test_local_candidate_is_temporary_and_never_applies_source(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="candidate-only\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            before = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            request = generator.request(
                task,
                confirmation=str(plan["confirmation_id"]),
            )
            with patch.object(
                fixture.runner,
                "_apply_verified_candidate",
            ) as apply_candidate:
                with generator.generate(
                    request,
                    source_workspace=fixture.root,
                    expected_source_fingerprint=before.fingerprint,
                    expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                ) as generated:
                    candidate_root = generated.workspace
                    self.assertTrue(candidate_root.is_dir())
                    self.assertNotEqual(candidate_root, fixture.root)
                    self.assertEqual(
                        (candidate_root / "tracked.txt").read_text(encoding="utf-8"),
                        "candidate-only\n",
                    )
                    self.assertEqual(
                        (fixture.root / "tracked.txt").read_text(encoding="utf-8"),
                        "initial\n",
                    )
                self.assertFalse(candidate_root.exists())
                apply_candidate.assert_not_called()
            after = snapshot_workspace(fixture.root, fixture.config.workspace.scope)

        self.assertEqual(plan["mode"], "assistant_bridge_candidate_plan")
        self.assertEqual(plan["route_receipt"]["route"], "local")
        self.assertEqual(before, after)

    def test_candidate_confirmation_is_intent_bound_and_one_shot(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="candidate-only\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            token = str(plan["confirmation_id"])
            with self.assertRaisesRegex(AssistantBridgeError, "confirmation"):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=token,
                )
            request = generator.request(task, confirmation=token)
            with generator.generate(
                request,
                source_workspace=fixture.root,
                expected_source_fingerprint=source.fingerprint,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            ):
                pass
            with self.assertRaisesRegex(AssistantBridgeError, "confirmation"):
                with generator.generate(
                    request,
                    source_workspace=fixture.root,
                    expected_source_fingerprint=source.fingerprint,
                    expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                ):
                    pass

    def test_local_quality_findings_are_observations_not_apply_authority(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="output without the required marker",
                write_relative="tracked.txt",
                write_content="candidate-with-finding\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            with generator.generate_candidate(
                task,
                source_workspace=fixture.root,
                expected_source_fingerprint=source.fingerprint,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                confirmation=str(plan["confirmation_id"]),
            ) as generated:
                self.assertEqual(
                    (generated.workspace / "tracked.txt").read_text(encoding="utf-8"),
                    "candidate-with-finding\n",
                )
                self.assertEqual(
                    (fixture.root / "tracked.txt").read_text(encoding="utf-8"),
                    "initial\n",
                )

    def test_quality_failure_escalates_without_source_apply(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="local quality finding",
                premium_output="VERIFIED premium candidate",
                write_relative="tracked.txt",
                write_content="premium-candidate\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="balanced",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            with generator.generate_candidate(
                task,
                source_workspace=fixture.root,
                expected_source_fingerprint=source.fingerprint,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                confirmation=str(plan["confirmation_id"]),
            ) as generated:
                self.assertEqual(
                    (generated.workspace / "tracked.txt").read_text(encoding="utf-8"),
                    "premium-candidate\n",
                )
            invocations = _read_jsonl(fixture.log)
            source_content = (fixture.root / "tracked.txt").read_text(encoding="utf-8")

        self.assertEqual([item["mode"] for item in invocations], ["local", "premium"])
        self.assertEqual(source_content, "initial\n")

    def test_balanced_local_quality_pass_does_not_escalate_for_attestation(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED local candidate",
                write_relative="tracked.txt",
                write_content="local-candidate\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="balanced",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=True,
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            with generator.generate_candidate(
                task,
                source_workspace=fixture.root,
                expected_source_fingerprint=source.fingerprint,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                confirmation=str(plan["confirmation_id"]),
            ) as generated:
                self.assertEqual(
                    (generated.workspace / "tracked.txt").read_text(encoding="utf-8"),
                    "local-candidate\n",
                )
            invocations = _read_jsonl(fixture.log)
            source_content = (fixture.root / "tracked.txt").read_text(encoding="utf-8")

        self.assertEqual([item["mode"] for item in invocations], ["local"])
        self.assertEqual(source_content, "initial\n")

    def test_missing_required_delta_is_not_stageable(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = fixture.runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )

            with self.assertRaisesRegex(
                AssistantBridgeError, "candidate_scope_invalid"
            ):
                with generator.generate_candidate(
                    task,
                    source_workspace=fixture.root,
                    expected_source_fingerprint=source.fingerprint,
                    expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                    confirmation=str(plan["confirmation_id"]),
                ):
                    pass

    def test_candidate_generator_reuses_injected_provider_registry(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="registry-candidate\n",
            ),
        ):
            delegate = default_provider_adapter_registry().require("codex_cli")
            recording = _RecordingProviderAdapter(delegate)
            runner = AssistantBridgeRunner.with_provider_adapters(
                fixture.config,
                adapter_registry=ProviderAdapterRegistry((recording,)),
                state_ledger=fixture.runner.state_ledger,
            )
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            source = snapshot_workspace(fixture.root, fixture.config.workspace.scope)
            generator = runner.candidate_generator()
            plan = generator.plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )
            with generator.generate_candidate(
                task,
                source_workspace=fixture.root,
                expected_source_fingerprint=source.fingerprint,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
                confirmation=str(plan["confirmation_id"]),
            ):
                pass

        self.assertGreater(recording.calls["build_command_plan"], 0)
        self.assertGreater(recording.calls["execute_command"], 0)

    def test_candidate_generator_requires_immutable_adapter_configuration(self) -> None:
        with _fake_bridge() as fixture:
            delegate = default_provider_adapter_registry().require("codex_cli")
            recording = _RecordingProviderAdapter(delegate)
            del recording.configuration_sha256
            runner = AssistantBridgeRunner.with_provider_adapters(
                fixture.config,
                adapter_registry=ProviderAdapterRegistry((recording,)),
                state_ledger=fixture.runner.state_ledger,
            )

            with self.assertRaisesRegex(
                AssistantBridgeError,
                "immutable configuration digest",
            ):
                _ = runner.candidate_generator().configuration_sha256

    def test_premium_write_without_remote_opt_in_falls_back_local(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="quality",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
                allow_remote_workspace=False,
            )
            plan = fixture.runner.candidate_generator().plan_candidate(
                task,
                workspace=fixture.root,
                expected_config_sha256=LIFECYCLE_CONFIG_SHA256,
            )

        self.assertEqual(plan["route_receipt"]["route"], "local")
        self.assertIn(
            "premium_capability_or_authority_gap",
            plan["route_receipt"]["rationale_codes"],
        )
        self.assertIsNotNone(plan["confirmation_id"])


if __name__ == "__main__":
    unittest.main()
