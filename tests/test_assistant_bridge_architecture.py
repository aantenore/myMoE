from __future__ import annotations

import ast
from importlib import metadata
import importlib
import inspect
import json
import os
from pathlib import Path
import pkgutil
import re
import subprocess
import sys
import tempfile
import unittest

import local_moe
import local_moe.assistant_bridge as assistant_bridge


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "local_moe"

# This is a minimum compatibility contract, not an exhaustive ``dir()`` snapshot.
# New public names may be added without weakening the characterization test; removing
# or renaming one of these names requires an explicit compatibility decision.
SUPPORTED_FACADE_API = frozenset(
    {
        "BRIDGE_SCHEMA_VERSION",
        "PROFILES",
        "RISK_LEVELS",
        "ROUTES",
        "AssistantBridgeConfig",
        "AssistantBridgeCandidateGenerator",
        "AssistantBridgeError",
        "AssistantBridgeRunner",
        "AssistantTaskBudget",
        "AssistantTaskEnvelope",
        "BoundVerifierPlan",
        "BridgeRunResult",
        "BridgeRuntimePolicy",
        "BridgeStatePolicy",
        "BridgeWorkspacePolicy",
        "CapabilityDemand",
        "CapsulePolicy",
        "CandidateGenerationRequest",
        "CommandPlan",
        "CommandResult",
        "CommandVerifierSpec",
        "DiffEvidence",
        "EscalationCapsule",
        "ExternalVerifierSpec",
        "GitIdentity",
        "PremiumAuthAttestation",
        "ProfilePolicy",
        "ProviderAdapter",
        "ProviderAdapterRegistry",
        "ProviderAdapterRegistryError",
        "ProviderAuthorityBinding",
        "ProviderSpec",
        "RouteDecisionReceipt",
        "StagedPremiumAuthAttestation",
        "VerificationEvidence",
        "WorkspaceAttestation",
        "attest_workspace",
        "build_assistant_task",
        "build_codex_command_plan",
        "build_escalation_capsule",
        "build_local_prompt",
        "build_premium_prompt",
        "collect_git_diff",
        "collect_git_evidence",
        "execute_codex_command",
        "default_provider_adapter_registry",
        "load_assistant_bridge_config",
        "load_assistant_task",
        "load_verification_evidence",
        "plan_assistant_route",
        "provider_gaps",
        "redact_and_bound",
        "verify_command_result",
    }
)

# These identities are compatibility seams today: callers and tests can patch them
# through the facade. Future modules may own the implementation, but the facade must
# continue to expose the same objects until a deliberate breaking release.
COMPATIBILITY_SEAM_REEXPORTS: dict[str, tuple[str, ...]] = {
    "local_moe.assistant_bridge_provider_registry": (
        "ProviderAdapterRegistry",
        "ProviderAdapterRegistryError",
    ),
    "local_moe.assistant_bridge_resources": (
        "VerifierResourceCapabilities",
        "VerifierResourceEnforcementReport",
        "VerifierResourceError",
        "VerifierResourcePlan",
        "VerifierResourcePolicy",
        "build_verifier_resource_enforcement_report",
        "build_verifier_resource_plan",
        "verifier_resource_capabilities",
    ),
    "local_moe.assistant_bridge_ledger": (
        "BridgeLedgerError",
        "BridgeStateLedger",
        "PremiumBudgetLease",
        "budget_key",
    ),
    "local_moe.assistant_bridge_runtime": (
        "AssistantBridgeRuntimeError",
        "ExecutableIdentity",
        "LauncherChainIdentity",
        "ProcessCleanupError",
        "ProcessExecutionPolicy",
        "ProcessLaunchLifecycleError",
        "ProcessLaunchNotAuthorizedError",
        "ProcessLaunchPermit",
        "execute_process",
        "fingerprint_environment",
        "resolve_executable",
        "resolve_launcher_chain",
        "runtime_capabilities",
        "validate_environment_name",
    ),
    "local_moe.assistant_bridge_secrets": (
        "ResidualAssuranceUnavailableError",
        "SecretRedactionPolicy",
        "redact_text",
        "redact_user_controlled_fields",
    ),
    "local_moe.assistant_bridge_workspace": (
        "GitIdentity",
        "IgnoredPathRule",
        "MaterializedWorkspace",
        "WorkspaceChange",
        "WorkspaceFile",
        "WorkspaceScopePolicy",
        "WorkspaceSecurityError",
        "WorkspaceSnapshot",
        "WorkspaceWriteCapability",
        "apply_changeset",
        "build_changeset",
        "materialize_workspace",
        "snapshot_workspace",
        "trusted_git_session",
        "workspace_write_capability",
    ),
}

PUBLIC_CALL_SIGNATURES = {
    "AssistantBridgeRunner": (
        "(config: 'AssistantBridgeConfig', *, state_ledger: "
        "'BridgeStateLedger | None' = None) -> 'None'"
    ),
    "attest_workspace": "(workspace: 'str | Path') -> 'WorkspaceAttestation'",
    "build_assistant_task": (
        "(objective: 'str', *, profile: 'str' = 'balanced', "
        "required_capabilities: 'Sequence[str]' = (), required_tools: "
        "'Sequence[str]' = (), risk_class: 'str' = 'read_only', constraints: "
        "'Sequence[str]' = (), no_change_expected: 'bool' = False, "
        "required_verifier_ids: 'Sequence[str]' = (), allow_remote: "
        "'bool | None' = None, allow_remote_workspace: 'bool' = False, "
        "max_premium_calls: 'int | None' = None) -> 'AssistantTaskEnvelope'"
    ),
    "build_codex_command_plan": (
        "(provider: 'ProviderSpec', *, prompt: 'str', workspace: 'str | Path', "
        "demand: 'CapabilityDemand | None' = None, output_path: "
        "'str | Path | None' = None, local_provider_override: 'str | None' = "
        "None, workspace_access: 'str | None' = None, runtime_policy: "
        "'BridgeRuntimePolicy | None' = None, ephemeral_workspace: 'bool' = "
        "False) -> 'CommandPlan'"
    ),
    "build_escalation_capsule": (
        "(task: 'AssistantTaskEnvelope', receipt: 'RouteDecisionReceipt', "
        "verification: 'Sequence[VerificationEvidence]', policy: "
        "'CapsulePolicy', *, failure_codes: 'Sequence[str]', diff_text: 'str' "
        "= '', diff_evidence: 'DiffEvidence | None' = None, "
        "workspace_fingerprint: 'str | None' = None) -> 'EscalationCapsule'"
    ),
    "build_local_prompt": "(task: 'AssistantTaskEnvelope') -> 'str'",
    "build_premium_prompt": "(capsule: 'EscalationCapsule') -> 'str'",
    "collect_git_diff": (
        "(workspace: 'str | Path', *, max_chars: 'int' = 100000) -> 'str'"
    ),
    "collect_git_evidence": (
        "(workspace: 'str | Path', policy: 'CapsulePolicy', *, include_excerpt: "
        "'bool', expected_snapshot: 'WorkspaceSnapshot | None' = None, "
        "workspace_policy: 'WorkspaceScopePolicy | None' = None) -> "
        "'DiffEvidence'"
    ),
    "execute_codex_command": (
        "(plan: 'CommandPlan', *, prompt: 'str', output_path: 'str | Path', "
        "timeout_seconds: 'float', environment_overrides: "
        "'Mapping[str, str] | None' = None, reserve_launch: "
        "'Callable[[], ProcessLaunchPermit | None] | None' = None) -> "
        "'CommandResult'"
    ),
    "default_provider_adapter_registry": (
        "() -> 'ProviderAdapterRegistry[ProviderAdapter]'"
    ),
    "load_assistant_bridge_config": (
        "(path: 'str | Path') -> 'AssistantBridgeConfig'"
    ),
    "load_assistant_task": "(path: 'str | Path') -> 'AssistantTaskEnvelope'",
    "load_verification_evidence": (
        "(path: 'str | Path') -> 'tuple[VerificationEvidence, ...]'"
    ),
    "plan_assistant_route": (
        "(task: 'AssistantTaskEnvelope', config: 'AssistantBridgeConfig', *, "
        "workspace: 'str | Path' = '.', local_provider_override: 'str | None' "
        "= None, workspace_snapshot: 'WorkspaceSnapshot | None' = None) -> "
        "'RouteDecisionReceipt'"
    ),
    "provider_gaps": (
        "(provider: 'ProviderSpec', task_or_demand: "
        "'AssistantTaskEnvelope | CapabilityDemand') -> 'tuple[str, ...]'"
    ),
    "redact_and_bound": (
        "(value: 'str', max_chars: 'int') -> 'tuple[str, int, bool]'"
    ),
    "verify_command_result": (
        "(result: 'CommandResult', config: 'AssistantBridgeConfig', *, task: "
        "'AssistantTaskEnvelope', workspace: 'WorkspaceAttestation', "
        "external_evidence: 'Sequence[VerificationEvidence]' = (), "
        "verifier_workspace: 'str | Path', verifier_plans: "
        "'Sequence[BoundVerifierPlan]' = ()) -> "
        "'tuple[VerificationEvidence, ...]'"
    ),
}

RUNNER_METHOD_SIGNATURES = {
    "candidate_generator": ("(self) -> 'AssistantBridgeCandidateGenerator'"),
    "plan": (
        "(self, task: 'AssistantTaskEnvelope', *, workspace: 'str | Path', "
        "local_provider_override: 'str | None' = None, external_evidence: "
        "'Sequence[VerificationEvidence]' = (), include_diff: 'bool' = False, "
        "capsule_out: 'str | Path | None' = None) -> 'dict[str, object]'"
    ),
    "run": (
        "(self, task: 'AssistantTaskEnvelope', *, workspace: 'str | Path', "
        "confirmation: 'str', local_provider_override: 'str | None' = None, "
        "external_evidence: 'Sequence[VerificationEvidence]' = (), "
        "include_diff: 'bool' = False, capsule_out: 'str | Path | None' = "
        "None) -> 'BridgeRunResult'"
    ),
    "with_provider_adapters": (
        "(config: 'AssistantBridgeConfig', *, adapter_registry: "
        "'ProviderAdapterRegistry[ProviderAdapter]', state_ledger: "
        "'BridgeStateLedger | None' = None) -> 'AssistantBridgeRunner'"
    ),
}

MINIMUM_SEAM_MODULES = frozenset(COMPATIBILITY_SEAM_REEXPORTS)


class AssistantBridgeFacadeContractTests(unittest.TestCase):
    def test_facade_exposes_the_supported_minimum_api(self) -> None:
        missing = sorted(SUPPORTED_FACADE_API.difference(dir(assistant_bridge)))

        self.assertEqual(missing, [])

    def test_public_call_signatures_remain_compatible(self) -> None:
        observed = {
            name: str(inspect.signature(getattr(assistant_bridge, name)))
            for name in sorted(PUBLIC_CALL_SIGNATURES)
        }

        self.assertEqual(observed, PUBLIC_CALL_SIGNATURES)

    def test_runner_method_signatures_remain_compatible(self) -> None:
        observed = {
            name: str(
                inspect.signature(getattr(assistant_bridge.AssistantBridgeRunner, name))
            )
            for name in sorted(RUNNER_METHOD_SIGNATURES)
        }

        self.assertEqual(observed, RUNNER_METHOD_SIGNATURES)

    def test_existing_seams_are_identity_reexports(self) -> None:
        mismatches: list[str] = []
        for module_name in sorted(COMPATIBILITY_SEAM_REEXPORTS):
            owner = importlib.import_module(module_name)
            for name in sorted(COMPATIBILITY_SEAM_REEXPORTS[module_name]):
                if getattr(assistant_bridge, name, None) is not getattr(owner, name):
                    mismatches.append(f"{module_name}:{name}")

        self.assertEqual(mismatches, [])


class AssistantBridgeImportContractTests(unittest.TestCase):
    def test_facade_and_seams_import_in_both_dependency_orders(self) -> None:
        seams = tuple(sorted(MINIMUM_SEAM_MODULES))
        orders = (
            ("local_moe.assistant_bridge", "local_moe.cli", *seams),
            (*seams, "local_moe.assistant_bridge", "local_moe.cli"),
        )
        script = "\n".join(
            (
                "import importlib",
                "import json",
                "import sys",
                "modules = json.loads(sys.argv[1])",
                "required = set(json.loads(sys.argv[2]))",
                "for module in modules:",
                "    importlib.import_module(module)",
                "facade = importlib.import_module('local_moe.assistant_bridge')",
                "missing = sorted(required.difference(dir(facade)))",
                "print(json.dumps({'missing': missing, 'modules': modules}, sort_keys=True))",
            )
        )
        env = dict(os.environ)
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(ROOT / "src"), existing_path) if part
        )

        with tempfile.TemporaryDirectory(prefix="mymoe-import-") as temporary:
            for order in orders:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script,
                        json.dumps(order),
                        json.dumps(sorted(SUPPORTED_FACADE_API)),
                    ],
                    cwd=temporary,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["missing"], [])
                self.assertEqual(payload["modules"], list(order))

    def test_seam_modules_do_not_depend_back_on_the_facade(self) -> None:
        violations: list[str] = []
        seam_paths = sorted(PACKAGE_ROOT.glob("assistant_bridge_*.py"))

        for path in seam_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(
                        alias.name == "local_moe.assistant_bridge"
                        for alias in node.names
                    ):
                        violations.append(path.name)
                elif isinstance(node, ast.ImportFrom):
                    absolute_facade = (
                        node.level == 0 and node.module == "local_moe.assistant_bridge"
                    )
                    relative_facade = node.level == 1 and node.module == "assistant_bridge"
                    if absolute_facade or relative_facade:
                        violations.append(path.name)

        self.assertEqual(sorted(set(violations)), [])


class AssistantBridgePackagingContractTests(unittest.TestCase):
    def test_src_discovery_and_package_exports_keep_bridge_modules_installable(
        self,
    ) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        discovery = _toml_section(pyproject, "tool.setuptools.packages.find")
        scripts = _toml_section(pyproject, "project.scripts")
        discovered_modules = {
            f"local_moe.{item.name}"
            for item in pkgutil.iter_modules(local_moe.__path__)
        }

        self.assertRegex(
            discovery,
            r"(?m)^\s*where\s*=\s*\[\s*[\"']src[\"']\s*\]\s*$",
        )
        self.assertIn('mymoe = "local_moe.cli:main"', scripts)
        self.assertIn(
            'mymoe-paired = "local_moe.paired_execution_cli:main"',
            scripts,
        )
        self.assertIn('mymoe-web = "local_moe.web:main"', scripts)
        self.assertIn("assistant_bridge", local_moe.__all__)
        self.assertTrue(MINIMUM_SEAM_MODULES.issubset(discovered_modules))
        self.assertIn("local_moe.assistant_bridge", discovered_modules)

    def test_installed_distribution_imports_the_facade_without_source_path_help(
        self,
    ) -> None:
        try:
            distribution = metadata.distribution("local-moe-orchestrator")
        except metadata.PackageNotFoundError:
            self.skipTest("installed distribution metadata is not available")

        entry_points = {
            (entry_point.group, entry_point.name, entry_point.value)
            for entry_point in distribution.entry_points
        }
        script = "\n".join(
            (
                "import importlib",
                "import json",
                "import sys",
                "required = set(json.loads(sys.argv[1]))",
                "facade = importlib.import_module('local_moe.assistant_bridge')",
                "missing = sorted(required.difference(dir(facade)))",
                "print(json.dumps({'missing': missing, 'module': facade.__name__}, sort_keys=True))",
            )
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() != "PYTHONPATH"
        }

        with tempfile.TemporaryDirectory(prefix="mymoe-installed-import-") as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    script,
                    json.dumps(sorted(SUPPORTED_FACADE_API)),
                ],
                cwd=temporary,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            json.loads(completed.stdout),
            {"missing": [], "module": "local_moe.assistant_bridge"},
        )
        self.assertTrue(
            {
                ("console_scripts", "mymoe", "local_moe.cli:main"),
                ("console_scripts", "mymoe-web", "local_moe.web:main"),
            }.issubset(entry_points)
        )


def _toml_section(source: str, name: str) -> str:
    marker = f"[{name}]"
    try:
        start = source.index(marker) + len(marker)
    except ValueError as exc:
        raise AssertionError(f"Missing {marker} in pyproject.toml") from exc
    remainder = source[start:]
    next_table = re.search(r"(?m)^\s*\[", remainder)
    return remainder[: next_table.start()] if next_table else remainder


if __name__ == "__main__":
    unittest.main()
