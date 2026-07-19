from __future__ import annotations

from collections import defaultdict
import unittest

from local_moe.config import ConfigError, parse_config
from local_moe.execution_scope import (
    ExecutionAttestation,
    ExecutionDeclaration,
    ExecutionPolicy,
    ExecutionScope,
    ExecutionScopeGuard,
    ExecutionTarget,
    ExecutionTransport,
    SCOPE_BLOCKED,
    ScopePolicyError,
    is_loopback_endpoint,
    normalized_execution_declaration,
)
from local_moe.orchestrator import LocalMoE
from local_moe.providers import ExpertResult, ProviderError, ProviderStreamEvent
from local_moe.router import RuleRouter


def _expert(
    expert_id: str,
    *,
    provider: str = "synthetic",
    base_url: str | None = None,
    weight: float = 1.0,
    execution: dict[str, str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": expert_id,
        "provider": provider,
        "model": f"{expert_id}-model",
        "role": expert_id,
        "weight": weight,
    }
    if base_url is not None:
        payload["base_url"] = base_url
    if execution is not None:
        payload["execution"] = execution
    return payload


def _config(
    experts: list[dict[str, object]],
    *,
    top_k: int = 1,
    aggregation: str = "best",
    fallback_order: list[str] | None = None,
    execution: dict[str, object] | None = None,
    rules: list[dict[str, object]] | None = None,
):
    raw: dict[str, object] = {
        "routing": {
            "top_k": top_k,
            "aggregation": aggregation,
            "fallback_order": fallback_order or [],
        },
        "experts": experts,
        "rules": rules or [],
    }
    if execution is not None:
        raw["execution"] = execution
    return parse_config(raw)


class FixedAttestor:
    def __init__(self, scopes: dict[str, ExecutionScope]):
        self._scopes = scopes
        self.calls: defaultdict[str, int] = defaultdict(int)

    def attest(self, target: ExecutionTarget) -> ExecutionAttestation:
        self.calls[target.expert_id] += 1
        return ExecutionAttestation(
            expert_id=target.expert_id,
            scope=self._scopes[target.expert_id],
            transport=target.declaration.transport or ExecutionTransport.GATEWAY,
            authority="test",
        )


class RecheckAttestor(FixedAttestor):
    def __init__(self, scopes: dict[str, ExecutionScope], *, blocked: set[str]):
        super().__init__(scopes)
        self._blocked = blocked

    def attest(self, target: ExecutionTarget) -> ExecutionAttestation:
        if self.calls[target.expert_id] >= 1 and target.expert_id in self._blocked:
            self.calls[target.expert_id] += 1
            raise ScopePolicyError(
                "fresh execution evidence is unavailable.",
                expert_id=target.expert_id,
            )
        return super().attest(target)


class CountingProvider:
    def __init__(self, *, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def generate(self, expert, req):
        self.calls += 1
        if self.fail:
            raise ProviderError("endpoint unavailable")
        return ExpertResult(
            expert_id=expert.id,
            model=expert.model,
            content="ok",
            correlation_id=req.correlation_id,
        )


class CountingStreamProvider(CountingProvider):
    def stream_generate(self, expert, req):
        self.calls += 1
        result = ExpertResult(
            expert_id=expert.id,
            model=expert.model,
            content="ok",
            correlation_id=req.correlation_id,
        )
        yield ProviderStreamEvent(content=result.content)
        yield ProviderStreamEvent(content=result.content, result=result)


class ExecutionScopeContractTests(unittest.TestCase):
    def test_default_policy_is_device_only_and_does_not_widen(self) -> None:
        policy = ExecutionPolicy()

        self.assertEqual(policy.max_scope, ExecutionScope.DEVICE_ONLY)
        self.assertEqual(policy.allowed_scopes, (ExecutionScope.DEVICE_ONLY,))
        self.assertFalse(policy.allow_scope_widening)

    def test_synthetic_and_loopback_direct_transport_infer_device_only(self) -> None:
        synthetic = normalized_execution_declaration(
            provider="synthetic",
            endpoint=None,
        )
        loopback = normalized_execution_declaration(
            provider="openai_compatible",
            endpoint="http://127.0.0.1:8101/v1",
        )

        self.assertEqual(synthetic.scope, ExecutionScope.DEVICE_ONLY)
        self.assertEqual(synthetic.transport, ExecutionTransport.DIRECT_LOCAL)
        self.assertEqual(loopback.scope, ExecutionScope.DEVICE_ONLY)
        self.assertEqual(loopback.transport, ExecutionTransport.DIRECT_LOCAL)

    def test_loopback_mesh_transport_never_infers_device_only(self) -> None:
        declaration = normalized_execution_declaration(
            provider="openai_compatible",
            endpoint="http://127.0.0.1:8101/v1",
            declaration=ExecutionDeclaration(
                transport=ExecutionTransport.MESH_LLM,
            ),
        )

        self.assertIsNone(declaration.scope)
        self.assertEqual(declaration.transport, ExecutionTransport.MESH_LLM)

    def test_loopback_detection_rejects_lookalike_hosts(self) -> None:
        self.assertTrue(is_loopback_endpoint("http://localhost:8101/v1"))
        self.assertTrue(is_loopback_endpoint("http://[::1]:8101/v1"))
        self.assertFalse(is_loopback_endpoint("http://127.0.0.1.example:8101/v1"))
        self.assertFalse(is_loopback_endpoint("file:///tmp/model"))

    def test_remote_endpoint_is_unresolved_and_blocked_by_default(self) -> None:
        config = _config(
            [
                _expert(
                    "remote",
                    provider="openai_compatible",
                    base_url="https://models.example.test/v1",
                )
            ]
        )

        with self.assertRaises(ScopePolicyError) as raised:
            RuleRouter(config).route("hello")

        self.assertEqual(raised.exception.reason_code, SCOPE_BLOCKED)
        self.assertIn(SCOPE_BLOCKED, str(raised.exception))

    def test_mesh_transport_requires_external_attestation_even_on_loopback(self) -> None:
        config = _config(
            [
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "device_only",
                        "transport": "mesh_llm",
                    },
                )
            ]
        )

        with self.assertRaises(ScopePolicyError) as raised:
            RuleRouter(config).route("hello")

        self.assertIn("external attestor", str(raised.exception))

    def test_external_attestor_can_prove_private_mesh_scope(self) -> None:
        config = _config(
            [
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "private_mesh",
                        "transport": "mesh_llm",
                    },
                )
            ],
            execution={"max_scope": "private_mesh"},
        )
        attestor = FixedAttestor({"mesh": ExecutionScope.PRIVATE_MESH})
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)

        decision = RuleRouter(config, execution_guard=guard).route("hello")

        self.assertEqual(decision.selected[0].expert_id, "mesh")
        self.assertEqual(attestor.calls["mesh"], 1)

    def test_fresh_attestation_preserves_its_original_authority(self) -> None:
        config = _config([_expert("local")])
        attestor = FixedAttestor({"local": ExecutionScope.DEVICE_ONLY})
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)

        attestation = guard.require_allowed(config.experts[0].execution_target)

        self.assertEqual(attestation.authority, "test")

    def test_attested_scope_must_match_the_declaration(self) -> None:
        config = _config(
            [
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "private_mesh",
                        "transport": "mesh_llm",
                    },
                )
            ],
            execution={"max_scope": "public_mesh"},
        )
        attestor = FixedAttestor({"mesh": ExecutionScope.PUBLIC_MESH})
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)

        with self.assertRaises(ScopePolicyError):
            RuleRouter(config, execution_guard=guard).route("hello")

    def test_router_filters_ineligible_expert_before_scoring(self) -> None:
        config = _config(
            [
                _expert(
                    "remote",
                    provider="openai_compatible",
                    base_url="https://models.example.test/v1",
                    weight=10,
                ),
                _expert("local", weight=1),
            ],
            rules=[
                {"expert_id": "remote", "keywords": ["remote"], "weight": 100}
            ],
        )

        decision = RuleRouter(config).route("remote task")

        self.assertEqual(tuple(item.expert_id for item in decision.selected), ("local",))

    def test_fallback_cannot_widen_scope_without_explicit_policy(self) -> None:
        config = _config(
            [
                _expert("local", weight=10),
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "private_mesh",
                        "transport": "mesh_llm",
                    },
                ),
            ],
            fallback_order=["mesh"],
            execution={"max_scope": "private_mesh"},
        )
        attestor = FixedAttestor(
            {
                "local": ExecutionScope.DEVICE_ONLY,
                "mesh": ExecutionScope.PRIVATE_MESH,
            }
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)

        decision = RuleRouter(config, execution_guard=guard).route("hello")

        self.assertEqual(decision.selected[0].expert_id, "local")
        self.assertEqual(decision.fallback_order, ())

    def test_fallback_can_widen_only_when_policy_opts_in(self) -> None:
        config = _config(
            [
                _expert("local", weight=10),
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "private_mesh",
                        "transport": "mesh_llm",
                    },
                ),
            ],
            fallback_order=["mesh"],
            execution={
                "max_scope": "private_mesh",
                "allow_scope_widening": True,
            },
        )
        attestor = FixedAttestor(
            {
                "local": ExecutionScope.DEVICE_ONLY,
                "mesh": ExecutionScope.PRIVATE_MESH,
            }
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)

        decision = RuleRouter(config, execution_guard=guard).route("hello")

        self.assertEqual(decision.fallback_order, ("mesh",))


class InvocationGuardTests(unittest.TestCase):
    def test_best_path_rechecks_immediately_before_provider_invocation(self) -> None:
        config = _config([_expert("local")])
        attestor = RecheckAttestor(
            {"local": ExecutionScope.DEVICE_ONLY},
            blocked={"local"},
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        provider = CountingProvider()
        moe = LocalMoE(config, execution_guard=guard)
        moe._providers["local"] = provider

        with self.assertRaises(ScopePolicyError) as raised:
            moe.generate("hello")

        self.assertEqual(raised.exception.reason_code, SCOPE_BLOCKED)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(attestor.calls["local"], 2)

    def test_stream_path_rechecks_before_stream_invocation(self) -> None:
        config = _config([_expert("local")])
        attestor = RecheckAttestor(
            {"local": ExecutionScope.DEVICE_ONLY},
            blocked={"local"},
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        provider = CountingStreamProvider()
        moe = LocalMoE(config, execution_guard=guard)
        moe._providers["local"] = provider

        with self.assertRaises(ScopePolicyError):
            list(moe.generate_stream("hello"))

        self.assertEqual(provider.calls, 0)

    def test_parallel_path_rechecks_each_expert_without_absorbing_scope_error(self) -> None:
        config = _config(
            [_expert("left"), _expert("right")],
            top_k=2,
            aggregation="compare",
        )
        attestor = RecheckAttestor(
            {
                "left": ExecutionScope.DEVICE_ONLY,
                "right": ExecutionScope.DEVICE_ONLY,
            },
            blocked={"right"},
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        left = CountingProvider()
        right = CountingProvider()
        moe = LocalMoE(config, execution_guard=guard)
        moe._providers.update({"left": left, "right": right})

        with self.assertRaises(ScopePolicyError):
            moe.generate("hello")

        self.assertEqual(left.calls, 1)
        self.assertEqual(right.calls, 0)

    def test_compare_fallback_rechecks_before_invocation(self) -> None:
        config = _config(
            [
                _expert("left", weight=10),
                _expert("right", weight=9),
                _expert("fallback", weight=1),
            ],
            top_k=2,
            aggregation="compare",
            fallback_order=["fallback"],
        )
        attestor = RecheckAttestor(
            {
                "left": ExecutionScope.DEVICE_ONLY,
                "right": ExecutionScope.DEVICE_ONLY,
                "fallback": ExecutionScope.DEVICE_ONLY,
            },
            blocked={"fallback"},
        )
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        left = CountingProvider()
        right = CountingProvider(fail=True)
        fallback = CountingProvider()
        moe = LocalMoE(config, execution_guard=guard)
        moe._providers.update(
            {"left": left, "right": right, "fallback": fallback}
        )

        with self.assertRaises(ScopePolicyError):
            moe.generate("hello")

        self.assertEqual(left.calls, 1)
        self.assertEqual(right.calls, 1)
        self.assertEqual(fallback.calls, 0)

    def test_blocked_fallback_is_never_invoked_after_local_failure(self) -> None:
        config = _config(
            [
                _expert("local", weight=10),
                _expert(
                    "remote",
                    provider="openai_compatible",
                    base_url="https://models.example.test/v1",
                ),
            ],
            fallback_order=["remote"],
        )
        local = CountingProvider(fail=True)
        remote = CountingProvider()
        moe = LocalMoE(config)
        moe._providers.update({"local": local, "remote": remote})

        with self.assertRaises(ProviderError):
            moe.generate("hello")

        self.assertEqual(local.calls, 1)
        self.assertEqual(remote.calls, 0)


class ExecutionConfigTests(unittest.TestCase):
    def test_parses_policy_and_expert_declaration(self) -> None:
        config = _config(
            [
                _expert(
                    "mesh",
                    provider="openai_compatible",
                    base_url="http://127.0.0.1:8080/v1",
                    execution={
                        "scope": "private_mesh",
                        "transport": "mesh_llm",
                    },
                )
            ],
            execution={
                "max_scope": "private_mesh",
                "allowed_scopes": ["private_mesh"],
                "allow_scope_widening": True,
            },
        )

        self.assertEqual(config.execution_policy.max_scope, ExecutionScope.PRIVATE_MESH)
        self.assertEqual(
            config.execution_policy.allowed_scopes,
            (ExecutionScope.PRIVATE_MESH,),
        )
        self.assertTrue(config.execution_policy.allow_scope_widening)
        self.assertEqual(
            config.experts[0].execution.transport,
            ExecutionTransport.MESH_LLM,
        )

    def test_rejects_scope_above_policy_maximum(self) -> None:
        with self.assertRaisesRegex(ConfigError, "cannot exceed"):
            _config(
                [_expert("local")],
                execution={
                    "max_scope": "device_only",
                    "allowed_scopes": ["private_mesh"],
                },
            )

    def test_rejects_unknown_transport(self) -> None:
        with self.assertRaisesRegex(ConfigError, "expert.execution.transport"):
            _config(
                [
                    _expert(
                        "local",
                        execution={"transport": "teleport"},
                    )
                ]
            )

    def test_rejects_non_boolean_scope_widening(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must be boolean"):
            _config(
                [_expert("local")],
                execution={"allow_scope_widening": "false"},
            )


if __name__ == "__main__":
    unittest.main()
