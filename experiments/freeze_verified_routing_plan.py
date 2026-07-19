from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from local_moe.assistant_bridge import (
    AssistantBridgeRunner,
    load_assistant_bridge_config,
)
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.assistant_bridge_two_phase_config import (
    load_two_phase_lifecycle_config,
)
from local_moe.paired_attestation_directory import (
    DirectoryPairedAttestationProducer,
)
from local_moe.paired_execution import (
    paired_execution_harness_sha256,
    paired_runner_source_sha256,
)
from local_moe.paired_execution_bridge import AssistantBridgePairedArmExecutor
from local_moe.paired_execution_pricing import PricingContract
from local_moe.route_policy import route_policy_from_payload
from local_moe.route_promotion import (
    build_evidence_plan,
    load_promotion_cases,
    load_promotion_gate_policy,
    load_strict_promotion_json,
    write_content_addressed_json,
)
from local_moe.route_scorecard import route_scorecard_from_payload
from local_moe.route_signals import MetadataTaskSignalProvider
from local_moe.verified_routing_contracts import VerifiedRoutingError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a content-addressed paired routing evidence plan."
    )
    parser.add_argument("--cases", required=True, help="Planned case JSON path.")
    parser.add_argument("--route-policy", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--gate-policy", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--canary-basis-points", required=True, type=int)
    parser.add_argument("--manifest-ttl-seconds", required=True, type=int)
    parser.add_argument("--assignment-salt-sha256", required=True)
    parser.add_argument("--assistant-bridge-config", required=True)
    parser.add_argument(
        "--attestation-config",
        required=True,
        help="Two-phase public trust and preinitialized CAS configuration.",
    )
    parser.add_argument(
        "--attestation-exchange-dir",
        required=True,
        help="Preinitialized signed-verifier exchange used by paired execution.",
    )
    parser.add_argument(
        "--pricing-contract",
        required=True,
        help="Canonical verified paired pricing contract JSON path.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cases = load_promotion_cases(args.cases)
    route_policy_raw = load_strict_promotion_json(args.route_policy)
    scorecard_raw = load_strict_promotion_json(args.scorecard)
    if not isinstance(route_policy_raw, dict) or not isinstance(scorecard_raw, dict):
        raise VerifiedRoutingError("Route policy and scorecard must be objects.")
    route_policy = route_policy_from_payload(route_policy_raw)
    scorecard = route_scorecard_from_payload(
        scorecard_raw, require_fresh=False
    )
    gate_policy = load_promotion_gate_policy(args.gate_policy)
    (
        bridge_config_sha256,
        attestation_policy_sha256,
        execution_harness_sha256,
        runner_source_sha256,
    ) = _inspect_execution_harness(
        assistant_bridge_config=args.assistant_bridge_config,
        attestation_config=args.attestation_config,
        attestation_exchange_dir=args.attestation_exchange_dir,
    )
    if any(case.config_sha256 != bridge_config_sha256 for case in cases):
        raise VerifiedRoutingError(
            "Every planned case must match the inspected Assistant Bridge config."
        )
    try:
        pricing_contract = PricingContract.from_json(
            Path(args.pricing_contract).read_bytes()
        )
    except OSError as exc:
        raise VerifiedRoutingError(
            f"Unable to read pricing contract: {args.pricing_contract}."
        ) from exc
    plan = build_evidence_plan(
        cases,
        route_policy=route_policy,
        scorecard=scorecard,
        gate_policy=gate_policy,
        created_at=args.created_at,
        canary_basis_points=args.canary_basis_points,
        manifest_ttl_seconds=args.manifest_ttl_seconds,
        assignment_salt_sha256=args.assignment_salt_sha256,
        attestation_policy_sha256=attestation_policy_sha256,
        execution_harness_sha256=paired_execution_harness_sha256(
            executor_harness_sha256=execution_harness_sha256,
            signal_provider_config_sha256=_single_signal_config_sha256(cases),
        ),
        runner_source_sha256=runner_source_sha256,
        pricing_contract=pricing_contract,
    )
    write_content_addressed_json(args.out, plan)
    print(
        json.dumps(
            {
                "cases": len(plan.cases),
                "plan_sha256": plan.plan_sha256,
                "split_sha256": plan.split_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _inspect_execution_harness(
    *,
    assistant_bridge_config: str | Path,
    attestation_config: str | Path,
    attestation_exchange_dir: str | Path,
) -> tuple[str, str, str, str]:
    bridge = load_assistant_bridge_config(assistant_bridge_config)
    lifecycle = load_two_phase_lifecycle_config(attestation_config)
    evidence_store = ContentAddressedStore(
        lifecycle.state.cas_path,
        create_if_missing=False,
    )
    executor = AssistantBridgePairedArmExecutor(
        AssistantBridgeRunner(bridge),
        attestation_producer=DirectoryPairedAttestationProducer(
            attestation_exchange_dir
        ),
        trust_config=lifecycle.trust,
        evidence_store=evidence_store,
    )
    return (
        bridge.source_sha256,
        lifecycle.trust.policy.policy_sha256,
        executor.execution_harness_sha256,
        paired_runner_source_sha256(),
    )


def _single_signal_config_sha256(cases: Sequence[object]) -> str:
    digests = {
        getattr(case, "signal_provider_config_sha256", None) for case in cases
    }
    if len(digests) != 1 or None in digests:
        raise VerifiedRoutingError(
            "Schema 1.0 requires one signal-provider config per evidence plan."
        )
    executable_digest = MetadataTaskSignalProvider().config_sha256
    if digests != {executable_digest}:
        raise VerifiedRoutingError(
            "Every planned case must use the executable MetadataTaskSignalProvider "
            "configuration."
        )
    return executable_digest


if __name__ == "__main__":
    raise SystemExit(main())
