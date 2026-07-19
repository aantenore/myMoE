from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import (
    ED25519_DSSE_ADAPTER_ID,
    create_ed25519_evaluation_dsse_envelope,
    ed25519_public_key_sha256,
)
from local_moe.assistant_bridge_cas import ContentAddressedStore
from local_moe.assistant_bridge_integrity import sha256_bytes
from local_moe.assistant_bridge_ledger import BridgeStateLedger
from local_moe.assistant_bridge_two_phase_config import (
    LoadedPublicVerifier,
    TwoPhaseTrustConfig,
)
from local_moe.assistant_bridge_two_phase_contracts import (
    AttestationCheck,
    CandidateBinding,
    VerificationPolicy,
    VerifierRequirement,
)
from local_moe.paired_execution_bridge import AssistantBridgePairedArmExecutor


class SigningPairedAttestationProducer:
    def __init__(
        self,
        requirements: tuple[VerifierRequirement, ...],
        private_keys: tuple[Ed25519PrivateKey, ...],
        *,
        passed: bool = True,
    ) -> None:
        self.requirements = requirements
        self.private_keys = private_keys
        self.passed = passed
        self.configuration_sha256 = sha256_bytes(
            b"test-signed-paired-attestation-producer/v1"
        )
        self.calls: list[tuple[CandidateBinding, Path, float]] = []
        self._state_paths: tuple[Path, ...] = ()

    @property
    def state_paths(self) -> tuple[Path, ...]:
        return self._state_paths

    def attest(
        self,
        binding: CandidateBinding,
        workspace: Path,
        deadline: float,
    ) -> tuple[bytes, ...]:
        self.calls.append((binding, workspace, deadline))
        return tuple(
            create_ed25519_evaluation_dsse_envelope(
                binding,
                requirement,
                private_key,
                attestation_id=(
                    f"paired-{requirement.verifier_id}-"
                    f"{binding.stage_idempotency_sha256[:24]}"
                ),
                issued_at=binding.created_at,
                expires_at=binding.expires_at,
                checks=(
                    AttestationCheck(
                        "signed-check",
                        self.passed,
                        sha256_bytes(
                            b"signed-check-passed"
                            if self.passed
                            else b"signed-check-failed"
                        ),
                    ),
                ),
            )
            for requirement, private_key in zip(
                self.requirements,
                self.private_keys,
                strict=True,
            )
        )


@dataclass(frozen=True)
class SignedPairedExecutorFixture:
    executor: AssistantBridgePairedArmExecutor
    producer: SigningPairedAttestationProducer
    trust_config: TwoPhaseTrustConfig
    evidence_store: ContentAddressedStore
    requirements: tuple[VerifierRequirement, ...]
    private_keys: tuple[Ed25519PrivateKey, ...]


def build_signed_paired_executor(
    bridge_fixture,
    evidence_root: str | Path,
    *,
    verifier_ids: tuple[str, ...] = ("focused-tests",),
    quorum: int | None = None,
) -> SignedPairedExecutorFixture:
    evidence_path = Path(evidence_root).expanduser().absolute()
    ledger_root = evidence_path.parent / f".{evidence_path.name}-bridge-state"
    ledger_root.mkdir(mode=0o700, exist_ok=True)
    ledger_root.chmod(0o700)
    state = bridge_fixture.runner.config.state
    bridge_fixture.runner.state_ledger = BridgeStateLedger(
        ledger_root / "ledger.json",
        namespace=state.namespace,
        lock_timeout_seconds=state.lock_timeout_seconds,
        stale_lock_seconds=state.stale_lock_seconds,
        budget_retention_seconds=state.budget_retention_seconds,
        max_budget_entries=state.max_budget_entries,
        confirmation_retention_seconds=state.confirmation_retention_seconds,
        max_confirmation_entries=state.max_confirmation_entries,
        budget_lease_ttl_seconds=state.budget_lease_ttl_seconds,
    )
    private_keys = tuple(Ed25519PrivateKey.generate() for _ in verifier_ids)
    requirements = tuple(
        VerifierRequirement(
            verifier_id=verifier_id,
            adapter_id=ED25519_DSSE_ADAPTER_ID,
            key_id=f"{verifier_id}-key",
            public_key_sha256=ed25519_public_key_sha256(private_key.public_key()),
            spec_sha256=(
                bridge_fixture.runner.config.external_verifiers[
                    verifier_id
                ].spec_sha256
            ),
        )
        for verifier_id, private_key in zip(
            verifier_ids,
            private_keys,
            strict=True,
        )
    )
    loaded_verifiers = []
    for requirement, private_key in zip(
        requirements,
        private_keys,
        strict=True,
    ):
        public_key_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        loaded_verifiers.append(
            LoadedPublicVerifier(
                requirement=requirement,
                public_key_path=(
                    Path(evidence_root)
                    / f"{requirement.verifier_id}-public.pem"
                ),
                public_key_pem=public_key_pem,
                public_key_file_sha256=sha256_bytes(public_key_pem),
            )
        )
    trust_config = TwoPhaseTrustConfig(
        policy=VerificationPolicy(
            "paired-policy-v1",
            len(requirements) if quorum is None else quorum,
            requirements,
        ),
        verifiers=tuple(loaded_verifiers),
    )
    evidence_store = ContentAddressedStore(evidence_root)
    producer = SigningPairedAttestationProducer(
        requirements,
        private_keys,
    )
    executor = AssistantBridgePairedArmExecutor(
        bridge_fixture.runner,
        attestation_producer=producer,
        trust_config=trust_config,
        evidence_store=evidence_store,
    )
    return SignedPairedExecutorFixture(
        executor=executor,
        producer=producer,
        trust_config=trust_config,
        evidence_store=evidence_store,
        requirements=requirements,
        private_keys=private_keys,
    )
