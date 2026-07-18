from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .assistant_bridge_attestation import (
    AttestationTrustStore,
    ED25519_DSSE_ADAPTER_ID,
    TrustedEd25519Verifier,
    ed25519_public_key_sha256,
    load_ed25519_public_key_pem,
)
from .assistant_bridge_integrity import canonical_sha256, sha256_bytes
from .assistant_bridge_two_phase_contracts import (
    VerificationPolicy,
    VerifierRequirement,
    require_safe_id,
    require_sha256,
)
from .assistant_bridge_two_phase_state import (
    TWO_PHASE_CONFIG_SCHEMA_VERSION,
    TwoPhaseConfigError,
    TwoPhaseStateConfig,
    _configured_path,
    _integer,
    _load_config_document,
    _object,
    _parse_state,
    _reject_unknown,
    _string,
    load_two_phase_state_config as load_two_phase_state_config,
    read_bounded_regular_file,
)


_MAX_PUBLIC_KEY_BYTES = 64 * 1024
_MAX_VERIFIERS = 64
_MAX_TRUST_KEY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LoadedPublicVerifier:
    requirement: VerifierRequirement
    public_key_path: Path = field(repr=False)
    public_key_pem: bytes = field(repr=False)
    public_key_file_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.public_key_file_sha256, "public key file sha256")
        if not isinstance(self.public_key_pem, bytes) or not self.public_key_pem:
            raise TwoPhaseConfigError("Verifier public key is unavailable.")

    def verifier(self) -> TrustedEd25519Verifier:
        return TrustedEd25519Verifier.from_pem(self.requirement, self.public_key_pem)

    def descriptor(self) -> dict[str, object]:
        return {
            "requirement": self.requirement.payload(),
            "publicKeyFileSha256": self.public_key_file_sha256,
        }


@dataclass(frozen=True)
class TwoPhaseTrustConfig:
    policy: VerificationPolicy
    verifiers: tuple[LoadedPublicVerifier, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.verifiers, key=lambda item: item.requirement.verifier_id)
        )
        if tuple(item.requirement for item in ordered) != self.policy.verifiers:
            raise TwoPhaseConfigError(
                "Trust adapters do not match the verification policy."
            )
        object.__setattr__(self, "verifiers", ordered)

    def build_trust_store(self) -> AttestationTrustStore:
        return AttestationTrustStore(tuple(item.verifier() for item in self.verifiers))

    def descriptor(self) -> dict[str, object]:
        return {
            "policy": self.policy.payload(),
            "publicKeys": [item.descriptor() for item in self.verifiers],
        }


@dataclass(frozen=True)
class TwoPhaseLifecycleConfig:
    state: TwoPhaseStateConfig
    trust: TwoPhaseTrustConfig
    source_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_sha256, "two-phase config source_sha256")

    @property
    def effective_sha256(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": TWO_PHASE_CONFIG_SCHEMA_VERSION,
                "sourceSha256": self.source_sha256,
                "state": self.state.payload(),
                "trust": self.trust.descriptor(),
            }
        )


def load_two_phase_lifecycle_config(path: str | Path) -> TwoPhaseLifecycleConfig:
    source, raw, source_sha256 = _load_config_document(path)
    state = _parse_state(raw, config_root=source.parent)
    trust_raw = _object(raw.get("trust"), "trust")
    _reject_unknown("trust", trust_raw, {"policy_id", "quorum", "verifiers"})
    raw_verifiers = trust_raw.get("verifiers")
    if not isinstance(raw_verifiers, list) or not raw_verifiers:
        raise TwoPhaseConfigError("trust.verifiers must be a non-empty list.")
    if len(raw_verifiers) > _MAX_VERIFIERS:
        raise TwoPhaseConfigError("trust.verifiers exceeds safe bounds.")
    loaded_items: list[LoadedPublicVerifier] = []
    trust_key_bytes = 0
    for index, item in enumerate(raw_verifiers):
        verifier = _load_public_verifier(
            item,
            index=index,
            config_root=source.parent,
        )
        trust_key_bytes += len(verifier.public_key_pem)
        if trust_key_bytes > _MAX_TRUST_KEY_BYTES:
            raise TwoPhaseConfigError(
                "Aggregate public verification material exceeds safe bounds."
            )
        loaded_items.append(verifier)
    loaded = tuple(loaded_items)
    try:
        policy = VerificationPolicy(
            policy_id=_string(trust_raw.get("policy_id"), "trust.policy_id"),
            quorum=_integer(trust_raw.get("quorum"), "trust.quorum"),
            verifiers=tuple(item.requirement for item in loaded),
        )
        trust = TwoPhaseTrustConfig(policy=policy, verifiers=loaded)
    except ValueError as exc:
        raise TwoPhaseConfigError(str(exc)) from exc
    return TwoPhaseLifecycleConfig(
        state=state,
        trust=trust,
        source_sha256=source_sha256,
    )


def _load_public_verifier(
    value: object,
    *,
    index: int,
    config_root: Path,
) -> LoadedPublicVerifier:
    label = f"trust.verifiers[{index}]"
    raw = _object(value, label)
    _reject_unknown(
        label,
        raw,
        {"adapter_id", "key_id", "public_key_file", "spec_sha256", "verifier_id"},
    )
    adapter_id = _string(raw.get("adapter_id"), f"{label}.adapter_id")
    if adapter_id != ED25519_DSSE_ADAPTER_ID:
        raise TwoPhaseConfigError(f"{label}.adapter_id is unsupported.")
    public_key_path = _configured_path(
        raw.get("public_key_file"), f"{label}.public_key_file", config_root
    )
    pem = read_bounded_regular_file(
        public_key_path,
        max_bytes=_MAX_PUBLIC_KEY_BYTES,
        label=f"{label} public key",
    )
    if b"PRIVATE KEY" in pem:
        raise TwoPhaseConfigError(
            f"{label} must contain public verification material only."
        )
    try:
        public_key = load_ed25519_public_key_pem(pem)
        requirement = VerifierRequirement(
            verifier_id=require_safe_id(
                _string(raw.get("verifier_id"), f"{label}.verifier_id"),
                f"{label}.verifier_id",
            ),
            adapter_id=adapter_id,
            key_id=require_safe_id(
                _string(raw.get("key_id"), f"{label}.key_id"),
                f"{label}.key_id",
            ),
            public_key_sha256=ed25519_public_key_sha256(public_key),
            spec_sha256=require_sha256(
                _string(raw.get("spec_sha256"), f"{label}.spec_sha256"),
                f"{label}.spec_sha256",
            ),
        )
    except ValueError as exc:
        raise TwoPhaseConfigError(str(exc)) from exc
    return LoadedPublicVerifier(
        requirement=requirement,
        public_key_path=public_key_path,
        public_key_pem=pem,
        public_key_file_sha256=sha256_bytes(pem),
    )
