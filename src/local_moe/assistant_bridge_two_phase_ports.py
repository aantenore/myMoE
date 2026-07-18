from __future__ import annotations

from pathlib import Path
from typing import Any, ContextManager, Mapping, Protocol, Sequence

from .assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    CandidateBinding,
    IndependentAttestation,
)


class EvidenceStore(Protocol):
    """Trusted application port for immutable evidence artifacts."""

    def put_bytes(
        self, value: bytes, *, media_type: str
    ) -> ArtifactDescriptor: ...

    def get_bytes(self, descriptor: ArtifactDescriptor) -> bytes: ...


class CandidateStore(Protocol):
    """Trusted application port for staged candidate artifacts."""

    def store_candidate(
        self,
        candidate_root: str | Path,
        candidate_files: Sequence[Mapping[str, Any]],
        changes: Sequence[Mapping[str, Any]],
        *,
        source_fingerprint: str,
        source_identity: Mapping[str, Any],
    ) -> tuple[ArtifactDescriptor, ArtifactDescriptor]: ...

    def load_candidate(
        self,
        manifest_descriptor: ArtifactDescriptor,
        changeset_descriptor: ArtifactDescriptor,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def validate_candidate_closure(
        self,
        manifest_descriptor: ArtifactDescriptor,
        changeset_descriptor: ArtifactDescriptor,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def materialize_candidate(
        self, manifest_descriptor: ArtifactDescriptor
    ) -> ContextManager[Path]: ...


class AttestationVerifier(Protocol):
    """Trusted application port that verifies raw independent evidence."""

    def verify(
        self,
        binding: CandidateBinding,
        envelope: bytes,
        *,
        now: float,
    ) -> IndependentAttestation: ...
