from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from ipaddress import ip_address
from typing import Protocol
from urllib.parse import urlparse


SCOPE_BLOCKED = "scope_blocked"


class ExecutionScope(str, Enum):
    DEVICE_ONLY = "device_only"
    PRIVATE_MESH = "private_mesh"
    PUBLIC_MESH = "public_mesh"
    PAID_REMOTE = "paid_remote"


class ExecutionTransport(str, Enum):
    DIRECT_LOCAL = "direct_local"
    MESH_LLM = "mesh_llm"
    GATEWAY = "gateway"


_SCOPE_ORDER = {
    ExecutionScope.DEVICE_ONLY: 0,
    ExecutionScope.PRIVATE_MESH: 1,
    ExecutionScope.PUBLIC_MESH: 2,
    ExecutionScope.PAID_REMOTE: 3,
}


class ScopePolicyError(RuntimeError):
    """Raised when an execution target cannot satisfy the active scope policy."""

    reason_code = SCOPE_BLOCKED

    def __init__(self, detail: str, *, expert_id: str | None = None):
        self.detail = detail
        self.expert_id = expert_id
        prefix = f"{SCOPE_BLOCKED}:"
        if expert_id:
            prefix = f"{prefix} expert {expert_id!r}"
        super().__init__(f"{prefix} {detail}")


@dataclass(frozen=True)
class ExecutionDeclaration:
    """Configuration claim. Claims for non-local transports still need attestation."""

    scope: ExecutionScope | None = None
    transport: ExecutionTransport | None = None


@dataclass(frozen=True)
class ExecutionPolicy:
    max_scope: ExecutionScope = ExecutionScope.DEVICE_ONLY
    allowed_scopes: tuple[ExecutionScope, ...] = (ExecutionScope.DEVICE_ONLY,)
    allow_scope_widening: bool = False

    def __post_init__(self) -> None:
        if len(set(self.allowed_scopes)) != len(self.allowed_scopes):
            raise ValueError("execution.allowed_scopes must not contain duplicates.")
        above_maximum = [
            scope
            for scope in self.allowed_scopes
            if scope_rank(scope) > scope_rank(self.max_scope)
        ]
        if above_maximum:
            rendered = ", ".join(scope.value for scope in above_maximum)
            raise ValueError(
                "execution.allowed_scopes cannot exceed execution.max_scope: "
                f"{rendered}."
            )

    def allows(self, scope: ExecutionScope) -> bool:
        return (
            scope in self.allowed_scopes
            and scope_rank(scope) <= scope_rank(self.max_scope)
        )


@dataclass(frozen=True)
class ExecutionTarget:
    expert_id: str
    provider: str
    endpoint: str | None
    declaration: ExecutionDeclaration


@dataclass(frozen=True)
class ExecutionAttestation:
    expert_id: str
    scope: ExecutionScope
    transport: ExecutionTransport
    authority: str


@dataclass(frozen=True)
class ExecutionEligibility:
    expert_id: str
    allowed: bool
    scope: ExecutionScope | None = None
    transport: ExecutionTransport | None = None
    reason_code: str | None = None
    detail: str = ""


class ExecutionAttestor(Protocol):
    """Provider-neutral boundary for fresh transport and scope evidence."""

    def attest(self, target: ExecutionTarget) -> ExecutionAttestation:
        ...


class ConfiguredDirectLocalAttestor:
    """Enforce the configured direct-local boundary without claiming compute proof.

    Loopback establishes only the first network hop. Operators that place a mesh or
    proxy behind loopback must declare that transport and provide a fresh attestor.
    """

    def attest(self, target: ExecutionTarget) -> ExecutionAttestation:
        declaration = normalized_execution_declaration(
            provider=target.provider,
            endpoint=target.endpoint,
            declaration=target.declaration,
        )
        transport = declaration.transport
        if transport != ExecutionTransport.DIRECT_LOCAL:
            raise ScopePolicyError(
                f"transport {transport.value!r} requires an external attestor.",
                expert_id=target.expert_id,
            )

        if target.provider != "synthetic" and not is_loopback_endpoint(target.endpoint):
            raise ScopePolicyError(
                "direct_local requires a loopback HTTP endpoint.",
                expert_id=target.expert_id,
            )

        if declaration.scope not in {None, ExecutionScope.DEVICE_ONLY}:
            raise ScopePolicyError(
                "direct_local evidence conflicts with the declared scope.",
                expert_id=target.expert_id,
            )

        return ExecutionAttestation(
            expert_id=target.expert_id,
            scope=ExecutionScope.DEVICE_ONLY,
            transport=ExecutionTransport.DIRECT_LOCAL,
            authority="configured_direct_local",
        )


class ExecutionScopeGuard:
    """Resolve fresh evidence and enforce one execution policy."""

    def __init__(
        self,
        policy: ExecutionPolicy | None = None,
        *,
        attestor: ExecutionAttestor | None = None,
    ):
        self.policy = policy or ExecutionPolicy()
        self._attestor = attestor or ConfiguredDirectLocalAttestor()

    def evaluate(self, target: ExecutionTarget) -> ExecutionEligibility:
        declaration = normalized_execution_declaration(
            provider=target.provider,
            endpoint=target.endpoint,
            declaration=target.declaration,
        )
        normalized_target = ExecutionTarget(
            expert_id=target.expert_id,
            provider=target.provider,
            endpoint=target.endpoint,
            declaration=declaration,
        )
        try:
            attestation = self._attestor.attest(normalized_target)
            self._validate_attestation(normalized_target, attestation)
        except ScopePolicyError as exc:
            return ExecutionEligibility(
                expert_id=target.expert_id,
                allowed=False,
                reason_code=exc.reason_code,
                detail=exc.detail,
            )

        if not self.policy.allows(attestation.scope):
            return ExecutionEligibility(
                expert_id=target.expert_id,
                allowed=False,
                scope=attestation.scope,
                transport=attestation.transport,
                reason_code=SCOPE_BLOCKED,
                detail=(
                    f"scope {attestation.scope.value!r} is outside the active policy."
                ),
            )

        return ExecutionEligibility(
            expert_id=target.expert_id,
            allowed=True,
            scope=attestation.scope,
            transport=attestation.transport,
        )

    def require_allowed(self, target: ExecutionTarget) -> ExecutionAttestation:
        eligibility = self.evaluate(target)
        if not eligibility.allowed:
            raise ScopePolicyError(
                eligibility.detail or "execution target is not eligible.",
                expert_id=target.expert_id,
            )
        if eligibility.scope is None or eligibility.transport is None:
            raise ScopePolicyError(
                "execution attestation is incomplete.",
                expert_id=target.expert_id,
            )
        return ExecutionAttestation(
            expert_id=target.expert_id,
            scope=eligibility.scope,
            transport=eligibility.transport,
            authority="guard",
        )

    def permits_fallback(
        self,
        *,
        selected_scopes: tuple[ExecutionScope, ...],
        fallback_scope: ExecutionScope,
    ) -> bool:
        if self.policy.allow_scope_widening or not selected_scopes:
            return True
        route_ceiling = max(scope_rank(scope) for scope in selected_scopes)
        return scope_rank(fallback_scope) <= route_ceiling

    @staticmethod
    def _validate_attestation(
        target: ExecutionTarget,
        attestation: ExecutionAttestation,
    ) -> None:
        if attestation.expert_id != target.expert_id:
            raise ScopePolicyError(
                "attestation identifies a different expert.",
                expert_id=target.expert_id,
            )
        transport = target.declaration.transport
        if transport is None or attestation.transport != transport:
            raise ScopePolicyError(
                "attested transport does not match the configured transport.",
                expert_id=target.expert_id,
            )
        declared_scope = target.declaration.scope
        if declared_scope is None:
            raise ScopePolicyError(
                "non-local execution requires an explicit scope declaration.",
                expert_id=target.expert_id,
            )
        if attestation.scope != declared_scope:
            raise ScopePolicyError(
                "attested scope does not match the configured scope.",
                expert_id=target.expert_id,
            )


def normalized_execution_declaration(
    *,
    provider: str,
    endpoint: str | None,
    declaration: ExecutionDeclaration | None = None,
) -> ExecutionDeclaration:
    configured = declaration or ExecutionDeclaration()
    transport = configured.transport
    if transport is None:
        if provider == "synthetic" or is_loopback_endpoint(endpoint):
            transport = ExecutionTransport.DIRECT_LOCAL
        else:
            transport = ExecutionTransport.GATEWAY

    scope = configured.scope
    if scope is None and (
        provider == "synthetic"
        or (
            transport == ExecutionTransport.DIRECT_LOCAL
            and is_loopback_endpoint(endpoint)
        )
    ):
        scope = ExecutionScope.DEVICE_ONLY

    return ExecutionDeclaration(scope=scope, transport=transport)


def is_loopback_endpoint(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    parsed = urlparse(str(endpoint).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def scope_rank(scope: ExecutionScope) -> int:
    return _SCOPE_ORDER[scope]
