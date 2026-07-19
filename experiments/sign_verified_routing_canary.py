from __future__ import annotations

import argparse
import base64
from datetime import datetime
import os
from pathlib import Path
import stat
import sys
from typing import Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge import load_assistant_bridge_config
from local_moe.assistant_bridge_attestation import (
    ed25519_public_key_sha256,
    load_ed25519_public_key_pem,
)
from local_moe.assistant_bridge_integrity import (
    canonical_json_bytes,
    canonical_sha256,
)
from local_moe.route_canary import (
    AUTHORIZATION_PAYLOAD_TYPE,
    VerifiedRoutingCanaryAuthorization,
    load_verified_routing_canary_manifest,
    load_verified_routing_runtime_config,
)
from local_moe.route_policy import load_route_policy
from local_moe.route_promotion import (
    evaluate_route_promotion,
    load_evidence_plan,
    load_promotion_gate_policy,
    load_promotion_outcome_payloads,
)
from local_moe.route_scorecard import load_route_scorecard
from local_moe.verified_routing_contracts import (
    CONTRACT_VERSION,
    require_utc_timestamp,
)


_MAX_KEY_BYTES = 64 * 1024


class CanarySigningError(ValueError):
    """Raised when operator authorization cannot be produced safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-sign a bounded verified-routing canary authorization."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gate-policy", required=True)
    parser.add_argument("--training-records", required=True)
    parser.add_argument("--holdout-records", required=True)
    parser.add_argument("--assistant-bridge-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--activation-id", required=True)
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument(
        "--maximum-canary-basis-points",
        required=True,
        type=int,
        help="Operator cap in basis points; the runtime contract limits this to 500 (five percent).",
    )
    parser.add_argument("--out", required=True)
    return parser


def sign_authorization(
    *,
    manifest_path: str | Path,
    plan_path: str | Path,
    gate_policy_path: str | Path,
    training_records_path: str | Path,
    holdout_records_path: str | Path,
    assistant_bridge_config_path: str | Path,
    runtime_config_path: str | Path,
    private_key_path: str | Path,
    activation_id: str,
    issued_at: str,
    not_before: str,
    expires_at: str,
    maximum_canary_basis_points: int,
) -> tuple[VerifiedRoutingCanaryAuthorization, bytes]:
    """Build a canonical, single-signature DSSE authorization envelope."""

    bridge = load_assistant_bridge_config(assistant_bridge_config_path)
    reference = bridge.verified_routing
    if not reference.enabled or reference.config_sha256 is None:
        raise CanarySigningError(
            "Assistant bridge verified routing must be enabled before authorization."
        )

    runtime_source = Path(runtime_config_path).expanduser().resolve()
    if runtime_source != Path(reference.config_path).resolve():
        raise CanarySigningError(
            "Runtime config does not match the enabled assistant bridge reference."
        )
    runtime = load_verified_routing_runtime_config(
        runtime_source,
        expected_source_sha256=reference.config_sha256,
    )

    manifest_source = Path(manifest_path).expanduser().resolve()
    if manifest_source != Path(runtime.manifest_path).resolve():
        raise CanarySigningError(
            "Manifest does not match the enabled verified-routing runtime config."
        )
    manifest = load_verified_routing_canary_manifest(manifest_source)

    normalized_issued_at = require_utc_timestamp(issued_at, "issued_at")
    normalized_not_before = require_utc_timestamp(not_before, "not_before")
    normalized_expires_at = require_utc_timestamp(expires_at, "expires_at")
    if not (
        _timestamp(manifest.not_before)
        <= _timestamp(normalized_not_before)
        < _timestamp(normalized_expires_at)
        <= _timestamp(manifest.expires_at)
    ):
        raise CanarySigningError(
            "Authorization not-before and expiry must be inside the manifest window."
        )

    route_policy = load_route_policy(runtime.route_policy_path)
    scorecard = load_route_scorecard(
        runtime.scorecard_path,
        now=normalized_not_before,
    )
    if (
        route_policy.digest != manifest.lineage["route_policy_digest"]
        or scorecard.digest != manifest.lineage["scorecard_digest"]
        or scorecard.source_digest
        != manifest.lineage["training_source_digest"]
    ):
        raise CanarySigningError(
            "Manifest route-policy, scorecard, or training lineage does not match runtime."
        )
    if any(
        cell.config_sha256 != bridge.source_sha256
        for cell in manifest.enabled_cells
    ):
        raise CanarySigningError(
            "Manifest cells do not match the enabled assistant bridge configuration."
        )

    reconstructed_report, reconstructed_manifest = evaluate_route_promotion(
        plan=load_evidence_plan(plan_path),
        gate_policy=load_promotion_gate_policy(gate_policy_path),
        route_policy=route_policy,
        scorecard=scorecard,
        training_records=load_promotion_outcome_payloads(
            training_records_path
        ),
        holdout_records=load_promotion_outcome_payloads(
            holdout_records_path
        ),
        evaluated_at=manifest.not_before,
    )
    if (
        reconstructed_manifest is None
        or reconstructed_report.digest
        != manifest.lineage["report_sha256"]
        or reconstructed_manifest.payload() != manifest.payload()
    ):
        raise CanarySigningError(
            "Manifest was not reconstructed from eligible paired evidence."
        )

    content: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "contract": "VerifiedRoutingCanaryAuthorization",
        "activation_id": activation_id,
        "operator_key_id": runtime.operator_key_id,
        "manifest_sha256": manifest.manifest_sha256,
        "bridge_config_sha256": bridge.source_sha256,
        "route_policy_digest": route_policy.digest,
        "scorecard_digest": scorecard.digest,
        "issued_at": normalized_issued_at,
        "not_before": normalized_not_before,
        "expires_at": normalized_expires_at,
        "maximum_canary_basis_points": maximum_canary_basis_points,
    }
    authorization = VerifiedRoutingCanaryAuthorization.from_payload(
        {
            **content,
            "authorization_sha256": canonical_sha256(content),
        }
    )
    if authorization.maximum_canary_basis_points < manifest.canary_basis_points:
        raise CanarySigningError(
            "Operator cap is smaller than the manifest canary cohort."
        )
    private_key = _load_private_key(Path(private_key_path))
    signing_key_sha256 = ed25519_public_key_sha256(private_key.public_key())
    trusted_public_key = load_ed25519_public_key_pem(
        _read_bounded_regular_file(
            Path(runtime.operator_public_key_path),
            label="Operator public key",
        )
    )
    trusted_key_sha256 = ed25519_public_key_sha256(trusted_public_key)
    if trusted_key_sha256 != runtime.operator_public_key_sha256:
        raise CanarySigningError(
            "Runtime operator public key does not match its configured digest."
        )
    if signing_key_sha256 != trusted_key_sha256:
        raise CanarySigningError(
            "Private signing key does not match the trusted runtime operator key."
        )

    payload = canonical_json_bytes(authorization.payload())
    signature = private_key.sign(_dsse_pae(AUTHORIZATION_PAYLOAD_TYPE, payload))
    envelope = canonical_json_bytes(
        {
            "payloadType": AUTHORIZATION_PAYLOAD_TYPE,
            "payload": base64.b64encode(payload).decode("ascii"),
            "signatures": [
                {
                    "keyid": runtime.operator_key_id,
                    "sig": base64.b64encode(signature).decode("ascii"),
                }
            ],
        }
    )
    return authorization, envelope


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        authorization, envelope = sign_authorization(
            manifest_path=args.manifest,
            plan_path=args.plan,
            gate_policy_path=args.gate_policy,
            training_records_path=args.training_records,
            holdout_records_path=args.holdout_records,
            assistant_bridge_config_path=args.assistant_bridge_config,
            runtime_config_path=args.runtime_config,
            private_key_path=args.private_key,
            activation_id=args.activation_id,
            issued_at=args.issued_at,
            not_before=args.not_before,
            expires_at=args.expires_at,
            maximum_canary_basis_points=args.maximum_canary_basis_points,
        )
        _write_exclusive(Path(args.out), envelope)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    summary = {
        "authorization_sha256": authorization.authorization_sha256,
        "manifest_sha256": authorization.manifest_sha256,
        "operator_key_id": authorization.operator_key_id,
        "out": str(Path(args.out)),
    }
    sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
    return 0


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    value = _read_bounded_regular_file(path, label="Private signing key")
    try:
        key = serialization.load_pem_private_key(value, password=None)
    except (TypeError, ValueError) as exc:
        raise CanarySigningError(
            "Private signing key must be an unencrypted Ed25519 PEM."
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise CanarySigningError("Private signing key must be Ed25519.")
    return key


def _read_bounded_regular_file(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CanarySigningError(f"{label} is unavailable.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_KEY_BYTES:
            raise CanarySigningError(f"{label} must be a bounded regular file.")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise CanarySigningError(f"{label} changed while it was read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CanarySigningError(f"{label} changed while it was read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise CanarySigningError("Authorization output already exists.") from exc
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written <= 0:
                raise OSError("Authorization output write failed.")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


if __name__ == "__main__":
    raise SystemExit(main())
