"""Empirical compatibility probe for a configured local Codex provider.

The Assistant Bridge configuration declares policy ceilings.  This module
checks a narrower runtime fact: can the configured local model actually use
the Codex tool protocol to read a file from a disposable workspace?  A passing
report is diagnostic evidence only; it never grants routing or write authority.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .assistant_bridge import (
    AssistantBridgeError,
    CapabilityDemand,
    ProviderSpec,
    _write_capsule_atomic,
    default_provider_adapter_registry,
    load_assistant_bridge_config,
)
from .assistant_bridge_provider_registry import ProviderAdapterRegistry
from .assistant_bridge_two_phase_state import (
    TwoPhaseConfigError,
    read_bounded_regular_file,
)


EXIT_COMPATIBLE = 0
EXIT_INCOMPATIBLE = 1
EXIT_CONTRACT = 2
EXIT_OPERATIONAL = 3

SCHEMA_VERSION = "assistant-provider-compatibility/v1"
PROBE_CONTRACT = "codex-disposable-workspace-read/v1"
_MARKER_FILENAME = "MYMOE_PROBE_MARKER.txt"
_EXPECTED_PREFIX = "MYMOE_TOOL_OK:"
_MAX_TIMEOUT_SECONDS = 300.0
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_OLLAMA_TAGS_BYTES = 1024 * 1024
_SAFE_RESULT_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_RAW_CONTENT_FIELDS = frozenset(
    {
        "content",
        "messages",
        "output",
        "prompt",
        "raw",
        "raw_output",
        "raw_prompt",
        "response",
        "stderr",
        "stdin_text",
        "stdout",
    }
)
_RESULT_METADATA_FIELDS = frozenset(
    {
        "provider_id",
        "status",
        "code",
        "returncode",
        "duration_ms",
        "output_sha256",
        "output_chars",
        "stdout_sha256",
        "stdout_bytes",
        "stderr_sha256",
        "stderr_bytes",
        "command_sha256",
        "usage",
    }
)
_RESULT_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "cost_status",
    }
)
_PLAN_METADATA_FIELDS = frozenset(
    {
        "provider_id",
        "adapter_id",
        "mode",
        "argv_sha256",
        "argv_shape",
        "stdin",
        "workspace_sha256",
        "output_path_sha256",
        "command_sha256",
        "sandbox",
        "permission_profile",
        "permission_profile_effective_attested",
        "permission_workspace_rule",
        "network_access",
        "shell_network_access",
        "web_search_mode",
        "workspace_access",
        "model",
        "local_provider",
        "environment_keys",
        "ephemeral_environment_keys",
        "executable",
        "environment_sha256",
        "runtime",
        "runtime_policy",
        "launcher_chain",
        "launcher_authority_sha256",
        "launcher_artifact_sha256",
    }
)
_PLAN_STDIN_FIELDS = frozenset(
    {"transport", "sha256", "characters", "content_in_argv"}
)


class AssistantProviderProbeError(RuntimeError):
    """Raised when the probe cannot produce a trustworthy diagnostic result."""


class AssistantProviderProbeOperationalError(AssistantProviderProbeError):
    """Raised when runtime conditions prevent a compatibility conclusion."""


def run_local_provider_probe(
    bridge_config_path: str | Path,
    *,
    timeout_seconds: float = 45.0,
    adapter_registry: ProviderAdapterRegistry | None = None,
    now: Callable[[], datetime] | None = None,
    marker_factory: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Run one bounded, read-only tool-protocol check against the local model.

    The random marker exists only inside a disposable workspace and is never
    included in the returned report.  Recovering it proves that the model used
    the workspace boundary; a conversational guess cannot pass the check.
    """

    selected_timeout = _timeout(timeout_seconds)
    declared_config = _read_declared_config(bridge_config_path)
    config = load_assistant_bridge_config(bridge_config_path)
    if _read_declared_config(bridge_config_path) != declared_config:
        raise AssistantProviderProbeError(
            "The Assistant Bridge configuration changed while it was loaded."
        )
    declared_config_sha256 = _sha256_bytes(declared_config)
    configured_provider = config.local
    provider = replace(
        configured_provider,
        capabilities=("filesystem",),
        tools=("shell",),
        max_risk="read_only",
        sandbox="read-only",
        workspace_access="read_only",
        timeout_seconds=selected_timeout,
    )
    registry = adapter_registry or default_provider_adapter_registry()
    try:
        adapter = registry.require(provider.adapter)
    except Exception as exc:
        raise AssistantProviderProbeError(
            "The configured local provider adapter is unavailable."
        ) from exc

    marker = (marker_factory or _new_marker)()
    if not _valid_marker(marker):
        raise AssistantProviderProbeError("The probe marker contract is invalid.")

    prompt = _probe_prompt()
    demand = CapabilityDemand(
        required=("filesystem",),
        tools=("shell",),
        risk_class="read_only",
    )
    model_before = _capture_model_identity(provider, selected_timeout)
    try:
        with tempfile.TemporaryDirectory(prefix="mymoe-provider-probe-") as temporary:
            workspace = Path(temporary)
            _write_private_marker(workspace / _MARKER_FILENAME, marker)
            output_path = workspace / "final.txt"
            try:
                plan = adapter.build_command_plan(
                    provider,
                    prompt=prompt,
                    workspace=workspace,
                    demand=demand,
                    output_path=output_path,
                    workspace_access="read_only",
                    runtime_policy=config.runtime,
                    ephemeral_workspace=True,
                )
                plan_payload = _public_plan_payload(
                    plan,
                    provider=provider,
                    prompt=prompt,
                    workspace=workspace,
                    output_path=output_path,
                )
            except (AssistantBridgeError, ValueError) as exc:
                raise AssistantProviderProbeError(
                    "The local provider cannot materialize the probe contract."
                ) from exc
            try:
                result = adapter.execute_command(
                    provider,
                    plan,
                    prompt=prompt,
                    output_path=output_path,
                )
            except (AssistantBridgeError, ValueError) as exc:
                raise AssistantProviderProbeError(
                    "The local provider violated the probe execution contract."
                ) from exc
            except Exception as exc:
                raise AssistantProviderProbeOperationalError(
                    "The local provider failed before returning bounded metadata."
                ) from exc
    except AssistantProviderProbeError:
        raise
    except OSError as exc:
        raise AssistantProviderProbeOperationalError(
            "The disposable probe workspace was unavailable."
        ) from exc
    except Exception as exc:
        raise AssistantProviderProbeOperationalError(
            "The local provider probe failed before returning bounded metadata."
        ) from exc

    model_after = _capture_model_identity(provider, selected_timeout)
    model_identity = _reconcile_model_identity(model_before, model_after)
    if getattr(result, "command_sha256", None) != plan.command_sha256:
        raise AssistantProviderProbeError(
            "The provider result does not match the inspected command plan."
        )
    if getattr(result, "provider_id", None) != provider.id:
        raise AssistantProviderProbeError(
            "The provider result does not match the inspected provider."
        )
    try:
        result_payload = _validated_result_metadata_payload(result)
    except Exception as exc:
        raise AssistantProviderProbeError(
            "The provider returned an invalid result contract."
        ) from exc

    expected = f"{_EXPECTED_PREFIX}{marker}"
    recovered = result.status == "completed" and result.output.strip() == expected
    if recovered:
        reason_codes = ["workspace_marker_recovered"]
        observed = ["codex_tool_protocol", "filesystem_read"]
        status = "compatible"
    elif result.status == "completed":
        reason_codes = ["workspace_marker_not_recovered"]
        observed = []
        status = "incompatible"
    else:
        reason_codes = [result.code]
        observed = []
        status = "indeterminate"

    checked_at = (now or _utc_now)().astimezone(timezone.utc)
    contract_sha256 = _sha256_text(
        _canonical_json(
            {
                "contract": PROBE_CONTRACT,
                "marker_filename": _MARKER_FILENAME,
                "expected_prefix": _EXPECTED_PREFIX,
                "prompt": prompt,
                "demand": demand.payload(),
            }
        )
    )
    probe_id = _sha256_text(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "provider_id": provider.id,
                "adapter": provider.adapter,
                "model": provider.model,
                "local_provider": provider.local_provider,
                "bridge_config_declared_bytes_sha256": declared_config_sha256,
                "bridge_config_effective_sha256": config.source_sha256,
                "contract_sha256": contract_sha256,
                "command_sha256": plan.command_sha256,
                "model_identity": model_identity,
                "checked_at": checked_at.isoformat(),
            }
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "checked_at": checked_at.isoformat(),
        "status": status,
        "diagnostic_only": True,
        "authorizes_routing": False,
        "reason_codes": reason_codes,
        "observed_capabilities": observed,
        "provider": {
            "id": configured_provider.id,
            "mode": configured_provider.mode,
            "adapter": configured_provider.adapter,
            "execution_scope": configured_provider.execution_scope,
            "local_provider": configured_provider.local_provider,
            "model": configured_provider.model,
            "declared_capabilities": list(configured_provider.capabilities),
            "declared_tools": list(configured_provider.tools),
            "declared_max_risk": configured_provider.max_risk,
            "declared_sandbox": configured_provider.sandbox,
            "declared_workspace_access": configured_provider.workspace_access,
        },
        "probe_authority": {
            "capabilities": list(provider.capabilities),
            "tools": list(provider.tools),
            "max_risk": provider.max_risk,
            "sandbox": provider.sandbox,
            "workspace_access": provider.workspace_access,
            "ephemeral": True,
        },
        "binding": {
            "bridge_config_declared_bytes_sha256": declared_config_sha256,
            "bridge_config_effective_sha256": config.source_sha256,
            "probe_contract": PROBE_CONTRACT,
            "probe_contract_sha256": contract_sha256,
            "command_sha256": plan.command_sha256,
        },
        "execution_identity": {
            "command_plan": plan_payload,
            "model": model_identity,
        },
        "result": result_payload,
    }
    _validate_metadata_only_fields(report)
    return report


def write_probe_report(path: str | Path, report: Mapping[str, object]) -> Path:
    """Durably replace a report without following target or ancestor links."""

    requested = Path(path).expanduser()
    target = Path(os.path.abspath(os.fspath(requested)))
    payload = (
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _write_capsule_atomic(target, payload)
    except AssistantBridgeError as exc:
        raise AssistantProviderProbeOperationalError(
            "Probe report could not be persisted through the hardened writer."
        ) from exc
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mymoe assistant-probe",
        description=(
            "Check whether the configured local model can use the Codex tool "
            "protocol in a disposable read-only workspace."
        ),
    )
    parser.add_argument(
        "--bridge-config",
        default="configs/assistant-bridge.json",
        help="Assistant Bridge configuration to inspect.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Bounded local-provider timeout (1-300 seconds).",
    )
    parser.add_argument(
        "--out",
        help="Optional metadata-only JSON report path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete metadata-only report as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        report = run_local_provider_probe(
            args.bridge_config,
            timeout_seconds=args.timeout_seconds,
        )
        if args.out:
            write_probe_report(args.out, report)
    except (AssistantProviderProbeOperationalError, OSError):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": {
                "code": "probe_operational_failure",
                "message": "The local provider probe could not reach a conclusion safely.",
            },
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return EXIT_OPERATIONAL
    except (AssistantBridgeError, AssistantProviderProbeError, ValueError):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": {
                "code": "probe_contract_or_runtime_invalid",
                "message": "The local provider probe could not run safely.",
            },
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return EXIT_CONTRACT

    if args.json:
        print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    else:
        provider = report["provider"]
        assert isinstance(provider, dict)
        reason_codes = report["reason_codes"]
        assert isinstance(reason_codes, list) and reason_codes
        print(
            "Local provider {provider} ({model}): {status} ({reason}). "
            "Use --json for evidence.".format(
                provider=provider["id"],
                model=provider["model"],
                status=report["status"],
                reason=reason_codes[0],
            )
        )
    if report["status"] == "compatible":
        return EXIT_COMPATIBLE
    if report["status"] == "incompatible":
        return EXIT_INCOMPATIBLE
    return EXIT_OPERATIONAL


def _probe_prompt() -> str:
    return (
        "This is a bounded compatibility check in a disposable read-only "
        f"workspace. Use the available shell tool to read {_MARKER_FILENAME}. "
        f"Return exactly one line in the form {_EXPECTED_PREFIX}<file contents>. "
        "Do not guess, ask questions, explain, edit files, or call any other tool."
    )


def _new_marker() -> str:
    return "mymoe-" + secrets.token_hex(16)


def _valid_marker(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("mymoe-")
        and len(value) == 38
        and all(character in "0123456789abcdef" for character in value[6:])
    )


def _write_private_marker(path: Path, marker: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        content = memoryview((marker + "\n").encode("ascii"))
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("marker write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssistantProviderProbeError("Probe timeout must be numeric.")
    selected = float(value)
    if not 1.0 <= selected <= _MAX_TIMEOUT_SECONDS:
        raise AssistantProviderProbeError("Probe timeout must be between 1 and 300 seconds.")
    return selected


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_declared_config(path: str | Path) -> bytes:
    try:
        return read_bounded_regular_file(
            path,
            max_bytes=_MAX_CONFIG_BYTES,
            label="Assistant Bridge configuration",
        )
    except TwoPhaseConfigError as exc:
        raise AssistantProviderProbeError(str(exc)) from exc


def _public_plan_payload(
    plan: Any,
    *,
    provider: ProviderSpec,
    prompt: str,
    workspace: Path,
    output_path: Path,
) -> dict[str, object]:
    payload_method = getattr(plan, "payload", None)
    if not callable(payload_method):
        raise AssistantProviderProbeError(
            "The provider command plan has no public identity payload."
        )
    payload = payload_method()
    if not isinstance(payload, Mapping):
        raise AssistantProviderProbeError(
            "The provider command plan public identity is invalid."
        )
    try:
        encoded = _canonical_json(payload)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssistantProviderProbeError(
            "The provider command plan public identity is not canonical JSON."
        ) from exc
    if not isinstance(normalized, dict):
        raise AssistantProviderProbeError(
            "The provider command plan public identity must be an object."
        )
    if set(normalized) != _PLAN_METADATA_FIELDS:
        raise AssistantProviderProbeError(
            "The provider command plan public identity schema is invalid."
        )
    command_sha256 = normalized.get("command_sha256")
    if (
        command_sha256 != getattr(plan, "command_sha256", None)
        or not _is_sha256(command_sha256)
    ):
        raise AssistantProviderProbeError(
            "The provider command plan digest identity is invalid."
        )
    adapter_id = normalized.get("adapter_id")
    if (
        not isinstance(adapter_id, str)
        or _SAFE_RESULT_CODE.fullmatch(adapter_id) is None
        or adapter_id != provider.adapter
    ):
        raise AssistantProviderProbeError(
            "The provider command plan adapter identity is invalid."
        )
    stdin = normalized.get("stdin")
    if (
        not isinstance(stdin, dict)
        or set(stdin) != _PLAN_STDIN_FIELDS
        or stdin.get("transport") != "stdin"
        or stdin.get("sha256") != _sha256_text(prompt)
        or stdin.get("characters") != len(prompt)
        or stdin.get("content_in_argv") is not False
    ):
        raise AssistantProviderProbeError(
            "The provider command plan stdin metadata is invalid."
        )
    expected_scalars = {
        "provider_id": provider.id,
        "mode": provider.mode,
        "sandbox": "read-only",
        "permission_profile": "mymoe_workspace_read",
        "permission_profile_effective_attested": False,
        "permission_workspace_rule": "read",
        "network_access": False,
        "shell_network_access": False,
        "web_search_mode": "disabled",
        "workspace_access": "read_only",
        "model": provider.model,
        "local_provider": provider.local_provider,
        "environment_keys": list(provider.environment_allowlist),
        "ephemeral_environment_keys": ["CODEX_HOME", "HOME"],
        "workspace_sha256": _sha256_text(str(workspace.resolve())),
        "output_path_sha256": _sha256_text(str(output_path.resolve())),
    }
    if any(normalized.get(key) != value for key, value in expected_scalars.items()):
        raise AssistantProviderProbeError(
            "The provider command plan authority metadata is inconsistent."
        )
    for key in (
        "argv_sha256",
        "environment_sha256",
        "launcher_authority_sha256",
    ):
        if not _is_sha256(normalized.get(key)):
            raise AssistantProviderProbeError(
                "The provider command plan digest metadata is invalid."
            )
    argv_shape = normalized.get("argv_shape")
    if not isinstance(argv_shape, list) or any(
        not isinstance(item, str) for item in argv_shape
    ):
        raise AssistantProviderProbeError(
            "The provider command plan argv shape is invalid."
        )
    executable = normalized.get("executable")
    runtime = normalized.get("runtime")
    runtime_policy = normalized.get("runtime_policy")
    launcher_chain = normalized.get("launcher_chain")
    if (
        not isinstance(executable, dict)
        or not _is_sha256(executable.get("sha256"))
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("schema_version"), str)
        or not isinstance(runtime.get("strict_tree_supported"), bool)
        or not isinstance(runtime_policy, dict)
        or runtime_policy.get("require_tree_isolation") is not True
        or not isinstance(launcher_chain, dict)
        or not _is_sha256(launcher_chain.get("fingerprint"))
        or not isinstance(launcher_chain.get("schema_version"), str)
        or launcher_chain.get("strict") is not True
    ):
        raise AssistantProviderProbeError(
            "The provider command plan runtime identity is incomplete."
        )
    launcher_artifacts = normalized.get("launcher_artifact_sha256")
    if not isinstance(launcher_artifacts, list) or any(
        not _is_sha256(item) for item in launcher_artifacts
    ):
        raise AssistantProviderProbeError(
            "The provider command plan launcher artifact identity is invalid."
        )
    _validate_metadata_only_fields(normalized)
    return normalized


def _validated_result_metadata_payload(result: Any) -> dict[str, object]:
    """Validate the result's fixed metadata schema without inspecting raw text.

    Raw output can be arbitrarily short, so substring checks against serialized
    JSON are not a sound exclusion mechanism.  The report instead admits only
    the fixed metadata fields below and binds output evidence to its digest and
    character count.
    """

    payload_method = getattr(result, "metadata_payload", None)
    if not callable(payload_method):
        raise AssistantProviderProbeError(
            "The provider result has no metadata-only identity payload."
        )
    try:
        normalized = json.loads(_canonical_json(payload_method()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssistantProviderProbeError(
            "The provider result metadata is not canonical JSON."
        ) from exc
    if not isinstance(normalized, dict) or set(normalized) != _RESULT_METADATA_FIELDS:
        raise AssistantProviderProbeError(
            "The provider result metadata schema is invalid."
        )
    usage = normalized.get("usage")
    if not isinstance(usage, dict) or set(usage) != _RESULT_USAGE_FIELDS:
        raise AssistantProviderProbeError(
            "The provider result usage metadata schema is invalid."
        )

    output = getattr(result, "output", None)
    if not isinstance(output, str):
        raise AssistantProviderProbeError("The provider result output is invalid.")
    expected_output_sha256 = _sha256_text(output) if output else None
    if (
        normalized.get("output_sha256") != expected_output_sha256
        or normalized.get("output_chars") != len(output)
    ):
        raise AssistantProviderProbeError(
            "The provider result output metadata does not match the raw result."
        )
    if normalized.get("provider_id") != getattr(result, "provider_id", None):
        raise AssistantProviderProbeError(
            "The provider result identity metadata is inconsistent."
        )
    if normalized.get("status") != getattr(result, "status", None) or normalized.get(
        "status"
    ) not in {"completed", "failed", "blocked"}:
        raise AssistantProviderProbeError(
            "The provider result status metadata is invalid."
        )
    code = normalized.get("code")
    if (
        code != getattr(result, "code", None)
        or not isinstance(code, str)
        or _SAFE_RESULT_CODE.fullmatch(code) is None
    ):
        raise AssistantProviderProbeError(
            "The provider result code metadata is invalid."
        )
    for field in ("returncode", "duration_ms", "stdout_bytes", "stderr_bytes"):
        expected = getattr(result, field, None)
        observed = normalized.get(field)
        if observed != expected or (
            observed is not None
            and (isinstance(observed, bool) or not isinstance(observed, int))
        ) or (field != "returncode" and observed is None):
            raise AssistantProviderProbeError(
                f"The provider result {field} metadata is invalid."
            )
    if normalized["duration_ms"] < 0:
        raise AssistantProviderProbeError(
            "The provider result duration metadata is invalid."
        )
    if normalized["stdout_bytes"] < 0 or normalized["stderr_bytes"] < 0:
        raise AssistantProviderProbeError(
            "The provider result stream-size metadata is invalid."
        )
    if normalized["status"] == "completed" and (
        normalized["code"] != "launcher_completed"
        or normalized["returncode"] != 0
    ):
        raise AssistantProviderProbeError(
            "The provider result completion metadata is contradictory."
        )
    for field in ("stdout_sha256", "stderr_sha256", "command_sha256"):
        expected = getattr(result, field, "") or None
        observed = normalized.get(field)
        if observed != expected or (
            observed is not None
            and (
                not _is_sha256(observed)
            )
        ):
            raise AssistantProviderProbeError(
                f"The provider result {field} metadata is invalid."
            )
    expected_usage = {
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "completion_tokens": getattr(result, "completion_tokens", None),
        "cost": None,
        "cost_status": "not_computed_without_pricing_contract",
    }
    if usage != expected_usage:
        raise AssistantProviderProbeError(
            "The provider result usage metadata is inconsistent."
        )
    for field in ("prompt_tokens", "completion_tokens"):
        value = usage[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise AssistantProviderProbeError(
                f"The provider result {field} metadata is invalid."
            )
    _validate_metadata_only_fields(normalized)
    return normalized


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_metadata_only_fields(value: object, *, path: str = "report") -> None:
    """Reject content-bearing field names anywhere in a public metadata tree."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise AssistantProviderProbeError(
                    "The metadata-only report contains a non-string field name."
                )
            if key.casefold() in _RAW_CONTENT_FIELDS:
                raise AssistantProviderProbeError(
                    f"The metadata-only report contains content field {path}.{key}."
                )
            _validate_metadata_only_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_metadata_only_fields(nested, path=f"{path}[{index}]")


def _capture_model_identity(
    provider: ProviderSpec,
    timeout_seconds: float,
) -> dict[str, object]:
    if provider.local_provider != "ollama":
        return _unverified_model_identity(
            provider.model,
            "local_provider_identity_not_supported",
        )
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        11434,
        timeout=max(0.25, min(1.0, timeout_seconds / 4.0)),
    )
    try:
        connection.request("GET", "/api/tags", headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            return _unverified_model_identity(
                provider.model,
                "ollama_tags_unavailable",
            )
        payload = response.read(_MAX_OLLAMA_TAGS_BYTES + 1)
        if len(payload) > _MAX_OLLAMA_TAGS_BYTES:
            return _unverified_model_identity(
                provider.model,
                "ollama_tags_response_too_large",
            )
    except (OSError, http.client.HTTPException):
        return _unverified_model_identity(
            provider.model,
            "ollama_tags_unavailable",
        )
    finally:
        connection.close()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _unverified_model_identity(
            provider.model,
            "ollama_tags_invalid",
        )
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        return _unverified_model_identity(
            provider.model,
            "ollama_tags_invalid",
        )
    matches: list[tuple[str, int]] = []
    for item in document["models"]:
        if not isinstance(item, dict):
            continue
        references = (item.get("name"), item.get("model"))
        if provider.model not in references:
            continue
        digest = item.get("digest")
        size = item.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            return _unverified_model_identity(
                provider.model,
                "ollama_model_identity_invalid",
            )
        matches.append((digest.lower(), size))
    if len(set(matches)) != 1:
        return _unverified_model_identity(
            provider.model,
            "ollama_model_reference_missing_or_ambiguous",
        )
    digest, size = matches[0]
    return {
        "status": "content_addressed",
        "reference": provider.model,
        "digest": f"sha256:{digest}",
        "size_bytes": size,
        "source": "ollama_loopback_tags_api",
    }


def _unverified_model_identity(reference: str, reason_code: str) -> dict[str, object]:
    return {
        "status": "mutable_reference_unverified",
        "reference": reference,
        "reason_code": reason_code,
    }


def _reconcile_model_identity(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    if (
        before.get("status") == "content_addressed"
        and after.get("status") == "content_addressed"
        and before.get("digest") == after.get("digest")
        and before.get("size_bytes") == after.get("size_bytes")
    ):
        identity = dict(after)
        identity["stable_during_probe"] = True
        return identity
    reference = str(after.get("reference") or before.get("reference") or "unknown")
    reason = "model_identity_changed_during_probe"
    if before.get("status") != "content_addressed" or after.get(
        "status"
    ) != "content_addressed":
        reason = "model_identity_unverified_during_probe"
    return _unverified_model_identity(reference, reason)


if __name__ == "__main__":
    raise SystemExit(main())
