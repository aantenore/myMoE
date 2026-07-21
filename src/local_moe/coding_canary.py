"""Fail-closed qualification canary for local Cline coding cells.

The canary deliberately proves one small capability: a pinned Cline/model/
gateway configuration can read two disposable files, edit exactly one source
file, and run one allowlisted test command.  It never grants routing authority.
Browser, MCP, desktop control, Git publication, and real project workspaces are
outside this contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .assistant_bridge import AssistantBridgeError, _write_capsule_atomic
from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    ProcessExecutionPolicy,
    ProcessExecutionResult,
    execute_process,
    resolve_executable,
)
from .assistant_bridge_two_phase_state import (
    TwoPhaseConfigError,
    read_bounded_regular_file,
)
from .assistant_bridge_verifier_isolation import (
    VerifierIsolationError,
    VerifierIsolationPolicy,
    build_verifier_isolation_plan,
    verifier_isolation_capability,
)
from .assistant_bridge_workspace import (
    WorkspaceFile,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    snapshot_materialized,
)
from .config import ConfigError, MoEConfig, parse_config, runtime_config_sha256
from .execution_scope import ExecutionScopeGuard, ScopePolicyError
from .hardware import detect_hardware


EXIT_QUALIFIED = 0
EXIT_INCOMPATIBLE = 1
EXIT_CONTRACT = 2
EXIT_INDETERMINATE = 3

SCHEMA_VERSION = "local-coding-canary/v1"
CANARY_CONTRACT = "cline-single-file-edit/v1"
SUPPORTED_CLINE_VERSION = "3.0.46"
FIXTURE_VERSION = "python-addition/v1"

_SOURCE_NAME = "calculator.py"
_TEST_NAME = "test_calculator.py"
_ALLOWED_COMMAND = "python3 -m unittest -q test_calculator.py"
_VERIFIER_SCRIPT = (
    "import os,runpy,sys;"
    "sys.path.insert(0,os.getcwd());"
    "runpy.run_path('test_calculator.py',run_name='__main__')"
)
_BROKER_REQUEST_LIMIT = 32
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_APPROVAL_BYTES = 64 * 1024
_MAX_PROXY_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_PROXY_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_PROXY_CONTENT_TYPE_CHARS = 1_024
_MAX_NDJSON_BYTES = 8 * 1024 * 1024
_AI_SDK_WARNING_LINE = (
    b"AI SDK Warning System: To turn off warning logging, set the "
    b"AI_SDK_LOG_WARNINGS global to false."
)
_SAFE_REASON = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_CLINE_VERSION = re.compile(r"(?:cline\s+)?(\d+\.\d+\.\d+)", re.IGNORECASE)
_HTTP_TOKEN = r"[-!#$%&'*+.^_`|~0-9A-Za-z]+"
_HTTP_QUOTED_STRING = (
    r'"(?:[\t\x20-\x21\x23-\x5b\x5d-\x7e\x80-\xff]'
    r'|\\[\t\x20-\x7e\x80-\xff])*"'
)
_MEDIA_TYPE_PATTERN = re.compile(
    rf"{_HTTP_TOKEN}/{_HTTP_TOKEN}"
    rf"(?:[ \t]*;[ \t]*{_HTTP_TOKEN}="
    rf"(?:{_HTTP_TOKEN}|{_HTTP_QUOTED_STRING}))*[ \t]*\Z"
)
_MACH_O_MAGICS = {
    b"\xca\xfe\xba\xbe",  # Universal binary.
    b"\xca\xfe\xba\xbf",  # Universal binary with 64-bit offsets.
    b"\xbe\xba\xfe\xca",
    b"\xbf\xba\xfe\xca",
    b"\xfe\xed\xfa\xce",  # 32-bit Mach-O.
    b"\xfe\xed\xfa\xcf",  # 64-bit Mach-O.
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
}

_BROKEN_SOURCE = b"def add(left: int, right: int) -> int:\n    return left - right\n"
_FIXED_SOURCE = b"def add(left: int, right: int) -> int:\n    return left + right\n"
_PRISTINE_TEST = (
    b"import unittest\n\n"
    b"from calculator import add\n\n\n"
    b"class CalculatorTests(unittest.TestCase):\n"
    b"    def test_adds_two_integers(self) -> None:\n"
    b"        self.assertEqual(add(2, 3), 5)\n\n\n"
    b"if __name__ == \"__main__\":\n"
    b"    unittest.main()\n"
)

_DISABLED_TOOLS = (
    "apply_patch",
    "ask_question",
    "fetch_web_content",
    "search_codebase",
    "skills",
    "spawn_agent",
    "submit_and_exit",
    "team_attach_outcome_fragment",
    "team_await_runs",
    "team_broadcast",
    "team_cancel_run",
    "team_cleanup",
    "team_create_outcome",
    "team_finalize_outcome",
    "team_list_outcomes",
    "team_list_runs",
    "team_mission_log",
    "team_read_mailbox",
    "team_review_outcome_fragment",
    "team_run_task",
    "team_send_message",
    "team_shutdown_teammate",
    "team_spawn_teammate",
    "team_status",
    "team_task",
)


class CodingCanaryError(RuntimeError):
    """Base error for a canary that cannot return trusted evidence."""


class CodingCanaryContractError(CodingCanaryError):
    """Raised for invalid caller input or an unsupported Cline contract."""


class CodingCanaryOperationalError(CodingCanaryError):
    """Raised when host conditions prevent a trustworthy conclusion."""


@dataclass(frozen=True)
class _Endpoint:
    host: str
    port: int
    base_path: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.base_path}"


@dataclass(frozen=True)
class _GatewayBinding:
    config_sha256: str
    expert_id: str
    expert_model_sha256: str
    expert_endpoint_sha256: str
    scope: str
    transport: str
    runtime_config_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "expert_id": self.expert_id,
            "expert_model_sha256": self.expert_model_sha256,
            "expert_endpoint_sha256": self.expert_endpoint_sha256,
            "scope": self.scope,
            "transport": self.transport,
            "mymoe_fallback": "disabled_by_device_only_config",
            "runtime_config_sha256": self.runtime_config_sha256,
        }


@dataclass(frozen=True)
class _ApprovalDecision:
    approved: bool
    category: str
    reason: str


@dataclass
class _ApprovalEvidence:
    observed: Counter[str] = field(default_factory=Counter)
    approved: Counter[str] = field(default_factory=Counter)
    denied: Counter[str] = field(default_factory=Counter)
    request_fingerprints: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, object]:
        return {
            "observed": dict(sorted(self.observed.items())),
            "approved": dict(sorted(self.approved.items())),
            "denied": dict(sorted(self.denied.items())),
            "request_fingerprint_sha256": _sha256_json(
                sorted(self.request_fingerprints)
            ),
            "broker_errors": len(self.errors),
            "sequence": list(self.sequence),
        }


@dataclass(frozen=True)
class _ClineEvents:
    valid: bool
    run_results: int
    finish_reason: str
    iterations: int | None
    usage: dict[str, int]
    tool_starts: Counter[str]
    tool_ends: Counter[str]
    tool_errors: int
    record_count: int
    tool_sequence: tuple[str, ...] = ()
    lifecycle_valid: bool = False
    protocol_errors: Counter[str] = field(default_factory=Counter)
    ignored_known_noise: Counter[str] = field(default_factory=Counter)
    tool_input_contract: str = "not_evaluated"
    tool_input_fingerprint_sha256: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "run_results": self.run_results,
            "finish_reason": self.finish_reason or None,
            "iterations": self.iterations,
            "usage": dict(sorted(self.usage.items())),
            "tool_starts": dict(sorted(self.tool_starts.items())),
            "tool_ends": dict(sorted(self.tool_ends.items())),
            "tool_errors": self.tool_errors,
            "record_count": self.record_count,
            "tool_sequence": list(self.tool_sequence),
            "lifecycle_valid": self.lifecycle_valid,
            "protocol_errors": dict(sorted(self.protocol_errors.items())),
            "ignored_known_noise": dict(
                sorted(self.ignored_known_noise.items())
            ),
            "tool_input_contract": self.tool_input_contract,
            "tool_input_fingerprint_sha256": (
                self.tool_input_fingerprint_sha256 or None
            ),
        }


@dataclass
class _ToolInputContract:
    source: str
    test: str
    phase: str = "reading"
    read_paths: set[str] = field(default_factory=set)
    sequence: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    valid: bool = True

    def observe(self, tool: str, value: object) -> None:
        self.fingerprints.append(_sha256_json({"tool": tool, "input": value}))
        if not self.valid or not isinstance(value, dict):
            self.valid = False
            return
        if tool == "read_files":
            approved = self._read(value)
        elif tool == "editor":
            approved = self._edit(value)
        elif tool == "run_commands":
            approved = self._command(value)
        else:
            approved = False
        if not approved:
            self.valid = False

    def invalidate(self) -> None:
        self.valid = False

    @property
    def status(self) -> str:
        if not self.valid:
            return "invalid"
        if (
            self.phase == "complete"
            and self.read_paths == {self.source, self.test}
            and 3 <= len(self.sequence) <= 4
            and self.sequence[-2:] == ["editor", "run_commands"]
            and all(item == "read_files" for item in self.sequence[:-2])
        ):
            return "complete"
        return "incomplete"

    def _read(self, value: Mapping[str, object]) -> bool:
        if self.phase != "reading" or set(value) != {"files"}:
            return False
        files = value.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 2:
            return False
        paths: set[str] = set()
        for item in files:
            if not isinstance(item, dict) or not {"path"} <= set(item) <= {
                "path",
                "start_line",
                "end_line",
            }:
                return False
            path = item.get("path")
            start = item.get("start_line")
            end = item.get("end_line")
            if not isinstance(path, str) or path not in {self.source, self.test}:
                return False
            if path in paths:
                return False
            if any(
                bound is not None
                and (
                    isinstance(bound, bool)
                    or not isinstance(bound, int)
                    or bound < 1
                )
                for bound in (start, end)
            ):
                return False
            if start is not None and end is not None and start > end:
                return False
            paths.add(path)
        if len(self.sequence) >= 2:
            return False
        self.read_paths.update(paths)
        self.sequence.append("read_files")
        if self.read_paths == {self.source, self.test}:
            self.phase = "editing"
        return True

    def _edit(self, value: Mapping[str, object]) -> bool:
        if self.phase != "editing" or not {"path", "new_text"} <= set(
            value
        ) <= {"path", "old_text", "new_text", "insert_line"}:
            return False
        old_text = value.get("old_text")
        new_text = value.get("new_text")
        allowed_edits = {
            (_BROKEN_SOURCE.decode("utf-8"), _FIXED_SOURCE.decode("utf-8")),
            ("return left - right", "return left + right"),
        }
        if (
            value.get("path") != self.source
            or value.get("insert_line") is not None
            or not isinstance(old_text, str)
            or not isinstance(new_text, str)
            or (old_text, new_text) not in allowed_edits
        ):
            return False
        self.sequence.append("editor")
        self.phase = "testing"
        return True

    def _command(self, value: Mapping[str, object]) -> bool:
        if (
            self.phase != "testing"
            or set(value) != {"commands"}
            or value.get("commands") != [_ALLOWED_COMMAND]
        ):
            return False
        self.sequence.append("run_commands")
        self.phase = "complete"
        return True


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_reason(value: object) -> str:
    candidate = str(value or "").strip().lower().replace("-", "_")
    return candidate if _SAFE_REASON.fullmatch(candidate) else "unclassified"


def _validate_timeout(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 10.0 <= value <= 900.0
    ):
        raise CodingCanaryContractError(
            "timeout_seconds must be between 10 and 900 seconds."
        )
    return float(value)


def _parse_loopback_endpoint(value: str) -> _Endpoint:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CodingCanaryContractError("The gateway URL is not canonical.")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise CodingCanaryContractError("The gateway URL is invalid.") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise CodingCanaryContractError(
            "The gateway must be an explicit numeric IPv4 loopback HTTP URL."
        )
    path = (parsed.path or "").rstrip("/")
    if path not in {"", "/v1"}:
        raise CodingCanaryContractError(
            "The gateway URL path must be empty or /v1."
        )
    canonical = f"http://127.0.0.1:{port}{path}"
    if value != canonical:
        raise CodingCanaryContractError("The gateway URL is not canonical.")
    return _Endpoint(host="127.0.0.1", port=port, base_path=path or "/v1")


def _load_gateway_binding(
    config_path: str | Path,
    *,
    model: str,
    endpoint: _Endpoint,
) -> _GatewayBinding:
    if not model.startswith("mymoe/") or len(model.split("/", 1)[1]) == 0:
        raise CodingCanaryContractError(
            "The canary requires a pinned mymoe/<expert-id> model alias."
        )
    raw = read_bounded_regular_file(
        config_path,
        max_bytes=_MAX_CONFIG_BYTES,
        label="coding canary gateway configuration",
    )
    try:
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("configuration is not an object")
        config: MoEConfig = parse_config(document)
    except (UnicodeDecodeError, json.JSONDecodeError, ConfigError, ValueError) as exc:
        raise CodingCanaryContractError(
            "The gateway configuration is invalid."
        ) from exc
    if read_bounded_regular_file(
        config_path,
        max_bytes=_MAX_CONFIG_BYTES,
        label="coding canary gateway configuration",
    ) != raw:
        raise CodingCanaryOperationalError(
            "The gateway configuration changed while it was inspected."
        )
    policy = config.execution_policy
    if (
        policy.max_scope.value != "device_only"
        or tuple(item.value for item in policy.allowed_scopes) != ("device_only",)
        or policy.allow_scope_widening
        or config.routing.fallback_order
    ):
        raise CodingCanaryContractError(
            "The gateway configuration must be device-only with no fallback order."
        )
    expert_id = model.split("/", 1)[1]
    expert = config.experts_by_id.get(expert_id)
    if expert is None:
        raise CodingCanaryContractError(
            "The pinned model alias is absent from the gateway configuration."
        )
    try:
        attestation = ExecutionScopeGuard(policy).require_allowed(
            expert.execution_target
        )
    except ScopePolicyError as exc:
        raise CodingCanaryContractError(
            "The pinned expert is not attested as direct local execution."
        ) from exc
    for candidate in config.experts:
        try:
            ExecutionScopeGuard(policy).require_allowed(
                candidate.execution_target
            )
        except ScopePolicyError as exc:
            raise CodingCanaryContractError(
                "Every configured expert must remain within the device-only boundary."
            ) from exc
    if endpoint.host != "127.0.0.1":  # Defensive after parser validation.
        raise CodingCanaryContractError("The gateway endpoint is not loopback.")
    return _GatewayBinding(
        config_sha256=_sha256_bytes(raw),
        expert_id=expert.id,
        expert_model_sha256=_sha256_text(expert.model),
        expert_endpoint_sha256=_sha256_text(str(expert.base_url)),
        scope=attestation.scope.value,
        transport=attestation.transport.value,
        runtime_config_sha256=runtime_config_sha256(config),
    )


def _write_new_private(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_new_private(path: Path, payload: Mapping[str, object]) -> None:
    _write_new_private(
        path,
        (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )


def _fixture_policy() -> WorkspaceScopePolicy:
    return WorkspaceScopePolicy(
        max_files=4,
        max_total_bytes=64 * 1024,
        max_file_bytes=32 * 1024,
    )


def _assert_fixture_entries(root: Path) -> None:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise WorkspaceSecurityError("Canary workspace is unavailable.") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise WorkspaceSecurityError("Canary workspace must be a real directory.")
    entries = list(os.scandir(root))
    for entry in entries:
        try:
            metadata = os.stat(entry.path, follow_symlinks=False)
        except OSError as exc:
            raise WorkspaceSecurityError(
                "Canary workspace entry could not be attested."
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise WorkspaceSecurityError(
                "Canary workspace contains a directory, link, or special file."
            )
        if metadata.st_nlink != 1:
            raise WorkspaceSecurityError(
                "Canary workspace files must not be hard-linked."
            )


def _snapshot_fixture(root: Path) -> tuple[WorkspaceFile, ...]:
    _assert_fixture_entries(root)
    first = snapshot_materialized(root, _fixture_policy())
    _assert_fixture_entries(root)
    second = snapshot_materialized(root, _fixture_policy())
    _assert_fixture_entries(root)
    if first != second:
        raise WorkspaceSecurityError(
            "Canary workspace changed while it was being attested."
        )
    return second


def _manifest_payload(files: Sequence[WorkspaceFile]) -> dict[str, object]:
    return {
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
        "manifest_sha256": _sha256_json([item.payload() for item in files]),
    }


def _fixture_change_reason(
    baseline: Sequence[WorkspaceFile],
    candidate: Sequence[WorkspaceFile],
) -> str:
    before = {item.path: item for item in baseline}
    after = {item.path: item for item in candidate}
    if set(after) != {_SOURCE_NAME, _TEST_NAME}:
        return "workspace_shape_changed"
    if set(before) != set(after):
        return "workspace_shape_changed"
    if after[_TEST_NAME] != before[_TEST_NAME]:
        return "pristine_test_changed"
    if after[_SOURCE_NAME].sha256 != _sha256_bytes(_FIXED_SOURCE):
        return "expected_source_edit_missing"
    if (
        after[_SOURCE_NAME].kind != before[_SOURCE_NAME].kind
        or after[_SOURCE_NAME].mode != before[_SOURCE_NAME].mode
        or after[_SOURCE_NAME].direction != before[_SOURCE_NAME].direction
    ):
        return "source_metadata_changed"
    changed = [name for name in sorted(before) if before[name] != after[name]]
    return "expected_single_file_change" if changed == [_SOURCE_NAME] else "unexpected_change_set"


class _ApprovalBroker:
    """Approve only the three exact operations in the disposable fixture."""

    def __init__(self, directory: Path, *, workspace: Path) -> None:
        self.directory = directory
        self.workspace = workspace.resolve(strict=True)
        self.source = self.workspace / _SOURCE_NAME
        self.test = self.workspace / _TEST_NAME
        self.evidence = _ApprovalEvidence()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[str] = set()
        self._approved_read_paths: set[str] = set()

    @property
    def contract_complete(self) -> bool:
        return (
            self._approved_read_paths == {str(self.source), str(self.test)}
            and self.evidence.approved["editor"] == 1
            and self.evidence.approved["run_commands"] == 1
        )

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(self.evidence.sequence)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="mymoe-cline-approval-broker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self.evidence.errors.append("approval_broker_stop_timeout")

    def _run(self) -> None:
        try:
            while not self._stop.wait(0.05):
                self._poll_once()
            self._poll_once()
        except Exception:
            self.evidence.errors.append("approval_broker_failed")

    def _poll_once(self) -> None:
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            self.evidence.errors.append("approval_directory_unavailable")
            self._stop.set()
            return
        for name in names:
            if ".request." not in name or not name.endswith(".json") or name in self._seen:
                continue
            if len(self._seen) >= _BROKER_REQUEST_LIMIT:
                self.evidence.errors.append("approval_request_limit_exceeded")
                self._stop.set()
                return
            if self._handle(name):
                self._seen.add(name)

    def _handle(self, name: str) -> bool:
        request_path = self.directory / name
        try:
            raw = read_bounded_regular_file(
                request_path,
                max_bytes=_MAX_APPROVAL_BYTES,
                label="Cline approval request",
            )
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request is not an object")
            decision = self._decide(request)
            fingerprint = _sha256_bytes(raw)
        except Exception:
            return False
        category = decision.category
        self.evidence.observed[category] += 1
        self.evidence.request_fingerprints.append(fingerprint)
        if decision.approved:
            self.evidence.approved[category] += 1
            self.evidence.sequence.append(category)
        else:
            self.evidence.denied[category] += 1
        decision_name = name.replace(".request.", ".decision.", 1)
        if decision_name == name:
            self.evidence.errors.append("approval_filename_invalid")
            return True
        try:
            _write_json_new_private(
                self.directory / decision_name,
                {"approved": decision.approved, "reason": decision.reason},
            )
        except OSError:
            self.evidence.errors.append("approval_decision_write_failed")
        return True

    def _decide(self, request: Mapping[str, object]) -> _ApprovalDecision:
        required = {
            "requestId",
            "sessionId",
            "createdAt",
            "toolCallId",
            "toolName",
            "input",
            "iteration",
            "agentId",
            "conversationId",
        }
        if set(request) != required:
            return _ApprovalDecision(False, "malformed", "invalid request envelope")
        tool = request.get("toolName")
        value = request.get("input")
        if not isinstance(tool, str) or not isinstance(value, dict):
            return _ApprovalDecision(False, "malformed", "invalid tool request")
        if tool == "read_files":
            return self._read_decision(value)
        if tool == "editor":
            return self._editor_decision(value)
        if tool == "run_commands":
            return self._command_decision(value)
        return _ApprovalDecision(False, "other_tool", "tool outside canary contract")

    def _read_decision(self, value: Mapping[str, object]) -> _ApprovalDecision:
        if self.evidence.approved["editor"] or self.evidence.approved["run_commands"]:
            return _ApprovalDecision(False, "read_files", "read requested out of order")
        if set(value) != {"files"} or not isinstance(value.get("files"), list):
            return _ApprovalDecision(False, "read_files", "invalid read contract")
        files = value["files"]
        if not 1 <= len(files) <= 2:
            return _ApprovalDecision(False, "read_files", "invalid read count")
        allowed = {str(self.source), str(self.test)}
        observed: set[str] = set()
        for item in files:
            if not isinstance(item, dict) or not {"path"} <= set(item) <= {
                "path",
                "start_line",
                "end_line",
            }:
                return _ApprovalDecision(False, "read_files", "invalid read item")
            path = item.get("path")
            if not isinstance(path, str) or path not in allowed or path in observed:
                return _ApprovalDecision(False, "read_files", "path outside fixture")
            observed.add(path)
            for bound in (item.get("start_line"), item.get("end_line")):
                if bound is not None and (
                    isinstance(bound, bool) or not isinstance(bound, int) or bound < 1
                ):
                    return _ApprovalDecision(False, "read_files", "invalid line range")
        if self.evidence.approved["read_files"] >= 2:
            return _ApprovalDecision(False, "read_files", "read budget exhausted")
        self._approved_read_paths.update(observed)
        return _ApprovalDecision(True, "read_files", "allowed disposable fixture read")

    def _editor_decision(self, value: Mapping[str, object]) -> _ApprovalDecision:
        if not {"path", "new_text"} <= set(value) <= {
            "path",
            "old_text",
            "new_text",
            "insert_line",
        }:
            return _ApprovalDecision(False, "editor", "invalid editor contract")
        if self.evidence.approved["editor"] >= 1:
            return _ApprovalDecision(False, "editor", "edit budget exhausted")
        if self._approved_read_paths != {str(self.source), str(self.test)}:
            return _ApprovalDecision(False, "editor", "required reads are incomplete")
        if value.get("path") != str(self.source):
            return _ApprovalDecision(False, "editor", "path outside editable fixture")
        old_text = value.get("old_text")
        new_text = value.get("new_text")
        insert_line = value.get("insert_line")
        allowed_edits = {
            (
                _BROKEN_SOURCE.decode("utf-8"),
                _FIXED_SOURCE.decode("utf-8"),
            ),
            ("return left - right", "return left + right"),
        }
        if (
            not isinstance(old_text, str)
            or not isinstance(new_text, str)
            or insert_line is not None
            or (old_text, new_text) not in allowed_edits
        ):
            return _ApprovalDecision(False, "editor", "edit is not the exact fixture patch")
        return _ApprovalDecision(True, "editor", "allowed exact fixture patch")

    def _command_decision(self, value: Mapping[str, object]) -> _ApprovalDecision:
        commands = value.get("commands")
        if set(value) != {"commands"} or commands != [_ALLOWED_COMMAND]:
            return _ApprovalDecision(False, "run_commands", "command outside allowlist")
        if self.evidence.approved["run_commands"] >= 1:
            return _ApprovalDecision(False, "run_commands", "command budget exhausted")
        if self.evidence.approved["editor"] != 1:
            return _ApprovalDecision(False, "run_commands", "edit is incomplete")
        return _ApprovalDecision(True, "run_commands", "allowed deterministic test")


_HOOK_STATE_SCHEMA = "mymoe-cline-hook-gate/v1"
_PRE_TOOL_HOOK = r'''#!/usr/bin/python3
import json
import os
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

STATE_PATH = os.environ["MYMOE_CANARY_HOOK_STATE"]
SOURCE = os.environ["MYMOE_CANARY_SOURCE"]
TEST = os.environ["MYMOE_CANARY_TEST"]
COMMAND = "python3 -m unittest -q test_calculator.py"
BROKEN = "def add(left: int, right: int) -> int:\n    return left - right\n"
FIXED = "def add(left: int, right: int) -> int:\n    return left + right\n"
ALLOWED_EDITS = {(BROKEN, FIXED), ("return left - right", "return left + right")}


def load_state():
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(STATE_PATH, flags)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size < 1 or metadata.st_size > 65536:
            raise ValueError("state size")
        remaining = metadata.st_size + 1
        chunks = []
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    state = json.loads(raw)
    required = {"schema_version", "phase", "read_paths", "sequence", "observed", "approved", "denied", "errors"}
    if not isinstance(state, dict) or set(state) != required or state["schema_version"] != "mymoe-cline-hook-gate/v1":
        raise ValueError("state shape")
    return state


def save_state(state):
    payload = (json.dumps(state, separators=(",", ":"), sort_keys=True) + "\n").encode()
    temporary = STATE_PATH + ".next." + str(os.getpid())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, STATE_PATH)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def count(state, bucket, key):
    values = state[bucket]
    values[key] = int(values.get(key, 0)) + 1


def decide(state, payload):
    call = payload.get("tool_call") if isinstance(payload, dict) else None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("input"), dict):
        return False, "malformed"
    tool = call["name"]
    value = call["input"]
    category = tool if tool in {"read_files", "editor", "run_commands"} else "other_tool"
    count(state, "observed", category)
    approved = False
    reason = "tool_outside_contract"
    if tool == "read_files" and state["phase"] == "reading":
        files = value.get("files")
        if set(value) == {"files"} and isinstance(files, list) and 1 <= len(files) <= 2:
            paths = []
            valid = True
            for item in files:
                if not isinstance(item, dict) or not {"path"} <= set(item) <= {"path", "start_line", "end_line"}:
                    valid = False
                    break
                path = item.get("path")
                start = item.get("start_line")
                end = item.get("end_line")
                if path not in {SOURCE, TEST} or path in paths:
                    valid = False
                    break
                if any(bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 1) for bound in (start, end)):
                    valid = False
                    break
                if start is not None and end is not None and start > end:
                    valid = False
                    break
                paths.append(path)
            if valid and len(state["sequence"]) < 2:
                approved = True
                state["read_paths"] = sorted(set(state["read_paths"]) | set(paths))
                state["sequence"].append("read_files")
                if set(state["read_paths"]) == {SOURCE, TEST}:
                    state["phase"] = "editing"
                reason = "allowed_fixture_read"
    elif tool == "editor" and state["phase"] == "editing":
        if {"path", "new_text"} <= set(value) <= {"path", "old_text", "new_text", "insert_line"}:
            edit = (value.get("old_text"), value.get("new_text"))
            if value.get("path") == SOURCE and value.get("insert_line") is None and edit in ALLOWED_EDITS:
                approved = True
                state["phase"] = "testing"
                state["sequence"].append("editor")
                reason = "allowed_fixture_edit"
    elif tool == "run_commands" and state["phase"] == "testing":
        if set(value) == {"commands"} and value.get("commands") == [COMMAND]:
            approved = True
            state["phase"] = "complete"
            state["sequence"].append("run_commands")
            reason = "allowed_fixture_test"
    if approved:
        count(state, "approved", category)
    else:
        count(state, "denied", category)
    return approved, reason


def main():
    approved = False
    reason = "hook_gate_failure"
    if fcntl is None:
        print("HOOK_CONTROL\t" + json.dumps({"cancel": True, "context": "hook_lock_unavailable"}, separators=(",", ":"), sort_keys=True))
        return
    lock_path = STATE_PATH + ".lock"
    lock_flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    lock = None
    state_persisted = False
    try:
        lock = os.open(lock_path, lock_flags, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load_state()
        raw = sys.stdin.buffer.read(131073)
        if len(raw) > 131072:
            state["errors"] = int(state["errors"]) + 1
            reason = "hook_payload_too_large"
        else:
            try:
                payload = json.loads(raw)
                approved, reason = decide(state, payload)
            except Exception:
                state["errors"] = int(state["errors"]) + 1
                reason = "hook_payload_invalid"
        save_state(state)
        state_persisted = True
    except Exception:
        approved = False
        reason = "hook_gate_failure"
    finally:
        if lock is not None:
            try:
                os.close(lock)
            except OSError:
                if not state_persisted:
                    approved = False
                    reason = "hook_gate_failure"
    print("HOOK_CONTROL\t" + json.dumps({"cancel": not approved, "context": reason}, separators=(",", ":"), sort_keys=True))


main()
'''


@dataclass(frozen=True)
class _HookGate:
    evidence: _ApprovalEvidence
    sequence: tuple[str, ...]
    phase: str
    read_scope_complete: bool

    @property
    def contract_complete(self) -> bool:
        reads = self.sequence[:-2]
        return (
            self.phase == "complete"
            and self.read_scope_complete
            and 1 <= len(reads) <= 2
            and all(item == "read_files" for item in reads)
            and self.sequence[-2:] == ("editor", "run_commands")
            and not self.evidence.denied
            and not self.evidence.errors
        )


def _initial_hook_state() -> dict[str, object]:
    return {
        "schema_version": _HOOK_STATE_SCHEMA,
        "phase": "reading",
        "read_paths": [],
        "sequence": [],
        "observed": {},
        "approved": {},
        "denied": {},
        "errors": 0,
    }


def _load_hook_gate(path: Path, *, workspace: Path) -> _HookGate:
    try:
        raw = read_bounded_regular_file(
            path,
            max_bytes=64 * 1024,
            label="Cline pre-tool policy state",
        )
    except (OSError, TwoPhaseConfigError) as exc:
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy state could not be inspected safely."
        ) from exc
    try:
        state = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy state is invalid."
        ) from exc
    expected_keys = set(_initial_hook_state())
    if not isinstance(state, dict) or set(state) != expected_keys or state.get("schema_version") != _HOOK_STATE_SCHEMA:
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy state contract is invalid."
        )
    phase = state.get("phase")
    sequence = state.get("sequence")
    read_paths = state.get("read_paths")
    errors = state.get("errors")
    if (
        phase not in {"reading", "editing", "testing", "complete"}
        or not isinstance(sequence, list)
        or len(sequence) > 4
        or any(item not in {"read_files", "editor", "run_commands"} for item in sequence)
        or not isinstance(read_paths, list)
        or len(read_paths) > 2
        or isinstance(errors, bool)
        or not isinstance(errors, int)
        or not 0 <= errors <= 100
    ):
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy state values are invalid."
        )
    counters: dict[str, Counter[str]] = {}
    for key in ("observed", "approved", "denied"):
        value = state.get(key)
        if not isinstance(value, dict) or len(value) > 8:
            raise CodingCanaryOperationalError(
                "The Cline pre-tool policy counters are invalid."
            )
        counter: Counter[str] = Counter()
        for name, count in value.items():
            if (
                not isinstance(name, str)
                or not _SAFE_REASON.fullmatch(name)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= _BROKER_REQUEST_LIMIT
            ):
                raise CodingCanaryOperationalError(
                    "The Cline pre-tool policy counters are invalid."
                )
            counter[name] = count
        counters[key] = counter
    if counters["observed"] != counters["approved"] + counters["denied"]:
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy counters do not reconcile."
        )
    allowed_paths = {
        str((workspace / _SOURCE_NAME).resolve(strict=True)),
        str((workspace / _TEST_NAME).resolve(strict=True)),
    }
    if any(not isinstance(item, str) or item not in allowed_paths for item in read_paths):
        raise CodingCanaryOperationalError(
            "The Cline pre-tool policy read scope is invalid."
        )
    evidence = _ApprovalEvidence(
        observed=counters["observed"],
        approved=counters["approved"],
        denied=counters["denied"],
        errors=[] if errors == 0 else ["hook_gate_error"],
        sequence=list(sequence),
    )
    return _HookGate(
        evidence=evidence,
        sequence=tuple(sequence),
        phase=str(phase),
        read_scope_complete=set(read_paths) == allowed_paths,
    )


def _parse_cline_events(
    raw: bytes,
    *,
    truncated: bool,
    workspace: Path | None = None,
) -> _ClineEvents:
    if truncated or len(raw) > _MAX_NDJSON_BYTES:
        return _ClineEvents(
            False,
            0,
            "",
            None,
            {},
            Counter(),
            Counter(),
            0,
            0,
            protocol_errors=Counter({"stream_truncated": 1}),
        )
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    run_results = 0
    finish_reason = ""
    iterations: int | None = None
    tool_errors = 0
    records = 0
    valid = True
    protocol_errors: Counter[str] = Counter()
    ignored_known_noise: Counter[str] = Counter()
    active_calls: dict[str, str] = {}
    completed_calls: set[str] = set()
    tool_sequence: list[str] = []
    run_result_seen = False
    input_contract = (
        None
        if workspace is None
        else _ToolInputContract(
            source=str((workspace / _SOURCE_NAME).resolve(strict=True)),
            test=str((workspace / _TEST_NAME).resolve(strict=True)),
        )
    )
    for line in raw.splitlines():
        if not line.strip():
            continue
        records += 1
        if records > 10_000:
            valid = False
            protocol_errors["record_limit_exceeded"] += 1
            break
        if line == _AI_SDK_WARNING_LINE:
            ignored_known_noise["ai_sdk_warning_banner"] += 1
            if (
                run_result_seen
                or ignored_known_noise["ai_sdk_warning_banner"] > 1
            ):
                valid = False
                protocol_errors["known_noise_order_invalid"] += 1
            continue
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid = False
            protocol_errors["record_json_invalid"] += 1
            continue
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            valid = False
            protocol_errors["record_shape_invalid"] += 1
            continue
        if item["type"] == "run_result":
            if run_result_seen or active_calls:
                valid = False
                protocol_errors["run_result_order_invalid"] += 1
            run_result_seen = True
            run_results += 1
            finish_reason = _safe_reason(item.get("finishReason"))
            raw_iterations = item.get("iterations")
            if isinstance(raw_iterations, int) and not isinstance(raw_iterations, bool):
                iterations = raw_iterations
            for container_name in ("usage", "aggregateUsage"):
                container = item.get(container_name)
                if isinstance(container, dict):
                    for key in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"):
                        value = container.get(key)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            usage[key] = max(usage[key], value)
        elif item["type"] == "agent_event":
            event = item.get("event")
            if not isinstance(event, dict) or event.get("contentType") != "tool":
                continue
            if run_result_seen:
                valid = False
                protocol_errors["tool_event_after_result"] += 1
            tool = event.get("toolName")
            if not isinstance(tool, str) or not _SAFE_REASON.fullmatch(tool):
                tool = "unknown_tool"
            if event.get("type") == "content_start":
                call_id = event.get("toolCallId")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or call_id in active_calls
                    or call_id in completed_calls
                ):
                    valid = False
                    protocol_errors["tool_start_identity_invalid"] += 1
                    if input_contract is not None:
                        input_contract.invalidate()
                    continue
                active_calls[call_id] = tool
                starts[tool] += 1
                tool_sequence.append(tool)
                if input_contract is not None:
                    input_contract.observe(tool, event.get("input"))
            elif event.get("type") == "content_end":
                call_id = event.get("toolCallId")
                if (
                    not isinstance(call_id, str)
                    or active_calls.get(call_id) != tool
                ):
                    valid = False
                    protocol_errors["tool_end_identity_invalid"] += 1
                    continue
                del active_calls[call_id]
                completed_calls.add(call_id)
                ends[tool] += 1
                if event.get("error") is not None:
                    tool_errors += 1
    if run_results != 1:
        valid = False
        protocol_errors["run_result_count_invalid"] += 1
    lifecycle_valid = run_result_seen and not active_calls and starts == ends
    if not lifecycle_valid:
        valid = False
        protocol_errors["tool_lifecycle_incomplete"] += 1
    return _ClineEvents(
        valid=valid,
        run_results=run_results,
        finish_reason=finish_reason,
        iterations=iterations,
        usage=dict(usage),
        tool_starts=starts,
        tool_ends=ends,
        tool_errors=tool_errors,
        record_count=records,
        tool_sequence=tuple(tool_sequence),
        lifecycle_valid=lifecycle_valid,
        protocol_errors=protocol_errors,
        ignored_known_noise=ignored_known_noise,
        tool_input_contract=(
            "not_evaluated" if input_contract is None else input_contract.status
        ),
        tool_input_fingerprint_sha256=(
            ""
            if input_contract is None
            else _sha256_json(input_contract.fingerprints)
        ),
    )


@dataclass
class _ProxyEvidence:
    requests: Counter[str] = field(default_factory=Counter)
    responses: Counter[str] = field(default_factory=Counter)
    violations: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def payload(self) -> dict[str, object]:
        public = {
            "requests": dict(sorted(self.requests.items())),
            "responses": dict(sorted(self.responses.items())),
            "violations": dict(sorted(self.violations.items())),
            "errors": dict(sorted(self.errors.items())),
        }
        return {**public, "receipt_sha256": _sha256_json(public)}


class _InferenceProxyServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = False

    def __init__(
        self,
        endpoint: _Endpoint,
        *,
        token: str,
        expected_model: str,
        deadline: float,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _InferenceProxyHandler)
        self.endpoint = endpoint
        self.token = token
        self.expected_model = expected_model
        self.deadline = deadline
        self.evidence = _ProxyEvidence()
        self.evidence_lock = threading.Lock()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def record(self, bucket: str, key: str) -> None:
        with self.evidence_lock:
            target: Counter[str] = getattr(self.evidence, bucket)
            target[_safe_reason(key)] += 1

    def snapshot_evidence(self) -> _ProxyEvidence:
        with self.evidence_lock:
            return _ProxyEvidence(
                requests=Counter(self.evidence.requests),
                responses=Counter(self.evidence.responses),
                violations=Counter(self.evidence.violations),
                errors=Counter(self.evidence.errors),
            )


class _InferenceProxyHandler(BaseHTTPRequestHandler):
    server: _InferenceProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
        self._forward("POST")

    def _reject(self, status: HTTPStatus, code: str) -> None:
        self.server.record("violations", code)
        body = b'{"error":"request denied by local canary broker"}\n'
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _forward(self, method: str) -> None:
        response_started = False
        if self.client_address[0] != "127.0.0.1":
            self._reject(HTTPStatus.FORBIDDEN, "non_loopback_client")
            return
        if time.monotonic() >= self.server.deadline:
            self._reject(HTTPStatus.GATEWAY_TIMEOUT, "broker_deadline_elapsed")
            return
        if self.headers.get("Authorization") != f"Bearer {self.server.token}":
            self._reject(HTTPStatus.UNAUTHORIZED, "broker_auth_invalid")
            return
        base = self.server.endpoint.base_path
        allowed = {
            ("GET", f"{base}/models"): "models",
            ("POST", f"{base}/chat/completions"): "chat_completions",
        }
        route = allowed.get((method, self.path))
        if route is None:
            self._reject(HTTPStatus.NOT_FOUND, "route_not_allowed")
            return
        content_length = self.headers.get("Content-Length")
        if method == "POST":
            try:
                length = int(content_length or "")
            except ValueError:
                self._reject(HTTPStatus.BAD_REQUEST, "content_length_invalid")
                return
            if not 1 <= length <= _MAX_PROXY_REQUEST_BYTES:
                self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self._reject(HTTPStatus.BAD_REQUEST, "request_truncated")
                return
            try:
                request = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reject(HTTPStatus.BAD_REQUEST, "request_json_invalid")
                return
            if not isinstance(request, dict):
                self._reject(HTTPStatus.BAD_REQUEST, "request_shape_invalid")
                return
            if request.get("model") != self.server.expected_model:
                self._reject(HTTPStatus.BAD_REQUEST, "model_alias_mismatch")
                return
        else:
            if content_length not in {None, "0"}:
                self._reject(HTTPStatus.BAD_REQUEST, "get_body_not_allowed")
                return
            body = None
        self.server.record("requests", route)
        remaining = max(0.1, self.server.deadline - time.monotonic())
        connection = http.client.HTTPConnection(
            self.server.endpoint.host,
            self.server.endpoint.port,
            timeout=min(remaining, 120.0),
        )
        response: http.client.HTTPResponse | None = None
        try:
            headers = {
                "Accept": self.headers.get("Accept", "application/json"),
                "Authorization": "Bearer local",
                "Content-Type": "application/json",
                "User-Agent": "mymoe-coding-canary/1",
            }
            connection.request(method, self.path, body=body, headers=headers)
            response = connection.getresponse()
            try:
                response_status = _validated_inference_response_status(response.status)
            except ValueError:
                self.server.record("errors", "upstream_response_invalid")
                response_started = True
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_response_invalid")
                return
            if 300 <= response_status < 400:
                self.server.record("errors", "upstream_redirect")
                response_started = True
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_redirect_denied")
                return
            try:
                content_type = _validated_upstream_content_type(
                    response.getheader("Content-Type")
                )
            except ValueError:
                self.server.record("errors", "upstream_response_invalid")
                response_started = True
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_response_invalid")
                return
            response_has_body = response_status not in {204, 205, 304}
            payload = (
                response.read(_MAX_PROXY_RESPONSE_BYTES + 1)
                if response_has_body
                else b""
            )
            if len(payload) > _MAX_PROXY_RESPONSE_BYTES:
                self.server.record("errors", "response_too_large")
                response_started = True
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_response_too_large")
                return
            if not 200 <= response_status < 300:
                self.server.record("errors", "upstream_non_success")
            self.server.record("responses", f"status_{response_status}")
            response_started = True
            self.send_response(response_status)
            self.send_header("Content-Type", content_type)
            if response_has_body:
                self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            if response_has_body:
                self.wfile.write(payload)
            self.close_connection = True
        except (OSError, http.client.HTTPException, TimeoutError):
            self.server.record("errors", "upstream_unavailable")
            if not response_started:
                self._reject(HTTPStatus.BAD_GATEWAY, "upstream_unavailable")
            else:
                self.close_connection = True
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                connection.close()


def _validated_inference_response_status(status: object) -> int:
    if type(status) is not int or not 200 <= status <= 599:
        raise ValueError("upstream response status is invalid")
    return status


def _validated_upstream_content_type(value: object) -> str:
    if value is None:
        return "application/json"
    if not isinstance(value, str) or len(value) > _MAX_PROXY_CONTENT_TYPE_CHARS:
        raise ValueError("upstream Content-Type is invalid")
    line_safe = value.replace("\r", "").replace("\n", "")
    if line_safe != value:
        raise ValueError("upstream Content-Type is invalid")
    if any(
        character != "\t"
        and (ord(character) < 0x20 or ord(character) == 0x7F or ord(character) > 0xFF)
        for character in line_safe
    ):
        raise ValueError("upstream Content-Type is invalid")
    safe_value = line_safe.strip(" \t")
    if _MEDIA_TYPE_PATTERN.fullmatch(safe_value) is None:
        raise ValueError("upstream Content-Type is invalid")
    return safe_value


class _InferenceProxy:
    def __init__(
        self,
        endpoint: _Endpoint,
        *,
        token: str,
        expected_model: str,
        deadline: float,
    ) -> None:
        self.server = _InferenceProxyServer(
            endpoint,
            token=token,
            expected_model=expected_model,
            deadline=deadline,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="mymoe-inference-proxy",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return self.server.port

    @property
    def evidence(self) -> _ProxyEvidence:
        return self.snapshot_evidence()

    def snapshot_evidence(self) -> _ProxyEvidence:
        return self.server.snapshot_evidence()

    def __enter__(self) -> _InferenceProxy:
        self.thread.start()
        if not self.thread.is_alive():
            raise CodingCanaryOperationalError("The inference broker did not start.")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            raise CodingCanaryOperationalError(
                "The inference broker could not be stopped safely."
            )
        self.server.server_close()


def _gateway_get_json(endpoint: _Endpoint, path: str) -> object:
    connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=5.0)
    try:
        connection.request(
            "GET",
            path,
            headers={"Accept": "application/json", "Authorization": "Bearer local"},
        )
        response = connection.getresponse()
        if response.status != HTTPStatus.OK.value:
            raise CodingCanaryOperationalError(
                "The local gateway models endpoint is unavailable."
            )
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise CodingCanaryOperationalError(
                "The local gateway model inventory exceeded its bound."
            )
        return json.loads(raw)
    except (
        OSError,
        http.client.HTTPException,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise CodingCanaryOperationalError(
            "The local gateway identity could not be inspected."
        ) from exc
    finally:
        connection.close()


def _capture_gateway_models(
    endpoint: _Endpoint,
    *,
    expected_model: str,
    binding: _GatewayBinding,
) -> dict[str, object]:
    payload = _gateway_get_json(endpoint, f"{endpoint.base_path}/models")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise CodingCanaryOperationalError(
            "The local gateway returned an invalid model inventory."
        )
    model_entries = {
        item.get("id"): item
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    identifiers = sorted(
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    if len(set(identifiers)) != len(identifiers):
        raise CodingCanaryOperationalError(
            "The local gateway returned an ambiguous model inventory."
        )
    if expected_model not in identifiers:
        raise CodingCanaryOperationalError(
            "The pinned model alias is not served by the local gateway."
        )
    selected = model_entries[expected_model]
    selected_metadata = selected.get("mymoe")
    if not isinstance(selected_metadata, dict) or (
        selected_metadata.get("selection") != "pinned"
        or selected_metadata.get("expert_id") != binding.expert_id
        or _sha256_text(str(selected_metadata.get("upstream_model")))
        != binding.expert_model_sha256
        or selected_metadata.get("execution_scope") != binding.scope
        or selected_metadata.get("execution_transport") != binding.transport
        or selected_metadata.get("eligible") is not True
    ):
        raise CodingCanaryOperationalError(
            "The live gateway model metadata does not match the declared config."
        )
    runtime = _gateway_get_json(endpoint, "/api/config")
    runtime_sha256 = (
        runtime.get("runtime_config_sha256")
        if isinstance(runtime, dict)
        else None
    )
    if (
        not isinstance(runtime_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None
        or not secrets.compare_digest(
            runtime_sha256, binding.runtime_config_sha256
        )
    ):
        raise CodingCanaryOperationalError(
            "The live gateway runtime does not match the declared config."
        )
    return {
        "inventory_sha256": _sha256_json(identifiers),
        "model_count": len(identifiers),
        "selected_present": True,
        "runtime_config_sha256": runtime_sha256,
    }


def _sandbox_quote(path: Path) -> str:
    return json.dumps(str(path.resolve(strict=True)))


def _build_macos_profile(
    *,
    canary_root: Path,
    cline_root: Path,
    forbidden_home: Path,
    broker_port: int,
    workspace: Path | None = None,
    editable_file: Path | None = None,
    read_only_paths: Sequence[Path] = (),
    writable_roots: Sequence[Path] = (),
) -> tuple[str, str]:
    denied = [forbidden_home, Path("/Volumes"), Path("/Network"), Path("/private/tmp")]
    var_folders = Path("/private/var/folders")
    if var_folders.exists():
        denied.append(var_folders)
    clauses = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
    ]
    clauses.append(
        f'(allow network-outbound (remote ip "localhost:{broker_port}"))'
    )
    clauses.append("(deny network-bind)")
    for root in denied:
        if root.exists():
            rendered = _sandbox_quote(root)
            clauses.append(f"(deny file-read* file-write* (subpath {rendered}))")
    for root, operations in (
        (canary_root, "file-read*"),
        (cline_root, "file-read* file-map-executable"),
    ):
        rendered = _sandbox_quote(root)
        clauses.append(f"(allow {operations} (subpath {rendered}))")
    for root in writable_roots:
        clauses.append(
            f"(allow file-write* (subpath {_sandbox_quote(root)}))"
        )
    if workspace is not None:
        clauses.append(
            f"(deny file-write* (subpath {_sandbox_quote(workspace)}))"
        )
    if editable_file is not None:
        clauses.append(
            f"(allow file-write* (literal {_sandbox_quote(editable_file)}))"
        )
    for path in read_only_paths:
        clauses.append(f"(deny file-write* (literal {_sandbox_quote(path)}))")
    clauses.append("(deny appleevent-send)")
    profile = "".join(clauses)
    semantic = profile.replace(str(canary_root.resolve()), "<canary-root>").replace(
        str(cline_root.resolve()), "<cline-root>"
    ).replace(str(forbidden_home.resolve()), "<host-home>")
    return profile, _sha256_text(semantic)


def _minimal_environment(
    *,
    home: Path,
    temporary: Path,
    cline_dir: Path,
    global_settings: Path,
    hook_state: Path,
    mcp_settings: Path,
    source: Path,
    test: Path,
) -> dict[str, str]:
    return {
        "CFFIXED_USER_HOME": str(home),
        "CLINE_GLOBAL_SETTINGS_PATH": str(global_settings),
        "CLINE_DIR": str(cline_dir),
        "CLINE_LOG_ENABLED": "0",
        "CLINE_MCP_SETTINGS_PATH": str(mcp_settings),
        "CLINE_NO_AUTO_UPDATE": "1",
        "CLINE_TELEMETRY_ENABLED": "0",
        "DO_NOT_TRACK": "1",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "mymoe-canary",
        "MYMOE_CANARY_HOOK_STATE": str(hook_state),
        "MYMOE_CANARY_SOURCE": str(source),
        "MYMOE_CANARY_TEST": str(test),
        "NO_COLOR": "1",
        "NO_TELEMETRY": "1",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SHELL": "/bin/sh",
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "USER": "mymoe-canary",
    }


def _process_payload(result: ProcessExecutionResult) -> dict[str, object]:
    return {
        "code": result.code,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "stdout_sha256": result.stdout_sha256,
        "stderr_sha256": result.stderr_sha256,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "duration_ms": result.duration_ms,
        "environment_sha256": result.environment.sha256,
        "cleanup": result.cleanup.payload(),
    }


def _execute_sandboxed(
    sandbox: ExecutableIdentity,
    *,
    profile: str,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    stdout_limit: int = _MAX_NDJSON_BYTES,
) -> ProcessExecutionResult:
    return execute_process(
        sandbox,
        ("-p", profile, *command),
        cwd=cwd,
        env=environment,
        timeout_seconds=timeout_seconds,
        policy=ProcessExecutionPolicy(
            stdin_limit_bytes=0,
            stdout_limit_bytes=stdout_limit,
            stderr_limit_bytes=1024 * 1024,
            require_tree_isolation=True,
            require_psutil=False,
        ),
    )


_POLICY_PROBE = r"""
import json, os, socket, sys
allowed_file, forbidden_file, forbidden_write, protected_file, allowed_port, denied_port = sys.argv[1:]
result = {}
try:
    with open(allowed_file, "wb") as handle:
        handle.write(b"ok")
    result["scratch_write"] = True
except OSError:
    result["scratch_write"] = False
try:
    with open(forbidden_file, "rb") as handle:
        handle.read(1)
    result["host_read_denied"] = False
except OSError:
    result["host_read_denied"] = True
try:
    descriptor = os.open(forbidden_write, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    result["host_write_denied"] = False
except OSError:
    result["host_write_denied"] = True
try:
    with open(protected_file, "ab") as handle:
        handle.write(b"unsafe")
    result["protected_write_denied"] = False
except OSError:
    result["protected_write_denied"] = True
try:
    connection = socket.create_connection(("127.0.0.1", int(allowed_port)), 1.0)
    connection.close()
    result["broker_connect"] = True
except OSError:
    result["broker_connect"] = False
try:
    connection = socket.create_connection(("127.0.0.1", int(denied_port)), 1.0)
    connection.close()
    result["other_port_denied"] = False
except OSError:
    result["other_port_denied"] = True
try:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.close()
    result["bind_denied"] = False
except OSError:
    result["bind_denied"] = True
print(json.dumps(result, separators=(",", ":"), sort_keys=True))
""".strip()


def _probe_macos_profile(
    sandbox: ExecutableIdentity,
    *,
    profile: str,
    canary_root: Path,
    workspace: Path,
    probe_directory: Path,
    protected_file: Path,
    forbidden_file: Path,
    broker_port: int,
    environment: Mapping[str, str],
) -> dict[str, object]:
    allowed_file = probe_directory / ".policy-probe"
    forbidden_write = Path("/private/var/tmp") / (
        f".mymoe-canary-policy-probe-{secrets.token_hex(12)}"
    )
    denied_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    denied_socket.bind(("127.0.0.1", 0))
    denied_socket.listen(1)
    denied_port = int(denied_socket.getsockname()[1])
    try:
        result = _execute_sandboxed(
            sandbox,
            profile=profile,
            command=(
                "/usr/bin/python3",
                "-I",
                "-c",
                _POLICY_PROBE,
                str(allowed_file),
                str(forbidden_file),
                str(forbidden_write),
                str(protected_file),
                str(broker_port),
                str(denied_port),
            ),
            cwd=workspace,
            environment=environment,
            timeout_seconds=10.0,
            stdout_limit=32 * 1024,
        )
    finally:
        denied_socket.close()
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodingCanaryOperationalError(
            "The macOS sandbox policy probe returned invalid evidence."
        ) from exc
    expected = {
        "bind_denied": True,
        "broker_connect": True,
        "host_read_denied": True,
        "host_write_denied": True,
        "other_port_denied": True,
        "protected_write_denied": True,
        "scratch_write": True,
    }
    try:
        allowed_file.unlink()
    except FileNotFoundError:
        pass
    try:
        forbidden_write.unlink()
    except FileNotFoundError:
        pass
    if not result.ok or payload != expected:
        raise CodingCanaryOperationalError(
            "The assembled macOS sandbox policy failed its live probe."
        )
    return {
        "status": "passed",
        "checks": sorted(expected),
        "process": _process_payload(result),
    }


def _read_candidate_source(path: Path) -> bytes:
    try:
        first = read_bounded_regular_file(
            path,
            max_bytes=32 * 1024,
            label="coding canary candidate source",
        )
        second = read_bounded_regular_file(
            path,
            max_bytes=32 * 1024,
            label="coding canary candidate source",
        )
    except (OSError, TwoPhaseConfigError) as exc:
        raise CodingCanaryOperationalError(
            "The candidate source could not be inspected safely."
        ) from exc
    if first != second:
        raise CodingCanaryOperationalError(
            "The candidate source changed while it was sealed."
        )
    if first != _FIXED_SOURCE:
        raise CodingCanaryOperationalError(
            "The sealed candidate no longer matches the attested exact fix."
        )
    return first


def _run_independent_verifier(
    *,
    candidate_source: bytes,
    verification_root: Path,
) -> dict[str, object]:
    verification_root.mkdir(mode=0o700)
    source = verification_root / _SOURCE_NAME
    test = verification_root / _TEST_NAME
    _write_new_private(source, candidate_source)
    _write_new_private(test, _PRISTINE_TEST)
    source.chmod(0o400)
    test.chmod(0o400)
    policy = VerifierIsolationPolicy()
    capability = verifier_isolation_capability(policy)
    if not capability.supported or capability.executable is None:
        raise CodingCanaryOperationalError(
            "Independent verifier isolation is unavailable."
        )
    namespace = secrets.token_hex(12)
    try:
        plan = build_verifier_isolation_plan(
            policy,
            capability,
            workspace=verification_root,
            command_argv=(
                sys.executable,
                "-I",
                "-c",
                _VERIFIER_SCRIPT,
            ),
            runtime_read_roots=("{python_runtime}",),
            temp_namespace=namespace,
            attested_read_artifacts=(sys.executable,),
        )
    except (OSError, ValueError, VerifierIsolationError) as exc:
        raise CodingCanaryOperationalError(
            "The independent verifier isolation plan could not be built."
        ) from exc
    internal_temp = Path(plan.internal_temp)
    internal_temp.mkdir(mode=0o700)
    verification_root.chmod(0o500)
    environment = {
        "HOME": str(internal_temp),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(internal_temp),
        "TMP": str(internal_temp),
        "TMPDIR": str(internal_temp),
    }
    try:
        result = execute_process(
            capability.executable,
            plan.argv,
            cwd=verification_root,
            env=environment,
            timeout_seconds=15.0,
            policy=ProcessExecutionPolicy(
                stdin_limit_bytes=0,
                stdout_limit_bytes=64 * 1024,
                stderr_limit_bytes=64 * 1024,
                require_tree_isolation=True,
            ),
        )
    except AssistantBridgeRuntimeError as exc:
        raise CodingCanaryOperationalError(
            "The independent verifier could not complete safely."
        ) from exc
    finally:
        verification_root.chmod(0o700)
        source.chmod(0o600)
        test.chmod(0o600)
        shutil.rmtree(internal_temp, ignore_errors=True)
    return {
        "passed": result.ok,
        "profile_sha256": plan.profile_sha256,
        "binding_sha256": plan.binding_sha256,
        "process": _process_payload(result),
        "test_source": "pristine_embedded_fixture",
        "network": "denied",
    }


def _cline_version(result: ProcessExecutionResult) -> str:
    if not result.ok or result.stdout_truncated or result.stderr_truncated:
        raise CodingCanaryContractError(
            "The Cline executable did not return a bounded version."
        )
    combined = (
        result.stdout
        + (b"\n" if result.stdout and result.stderr else b"")
        + result.stderr
    )
    text = " ".join(combined.decode("utf-8", errors="replace").split())
    match = _CLINE_VERSION.search(text)
    if match is None or match.group(1) != SUPPORTED_CLINE_VERSION:
        raise CodingCanaryContractError(
            f"This canary is pinned to Cline {SUPPORTED_CLINE_VERSION}."
        )
    return match.group(1)


def _select_forbidden_probe_file(
    *,
    home: Path,
    config_path: str | Path,
    cline_root: Path,
    canary_root: Path,
) -> Path:
    candidates = (
        Path(config_path).expanduser(),
        Path(__file__),
        Path.cwd() / "README.md",
        home / ".zshrc",
        home / ".profile",
        home / ".bash_profile",
        home / ".gitconfig",
        home / "Library" / "Preferences" / ".GlobalPreferences.plist",
    )
    denied_roots = tuple(
        root.resolve(strict=True)
        for root in (
            home,
            Path("/Volumes"),
            Path("/Network"),
            Path("/private/tmp"),
            Path("/private/var/folders"),
        )
        if root.exists()
    )
    allowed_roots = (
        cline_root.resolve(strict=True),
        canary_root.resolve(strict=True),
    )
    for candidate in candidates:
        try:
            before = candidate.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                continue
            resolved = candidate.resolve(strict=True)
            after = resolved.lstat()
            if (
                stat.S_ISLNK(after.st_mode)
                or not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or any(_path_is_within(resolved, root) for root in allowed_roots)
                or not any(_path_is_within(resolved, root) for root in denied_roots)
            ):
                continue
        except OSError:
            continue
        return resolved
    raise CodingCanaryOperationalError(
        "No safe host file was available to exercise the sandbox deny rule."
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _same_executable(left: ExecutableIdentity, right: ExecutableIdentity) -> bool:
    return (
        left.resolved_path == right.resolved_path
        and left.sha256 == right.sha256
        and left.size_bytes == right.size_bytes
        and left.mtime_ns == right.mtime_ns
        and left.device_id == right.device_id
        and left.inode == right.inode
    )


def _require_direct_macos_native_executable(identity: ExecutableIdentity) -> None:
    """Reject scripts and launchers before running any Cline-controlled code.

    Cline's npm package exposes a JavaScript ``bin/cline`` resolver whose bytes
    do not bind the compiled ``bin/.cline`` process that it launches. The
    qualification contract therefore accepts only the direct native Mach-O
    executable and pins that file itself.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(identity.resolved_path, flags)
    except OSError as exc:
        raise CodingCanaryOperationalError(
            "The pinned Cline executable could not be re-opened safely."
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        observed = (
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
        )
        expected = (
            identity.size_bytes,
            identity.mtime_ns,
            identity.device_id,
            identity.inode,
            identity.mode,
        )
        if observed != expected:
            raise CodingCanaryOperationalError(
                "The pinned Cline executable changed before format attestation."
            )
        magic = os.read(descriptor, 4)
    finally:
        os.close(descriptor)
    if magic not in _MACH_O_MAGICS:
        raise CodingCanaryContractError(
            "The canary requires the direct native Cline Mach-O executable; "
            "scripts and launcher wrappers are not admitted."
        )


def _reattest_cline_executable(
    path: str,
    *,
    environment: Mapping[str, str],
) -> ExecutableIdentity:
    try:
        return resolve_executable(path, env=environment)
    except (AssistantBridgeRuntimeError, OSError) as exc:
        raise CodingCanaryOperationalError(
            "The Cline executable could not be re-attested after the run."
        ) from exc


def _base_report(
    *,
    status: str,
    reasons: Sequence[str],
    model: str,
    gateway: _GatewayBinding | None,
    cline: ExecutableIdentity | None,
    cline_version: str | None,
) -> dict[str, object]:
    hardware = detect_hardware()
    hardware_binding = {
        "machine": hardware.machine,
        "cpu_brand": hardware.cpu_brand,
        "memory_bytes": hardware.memory_bytes,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CANARY_CONTRACT,
        "status": status,
        "reason_codes": [_safe_reason(item) for item in reasons],
        "checked_at": _utc_now(),
        "diagnostic_only": True,
        "authorizes_routing": False,
        "qualified_scope": (
            "single_disposable_file_edit_and_pristine_test"
            if status == "qualified"
            else None
        ),
        "excluded_capabilities": [
            "browser",
            "desktop_control",
            "git_publication",
            "mcp",
            "real_workspace_access",
            "remote_network",
        ],
        "fixture": {
            "version": FIXTURE_VERSION,
            "editable_paths": [_SOURCE_NAME],
            "pristine_test_paths": [_TEST_NAME],
            "fixture_sha256": _sha256_json(
                {
                    _SOURCE_NAME: _sha256_bytes(_BROKEN_SOURCE),
                    _TEST_NAME: _sha256_bytes(_PRISTINE_TEST),
                }
            ),
        },
        "model": {
            "alias": model,
            "alias_sha256": _sha256_text(model),
        },
        "gateway": None if gateway is None else gateway.payload(),
        "cline": (
            None
            if cline is None
            else {
                "version": cline_version,
                "artifact_type": "direct_macos_native_executable",
                "executable_sha256": cline.sha256,
                "executable_size_bytes": cline.size_bytes,
            }
        ),
        "hardware": {
            **hardware_binding,
            "fingerprint_sha256": _sha256_json(hardware_binding),
        },
    }


def _coding_prompt(workspace: Path) -> str:
    source = workspace / _SOURCE_NAME
    test = workspace / _TEST_NAME
    return (
        "This is a deterministic local qualification task in a disposable "
        "workspace. Use only read_files, editor, and run_commands. First read "
        f"both {source} and {test}. Edit only {source}: replace the exact text "
        "`return left - right` with `return left + right`. Then call "
        f"run_commands exactly once with `{_ALLOWED_COMMAND}`. Do not create, "
        "delete, rename, or access any other file. Do not use MCP, web, browser, "
        "subagents, teams, Git, or desktop tools. Do not merely explain the fix; "
        "finish only after the allowlisted test command completes."
    )


def _classify_completed_run(
    *,
    broker: _ApprovalBroker | _HookGate,
    events: _ClineEvents,
    change_reason: str,
    verifier: Mapping[str, object],
    proxy: _ProxyEvidence,
) -> tuple[str, list[str]]:
    if broker.evidence.errors:
        return "indeterminate", ["tool_gate_observation_incomplete"]
    if proxy.errors:
        return "indeterminate", ["inference_broker_upstream_incomplete"]
    if broker.evidence.denied:
        return "incompatible", ["tool_request_denied"]
    if proxy.violations:
        return "incompatible", ["inference_broker_policy_violation"]
    if events.tool_input_contract == "invalid":
        return "incompatible", ["tool_input_contract_mismatch"]
    if not events.valid:
        return "indeterminate", ["cline_event_stream_ambiguous"]
    if events.finish_reason != "completed":
        return "incompatible", ["cline_run_not_completed"]
    if change_reason != "expected_single_file_change":
        return "incompatible", [change_reason]
    if not broker.contract_complete:
        return "incompatible", ["tool_contract_incomplete"]
    if events.tool_input_contract != "complete":
        return "incompatible", ["tool_input_contract_incomplete"]
    expected = broker.evidence.approved
    if (
        events.tool_starts != expected
        or events.tool_ends != expected
        or events.tool_errors
        or events.tool_sequence != broker.sequence
    ):
        return "incompatible", ["tool_event_contract_mismatch"]
    if proxy.requests["chat_completions"] < 1:
        return "incompatible", ["model_request_missing"]
    if verifier.get("passed") is not True:
        return "incompatible", ["pristine_verifier_failed"]
    return "qualified", ["all_qualification_gates_passed"]


def run_coding_canary(
    cline_executable: str | Path,
    *,
    base_url: str,
    gateway_config: str | Path,
    model: str = "mymoe/coder",
    timeout_seconds: float = 180.0,
    expected_cline_sha256: str | None = None,
) -> dict[str, object]:
    """Run one bounded local coding qualification and return metadata only."""

    selected_timeout = _validate_timeout(timeout_seconds)
    if (
        not isinstance(expected_cline_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_cline_sha256) is None
    ):
        raise CodingCanaryContractError(
            "The canary requires a lowercase caller-pinned Cline SHA-256."
        )
    endpoint = _parse_loopback_endpoint(base_url)
    gateway = _load_gateway_binding(
        gateway_config,
        model=model,
        endpoint=endpoint,
    )
    if sys.platform != "darwin":
        return _base_report(
            status="indeterminate",
            reasons=("coding_canary_platform_not_supported",),
            model=model,
            gateway=gateway,
            cline=None,
            cline_version=None,
        )

    try:
        preflight_cline = resolve_executable(cline_executable, env=os.environ)
    except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
        raise CodingCanaryContractError(
            "The pinned Cline executable is unavailable."
        ) from exc
    if not secrets.compare_digest(
        preflight_cline.sha256, expected_cline_sha256
    ):
        raise CodingCanaryContractError(
            "The Cline executable does not match the caller-pinned SHA-256."
        )
    _require_direct_macos_native_executable(preflight_cline)

    temporary_parent = Path("/private/tmp")
    if not temporary_parent.is_dir():
        raise CodingCanaryOperationalError(
            "The fixed local canary temporary root is unavailable."
        )
    with tempfile.TemporaryDirectory(
        prefix="mymoe-coding-canary-",
        dir=temporary_parent,
    ) as temporary:
        root = Path(temporary).resolve(strict=True)
        root.chmod(0o700)
        workspace = root / "workspace"
        state = root / "cline-state"
        config = root / "cline-config"
        home = root / "home"
        scratch = root / "tmp"
        verification = root / "verification"
        for directory in (workspace, state, config, home, scratch):
            directory.mkdir(mode=0o700)
        _write_new_private(workspace / _SOURCE_NAME, _BROKEN_SOURCE)
        _write_new_private(workspace / _TEST_NAME, _PRISTINE_TEST)
        baseline = _snapshot_fixture(workspace)
        global_settings = state / "global-settings.json"
        mcp_settings = state / "mcp-settings.json"
        hook_state = state / "pre-tool-gate.json"
        hook_config = config / "hooks"
        hook_config.mkdir(mode=0o700)
        pre_tool_hook = hook_config / "PreToolUse.py"
        _write_json_new_private(
            global_settings,
            {"disabledTools": list(_DISABLED_TOOLS)},
        )
        _write_json_new_private(mcp_settings, {"mcpServers": {}})
        _write_json_new_private(hook_state, _initial_hook_state())
        _write_new_private(pre_tool_hook, _PRE_TOOL_HOOK.encode("utf-8"))
        environment = _minimal_environment(
            home=home,
            temporary=scratch,
            cline_dir=config,
            global_settings=global_settings,
            hook_state=hook_state,
            mcp_settings=mcp_settings,
            source=(workspace / _SOURCE_NAME).resolve(strict=True),
            test=(workspace / _TEST_NAME).resolve(strict=True),
        )
        try:
            cline_identity = _reattest_cline_executable(
                preflight_cline.resolved_path,
                environment=environment,
            )
            if not _same_executable(preflight_cline, cline_identity):
                raise CodingCanaryOperationalError(
                    "The pinned Cline executable changed after preflight."
                )
            _require_direct_macos_native_executable(cline_identity)
            version_probe = execute_process(
                cline_identity,
                ("--version",),
                env=environment,
                timeout_seconds=5.0,
                policy=ProcessExecutionPolicy(
                    stdin_limit_bytes=0,
                    stdout_limit_bytes=32 * 1024,
                    stderr_limit_bytes=32 * 1024,
                    require_tree_isolation=True,
                ),
            )
            version = _cline_version(version_probe)
            sandbox = resolve_executable("/usr/bin/sandbox-exec", env=environment)
        except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
            raise CodingCanaryContractError(
                "The pinned Cline or macOS sandbox executable is unavailable."
            ) from exc
        base_report = _base_report(
            status="indeterminate",
            reasons=("canary_not_completed",),
            model=model,
            gateway=gateway,
            cline=cline_identity,
            cline_version=version,
        )
        cline_root = Path(cline_identity.resolved_path).parent.resolve(strict=True)
        host_home = Path.home().resolve(strict=True)
        forbidden_file = _select_forbidden_probe_file(
            home=host_home,
            config_path=gateway_config,
            cline_root=cline_root,
            canary_root=root,
        )
        token = secrets.token_urlsafe(32)
        gateway_before = _capture_gateway_models(
            endpoint,
            expected_model=model,
            binding=gateway,
        )
        deadline = time.monotonic() + selected_timeout + 45.0

        try:
            with _InferenceProxy(
                endpoint,
                token=token,
                expected_model=model,
                deadline=deadline,
            ) as inference_proxy:
                auth_profile, auth_profile_sha256 = _build_macos_profile(
                    canary_root=root,
                    cline_root=cline_root,
                    forbidden_home=host_home,
                    broker_port=inference_proxy.port,
                    workspace=workspace,
                    read_only_paths=(
                        workspace / _SOURCE_NAME,
                        workspace / _TEST_NAME,
                        global_settings,
                        mcp_settings,
                        pre_tool_hook,
                    ),
                    writable_roots=(state, config, home, scratch),
                )
                profile, profile_sha256 = _build_macos_profile(
                    canary_root=root,
                    cline_root=cline_root,
                    forbidden_home=host_home,
                    broker_port=inference_proxy.port,
                    workspace=workspace,
                    editable_file=workspace / _SOURCE_NAME,
                    read_only_paths=(
                        workspace / _TEST_NAME,
                        global_settings,
                        mcp_settings,
                        pre_tool_hook,
                    ),
                    writable_roots=(state, home, scratch),
                )
                policy_probe = _probe_macos_profile(
                    sandbox,
                    profile=profile,
                    canary_root=root,
                    workspace=workspace,
                    probe_directory=scratch,
                    forbidden_file=forbidden_file,
                    protected_file=workspace / _TEST_NAME,
                    broker_port=inference_proxy.port,
                    environment=environment,
                )
                broker_url = f"http://127.0.0.1:{inference_proxy.port}/v1"
                auth = _execute_sandboxed(
                    sandbox,
                    profile=auth_profile,
                    command=(
                        cline_identity.resolved_path,
                        "auth",
                        "--provider",
                        "openai-compatible",
                        "--apikey",
                        token,
                        "--modelid",
                        model,
                        "--baseurl",
                        broker_url,
                        "--config",
                        str(config),
                        "--cwd",
                        str(workspace),
                        "--data-dir",
                        str(state),
                    ),
                    cwd=workspace,
                    environment=environment,
                    timeout_seconds=15.0,
                    stdout_limit=256 * 1024,
                )
                if not auth.ok:
                    return {
                        **base_report,
                        "reason_codes": ["cline_auth_incomplete"],
                        "isolation": {
                            "backend": "sandbox-exec",
                            "profile_sha256": profile_sha256,
                            "auth_profile_sha256": auth_profile_sha256,
                            "live_probe": policy_probe,
                            "strength": "targeted_host_data_and_egress_policy",
                        },
                        "auth": _process_payload(auth),
                    }
                run = _execute_sandboxed(
                    sandbox,
                    profile=profile,
                    command=(
                        cline_identity.resolved_path,
                        "--json",
                        "--auto-approve",
                        "true",
                        "--cwd",
                        str(workspace),
                        "--compaction",
                        "off",
                        "--retries",
                        "2",
                        "--timeout",
                        str(int(selected_timeout)),
                        "--provider",
                        "openai-compatible",
                        "--key",
                        token,
                        "--model",
                        model,
                        "--config",
                        str(config),
                        "--data-dir",
                        str(state),
                        _coding_prompt(workspace),
                    ),
                    cwd=workspace,
                    environment=environment,
                    timeout_seconds=selected_timeout + 10.0,
                )
            proxy_evidence = inference_proxy.snapshot_evidence()
        except CodingCanaryOperationalError:
            raise
        except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
            raise CodingCanaryOperationalError(
                "The sandboxed Cline process could not be observed safely."
            ) from exc

        tool_gate = _load_hook_gate(hook_state, workspace=workspace)
        if run.timed_out or run.code in {
            "io_failed",
            "stderr_limit_exceeded",
            "stdout_limit_exceeded",
            "timed_out",
        }:
            return {
                **base_report,
                "reason_codes": ["cline_execution_incomplete"],
                "run": _process_payload(run),
                "tool_gate": tool_gate.evidence.payload(),
                "inference_broker": proxy_evidence.payload(),
            }
        events = _parse_cline_events(
            run.stdout,
            truncated=run.stdout_truncated,
            workspace=workspace,
        )
        try:
            candidate = _snapshot_fixture(workspace)
            change_reason = _fixture_change_reason(baseline, candidate)
        except WorkspaceSecurityError:
            candidate = ()
            change_reason = "workspace_unsafe"
        cline_after = _reattest_cline_executable(
            cline_identity.resolved_path,
            environment=environment,
        )
        if not _same_executable(cline_identity, cline_after):
            return {
                **base_report,
                "reason_codes": ["cline_executable_identity_drift"],
                "run": _process_payload(run),
            }
        gateway_after = _capture_gateway_models(
            endpoint,
            expected_model=model,
            binding=gateway,
        )
        if gateway_after != gateway_before:
            return {
                **base_report,
                "reason_codes": ["gateway_model_identity_drift"],
                "run": _process_payload(run),
            }
        if not run.ok and not tool_gate.evidence.denied:
            return {
                **base_report,
                "reason_codes": ["cline_process_nonzero_without_policy_evidence"],
                "run": _process_payload(run),
                "events": events.payload(),
            }
        verifier: dict[str, object]
        if change_reason == "expected_single_file_change":
            candidate_source = _read_candidate_source(workspace / _SOURCE_NAME)
            verifier = _run_independent_verifier(
                candidate_source=candidate_source,
                verification_root=verification,
            )
        else:
            verifier = {"passed": False, "status": "not_run_candidate_invalid"}
        status, reasons = _classify_completed_run(
            broker=tool_gate,
            events=events,
            change_reason=change_reason,
            verifier=verifier,
            proxy=proxy_evidence,
        )
        final_report = _base_report(
            status=status,
            reasons=reasons,
            model=model,
            gateway=gateway,
            cline=cline_identity,
            cline_version=version,
        )
        final_report.update(
            {
                "isolation": {
                    "backend": "sandbox-exec",
                    "profile_sha256": profile_sha256,
                    "auth_profile_sha256": auth_profile_sha256,
                    "live_probe": policy_probe,
                    "strength": "targeted_host_data_and_egress_policy",
                    "process_tree": "observed_cleanup_not_vm_containment",
                },
                "gateway_runtime": {
                    "before": gateway_before,
                    "after": gateway_after,
                },
                "baseline": _manifest_payload(baseline),
                "candidate": (
                    {"status": "unsafe"}
                    if not candidate
                    else {
                        **_manifest_payload(candidate),
                        "change": change_reason,
                        "changed_paths": (
                            [_SOURCE_NAME]
                            if change_reason == "expected_single_file_change"
                            else []
                        ),
                    }
                ),
                "auth": _process_payload(auth),
                "run": _process_payload(run),
                "events": events.payload(),
                "tool_gate": {
                    **tool_gate.evidence.payload(),
                    "sequence": list(tool_gate.sequence),
                    "phase": tool_gate.phase,
                    "read_scope_complete": tool_gate.read_scope_complete,
                    "contract_complete": tool_gate.contract_complete,
                },
                "inference_broker": proxy_evidence.payload(),
                "verifier": verifier,
            }
        )
        _validate_report_metadata(final_report)
        return final_report


def _validate_report_metadata(report: Mapping[str, object]) -> None:
    forbidden_keys = {
        "api_key",
        "apikey",
        "authorization",
        "content",
        "input",
        "message",
        "messages",
        "output",
        "prompt",
        "raw",
        "reasoning",
        "request_body",
        "response_body",
        "secret",
        "stderr",
        "stdin",
        "stdout",
        "token",
    }

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or key.lower() in forbidden_keys:
                    raise CodingCanaryOperationalError(
                        "The coding canary report contains a raw-content field."
                    )
                inspect(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
            return
        if isinstance(value, str):
            if (
                value.startswith("/")
                or re.match(r"[A-Za-z]:[\\/]", value)
                or "return left - right" in value
                or "return left + right" in value
                or "Bearer " in value
                or str(Path.home()) in value
            ):
                raise CodingCanaryOperationalError(
                    "The coding canary report contains path, content, or credential data."
                )

    inspect(dict(report))


def write_coding_canary_report(
    path: str | Path,
    report: Mapping[str, object],
) -> Path:
    _validate_report_metadata(report)
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _write_capsule_atomic(target, payload)
    except AssistantBridgeError as exc:
        raise CodingCanaryOperationalError(
            "The canary report could not be persisted safely."
        ) from exc
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mymoe coding-canary",
        description=(
            "Qualify one pinned local Cline/model cell with a disposable "
            "single-file edit and pristine independent test."
        ),
    )
    parser.add_argument(
        "--cline",
        default="cline",
        help=(
            f"Direct native Cline {SUPPORTED_CLINE_VERSION} Mach-O executable "
            "path; npm wrapper scripts are rejected."
        ),
    )
    parser.add_argument(
        "--cline-sha256",
        required=True,
        help="Expected SHA-256 of the trusted Cline executable.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8089/v1",
        help="Numeric loopback myMoE OpenAI-compatible URL.",
    )
    parser.add_argument(
        "--gateway-config",
        default="configs/moe.live.qwen3-coder-mlx.example.json",
        help="Device-only myMoE configuration served by the gateway.",
    )
    parser.add_argument(
        "--model",
        default="mymoe/coder",
        help="Pinned mymoe/<expert-id> model alias.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Cline task deadline, between 10 and 900 seconds.",
    )
    parser.add_argument("--out", help="Optional metadata-only JSON report path.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete metadata-only report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        report = run_coding_canary(
            args.cline,
            base_url=args.base_url,
            gateway_config=args.gateway_config,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            expected_cline_sha256=args.cline_sha256,
        )
        if args.out:
            write_coding_canary_report(args.out, report)
    except (
        AssistantBridgeRuntimeError,
        CodingCanaryOperationalError,
        WorkspaceSecurityError,
        OSError,
    ):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "indeterminate",
            "reason_codes": ["canary_operational_failure"],
            "diagnostic_only": True,
            "authorizes_routing": False,
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return EXIT_INDETERMINATE
    except (
        CodingCanaryContractError,
        ConfigError,
        ScopePolicyError,
        TwoPhaseConfigError,
        ValueError,
    ):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "reason_codes": ["canary_contract_invalid"],
            "diagnostic_only": True,
            "authorizes_routing": False,
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return EXIT_CONTRACT

    if args.json:
        print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    else:
        reason_codes = report.get("reason_codes", ["unclassified"])
        reason = reason_codes[0] if isinstance(reason_codes, list) else "unclassified"
        print(
            f"Local coding cell: {report['status']} ({reason}). "
            "Use --json for metadata-only evidence."
        )
    if report["status"] == "qualified":
        return EXIT_QUALIFIED
    if report["status"] == "incompatible":
        return EXIT_INCOMPATIBLE
    return EXIT_INDETERMINATE


if __name__ == "__main__":
    raise SystemExit(main())
