from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Mapping, Sequence
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ProcessExecutionPolicy,
    ProcessExecutionResult,
    execute_process,
    resolve_executable,
)
from local_moe.assistant_bridge_two_phase_state import TwoPhaseConfigError
from local_moe.assistant_bridge_workspace import (
    WorkspaceSecurityError,
    snapshot_materialized,
)
from local_moe.coding_canary import (
    EXIT_INCOMPATIBLE,
    EXIT_QUALIFIED,
    _ALLOWED_COMMAND,
    _AI_SDK_WARNING_LINE,
    _BROKEN_SOURCE,
    _FIXED_SOURCE,
    _PRE_TOOL_HOOK,
    _POLICY_PROBE,
    _PRISTINE_TEST,
    _SOURCE_NAME,
    _TEST_NAME,
    _ApprovalBroker,
    _ClineEvents,
    _GatewayBinding,
    _InferenceProxy,
    _ProxyEvidence,
    _assert_fixture_entries,
    _base_report,
    _build_macos_profile,
    _capture_gateway_models,
    _classify_completed_run,
    _cline_version,
    _fixture_change_reason,
    _initial_hook_state,
    _load_hook_gate,
    _load_gateway_binding,
    _parse_cline_events,
    _parse_loopback_endpoint,
    _read_candidate_source,
    _require_direct_macos_native_executable,
    _reattest_cline_executable,
    _run_independent_verifier,
    _snapshot_fixture,
    _validate_report_metadata,
    _validated_inference_response_status,
    _validated_upstream_content_type,
    CodingCanaryContractError,
    CodingCanaryOperationalError,
    main,
    run_coding_canary,
)


ROOT = Path(__file__).resolve().parents[1]
CODER_CONFIG = ROOT / "configs" / "moe.live.qwen3-coder-mlx.example.json"

_FAKE_DIRECT_NATIVE_CLINE_HELPER = r'''#!/usr/bin/python3
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse


def option(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def emit(event_type, tool, call_id, value=None):
    event = {
        "type": event_type,
        "contentType": "tool",
        "toolName": tool,
        "toolCallId": call_id,
    }
    if value is not None:
        event["input"] = value
    print(
        json.dumps({"type": "agent_event", "event": event}, separators=(",", ":")),
        flush=True,
    )


def require_hook_approval(hook, tool, value):
    payload = json.dumps(
        {"tool_call": {"name": tool, "input": value}},
        separators=(",", ":"),
    ).encode("utf-8")
    completed = subprocess.run(
        ["/usr/bin/python3", str(hook)],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=2,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError("pre-tool hook failed")
    prefix, raw = completed.stdout.decode("utf-8").strip().split("\t", 1)
    decision = json.loads(raw)
    if prefix != "HOOK_CONTROL" or decision.get("cancel") is not False:
        raise RuntimeError("pre-tool hook denied deterministic fixture action")


def call_model(base_url, key, model):
    parsed = urlparse(base_url)
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=2)
    try:
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": "canary"}]},
            separators=(",", ":"),
        ).encode("utf-8")
        connection.request(
            "POST",
            parsed.path.rstrip("/") + "/chat/completions",
            body=body,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise RuntimeError("deterministic model endpoint failed")
    finally:
        connection.close()


def main():
    arguments = sys.argv[1:]
    if arguments == ["--version"]:
        print("Cline CLI 3.0.46")
        return

    data_dir = Path(option("--data-dir"))
    auth_path = data_dir / "fake-direct-native-auth.json"
    if arguments[0] == "auth":
        auth_path.write_text(
            json.dumps({"base_url": option("--baseurl")}),
            encoding="utf-8",
        )
        return

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    model = option("--model")
    call_model(auth["base_url"], option("--key"), model)

    workspace = Path(option("--cwd")).resolve()
    source = (workspace / "calculator.py").resolve()
    test = (workspace / "test_calculator.py").resolve()
    hook = Path(option("--config")) / "hooks" / "PreToolUse.py"
    actions = (
        (
            "read_files",
            "read",
            {"files": [{"path": str(source)}, {"path": str(test)}]},
        ),
        (
            "editor",
            "edit",
            {
                "path": str(source),
                "old_text": "return left - right",
                "new_text": "return left + right",
            },
        ),
        (
            "run_commands",
            "test",
            {"commands": ["python3 -m unittest -q test_calculator.py"]},
        ),
    )

    for tool, call_id, value in actions:
        emit("content_start", tool, call_id, value)
        require_hook_approval(hook, tool, value)
        if tool == "read_files":
            source.read_bytes()
            test.read_bytes()
        elif tool == "editor":
            before = source.read_text(encoding="utf-8")
            if before.count("return left - right") != 1:
                raise RuntimeError("fixture source was not pristine")
            source.write_text(
                before.replace("return left - right", "return left + right"),
                encoding="utf-8",
            )
        else:
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-m",
                    "unittest",
                    "-q",
                    "test_calculator.py",
                ],
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                raise RuntimeError("fixture test failed")
        emit("content_end", tool, call_id)

    print(
        json.dumps(
            {
                "type": "run_result",
                "finishReason": "completed",
                "iterations": 3,
                "usage": {"inputTokens": 8, "outputTokens": 3},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


main()
'''


def _write_fixture(root: Path) -> None:
    (root / _SOURCE_NAME).write_bytes(_BROKEN_SOURCE)
    (root / _TEST_NAME).write_bytes(_PRISTINE_TEST)


def _request(tool: str, value: dict[str, object], request_id: str) -> dict[str, object]:
    return {
        "requestId": request_id,
        "sessionId": "session",
        "createdAt": "2026-07-20T12:00:00Z",
        "toolCallId": request_id,
        "toolName": tool,
        "input": value,
        "iteration": 1,
        "agentId": "agent",
        "conversationId": "conversation",
    }


def _hook_payload(tool: str, value: dict[str, object]) -> bytes:
    return json.dumps({"tool_call": {"name": tool, "input": value}}).encode(
        "utf-8"
    )


def _run_hook(
    hook: Path,
    *,
    state: Path,
    source: Path,
    test: Path,
    tool: str,
    value: dict[str, object],
) -> dict[str, object]:
    environment = {
        **os.environ,
        "MYMOE_CANARY_HOOK_STATE": str(state),
        "MYMOE_CANARY_SOURCE": str(source),
        "MYMOE_CANARY_TEST": str(test),
    }
    completed = subprocess.run(
        [sys.executable, str(hook)],
        input=_hook_payload(tool, value),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=5,
        env=environment,
    )
    prefix, payload = completed.stdout.decode("utf-8").strip().split("\t", 1)
    if prefix != "HOOK_CONTROL":
        raise AssertionError(f"unexpected hook control prefix: {prefix}")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise AssertionError("hook control payload must be an object")
    return result


def _agent_tool_event(
    event_type: str,
    tool: str,
    call_id: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "type": "agent_event",
        "event": {
            "type": event_type,
            "contentType": "tool",
            "toolName": tool,
            "toolCallId": call_id,
            **extra,
        },
    }


def _ndjson(*records: dict[str, object]) -> bytes:
    return b"\n".join(json.dumps(item).encode("utf-8") for item in records)


class CodingCanaryTests(unittest.TestCase):
    def test_endpoint_requires_numeric_ipv4_loopback_and_v1(self) -> None:
        endpoint = _parse_loopback_endpoint("http://127.0.0.1:8089/v1")
        self.assertEqual(endpoint.port, 8089)
        self.assertEqual(endpoint.base_path, "/v1")
        self.assertEqual(
            _parse_loopback_endpoint("http://127.0.0.1:8089").base_url,
            "http://127.0.0.1:8089/v1",
        )

        invalid = (
            "https://127.0.0.1:8089/v1",
            "http://localhost:8089/v1",
            "http://192.168.1.2:8089/v1",
            "http://127.0.0.1:0/v1",
            "http://127.0.0.1:65536/v1",
            "http://127.0.0.1:08089/v1",
            "http://127.0.0.1:8089/other",
            "http://127.0.0.1:8089/v1/",
            "http://user@127.0.0.1:8089/v1",
            "http://127.0.0.1:8089/v1?debug=true",
            " http://127.0.0.1:8089/v1",
            "http://127.0.0.1:8089/v1 ",
            "http://127.0.0.1:8089/v1\n",
            "http://127.0.0.1:\t8089/v1",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(
                CodingCanaryContractError
            ):
                _parse_loopback_endpoint(value)

    def test_gateway_binding_requires_pinned_device_only_expert(self) -> None:
        binding = _load_gateway_binding(
            CODER_CONFIG,
            model="mymoe/coder",
            endpoint=_parse_loopback_endpoint("http://127.0.0.1:8089/v1"),
        )
        self.assertEqual(binding.expert_id, "coder")
        self.assertEqual(binding.scope, "device_only")
        self.assertEqual(binding.transport, "direct_local")

        with self.assertRaises(CodingCanaryContractError):
            _load_gateway_binding(
                CODER_CONFIG,
                model="mymoe",
                endpoint=_parse_loopback_endpoint("http://127.0.0.1:8089/v1"),
            )
        with self.assertRaises(CodingCanaryContractError):
            _load_gateway_binding(
                CODER_CONFIG,
                model="mymoe/missing",
                endpoint=_parse_loopback_endpoint("http://127.0.0.1:8089/v1"),
            )

    def test_live_gateway_requires_exact_full_runtime_config_digest(self) -> None:
        endpoint = _parse_loopback_endpoint("http://127.0.0.1:8089/v1")
        binding = _load_gateway_binding(
            CODER_CONFIG,
            model="mymoe/coder",
            endpoint=endpoint,
        )
        declared = json.loads(CODER_CONFIG.read_text(encoding="utf-8"))
        model_payload = {
            "data": [
                {
                    "id": "mymoe/coder",
                    "mymoe": {
                        "selection": "pinned",
                        "expert_id": "coder",
                        "upstream_model": declared["experts"][0]["model"],
                        "execution_scope": "device_only",
                        "execution_transport": "direct_local",
                        "eligible": True,
                    },
                }
            ]
        }
        with patch(
            "local_moe.coding_canary._gateway_get_json",
            side_effect=[
                model_payload,
                {"runtime_config_sha256": "0" * 64},
            ],
        ), self.assertRaises(CodingCanaryOperationalError):
            _capture_gateway_models(
                endpoint,
                expected_model="mymoe/coder",
                binding=binding,
            )

        with patch(
            "local_moe.coding_canary._gateway_get_json",
            side_effect=[
                model_payload,
                {"runtime_config_sha256": binding.runtime_config_sha256},
            ],
        ):
            evidence = _capture_gateway_models(
                endpoint,
                expected_model="mymoe/coder",
                binding=binding,
            )
        self.assertEqual(
            evidence["runtime_config_sha256"],
            binding.runtime_config_sha256,
        )

        duplicate_payload = {
            "data": [model_payload["data"][0], model_payload["data"][0]]
        }
        with patch(
            "local_moe.coding_canary._gateway_get_json",
            return_value=duplicate_payload,
        ), self.assertRaises(CodingCanaryOperationalError):
            _capture_gateway_models(
                endpoint,
                expected_model="mymoe/coder",
                binding=binding,
            )

    def test_approval_broker_allows_only_exact_disposable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            approvals = root / "approvals"
            workspace.mkdir()
            approvals.mkdir()
            _write_fixture(workspace)
            broker = _ApprovalBroker(approvals, workspace=workspace)
            source = str((workspace / _SOURCE_NAME).resolve())
            test = str((workspace / _TEST_NAME).resolve())
            requests = (
                (
                    "read",
                    _request(
                        "read_files",
                        {"files": [{"path": source}, {"path": test}]},
                        "read",
                    ),
                ),
                (
                    "edit",
                    _request(
                        "editor",
                        {
                            "path": source,
                            "old_text": "return left - right",
                            "new_text": "return left + right",
                        },
                        "edit",
                    ),
                ),
                (
                    "command",
                    _request(
                        "run_commands",
                        {"commands": [_ALLOWED_COMMAND]},
                        "command",
                    ),
                ),
            )
            for request_id, payload in requests:
                path = approvals / f"session.request.{request_id}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                broker._poll_once()
                decision = json.loads(
                    (approvals / f"session.decision.{request_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(decision["approved"])

            self.assertTrue(broker.contract_complete)
            self.assertEqual(
                broker.evidence.approved,
                Counter({"read_files": 1, "editor": 1, "run_commands": 1}),
            )

            denied = _request(
                "run_commands",
                {"commands": ["git status"]},
                "denied",
            )
            (approvals / "session.request.denied.json").write_text(
                json.dumps(denied), encoding="utf-8"
            )
            broker._poll_once()
            decision = json.loads(
                (approvals / "session.decision.denied.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(decision["approved"])
            self.assertEqual(broker.evidence.denied["run_commands"], 1)

    def test_approval_broker_rejects_test_edits_and_unknown_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            approvals = root / "approvals"
            workspace.mkdir()
            approvals.mkdir()
            _write_fixture(workspace)
            broker = _ApprovalBroker(approvals, workspace=workspace)
            test_path = str((workspace / _TEST_NAME).resolve())

            edit = broker._decide(
                _request(
                    "editor",
                    {
                        "path": test_path,
                        "old_text": "5",
                        "new_text": "-1",
                    },
                    "edit-test",
                )
            )
            unknown = broker._decide(
                _request("fetch_web_content", {"url": "http://example.com"}, "web")
            )
            self.assertFalse(edit.approved)
            self.assertFalse(unknown.approved)
            self.assertEqual(unknown.category, "other_tool")

    @unittest.skipIf(os.name == "nt", "embedded hook requires POSIX flock")
    def test_pre_tool_hook_enforces_exact_read_edit_test_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_fixture(workspace)
            source = (workspace / _SOURCE_NAME).resolve()
            test = (workspace / _TEST_NAME).resolve()
            state = root / "hook-state.json"
            hook = root / "PreToolUse.py"
            state.write_text(json.dumps(_initial_hook_state()), encoding="utf-8")
            hook.write_text(_PRE_TOOL_HOOK, encoding="utf-8")

            read = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="read_files",
                value={"files": [{"path": str(source)}, {"path": str(test)}]},
            )
            edit = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="editor",
                value={
                    "path": str(source),
                    "old_text": "return left - right",
                    "new_text": "return left + right",
                },
            )
            command = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="run_commands",
                value={"commands": [_ALLOWED_COMMAND]},
            )

            self.assertEqual(
                [read["cancel"], edit["cancel"], command["cancel"]],
                [False, False, False],
            )
            gate = _load_hook_gate(state, workspace=workspace)
            self.assertTrue(gate.contract_complete)
            self.assertEqual(
                gate.sequence,
                ("read_files", "editor", "run_commands"),
            )
            self.assertEqual(
                gate.evidence.approved,
                Counter({"read_files": 1, "editor": 1, "run_commands": 1}),
            )

    @unittest.skipIf(os.name == "nt", "embedded hook requires POSIX flock")
    def test_pre_tool_hook_denies_out_of_order_edit_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_fixture(workspace)
            source = (workspace / _SOURCE_NAME).resolve()
            test = (workspace / _TEST_NAME).resolve()
            state = root / "hook-state.json"
            hook = root / "PreToolUse.py"
            state.write_text(json.dumps(_initial_hook_state()), encoding="utf-8")
            hook.write_text(_PRE_TOOL_HOOK, encoding="utf-8")

            edit = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="editor",
                value={
                    "path": str(source),
                    "old_text": "return left - right",
                    "new_text": "return left + right",
                },
            )
            read = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="read_files",
                value={"files": [{"path": str(source)}, {"path": str(test)}]},
            )
            command = _run_hook(
                hook,
                state=state,
                source=source,
                test=test,
                tool="run_commands",
                value={"commands": [_ALLOWED_COMMAND]},
            )

            self.assertTrue(edit["cancel"])
            self.assertFalse(read["cancel"])
            self.assertTrue(command["cancel"])
            gate = _load_hook_gate(state, workspace=workspace)
            self.assertFalse(gate.contract_complete)
            self.assertEqual(gate.phase, "editing")
            self.assertEqual(gate.sequence, ("read_files",))
            self.assertEqual(
                gate.evidence.denied,
                Counter({"editor": 1, "run_commands": 1}),
            )

    def test_pre_tool_hook_denies_without_a_supported_lock_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_fixture(workspace)
            source = (workspace / _SOURCE_NAME).resolve()
            test = (workspace / _TEST_NAME).resolve()
            state = root / "hook-state.json"
            hook = root / "PreToolUse.py"
            initial = _initial_hook_state()
            state.write_text(json.dumps(initial), encoding="utf-8")
            hook.write_text(_PRE_TOOL_HOOK, encoding="utf-8")
            bootstrap = root / "run-hook-without-fcntl.py"
            bootstrap.write_text(
                "import builtins\n"
                "import runpy\n"
                "real_import = builtins.__import__\n"
                "def blocked_import(name, *args, **kwargs):\n"
                "    if name == 'fcntl':\n"
                "        raise ImportError('fcntl unavailable')\n"
                "    return real_import(name, *args, **kwargs)\n"
                "builtins.__import__ = blocked_import\n"
                f"runpy.run_path({str(hook)!r}, run_name='__main__')\n",
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "MYMOE_CANARY_HOOK_STATE": str(state),
                "MYMOE_CANARY_SOURCE": str(source),
                "MYMOE_CANARY_TEST": str(test),
            }

            completed = subprocess.run(
                [sys.executable, str(bootstrap)],
                input=_hook_payload(
                    "read_files",
                    {"files": [{"path": str(source)}, {"path": str(test)}]},
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, b"")
            prefix, payload = completed.stdout.decode("utf-8").strip().split(
                "\t", 1
            )
            self.assertEqual(prefix, "HOOK_CONTROL")
            self.assertEqual(
                json.loads(payload),
                {"cancel": True, "context": "hook_lock_unavailable"},
            )
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8")),
                initial,
            )
            self.assertFalse(Path(str(state) + ".lock").exists())
            self.assertEqual(list(root.glob("hook-state.json.next.*")), [])

    @unittest.skipIf(os.name == "nt", "embedded hook requires POSIX flock")
    def test_pre_tool_hook_keeps_a_persisted_decision_on_close_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_fixture(workspace)
            source = (workspace / _SOURCE_NAME).resolve()
            test = (workspace / _TEST_NAME).resolve()
            state = root / "hook-state.json"
            hook = root / "PreToolUse.py"
            state.write_text(json.dumps(_initial_hook_state()), encoding="utf-8")
            hook.write_text(_PRE_TOOL_HOOK, encoding="utf-8")
            bootstrap = root / "run-hook-with-close-error.py"
            injection_marker = root / "close-error-injected"
            bootstrap.write_text(
                "import os\n"
                "import runpy\n"
                "real_close = os.close\n"
                "close_calls = 0\n"
                "def close_with_error(descriptor):\n"
                "    global close_calls\n"
                "    close_calls += 1\n"
                "    real_close(descriptor)\n"
                "    if close_calls == 3:\n"
                f"        with open({str(injection_marker)!r}, 'w', encoding='utf-8') as marker:\n"
                "            marker.write('injected\\n')\n"
                "        raise OSError('reported close failure')\n"
                "os.close = close_with_error\n"
                f"runpy.run_path({str(hook)!r}, run_name='__main__')\n",
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "MYMOE_CANARY_HOOK_STATE": str(state),
                "MYMOE_CANARY_SOURCE": str(source),
                "MYMOE_CANARY_TEST": str(test),
            }

            completed = subprocess.run(
                [sys.executable, str(bootstrap)],
                input=_hook_payload(
                    "read_files",
                    {"files": [{"path": str(source)}, {"path": str(test)}]},
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, b"")
            self.assertEqual(
                injection_marker.read_text(encoding="utf-8"),
                "injected\n",
            )
            prefix, payload = completed.stdout.decode("utf-8").strip().split(
                "\t", 1
            )
            self.assertEqual(prefix, "HOOK_CONTROL")
            self.assertEqual(
                json.loads(payload),
                {"cancel": False, "context": "allowed_fixture_read"},
            )
            persisted = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["phase"], "editing")
            self.assertEqual(persisted["sequence"], ["read_files"])
            self.assertEqual(persisted["approved"], {"read_files": 1})
            self.assertEqual(list(root.glob("hook-state.json.next.*")), [])

    def test_hook_state_read_failure_is_operational(self) -> None:
        with (
            patch(
                "local_moe.coding_canary.read_bounded_regular_file",
                side_effect=TwoPhaseConfigError("unstable state"),
            ),
            self.assertRaises(CodingCanaryOperationalError),
        ):
            _load_hook_gate(Path("unused"), workspace=Path("unused"))

    def test_policy_probe_labels_the_writable_scratch_area_honestly(self) -> None:
        self.assertIn('result["scratch_write"]', _POLICY_PROBE)
        self.assertNotIn('result["workspace_write"]', _POLICY_PROBE)

    def test_fixture_snapshot_rejects_hardlinks_and_extra_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            snapshot = _snapshot_fixture(root)
            self.assertEqual({item.path for item in snapshot}, {_SOURCE_NAME, _TEST_NAME})
            os.link(root / _SOURCE_NAME, root / "alias.py")
            with self.assertRaises(WorkspaceSecurityError):
                _snapshot_fixture(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            (root / "nested").mkdir()
            with self.assertRaises(WorkspaceSecurityError):
                _snapshot_fixture(root)

    def test_fixture_entry_check_uses_fresh_path_stat(self) -> None:
        class CachedEntry:
            def __init__(self, path: Path) -> None:
                self.path = str(path)

            def stat(self, *, follow_symlinks: bool = True) -> object:
                raise AssertionError("cached DirEntry stat must not be trusted")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            entries = [
                CachedEntry(root / _SOURCE_NAME),
                CachedEntry(root / _TEST_NAME),
            ]
            with patch(
                "local_moe.coding_canary.os.scandir",
                return_value=entries,
            ):
                _assert_fixture_entries(root)

    def test_fixture_snapshot_detects_hardlink_created_during_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            _write_fixture(workspace)
            outside_alias = root / "late-alias.py"
            calls = 0

            def snapshot_then_link(
                candidate: Path,
                policy: object,
            ) -> tuple[object, ...]:
                nonlocal calls
                result = snapshot_materialized(candidate, policy)
                calls += 1
                if calls == 1:
                    os.link(workspace / _SOURCE_NAME, outside_alias)
                return result

            with patch(
                "local_moe.coding_canary.snapshot_materialized",
                side_effect=snapshot_then_link,
            ), self.assertRaises(WorkspaceSecurityError):
                _snapshot_fixture(workspace)

            self.assertEqual(calls, 1)

    def test_change_contract_requires_exact_source_and_pristine_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            baseline = _snapshot_fixture(root)
            (root / _SOURCE_NAME).write_bytes(_FIXED_SOURCE)
            candidate = _snapshot_fixture(root)
            self.assertEqual(
                _fixture_change_reason(baseline, candidate),
                "expected_single_file_change",
            )
            (root / _TEST_NAME).write_text("pass\n", encoding="utf-8")
            changed_test = _snapshot_fixture(root)
            self.assertEqual(
                _fixture_change_reason(baseline, changed_test),
                "pristine_test_changed",
            )

    def test_candidate_source_requires_exact_fixed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / _SOURCE_NAME
            source.write_bytes(_FIXED_SOURCE)
            self.assertEqual(_read_candidate_source(source), _FIXED_SOURCE)

            source.write_bytes(_FIXED_SOURCE + b"# unexpected stable content\n")
            with self.assertRaises(CodingCanaryOperationalError):
                _read_candidate_source(source)

    def test_candidate_source_read_failure_is_operational(self) -> None:
        with (
            patch(
                "local_moe.coding_canary.read_bounded_regular_file",
                side_effect=TwoPhaseConfigError("unstable candidate"),
            ),
            self.assertRaises(CodingCanaryOperationalError),
        ):
            _read_candidate_source(Path("unused"))

    def test_cline_reattest_failure_is_operational(self) -> None:
        with (
            patch(
                "local_moe.coding_canary.resolve_executable",
                side_effect=AssistantBridgeRuntimeError("identity drift"),
            ),
            self.assertRaises(CodingCanaryOperationalError),
        ):
            _reattest_cline_executable("/pinned/cline", environment={})

    def test_ndjson_parser_keeps_only_bounded_metadata(self) -> None:
        records = (
            {
                "ts": "now",
                "type": "agent_event",
                "event": {
                    "type": "content_start",
                    "contentType": "tool",
                    "toolName": "editor",
                    "toolCallId": "one",
                    "input": {"new_text": "private source"},
                },
            },
            {
                "ts": "now",
                "type": "agent_event",
                "event": {
                    "type": "content_end",
                    "contentType": "tool",
                    "toolName": "editor",
                    "toolCallId": "one",
                    "output": "private source",
                },
            },
            {
                "ts": "now",
                "type": "run_result",
                "finishReason": "completed",
                "iterations": 2,
                "usage": {"inputTokens": 10, "outputTokens": 2},
                "text": "private source",
            },
        )
        raw = b"\n".join(json.dumps(item).encode("utf-8") for item in records)
        parsed = _parse_cline_events(raw, truncated=False)

        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.tool_starts, Counter({"editor": 1}))
        self.assertEqual(parsed.tool_ends, Counter({"editor": 1}))
        self.assertEqual(parsed.usage, {"inputTokens": 10, "outputTokens": 2})
        self.assertNotIn("private source", json.dumps(parsed.payload()))
        self.assertFalse(_parse_cline_events(raw, truncated=True).valid)

    def test_ndjson_parser_accepts_only_the_exact_known_ai_sdk_banner(self) -> None:
        run_result = _ndjson(
            {
                "type": "run_result",
                "finishReason": "completed",
                "iterations": 1,
            }
        )
        parsed = _parse_cline_events(
            _AI_SDK_WARNING_LINE + b"\n" + run_result,
            truncated=False,
        )
        self.assertTrue(parsed.valid)
        self.assertEqual(
            parsed.ignored_known_noise,
            Counter({"ai_sdk_warning_banner": 1}),
        )
        self.assertFalse(
            _parse_cline_events(
                _AI_SDK_WARNING_LINE + b" altered\n" + run_result,
                truncated=False,
            ).valid
        )
        self.assertFalse(
            _parse_cline_events(
                _AI_SDK_WARNING_LINE
                + b"\n"
                + _AI_SDK_WARNING_LINE
                + b"\n"
                + run_result,
                truncated=False,
            ).valid
        )

    def test_ndjson_parser_requires_correlated_tool_lifecycle(self) -> None:
        run_result = {
            "type": "run_result",
            "finishReason": "completed",
            "iterations": 3,
            "usage": {"inputTokens": 10, "outputTokens": 2},
        }
        valid_raw = _ndjson(
            _agent_tool_event("content_start", "read_files", "read"),
            _agent_tool_event("content_end", "read_files", "read"),
            _agent_tool_event("content_start", "editor", "edit"),
            _agent_tool_event("content_end", "editor", "edit"),
            _agent_tool_event("content_start", "run_commands", "test"),
            _agent_tool_event("content_end", "run_commands", "test"),
            run_result,
        )
        parsed = _parse_cline_events(valid_raw, truncated=False)
        self.assertTrue(parsed.valid)
        self.assertTrue(parsed.lifecycle_valid)
        self.assertEqual(
            parsed.tool_sequence,
            ("read_files", "editor", "run_commands"),
        )

        invalid_streams = {
            "end_without_start": _ndjson(
                _agent_tool_event("content_end", "editor", "missing"),
                run_result,
            ),
            "mismatched_tool": _ndjson(
                _agent_tool_event("content_start", "editor", "one"),
                _agent_tool_event("content_end", "read_files", "one"),
                run_result,
            ),
            "mismatched_call_id": _ndjson(
                _agent_tool_event("content_start", "editor", "one"),
                _agent_tool_event("content_end", "editor", "two"),
                run_result,
            ),
            "duplicate_start": _ndjson(
                _agent_tool_event("content_start", "editor", "one"),
                _agent_tool_event("content_start", "editor", "one"),
                _agent_tool_event("content_end", "editor", "one"),
                run_result,
            ),
            "reused_completed_call": _ndjson(
                _agent_tool_event("content_start", "editor", "one"),
                _agent_tool_event("content_end", "editor", "one"),
                _agent_tool_event("content_start", "editor", "one"),
                run_result,
            ),
            "unfinished_call": _ndjson(
                _agent_tool_event("content_start", "editor", "one"),
                run_result,
            ),
            "duplicate_run_result": _ndjson(run_result, run_result),
        }
        for reason, raw in invalid_streams.items():
            with self.subTest(reason=reason):
                self.assertFalse(_parse_cline_events(raw, truncated=False).valid)

    def test_ndjson_parser_validates_exact_tool_inputs_without_retaining_them(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _write_fixture(workspace)
            source = str((workspace / _SOURCE_NAME).resolve())
            test = str((workspace / _TEST_NAME).resolve())
            run_result = {
                "type": "run_result",
                "finishReason": "completed",
                "iterations": 3,
            }
            records = (
                _agent_tool_event(
                    "content_start",
                    "read_files",
                    "read",
                    input={"files": [{"path": source}, {"path": test}]},
                ),
                _agent_tool_event("content_end", "read_files", "read"),
                _agent_tool_event(
                    "content_start",
                    "editor",
                    "edit",
                    input={
                        "path": source,
                        "old_text": "return left - right",
                        "new_text": "return left + right",
                    },
                ),
                _agent_tool_event("content_end", "editor", "edit"),
                _agent_tool_event(
                    "content_start",
                    "run_commands",
                    "test",
                    input={"commands": [_ALLOWED_COMMAND]},
                ),
                _agent_tool_event("content_end", "run_commands", "test"),
                run_result,
            )
            parsed = _parse_cline_events(
                _ndjson(*records),
                truncated=False,
                workspace=workspace,
            )
            self.assertEqual(parsed.tool_input_contract, "complete")
            self.assertEqual(len(parsed.tool_input_fingerprint_sha256), 64)
            self.assertNotIn(source, json.dumps(parsed.payload()))
            self.assertNotIn(_ALLOWED_COMMAND, json.dumps(parsed.payload()))

            malicious = list(records)
            malicious[4] = _agent_tool_event(
                "content_start",
                "run_commands",
                "test",
                input={"commands": ["git status"]},
            )
            rejected = _parse_cline_events(
                _ndjson(*malicious),
                truncated=False,
                workspace=workspace,
            )
            self.assertEqual(rejected.tool_input_contract, "invalid")

    def test_tri_state_classification_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            approvals = root / "approvals"
            workspace.mkdir()
            approvals.mkdir()
            _write_fixture(workspace)
            broker = _ApprovalBroker(approvals, workspace=workspace)
            broker._approved_read_paths = {
                str((workspace / _SOURCE_NAME).resolve()),
                str((workspace / _TEST_NAME).resolve()),
            }
            broker.evidence.approved.update(
                {"read_files": 1, "editor": 1, "run_commands": 1}
            )
            broker.evidence.sequence.extend(
                ("read_files", "editor", "run_commands")
            )
            events = _ClineEvents(
                valid=True,
                run_results=1,
                finish_reason="completed",
                iterations=2,
                usage={},
                tool_starts=Counter(broker.evidence.approved),
                tool_ends=Counter(broker.evidence.approved),
                tool_errors=0,
                record_count=7,
                tool_sequence=tuple(broker.evidence.sequence),
                lifecycle_valid=True,
                tool_input_contract="complete",
                tool_input_fingerprint_sha256="a" * 64,
            )
            proxy = _ProxyEvidence(requests=Counter({"chat_completions": 1}))

            status, _ = _classify_completed_run(
                broker=broker,
                events=events,
                change_reason="expected_single_file_change",
                verifier={"passed": True},
                proxy=proxy,
            )
            self.assertEqual(status, "qualified")

            status, reasons = _classify_completed_run(
                broker=broker,
                events=replace(
                    events,
                    tool_sequence=("editor", "read_files", "run_commands"),
                ),
                change_reason="expected_single_file_change",
                verifier={"passed": True},
                proxy=proxy,
            )
            self.assertEqual(status, "incompatible")
            self.assertEqual(reasons, ["tool_event_contract_mismatch"])

            status, reasons = _classify_completed_run(
                broker=broker,
                events=replace(events, finish_reason="aborted"),
                change_reason="expected_single_file_change",
                verifier={"passed": True},
                proxy=proxy,
            )
            self.assertEqual(status, "incompatible")
            self.assertEqual(reasons, ["cline_run_not_completed"])

            broker.evidence.denied["other_tool"] = 1
            status, reasons = _classify_completed_run(
                broker=broker,
                events=replace(events, valid=False, finish_reason="aborted"),
                change_reason="expected_single_file_change",
                verifier={"passed": True},
                proxy=proxy,
            )
            self.assertEqual(status, "incompatible")
            self.assertEqual(reasons, ["tool_request_denied"])

            broker.evidence.errors.append("lost request")
            status, _ = _classify_completed_run(
                broker=broker,
                events=events,
                change_reason="expected_single_file_change",
                verifier={"passed": True},
                proxy=proxy,
            )
            self.assertEqual(status, "indeterminate")

    def test_report_validator_rejects_paths_content_and_credentials(self) -> None:
        binding = _GatewayBinding(
            "a" * 64,
            "coder",
            "b" * 64,
            "c" * 64,
            "device_only",
            "direct_local",
            "d" * 64,
        )
        report = _base_report(
            status="incompatible",
            reasons=("expected_source_edit_missing",),
            model="mymoe/coder",
            gateway=binding,
            cline=None,
            cline_version=None,
        )
        _validate_report_metadata(report)
        _validate_report_metadata(
            {
                "events": {
                    "usage": {"inputTokens": 10, "outputTokens": 2},
                    "sequence": ["read_files", "editor", "run_commands"],
                }
            }
        )

        for secret in (str(Path.home()), "return left + right", "Bearer token"):
            with self.subTest(secret=secret), self.assertRaises(
                CodingCanaryOperationalError
            ):
                _validate_report_metadata({"value": secret})

        recursive_violations = (
            {"outer": [{"Authorization": "redacted"}]},
            {"outer": {"nested": ["/private/tmp/canary"]}},
            {"outer": {"nested": {"PROMPT": "redacted"}}},
            {"outer": [r"C:\private\canary"]},
        )
        for payload in recursive_violations:
            with self.subTest(payload=payload), self.assertRaises(
                CodingCanaryOperationalError
            ):
                _validate_report_metadata(payload)

    def test_macos_profile_denies_general_network_and_host_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canary = root / "canary"
            cline = root / "cline"
            canary.mkdir()
            cline.mkdir()
            profile, digest = _build_macos_profile(
                canary_root=canary,
                cline_root=cline,
                forbidden_home=Path.home(),
                broker_port=54321,
            )
        self.assertIn("(deny network*)", profile)
        self.assertIn('localhost:54321', profile)
        self.assertIn("(deny network-bind)", profile)
        self.assertEqual(len(digest), 64)

    def test_inference_proxy_pins_model_and_records_upstream_non_success(self) -> None:
        class UpstreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.server.seen_requests += 1
                payload = b'{"error":"deterministic upstream failure"}\n'
                self.send_response(self.server.response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream.response_status = 503
        upstream.seen_requests = 0
        upstream_thread = threading.Thread(
            target=upstream.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        upstream_thread.start()
        endpoint = _parse_loopback_endpoint(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        )
        evidence: _ProxyEvidence
        try:
            with _InferenceProxy(
                endpoint,
                token="canary-secret",
                expected_model="mymoe/coder",
                deadline=time.monotonic() + 5.0,
            ) as proxy:
                connection = http.client.HTTPConnection(
                    "127.0.0.1", proxy.port, timeout=2.0
                )
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {"model": "mymoe/other", "messages": []}
                    ).encode("utf-8"),
                    headers={
                        "Authorization": "Bearer canary-secret",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 400)
                connection.close()
                self.assertEqual(upstream.seen_requests, 0)

                connection = http.client.HTTPConnection(
                    "127.0.0.1", proxy.port, timeout=2.0
                )
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {"model": "mymoe/coder", "messages": []}
                    ).encode("utf-8"),
                    headers={
                        "Authorization": "Bearer canary-secret",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 503)
                connection.close()
                evidence = proxy.evidence
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=2.0)

        self.assertFalse(upstream_thread.is_alive())
        self.assertEqual(upstream.seen_requests, 1)
        self.assertEqual(evidence.violations, Counter({"model_alias_mismatch": 1}))
        self.assertEqual(evidence.requests, Counter({"chat_completions": 1}))
        self.assertEqual(evidence.responses, Counter({"status_503": 1}))
        self.assertEqual(evidence.errors, Counter({"upstream_non_success": 1}))

    def test_inference_proxy_rejects_unsafe_upstream_response_metadata(self) -> None:
        unsafe_content_types = (
            "application/json\r\nX-Injected: true",
            "application/json\rX-Injected: true",
            "application/json\nX-Injected: true",
            "application/json\x00",
            "application/json\x7f",
            "application/json\u0100",
            "not-a-media-type",
            "application/json; broken",
            "application/json; =oops",
            "application/json; x=",
            "application/json;",
            'application/json; x="unterminated',
            "x" * 1_025,
            object(),
        )
        for value in unsafe_content_types:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "Content-Type",
            ):
                _validated_upstream_content_type(value)

        self.assertEqual(
            _validated_upstream_content_type(" application/json; charset=utf-8 "),
            "application/json; charset=utf-8",
        )
        self.assertEqual(
            _validated_upstream_content_type(
                'application/problem+json; note="quoted; value\\\""'
            ),
            'application/problem+json; note="quoted; value\\\""',
        )
        self.assertEqual(
            _validated_upstream_content_type('text/plain; note="caf\xe9"'),
            'text/plain; note="caf\xe9"',
        )
        self.assertEqual(_validated_upstream_content_type(None), "application/json")
        for status in (199, 600, True, "200", None):
            with self.subTest(status=status), self.assertRaisesRegex(
                ValueError,
                "response status",
            ):
                _validated_inference_response_status(status)
        self.assertEqual(_validated_inference_response_status(200), 200)
        self.assertEqual(_validated_inference_response_status(599), 599)

        class UpstreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                payload = b'{"upstream":"sentinel"}\n'
                self.send_response(200, "UPSTREAM-SENTINEL")
                self.send_header(
                    "Content-Type",
                    "application/json\r\n X-Injected: true",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream_thread = threading.Thread(
            target=upstream.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        upstream_thread.start()
        endpoint = _parse_loopback_endpoint(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        )
        body = json.dumps({"model": "mymoe/coder", "messages": []}).encode("utf-8")
        try:
            with _InferenceProxy(
                endpoint,
                token="canary-secret",
                expected_model="mymoe/coder",
                deadline=time.monotonic() + 5.0,
            ) as proxy:
                with socket.create_connection(
                    ("127.0.0.1", proxy.port),
                    timeout=2.0,
                ) as client:
                    client.sendall(
                        (
                            "POST /v1/chat/completions HTTP/1.1\r\n"
                            "Host: 127.0.0.1\r\n"
                            "Authorization: Bearer canary-secret\r\n"
                            "Content-Type: application/json\r\n"
                            f"Content-Length: {len(body)}\r\n"
                            "Connection: close\r\n\r\n"
                        ).encode("ascii")
                        + body
                    )
                    chunks: list[bytes] = []
                    while True:
                        chunk = client.recv(65_536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                received = b"".join(chunks)
                evidence = proxy.evidence

                with patch(
                    "local_moe.coding_canary._InferenceProxyHandler._reject",
                    side_effect=OSError("synthetic reject transport failure"),
                ) as reject:
                    connection = http.client.HTTPConnection(
                        "127.0.0.1",
                        proxy.port,
                        timeout=2.0,
                    )
                    try:
                        connection.request(
                            "POST",
                            "/v1/chat/completions",
                            body=body,
                            headers={
                                "Authorization": "Bearer canary-secret",
                                "Content-Type": "application/json",
                            },
                        )
                        connection.getresponse()
                    except (OSError, http.client.HTTPException):
                        pass
                    else:  # pragma: no cover - the synthetic reject must abort.
                        self.fail("failed reject unexpectedly produced a response")
                    finally:
                        connection.close()
                    self.assertEqual(reject.call_count, 1)
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=2.0)

        self.assertTrue(received.startswith(b"HTTP/1.1 502 "), received)
        self.assertEqual(received.count(b"HTTP/1.1 "), 1)
        self.assertNotIn(b"UPSTREAM-SENTINEL", received)
        self.assertNotIn(b"X-Injected", received)
        self.assertNotIn(b'"upstream":"sentinel"', received)
        self.assertTrue(received.endswith(b"local canary broker\"}\n"), received)
        self.assertEqual(
            evidence.errors,
            Counter({"upstream_response_invalid": 1}),
        )
        self.assertEqual(
            evidence.violations,
            Counter({"upstream_response_invalid": 1}),
        )

    def test_inference_proxy_drops_bodyless_payloads_and_rejects_redirects(self) -> None:
        class UpstreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                payload = b"body-sentinel"
                self.send_response(self.server.response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        upstream.response_status = 204
        upstream_thread = threading.Thread(
            target=upstream.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        upstream_thread.start()
        endpoint = _parse_loopback_endpoint(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        )
        body = json.dumps({"model": "mymoe/coder", "messages": []}).encode("utf-8")
        try:
            with _InferenceProxy(
                endpoint,
                token="canary-secret",
                expected_model="mymoe/coder",
                deadline=time.monotonic() + 5.0,
            ) as proxy:
                for upstream_status, expected_status in ((204, 204), (205, 205), (304, 502)):
                    with self.subTest(upstream_status=upstream_status):
                        upstream.response_status = upstream_status
                        connection = http.client.HTTPConnection(
                            "127.0.0.1",
                            proxy.port,
                            timeout=2.0,
                        )
                        connection.request(
                            "POST",
                            "/v1/chat/completions",
                            body=body,
                            headers={
                                "Authorization": "Bearer canary-secret",
                                "Content-Type": "application/json",
                            },
                        )
                        response = connection.getresponse()
                        response_body = response.read()
                        self.assertEqual(response.status, expected_status)
                        self.assertNotIn(b"body-sentinel", response_body)
                        if upstream_status in {204, 205}:
                            self.assertIsNone(response.getheader("Content-Length"))
                            self.assertEqual(response_body, b"")
                        connection.close()
        finally:
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=2.0)

        self.assertFalse(upstream_thread.is_alive())

    def test_inference_proxy_waits_for_handlers_and_snapshots_evidence(self) -> None:
        request_started = threading.Event()
        release_upstream = threading.Event()

        class SlowUpstreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                request_started.set()
                if not release_upstream.wait(timeout=1.0):
                    raise AssertionError("test upstream release timed out")
                payload = b'{"ok":true}\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), SlowUpstreamHandler)
        upstream_thread = threading.Thread(
            target=upstream.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        upstream_thread.start()
        endpoint = _parse_loopback_endpoint(
            f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        )
        client_done = threading.Event()
        client_result: dict[str, object] = {}
        client_errors: list[BaseException] = []

        def request_through_proxy(port: int) -> None:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            try:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {"model": "mymoe/coder", "messages": []}
                    ).encode("utf-8"),
                    headers={
                        "Authorization": "Bearer canary-secret",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                client_result["status"] = response.status
                response.read()
            except BaseException as exc:  # pragma: no cover - assertion reports it.
                client_errors.append(exc)
            finally:
                connection.close()
                client_done.set()

        proxy = _InferenceProxy(
            endpoint,
            token="canary-secret",
            expected_model="mymoe/coder",
            deadline=time.monotonic() + 5.0,
        )
        client_thread: threading.Thread | None = None
        releaser_thread: threading.Thread | None = None
        evidence: _ProxyEvidence | None = None
        exit_elapsed = 0.0
        try:
            with proxy:
                client_thread = threading.Thread(
                    target=request_through_proxy,
                    args=(proxy.port,),
                    name="coding-canary-proxy-test-client",
                )
                client_thread.start()
                self.assertTrue(request_started.wait(timeout=1.0))

                def release_after_bound() -> None:
                    time.sleep(0.2)
                    release_upstream.set()

                releaser_thread = threading.Thread(
                    target=release_after_bound,
                    name="coding-canary-proxy-test-release",
                    daemon=True,
                )
                releaser_thread.start()
                exit_started = time.monotonic()
            exit_elapsed = time.monotonic() - exit_started
            evidence = proxy.snapshot_evidence()
        finally:
            release_upstream.set()
            if releaser_thread is not None:
                releaser_thread.join(timeout=1.0)
            if client_thread is not None:
                client_thread.join(timeout=2.0)
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(timeout=2.0)

        if evidence is None:
            self.fail("inference proxy evidence snapshot was not produced")
        self.assertGreaterEqual(exit_elapsed, 0.15)
        self.assertTrue(client_done.is_set())
        self.assertFalse(client_errors)
        self.assertEqual(client_result, {"status": 200})
        self.assertEqual(evidence.requests, Counter({"chat_completions": 1}))
        self.assertEqual(evidence.responses, Counter({"status_200": 1}))

        for bucket in ("requests", "responses", "violations", "errors"):
            snapshot_counter = getattr(evidence, bucket)
            live_counter = getattr(proxy.server.evidence, bucket)
            self.assertIsNot(snapshot_counter, live_counter)
            live_counter["late_mutation"] += 1
            self.assertNotIn("late_mutation", snapshot_counter)

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS verifier")
    def test_run_coding_canary_qualifies_deterministic_direct_native_cell(
        self,
    ) -> None:
        class GatewayHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send_json(self, payload: object) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
                with self.server.request_lock:
                    self.server.requests[self.path] += 1
                if self.path == "/v1/models":
                    binding = self.server.binding
                    self._send_json(
                        {
                            "data": [
                                {
                                    "id": "mymoe/coder",
                                    "mymoe": {
                                        "selection": "pinned",
                                        "expert_id": binding.expert_id,
                                        "upstream_model": self.server.upstream_model,
                                        "execution_scope": binding.scope,
                                        "execution_transport": binding.transport,
                                        "eligible": True,
                                    },
                                }
                            ]
                        }
                    )
                    return
                if self.path == "/api/config":
                    self._send_json(
                        {
                            "runtime_config_sha256": (
                                self.server.binding.runtime_config_sha256
                            )
                        }
                    )
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                with self.server.request_lock:
                    self.server.requests[self.path] += 1
                    self.server.models.append(request.get("model"))
                self._send_json(
                    {
                        "id": "deterministic-local-response",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            helper = root / "fake_cline_helper.py"
            helper.write_text(
                _FAKE_DIRECT_NATIVE_CLINE_HELPER,
                encoding="utf-8",
            )
            source = root / "fake_cline.c"
            executable = root / "fake-cline"
            helper_literal = json.dumps(str(helper))
            source.write_text(
                "#include <stdlib.h>\n"
                "#include <unistd.h>\n"
                "int main(int argc, char **argv) {\n"
                "    char **forwarded = calloc((size_t)argc + 2, sizeof(char *));\n"
                "    if (forwarded == NULL) return 70;\n"
                '    forwarded[0] = "/usr/bin/python3";\n'
                f"    forwarded[1] = {helper_literal};\n"
                "    for (int index = 1; index < argc; ++index) {\n"
                "        forwarded[index + 1] = argv[index];\n"
                "    }\n"
                "    forwarded[argc + 1] = NULL;\n"
                '    execv("/usr/bin/python3", forwarded);\n'
                "    free(forwarded);\n"
                "    return 71;\n"
                "}\n",
                encoding="utf-8",
            )
            compiled = subprocess.run(
                [
                    "/usr/bin/clang",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(source),
                    "-o",
                    str(executable),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            self.assertEqual(
                compiled.returncode,
                0,
                compiled.stderr.decode("utf-8", errors="replace"),
            )
            cline_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()

            gateway = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
            endpoint = _parse_loopback_endpoint(
                f"http://127.0.0.1:{gateway.server_address[1]}/v1"
            )
            binding = _load_gateway_binding(
                CODER_CONFIG,
                model="mymoe/coder",
                endpoint=endpoint,
            )
            config = json.loads(CODER_CONFIG.read_text(encoding="utf-8"))
            gateway.binding = binding
            gateway.upstream_model = config["experts"][0]["model"]
            gateway.requests = Counter()
            gateway.models = []
            gateway.request_lock = threading.Lock()
            gateway_thread = threading.Thread(
                target=gateway.serve_forever,
                kwargs={"poll_interval": 0.01},
                daemon=True,
            )
            gateway_thread.start()

            def execute_directly(
                _sandbox: object,
                *,
                profile: str,
                command: Sequence[str],
                cwd: Path,
                environment: Mapping[str, str],
                timeout_seconds: float,
                stdout_limit: int = 8 * 1024 * 1024,
            ) -> ProcessExecutionResult:
                self.assertTrue(profile.startswith("(version 1)"))
                identity = resolve_executable(command[0], env=environment)
                return execute_process(
                    identity,
                    tuple(command[1:]),
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

            output = io.StringIO()
            try:
                with (
                    patch(
                        "local_moe.coding_canary._probe_macos_profile",
                        return_value={
                            "status": "passed",
                            "checks": ["deterministic_sandbox_test_boundary"],
                        },
                    ) as probe,
                    patch(
                        "local_moe.coding_canary._execute_sandboxed",
                        side_effect=execute_directly,
                    ) as sandboxed,
                    redirect_stdout(output),
                ):
                    exit_code = main(
                        [
                            "--cline",
                            str(executable),
                            "--cline-sha256",
                            cline_sha256,
                            "--base-url",
                            endpoint.base_url,
                            "--gateway-config",
                            str(CODER_CONFIG),
                            "--model",
                            "mymoe/coder",
                            "--timeout-seconds",
                            "20",
                            "--json",
                        ]
                    )
            finally:
                gateway.shutdown()
                gateway.server_close()
                gateway_thread.join(timeout=2.0)

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, EXIT_QUALIFIED, output.getvalue())
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(
            report["reason_codes"],
            ["all_qualification_gates_passed"],
        )
        self.assertFalse(report["authorizes_routing"])
        self.assertEqual(
            report["cline"]["artifact_type"],
            "direct_macos_native_executable",
        )
        self.assertEqual(
            report["candidate"]["change"],
            "expected_single_file_change",
        )
        self.assertTrue(report["verifier"]["passed"])
        self.assertEqual(report["events"]["tool_input_contract"], "complete")
        self.assertEqual(
            report["tool_gate"]["sequence"],
            ["read_files", "editor", "run_commands"],
        )
        self.assertEqual(
            report["inference_broker"]["requests"],
            {"chat_completions": 1},
        )
        self.assertEqual(
            gateway.requests,
            Counter(
                {
                    "/v1/models": 2,
                    "/api/config": 2,
                    "/v1/chat/completions": 1,
                }
            ),
        )
        self.assertEqual(gateway.models, ["mymoe/coder"])
        probe.assert_called_once()
        self.assertEqual(sandboxed.call_count, 2)

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS sandbox-exec")
    def test_independent_verifier_uses_pristine_test_in_second_sandbox(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary) / "verification"
            passed = _run_independent_verifier(
                candidate_source=_FIXED_SOURCE,
                verification_root=root,
            )
            self.assertTrue(passed["passed"])

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary) / "verification"
            failed = _run_independent_verifier(
                candidate_source=_BROKEN_SOURCE,
                verification_root=root,
            )
            self.assertFalse(failed["passed"])

    def test_unsupported_platform_returns_indeterminate_without_running_cline(self) -> None:
        with patch.object(sys, "platform", "linux"):
            report = run_coding_canary(
                "/missing/cline",
                base_url="http://127.0.0.1:8089/v1",
                gateway_config=CODER_CONFIG,
                model="mymoe/coder",
                expected_cline_sha256="a" * 64,
            )
        self.assertEqual(report["status"], "indeterminate")
        self.assertFalse(report["authorizes_routing"])

    def test_wrong_cline_sha_is_rejected_before_any_process_runs(self) -> None:
        with (
            patch.object(sys, "platform", "darwin"),
            patch("local_moe.coding_canary.execute_process") as execute_process,
            self.assertRaises(CodingCanaryContractError),
        ):
            run_coding_canary(
                sys.executable,
                base_url="http://127.0.0.1:8089/v1",
                gateway_config=CODER_CONFIG,
                model="mymoe/coder",
                expected_cline_sha256="0" * 64,
            )

        execute_process.assert_not_called()

    def test_cline_wrapper_is_rejected_before_any_process_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary) / "cline"
            wrapper.write_text(
                "#!/usr/bin/env node\n"
                "require('child_process').spawnSync(__dirname + '/.cline', "
                "process.argv.slice(2), {stdio: 'inherit'});\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o700)
            digest = hashlib.sha256(wrapper.read_bytes()).hexdigest()
            with (
                patch.object(sys, "platform", "darwin"),
                patch("local_moe.coding_canary.execute_process") as execute_process,
                self.assertRaises(CodingCanaryContractError),
            ):
                run_coding_canary(
                    wrapper,
                    base_url="http://127.0.0.1:8089/v1",
                    gateway_config=CODER_CONFIG,
                    model="mymoe/coder",
                    expected_cline_sha256=digest,
                )

            execute_process.assert_not_called()

    def test_direct_native_cline_artifact_is_admitted_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "cline"
            executable.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 32)
            executable.chmod(0o700)
            with patch(
                "local_moe.assistant_bridge_runtime.shutil.which",
                return_value=str(executable),
            ):
                identity = resolve_executable(executable, env=os.environ)

            _require_direct_macos_native_executable(identity)

    def test_cline_version_uses_bounded_process_result(self) -> None:
        result = ProcessExecutionResult(
            code="completed",
            returncode=0,
            timed_out=False,
            stdout=b"Cline CLI 3.0.46\n",
            stderr=b"",
            stdout_bytes=17,
            stderr_bytes=0,
            stdout_sha256="a" * 64,
            stderr_sha256="b" * 64,
            stdout_truncated=False,
            stderr_truncated=False,
            stdin_bytes_written=0,
            execution_duration_ms=1,
            duration_ms=1,
            executable=None,
            environment=None,
            cleanup=None,
        )

        self.assertEqual(_cline_version(result), "3.0.46")

    def test_cli_maps_incompatible_report_to_exit_one(self) -> None:
        report = {
            "schema_version": "local-coding-canary/v1",
            "status": "incompatible",
            "reason_codes": ["expected_source_edit_missing"],
            "diagnostic_only": True,
            "authorizes_routing": False,
        }
        output = io.StringIO()
        with patch(
            "local_moe.coding_canary.run_coding_canary",
            return_value=report,
        ), redirect_stdout(output):
            code = main(["--cline-sha256", "a" * 64, "--json"])

        self.assertEqual(code, EXIT_INCOMPATIBLE)
        self.assertEqual(json.loads(output.getvalue()), report)

    def test_cli_maps_runtime_failure_to_indeterminate_without_traceback(self) -> None:
        output = io.StringIO()
        with patch(
            "local_moe.coding_canary.run_coding_canary",
            side_effect=AssistantBridgeRuntimeError("executable drift"),
        ), redirect_stdout(output):
            code = main(["--cline-sha256", "a" * 64, "--json"])

        self.assertEqual(code, 3)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "schema_version": "local-coding-canary/v1",
                "status": "indeterminate",
                "reason_codes": ["canary_operational_failure"],
                "diagnostic_only": True,
                "authorizes_routing": False,
            },
        )

    def test_primary_cli_dispatches_coding_canary(self) -> None:
        from local_moe import cli

        with (
            patch.object(sys, "argv", ["mymoe", "coding-canary", "--json"]),
            patch("local_moe.coding_canary.main", return_value=0) as canary_main,
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, 0)
        canary_main.assert_called_once_with(["--json"])


if __name__ == "__main__":
    unittest.main()
