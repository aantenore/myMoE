from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from unittest import mock
from urllib import error

import local_moe.llama_cpp_runtime_supervisor as supervisor_module
from local_moe.llama_cpp_runtime_supervisor import (
    BoundedLoopbackJsonTransport,
    JsonHttpResponse,
    LlamaCppRuntimeSpec,
    LlamaCppRuntimeSupervisor,
    LlamaCppRuntimeSupervisorError,
    LlamaCppTransportError,
    SubprocessLauncher,
)
from local_moe.runtime_process_observer import (
    EndpointOwnershipEvidence,
    ProcessTreeEvidence,
)


ROOT_PID = 4207
EXECUTABLE_SHA256 = "a" * 64
HOST = "127.0.0.1"
PORT = 8188
_FIXTURE_ROOT = Path(__file__).resolve().parent / "runtime-supervisor-fixture"
EXECUTABLE_PATH = str(_FIXTURE_ROOT / "llama-server")
MODEL_PATH = str(_FIXTURE_ROOT / "qwen-coder.gguf")
WORKING_DIRECTORY = str(_FIXTURE_ROOT / "work")
MODEL_ID = "qwen-coder-local"


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _process_evidence(
    *,
    pid: int = ROOT_PID,
    executable_sha256: str = EXECUTABLE_SHA256,
    create_time_ns: int = 1_725_000_000_125_000_000,
) -> ProcessTreeEvidence:
    return ProcessTreeEvidence(
        root_pid=pid,
        create_time_ns=create_time_ns,
        process_count=1,
        pids_digest=_sha256_json({"pids": [pid]}),
        root_executable_sha256=executable_sha256,
        root_only=True,
    )


def _endpoint(
    pids: tuple[int, ...],
    *,
    expected_root: int | None = ROOT_PID,
    ambiguous: bool = False,
) -> EndpointOwnershipEvidence:
    return EndpointOwnershipEvidence(
        host=HOST,
        port=PORT,
        listener_pids=pids,
        listener_pids_digest=_sha256_json({"pids": list(pids)}),
        owned_by_root=(
            expected_root is not None
            and not ambiguous
            and pids == (expected_root,)
        ),
        ambiguous=ambiguous,
    )


def _spec(**overrides) -> LlamaCppRuntimeSpec:
    values = {
        "executable_path": EXECUTABLE_PATH,
        "executable_sha256": EXECUTABLE_SHA256,
        "model_path": MODEL_PATH,
        "model_id": MODEL_ID,
        "working_directory": WORKING_DIRECTORY,
        "host": HOST,
        "port": PORT,
        "sleep_idle_seconds": 45,
        "startup_timeout_seconds": 1.0,
        "poll_interval_seconds": 0.01,
    }
    values.update(overrides)
    return LlamaCppRuntimeSpec(**values)


class _FakeProcess:
    def __init__(self, pid: int = ROOT_PID) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.returncode is None:
            raise TimeoutError
        return self.returncode


class _FakeLauncher:
    def __init__(self, process: _FakeProcess | None = None) -> None:
        self.process = process or _FakeProcess()
        self.calls: list[tuple[tuple[str, ...], dict[str, str], str]] = []

    def launch(self, argv, *, environment, working_directory):
        self.calls.append((tuple(argv), dict(environment), working_directory))
        return self.process


class _FakeObserver:
    def __init__(
        self,
        endpoints: list[EndpointOwnershipEvidence],
        *,
        process_evidence: ProcessTreeEvidence | None = None,
        later_process_evidence: ProcessTreeEvidence | None = None,
    ) -> None:
        self.endpoints = list(endpoints)
        self.last_endpoint = self.endpoints[-1]
        self.process_evidence = process_evidence or _process_evidence()
        self.later_process_evidence = later_process_evidence
        self.process_calls = 0
        self.endpoint_calls: list[tuple[str, int, int | None]] = []

    def observe_process_tree(self, root_pid: int) -> ProcessTreeEvidence:
        self.process_calls += 1
        if self.process_calls > 1 and self.later_process_evidence is not None:
            return self.later_process_evidence
        return self.process_evidence

    def observe_endpoint_ownership(self, *, host: str, port: int, root_pid: int | None):
        self.endpoint_calls.append((host, port, root_pid))
        if self.endpoints:
            self.last_endpoint = self.endpoints.pop(0)
        return self.last_endpoint


class _FakeTransport:
    def __init__(self, responses: dict[str, JsonHttpResponse] | None = None) -> None:
        self.responses = responses or _ready_responses()
        self.calls: list[tuple[str, int, str, float, int]] = []

    def get_json(
        self,
        *,
        host: str,
        port: int,
        path: str,
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> JsonHttpResponse:
        self.calls.append((host, port, path, timeout_seconds, maximum_bytes))
        return self.responses[path]


class _SequencedTransport(_FakeTransport):
    def __init__(self, rounds: list[dict[str, JsonHttpResponse]]) -> None:
        super().__init__(rounds[-1])
        self.rounds = rounds
        self.round_index = 0

    def get_json(
        self,
        *,
        host: str,
        port: int,
        path: str,
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> JsonHttpResponse:
        self.calls.append((host, port, path, timeout_seconds, maximum_bytes))
        response = self.rounds[min(self.round_index, len(self.rounds) - 1)][path]
        if path == "/v1/models":
            self.round_index += 1
        return response


def _ready_responses() -> dict[str, JsonHttpResponse]:
    return {
        "/health": JsonHttpResponse(200, {"status": "ok"}),
        "/props": JsonHttpResponse(
            200,
            {
                "model_path": MODEL_PATH,
                "build_info": "b9000-deadbeef",
                "is_sleeping": False,
            },
        ),
        "/v1/models": JsonHttpResponse(
            200,
            {"object": "list", "data": [{"id": MODEL_ID}]},
        ),
    }


class LlamaCppRuntimeSupervisorTests(unittest.TestCase):
    def test_injected_spec_fixture_paths_are_host_native_and_absolute(self) -> None:
        spec = _spec()

        self.assertTrue(Path(spec.executable_path).is_absolute())
        self.assertTrue(Path(spec.model_path).is_absolute())
        self.assertTrue(Path(spec.working_directory).is_absolute())
        self.assertEqual(spec.executable_path, EXECUTABLE_PATH)
        self.assertEqual(spec.model_path, MODEL_PATH)
        self.assertEqual(spec.working_directory, WORKING_DIRECTORY)

    def test_start_requires_owned_listener_and_three_bounded_get_probes(self) -> None:
        process = _FakeProcess()
        launcher = _FakeLauncher(process)
        observer = _FakeObserver([_endpoint((), expected_root=None), _endpoint((ROOT_PID,))])
        transport = _FakeTransport()
        observed = []
        launched = []
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=observer,
            launcher=launcher,
            transport=transport,
            environment={
                "PATH": "/usr/bin",
                "HTTP_PROXY": "http://proxy.invalid",
                "LLAMA_ARG_MODELS_AUTOLOAD": "1",
                "LD_PRELOAD": "/tmp/unbound-library.so",
            },
        )

        evidence = supervisor.start(
            on_process_launched=lambda: launched.append(True),
            on_process_observed=observed.append,
        )

        self.assertTrue(evidence.application.ready)
        self.assertEqual(evidence.application.model_ids, (MODEL_ID,))
        self.assertEqual(evidence.process.root_pid, ROOT_PID)
        self.assertEqual(evidence.endpoint.listener_pids, (ROOT_PID,))
        self.assertEqual(
            evidence.content_payload()["process_tree_sha256"],
            evidence.process.digest,
        )
        self.assertEqual(
            evidence.content_payload()["endpoint_evidence_sha256"],
            evidence.endpoint.digest,
        )
        self.assertEqual(evidence.payload()["digest"], evidence.digest)
        self.assertNotIn(MODEL_PATH, json.dumps(evidence.payload()))
        self.assertTrue(supervisor.owns_process)
        self.assertEqual(launched, [True])
        self.assertEqual(observed, [evidence.process])
        self.assertEqual(
            [call[2] for call in transport.calls],
            ["/health", "/props", "/v1/models"],
        )
        argv, environment, working_directory = launcher.calls[0]
        self.assertEqual(argv[0], EXECUTABLE_PATH)
        self.assertIn("-m", argv)
        self.assertIn(MODEL_PATH, argv)
        self.assertIn("--offline", argv)
        self.assertIn("--no-ui", argv)
        self.assertIn("--no-agent", argv)
        self.assertNotIn("--models-preset", argv)
        self.assertNotIn("--hf-repo", argv)
        self.assertEqual(
            environment,
            {
                "DO_NOT_TRACK": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "LANG": "C",
                "LC_ALL": "C",
                "TRANSFORMERS_OFFLINE": "1",
                "TZ": "UTC",
            },
        )
        self.assertNotIn("PATH", environment)
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertNotIn("LLAMA_ARG_MODELS_AUTOLOAD", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(working_directory, WORKING_DIRECTORY)

    def test_refuses_preexisting_listener_instead_of_attaching(self) -> None:
        launcher = _FakeLauncher()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver([_endpoint((9911,), expected_root=None)]),
            launcher=launcher,
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "endpoint_in_use")
        self.assertEqual(launcher.calls, [])
        self.assertFalse(supervisor.owns_process)

    def test_refuses_ambiguous_preexisting_listener(self) -> None:
        launcher = _FakeLauncher()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [_endpoint((9911, 9912), expected_root=None, ambiguous=True)]
            ),
            launcher=launcher,
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "endpoint_in_use")
        self.assertEqual(launcher.calls, [])

    def test_executable_mismatch_fails_closed_and_cleans_root(self) -> None:
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [_endpoint((), expected_root=None)],
                process_evidence=_process_evidence(executable_sha256="b" * 64),
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "executable_identity_mismatch")
        self.assertEqual(process.terminate_calls, 1)
        self.assertFalse(supervisor.owns_process)

    def test_listener_owned_by_another_pid_fails_closed(self) -> None:
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [
                    _endpoint((), expected_root=None),
                    _endpoint((9911,), expected_root=ROOT_PID),
                    _endpoint((), expected_root=ROOT_PID),
                ]
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "endpoint_owner_mismatch")
        self.assertEqual(process.terminate_calls, 1)

    def test_model_identity_mismatch_is_terminal_and_cleans_root(self) -> None:
        responses = _ready_responses()
        responses["/v1/models"] = JsonHttpResponse(
            200, {"data": [{"id": "different-model"}]}
        )
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [
                    _endpoint((), expected_root=None),
                    _endpoint((ROOT_PID,)),
                    _endpoint((), expected_root=ROOT_PID),
                ]
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(responses),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "application_identity_mismatch")
        self.assertIn("model_id_mismatch", raised.exception.reason_codes)
        self.assertEqual(process.terminate_calls, 1)

    def test_start_treats_loading_schemas_as_transient_until_health_is_ready(self) -> None:
        loading = {
            "/health": JsonHttpResponse(503, {"status": "loading model"}),
            "/props": JsonHttpResponse(503, {"error": "model loading"}),
            "/v1/models": JsonHttpResponse(503, {"error": "model loading"}),
        }
        transport = _SequencedTransport([loading, _ready_responses()])
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [_endpoint((), expected_root=None), _endpoint((ROOT_PID,))]
            ),
            launcher=_FakeLauncher(),
            transport=transport,
        )

        evidence = supervisor.start()

        self.assertTrue(evidence.application.ready)
        self.assertEqual(evidence.application.model_ids, (MODEL_ID,))
        self.assertEqual(len(transport.calls), 6)

    def test_inspection_never_attaches_and_detects_pid_reuse(self) -> None:
        process = _FakeProcess()
        observer = _FakeObserver(
            [_endpoint((), expected_root=None), _endpoint((ROOT_PID,)), _endpoint((ROOT_PID,))],
        )
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=observer,
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )

        supervisor.start()
        observer.later_process_evidence = _process_evidence(
            create_time_ns=1_725_000_000_999_000_000
        )
        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.inspect()

        self.assertEqual(raised.exception.code, "process_identity_changed")

        detached = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver([_endpoint((ROOT_PID,))]),
            launcher=_FakeLauncher(),
            transport=_FakeTransport(),
        )
        with self.assertRaises(LlamaCppRuntimeSupervisorError) as not_owned:
            detached.inspect()
        self.assertEqual(not_owned.exception.code, "process_not_owned")

    def test_inspection_fails_closed_when_application_is_no_longer_ready(self) -> None:
        process = _FakeProcess()
        observer = _FakeObserver(
            [
                _endpoint((), expected_root=None),
                _endpoint((ROOT_PID,)),
                _endpoint((ROOT_PID,)),
                _endpoint((ROOT_PID,)),
            ]
        )
        transport = _FakeTransport()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=observer,
            launcher=_FakeLauncher(process),
            transport=transport,
        )
        supervisor.start()
        transport.responses["/health"] = JsonHttpResponse(
            503, {"status": "loading model"}
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.inspect()
        self.assertEqual(raised.exception.code, "application_not_ready")
        self.assertIn("health_not_ready", raised.exception.reason_codes)

    def test_stop_terminates_only_owned_root_and_is_idempotent(self) -> None:
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [
                    _endpoint((), expected_root=None),
                    _endpoint((ROOT_PID,)),
                    _endpoint((ROOT_PID,)),
                    _endpoint((), expected_root=ROOT_PID),
                ]
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )
        supervisor.start()

        supervisor.stop()
        supervisor.stop()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertFalse(supervisor.owns_process)

    def test_listener_substitution_during_get_probes_never_becomes_ready(self) -> None:
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [
                    _endpoint((), expected_root=None),
                    _endpoint((ROOT_PID,)),
                    _endpoint((9911,), expected_root=ROOT_PID),
                    _endpoint((), expected_root=ROOT_PID),
                ]
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
            supervisor.start()

        self.assertEqual(raised.exception.code, "endpoint_owner_mismatch")
        self.assertEqual(process.terminate_calls, 1)
        self.assertFalse(supervisor.owns_process)

    def test_unverified_cleanup_is_sticky_and_blocks_restart(self) -> None:
        process = _FakeProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver(
                [
                    _endpoint((), expected_root=None),
                    _endpoint((9911,), expected_root=ROOT_PID),
                ]
            ),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as failed:
            supervisor.start()
        self.assertEqual(failed.exception.code, "cleanup_unverified")
        self.assertTrue(supervisor.cleanup_unknown)
        self.assertTrue(supervisor.owns_process)

        with self.assertRaises(LlamaCppRuntimeSupervisorError) as restarted:
            supervisor.start()
        self.assertEqual(restarted.exception.code, "cleanup_unverified")

    def test_spec_rejects_router_download_and_listener_overrides(self) -> None:
        forbidden = (
            ("--models-preset", "/tmp/router.ini"),
            ("--models-max=2",),
            ("--hf-repo", "owner/model"),
            ("--hf-repo-draft", "owner/draft"),
            ("--fim-qwen-7b-default",),
            ("--model-url=https://example.test/model.gguf",),
            ("--host", "127.0.0.2"),
            ("--reuse-port",),
            ("--tools", "all"),
        )
        for arguments in forbidden:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, "permits only"
            ):
                _spec(extra_args=arguments)

        self.assertIn(
            "--n-gpu-layers",
            _spec(extra_args=("--n-gpu-layers", "12")).argv(),
        )

        for host in ("localhost", "0.0.0.0", "10.0.0.1"):
            with self.subTest(host=host), self.assertRaisesRegex(
                ValueError, "numeric loopback"
            ):
                _spec(host=host)

        with self.assertRaisesRegex(ValueError, "must be 'off'"):
            _spec(fit_mode="on")

    @unittest.skipUnless(os.name == "posix", "process-bound v1 is POSIX-only")
    def test_production_launcher_uses_exact_cwd_and_new_posix_session(self) -> None:
        process = _FakeProcess()
        with mock.patch.object(
            supervisor_module.subprocess, "Popen", return_value=process
        ) as popen:
            launched = SubprocessLauncher().launch(
                ("/absolute/llama-server", "--offline"),
                environment={"PATH": "/usr/bin"},
                working_directory="/absolute/work",
            )

        self.assertIs(launched, process)
        self.assertEqual(popen.call_args.args[0][0], "/absolute/llama-server")
        self.assertEqual(popen.call_args.kwargs["cwd"], "/absolute/work")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertFalse(popen.call_args.kwargs["shell"])

    @unittest.skipIf(os.name == "posix", "non-POSIX regression")
    def test_production_launcher_fails_closed_before_spawn_on_non_posix(self) -> None:
        with mock.patch.object(supervisor_module.subprocess, "Popen") as popen:
            with self.assertRaises(LlamaCppRuntimeSupervisorError) as raised:
                SubprocessLauncher().launch(
                    (EXECUTABLE_PATH, "--offline"),
                    environment={},
                    working_directory=WORKING_DIRECTORY,
                )

        self.assertEqual(raised.exception.code, "platform_unsupported")
        popen.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "process-bound v1 is POSIX-only")
    def test_posix_group_cleanup_escalates_from_term_to_kill(self) -> None:
        class _GroupProcess(_FakeProcess):
            def wait(self, timeout: float | None = None) -> int:
                self.wait_calls.append(timeout)
                if len(self.wait_calls) == 1:
                    raise supervisor_module.subprocess.TimeoutExpired(
                        "llama-server", timeout
                    )
                self.returncode = -9
                return self.returncode

        process = _GroupProcess()
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver([_endpoint((), expected_root=None)]),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )
        with (
            mock.patch.object(supervisor_module.os, "getpgid", return_value=ROOT_PID),
            mock.patch.object(
                supervisor_module.os,
                "killpg",
                side_effect=(None, None, ProcessLookupError()),
            ) as killpg,
        ):
            supervisor._terminate_posix_group(process)  # noqa: SLF001

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(ROOT_PID, supervisor_module.signal.SIGTERM),
                mock.call(ROOT_PID, supervisor_module.signal.SIGKILL),
                mock.call(ROOT_PID, 0),
            ],
        )

    @unittest.skipUnless(os.name == "posix", "process-bound v1 is POSIX-only")
    def test_posix_cleanup_kills_a_surviving_owned_process_group(self) -> None:
        process = _FakeProcess()
        process.returncode = 0
        supervisor = LlamaCppRuntimeSupervisor(
            _spec(),
            observer=_FakeObserver([_endpoint((), expected_root=None)]),
            launcher=_FakeLauncher(process),
            transport=_FakeTransport(),
        )
        with (
            mock.patch.object(
                supervisor,
                "_process_group_exists",
                side_effect=(True, True, False),
            ),
            mock.patch.object(supervisor_module.os, "killpg") as killpg,
        ):
            supervisor._terminate_posix_group(process)  # noqa: SLF001

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(ROOT_PID, supervisor_module.signal.SIGTERM),
                mock.call(ROOT_PID, supervisor_module.signal.SIGKILL),
            ],
        )


class _HttpResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def read(self, maximum: int) -> bytes:
        return self.body[:maximum]

    def close(self) -> None:
        self.closed = True


class _HttpOpener:
    def __init__(self, response: _HttpResponse | Exception) -> None:
        self.response = response
        self.calls = []

    def open(self, outbound, *, timeout: float):
        self.calls.append((outbound, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class BoundedLoopbackJsonTransportTests(unittest.TestCase):
    def test_default_transport_builds_an_empty_proxy_and_no_redirect_opener(self) -> None:
        sentinel = object()
        with mock.patch.object(
            supervisor_module.request,
            "build_opener",
            return_value=sentinel,
        ) as build_opener:
            transport = BoundedLoopbackJsonTransport()

        proxy_handler, redirect_handler = build_opener.call_args.args
        self.assertEqual(proxy_handler.proxies, {})
        self.assertIsInstance(redirect_handler, supervisor_module._NoRedirectHandler)
        self.assertIs(transport._opener, sentinel)

    def test_transport_uses_get_on_exact_numeric_loopback_path(self) -> None:
        response = _HttpResponse(b'{"status":"ok"}')
        opener = _HttpOpener(response)
        transport = BoundedLoopbackJsonTransport(opener=opener)

        result = transport.get_json(
            host="127.0.0.1",
            port=PORT,
            path="/health",
            timeout_seconds=0.5,
            maximum_bytes=100,
        )

        outbound, timeout = opener.calls[0]
        self.assertEqual(outbound.get_method(), "GET")
        self.assertEqual(outbound.full_url, f"http://127.0.0.1:{PORT}/health")
        self.assertEqual(timeout, 0.5)
        self.assertEqual(result.payload, {"status": "ok"})
        self.assertTrue(response.closed)

    def test_transport_rejects_redirects_forbidden_paths_and_large_bodies(self) -> None:
        redirect = error.HTTPError(
            f"http://127.0.0.1:{PORT}/health",
            302,
            "redirect",
            {"Location": "http://127.0.0.1:9999/health"},
            BytesIO(b"{}"),
        )
        with self.assertRaises(LlamaCppTransportError) as redirected:
            BoundedLoopbackJsonTransport(opener=_HttpOpener(redirect)).get_json(
                host=HOST,
                port=PORT,
                path="/health",
                timeout_seconds=0.5,
                maximum_bytes=100,
            )
        self.assertEqual(redirected.exception.code, "redirect_forbidden")

        opener = _HttpOpener(_HttpResponse(b"{}"))
        with self.assertRaises(LlamaCppTransportError) as forbidden:
            BoundedLoopbackJsonTransport(opener=opener).get_json(
                host=HOST,
                port=PORT,
                path="/models/load",
                timeout_seconds=0.5,
                maximum_bytes=100,
            )
        self.assertEqual(forbidden.exception.code, "path_forbidden")
        self.assertEqual(opener.calls, [])

        oversized_response = _HttpResponse(b"{" + (b"x" * 100) + b"}")
        with self.assertRaises(LlamaCppTransportError) as oversized:
            BoundedLoopbackJsonTransport(
                opener=_HttpOpener(oversized_response)
            ).get_json(
                host=HOST,
                port=PORT,
                path="/props",
                timeout_seconds=0.5,
                maximum_bytes=16,
            )
        self.assertEqual(oversized.exception.code, "response_too_large")

    def test_transport_rejects_non_json_duplicate_and_non_finite_payloads(self) -> None:
        cases = (
            (
                _HttpResponse(b"{}", headers={"Content-Type": "text/plain"}),
                "content_type_invalid",
            ),
            (
                _HttpResponse(b'{"status":"ok","status":"ok"}'),
                "response_invalid",
            ),
            (
                _HttpResponse(b'{"value":NaN}'),
                "response_invalid",
            ),
        )
        for response, expected_code in cases:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                LlamaCppTransportError
            ) as raised:
                BoundedLoopbackJsonTransport(
                    opener=_HttpOpener(response)
                ).get_json(
                    host=HOST,
                    port=PORT,
                    path="/health",
                    timeout_seconds=0.5,
                    maximum_bytes=100,
                )
            self.assertEqual(raised.exception.code, expected_code)


if __name__ == "__main__":
    unittest.main()
