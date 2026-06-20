from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .app_config import app_config_payload, load_app_config
from .audit import AuditLogStore, audit_log_payload, audit_prune_payload
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .chat_store import ChatSession, FileChatStore, chat_session_payload, chat_summary_payload
from .compaction import LocalCompactionProvider
from .config import load_config
from .config_profiles import discover_config_profiles
from .context import ContextBundle, ConversationTurn, MemorySnippet, build_context_bundle
from .context_policy import load_context_policy
from .data_bundle import (
    build_local_data_bundle,
    local_data_restore_payload,
    restore_local_data_bundle,
)
from .doctor import build_doctor_report, doctor_report_filename, render_doctor_report_markdown
from .evaluator import evaluate_router, load_eval_cases
from .extensions import (
    ExtensionError,
    audit_extension_registry,
    configure_extension_entry,
    create_plugin_scaffold,
    extension_configuration_templates,
    load_extension_registry,
    registry_payload,
)
from .health import check_runtime_health, runtime_health_payload
from .memory import (
    FileMemoryStore,
    memory_maintenance_payload,
    memory_prune_payload,
    memory_record_payload,
)
from .model_servers import ModelServerManager, model_server_action_payload
from .orchestrator import LocalMoE
from .performance_report import (
    build_performance_report,
    performance_report_filename,
    render_performance_report_markdown,
)
from .providers import ProviderError
from .scheduler import BackgroundCronRunner, cron_status, cron_summary_payload, run_due_jobs
from .setup_status import inspect_setup_status, setup_status_payload
from .setup_runner import run_runtime_setup, setup_run_payload
from .support_bundle import build_support_bundle, support_bundle_filename
from .tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="myMoE local web UI")
    parser.add_argument("--config")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    server = build_server(args.config or app_config.default_moe_config, args.host, args.port, app_config_path=args.app_config)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"myMoE UI listening on {url}")
    cron_runner = getattr(server, "background_cron_runner", None)
    if cron_runner is not None:
        cron_runner.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        if cron_runner is not None:
            cron_runner.stop()
        model_manager = getattr(server, "model_server_manager", None)
        if model_manager is not None:
            model_manager.close()
        server.server_close()


def build_server(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 8089,
    *,
    app_config_path: str = "configs/app.json",
) -> ThreadingHTTPServer:
    app_config = load_app_config(app_config_path)
    config = load_config(config_path)
    context_policy = load_context_policy(
        app_config.runtime.context_policy_config,
        app_config.runtime.context_policy_profile,
    )
    moe = LocalMoE(config)
    registry = _load_registry(app_config)
    audit_store = AuditLogStore(_audit_log_path(app_config))
    chat_store = FileChatStore(_chat_store_path(app_config))
    memory_store = FileMemoryStore(_memory_store_path(app_config))
    model_manager = ModelServerManager.from_config(
        config,
        preferred_backends=app_config.runtime.preferred_backends,
        work_dir=app_config.runtime.work_dir,
    )
    cron_runner = BackgroundCronRunner(
        registry.cron_jobs,
        state_path=_cron_state_path(app_config),
        poll_seconds=app_config.runtime.cron_poll_seconds,
        confirm_writes=app_config.runtime.cron_confirm_writes,
        enabled=app_config.runtime.cron_auto_run,
        registry=registry,
    )
    handler = _make_handler(
        config_path=config_path,
        app_config_path=app_config_path,
        app_config=app_config,
        config=config,
        context_policy=context_policy,
        moe=moe,
        registry=registry,
        audit_store=audit_store,
        chat_store=chat_store,
        memory_store=memory_store,
        model_manager=model_manager,
        cron_runner=cron_runner,
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.background_cron_runner = cron_runner  # type: ignore[attr-defined]
    server.model_server_manager = model_manager  # type: ignore[attr-defined]
    return server


def _make_handler(
    config_path: str,
    app_config_path: str,
    app_config: object,
    config: object,
    context_policy: object,
    moe: LocalMoE,
    registry: object,
    audit_store: AuditLogStore,
    chat_store: FileChatStore,
    memory_store: FileMemoryStore,
    model_manager: ModelServerManager,
    cron_runner: BackgroundCronRunner,
) -> type[BaseHTTPRequestHandler]:
    class MyMoEHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path in {"/", "/index.html"}:
                _send(
                    self,
                    HTTPStatus.OK,
                    _asset("index.html").encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return

            if path == "/api/config":
                _send_json(self, _config_payload(config_path, config, app_config))
                return

            if path == "/api/config/profiles":
                _send_json(
                    self,
                    discover_config_profiles(
                        active_config_path=config_path,
                        app_config=app_config,
                        app_config_path=app_config_path,
                    ),
                )
                return

            if path == "/api/doctor":
                _send_json(
                    self,
                    build_doctor_report(
                        config_path=config_path,
                        config=config,
                        app_config=app_config,
                        app_config_path=app_config_path,
                        registry=registry,
                        model_manager=model_manager,
                    ),
                )
                return

            if path == "/api/doctor/report.md":
                report = build_doctor_report(
                    config_path=config_path,
                    config=config,
                    app_config=app_config,
                    app_config_path=app_config_path,
                    registry=registry,
                    model_manager=model_manager,
                )
                _send_download(
                    self,
                    render_doctor_report_markdown(report).encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                    filename=doctor_report_filename(),
                )
                return

            if path == "/api/support-bundle":
                _send_json(
                    self,
                    build_support_bundle(
                        config_path=config_path,
                        config=config,
                        app_config=app_config,
                        app_config_path=app_config_path,
                        registry=registry,
                        model_manager=model_manager,
                    ),
                )
                return

            if path == "/api/support-bundle/download.json":
                bundle = build_support_bundle(
                    config_path=config_path,
                    config=config,
                    app_config=app_config,
                    app_config_path=app_config_path,
                    registry=registry,
                    model_manager=model_manager,
                )
                _send_download(
                    self,
                    json.dumps(bundle, indent=2).encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                    filename=support_bundle_filename(),
                )
                return

            if path == "/api/performance":
                _send_json(self, build_performance_report())
                return

            if path == "/api/performance/report.md":
                report = build_performance_report()
                _send_download(
                    self,
                    render_performance_report_markdown(report).encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                    filename=performance_report_filename(),
                )
                return

            if path == "/api/extensions":
                _send_json(self, registry_payload(registry))
                return

            if path == "/api/extensions/templates":
                _send_json(self, extension_configuration_templates())
                return

            if path == "/api/extensions/audit":
                _send_json(
                    self,
                    {
                        "audit": audit_extension_registry(registry),
                        "extensions": registry_payload(registry),
                    },
                )
                return

            if path == "/api/audit":
                query = parse_qs(parsed_url.query)
                events = audit_store.list_events(
                    limit=_query_limit(query.get("limit", ["100"])[0], default=100, maximum=500),
                    action=_optional_str(query.get("action", [""])[0]),
                    status=_optional_str(query.get("status", [""])[0]),
                )
                _send_json(self, audit_log_payload(events))
                return

            if path == "/api/runtime":
                plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
                _send_json(self, runtime_plan_payload(plan))
                return

            if path == "/api/models/processes":
                _send_json(self, model_manager.status())
                return

            if path == "/api/models/logs":
                query = parse_qs(parsed_url.query)
                _send_json(
                    self,
                    model_manager.logs(
                        expert_id=_optional_str(query.get("expert_id", [""])[0]),
                        max_lines=_bounded_int(
                            query.get("lines", ["120"])[0],
                            default=120,
                            minimum=1,
                            maximum=1000,
                        ),
                    ),
                )
                return

            if path == "/api/setup":
                _send_json(
                    self,
                    setup_status_payload(
                        inspect_setup_status(
                            config_path,
                            config,
                            app_config,
                            app_config_path=app_config_path,
                        )
                    ),
                )
                return

            if path == "/api/health":
                _send_json(self, runtime_health_payload(check_runtime_health(config)))
                return

            if path == "/api/cron":
                status = cron_status(
                    registry.cron_jobs,
                    state_path=_cron_state_path(app_config),
                )
                status["auto"] = cron_runner.status_payload()
                _send_json(
                    self,
                    status,
                )
                return

            if path == "/api/chats":
                query = _optional_str(parse_qs(parsed_url.query).get("query", [""])[0])
                sessions = chat_store.search_sessions(query) if query else chat_store.list_sessions()
                _send_json(
                    self,
                    {
                        "count": len(sessions),
                        "sessions": [chat_summary_payload(summary) for summary in sessions],
                    },
                )
                return

            if path.startswith("/api/chats/"):
                if path.endswith("/export.md"):
                    session_id = _path_tail(path[: -len("/export.md")], "/api/chats/")
                    try:
                        markdown = chat_store.export_markdown(session_id)
                    except KeyError:
                        _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                        return
                    _audit(
                        audit_store,
                        "chat.export",
                        "ok",
                        risk_class="read_only",
                        subject=session_id,
                    )
                    _send_download(
                        self,
                        markdown.encode("utf-8"),
                        content_type="text/markdown; charset=utf-8",
                        filename=f"{_safe_filename(session_id)}.md",
                    )
                    return
                session_id = _path_tail(path, "/api/chats/")
                session = chat_store.get_session(session_id)
                if session is None:
                    _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                _send_json(self, chat_session_payload(session))
                return

            if path == "/api/memory":
                query = _optional_str(parse_qs(parsed_url.query).get("query", [""])[0])
                scope = _optional_str(parse_qs(parsed_url.query).get("scope", ["default"])[0])
                if query:
                    results = memory_store.search(query, scope=scope)
                    _send_json(
                        self,
                        {
                            "count": len(results),
                            "records": [
                                memory_record_payload(record, score=score)
                                for record, score in results
                            ],
                        },
                    )
                    return
                records = memory_store.list(scope=scope)
                _send_json(
                    self,
                    {
                        "count": len(records),
                        "records": [memory_record_payload(record) for record in records],
                    },
                )
                return

            if path == "/api/memory/maintenance":
                _send_json(self, memory_maintenance_payload(memory_store.maintenance_report()))
                return

            if path == "/api/knowledge":
                scope = _optional_str(parse_qs(parsed_url.query).get("scope", ["default"])[0])
                records = [
                    record
                    for record in memory_store.list(scope=scope)
                    if record.kind == "knowledge"
                ]
                _send_json(
                    self,
                    {
                        "count": len(records),
                        "records": [memory_record_payload(record) for record in records],
                    },
                )
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            nonlocal registry
            path = urlparse(self.path).path
            if path == "/api/generate/stream":
                payload = _read_json(self)
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    _send_json(
                        self,
                        {"error": "prompt is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                session_id = _optional_str(payload.get("session_id"))
                session = None
                if session_id:
                    session = chat_store.get_session(session_id)
                    if session is None:
                        _send_json(
                            self,
                            {"error": "not_found", "message": "Chat session not found."},
                            status=HTTPStatus.NOT_FOUND,
                        )
                        return
                model_context = _build_model_context(
                    session,
                    prompt,
                    context_policy,
                    memory_store=memory_store,
                )

                _send_sse_headers(self)
                try:
                    for event in moe.generate_stream(
                        model_context["prompt"],
                        correlation_id=_optional_str(payload.get("correlation_id")),
                        route_prompt=prompt,
                    ):
                        if event.kind == "route" and event.route is not None:
                            _send_sse_event(self, "route", {"route": _route_payload(event.route)})
                            continue
                        if event.kind == "content":
                            _send_sse_event(self, "content", {"content": event.content})
                            continue
                        if event.kind == "final" and event.response is not None:
                            response_payload = _response_payload(event.response)
                            response_payload["context"] = model_context["payload"]
                            try:
                                session = chat_store.append_exchange(
                                    session_id=session_id,
                                    user_content=prompt,
                                    assistant_content=event.response.content,
                                    assistant_meta={
                                        "correlation_id": response_payload["correlation_id"],
                                        "route": response_payload["route"],
                                        "results": response_payload["results"],
                                        "errors": response_payload["errors"],
                                        "disagreement": response_payload["disagreement"],
                                        "context": response_payload["context"],
                                    },
                                )
                            except KeyError:
                                _send_sse_event(
                                    self,
                                    "error",
                                    {"error": "not_found", "message": "Chat session not found."},
                                )
                                return
                            response_payload["session_id"] = session.id
                            response_payload["session"] = chat_session_payload(session)
                            _send_sse_event(self, "final", response_payload)
                            return
                    _send_sse_event(
                        self,
                        "error",
                        {"error": "provider_error", "message": "Stream ended without a final response."},
                    )
                except ProviderError as exc:
                    _send_sse_event(
                        self,
                        "error",
                        {"error": "provider_error", "message": str(exc)},
                    )
                return

            if path == "/api/generate":
                payload = _read_json(self)
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    _send_json(
                        self,
                        {"error": "prompt is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                session_id = _optional_str(payload.get("session_id"))
                session = None
                if session_id:
                    session = chat_store.get_session(session_id)
                    if session is None:
                        _send_json(
                            self,
                            {"error": "not_found", "message": "Chat session not found."},
                            status=HTTPStatus.NOT_FOUND,
                        )
                        return
                model_context = _build_model_context(
                    session,
                    prompt,
                    context_policy,
                    memory_store=memory_store,
                )

                try:
                    response = moe.generate(
                        model_context["prompt"],
                        correlation_id=_optional_str(payload.get("correlation_id")),
                        route_prompt=prompt,
                    )
                except ProviderError as exc:
                    _send_json(
                        self,
                        {"error": "provider_error", "message": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return

                response_payload = _response_payload(response)
                response_payload["context"] = model_context["payload"]
                try:
                    session = chat_store.append_exchange(
                        session_id=session_id,
                        user_content=prompt,
                        assistant_content=response.content,
                        assistant_meta={
                            "correlation_id": response_payload["correlation_id"],
                            "route": response_payload["route"],
                            "results": response_payload["results"],
                            "errors": response_payload["errors"],
                            "disagreement": response_payload["disagreement"],
                            "context": response_payload["context"],
                        },
                    )
                except KeyError:
                    _send_json(
                        self,
                        {"error": "not_found", "message": "Chat session not found."},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                response_payload["session_id"] = session.id
                response_payload["session"] = chat_session_payload(session)
                _send_json(self, response_payload)
                return

            if path == "/api/chats":
                payload = _read_json(self)
                session = chat_store.create_session(title=_optional_str(payload.get("title")))
                _audit(
                    audit_store,
                    "chat.create",
                    "ok",
                    risk_class="write_local",
                    subject=session.id,
                )
                _send_json(self, chat_session_payload(session), status=HTTPStatus.CREATED)
                return

            if path == "/api/memory":
                payload = _read_json(self)
                text = _optional_str(payload.get("text"))
                if not text:
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "text is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "metadata must be a JSON object"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                record = memory_store.add(
                    text,
                    scope=_optional_str(payload.get("scope")) or "default",
                    kind=_optional_str(payload.get("kind")) or "fact",
                    metadata=metadata,
                    valid_from=_optional_str(payload.get("valid_from")),
                    valid_until=_optional_str(payload.get("valid_until")),
                )
                _audit(
                    audit_store,
                    "memory.add",
                    "ok",
                    risk_class="write_local",
                    subject=record.id,
                    metadata={"scope": record.scope, "kind": record.kind},
                )
                _send_json(self, memory_record_payload(record), status=HTTPStatus.CREATED)
                return

            if path == "/api/knowledge":
                payload = _read_json(self)
                if payload.get("confirm") is not True:
                    _audit(
                        audit_store,
                        "knowledge.ingest",
                        "confirmation_required",
                        risk_class="write_local",
                        metadata={"scope": _optional_str(payload.get("scope")) or "default"},
                    )
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Knowledge import requires confirm=true because it writes local memory records.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "metadata must be a JSON object"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    report = memory_store.ingest_document(
                        _required_payload_text(payload, "content"),
                        title=_required_payload_text(payload, "title"),
                        scope=_optional_str(payload.get("scope")) or "default",
                        chunk_chars=int(payload.get("chunk_chars", 1200)),
                        metadata=metadata,
                    )
                except (ValueError, TypeError) as exc:
                    _audit(
                        audit_store,
                        "knowledge.ingest",
                        "error",
                        risk_class="write_local",
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "bad_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _audit(
                    audit_store,
                    "knowledge.ingest",
                    "ok",
                    risk_class="write_local",
                    subject=report.document_id,
                    metadata={
                        "scope": report.scope,
                        "chunk_count": report.chunk_count,
                        "record_count": len(report.record_ids),
                    },
                )
                _send_json(
                    self,
                    {
                        "document_id": report.document_id,
                        "title": report.title,
                        "scope": report.scope,
                        "chunk_count": report.chunk_count,
                        "record_ids": list(report.record_ids),
                    },
                    status=HTTPStatus.CREATED,
                )
                return

            if path == "/api/data/export":
                payload = _read_json(self)
                if payload.get("confirm") is not True:
                    _audit(audit_store, "data.export", "confirmation_required", risk_class="read_only")
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Local data export requires confirm=true because it returns private chat and memory data.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                bundle = build_local_data_bundle(chat_store=chat_store, memory_store=memory_store)
                _audit(
                    audit_store,
                    "data.export",
                    "ok",
                    risk_class="read_only",
                    metadata=bundle["counts"],
                )
                _send_json(
                    self,
                    bundle,
                )
                return

            if path == "/api/data/import":
                payload = _read_json(self)
                if payload.get("confirm") is not True:
                    _audit(audit_store, "data.import", "confirmation_required", risk_class="write_local")
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Local data import requires confirm=true because it writes chat and memory data.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                bundle = payload.get("bundle", {})
                if not isinstance(bundle, dict):
                    _audit(
                        audit_store,
                        "data.import",
                        "error",
                        risk_class="write_local",
                        metadata={"message": "bundle must be a JSON object."},
                    )
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "bundle must be a JSON object."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    report = restore_local_data_bundle(
                        bundle,
                        chat_store=chat_store,
                        memory_store=memory_store,
                        mode=_optional_str(payload.get("mode")) or "merge",
                    )
                except ValueError as exc:
                    _audit(
                        audit_store,
                        "data.import",
                        "error",
                        risk_class="write_local",
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "bad_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _audit(
                    audit_store,
                    "data.import",
                    "ok",
                    risk_class="write_local",
                    metadata=local_data_restore_payload(report),
                )
                _send_json(self, local_data_restore_payload(report))
                return

            if path == "/api/memory/prune-expired":
                payload = _read_json(self)
                if payload.get("confirm") is not True:
                    _audit(
                        audit_store,
                        "memory.prune_expired",
                        "confirmation_required",
                        risk_class="write_local",
                    )
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Expired memory pruning requires confirm=true because it deletes local memory records.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                report = memory_store.prune_expired()
                _audit(
                    audit_store,
                    "memory.prune_expired",
                    "ok",
                    risk_class="write_local",
                    metadata={
                        "removed_count": report.removed_count,
                        "remaining_count": report.remaining_count,
                    },
                )
                _send_json(self, memory_prune_payload(report))
                return

            if path == "/api/audit/prune":
                payload = _read_json(self)
                keep = _bounded_int(payload.get("keep"), default=500, minimum=1, maximum=50000)
                if payload.get("confirm") is not True:
                    _audit(
                        audit_store,
                        "audit.prune",
                        "confirmation_required",
                        risk_class="write_local",
                        metadata={"keep": keep},
                    )
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Audit pruning requires confirm=true because it permanently removes older audit events.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                report = audit_store.prune(keep=keep)
                _send_json(self, audit_prune_payload(report))
                return

            if path.startswith("/api/chats/") and path.endswith("/compact"):
                session_id = _path_tail(path[: -len("/compact")], "/api/chats/")
                session = chat_store.get_session(session_id)
                if session is None:
                    _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                turns = _conversation_turns(session)
                if not turns:
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "Chat session has no turns to compact."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                payload = _read_json(self)
                try:
                    compacted = LocalCompactionProvider(
                        config,
                        expert_id=_optional_str(payload.get("expert_id")),
                    ).compact(
                        turns=turns,
                        existing_summary=session.summary,
                        correlation_id=_optional_str(payload.get("correlation_id")),
                    )
                except ProviderError as exc:
                    _send_json(
                        self,
                        {"error": "provider_error", "message": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return
                try:
                    updated = chat_store.update_summary(
                        session.id,
                        compacted.summary,
                        meta={
                            "expert_id": compacted.expert_id,
                            "model": compacted.model,
                            "correlation_id": compacted.correlation_id,
                            "input_messages": len(turns),
                            "previous_summary_present": bool(session.summary.strip()),
                        },
                    )
                except ValueError as exc:
                    _send_json(
                        self,
                        {"error": "bad_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _send_json(
                    self,
                    {
                        "session": chat_session_payload(updated),
                        "summary": updated.summary,
                        "summary_updated_at": updated.summary_updated_at,
                        "compaction": {
                            "expert_id": compacted.expert_id,
                            "model": compacted.model,
                            "correlation_id": compacted.correlation_id,
                            "input_messages": len(turns),
                            "previous_summary_present": bool(session.summary.strip()),
                        },
                    },
                )
                return

            if path == "/api/evaluate":
                payload = _read_json(self)
                eval_path = str(payload.get("eval_path", "experiments/eval_set.jsonl"))
                result = evaluate_router(config, load_eval_cases(eval_path))
                _send_json(self, result)
                return

            if path == "/api/setup/run":
                payload = _read_json(self)
                result = run_runtime_setup(
                    config_path=config_path,
                    app_config_path=app_config_path,
                    execute=bool(payload.get("execute", False)),
                    download_models=bool(payload.get("download_models", False)),
                    confirm=bool(payload.get("confirm", False)),
                )
                _audit(
                    audit_store,
                    "setup.run",
                    result.status,
                    risk_class="process_execution" if result.execute or result.download_models else "read_only",
                    metadata={
                        "execute": result.execute,
                        "download_models": result.download_models,
                        "confirmed": result.confirmed,
                    },
                )
                _send_json(self, setup_run_payload(result))
                return

            if path == "/api/plugins":
                payload = _read_json(self)
                if payload.get("confirm") is not True:
                    _audit(audit_store, "plugin.create", "confirmation_required", risk_class="write_local")
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Plugin creation requires confirm=true because it writes local files.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    path_created = create_plugin_scaffold(
                        _required_payload_text(payload, "plugin_id"),
                        root=app_config.extensions.plugins_dir,
                        name=_optional_str(payload.get("name")),
                        description=_optional_str(payload.get("description")),
                        risk_class=_optional_str(payload.get("risk_class")) or "read_only",
                    )
                except (ExtensionError, ValueError) as exc:
                    _audit(
                        audit_store,
                        "plugin.create",
                        "error",
                        risk_class="write_local",
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "plugin_error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                registry = _load_registry(app_config)
                _audit(
                    audit_store,
                    "plugin.create",
                    "ok",
                    risk_class=_optional_str(payload.get("risk_class")) or "read_only",
                    subject=path_created.name,
                )
                _send_json(
                    self,
                    {
                        "created": True,
                        "plugin_id": path_created.name,
                        "path": str(path_created),
                        "manifest": str(path_created / "plugin.json"),
                        "skill": str(path_created / "SKILL.md"),
                        "audit": audit_extension_registry(registry),
                        "extensions": registry_payload(registry),
                    },
                    status=HTTPStatus.CREATED,
                )
                return

            if path == "/api/extensions/configure":
                payload = _read_json(self)
                surface = str(payload.get("surface", "")).strip()
                mode = str(payload.get("mode", "upsert")).strip() or "upsert"
                definition = payload.get("definition", {})
                if payload.get("confirm") is not True:
                    _audit(
                        audit_store,
                        "extension.configure",
                        "confirmation_required",
                        risk_class="write_local",
                        subject=surface or None,
                    )
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Extension configuration requires confirm=true because it writes registry files.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                if not isinstance(definition, dict):
                    _audit(
                        audit_store,
                        "extension.configure",
                        "bad_request",
                        risk_class="write_local",
                        subject=surface or None,
                        metadata={"message": "definition must be a JSON object"},
                    )
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "definition must be a JSON object"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    result = configure_extension_entry(
                        surface,
                        definition,
                        mode=mode,
                        mcp_config=app_config.extensions.mcp_config,
                        cron_config=app_config.extensions.cron_config,
                    )
                except (ExtensionError, ValueError) as exc:
                    _audit(
                        audit_store,
                        "extension.configure",
                        "error",
                        risk_class="write_local",
                        subject=surface or None,
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "extension_configure_error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                registry = _load_registry(app_config)
                cron_runner.replace_registry(registry)
                audit = audit_extension_registry(registry)
                _audit(
                    audit_store,
                    "extension.configure",
                    "ok",
                    risk_class="write_local",
                    subject=result["id"],
                    metadata={
                        "surface": result["surface"],
                        "mode": result["mode"],
                        "action": result["action"],
                    },
                )
                _send_json(
                    self,
                    {
                        "configured": True,
                        **result,
                        "audit": audit,
                        "extensions": registry_payload(registry),
                    },
                )
                return

            if path == "/api/models/start":
                payload = _read_json(self)
                action = model_manager.start(
                    confirm=bool(payload.get("confirm", False)),
                    only_first=bool(payload.get("only_first", False)),
                )
                _audit(
                    audit_store,
                    "models.start",
                    action.status,
                    risk_class="process_execution",
                    metadata={
                        "confirmed": action.confirmed,
                        "only_first": action.only_first,
                        "result_count": len(action.results),
                    },
                )
                _send_json(self, model_server_action_payload(action))
                return

            if path == "/api/models/stop":
                payload = _read_json(self)
                action = model_manager.stop(confirm=bool(payload.get("confirm", False)))
                _audit(
                    audit_store,
                    "models.stop",
                    action.status,
                    risk_class="process_execution",
                    metadata={"confirmed": action.confirmed, "result_count": len(action.results)},
                )
                _send_json(self, model_server_action_payload(action))
                return

            if path == "/api/cron/run":
                payload = _read_json(self)
                dry_run = bool(payload.get("dry_run", False))
                confirm_writes = bool(payload.get("confirm_writes", False))
                summary = run_due_jobs(
                    registry.cron_jobs,
                    state_path=_cron_state_path(app_config),
                    dry_run=dry_run,
                    confirm_writes=confirm_writes,
                    registry=registry,
                )
                _audit(
                    audit_store,
                    "cron.run",
                    "dry_run" if dry_run else "ok",
                    risk_class="write_local" if confirm_writes else "compute_only",
                    metadata={
                        "dry_run": dry_run,
                        "confirm_writes": confirm_writes,
                        "job_count": len(summary.results),
                    },
                )
                _send_json(self, cron_summary_payload(summary))
                return

            if path == "/api/tools/run":
                payload = _read_json(self)
                name = str(payload.get("name", "")).strip()
                tool_input = payload.get("input", {})
                if not isinstance(tool_input, dict):
                    _audit(
                        audit_store,
                        "tool.run",
                        "bad_request",
                        risk_class="compute_only",
                        subject=name,
                        metadata={"message": "input must be a JSON object"},
                    )
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "input must be a JSON object"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    result = LocalToolRunner(
                        registry,
                        app_config=app_config,
                        moe_config=config,
                    ).run(name, tool_input)
                except ToolExecutionError as exc:
                    _audit(
                        audit_store,
                        "tool.run",
                        "tool_error",
                        risk_class="compute_only",
                        subject=name,
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "tool_error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                except ProviderError as exc:
                    _audit(
                        audit_store,
                        "tool.run",
                        "provider_error",
                        risk_class="compute_only",
                        subject=name,
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "provider_error", "message": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return
                payload = tool_result_payload(result)
                _audit(
                    audit_store,
                    "tool.run",
                    result.status,
                    risk_class=result.risk_class,
                    subject=result.name,
                    metadata={"side_effects": result.side_effects},
                )
                if result.name == "extension.configure":
                    registry = _load_registry(app_config)
                    cron_runner.replace_registry(registry)
                    payload["payload"]["audit"] = audit_extension_registry(registry)
                    payload["payload"]["extensions"] = registry_payload(registry)
                _send_json(self, payload)
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/chats/"):
                session_id = _path_tail(path, "/api/chats/")
                payload = _read_json(self)
                title = _optional_str(payload.get("title"))
                if not title:
                    _send_json(
                        self,
                        {"error": "bad_request", "message": "title is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    session = chat_store.rename_session(session_id, title)
                except KeyError:
                    _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                except ValueError as exc:
                    _send_json(
                        self,
                        {"error": "bad_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _audit(audit_store, "chat.rename", "ok", risk_class="write_local", subject=session_id)
                _send_json(self, chat_session_payload(session))
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            confirm = parse_qs(parsed_url.query).get("confirm", ["false"])[0] == "true"
            if path.startswith("/api/memory/"):
                if not confirm:
                    _audit(audit_store, "memory.forget", "confirmation_required", risk_class="write_local")
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Memory deletion requires confirm=true.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                record_id = _path_tail(path, "/api/memory/")
                try:
                    report = memory_store.forget_record(record_id)
                except ValueError as exc:
                    _audit(
                        audit_store,
                        "memory.forget",
                        "error",
                        risk_class="write_local",
                        subject=record_id,
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "invalid_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _audit(
                    audit_store,
                    "memory.forget",
                    "ok",
                    risk_class="write_local",
                    subject=report.target,
                    metadata={"removed_count": report.removed_count, "remaining_count": report.remaining_count},
                )
                _send_json(
                    self,
                    {
                        "deleted": report.removed_count > 0,
                        "target": report.target,
                        "removed_count": report.removed_count,
                        "remaining_count": report.remaining_count,
                        "removed_ids": list(report.removed_ids),
                    },
                )
                return

            if path.startswith("/api/knowledge/"):
                if not confirm:
                    _audit(audit_store, "knowledge.forget", "confirmation_required", risk_class="write_local")
                    _send_json(
                        self,
                        {
                            "error": "confirmation_required",
                            "message": "Knowledge deletion requires confirm=true.",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                document_id = _path_tail(path, "/api/knowledge/")
                try:
                    report = memory_store.forget_document(document_id)
                except ValueError as exc:
                    _audit(
                        audit_store,
                        "knowledge.forget",
                        "error",
                        risk_class="write_local",
                        subject=document_id,
                        metadata={"message": str(exc)},
                    )
                    _send_json(
                        self,
                        {"error": "invalid_request", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                _audit(
                    audit_store,
                    "knowledge.forget",
                    "ok",
                    risk_class="write_local",
                    subject=report.target,
                    metadata={"removed_count": report.removed_count, "remaining_count": report.remaining_count},
                )
                _send_json(
                    self,
                    {
                        "deleted": report.removed_count > 0,
                        "target": report.target,
                        "removed_count": report.removed_count,
                        "remaining_count": report.remaining_count,
                        "removed_ids": list(report.removed_ids),
                    },
                )
                return

            if path.startswith("/api/chats/"):
                session_id = _path_tail(path, "/api/chats/")
                deleted = chat_store.delete_session(session_id)
                if not deleted:
                    _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                _audit(audit_store, "chat.delete", "ok", risk_class="write_local", subject=session_id)
                _send_json(self, {"deleted": True, "id": session_id})
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

    return MyMoEHandler


def _asset(name: str) -> str:
    return (Path(__file__).parent / "ui" / name).read_text(encoding="utf-8")


def _load_registry(app_config: object) -> object:
    return load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


def _send_json(
    handler: BaseHTTPRequestHandler,
    payload: object,
    *,
    status: HTTPStatus = HTTPStatus.OK,
) -> None:
    _send(
        handler,
        status,
        json.dumps(payload, indent=2).encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )


def _send(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    body: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_download(
    handler: BaseHTTPRequestHandler,
    body: bytes,
    *,
    content_type: str,
    filename: str,
) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(filename)}")
    handler.end_headers()
    handler.wfile.write(body)


def _send_sse_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()


def _send_sse_event(handler: BaseHTTPRequestHandler, event: str, payload: object) -> None:
    body = f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
    handler.wfile.write(body.encode("utf-8"))
    handler.wfile.flush()


def _config_payload(config_path: str, config: object, app_config: object) -> dict[str, object]:
    return {
        "app": app_config_payload(app_config),
        "config_path": config_path,
        "requires_model": True,
        "routing": _routing_payload(config.routing),
        "experts": [
            {
                "id": expert.id,
                "provider": expert.provider,
                "model": expert.model,
                "role": expert.role,
                "base_url": expert.base_url,
                "weight": expert.weight,
                "runtime_backend": str(expert.params.get("runtime_backend", "provider_default")),
            }
            for expert in config.experts
        ],
        "rules": [rule.__dict__ for rule in config.rules],
    }


def _cron_state_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/cron-state.json"


def _chat_store_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/chats.json"


def _memory_store_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/memory.jsonl"


def _audit_log_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/audit.jsonl"


def _path_tail(path: str, prefix: str) -> str:
    return unquote(path[len(prefix) :]).strip("/")


def _query_limit(raw: object, *, default: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def _bounded_int(raw: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


def _audit(
    audit_store: AuditLogStore,
    action: str,
    status: str,
    *,
    risk_class: str = "read_only",
    subject: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        audit_store.record(
            action,
            status,
            risk_class=risk_class,
            subject=subject,
            metadata=metadata,
        )
    except Exception:
        return


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")
    return safe or "chat"


def _routing_payload(routing: object) -> dict[str, object]:
    semantic = routing.semantic
    distilled = routing.distilled
    return {
        "top_k": routing.top_k,
        "fallback_order": list(routing.fallback_order),
        "aggregation": routing.aggregation,
        "strategy": routing.strategy,
        "semantic": {
            "enabled": semantic.enabled,
            "method": semantic.method,
            "min_score": semantic.min_score,
            "margin": semantic.margin,
            "weight": semantic.weight,
            "ngram_min": semantic.ngram_min,
            "ngram_max": semantic.ngram_max,
            "examples": [
                {
                    "expert_id": example.expert_id,
                    "utterances": list(example.utterances),
                    "weight": example.weight,
                }
                for example in semantic.examples
            ],
        },
        "distilled": {
            "enabled": distilled.enabled,
            "artifact_path": distilled.artifact_path,
            "min_confidence": distilled.min_confidence,
            "weight": distilled.weight,
        },
    }


def _response_payload(response: object) -> dict[str, object]:
    return {
        "content": response.content,
        "correlation_id": response.correlation_id,
        "route": {
            "selected": [item.__dict__ for item in response.route.selected],
            "fallback_order": list(response.route.fallback_order),
        },
        "results": [item.__dict__ for item in response.results],
        "errors": list(response.errors),
        "disagreement": asdict(response.disagreement) if response.disagreement else None,
        "context": {},
    }


def _route_payload(route: object) -> dict[str, object]:
    return {
        "selected": [item.__dict__ for item in route.selected],
        "fallback_order": list(route.fallback_order),
    }


def _build_model_context(
    session: ChatSession | None,
    prompt: str,
    context_policy: object,
    *,
    memory_store: FileMemoryStore,
) -> dict[str, object]:
    memories = _memory_snippets(memory_store, prompt, limit=context_policy.max_memory_items)
    if session is None or not session.messages:
        bundle = build_context_bundle(
            system_prompt="",
            current_prompt=prompt,
            turns=(),
            memories=memories,
            policy=context_policy,
        )
        model_prompt = bundle.as_prompt() if memories else prompt
        return {"prompt": model_prompt, "payload": _context_payload(bundle, memories=memories)}

    turns = _conversation_turns(session)
    bundle = build_context_bundle(
        system_prompt=(
            "You are continuing a local chat session. "
            "Use prior messages only as context for the current user message."
        ),
        current_prompt=f"Current user message:\n{prompt}",
        turns=turns,
        summary=session.summary,
        memories=memories,
        policy=context_policy,
    )
    return {"prompt": bundle.as_prompt(), "payload": _context_payload(bundle, memories=memories)}


def _context_payload(
    bundle: ContextBundle,
    *,
    memories: tuple[MemorySnippet, ...] = (),
) -> dict[str, object]:
    return {
        "token_estimate": bundle.token_estimate,
        "budget_tokens": bundle.budget_tokens,
        "compaction_needed": bundle.compaction_needed,
        "dropped_turns": bundle.dropped_turns,
        "sections": bundle.by_section(),
        "memory_ids": [memory.id for memory in memories],
    }


def _memory_snippets(
    memory_store: FileMemoryStore,
    prompt: str,
    *,
    limit: int,
) -> tuple[MemorySnippet, ...]:
    return tuple(
        MemorySnippet(id=record.id, text=record.text, score=score)
        for record, score in memory_store.search(prompt, scope="default", limit=limit)
    )


def _conversation_turns(session: ChatSession) -> tuple[ConversationTurn, ...]:
    return tuple(
        ConversationTurn(role=message.role, content=message.content)
        for message in session.messages
        if message.role in {"user", "assistant"}
    )


def _optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _required_payload_text(payload: dict[str, Any], key: str) -> str:
    value = _optional_str(payload.get(key))
    if value is None:
        raise ValueError(f"{key} is required.")
    return value


if __name__ == "__main__":
    main()
