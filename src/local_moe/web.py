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
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .chat_store import ChatSession, FileChatStore, chat_session_payload, chat_summary_payload
from .compaction import LocalCompactionProvider
from .config import load_config
from .context import ContextBundle, ConversationTurn, build_context_bundle
from .context_policy import load_context_policy
from .evaluator import evaluate_router, load_eval_cases
from .extensions import load_extension_registry, registry_payload
from .health import check_runtime_health, runtime_health_payload
from .orchestrator import LocalMoE
from .providers import ProviderError
from .scheduler import cron_status, cron_summary_payload, run_due_jobs
from .setup_status import inspect_setup_status, setup_status_payload
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
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
    registry = load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )
    chat_store = FileChatStore(_chat_store_path(app_config))
    handler = _make_handler(
        config_path=config_path,
        app_config_path=app_config_path,
        app_config=app_config,
        config=config,
        context_policy=context_policy,
        moe=moe,
        registry=registry,
        chat_store=chat_store,
    )
    return ThreadingHTTPServer((host, port), handler)


def _make_handler(
    config_path: str,
    app_config_path: str,
    app_config: object,
    config: object,
    context_policy: object,
    moe: LocalMoE,
    registry: object,
    chat_store: FileChatStore,
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

            if path == "/api/extensions":
                _send_json(self, registry_payload(registry))
                return

            if path == "/api/runtime":
                plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
                _send_json(self, runtime_plan_payload(plan))
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
                _send_json(
                    self,
                    cron_status(
                        registry.cron_jobs,
                        state_path=_cron_state_path(app_config),
                    ),
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

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
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
                model_context = _build_model_context(session, prompt, context_policy)

                try:
                    response = moe.generate(
                        model_context["prompt"],
                        correlation_id=_optional_str(payload.get("correlation_id")),
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
                _send_json(self, chat_session_payload(session), status=HTTPStatus.CREATED)
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

            if path == "/api/cron/run":
                payload = _read_json(self)
                summary = run_due_jobs(
                    registry.cron_jobs,
                    state_path=_cron_state_path(app_config),
                    dry_run=bool(payload.get("dry_run", False)),
                    confirm_writes=bool(payload.get("confirm_writes", False)),
                    registry=registry,
                )
                _send_json(self, cron_summary_payload(summary))
                return

            if path == "/api/tools/run":
                payload = _read_json(self)
                name = str(payload.get("name", "")).strip()
                tool_input = payload.get("input", {})
                if not isinstance(tool_input, dict):
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
                    _send_json(
                        self,
                        {"error": "tool_error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                except ProviderError as exc:
                    _send_json(
                        self,
                        {"error": "provider_error", "message": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return
                _send_json(self, tool_result_payload(result))
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
                _send_json(self, chat_session_payload(session))
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/chats/"):
                session_id = _path_tail(path, "/api/chats/")
                deleted = chat_store.delete_session(session_id)
                if not deleted:
                    _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return
                _send_json(self, {"deleted": True, "id": session_id})
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

    return MyMoEHandler


def _asset(name: str) -> str:
    return (Path(__file__).parent / "ui" / name).read_text(encoding="utf-8")


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


def _path_tail(path: str, prefix: str) -> str:
    return unquote(path[len(prefix) :]).strip("/")


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


def _build_model_context(
    session: ChatSession | None,
    prompt: str,
    context_policy: object,
) -> dict[str, object]:
    if session is None or not session.messages:
        bundle = build_context_bundle(
            system_prompt="",
            current_prompt=prompt,
            turns=(),
            policy=context_policy,
        )
        return {"prompt": prompt, "payload": _context_payload(bundle)}

    turns = _conversation_turns(session)
    bundle = build_context_bundle(
        system_prompt=(
            "You are continuing a local chat session. "
            "Use prior messages only as context for the current user message."
        ),
        current_prompt=f"Current user message:\n{prompt}",
        turns=turns,
        summary=session.summary,
        policy=context_policy,
    )
    return {"prompt": bundle.as_prompt(), "payload": _context_payload(bundle)}


def _context_payload(bundle: ContextBundle) -> dict[str, object]:
    return {
        "token_estimate": bundle.token_estimate,
        "budget_tokens": bundle.budget_tokens,
        "compaction_needed": bundle.compaction_needed,
        "dropped_turns": bundle.dropped_turns,
        "sections": bundle.by_section(),
    }


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


if __name__ == "__main__":
    main()
