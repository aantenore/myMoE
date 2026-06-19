from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .app_config import app_config_payload, load_app_config
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .config import load_config
from .evaluator import evaluate_router, load_eval_cases
from .extensions import load_extension_registry, registry_payload
from .orchestrator import LocalMoE
from .providers import ProviderError


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
    moe = LocalMoE(config)
    registry = load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )
    handler = _make_handler(
        config_path=config_path,
        app_config=app_config,
        config=config,
        moe=moe,
        registry=registry,
    )
    return ThreadingHTTPServer((host, port), handler)


def _make_handler(
    config_path: str,
    app_config: object,
    config: object,
    moe: LocalMoE,
    registry: object,
) -> type[BaseHTTPRequestHandler]:
    class MyMoEHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                _send(
                    self,
                    HTTPStatus.OK,
                    _asset("index.html").encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                )
                return

            if self.path == "/api/config":
                _send_json(self, _config_payload(config_path, config, app_config))
                return

            if self.path == "/api/extensions":
                _send_json(self, registry_payload(registry))
                return

            if self.path == "/api/runtime":
                plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
                _send_json(self, runtime_plan_payload(plan))
                return

            _send_json(self, {"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/generate":
                payload = _read_json(self)
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    _send_json(
                        self,
                        {"error": "prompt is required"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                try:
                    response = moe.generate(
                        prompt,
                        correlation_id=_optional_str(payload.get("correlation_id")),
                    )
                except ProviderError as exc:
                    _send_json(
                        self,
                        {"error": "provider_error", "message": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return

                _send_json(self, _response_payload(response))
                return

            if self.path == "/api/evaluate":
                payload = _read_json(self)
                eval_path = str(payload.get("eval_path", "experiments/eval_set.jsonl"))
                result = evaluate_router(config, load_eval_cases(eval_path))
                _send_json(self, result)
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


def _routing_payload(routing: object) -> dict[str, object]:
    semantic = routing.semantic
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
    }


def _optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


if __name__ == "__main__":
    main()
