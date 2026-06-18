from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import load_config
from .evaluator import evaluate_router, load_eval_cases
from .orchestrator import LocalMoE
from .providers import ProviderError


def main() -> None:
    parser = argparse.ArgumentParser(description="myMoE local web UI")
    parser.add_argument("--config", default="configs/moe.mock.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()

    server = build_server(args.config, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"myMoE UI listening on {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def build_server(config_path: str, host: str = "127.0.0.1", port: int = 8089) -> ThreadingHTTPServer:
    config = load_config(config_path)
    moe = LocalMoE(config)
    handler = _make_handler(config_path=config_path, config=config, moe=moe)
    return ThreadingHTTPServer((host, port), handler)


def _make_handler(config_path: str, config: object, moe: LocalMoE) -> type[BaseHTTPRequestHandler]:
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
                _send_json(self, _config_payload(config_path, config))
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


def _config_payload(config_path: str, config: object) -> dict[str, object]:
    return {
        "config_path": config_path,
        "routing": config.routing.__dict__,
        "experts": [
            {
                "id": expert.id,
                "provider": expert.provider,
                "model": expert.model,
                "role": expert.role,
                "base_url": expert.base_url,
                "weight": expert.weight,
            }
            for expert in config.experts
        ],
        "rules": [rule.__dict__ for rule in config.rules],
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
    }


def _optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


if __name__ == "__main__":
    main()
