from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from unittest import mock
import threading
import unittest
from urllib import request

from local_moe.http_boundary import (
    HttpBoundaryError,
    build_model_endpoint_opener,
    open_model_endpoint,
)


class _EndpointHandler(BaseHTTPRequestHandler):
    cross_origin_target = ""
    final_bodies: list[bytes] = []
    final_gets = 0

    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        if self.path == "/final":
            type(self).final_gets += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path == "/cross-origin":
            self.send_response(302)
            self.send_header("Location", type(self).cross_origin_target)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path in {"/post-redirect", "/post-redirect-308"}:
            self.send_response(308 if self.path.endswith("308") else 307)
            self.send_header("Location", "/post-final")
            self.end_headers()
            return
        if self.path == "/cross-origin":
            self.send_response(307)
            self.send_header("Location", type(self).cross_origin_target)
            self.end_headers()
            return
        if self.path == "/post-final":
            type(self).final_bodies.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class _CrossOriginHandler(BaseHTTPRequestHandler):
    received_bodies: list[bytes] = []
    received_gets = 0

    def do_GET(self) -> None:
        type(self).received_gets += 1
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).received_bodies.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class _ProxyHandler(BaseHTTPRequestHandler):
    requests_seen = 0

    def do_GET(self) -> None:
        type(self).requests_seen += 1
        self.send_response(502)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _server(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class HttpBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        _EndpointHandler.cross_origin_target = ""
        _EndpointHandler.final_bodies = []
        _EndpointHandler.final_gets = 0
        _CrossOriginHandler.received_bodies = []
        _CrossOriginHandler.received_gets = 0
        _ProxyHandler.requests_seen = 0

    def test_allows_same_origin_redirect_and_preserves_post_body(self) -> None:
        with _server(_EndpointHandler) as endpoint:
            host, port = endpoint.server_address
            payload = b'{"prompt":"hello"}'
            http_request = request.Request(
                f"http://{host}:{port}/post-redirect",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with open_model_endpoint(http_request, timeout=1) as response:
                self.assertEqual(response.read(), b"ok")
                self.assertEqual(
                    response.geturl(),
                    f"http://{host}:{port}/post-final",
                )

        self.assertEqual(_EndpointHandler.final_bodies, [payload])

    def test_rejects_cross_origin_redirect_before_target_receives_body(self) -> None:
        with _server(_CrossOriginHandler) as target, _server(_EndpointHandler) as endpoint:
            target_host, target_port = target.server_address
            endpoint_host, endpoint_port = endpoint.server_address
            _EndpointHandler.cross_origin_target = (
                f"http://{target_host}:{target_port}/receive"
            )
            http_request = request.Request(
                f"http://{endpoint_host}:{endpoint_port}/cross-origin",
                data=b"sensitive prompt",
                method="POST",
            )

            with self.assertRaisesRegex(HttpBoundaryError, "crossed"):
                open_model_endpoint(http_request, timeout=1)

        self.assertEqual(_CrossOriginHandler.received_bodies, [])

    def test_allows_same_origin_308_and_preserves_post_body(self) -> None:
        with _server(_EndpointHandler) as endpoint:
            host, port = endpoint.server_address
            payload = b'{"prompt":"hello-308"}'
            http_request = request.Request(
                f"http://{host}:{port}/post-redirect-308",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with open_model_endpoint(http_request, timeout=1) as response:
                self.assertEqual(response.read(), b"ok")

        self.assertEqual(_EndpointHandler.final_bodies, [payload])

    def test_rejects_cross_origin_get_redirect_before_target_receives_request(self) -> None:
        with _server(_CrossOriginHandler) as target, _server(_EndpointHandler) as endpoint:
            target_host, target_port = target.server_address
            endpoint_host, endpoint_port = endpoint.server_address
            _EndpointHandler.cross_origin_target = (
                f"http://{target_host}:{target_port}/receive"
            )

            with self.assertRaisesRegex(HttpBoundaryError, "crossed"):
                open_model_endpoint(
                    f"http://{endpoint_host}:{endpoint_port}/cross-origin",
                    timeout=1,
                )

        self.assertEqual(_CrossOriginHandler.received_gets, 0)

    def test_loopback_opener_has_an_empty_proxy_map(self) -> None:
        captured_handlers: list[object] = []

        class _UnusedOpener:
            pass

        def builder(*handlers: object) -> _UnusedOpener:
            captured_handlers.extend(handlers)
            return _UnusedOpener()

        build_model_endpoint_opener(
            "http://127.0.0.1:8101/v1",
            opener_builder=builder,
        )

        proxy_handlers = [
            handler
            for handler in captured_handlers
            if isinstance(handler, request.ProxyHandler)
        ]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies, {})

    def test_loopback_request_ignores_ambient_http_proxy(self) -> None:
        with _server(_ProxyHandler) as proxy, _server(_EndpointHandler) as endpoint:
            proxy_host, proxy_port = proxy.server_address
            endpoint_host, endpoint_port = endpoint.server_address
            proxy_url = f"http://{proxy_host}:{proxy_port}"
            environment = {
                "http_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "no_proxy": "",
                "NO_PROXY": "",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                with open_model_endpoint(
                    f"http://{endpoint_host}:{endpoint_port}/final",
                    timeout=1,
                ) as response:
                    self.assertEqual(response.read(), b"ok")

        self.assertEqual(_EndpointHandler.final_gets, 1)
        self.assertEqual(_ProxyHandler.requests_seen, 0)

    def test_rejects_an_untrusted_final_url_and_closes_response(self) -> None:
        class _Response:
            closed = False

            def geturl(self) -> str:
                return "https://other.example/v1/models"

            def close(self) -> None:
                self.closed = True

        response = _Response()

        class _Opener:
            def open(self, *_args: object, **_kwargs: object) -> _Response:
                return response

        with self.assertRaisesRegex(HttpBoundaryError, "final response"):
            open_model_endpoint(
                "https://models.example/v1/models",
                timeout=1,
                opener_builder=lambda *_handlers: _Opener(),
            )

        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
