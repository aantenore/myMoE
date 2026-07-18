from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Callable, Protocol
from urllib import error, request
from urllib.parse import urljoin, urlsplit


class HttpBoundaryError(error.URLError):
    """Raised when a model endpoint request crosses its HTTP origin boundary."""


@dataclass(frozen=True)
class HttpOrigin:
    scheme: str
    host: str
    port: int


class HttpOpener(Protocol):
    def open(
        self,
        target: str | request.Request,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> Any:
        ...


OpenerBuilder = Callable[..., HttpOpener]


class SameOriginRedirectHandler(request.HTTPRedirectHandler):
    """Allow redirects only when scheme, host, and effective port are unchanged."""

    def __init__(self, origin: HttpOrigin):
        super().__init__()
        self._origin = origin

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> request.Request | None:
        redirect_url = urljoin(req.full_url, newurl)
        _require_same_origin(self._origin, redirect_url, phase="redirect")

        if code in {307, 308} and req.get_method() not in {"GET", "HEAD"}:
            return request.Request(
                redirect_url,
                data=req.data,
                headers=dict(req.header_items()),
                origin_req_host=req.origin_req_host,
                unverifiable=True,
                method=req.get_method(),
            )
        return super().redirect_request(req, fp, code, msg, headers, redirect_url)


def open_model_endpoint(
    target: str | request.Request,
    *,
    timeout: float,
    opener_builder: OpenerBuilder | None = None,
) -> Any:
    """Open an HTTP model endpoint without allowing origin changes.

    A fresh opener is built for each call so loopback requests can explicitly
    ignore ambient proxy configuration without changing process-wide state.
    ``opener_builder`` keeps the transport construction injectable for tests.
    """

    target_url = _target_url(target)
    origin = http_origin(target_url)
    opener = build_model_endpoint_opener(
        target_url,
        opener_builder=opener_builder,
    )
    response = opener.open(target, timeout=timeout)
    try:
        final_url = response.geturl()
        if not isinstance(final_url, str) or not final_url:
            raise HttpBoundaryError("model endpoint response has no final URL")
        _require_same_origin(origin, final_url, phase="final response")
    except Exception:
        response.close()
        raise
    return response


def build_model_endpoint_opener(
    target_url: str,
    *,
    opener_builder: OpenerBuilder | None = None,
) -> HttpOpener:
    """Build the isolated opener used for one model endpoint origin."""

    origin = http_origin(target_url)
    handlers: list[Any] = []
    if _host_is_loopback(origin.host):
        handlers.append(request.ProxyHandler({}))
    handlers.append(SameOriginRedirectHandler(origin))
    builder = opener_builder or request.build_opener
    return builder(*handlers)


def http_origin(url: str) -> HttpOrigin:
    try:
        parsed = urlsplit(str(url).strip())
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise HttpBoundaryError(f"invalid model endpoint URL: {exc}") from exc

    if scheme not in {"http", "https"} or not host:
        raise HttpBoundaryError("model endpoint URL must use HTTP(S) with a host")
    if parsed.username is not None or parsed.password is not None:
        raise HttpBoundaryError("model endpoint URL must not contain credentials")

    normalized_host = _normalize_host(host)
    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    return HttpOrigin(scheme=scheme, host=normalized_host, port=effective_port)


def is_loopback_http_url(url: str) -> bool:
    try:
        return _host_is_loopback(http_origin(url).host)
    except HttpBoundaryError:
        return False


def _target_url(target: str | request.Request) -> str:
    if isinstance(target, request.Request):
        return target.full_url
    return str(target)


def _require_same_origin(expected: HttpOrigin, url: str, *, phase: str) -> None:
    actual = http_origin(url)
    if actual != expected:
        raise HttpBoundaryError(
            f"model endpoint {phase} crossed the configured HTTP origin"
        )


def _normalize_host(host: str) -> str:
    normalized = host.rstrip(".").lower()
    try:
        return normalized.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise HttpBoundaryError("model endpoint host is invalid") from exc


def _host_is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)
