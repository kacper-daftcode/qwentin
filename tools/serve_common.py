"""Request-side hardening of tools/serve_batched.py -- no engine involvement.

What the server gets from here (flags via add_hardening_args):
  --host              default 127.0.0.1 (DEFAULT_HOST): the server binds to loopback unless told otherwise
  --api-key KEY       when set, every path except /health needs `Authorization: Bearer KEY` (401 otherwise)
  --max-body-bytes N  request bodies above N bytes are refused with 413 before they are read (default 32 MiB)
  --max-tokens-cap N  a request's max_tokens above N is clamped to N; the clamp is reported in the response's
                      x_qwentin (max_tokens_clamped_from / max_tokens_cap) and finish_reason reads "length" (default 16384)
  --socket-timeout S  read / write timeout of a client socket: a connection that stops sending its request or stops
                      reading the response is closed after S seconds of blocking (default 600; the server's --timeout is
                      the admission wait, a different thing)

Pure stdlib, no GPU: tests/test_serve_common.py exercises every function here and the handler wiring of
serve_batched.py with a fake engine.
"""

from __future__ import annotations

import hmac
import json

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOKENS_CAP = 16384
DEFAULT_SOCKET_TIMEOUT_S = 600.0
OPEN_PATHS = ("/health",)


def add_hardening_args(ap, host_default: str = DEFAULT_HOST, max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP):
    """The hardening flags. --host is added here so the server defaults to loopback."""
    ap.add_argument("--host", default=host_default,
                    help=f"bind address (default {host_default}: loopback only; 0.0.0.0 exposes the server to the network -- "
                         "pair that with --api-key)")
    ap.add_argument("--api-key", default=None,
                    help="require `Authorization: Bearer <key>` on every path except /health (401 otherwise); unset = open")
    ap.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES,
                    help=f"refuse request bodies larger than this with 413 before reading them (default {DEFAULT_MAX_BODY_BYTES} = 32 MiB)")
    ap.add_argument("--max-tokens-cap", type=int, default=max_tokens_cap,
                    help=f"clamp a request's max_tokens to this (default {max_tokens_cap}); the clamp is reported in x_qwentin "
                         "(max_tokens_clamped_from, max_tokens_cap)")
    ap.add_argument("--socket-timeout", type=float, default=DEFAULT_SOCKET_TIMEOUT_S,
                    help=f"read / write timeout in seconds of a client connection (default {DEFAULT_SOCKET_TIMEOUT_S:.0f}): idle "
                         "keep-alive connections, half-sent requests and clients that stop reading a stream are closed after it")
    return ap


def is_open_path(path: str) -> bool:
    """Paths that never need the API key (/health, for load balancers and the verification scripts' readiness polls)."""
    p = path.split("?", 1)[0].rstrip("/") or "/"
    return any(p == o for o in OPEN_PATHS)


def auth_ok(headers, api_key: str | None, path: str = "/v1/") -> bool:
    """True when no key is configured, the path is open, or the request carries `Authorization: Bearer <api_key>`
    (compared in constant time). `headers` is anything with .get() (http.server's message object, a dict in tests)."""
    if not api_key or is_open_path(path):
        return True
    value = headers.get("Authorization") or headers.get("authorization") or ""
    scheme, _, token = value.strip().partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), api_key.encode())


def unauthorized_body() -> dict:
    return {"error": {"message": "missing or invalid API key (Authorization: Bearer <key>)", "type": "authentication_error",
                      "code": 401}}


class BodyError(Exception):
    """A request body that must be refused: .status (411 / 413 / 400) and .body (the JSON error object)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": "invalid_request_error", "code": status}}


def body_length(headers, max_bytes: int) -> int:
    """Validate Content-Length against the limit WITHOUT reading the body: 411 when absent (chunked uploads are not
    accepted), 400 when not a non-negative integer, 413 when above max_bytes. Returns the length to read."""
    raw = headers.get("Content-Length")
    if raw is None:
        raw = headers.get("content-length")
    if raw is None:
        if (headers.get("Transfer-Encoding") or headers.get("transfer-encoding") or "").lower().strip() == "chunked":
            raise BodyError(411, "chunked request bodies are not accepted; send Content-Length")
        raise BodyError(411, "Content-Length required")
    try:
        n = int(raw.strip())
    except (ValueError, AttributeError):
        raise BodyError(400, f"bad Content-Length: {raw!r}") from None
    if n < 0:
        raise BodyError(400, f"bad Content-Length: {raw!r}")
    if n > max_bytes:
        raise BodyError(413, f"request body of {n} bytes exceeds the limit of {max_bytes} bytes (--max-body-bytes)")
    return n


def read_json_body(handler, max_bytes: int) -> dict:
    """Read and parse the JSON body of an http.server request handler within the size limit; raises BodyError
    (411 / 413 / 400) -- the caller answers with .status / .body. An empty body is {}."""
    n = body_length(handler.headers, max_bytes)
    raw = handler.rfile.read(n) if n else b""
    if len(raw) != n:
        raise BodyError(400, f"short body: {len(raw)} of {n} bytes")
    if not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise BodyError(400, f"bad json: {e}") from None
    if not isinstance(obj, dict):
        raise BodyError(400, "the request body must be a JSON object")
    return obj


def clamp_max_tokens(requested, cap: int, default: int) -> tuple[int, int | None]:
    """(max_tokens to use, the requested value when it was clamped else None). Non-numeric / missing / <= 0 requests
    get `default` (then clamped as well); the cap applies to the server's own default too."""
    try:
        n = int(requested) if requested is not None else default
    except (TypeError, ValueError):
        n = default
    if n <= 0:
        n = default
    if cap and n > cap:
        return cap, n
    return n, None


def clamp_fields(clamped_from: int | None, cap: int) -> dict:
    """The x_qwentin fields that report a clamp: {} when nothing was clamped."""
    if clamped_from is None:
        return {}
    return {"max_tokens_clamped_from": int(clamped_from), "max_tokens_cap": int(cap)}


def apply_socket_timeout(handler_class, seconds: float | None):
    """http.server applies handler_class.timeout to every accepted connection (settimeout in StreamRequestHandler.setup),
    which bounds blocking reads (idle keep-alive, half-sent bodies) and writes (clients that stop reading a stream);
    a timed-out request is logged and the connection closed (handle_one_request catches TimeoutError)."""
    handler_class.timeout = float(seconds) if seconds and seconds > 0 else None
    return handler_class
