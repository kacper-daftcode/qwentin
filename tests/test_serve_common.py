"""CPU-only tests of the server hardening: tools/serve_common.py directly, and tools/serve_batched.py's handler wired
to a fake engine + fake tokenizer over a real loopback socket (no CUDA, no torch, no transformers)."""

from __future__ import annotations

import http.client
import io
import json
import os
import queue
import sys
import threading
import time
import types
from http.server import ThreadingHTTPServer

import pytest

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)

import serve_common as sc  # noqa: E402

# ----------------------------------------------------------------------------- serve_common


def test_open_paths_and_auth():
    assert sc.is_open_path("/health")
    assert sc.is_open_path("/health/")
    assert sc.is_open_path("/health?x=1")
    assert not sc.is_open_path("/healthz")
    assert not sc.is_open_path("/v1/models")
    assert not sc.is_open_path("/spec?on=0")
    # no key configured -> everything open
    assert sc.auth_ok({}, None, "/v1/chat/completions")
    assert sc.auth_ok({}, "", "/v1/chat/completions")
    # key configured
    assert sc.auth_ok({}, "s3cret", "/health")
    assert not sc.auth_ok({}, "s3cret", "/v1/models")
    assert sc.auth_ok({"Authorization": "Bearer s3cret"}, "s3cret", "/v1/models")
    assert sc.auth_ok({"authorization": "bearer s3cret"}, "s3cret", "/v1/models")
    assert not sc.auth_ok({"Authorization": "Bearer s3cre"}, "s3cret", "/v1/models")
    assert not sc.auth_ok({"Authorization": "Basic s3cret"}, "s3cret", "/v1/models")
    assert not sc.auth_ok({"Authorization": "s3cret"}, "s3cret", "/v1/models")
    assert sc.unauthorized_body()["error"]["code"] == 401


def test_body_length():
    assert sc.body_length({"Content-Length": "10"}, 100) == 10
    assert sc.body_length({"content-length": " 0 "}, 100) == 0
    with pytest.raises(sc.BodyError) as e:
        sc.body_length({"Content-Length": "101"}, 100)
    assert e.value.status == 413 and "101" in e.value.body["error"]["message"]
    with pytest.raises(sc.BodyError) as e:
        sc.body_length({}, 100)
    assert e.value.status == 411
    with pytest.raises(sc.BodyError) as e:
        sc.body_length({"Transfer-Encoding": "chunked"}, 100)
    assert e.value.status == 411
    for bad in ("abc", "-5", ""):
        with pytest.raises(sc.BodyError) as e:
            sc.body_length({"Content-Length": bad}, 100)
        assert e.value.status == 400


def _fake_handler(body: bytes, headers: dict | None = None):
    h = types.SimpleNamespace()
    h.headers = headers if headers is not None else {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    return h


def test_read_json_body():
    assert sc.read_json_body(_fake_handler(b'{"a": 1}'), 1 << 20) == {"a": 1}
    assert sc.read_json_body(_fake_handler(b""), 1 << 20) == {}
    assert sc.read_json_body(_fake_handler(b"   "), 1 << 20) == {}
    with pytest.raises(sc.BodyError) as e:
        sc.read_json_body(_fake_handler(b"{not json"), 1 << 20)
    assert e.value.status == 400 and "bad json" in e.value.body["error"]["message"]
    with pytest.raises(sc.BodyError) as e:
        sc.read_json_body(_fake_handler(b"[1, 2]"), 1 << 20)
    assert e.value.status == 400
    with pytest.raises(sc.BodyError) as e:
        sc.read_json_body(_fake_handler(b"x" * 64), 32)
    assert e.value.status == 413
    # a declared length longer than what arrives is a 400, not a hang
    with pytest.raises(sc.BodyError) as e:
        sc.read_json_body(_fake_handler(b'{"a": 1}', {"Content-Length": "500"}), 1 << 20)
    assert e.value.status == 400


def test_clamp_max_tokens():
    assert sc.clamp_max_tokens(100, 4096, 128) == (100, None)
    assert sc.clamp_max_tokens(5000, 4096, 128) == (4096, 5000)
    assert sc.clamp_max_tokens("5000", 4096, 128) == (4096, 5000)
    assert sc.clamp_max_tokens(None, 4096, 128) == (128, None)
    assert sc.clamp_max_tokens("x", 4096, 128) == (128, None)
    assert sc.clamp_max_tokens(0, 4096, 128) == (128, None)
    assert sc.clamp_max_tokens(-3, 4096, 128) == (128, None)
    # the server's own default is capped too (serve_openai's Responses default 8192 with the 4096 cap)
    assert sc.clamp_max_tokens(None, 4096, 8192) == (4096, 8192)
    # cap 0 / None disables
    assert sc.clamp_max_tokens(10**9, 0, 128) == (10**9, None)
    assert sc.clamp_fields(None, 4096) == {}
    assert sc.clamp_fields(5000, 4096) == {"max_tokens_clamped_from": 5000, "max_tokens_cap": 4096}


def test_apply_socket_timeout():
    class H:
        timeout = None

    assert sc.apply_socket_timeout(H, 12.5).timeout == 12.5
    assert sc.apply_socket_timeout(H, 0).timeout is None
    assert sc.apply_socket_timeout(H, None).timeout is None


def test_add_hardening_args_defaults():
    import argparse

    ap = argparse.ArgumentParser()
    sc.add_hardening_args(ap)
    a = ap.parse_args([])
    assert a.host == "127.0.0.1"
    assert a.api_key is None
    assert a.max_body_bytes == 32 * 1024 * 1024
    assert a.max_tokens_cap == 4096
    assert a.socket_timeout == 600.0
    a = ap.parse_args(["--host", "0.0.0.0", "--api-key", "k", "--max-body-bytes", "1024", "--max-tokens-cap", "16", "--socket-timeout", "5"])
    assert (a.host, a.api_key, a.max_body_bytes, a.max_tokens_cap, a.socket_timeout) == ("0.0.0.0", "k", 1024, 16, 5.0)


# ----------------------------------------------------------------------------- serve_batched.py handler wiring


class _FakeReq:
    """What BatchedEngine.submit() returns, as the handler sees it: finished immediately with max_new tokens."""

    def __init__(self, ids, max_new, eos):
        self.ids, self.max_new, self.eos = list(ids), max_new, set(eos)
        self.out = list(range(1000, 1000 + max_new))
        self.n_prompt, self.start, self.slot = len(self.ids), 0, 0
        self.t_submit = time.time()
        self.t_first = self.t_submit + 0.001
        self.err = None
        self.started = True
        self.done = threading.Event()
        self.done.set()
        self.tokens = queue.Queue()
        for t in self.out:
            self.tokens.put(t)
        self.tokens.put(None)


class _FakeEng:
    log_admits = False
    q, active, parked, prefills, host_parked = [], {}, {}, {}, {}
    cache_hits = cache_tokens = host_bytes = host_hits = host_saves = host_evict_lru = host_evict_ttl = 0
    host_save_s = host_restore_s = host_save_bytes = host_restore_bytes = 0.0
    park_ram_ttl = park_ram_bytes = prefilled_tokens = decoded_tokens = steps = rounds = spec_tokens = spec_cols = spec_fallbacks = 0
    spec = spec_on = False
    spec_k = "auto"
    spec_draft_s = spec_verify_s = spec_commit_s = 0.0

    def _stats(self):
        return 10, 10, 128, 0

    def submit(self, ids, max_new, eos):
        return _FakeReq(ids, max_new, eos)

    def cancel_queued(self, req):
        return False


class _FakeTok:
    eos_token_id = 2

    def convert_tokens_to_ids(self, name):
        return 3

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        return " ".join(m.get("content", "") for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[7] * max(1, len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


@pytest.fixture(scope="module")
def batched_server():
    import serve_batched

    args = types.SimpleNamespace(model_name="fake", timeout=5.0, api_key="k3y", max_body_bytes=4096, max_tokens_cap=16,
                                 socket_timeout=5.0)
    H = serve_batched.make_handler(_FakeEng(), _FakeTok(), args)
    assert H.timeout == 5.0  # the socket timeout is installed on the handler class
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address
    httpd.shutdown()


def _req(addr, method, path, body=None, headers=None):
    c = http.client.HTTPConnection(*addr, timeout=10)
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    hdr = dict(headers or {})
    if data is not None and "Content-Length" not in hdr:
        hdr["Content-Length"] = str(len(data))
    c.request(method, path, body=data, headers=hdr)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw) if raw else None, r.getheaders()
    except ValueError:
        return r.status, raw.decode(errors="replace"), r.getheaders()


AUTH = {"Authorization": "Bearer k3y"}


def test_batched_health_is_open_and_v1_needs_key(batched_server):
    status, body, _ = _req(batched_server, "GET", "/health")
    assert status == 200 and body["status"] == "ok"
    status, body, headers = _req(batched_server, "GET", "/v1/models")
    assert status == 401 and body["error"]["type"] == "authentication_error"
    assert dict(headers).get("WWW-Authenticate") == "Bearer"
    status, body, _ = _req(batched_server, "GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
    assert status == 401
    status, body, _ = _req(batched_server, "GET", "/v1/models", headers=AUTH)
    assert status == 200 and body["data"][0]["id"] == "fake"
    # the run-time switch is protected as well
    status, body, _ = _req(batched_server, "GET", "/spec?on=0")
    assert status == 401
    status, body, _ = _req(batched_server, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 401


def test_batched_body_limit(batched_server):
    big = b'{"messages": [], "pad": "' + b"x" * 5000 + b'"}'
    status, body, _ = _req(batched_server, "POST", "/v1/chat/completions", big, headers=AUTH)
    assert status == 413 and "--max-body-bytes" in body["error"]["message"]
    status, body, _ = _req(batched_server, "POST", "/v1/chat/completions", b"{nope", headers=AUTH)
    assert status == 400 and "bad json" in body["error"]["message"]
    c = http.client.HTTPConnection(*batched_server, timeout=10)
    c.putrequest("POST", "/v1/chat/completions")
    c.putheader("Authorization", "Bearer k3y")
    c.endheaders()                       # no Content-Length at all
    r = c.getresponse()
    assert r.status == 411
    c.close()


def test_batched_max_tokens_clamp_reported(batched_server):
    msg = {"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 500}
    status, body, _ = _req(batched_server, "POST", "/v1/chat/completions", msg, headers=AUTH)
    assert status == 200
    assert body["x_qwentin"]["max_tokens_clamped_from"] == 500
    assert body["x_qwentin"]["max_tokens_cap"] == 16
    assert body["usage"]["completion_tokens"] == 16
    assert body["choices"][0]["finish_reason"] == "length"
    # under the cap: no clamp fields
    msg["max_tokens"] = 4
    status, body, _ = _req(batched_server, "POST", "/v1/chat/completions", msg, headers=AUTH)
    assert status == 200 and "max_tokens_clamped_from" not in body["x_qwentin"]
    assert body["usage"]["completion_tokens"] == 4
    # streaming: the clamp rides in the final chunk's x_qwentin
    msg["max_tokens"] = 99
    msg["stream"] = True
    status, text, _ = _req(batched_server, "POST", "/v1/chat/completions", msg, headers=AUTH)
    assert status == 200
    chunks = [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:") and line[5:].strip() != "[DONE]"]
    final = [c for c in chunks if "x_qwentin" in c]
    assert final and final[-1]["x_qwentin"]["max_tokens_clamped_from"] == 99
    assert final[-1]["choices"][0]["finish_reason"] == "length"
