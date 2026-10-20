"""tools/serve_batched.py starts the engine through qwn_init_with: its arguments fill every switch the TQ_* environment
leaves unset or empty, an explicit environment value wins. Needs a built library: $QWENTIN_LIB or
build-qwen/libforward_qwen.so (skipped otherwise, e.g. in the CPU CI); qwn_config_from_env needs no GPU."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("QWENTIN_LIB", os.path.join(ROOT, "build-qwen", "libforward_qwen.so"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import qwentin_abi  # noqa: E402
import serve_batched as sb  # noqa: E402

SERVED = dict(wide_attn_mma=True, page=128, embed_host=True, mtp=True, kv_v4=True, kv_t4=True, kv_t4_codebook=None,
              pdl=True, dstate=4, gdn_prefill=True)
pytestmark = pytest.mark.skipif(not os.path.exists(LIB), reason="no built libforward_qwen.so")


def _config(monkeypatch, env=None, **args):
    for k in list(os.environ):
        if k.startswith("TQ_"):
            monkeypatch.delenv(k)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return sb.engine_config(sb.load_lib(LIB), sb.config_overrides(SimpleNamespace(**{**SERVED, **args})))


def _defaults():
    import ctypes
    c = qwentin_abi.Config()
    lib = sb.load_lib(LIB)
    lib.qwn_config_defaults.argtypes = [ctypes.c_void_p]
    lib.qwn_config_defaults(ctypes.byref(c))
    return c


def _fields(c):
    return {f: getattr(c, f) for f, _ in qwentin_abi.Config._fields_}


def test_the_served_arguments_are_the_engine_defaults(monkeypatch):
    assert _fields(_config(monkeypatch)) == _fields(_defaults())


def test_arguments_fill_what_the_environment_leaves_unset(monkeypatch):
    c = _config(monkeypatch, kv_t4=False, pdl=False, dstate=0, mtp=False, kv_t4_codebook="/x/cb.f32")
    assert (c.kv_t4, c.pdl, c.dstate, c.no_mtp, c.kv_t4_codebook) == (0, 0, 0, 1, b"/x/cb.f32")


def test_explicit_environment_wins(monkeypatch):
    c = _config(monkeypatch, env={"TQ_KV_T4": "0", "TQ_PDL": "2", "TQ_GRAPH": "0"}, kv_t4=True, pdl=True)
    assert (c.kv_t4, c.pdl, c.graph) == (0, 2, 0)


def test_an_empty_variable_counts_as_unset(monkeypatch):
    c = _config(monkeypatch, env={"TQ_PDL": "", "TQ_KV_V4": ""}, pdl=False, kv_v4=False, kv_t4=False)
    assert (c.pdl, c.kv_v4) == (0, 0)


def test_the_wide_attention_needs_128_token_pages(monkeypatch):
    assert _config(monkeypatch, page=64).wide_attn_mma == 0
