"""The engine's built-in trellis-KV codebook (src/kv_t4_codebook.inc) is tools/codebooks/lloyd_kv12a.f32 bit for bit."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gen_codebook_inc as g  # noqa: E402


def test_builtin_codebook_matches_the_shipped_file():
    with open(os.path.join(ROOT, g.SRC), "rb") as f:
        expected = g.render(f.read())
    with open(os.path.join(ROOT, g.OUT)) as f:
        assert f.read() == expected, "src/kv_t4_codebook.inc is stale: run python3 tools/gen_codebook_inc.py"
