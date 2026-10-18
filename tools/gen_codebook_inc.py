#!/usr/bin/env python3
"""Write src/kv_t4_codebook.inc, the trellis-KV codebook compiled into the engine, from tools/codebooks/lloyd_kv12a.f32
(4096 raw little-endian fp32: the Lloyd-trained 12-bit-state codebook of `--kv-t4`). TQ_KV_T4_CODEBOOK=<file> still
overrides the built-in table at run time; tests/test_codebook_inc.py fails when the two files disagree.

  python3 tools/gen_codebook_inc.py
"""

from __future__ import annotations

import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "tools/codebooks/lloyd_kv12a.f32"
OUT = "src/kv_t4_codebook.inc"
N = 4096


def render(raw: bytes) -> str:
    """The .inc text for the codebook file's bytes; entries are written as fp32 bit patterns so the table is exact."""
    if len(raw) != 4 * N:
        raise ValueError(f"{SRC}: {len(raw)} bytes, want {4 * N}")
    words = struct.unpack(f"<{N}I", raw)
    rows = ["    " + " ".join(f"0x{w:08x}u," for w in words[i:i + 8]) for i in range(0, N, 8)]
    return (f"// Generated from {SRC} by tools/gen_codebook_inc.py -- do not edit.\n"
            f"static const uint32_t k_t4_codebook_lloyd_kv12a[{N}] = {{\n" + "\n".join(rows) + "\n};\n")


if __name__ == "__main__":
    with open(os.path.join(ROOT, SRC), "rb") as f:
        text = render(f.read())
    with open(os.path.join(ROOT, OUT), "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({N} entries)")
