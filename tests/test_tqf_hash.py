"""The engine's BLAKE2b-256 (qwn_tqf_blake2b256, the .tqf integrity hash) equals hashlib's around the 128-byte block
edges. Needs a built library: $QWENTIN_LIB or build-qwen/libforward_qwen.so (skipped otherwise, e.g. in the CPU CI)."""

from __future__ import annotations

import ctypes
import hashlib
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("QWENTIN_LIB", os.path.join(ROOT, "build-qwen", "libforward_qwen.so"))


@pytest.mark.skipif(not os.path.exists(LIB), reason="no built libforward_qwen.so")
def test_blake2b256_matches_hashlib():
    lib = ctypes.CDLL(LIB)
    lib.qwn_tqf_blake2b256.restype = ctypes.c_int
    lib.qwn_tqf_blake2b256.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    data = bytes((i * 131 + 7) & 0xFF for i in range(1 << 17))
    for n in (0, 1, 63, 64, 127, 128, 129, 255, 256, 257, 1000, 1 << 17):
        out = (ctypes.c_uint8 * 32)()
        assert lib.qwn_tqf_blake2b256(data[:n], n, out) == 0
        assert bytes(out) == hashlib.blake2b(data[:n], digest_size=32).digest(), n


def _tiny_v1_header(flags: int = 0x1c1d) -> bytes:
    import struct
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import inspect_tqf as it
    fixed = struct.pack(it.FIXED_HEADER_FMT, flags, 1, 2, 64, 128, 2, 256, 2, 1, 32, 2, 16, 16, 16, 4, 4096,
                        1e-6, 10000.0, 0.25, 0, 0, 0, 0, 0, 1, 1, 1)
    return b"TQF1" + struct.pack("<I", 8 + len(fixed) + 2) + fixed + bytes([1, 2])


@pytest.mark.skipif(not os.path.exists(LIB), reason="no built libforward_qwen.so")
def test_loader_refuses_bad_v2_files(tmp_path, monkeypatch):
    """qwn_probe (parse without upload, no GPU needed) on crafted files: each defect gets its own code."""
    import struct
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import inspect_tqf as it
    lib = ctypes.CDLL(LIB)
    lib.qwn_probe.restype = ctypes.c_int
    lib.qwn_probe.argtypes = [ctypes.c_char_p]
    payload = bytes(4096)
    good = it.seal_header(it.add_extension(_tiny_v1_header()), len(payload), hashlib.blake2b(payload, digest_size=32).digest())
    ext = len(good) - it.EXT_SIZE

    def probe(name, header, body=payload):
        p = tmp_path / name
        p.write_bytes(header + body)
        return lib.qwn_probe(str(p).encode())

    def resealed(version=2, ext_bytes=it.EXT_SIZE, flags=None):
        h = bytearray(it.add_extension(_tiny_v1_header(flags if flags is not None else 0x1c1d)))
        h[ext:ext + 8] = struct.pack("<II", version, ext_bytes)
        h = it.seal_header(bytes(h), len(payload), hashlib.blake2b(payload, digest_size=32).digest())
        h = bytearray(h); h[ext:ext + 8] = struct.pack("<II", version, ext_bytes)       # seal_header rewrote them
        h[ext + 48:ext + 80] = it.header_hash(bytes(h), ext)
        return bytes(h)

    assert probe("short.tqf", good, payload[:-1]) == -41
    flipped = bytearray(good); flipped[8 + 4 * 15] ^= 1                                 # max_position_embeddings
    assert probe("hdr.tqf", bytes(flipped)) == -42
    assert probe("v3.tqf", resealed(version=3)) == -40
    assert probe("flags.tqf", resealed(flags=0x1c1d | (1 << 20))) == -44
    assert probe("ext.tqf", resealed(ext_bytes=96)) == -45
    monkeypatch.setenv("TQ_VERIFY_TQF", "1")
    assert probe("payload.tqf", good, bytes([1]) + payload[1:]) == -43
    assert probe("ok.tqf", good) not in (-40, -41, -42, -43, -44, -45)    # passes the checks, then the tiny payload fails
