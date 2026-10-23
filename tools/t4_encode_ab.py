#!/usr/bin/env python3
"""The engine's trellis page encoder (k_t4_encode, the fused Viterbi) on fixed rows: codes + scales of one library, for
a bitwise A/B between builds (session 17, the cheaper encoder). The rows go through qwn_paged_t4_test_encode (staging ->
encoder -> pool, full-attention layer 0), <= page * nkv * 2 = 1024 per call; --batches seeded batches cover 1024 x
batches rows: Gaussian with a per-row scale spread, heavy tails with an outlier channel, values on a coarse grid (exact
cost ties -> the first-occurrence rule), near-zero rows (the rms clamp), constant rows, a zero row.
usage: t4_encode_ab.py LIB OUT.npz --tqf MODEL.tqf [--batches 8] [--reps 0]
       t4_encode_ab.py --cmp A.npz B.npz
--reps R re-runs the last batch R more times (for nsys / ncu on k_t4_encode); the served environment (no TQ_*): T4 is the default.
Vendored from the qwen256k research harness (analysis/t4_encode_ab.py)."""
import argparse, ctypes, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("lib", nargs="?")
ap.add_argument("out", nargs="?")
ap.add_argument("--tqf", default="/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf")
ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--reps", type=int, default=0)
ap.add_argument("--cmp", nargs=2)
a = ap.parse_args()

if a.cmp:
    A = np.load(a.cmp[0]); B = np.load(a.cmp[1])
    same_c = (A["codes"] == B["codes"]).all(1); same_s = A["scales"] == B["scales"]
    n = len(same_c)
    print(f"[t4-ab] rows {n}: codes identical {same_c.sum()} / {n}, scales identical {same_s.sum()} / {n} -> "
          f"{'BITWISE' if same_c.all() and same_s.all() else 'DIFFERENT'}")
    if not same_c.all():
        bad = np.where(~same_c)[0]
        print(f"   first differing rows {bad[:8].tolist()}; differing bytes per row (first 8): "
              f"{[int((A['codes'][i] != B['codes'][i]).sum()) for i in bad[:8]]}")
    sys.exit(0 if same_c.all() and same_s.all() else 1)

os.environ.setdefault("TQ_NO_MTP", "1")
L = ctypes.CDLL(a.lib)
L.qwn_init.argtypes = [ctypes.c_char_p]
L.qwn_paged_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
L.qwn_paged_t4_test_encode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
assert L.qwn_init(a.tqf.encode()) == 0, "qwn_init"
rc = L.qwn_paged_init(2, 8, 128)
assert rc == 0, f"qwn_paged_init rc={rc}"
N = 1024                                                                     # page 128 x nkv 4 x {K, V}


def rows(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N, 256, device="cuda", generator=g)
    x[:256] *= torch.exp(torch.randn(256, 1, device="cuda", generator=g))      # per-row scale spread
    x[256:384] *= 3.0
    x[256:264, 5] += 40.0                                                      # an outlier channel
    x[384:512] = torch.round(x[384:512] * 2.0) / 2.0                           # a coarse grid: exact cost ties
    x[512:640] = torch.round(x[512:640] * 0.5) * 2.0                           # coarser still
    x[640:768] *= 1e-6                                                         # near zero (the rms clamp region)
    x[768:896] = x[768:896, :1]                                                # constant rows
    x[896:1023] = torch.randn(127, 256, device="cuda", generator=g) * torch.linspace(0.01, 30.0, 127, device="cuda")[:, None]
    x[1023] = 0.0                                                              # a zero row
    return x.to(torch.float16).cpu().numpy().view(np.uint16)


codes = np.zeros((a.batches, N, 129), dtype=np.uint8)
scales = np.zeros((a.batches, N), dtype=np.uint16)
for b in range(a.batches):
    r = np.ascontiguousarray(rows(1000 + b))
    rc = L.qwn_paged_t4_test_encode(r.ctypes.data, N, codes[b].ctypes.data, scales[b].ctypes.data)
    assert rc == 0, f"qwn_paged_t4_test_encode rc={rc} (batch {b})"
np.savez(a.out, codes=codes.reshape(-1, 129), scales=scales.reshape(-1))
print(f"[t4-ab] {a.batches * N} rows encoded -> {a.out}; distinct code bytes {len(np.unique(codes))}, "
      f"scale range {scales.view(np.float16).astype(np.float32).min():.3g} .. {scales.view(np.float16).astype(np.float32).max():.3g}")
if a.reps:
    r = np.ascontiguousarray(rows(1000 + a.batches - 1))
    c = np.zeros((N, 129), dtype=np.uint8); s = np.zeros(N, dtype=np.uint16)
    t0 = time.perf_counter()
    for _ in range(a.reps):
        assert L.qwn_paged_t4_test_encode(r.ctypes.data, N, c.ctypes.data, s.ctypes.data) == 0
    dt = (time.perf_counter() - t0) / a.reps
    print(f"[t4-ab] {a.reps} more calls of 1024 rows: {dt * 1e3:.3f} ms per call (host wall incl. the copies; the kernel time: nsys / ncu)")
