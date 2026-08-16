#!/usr/bin/env python3
"""Parity gate for the engine-side LLobotomy bark (qwn_ot_hook_add / k_tq_ot_hook2).

Arms an OT hook from a save_maps JSON exactly like serve_openai.py --bark-all-day
would, then drives qwn_ot_apply_debug with random residual rows and compares the
device result against the fp32/fp64 numpy reference of the lab formula:

    y = P^T (x - mu_H);  z = (A_k - I)^T y;  x += scale * (mean_shift + P z)

Usage:
    python3 tools/ot_hook_check.py --lib build-qwen13/libforward_qwen.so \
        --tqf /root/models/qwen3_8-27b-fp6-e2m3-mtp.tqf --maps /tmp/maps_q38.json

Exit 0 = parity within fp32 reduction tolerance on every checked layer.
"""
from __future__ import annotations
import argparse, ctypes, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, ck  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True)
ap.add_argument("--tqf", required=True)
ap.add_argument("--maps", required=True)
ap.add_argument("--layers", default="37,38")
ap.add_argument("--scale", type=float, default=0.21)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

import numpy as np

LIB = load_lib(args.lib)
ck(LIB.qwn_init(args.tqf.encode()), "init")
H = int(LIB.qwn_hidden_size())

LIB.qwn_ot_hook_add.restype = ctypes.c_int
LIB.qwn_ot_hook_add.argtypes = [ctypes.c_int, ctypes.c_float] + [ctypes.POINTER(ctypes.c_float)] * 4
LIB.qwn_ot_apply_debug.restype = ctypes.c_int
LIB.qwn_ot_apply_debug.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]


def f32p(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


maps = json.load(open(args.maps))["ot_maps"]
rng = np.random.default_rng(args.seed)
ok = True
for ln in [int(x) for x in args.layers.split(",") if x.strip()]:
    m = maps[str(ln)]
    P = np.ascontiguousarray(np.asarray(m["P"], dtype=np.float32))            # H x 2
    A = np.ascontiguousarray(np.asarray(m["A_k_minus_I"], dtype=np.float32))  # 2 x 2
    mu = np.ascontiguousarray(np.asarray(m["mu_H"], dtype=np.float32))
    ms = np.ascontiguousarray(np.asarray(m["mean_shift"], dtype=np.float32))
    assert P.shape == (H, 2) and A.shape == (2, 2) and mu.shape == (H,) and ms.shape == (H,)
    ck(LIB.qwn_ot_hook_add(ln, ctypes.c_float(args.scale), f32p(P), f32p(A), f32p(mu), f32p(ms)),
       f"hook_add {ln}")

    # realistic residual magnitudes: unit-normal rows scaled up a bit
    x = (rng.standard_normal((args.rows, H)) * 2.0).astype(np.float32)
    ref = x.astype(np.float64) + args.scale * (
        ms.astype(np.float64)[None, :]
        + ((x.astype(np.float64) - mu.astype(np.float64)[None, :]) @ P.astype(np.float64)
           @ A.astype(np.float64).T) @ P.astype(np.float64).T)
    got = x.copy()
    ck(LIB.qwn_ot_apply_debug(ln, args.rows, f32p(got)), f"apply_debug {ln}")

    d = np.abs(got.astype(np.float64) - ref)
    denom = np.abs(ref).max()
    print(f"layer {ln}: max|diff|={d.max():.3e}  (rel {d.max() / max(denom, 1e-12):.3e}, "
          f"ref|max|={denom:.3f})  effect|max|={np.abs(got - x).max():.4f}")
    if d.max() > 1e-3:
        ok = False
        print(f"layer {ln}: FAIL (fp32 reduction tolerance 1e-3 exceeded)")

print("PARITY OK" if ok else "PARITY FAIL")
sys.exit(0 if ok else 1)
