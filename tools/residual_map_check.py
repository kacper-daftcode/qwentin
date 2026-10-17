#!/usr/bin/env python3
"""Parity check of the engine-side residual-stream hook (qwn_ot_hook_add / k_tq_ot_hook2;
the engine symbols keep their historical names until the deferred engine-side rename).

Arms the maps file exactly like serve_openai.py --residual-map would, then drives
qwn_ot_apply_debug with random residual rows and compares the device result against the
fp64 numpy reference of the same affine map (tools/residual_map.py apply_reference):

    y = P^T (x - mu);  z = (A - I)^T y;  x += scale * (shift + P z)

Usage:
    python3 tools/residual_map_check.py --lib build-qwen/libforward_qwen.so \
        --tqf /path/to/model.tqf --residual-map maps.json [--layers 37,38] [--scale 0.3]

Exit 0 = parity within fp32 reduction tolerance (1e-3 absolute) on every checked layer.
Run it on every new maps file before serving it; GPU required (the hook runs on the device).
"""
from __future__ import annotations
import argparse, ctypes, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, ck  # noqa: E402
from residual_map import load_residual_map, arm_residual_map, apply_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", required=True)
ap.add_argument("--tqf", required=True)
ap.add_argument("--residual-map", "--maps", dest="residual_map", required=True, help="the maps JSON")
ap.add_argument("--layers", default=None, help="override the file's hooked layers (comma-separated)")
ap.add_argument("--scale", type=float, default=None, help="override the file's scale (any non-zero value checks the kernel)")
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

import numpy as np

LIB = load_lib(args.lib)
ck(LIB.qwn_init(args.tqf.encode()), "init")
H = int(LIB.qwn_hidden_size())

rm = load_residual_map(args.residual_map, layers=args.layers, scale=args.scale, hidden_size=H)
armed = arm_residual_map(LIB, rm)
LIB.qwn_ot_apply_debug.restype = ctypes.c_int
LIB.qwn_ot_apply_debug.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

rng = np.random.default_rng(args.seed)
ok = True
print(f"armed layers {armed} at scale {rm['scale']:g} from {args.residual_map} (H={H})")
for ln in armed:
    # realistic residual magnitudes: unit-normal rows scaled up a bit
    x = (rng.standard_normal((args.rows, H)) * 2.0).astype(np.float32)
    ref = apply_reference(x, rm["maps"][ln], rm["scale"])
    got = x.copy()
    ck(LIB.qwn_ot_apply_debug(ln, args.rows, got.ctypes.data_as(ctypes.POINTER(ctypes.c_float))), f"apply_debug {ln}")

    d = np.abs(got.astype(np.float64) - ref)
    denom = np.abs(ref).max()
    print(f"layer {ln}: max|diff|={d.max():.3e}  (rel {d.max() / max(denom, 1e-12):.3e}, "
          f"ref|max|={denom:.3f})  effect|max|={np.abs(got - x).max():.4f}")
    if d.max() > 1e-3:
        ok = False
        print(f"layer {ln}: FAIL (fp32 reduction tolerance 1e-3 exceeded)")

print("PARITY OK" if ok else "PARITY FAIL")
sys.exit(0 if ok else 1)
