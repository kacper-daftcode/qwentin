#!/usr/bin/env python3
"""Reusable converter-side 2:4 WANDA pack for the sparse-FP4 draft path.

Produces exactly the byte layout validated bit-exact by the probe kernel
`tl_probe_qmma_sp_gemv_e2m3` (src/qmma_sp_probe.cu):
  - A: compressed E2M3 values in `group_perm` K-order, 6-bit packed, 3 uint32/lane,
       tq_qmma_a_byte byte placement; stride Kt64*96 uint32 per m16 tile.
  - metadata: ordered_metadata nibble `(hi<<2)|lo` at the placement closed form
       lane=(r>>1)*4+(r&1)+((c>>1)&1)*2, nibble=(c&1)*4+(c>>2).

Run directly for the self-test (signed weights, multi K-tile, bit-exact vs the
dense kept-mask reference through the real kernel).
"""

from __future__ import annotations

import math

import numpy as np

GROUP_PERM = [0, 4, 2, 6, 8, 12, 10, 14, 1, 5, 3, 7, 9, 13, 11, 15]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def e2m3_code(v: float) -> int:
    """Signed E2M3 (1 sign, 2 exp bias-1, 3 mantissa). Exact on the E2M1 grid."""
    s = 0
    if v < 0:
        s, v = 1, -v
    if v == 0.0:
        return s << 5
    if v < 1.0:  # subnormal e=0
        e, m = 0, int(round(v * 8))
        if m == 8:
            e, m = 1, 0
    else:
        e = int(math.floor(math.log2(v))) + 1
        m = int(round((v / 2.0 ** (e - 1) - 1.0) * 8))
        if m == 8:
            e, m = e + 1, 0
    if e > 3:  # saturate (should not happen after block scaling)
        e, m = 3, 7
    return (s << 5) | (e << 3) | m


def e4m3_code(v: float) -> int:
    s = 0
    if v < 0:
        s, v = 1, -v
    if v == 0.0:
        return s << 7
    e = int(math.floor(math.log2(v)))
    E = e + 7
    m = int(round((v / 2.0 ** e - 1.0) * 8))
    if m == 8:
        E, m = E + 1, 0
    E = max(0, min(15, E))
    return (s << 7) | (E << 3) | m


def meta_lane_nib(rl: int, c: int):
    """(local row 0..15, k64-group 0..15) -> (lane 0..31, nibble-slot 0..7)."""
    return (rl >> 1) * 4 + (rl & 1) + ((c >> 1) & 1) * 2, (c & 1) * 4 + (c >> 2)


def wanda_2of4_mask(W: np.ndarray, act_norm: np.ndarray | None = None) -> np.ndarray:
    """Per (row, group-of-4 along K) keep the 2 highest-importance lanes.
    Importance = |W| * ||act||_2 (WANDA); falls back to |W| if act_norm is None.
    Returns int array [M, K//4, 2] of sorted kept indices (lo<hi)."""
    M, K = W.shape
    imp = np.abs(W).astype(np.float64)
    if act_norm is not None:
        imp = imp * act_norm[None, :]
    g = imp.reshape(M, K // 4, 4)
    # keep top-2 per group; return sorted (lo<hi)
    keep = np.sort(np.argsort(-g, axis=2)[:, :, :2], axis=2)
    return keep.astype(np.int64)


_GP = np.array(GROUP_PERM, np.int64)


def _build_idx_tables():
    """Fixed index maps (data-independent): (rl,cp,slot)->(lane,byte) for the
    compressed-A fragment, (rl,cp)->(lane,nibble) for ordered metadata, and the
    logical group c=group_perm[cp]."""
    rl = np.arange(16)[:, None, None]
    cp = np.arange(16)[None, :, None]
    sl = np.arange(2)[None, None, :]
    c = _GP[cp]
    jc = 2 * cp + sl
    rl_b, c_b, jc_b = (np.broadcast_to(a, (16, 16, 2)) for a in (rl, c, jc))
    lane = (rl_b >> 1) * 4 + (jc_b >> 3)
    byte = ((jc_b & 7) >> 2) * 8 + (rl_b & 1) * 4 + (jc_b & 3)
    rlm = np.arange(16)[:, None]
    cm = _GP[np.arange(16)[None, :]]
    lane2 = (rlm >> 1) * 4 + (rlm & 1) + ((cm >> 1) & 1) * 2
    nib = (cm & 1) * 4 + (cm >> 2)
    return (lane.astype(np.int64), byte.astype(np.int64), c_b.astype(np.int64),
            np.broadcast_to(lane2, (16, 16)).astype(np.int64),
            np.broadcast_to(nib, (16, 16)).astype(np.int64))


_LANE, _BYTE, _CSLOT, _LANE2, _NIB = _build_idx_tables()


def pack_2of4_e2m3_codes(codes: np.ndarray, mask: np.ndarray):
    """codes[M,K] uint8 E2M3 codes (already quantized), mask[M,K//4,2].
    Returns (a_u32[n_mt*Kt64*96], meta_u32[n_mt*Kt64*32]). Vectorized (numpy)
    equivalent of the per-tile loop; the kept values are the given codes verbatim,
    so the draft is a true sub-read of the dense E2M3 payload."""
    M, K = codes.shape
    assert M % 16 == 0 and K % 64 == 0, f"sparse pack needs M%16==0, K%64==0; got {M}x{K}"
    n_mt, Kt64 = M // 16, K // 64
    full = (n_mt, Kt64, 16, 16, 2)
    mt = np.arange(n_mt)[:, None, None, None, None]
    tk = np.arange(Kt64)[None, :, None, None, None]
    rl = np.arange(16)[None, None, :, None, None]
    sl = np.arange(2)[None, None, None, None, :]
    c = _CSLOT[None, None]
    row = np.broadcast_to(mt * 16 + rl, full)
    g = np.broadcast_to(tk * 16 + c, full)
    sl_b = np.broadcast_to(sl, full)
    kept = mask[row, g, sl_b]
    code_vals = codes[row, 4 * g + kept]
    frag = np.zeros((n_mt, Kt64, 32, 16), np.uint8)
    frag[np.broadcast_to(mt, full), np.broadcast_to(tk, full),
         np.broadcast_to(_LANE[None, None], full), np.broadcast_to(_BYTE[None, None], full)] = code_vals
    aw = np.zeros((n_mt, Kt64, 32, 3), np.uint64)
    for j in range(16):
        bit = 6 * j
        word = bit >> 5
        off = bit & 31
        cj = frag[..., j].astype(np.uint64)
        aw[..., word] |= cj << np.uint64(off)
        if off > 26:
            aw[..., word + 1] |= cj >> np.uint64(32 - off)
    a = (aw & np.uint64(0xFFFFFFFF)).astype(np.uint32).reshape(-1)

    full2 = (n_mt, Kt64, 16, 16)
    mt2 = np.arange(n_mt)[:, None, None, None]
    tk2 = np.arange(Kt64)[None, :, None, None]
    rlm = np.arange(16)[None, None, :, None]
    cm = _GP[np.arange(16)[None, None, None, :]]
    row2 = np.broadcast_to(mt2 * 16 + rlm, full2)
    g2 = np.broadcast_to(tk2 * 16 + cm, full2)
    val = ((mask[:, :, 1].astype(np.uint32) << 2) | mask[:, :, 0].astype(np.uint32))
    valg = val[row2, g2]
    nibbles = np.zeros((n_mt, Kt64, 32, 8), np.uint32)
    nibbles[np.broadcast_to(mt2, full2), np.broadcast_to(tk2, full2),
            np.broadcast_to(_LANE2[None, None], full2), np.broadcast_to(_NIB[None, None], full2)] = valg
    mw = np.zeros((n_mt, Kt64, 32), np.uint32)
    for nib in range(8):
        mw |= nibbles[..., nib] << np.uint32(4 * nib)
    return a, mw.reshape(-1)


def pack_2of4_e2m3(Wg: np.ndarray, mask: np.ndarray):
    """Wg[M,K] on the E2M3/E2M1 grid (post block-scale division), mask[M,K//4,2]."""
    M, K = Wg.shape
    codes = np.zeros((M, K), np.uint8)
    for r in range(M):
        for k in range(K):
            codes[r, k] = e2m3_code(float(Wg[r, k]))
    return pack_2of4_e2m3_codes(codes, mask)


def pack_acts_e4m3(x: np.ndarray) -> np.ndarray:
    """x[K] -> b_u32[Kt64*128] (n=0 column at contiguous bytes 0..63 per k64-tile)."""
    K = x.shape[0]
    Kt64 = K // 64
    b = np.zeros(Kt64 * 128, np.uint32)
    for tk in range(Kt64):
        bb = np.zeros(512, np.uint8)
        for k in range(64):
            bb[k] = e4m3_code(float(x[tk * 64 + k]))
        b[tk * 128:(tk + 1) * 128] = bb.view(np.uint32)
    return b


def _self_test():
    import ctypes as C
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_qmma_sp import Driver, compile_cubin

    ROOT = Path(__file__).resolve().parents[1]
    CUBIN = ROOT / "build-qwen/qmma_sp_probe.cubin"
    compile_cubin(ROOT / "src/qmma_sp_probe.cu", CUBIN)

    E2M1 = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    XV = np.array([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3], np.float32)
    rng = np.random.default_rng(5)
    M, K = 32, 256
    Kt64 = K // 64
    Wg = rng.choice(E2M1, size=(M, K)).astype(np.float32)  # signed, on grid
    x = rng.choice(XV, size=K).astype(np.float32)
    mask = wanda_2of4_mask(Wg, act_norm=np.abs(x) + 0.1)  # WANDA selection
    pows = np.array([0.5, 1.0, 2.0], np.float32)
    n_sr = ((M // 16 - 1) >> 3) + 1
    wscale = rng.choice(pows, size=(n_sr, Kt64)).astype(np.float32)
    ascale = rng.choice(pows, size=Kt64).astype(np.float32)

    ref = np.zeros(M, np.float64)
    for r in range(M):
        sr = (r // 16) >> 3
        for tk in range(Kt64):
            acc = 0.0
            for c in range(16):
                g = tk * 16 + c
                lo, hi = int(mask[r, g, 0]), int(mask[r, g, 1])
                acc += Wg[r, 4 * g + lo] * x[4 * g + lo] + Wg[r, 4 * g + hi] * x[4 * g + hi]
            ref[r] += float(wscale[sr, tk]) * float(ascale[tk]) * acc

    a, meta = pack_2of4_e2m3(Wg, mask)
    b = pack_acts_e4m3(x)

    drv = Driver()
    out = np.zeros(M, np.float32)
    d_out, d_a, d_b, d_meta = (drv.alloc_copy(out), drv.alloc_copy(a),
                               drv.alloc_copy(b), drv.alloc_copy(meta))
    d_ws, d_as = drv.alloc_copy(wscale.reshape(-1).copy()), drv.alloc_copy(ascale.copy())
    mod, fn = C.c_void_p(), C.c_void_p()
    drv._ck(drv.cuda.cuModuleLoad(C.byref(mod), str(CUBIN).encode()), "load")
    drv._ck(drv.cuda.cuModuleGetFunction(C.byref(fn), mod, b"tl_probe_qmma_sp_gemv_e2m3"), "fn")
    M_c, Kt_c, ws_c = C.c_int(M), C.c_int(Kt64), C.c_int(Kt64)
    raw = [d_out, d_a, d_b, d_meta, d_ws, d_as, M_c, Kt_c, ws_c]
    storage = [C.c_uint64(v.value) if isinstance(v, C.c_uint64) else v for v in raw]
    pp = (C.c_void_p * len(storage))()
    for i, s in enumerate(storage):
        pp[i] = C.cast(C.pointer(s), C.c_void_p)
    drv._ck(drv.cuda.cuLaunchKernel(fn, M // 16, 1, 1, 32, 1, 1, 0, None, pp, None), "launch")
    drv._ck(drv.cuda.cuCtxSynchronize(), "sync")
    got = drv.read(d_out, (M,), np.float32)
    err = float(np.abs(got - ref).max())
    print(f"signed E2M3, WANDA 2:4, M={M} K={K} Kt64={Kt64}")
    print(f"ref[:5]={[round(float(v),2) for v in ref[:5]]}")
    print(f"got[:5]={[round(float(v),2) for v in got[:5]]}")
    print(f"max_err={err:.5f}   SPARSE_PACK_PASS={err < 1e-3}")


if __name__ == "__main__":
    _self_test()
