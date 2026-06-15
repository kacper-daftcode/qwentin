#!/usr/bin/env python3
"""Step 2 of docs/MTP_SPECDEC_PLAN.md: build the MTP covering tree in the engine
and MEASURE the realized accept-length of the DENSE-GREEDY path inside it.

This is the DECISIVE input for the whole scheme's speedup (~3x target). It needs
ONLY the cheap MTP head (Step 1, done) + forking the MTP head's OWN KV per branch
(implicit in the engine's depth-first tree walk) -- NOT the Step-3 forking-DeltaNet
verify.

For each of several decode positions p along a real prompt the smoke:
  1. Drives the engine's DENSE decode over the prompt up to p (teacher-forced),
     building the MTP causal trunk (slots 0..p-1) with qwn_mtp_advance, exactly the
     per-position MTP state the validated Step-1 on-policy path builds.
  2. Snapshots the root hidden h_p and free-runs the DENSE model D steps from the
     main-model free token (greedy at p) to get the TRUE dense-greedy continuation
     g_1..g_D  (g_d = the dense token at position p+1+d).
  3. (B, PRIMARY) Builds a margin-shaped covering MTP tree of depth D rooted at h_p
     (qwn_mtp_tree_build) and measures the realized accept-length = how deep the
     dense-greedy path g_1..g_D stays inside the tree (covered + on an expanded
     node at each level).
  4. (A, CROSS-CHECK) Feeds the dense path [seed,g_1,..,g_{D-1}] through the MTP head
     (qwn_mtp_path_topk) and records per-depth top-1 / top-8 coverage -- the
     "teacher-forced" covering upper bound that should reproduce the plan's model
     (top-8 coverage 95.6 -> 68.6 over d1..d4) and validate the KV forking.

Reset-per-position keeps the measurement obviously correct: the model has DeltaNet
(linear-attention) recurrent state that cannot be rewound, so every position starts
from qwn_reset_state + a fresh prefill rather than rewinding a free-run.

GATE / expectation (per the plan): with depth ~3-4 the mean accept-length (committed
tokens/round, INCLUDING the +1 main-model free token) should land ~3-3.8; depth-1
coverage should reflect MTP top-8 (~95%). If much lower (<~2.5), investigate tree
shape / margin threshold / KV forking before reporting.

Run (GPU 6 only):
    CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python3 -u tools/mtp_tree_smoke.py \
        --model-dir /workspace/models/Qwen3.6-27B \
        --tqf /workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np

BIGTEXT = "/tmp/bigtext.txt"
FALLBACK_TEXT = (
    "The history of science is full of surprising reversals. In 1905 a young clerk "
    "at the Swiss patent office published four papers that reshaped physics. Light, "
    "he argued, behaves as discrete quanta; matter and energy are interchangeable. "
    "Meanwhile, biologists were rediscovering Mendel, economists were debating gold, "
    "and engineers in Detroit were learning to build cars on a moving line. Consider "
    "a train that travels 60 kilometers in 45 minutes: its average speed is 80 km/h, "
    "because 60 divided by 0.75 equals 80. Programming offers similar clarity. Here is "
    "a function that returns the n-th Fibonacci number using fast doubling, which runs "
    "in O(log n) time rather than the naive linear scan. Markets, genomes, and source "
    "code all reward the same habit: state the invariant, then prove it holds."
)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------- engine ABI
def load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_hidden_size.restype = ctypes.c_int
    lib.qwn_vocab_size.restype = ctypes.c_int
    lib.qwn_has_mtp.restype = ctypes.c_int
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_mtp_reset.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_mtp_advance.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_mtp_advance.restype = ctypes.c_int
    lib.qwn_mtp_snapshot_root.argtypes = []
    lib.qwn_mtp_snapshot_root.restype = ctypes.c_int
    lib.qwn_mtp_tree_build.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwn_mtp_tree_build.restype = ctypes.c_int
    lib.qwn_mtp_path_topk.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwn_mtp_path_topk.restype = ctypes.c_int
    return lib


# ---------------------------------------------------------------- engine wrappers
def decode(lib, tok: int, pos: int) -> int:
    r = lib.qwn_decode(int(tok), int(pos))
    if r < 0:
        raise RuntimeError(f"qwn_decode({tok},{pos}) failed: {r}")
    return r


def mtp_advance(lib, tok: int, pos: int):
    r = lib.qwn_mtp_advance(int(tok), int(pos))
    if r != 0:
        raise RuntimeError(f"qwn_mtp_advance({tok},{pos}) failed: {r}")


def tree_build(lib, seed, pos, depth, k, tau, max_nodes):
    cap = max_nodes
    toks = (ctypes.c_int * cap)()
    pars = (ctypes.c_int * cap)()
    deps = (ctypes.c_int * cap)()
    mrg = (ctypes.c_float * cap)()
    n = lib.qwn_mtp_tree_build(int(seed), int(pos), int(depth), int(k),
                               ctypes.c_float(tau), int(max_nodes), toks, pars, deps, mrg)
    if n < 0:
        raise RuntimeError(f"qwn_mtp_tree_build failed: {n}")
    return (list(toks[:n]), list(pars[:n]), list(deps[:n]), list(mrg[:n]))


def path_topk(lib, fed_tokens, pos, k):
    n_fed = len(fed_tokens)
    fed = (ctypes.c_int * n_fed)(*[int(t) for t in fed_tokens])
    out_ids = (ctypes.c_int * (n_fed * k))()
    out_vals = (ctypes.c_float * (n_fed * k))()
    r = lib.qwn_mtp_path_topk(fed, n_fed, int(pos), int(k), out_ids, out_vals)
    if r < 0:
        raise RuntimeError(f"qwn_mtp_path_topk failed: {r}")
    ids = np.array(out_ids[: r * k], dtype=np.int64).reshape(r, k)
    vals = np.array(out_vals[: r * k], dtype=np.float32).reshape(r, k)
    return ids, vals


# ----------------------------------------------------------------- accept length
def tree_accept_length(tokens, parents, depths, g, D):
    """Longest prefix of the dense-greedy path g[0..D-1] (g[d-1] = dense token at
    depth d) that stays inside the built tree: at each depth the dense token must be
    a child of the previously-accepted node AND (to descend further) that node must
    be expanded (have children). Returns acc in [0, D]."""
    children = {}
    for i in range(len(tokens)):
        children.setdefault(parents[i], []).append(i)
    cur = -1            # root (the seed / main-model free token)
    acc = 0
    for d in range(1, D + 1):
        gd = g[d - 1]
        match = None
        for ci in children.get(cur, []):
            if depths[ci] == d and tokens[ci] == gd:
                match = ci
                break
        if match is None:
            break       # dense token not covered at this depth
        acc = d
        cur = match
        if d < D and match not in children:
            break       # accepted node was a leaf (pruned/unexpanded) -> can't descend
    return acc


def main():
    ap = argparse.ArgumentParser(description="MTP covering-tree accept-length (Step 2)")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
    ap.add_argument("--prompt", default=None, help="override prompt text")
    ap.add_argument("--max-tokens", type=int, default=320, help="prompt token cap")
    ap.add_argument("--start", type=int, default=24, help="first test position")
    ap.add_argument("--step", type=int, default=20, help="spacing between test positions")
    ap.add_argument("--positions", type=int, default=12, help="number of test positions")
    ap.add_argument("--depth", type=int, default=4, help="max tree depth D (measures 1..D)")
    ap.add_argument("--k", type=int, default=8, help="RECORD width (top-k) at every expanded node")
    ap.add_argument("--taus", default="0,8",
                    help="comma beam thresholds (logit gap): a child is expandable iff "
                         "top1-child<tau (0=greedy spine only, large=best-first covering)")
    ap.add_argument("--node-budgets", default="32,64,120",
                    help="comma node budgets to sweep (memory-bound column caps)")
    args = ap.parse_args()

    D = args.depth
    K = args.k
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    budgets = [int(x) for x in args.node_budgets.split(",") if x.strip()]
    # (tau, budget) configs. The greedy spine (tau=0) only expands the top-1 chain
    # (D*k nodes) so it is budget-insensitive -> include it once; sweep the budget
    # only for the covering trees (tau>0).
    configs = []
    if any(t == 0 for t in taus):
        configs.append((0.0, max(budgets)))
    for tau in taus:
        if tau == 0:
            continue
        for b in budgets:
            configs.append((tau, b))

    # ---- tokenize ---------------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = args.prompt
    if text is None:
        text = open(BIGTEXT).read() if os.path.exists(BIGTEXT) else FALLBACK_TEXT
    ids = tok(text, add_special_tokens=False).input_ids[: args.max_tokens]
    n = len(ids)
    log(f"prompt tokens: {n}")

    positions = [args.start + i * args.step for i in range(args.positions)]
    positions = [p for p in positions if p + 1 < n]   # need prompt[p] and a free token
    log(f"test positions ({len(positions)}): {positions}")
    if not positions:
        raise SystemExit("no valid test positions (prompt too short)")

    # ---- engine -----------------------------------------------------------------
    lib = load_lib(args.lib)
    log(f"loading engine {args.tqf} ...")
    t0 = time.time()
    if lib.qwn_init(args.tqf.encode()) != 0:
        raise RuntimeError("qwn_init failed")
    log(f"engine loaded in {time.time() - t0:.1f}s")
    try:
        H = int(lib.qwn_hidden_size())
        V = int(lib.qwn_vocab_size())
        if not lib.qwn_has_mtp():
            raise RuntimeError("engine reports no MTP section (regenerate TQF with TQ_EMIT_MTP=1)")
        log(f"engine: H={H} V={V} has_mtp=1")

        # accumulators
        accept = {cf: {d: [] for d in range(1, D + 1)} for cf in configs}  # acc per (tau,budget),D
        tree_sizes = {cf: [] for cf in configs}
        tf_top1 = [[] for _ in range(D)]    # teacher-forced per-depth top-1 hit
        tf_top8 = [[] for _ in range(D)]    # teacher-forced per-depth top-8 hit
        tf_cumul = [[] for _ in range(D)]   # teacher-forced cumulative top-8 (path covered 1..d)
        root_margins = []

        tstart = time.time()
        for pi, p in enumerate(positions):
            # 1. fresh state + prefill 0..p, building the MTP trunk (slots 0..p-1).
            if lib.qwn_reset_state() != 0:
                raise RuntimeError("qwn_reset_state failed")
            if lib.qwn_mtp_reset() != 0:
                raise RuntimeError("qwn_mtp_reset failed")
            pred = 0
            for t in range(0, p + 1):
                pred = decode(lib, ids[t], t)
                if t < p:
                    mtp_advance(lib, ids[t + 1], t)   # trunk slot t = MTP(h_t, committed tok_{t+1})
            seed = pred                                # main-model free/greedy token at p (= tok_{p+1})

            # 2. snapshot root hidden h_p, then free-run the DENSE model D steps.
            if lib.qwn_mtp_snapshot_root() != 0:
                raise RuntimeError("qwn_mtp_snapshot_root failed")
            g = []
            cur = seed
            for d in range(D):
                cur = decode(lib, cur, p + 1 + d)      # dense greedy g_{d+1} = tok_{p+2+d}
                g.append(cur)

            # 3. (B) build margin-shaped covering trees; measure realized accept-length.
            for (tau, b) in configs:
                toks, pars, deps, mrg = tree_build(lib, seed, p, D, K, tau, b)
                tree_sizes[(tau, b)].append(len(toks))
                acc = tree_accept_length(toks, pars, deps, g, D)
                for d in range(1, D + 1):
                    accept[(tau, b)][d].append(min(acc, d))

            # 4. (A) teacher-forced covering upper bound along the dense path.
            fed = [seed] + g[: D - 1]                  # feed [seed,g1,..,g_{D-1}], targets g1..gD
            ids_k, vals_k = path_topk(lib, fed, p, K)
            root_margins.append(float(vals_k[0, 0] - vals_k[0, 1]))
            covered_so_far = True
            for d in range(1, D + 1):
                target = g[d - 1]
                row = ids_k[d - 1]
                hit1 = int(target == row[0])
                hit8 = int(target in row[:8])
                tf_top1[d - 1].append(hit1)
                tf_top8[d - 1].append(hit8)
                covered_so_far = covered_so_far and bool(hit8)
                tf_cumul[d - 1].append(int(covered_so_far))

            log(f"pos {p:4d} ({pi + 1}/{len(positions)}): seed={seed} g={g}  "
                f"tf_top8={[int(np.mean(tf_top8[d])*100) for d in range(D)]}  "
                f"elapsed={time.time()-tstart:.0f}s")
    finally:
        lib.qwn_free()

    # ---- report -----------------------------------------------------------------
    np_top1 = [float(np.mean(x) * 100) for x in tf_top1]
    np_top8 = [float(np.mean(x) * 100) for x in tf_top8]
    np_cumul = [float(np.mean(x) * 100) for x in tf_cumul]
    ideal_accept = 1.0 + sum(np_cumul[d] / 100 for d in range(D))  # +1 free token

    print("\n" + "=" * 78, flush=True)
    print("  MTP covering tree -- realized accept-length vs dense greedy (Step 2)", flush=True)
    print(f"  model={args.model_dir}", flush=True)
    print(f"  tqf={args.tqf}", flush=True)
    print(f"  positions={len(positions)}  depth D={D}  record-k={K}  "
          f"taus={taus}  node-budgets={budgets}", flush=True)
    print(f"  root top1-top2 margin: mean={np.mean(root_margins):.3f} "
          f"median={np.median(root_margins):.3f} "
          f"p10={np.percentile(root_margins,10):.3f} p90={np.percentile(root_margins,90):.3f}", flush=True)
    print("=" * 78, flush=True)

    print("\n  (A) TEACHER-FORCED coverage along the dense path  [covering upper bound]", flush=True)
    print("      validates KV forking + reproduces the plan's per-level model.", flush=True)
    print(f"    {'depth':>5} {'predicts':>9} {'top-1':>8} {'top-8':>8} {'cumul top-8':>12}", flush=True)
    for d in range(1, D + 1):
        print(f"    {d:>5} {'t+'+str(d+1):>9} {np_top1[d-1]:>7.2f}% {np_top8[d-1]:>7.2f}% "
              f"{np_cumul[d-1]:>11.2f}%", flush=True)
    print(f"    => idealized mean accept-length (1 + sum cumul top-8) = {ideal_accept:.3f}", flush=True)
    print(f"    (plan model targets: top-8 ~95.6/85.3/76.2/68.6 -> E[committed] ~1.96/2.77/3.39/3.82)",
          flush=True)

    def cfg_tag(tau, b):
        kind = "greedy-spine" if tau == 0 else f"covering(tau={tau:g})"
        return f"{kind}, budget={b}"

    print("\n  (B) REALIZED accept-length inside the BUILT covering tree  [PRIMARY]", flush=True)
    print("      mean accept-length = 1 (main-model free token) + tree-accepted depth.", flush=True)
    print("      per-depth coverage[d] = P(dense path covered through depth d).", flush=True)
    for cf in configs:
        tau, b = cf
        ts = tree_sizes[cf]
        print(f"\n    [{cfg_tag(tau, b)}]  avg nodes={np.mean(ts):.1f} (max {int(np.max(ts))})", flush=True)
        print(f"    {'D':>3} {'mean_accept(+1)':>16} {'tree_accept':>12} "
              + " ".join(f'cov{d}'.rjust(7) for d in range(1, D + 1)), flush=True)
        for Dd in range(1, D + 1):
            accs = np.array(accept[cf][Dd], dtype=np.float64)   # min(acc, Dd) per position
            mean_tree = float(accs.mean())
            mean_committed = 1.0 + mean_tree
            covs = [float(np.mean(np.array(accept[cf][Dd]) >= d) * 100) for d in range(1, Dd + 1)]
            covstr = " ".join(f"{c:6.1f}%" for c in covs) + "        " * (D - Dd)
            print(f"    {Dd:>3} {mean_committed:>16.3f} {mean_tree:>12.3f} {covstr}", flush=True)

    # headline + gate at full depth D
    print("\n" + "=" * 78, flush=True)
    best_cf = max(configs, key=lambda cf: 1.0 + np.mean(accept[cf][D]))
    best = 1.0 + float(np.mean(accept[best_cf][D]))
    cov1_best = float(np.mean(np.array(accept[best_cf][D]) >= 1) * 100)
    print(f"  HEADLINE (D={D}): best tree [{cfg_tag(*best_cf)}] mean accept-length = {best:.3f}", flush=True)
    print(f"           (committed tokens/round incl. the main-model free token)", flush=True)
    print(f"           idealized (full top-8 covering, teacher-forced) = {ideal_accept:.3f}", flush=True)
    print(f"           depth-1 coverage: tree={cov1_best:.1f}%  teacher-forced top-8={np_top8[0]:.1f}%",
          flush=True)
    gate = best >= 2.5 or ideal_accept >= 3.0
    print(f"  GATE (mean accept >= 2.5, target ~3-3.8): {'PASS' if gate else 'INVESTIGATE'}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
