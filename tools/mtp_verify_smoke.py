#!/usr/bin/env python3
"""Step 3 + 4 of docs/MTP_SPECDEC_PLAN.md: dense FP6 verify over the MTP covering
tree, with FORKING DeltaNet/conv/KV state per branch, descent by dense argmax, and
single-path commit -> token-perfect output.

The verify reuses the existing dense decode (qwn_decode = the E2M3/FP6 27B model, all
64 layers) node-by-node by REPLAYING each tree branch from the committed state (the
task-endorsed "correctness first" option). The forking primitive is:

  * Full-attention layers fork IMPLICITLY through the causal KV cache + position: a
    branch writes cache slots pos+1.. and never reads stale deeper slots (exactly the
    Step-2 MTP tree mechanism). The committed prefix KV[0..pos] is never overwritten.
  * DeltaNet (linear-attention) layers carry ONE recurrent + conv state per layer that
    qwn_decode mutates in place. We snapshot the committed state with the NEW engine
    primitive qwn_delta_state_save(slot) and qwn_delta_state_restore(slot), so each
    branch can be replayed from the committed state and a different branch restored
    afterward. This is the per-branch DeltaNet fork (the crux).

GATES (must pass):
  (a) FORKING CORRECTNESS: for a small tree, each node's final dense hidden from the
      tree-verify (restore committed state -> replay branch) must match the dense model
      run INDEPENDENTLY on that node's full branch (reset -> feed prompt + branch).
      Require cos > 0.999 AND identical lm_head top-1 per node. Proves DeltaNet/conv/KV
      forking is exact.
  (b) END-TO-END TOKEN-PERFECT: run the full spec-decode (draft tree -> forking verify
      -> descend by dense argmax -> single-path commit) for N tokens and assert the
      committed token IDs are BIT-IDENTICAL to plain dense greedy (qwn_decode) on the
      same prompt. Any divergence = a forking/descent/commit bug.

Run (GPU 6 only):
    CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python3 -u tools/mtp_verify_smoke.py \
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
    "Consider a train that travels 60 kilometers in 45 minutes: its average speed is "
    "80 km/h, because 60 divided by 0.75 equals 80."
)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------- engine ABI
def load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.qwn_init.argtypes = [ctypes.c_char_p]; lib.qwn_init.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_hidden_size.restype = ctypes.c_int
    lib.qwn_vocab_size.restype = ctypes.c_int
    lib.qwn_has_mtp.restype = ctypes.c_int
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_mtp_reset.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_mtp_advance.argtypes = [ctypes.c_int, ctypes.c_int]; lib.qwn_mtp_advance.restype = ctypes.c_int
    lib.qwn_mtp_snapshot_root.restype = ctypes.c_int
    lib.qwn_mtp_tree_build.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ]
    lib.qwn_mtp_tree_build.restype = ctypes.c_int
    lib.qwn_copy_last_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_copy_last_norm.restype = ctypes.c_int
    lib.qwn_delta_state_save.argtypes = [ctypes.c_int]; lib.qwn_delta_state_save.restype = ctypes.c_int
    lib.qwn_delta_state_restore.argtypes = [ctypes.c_int]; lib.qwn_delta_state_restore.restype = ctypes.c_int
    return lib


# ---------------------------------------------------------------- thin wrappers
def decode(lib, tok, pos):
    r = lib.qwn_decode(int(tok), int(pos))
    if r < 0:
        raise RuntimeError(f"qwn_decode({tok},{pos}) failed: {r}")
    return r


def mtp_advance(lib, tok, pos):
    r = lib.qwn_mtp_advance(int(tok), int(pos))
    if r != 0:
        raise RuntimeError(f"qwn_mtp_advance({tok},{pos}) failed: {r}")


def delta_save(lib, slot):
    r = lib.qwn_delta_state_save(int(slot))
    if r != 0:
        raise RuntimeError(f"qwn_delta_state_save({slot}) failed: {r}")


def delta_restore(lib, slot):
    r = lib.qwn_delta_state_restore(int(slot))
    if r != 0:
        raise RuntimeError(f"qwn_delta_state_restore({slot}) failed: {r}")


def copy_norm(lib, H, buf):
    n = lib.qwn_copy_last_norm(buf, H)
    return np.ctypeslib.as_array(buf)[:n].copy()


def tree_build(lib, seed, pos, depth, k, tau, max_nodes):
    toks = (ctypes.c_int * max_nodes)(); pars = (ctypes.c_int * max_nodes)()
    deps = (ctypes.c_int * max_nodes)(); mrg = (ctypes.c_float * max_nodes)()
    n = lib.qwn_mtp_tree_build(int(seed), int(pos), int(depth), int(k),
                               ctypes.c_float(tau), int(max_nodes), toks, pars, deps, mrg)
    if n < 0:
        raise RuntimeError(f"qwn_mtp_tree_build failed: {n}")
    return list(toks[:n]), list(pars[:n]), list(deps[:n])


def branch_tokens(idx, tokens, parents):
    """Tokens from the root's first child down to node idx (excludes the seed)."""
    out = []
    while idx != -1:
        out.append(tokens[idx])
        idx = parents[idx]
    out.reverse()
    return out


def cosine(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def prefill(lib, ids, p, build_trunk=True):
    """Decode prompt ids[0..p] (positions 0..p), building the MTP trunk (slots 0..p-1)
    so the tree has realistic context. Returns the greedy at p (= seed / free token)."""
    lib.qwn_reset_state()
    lib.qwn_mtp_reset()
    pred = 0
    for t in range(0, p + 1):
        pred = decode(lib, ids[t], t)
        if build_trunk and t < p:
            mtp_advance(lib, ids[t + 1], t)
    return pred


# --------------------------------------------------------------------- gate (a)
def gate_a(lib, ids, H, p, depth, k, tau, max_nodes):
    log(f"GATE (a) forking correctness: p={p} depth={depth} k={k} tau={tau} max_nodes={max_nodes}")
    buf = (ctypes.c_float * H)()

    # Phase 0: committed state at p + small tree.
    seed = prefill(lib, ids, p)
    delta_save(lib, 0)                       # S_p (committed DeltaNet state)
    if lib.qwn_mtp_snapshot_root() != 0:
        raise RuntimeError("snapshot_root failed")
    tokens, parents, depths = tree_build(lib, seed, p, depth, k, tau, max_nodes)
    log(f"  built tree: {len(tokens)} nodes  (seed={seed})")

    # The verify processes the seed-node (branch [seed]) + every tree node.
    branches = [("seed", [seed])]
    for i in range(len(tokens)):
        branches.append((i, [seed] + branch_tokens(i, tokens, parents)))

    # Phase 1: VERIFY hidden/argmax via fork (restore S_p -> replay branch). Uses the
    # live committed KV[0..p]; DeltaNet restored to S_p before each branch.
    verify = {}
    for key, branch in branches:
        delta_restore(lib, 0)
        am = None
        for i, tok in enumerate(branch):
            am = decode(lib, tok, p + 1 + i)
        verify[key] = (am, copy_norm(lib, H, buf))

    # Phase 2: INDEPENDENT reference (reset -> feed prompt[0..p] + branch from scratch).
    indep = {}
    for key, branch in branches:
        lib.qwn_reset_state()
        for t in range(0, p + 1):
            decode(lib, ids[t], t)
        am = None
        for i, tok in enumerate(branch):
            am = decode(lib, tok, p + 1 + i)
        indep[key] = (am, copy_norm(lib, H, buf))

    # Phase 3: compare.
    worst_cos = 1.0
    argmax_mismatch = 0
    rows = []
    for key, branch in branches:
        v_am, v_h = verify[key]
        i_am, i_h = indep[key]
        c = cosine(v_h, i_h)
        worst_cos = min(worst_cos, c)
        ok_am = (v_am == i_am)
        if not ok_am:
            argmax_mismatch += 1
        d = (len(branch) - 1)
        rows.append((key, d, c, v_am, i_am, ok_am))

    print("\n" + "=" * 74, flush=True)
    print("  GATE (a) -- DeltaNet/conv/KV forking correctness (per node)", flush=True)
    print(f"  tree nodes (+seed) = {len(branches)}   depth = {depth}", flush=True)
    print(f"  {'node':>6} {'depth':>5} {'cos(hidden)':>12} {'verify_am':>10} {'indep_am':>9} {'ok':>4}",
          flush=True)
    for key, d, c, v_am, i_am, ok in rows[:24]:
        ks = "seed" if key == "seed" else str(key)
        print(f"  {ks:>6} {d:>5} {c:>12.6f} {v_am:>10} {i_am:>9} {'Y' if ok else 'N':>4}", flush=True)
    if len(rows) > 24:
        print(f"  ... ({len(rows)-24} more)", flush=True)
    passed = (worst_cos > 0.999) and (argmax_mismatch == 0)
    print(f"  worst cos = {worst_cos:.6f}   argmax mismatches = {argmax_mismatch}/{len(branches)}", flush=True)
    print(f"  GATE (a): {'PASS' if passed else 'FAIL'}", flush=True)
    print("=" * 74, flush=True)
    return passed


# --------------------------------------------------------------------- gate (b)
def dense_greedy(lib, ids, P, N):
    """Plain dense greedy: feed prompt, then repeatedly feed argmax. Returns N tokens."""
    lib.qwn_reset_state()
    am = 0
    for t in range(P):
        am = decode(lib, ids[t], t)
    gen = []
    cur = am             # first generated token (greedy after prompt, position P)
    p = P - 1
    for _ in range(N):
        gen.append(cur)
        cur = decode(lib, cur, p + 1)
        p += 1
    return gen


def spec_decode(lib, ids, P, N, depth, k, tau, max_nodes, verbose=False):
    """MTP-tree spec-decode: per round build the tree, fork-verify by dense argmax,
    descend, commit the accepted prefix + correction, advance the real state by the
    committed tokens (single winning path). Returns (committed[:N], stats)."""
    seed = prefill(lib, ids, P - 1)          # committed prompt[0..P-1]; seed=greedy at P-1
    p = P - 1
    committed = []
    round_lens = []
    while len(committed) < N:
        delta_save(lib, 0)                   # S_p (committed DeltaNet)
        if lib.qwn_mtp_snapshot_root() != 0:
            raise RuntimeError("snapshot_root failed")
        tokens, parents, depths = tree_build(lib, seed, p, depth, k, tau, max_nodes)
        children = {}
        for i in range(len(tokens)):
            children.setdefault(parents[i], []).append((i, tokens[i]))

        # Fork-verify + descend. Each accepted node is forked from S_p (restore ->
        # replay the branch) so the descent argmax is the true dense argmax of that
        # branch -- if the fork is wrong the argmax is wrong and (b) diverges.
        delta_restore(lib, 0)
        a = decode(lib, seed, p + 1)         # a_1 = dense argmax at the seed-node
        accepted = [a]
        cur = -1                              # root (seed-node)
        branch = [seed]
        while True:
            match = None
            for (ci, ctok) in children.get(cur, []):
                if ctok == a:
                    match = ci
                    break
            if match is None:
                break                         # a not covered -> stop (a is the correction)
            branch = branch + [a]             # descend onto the matched node (token a)
            delta_restore(lib, 0)             # fork from the committed state
            aa = None
            for i, tok in enumerate(branch):
                aa = decode(lib, tok, p + 1 + i)
            a = aa
            accepted.append(a)
            cur = match

        round_tokens = [seed] + accepted      # [seed, a_1, ..., a_{m+1}]
        round_lens.append(len(round_tokens))

        # Single-path commit: restore S_p, re-feed the committed tokens to advance the
        # REAL DeltaNet + conv + full-attn KV (discarding all other forked branches).
        delta_restore(lib, 0)
        last_am = None
        for i, tok in enumerate(round_tokens):
            last_am = decode(lib, tok, p + 1 + i)
        committed.extend(round_tokens)
        p += len(round_tokens)
        seed = last_am                        # next free token (greedy after commit)
        if verbose:
            log(f"  round: committed {len(round_tokens)} (accepted {len(accepted)-1} in-tree) "
                f"total={len(committed)}")
    return committed[:N], {"rounds": len(round_lens), "round_lens": round_lens}


def gate_b(lib, ids, P, N, depth, k, tau, max_nodes):
    log(f"GATE (b) end-to-end token-perfect: P={P} N={N} depth={depth} k={k} "
        f"tau={tau} max_nodes={max_nodes}")
    t0 = time.time()
    dense = dense_greedy(lib, ids, P, N)
    log(f"  dense greedy done ({time.time()-t0:.0f}s); first 12 = {dense[:12]}")
    t0 = time.time()
    spec, stats = spec_decode(lib, ids, P, N, depth, k, tau, max_nodes)
    log(f"  spec-decode done ({time.time()-t0:.0f}s); rounds={stats['rounds']} "
        f"mean commit/round={np.mean(stats['round_lens']):.2f}; first 12 = {spec[:12]}")

    match = (spec == dense)
    print("\n" + "=" * 74, flush=True)
    print("  GATE (b) -- spec-decode committed == dense greedy (token-perfect)", flush=True)
    print(f"  tokens compared = {N}   rounds = {stats['rounds']}   "
          f"mean committed/round = {np.mean(stats['round_lens']):.2f}", flush=True)
    if not match:
        # report first divergence
        first = next((i for i in range(min(len(spec), len(dense))) if spec[i] != dense[i]), None)
        print(f"  DIVERGENCE at index {first}: spec={spec[first] if first is not None else '?'} "
              f"dense={dense[first] if first is not None else '?'}", flush=True)
        print(f"  spec [{max(0,(first or 0)-3)}:+8]  = {spec[max(0,(first or 0)-3):(first or 0)+5]}", flush=True)
        print(f"  dense[{max(0,(first or 0)-3)}:+8]  = {dense[max(0,(first or 0)-3):(first or 0)+5]}", flush=True)
    print(f"  GATE (b): {'PASS (bit-identical)' if match else 'FAIL'}", flush=True)
    print("=" * 74, flush=True)
    return match


def main():
    ap = argparse.ArgumentParser(description="MTP-tree dense verify (Step 3+4)")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=128, help="prompt token cap")
    ap.add_argument("--gate", choices=("a", "b", "both"), default="both")
    # gate (a) small tree
    ap.add_argument("--a-pos", type=int, default=32)
    ap.add_argument("--a-depth", type=int, default=3)
    ap.add_argument("--a-k", type=int, default=4)
    ap.add_argument("--a-tau", type=float, default=20.0)
    ap.add_argument("--a-max-nodes", type=int, default=20)
    # gate (b) end-to-end
    ap.add_argument("--b-prompt-tokens", type=int, default=48)
    ap.add_argument("--b-gen", type=int, default=64)
    ap.add_argument("--b-depth", type=int, default=4)
    ap.add_argument("--b-k", type=int, default=8)
    ap.add_argument("--b-tau", type=float, default=8.0)
    ap.add_argument("--b-max-nodes", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = args.prompt or (open(BIGTEXT).read() if os.path.exists(BIGTEXT) else FALLBACK_TEXT)
    ids = tok(text, add_special_tokens=False).input_ids[: args.max_tokens]
    log(f"prompt tokens available: {len(ids)}")

    lib = load_lib(args.lib)
    log(f"loading engine {args.tqf} ...")
    t0 = time.time()
    if lib.qwn_init(args.tqf.encode()) != 0:
        raise RuntimeError("qwn_init failed")
    log(f"engine loaded in {time.time()-t0:.1f}s")
    results = {}
    try:
        H = int(lib.qwn_hidden_size())
        if not lib.qwn_has_mtp():
            raise RuntimeError("engine reports no MTP section")
        log(f"engine: H={H} V={int(lib.qwn_vocab_size())} has_mtp=1")
        if args.gate in ("a", "both"):
            results["a"] = gate_a(lib, ids, H, args.a_pos, args.a_depth, args.a_k,
                                  args.a_tau, args.a_max_nodes)
        if args.gate in ("b", "both"):
            results["b"] = gate_b(lib, ids, args.b_prompt_tokens, args.b_gen, args.b_depth,
                                  args.b_k, args.b_tau, args.b_max_nodes)
    finally:
        lib.qwn_free()

    print("\n" + "#" * 74, flush=True)
    for g in ("a", "b"):
        if g in results:
            print(f"#  GATE ({g}): {'PASS' if results[g] else 'FAIL'}", flush=True)
    print("#" * 74, flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
