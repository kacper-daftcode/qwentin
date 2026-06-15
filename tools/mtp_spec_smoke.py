#!/usr/bin/env python3
"""Step 5: end-to-end MTP-tree speculative decode (TQ_SPEC_DECODE path), MEASURED.

Builds the working spec-decode loop on top of the proven pieces:
  - qwn_mtp_tree_build  : MTP covering tree (<=k nodes) at the current position
  - qwn_spec_forward_test: batched k-split nk dense FP6 verify over the tree nodes
                           (reads each weight ONCE for all <=8 nodes) -> full-vocab argmax
  - dense-argmax descent : accept the covered prefix
  - single-path commit   : re-run the batched forward over the accepted chain to advance
                           the real DeltaNet/conv/full-attn-KV state, hand the post-commit
                           pre-final-norm hidden to the next round's tree root, and extend
                           the MTP trunk (qwn_spec_set_root_from / qwn_mtp_advance_from_spec)

gate-a is RELAXED to "lossless up to float-eps": the batched verify is an equally-valid
forward of the same model; we MEASURE the per-step divergence vs plain dense greedy
(teacher-forced flip rate) rather than requiring bit-identical IDs.

Reports (GPU 6, 27B qwen3_6-27b-e2m3-mtp.tqf, same prompt):
  (b) tok/s: baseline dense -> spec -> ratio
  relaxed-(a): per-step divergence rate (spec vs per-token argmax, same prefix)
  mean committed tokens / round (accept-length)

Run:
    CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python3 -u tools/mtp_spec_smoke.py
"""
from __future__ import annotations
import argparse, ctypes, os, time
import numpy as np
from transformers import AutoTokenizer

BIGTEXT = os.environ.get("TQ_BENCH_TEXT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "forward_qwen.cu"))


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_mtp_reset.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_decode_graph.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode_graph.restype = ctypes.c_int
    L.qwn_mtp_advance.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_mtp_advance.restype = ctypes.c_int
    L.qwn_mtp_snapshot_root.restype = ctypes.c_int
    L.qwn_mtp_tree_build.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    L.qwn_mtp_tree_build.restype = ctypes.c_int
    L.qwn_delta_state_save.argtypes = [ctypes.c_int]; L.qwn_delta_state_save.restype = ctypes.c_int
    L.qwn_delta_state_restore.argtypes = [ctypes.c_int]; L.qwn_delta_state_restore.restype = ctypes.c_int
    L.qwn_spec_forward_test.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4 + [ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    L.qwn_spec_forward_test.restype = ctypes.c_int
    L.qwn_capture_spec_forward.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4 + [ctypes.c_int]
    L.qwn_capture_spec_forward.restype = ctypes.c_int
    L.qwn_spec_forward_graph.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4 + [ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    L.qwn_spec_forward_graph.restype = ctypes.c_int
    L.qwn_spec_set_root_from.argtypes = [ctypes.c_int]; L.qwn_spec_set_root_from.restype = ctypes.c_int
    L.qwn_mtp_advance_from_spec.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.qwn_mtp_advance_from_spec.restype = ctypes.c_int
    L.qwn_spec_commit.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    L.qwn_spec_commit.restype = ctypes.c_int
    L.qwn_spec_round.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    L.qwn_spec_round.restype = ctypes.c_int
    return L


def ck(r, what):
    if (isinstance(r, int) and r < 0):
        raise RuntimeError(f"{what} failed: {r}")
    return r


class Eng:
    def __init__(self, L): self.L = L; self.H = L.qwn_hidden_size()
    def decode(self, t, p): return ck(self.L.qwn_decode(int(t), int(p)), "decode")
    def decode_graph(self, t, p): return ck(self.L.qwn_decode_graph(int(t), int(p)), "decode_graph")
    def mtp_advance(self, t, p): ck(self.L.qwn_mtp_advance(int(t), int(p)), "mtp_advance")
    def save(self, s): ck(self.L.qwn_delta_state_save(s), "save")
    def restore(self, s): ck(self.L.qwn_delta_state_restore(s), "restore")
    def snapshot_root(self): ck(self.L.qwn_mtp_snapshot_root(), "snapshot_root")
    def set_root_from(self, i): ck(self.L.qwn_spec_set_root_from(int(i)), "set_root_from")
    def advance_from_spec(self, i, t, p): ck(self.L.qwn_mtp_advance_from_spec(int(i), int(t), int(p)), "adv_spec")
    def commit(self, path, path_pos):
        m = len(path)
        cp = (ctypes.c_int * m)(*path); cq = (ctypes.c_int * m)(*path_pos)
        ck(self.L.qwn_spec_commit(cp, cq, m), "spec_commit")

    def tree(self, seed, pos, depth, k, tau, maxn):
        cap = maxn
        toks = (ctypes.c_int * cap)(); pars = (ctypes.c_int * cap)()
        deps = (ctypes.c_int * cap)(); mrg = (ctypes.c_float * cap)()
        n = ck(self.L.qwn_mtp_tree_build(int(seed), int(pos), int(depth), int(k),
               ctypes.c_float(tau), int(maxn), toks, pars, deps, mrg), "tree_build")
        return list(toks[:n]), list(pars[:n]), list(deps[:n])

    def spec_forward(self, toks, pars, deps, posv):
        N = len(toks)
        ct = (ctypes.c_int * N)(*toks); cp = (ctypes.c_int * N)(*pars)
        cd = (ctypes.c_int * N)(*deps); cpos = (ctypes.c_int * N)(*posv)
        am = (ctypes.c_int * N)()
        ck(self.L.qwn_spec_forward_test(ct, cp, cd, cpos, N, am, None), "spec_forward")
        return list(am)

    def spec_forward_graph(self, toks, pars, deps, posv):
        N = len(toks)
        ct = (ctypes.c_int * N)(*toks); cp = (ctypes.c_int * N)(*pars)
        cd = (ctypes.c_int * N)(*deps); cpos = (ctypes.c_int * N)(*posv)
        am = (ctypes.c_int * N)()
        ck(self.L.qwn_spec_forward_graph(ct, cp, cd, cpos, N, am, None), "spec_forward_graph")
        return list(am)


def prefill(e, ids, p, build_trunk=True, start=0, anchor=0):
    """Prefill ids[start..p] (positions start..p). start=0 (default) resets all
    engine state first; start>0 CONTINUES from existing state -- the caller
    guarantees that the live KV/DeltaNet/trunk state corresponds to ids[0..start-1]
    (prefix-cache path: positions are explicit through the whole API, nothing
    below assumes pos 0).

    BIT-IDENTITY: the chunked prefill cuts 16-token chains from `start`, and
    per-position numerics depend on the chunk seams (equally-valid forward,
    float-eps differences -> near-tie argmax flips downstream). A continuation
    reproduces the start=0 prefill bit-exactly iff `start` is 16-aligned.

    anchor=A (16-aligned, start < A <= p): snapshot the DeltaNet state into
    slot 0 right after the chunk ending at A-1 commits, so a later
    qwn_delta_state_restore(0) + prefill(start=A) replays positions A.. through
    the exact same chunking as a fresh full prefill (prefix-cache anchor)."""
    if start <= 0:
        e.L.qwn_reset_state(); e.L.qwn_mtp_reset()
        start = 0
    if start > p:
        raise ValueError(f"prefill: start {start} beyond final position {p}")
    # D.1(2) wide prefill (TQ_WIDE_PREFILL=1): the fast wide N=128 path (wide FP6 GEMM +
    # chunkwise-parallel DeltaNet + flash wide-attn). LENGTH-GATED hybrid: it wins for prompts
    # up to TQ_WIDE_MAX tokens (default 16384) -- 1.5-2.75x vs the N=16 baseline; beyond that
    # the (scalar) wide-attn loses to the baseline tensor-core attention, so we fall through to
    # the baseline chunked path (best-of-both, no long-ctx regression).
    #
    # TRUNK BRIDGE (D-finish 2026-06-14): the wide path now builds the MTP draft trunk too
    # (qwn_wide_advance_trunk gathers the per-token hidden from g_wide_h), so spec-decode
    # (qwn_spec_round / snapshot_root) runs after a wide prefill -- it is usable from the
    # OpenAI server, not just the smoke harness. snapshot_root already works (qwn_prefill_wide
    # lands the last hidden in d_debug_x). The anchor (prefix-cache) is honoured by cutting the
    # chunk grid at the anchor and snapshotting the DeltaNet state there.
    #
    # KV: the wide attn/KV store now supports Q4(K)+E4M3(V) (roadmap E milestone 2) in addition
    # to fp32; only PURE FP8 (TQ_KV_FP8 without TQ_KV_Q4) is still rejected (-3), so the wide
    # path runs under the prod TQ_KV_Q4 config too.
    # HYBRID GATE (E.perf step 3): the SCALAR wide-attn lost to baseline tensor-core attention
    # past ~16k, so wide prefill was capped at TQ_WIDE_MAX=16384. The MMA wide-prefill attention
    # (TQ_WIDE_ATTN_MMA=1) uses the SAME tensor cores as baseline and MEASURES faster at every
    # length (contiguous Q4: 2.86x@8k 2.57x@16k 2.31x@32k ~1.8x@64k; paged: 2.1x@8k 2.9x@16k
    # 4.1x@32k) -> no upper crossover. So with MMA on the cap lifts to the full context (always
    # wide+MMA); with MMA off the 16k scalar cap stays. Explicit TQ_WIDE_MAX always overrides.
    mma_on = os.environ.get("TQ_WIDE_ATTN_MMA", "") not in ("", "0")
    wide_max = int(os.environ.get("TQ_WIDE_MAX", str(10**9 if mma_on else 16384)))
    kv_fp8_only = (os.environ.get("TQ_KV_FP8", "") not in ("", "0")
                   and os.environ.get("TQ_KV_Q4", "") in ("", "0"))
    if (os.environ.get("TQ_WIDE_PREFILL", "") not in ("", "0")
            and not kv_fp8_only and (p + 1 - start) <= wide_max):
        e.L.qwn_prefill_wide.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                         ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        e.L.qwn_prefill_wide.restype = ctypes.c_int
        e.L.qwn_wide_advance_trunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                               ctypes.c_int]
        e.L.qwn_wide_advance_trunk.restype = ctypes.c_int
        am = ctypes.c_int(0)
        CH = int(os.environ.get("TQ_PREFILL_CHUNK", "128"))
        if CH < 1:
            CH = 1
        t = start
        while t <= p:
            n = min(CH, p + 1 - t)
            if anchor and t < anchor < t + n:    # cut so a chunk ends exactly at the anchor
                n = anchor - t
            last = (t + n > p)                    # only the final chunk needs the seed argmax
            chunk = (ctypes.c_int * n)(*ids[t:t + n])
            seed_ptr = ctypes.byref(am) if last else None
            ck(e.L.qwn_prefill_wide(chunk, t - 1, n, seed_ptr), "prefill_wide")
            if build_trunk:
                # trunk advance over this chunk's positions: node j (hidden at pos t+j) feeds
                # the NEXT token ids[t+1+j]. The final prompt position p has no next token
                # (the spec tree root covers it), so the last chunk advances n-1 nodes.
                m = n if t + n <= p else n - 1
                if m > 0:
                    ntoks = (ctypes.c_int * m)(*ids[t + 1:t + 1 + m])
                    ck(e.L.qwn_wide_advance_trunk(ntoks, t, m), "wide_advance_trunk")
            t += n
            if anchor and t == anchor:
                e.save(0)              # DeltaNet state after positions 0..anchor-1
        return am.value
    # Chunked prefill (TQ_FAST_PREFILL=0 reverts to per-token): 16-token chains
    # through the batched verify machinery -- weights read once per chunk.
    if os.environ.get("TQ_FAST_PREFILL", "1") not in ("", "0"):
        e.L.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                          ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        e.L.qwn_mtp_advance_wave.argtypes = [ctypes.POINTER(ctypes.c_int),
                                             ctypes.POINTER(ctypes.c_int),
                                             ctypes.c_int, ctypes.c_int]
        am = ctypes.c_int(0)
        # Verify chunk size: default 16 (the TQ_SPEC_MAX_N=16 build). The wide-prefill
        # stopgap (roadmap D) sets TQ_PREFILL_CHUNK=32 against a -DTQ_SPEC_MAX_N=32 build
        # to halve the number of weight passes. The MTP trunk-advance wave stays <=16
        # (its g_mtpb_* buffers are hardcoded to 16), so it is split into sub-waves.
        CH = int(os.environ.get("TQ_PREFILL_CHUNK", "16"))
        if CH < 1:
            CH = 1
        t = start
        n = 1
        while t <= p:
            n = min(CH, p + 1 - t)
            chunk = (ctypes.c_int * n)(*ids[t:t + n])
            ck(e.L.qwn_prefill_chunk(chunk, t - 1, n, ctypes.byref(am)), "prefill_chunk")
            if build_trunk:
                # trunk advance at positions t..: feeds the NEXT prompt token; the
                # final prompt position p has no next token (the tree root covers it)
                m = n if t + n <= p else n - 1
                off = 0
                while off < m:                       # <=16-wide sub-waves (buffer cap)
                    mm = min(16, m - off)
                    nodes = (ctypes.c_int * mm)(*range(off, off + mm))
                    toks = (ctypes.c_int * mm)(*ids[t + 1 + off:t + 1 + off + mm])
                    ck(e.L.qwn_mtp_advance_wave(nodes, toks, t + off, mm), "advance_wave")
                    off += mm
            t += n
            if anchor and t == anchor:
                e.save(0)              # DeltaNet state after positions 0..anchor-1
        # land the last hidden in d_debug_x (callers snapshot_root from it) and
        # take the seed from the canonical per-token lm_head path
        return ck(e.L.qwn_spec_lmhead_from_node(n - 1), "lmhead_from_node")
    am = 0
    for t in range(start, p + 1):
        am = e.decode(ids[t], t)
        if build_trunk and t < p:
            e.mtp_advance(ids[t + 1], t)
        if anchor and t + 1 == anchor:
            e.save(0)
    return am   # greedy at p (= seed / free token tok_{p+1})


def dense_greedy(e, ids, P, N):
    e.L.qwn_reset_state()
    am = 0
    for t in range(P):
        am = e.decode(ids[t], t)
    gen = []; cur = am; p = P - 1
    for _ in range(N):
        gen.append(cur)
        cur = e.decode(cur, p + 1); p += 1
    return gen


def dfs_order(seed, tree_tok, tree_par, tree_dep, base_pos):
    """Verify node list: node0=seed (root), tree nodes offset by 1; reorder to DFS
    (so the shared full-attn KV holds each node's branch when it attends)."""
    n = len(tree_tok)
    vtok = [seed] + tree_tok
    vpar = [-1] + [(0 if tree_par[t] == -1 else tree_par[t] + 1) for t in range(n)]
    vdep = [0] + tree_dep
    children = {}
    for i in range(1, n + 1):
        children.setdefault(vpar[i], []).append(i)
    order = []
    stack = [0]
    while stack:
        x = stack.pop(); order.append(x)
        for c in reversed(children.get(x, [])):
            stack.append(c)
    pos_of = {old: new for new, old in enumerate(order)}
    dtok = [vtok[o] for o in order]
    dpar = [(-1 if vpar[o] == -1 else pos_of[vpar[o]]) for o in order]
    ddep = [vdep[o] for o in order]
    dpos = [base_pos + 1 + vdep[o] for o in order]
    # children map in DFS indices, with tokens, for descent
    dchildren = {}
    for i in range(len(order)):
        pp = dpar[i]
        if pp >= 0:
            dchildren.setdefault(pp, []).append((i, dtok[i]))
    return dtok, dpar, ddep, dpos, dchildren


def spec_loop(e, ids, P, N, depth, k, tau, maxnodes, prof=None):
    seed = prefill(e, ids, P - 1)        # committed prompt[0..P-1]; seed = greedy at P-1
    e.snapshot_root()                    # d_mtp_root_hidden = h_{P-1}
    e.save(0)                            # committed DeltaNet
    base_pos = P - 1
    committed = []; round_lens = []
    use_decode_graph = bool(os.environ.get("TQ_DECODE_GRAPH", ""))
    no_decode = bool(os.environ.get("TQ_NO_DECODE", ""))
    use_spec_graph = bool(os.environ.get("TQ_SPEC_GRAPH_CACHED", ""))
    bonus_pending = False
    verify_times = []
    use_lmhead_only = bool(os.environ.get("TQ_LMHEAD_ONLY", ""))
    # In the lmhead-only/no-decode loop nothing touches the LIVE DeltaNet state
    # between qwn_spec_commit (archive -> live) and the next verify (which forks
    # from live), so the per-round 159 MB save/restore round-trip is a no-op.
    skip_snap = no_decode or use_lmhead_only
    # C-side round driver (TQ_C_ROUND=1): tree+verify+descent+commit+advance in one
    # ctypes call, removing the Python orchestration from the round.
    if skip_snap and os.environ.get("TQ_C_ROUND", "1") not in ("", "0"):
        chain_buf = (ctypes.c_int * (maxnodes + 2))()
        state = (ctypes.c_int * 2)()
        while len(committed) < N:
            t = time.time()
            cl = e.L.qwn_spec_round(int(seed), int(base_pos), int(depth), int(k),
                                    ctypes.c_float(tau), int(maxnodes), chain_buf, state)
            ck(cl, "spec_round")
            if prof is not None: prof["verify"] += time.time() - t
            verify_times.append(time.time() - t)
            chain = list(chain_buf[:cl])
            if bonus_pending:
                committed.extend(chain[1:])
            else:
                committed.extend(chain)
            round_lens.append(cl)
            seed = state[0]
            base_pos = state[1]
            bonus_pending = True
        return committed[:N], round_lens, verify_times
    while len(committed) < N:
        t = time.time()
        tree_tok, tree_par, tree_dep = e.tree(seed, base_pos, depth, k, tau, maxnodes - 1)
        if prof is not None: prof["tree"] += time.time() - t
        dtok, dpar, ddep, dpos, dch = dfs_order(seed, tree_tok, tree_par, tree_dep, base_pos)
        if not skip_snap:
            e.restore(0)
        t = time.time()
        if use_spec_graph:
            am = e.spec_forward_graph(dtok, dpar, ddep, dpos)
        else:
            am = e.spec_forward(dtok, dpar, ddep, dpos)
        vt = time.time() - t
        if prof is not None: prof["verify"] += vt
        verify_times.append(vt)
        a = am[0]; accepted = [a]; path = [0]; cur = 0
        while True:
            nxt = None
            for (ci, ctok) in dch.get(cur, []):
                if ctok == a:
                    nxt = ci; break
            if nxt is None:
                break
            cur = nxt; a = am[cur]; accepted.append(a); path.append(cur)
        chain = [seed] + accepted
        m2 = len(chain); leaf = path[-1]
        path_pos = [base_pos + 1 + j for j in range(len(path))]
        t = time.time()
        e.commit(path, path_pos)
        if prof is not None: prof["commit"] += time.time() - t

        if no_decode or use_lmhead_only:
            t = time.time()
            e.set_root_from(leaf)
            for j in range(m2 - 1):
                e.advance_from_spec(path[j], accepted[j], base_pos + 1 + j)
            if not skip_snap:
                e.save(0)
            if prof is not None: prof["mtp+save"] += time.time() - t
            if bonus_pending:
                committed.extend(chain[1:])
            else:
                committed.extend(chain)
            round_lens.append(m2)
            base_pos += m2 - 1
            seed = accepted[-1]
            bonus_pending = True
        else:
            t = time.time()
            if use_decode_graph:
                next_seed = e.decode_graph(accepted[-1], base_pos + m2)
            else:
                next_seed = e.decode(accepted[-1], base_pos + m2)
            e.snapshot_root()
            if prof is not None: prof["decode"] += time.time() - t
            t = time.time()
            for j in range(m2 - 1):
                e.advance_from_spec(path[j], accepted[j], base_pos + 1 + j)
            e.save(0)
            if prof is not None: prof["mtp+save"] += time.time() - t
            committed.extend(chain); round_lens.append(m2)
            base_pos += m2; seed = next_seed
    return committed[:N], round_lens, verify_times


def divergence_rate(e, ids, P, committed):
    """Teacher-forced per-step flip rate: feed prompt + committed via per-token decode;
    fraction of positions where the per-token argmax != the spec's committed next token."""
    e.L.qwn_reset_state()
    for t in range(P):
        e.decode(ids[t], t)
    flips = 0; checked = 0
    p = P - 1
    for i in range(len(committed)):
        am = e.decode(committed[i], p + 1); p += 1
        if i + 1 < len(committed):
            checked += 1
            if am != committed[i + 1]:
                flips += 1
    return flips, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
    ap.add_argument("--prompt-tokens", type=int, default=48)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--max-nodes", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = open(BIGTEXT).read() if os.path.exists(BIGTEXT) else "The history of cartography is"
    ids = tok(text, add_special_tokens=False).input_ids
    P, N = args.prompt_tokens, args.gen
    L = load_lib(args.lib)
    print(f"loading {args.tqf} ...", flush=True)
    ck(L.qwn_init(args.tqf.encode()), "init")
    e = Eng(L)
    try:
        # baseline
        t0 = time.time(); dense = dense_greedy(e, ids, P, N); tb = time.time() - t0
        base_tps = N / tb
        print(f"baseline dense: {N} tok in {tb:.2f}s = {base_tps:.2f} tok/s", flush=True)
        # spec
        no_decode = bool(os.environ.get("TQ_NO_DECODE", ""))
        prof = {"tree": 0.0, "verify": 0.0, "commit": 0.0, "mtp+save": 0.0}
        if not no_decode:
            prof["decode"] = 0.0
        t0 = time.time(); committed, rl, verify_times = spec_loop(e, ids, P, N, args.depth, args.k, args.tau, args.max_nodes, prof)
        ts = time.time() - t0
        spec_tps = len(committed) / ts
        print(f"spec-decode:   {len(committed)} tok in {ts:.2f}s = {spec_tps:.2f} tok/s  "
              f"rounds={len(rl)} mean_commit/round={np.mean(rl):.2f}", flush=True)
        # divergence
        flips, checked = divergence_rate(e, ids, P, committed)
        div = 100.0 * flips / max(1, checked)
        print("\n" + "=" * 72, flush=True)
        print("  Step 5 end-to-end MTP-tree spec decode (relaxed gate-a)", flush=True)
        print(f"  prompt={P} gen={N}  depth={args.depth} k={args.k} tau={args.tau} max_nodes={args.max_nodes}", flush=True)
        print("=" * 72, flush=True)
        print(f"  (b) tok/s   : baseline {base_tps:.2f} -> spec {spec_tps:.2f}  = {spec_tps/base_tps:.2f}x", flush=True)
        print(f"  accept-len  : mean {np.mean(rl):.2f} committed tokens/round ({len(rl)} rounds)", flush=True)
        tot = sum(prof.values())
        print(f"  phase ms/round ({len(rl)} rounds): " + "  ".join(
            f"{k}={1000*v/len(rl):.2f}({100*v/tot:.0f}%)" for k, v in prof.items()), flush=True)
        if verify_times:
            print(f"  verify R0={1000*verify_times[0]:.1f}ms  R1-end avg={1000*np.mean(verify_times[1:]):.1f}ms (graph warm)", flush=True)
        print(f"  relaxed-(a) : per-step divergence {flips}/{checked} = {div:.2f}%  (float-eps near-ties)", flush=True)
        print(f"  first 16 spec : {committed[:16]}", flush=True)
        print(f"  first 16 dense: {dense[:16]}", flush=True)
        exact = sum(1 for a, b in zip(committed, dense) if a == b)
        print(f"  free-run exact-prefix match: {exact}/{N} (diverges after first near-tie flip)", flush=True)
    finally:
        L.qwn_free()


if __name__ == "__main__":
    main()
