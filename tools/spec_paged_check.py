#!/usr/bin/env python3
"""Exactness check of qwn_paged_verify + qwn_paged_commit against plain qwn_paged_decode_step (task 10, stages 1+2).

Teacher-forced "drafts": every slot's draft = the model's own greedy continuation, so every verify column's logits
must equal the decode step's logits at that position bit for bit, and after committing j columns the slot state must
equal the state after j decode steps. The split pickers are pinned (TQ_QMMA_SF_K_SPLIT, TQ_PAGED_GQA_SPLIT) so a
column's GEMM k-splits / attention key-splits do not depend on the wave width (--no-pin shows the unpinned band).

Per (ctx, N):
  1. prefill N slots with distinct prefixes of forward_qwen.cu (--distinct of them prefilled, the rest clones via
     slot save/restore), snapshot every slot;
  2. reference: S plain lockstep decode steps (greedy); the logits of every step per slot (logits sink) + the final
     slot snapshots;
  3. spec (teacher drafts): restore; R rounds of verify (per-slot K_i in 0..4, mixed; K_i = 0 = a plain step through
     the verify path) + commit (per-slot accepted columns incl. partial and zero), every column's logits vs the
     reference; then plain decode steps up to the reference cursor (their logits vs the reference) and the slot
     snapshots vs the reference's;
  4. spec (wrong drafts): drafts half teacher / half random, the server's acceptance rule (longest matching prefix +
     bonus): the committed tokens must be the greedy sequence, rejected columns' KV must get overwritten, final
     state vs the reference;
  5. parking: save / reset / restore a slot right after a commit (inside phase 3).
--bench: verify-wave ms vs decode-step ms at T columns, and the commit replay ms.
usage: spec_paged_check.py --tqf MODEL.tqf --model-dir HF_DIR [--lib LIB] [--ctx 4096] [--n 1,3,8] [--rounds 6]
                           [--distinct 3] [--no-pin] [--bench]
The README's "spec exactness ALL PASS" gate is `--no-pin --ctx 4096` and `--ctx 32768` with the served env in front
(TQ_KV_T4=1; the per-column key splits of the verify wave make it bitwise without pins). Vendored from the qwen256k
research harness (analysis/spec_paged_check.py); run from the repo root (it reads src/forward_qwen.cu as its text).
"""
import argparse, ctypes, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PIN = {"TQ_QMMA_SF_K_SPLIT": "4", "TQ_PAGED_GQA_SPLIT": "8"}
ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=os.path.join(REPO, "build-qwen", "libforward_qwen.so"))
ap.add_argument("--tqf", required=True, help="the model file (needs no MTP section: the drafts are teacher-forced)")
ap.add_argument("--model-dir", required=True, help="HF checkpoint dir (tokenizer)")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--n", default="1,3,8")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--distinct", type=int, default=3, help="slots prefilled with distinct text; the rest are clones")
ap.add_argument("--no-pin", action="store_true", help="do not pin the split pickers (expect near-tie flips only)")
ap.add_argument("--bench", action="store_true")
ap.add_argument("--bench-iters", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--band", type=int, default=0,
                help="also compare plain decode at N against the same slots inside a lockstep decode of this width "
                     "(teacher-forced): the decode path's own batch-size band of near-tie flips, for reference")
args = ap.parse_args()

for k, v in {"TQ_CTX": "262144", "TQ_KV_Q4": "1", "TQ_KV_V4": "1", "TQ_EMBED_FP8": "2", "TQ_PAGED_ATTN_MMA": "1",
             "TQ_WIDE_ATTN_MMA": "1", "TQ_EMBED_HOST": "1", "TQ_NO_MTP": "1", "TQ_PDL": "1"}.items():
    os.environ.setdefault(k, v)
if args.bench:
    args.no_pin = True                           # timings always with the production (auto) split pickers
if not args.no_pin:
    for k, v in PIN.items():
        os.environ.setdefault(k, v)
import torch  # noqa: E402
sys.path.insert(0, HERE)
from serve_batched import load_lib, ck, paged_prefill_slot, _ci  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

Ns = [int(x) for x in args.n.split(",")]
Nmax = max(max(Ns), args.band)
KMAX = 4
R = args.rounds
S = (KMAX + 1) * R + 4                       # teacher tokens per slot (rounds commit <= 5 each) + spare steps
L = load_lib(args.lib)
L.qwn_vocab_size.restype = ctypes.c_int
L.qwn_paged_logits_sink.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
L.qwn_paged_verify.argtypes = [ctypes.POINTER(ctypes.c_int)] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
L.qwn_paged_verify.restype = ctypes.c_int
L.qwn_paged_commit.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
L.qwn_paged_commit.restype = ctypes.c_int
ck(L.qwn_init(args.tqf.encode()), "init")
V = L.qwn_vocab_size()
blocks_per = (args.ctx + S + 8 + 127) // 128 + 1
ck(L.qwn_paged_init(Nmax, Nmax * blocks_per + 4, 128), "paged_init")
SINK_CAP = 64
sink = torch.zeros(SINK_CAP, V, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()


def sink_on(on):
    """The sink makes every prefill wave compute all its columns' logits (slow): only on around decode/verify."""
    if on:
        ck(L.qwn_paged_logits_sink(ctypes.c_void_p(sink.data_ptr()), SINK_CAP, 8), "sink")
    else:
        ck(L.qwn_paged_logits_sink(None, 0, 0), "sink off")
tok = AutoTokenizer.from_pretrained(args.model_dir)
text = open(os.path.join(REPO, "src", "forward_qwen.cu")).read()
all_ids = tok(text, add_special_tokens=False).input_ids
print(f"[spec-check] lib={args.lib} ctx={args.ctx} Ns={Ns} rounds={R} S={S} pin={'off' if args.no_pin else PIN} "
      f"env TQ_PDL={os.environ['TQ_PDL']} corpus={len(all_ids)} tokens", flush=True)
rng = random.Random(args.seed)
fails = 0


def fail(msg):
    global fails
    fails += 1
    print("FAIL " + msg, flush=True)


def decode(tokens, slots, poss):
    n = len(slots)
    out = (ctypes.c_int * n)()
    ck(L.qwn_paged_decode_step(_ci(tokens), _ci(slots), _ci(poss), n, out), "decode")
    torch.cuda.synchronize()
    return [int(out[j]) for j in range(n)], sink[:n].cpu().clone()


def verify(cols, segs):
    """cols: list of (token, slot, pos); segs: list of (slot, off, len). Returns (argmax[T], logits[T, V])."""
    T = len(cols)
    out = (ctypes.c_int * T)()
    rc = L.qwn_paged_verify(_ci([c[0] for c in cols]), _ci([c[1] for c in cols]), _ci([c[2] for c in cols]),
                            _ci([s[0] for s in segs]), _ci([s[1] for s in segs]), _ci([s[2] for s in segs]),
                            len(segs), T, out)
    ck(rc, "verify")
    torch.cuda.synchronize()
    return [int(out[j]) for j in range(T)], sink[:T].cpu().clone()


def commit(seg_idx, n_acc):
    ck(L.qwn_paged_commit(_ci(seg_idx), _ci(n_acc), len(seg_idx)), "commit")


def snapshot(slot, upto):
    nb = int(L.qwn_paged_slot_bytes(upto))
    buf = (ctypes.c_char * nb)()
    ck(L.qwn_paged_slot_save(slot, upto, buf, nb), "save")
    return buf


def restore(slot, upto, buf):
    ck(L.qwn_paged_reset_slot(slot), "reset")
    ck(L.qwn_paged_slot_restore(slot, upto, buf, len(buf)), "restore")


def same_bytes(a, b):
    return len(a) == len(b) and ctypes.string_at(a, len(a)) == ctypes.string_at(b, len(b))


BASE = {"seeds": [], "snaps": []}                      # distinct prefixes, prefilled once per process


def prefill_base(D):
    p = args.ctx - 1
    stride = 997
    t0 = time.time()
    sink_on(False)
    for s in range(D):
        ck(L.qwn_paged_reset_slot(s), "reset")
        ids = all_ids[s * stride: s * stride + args.ctx]
        assert len(ids) == args.ctx
        BASE["seeds"].append(paged_prefill_slot(L, s, ids, 128))
        BASE["snaps"].append(snapshot(s, p))
    print(f"  prefill {D} distinct x {args.ctx} in {time.time() - t0:.1f}s", flush=True)


DIFF = {"cols": 0, "logit": 0, "argmax": 0, "maxd": 0.0, "gaps": []}   # unpinned band (--no-pin): per-column statistics


def top2_gap(lg):
    v = torch.topk(lg, 2).values
    return (v[0] - v[1]).item()


def band_report(tag, cols, flips, gaps, all_gaps):
    """flips per 1000 columns and the top-2 gap (reference logits) at the flip positions vs all positions."""
    gs = sorted(gaps); ag = sorted(all_gaps)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
    print(f"[band] {tag}: {flips} argmax flips in {cols} columns = {1000.0 * flips / max(1, cols):.2f} per 1000; "
          f"top-2 gap at flips: max {gs[-1] if gs else float('nan'):.4f}, median {q(gs, 0.5):.4f}; "
          f"all columns: median gap {q(ag, 0.5):.3f}, 1 % quantile {q(ag, 0.01):.4f}, "
          f"fraction of columns with gap < max-flip-gap: {sum(1 for g in ag if g < (gs[-1] if gs else 0)) / max(1, len(ag)):.4f}",
          flush=True)


def run_case(N):
    global fails
    f0 = fails
    p = args.ctx - 1                                    # last prefilled position
    D = min(N, len(BASE["snaps"]))
    sink_on(False)
    seeds, snaps = [], []
    for s in range(N):                                  # slot s = prefix s % D
        seeds.append(BASE["seeds"][s % D]); snaps.append(BASE["snaps"][s % D])
        restore(s, p, snaps[s])
    print(f"  N={N}: {D} distinct prefixes + {N - D} clones", flush=True)
    sink_on(True)

    # ---- 2. reference: S lockstep greedy decode steps; t[s][i] fed at p+1+i, logits ref_lg[s][i]
    t = [[seeds[s]] for s in range(N)]
    ref_lg = [[] for _ in range(N)]
    t0 = time.time()
    for i in range(S):
        am, lg = decode([t[s][i] for s in range(N)], list(range(N)), [p + 1 + i] * N)
        for s in range(N):
            t[s].append(am[s]); ref_lg[s].append(lg[s])
    ref_snap = [snapshot(s, p + S) for s in range(N)]
    print(f"  reference {S} steps in {time.time() - t0:.1f}s; slot0 text={tok.decode(t[0][1:12])!r}", flush=True)
    all_gaps = [top2_gap(ref_lg[s][i]) for s in range(N) for i in range(S)]
    d0 = dict(DIFF); d0["gaps"] = len(DIFF["gaps"]); d0["ndmax"] = len(DIFF.get("dmax", []))

    # ---- decode-vs-decode band: the same N slots inside a wider lockstep decode (args.band columns), teacher-forced
    if args.band > N:
        B = args.band
        for s in range(B):
            restore(s, p, snaps[s % N] if s < N else snaps[s % N])
        nxt = [t[s % N][0] for s in range(B)]
        flips = 0; gaps = []; dmax = []
        for i in range(S):
            am, lg = decode(nxt, list(range(B)), [p + 1 + i] * B)
            for s in range(N):
                if am[s] != t[s][i + 1]:
                    flips += 1; gaps.append(top2_gap(ref_lg[s][i]))
                dmax.append((lg[s] - ref_lg[s][i]).abs().max().item())
                nxt[s] = t[s][i + 1]                       # teacher-forced on the reference tokens
            for s in range(N, B):
                nxt[s] = am[s]
        band_report(f"plain decode N={N} vs the same slots in N={B}", N * S, flips, gaps, all_gaps)
        dm = sorted(dmax)
        print(f"[band]   logits max |d| per column: max {dm[-1]:.3f}, p99 {dm[int(0.99 * len(dm))]:.3f}, median {dm[len(dm) // 2]:.3f}", flush=True)

    def check_cols(tag, cols, segs, am, lg):
        """every column j of segment k (slot s, cursor c): logits == ref_lg[s][c + j], argmax == t[s][c + j + 1]."""
        bad = 0
        for (s, off, n) in segs:
            c = cols[off][2] - (p + 1)
            for j in range(n):
                DIFF["cols"] += 1
                if am[off + j] != t[s][c + j + 1]:
                    bad += 1; DIFF["argmax"] += 1; DIFF["gaps"].append(top2_gap(ref_lg[s][c + j]))
                    if bad <= 3:
                        print(f"    {tag}: slot {s} col {j} pos {cols[off + j][2]} argmax {am[off + j]} != ref {t[s][c + j + 1]}")
                if not torch.equal(lg[off + j], ref_lg[s][c + j]):
                    d = (lg[off + j] - ref_lg[s][c + j]).abs().max().item()
                    bad += 1; DIFF["logit"] += 1; DIFF["maxd"] = max(DIFF["maxd"], d); DIFF.setdefault("dmax", []).append(d)
                    if bad <= 3:
                        print(f"    {tag}: slot {s} col {j} pos {cols[off + j][2]} logits differ (max |d| {d:.3e}, "
                              f"{(lg[off + j] != ref_lg[s][c + j]).sum().item()} of {V} elems)")
        return bad

    def finish(tag, cur):
        """Bring every slot to cursor S with N-wide verify waves (teacher drafts; a slot already at S rides along
        with one discarded column so every wave keeps N segments = the reference's client count), checking the
        columns' logits, then compare the slot snapshots with the reference's."""
        bad = 0
        while any(cur[s] < S for s in range(N)):
            cols, segs, acc = [], [], []
            for s in range(N):
                rem = S - cur[s]
                K = 0 if rem <= 0 else min(KMAX, rem - 1)
                off = len(cols)
                for j in range(K + 1):
                    cols.append((t[s][cur[s] + j], s, p + 1 + cur[s] + j))
                segs.append((s, off, K + 1)); acc.append(0 if rem <= 0 else K + 1)
            am, lg = verify(cols, segs)
            bad += check_cols(f"{tag} top-up", cols, [sg for sg, a in zip(segs, acc) if a > 0], am, lg)
            commit(list(range(N)), acc)
            for s in range(N):
                cur[s] += acc[s]
        for s in range(N):
            if not same_bytes(snapshot(s, p + S), ref_snap[s]):
                bad += 1
                print(f"    {tag}: slot {s} final state (save bytes) differs from the reference")
        return bad

    # ---- 3. spec rounds with teacher drafts, mixed K_i, partial / zero commits, parking of one slot
    for s in range(N):
        restore(s, p, snaps[s])
    cur = [0] * N
    ncols = nver = 0
    t0 = time.time()
    bad = 0
    for r in range(R):
        cols, segs = [], []
        Ks = []
        for s in range(N):
            K = (s + r) % (KMAX + 1)
            Ks.append(K)
            off = len(cols)
            for j in range(K + 1):
                cols.append((t[s][cur[s] + j], s, p + 1 + cur[s] + j))
            segs.append((s, off, K + 1))
        am, lg = verify(cols, segs)
        nver += 1; ncols += len(cols)
        bad += check_cols(f"round {r}", cols, segs, am, lg)
        acc = []
        for s in range(N):
            K = Ks[s]
            if r % 3 == 2 and s == r % N:
                n_acc = 0                                         # discard everything (state must not move)
            elif (r + s) % 2 == 0:
                n_acc = K + 1                                     # all
            else:
                n_acc = 1 + rng.randrange(K + 1)                  # partial (1..K+1)
            acc.append(n_acc)
        commit(list(range(N)), acc)
        for s in range(N):
            cur[s] += acc[s]
        if r == 1:                                                # parking right after a commit
            s = N - 1
            buf = snapshot(s, p + cur[s])
            restore(s, p + cur[s], buf)
    if args.no_pin:
        band_report(f"verify waves (teacher drafts, mixed K) vs plain decode N={N}", DIFF["cols"] - d0["cols"],
                    DIFF["argmax"] - d0["argmax"], DIFF["gaps"][d0["gaps"]:], all_gaps)
        dm = sorted(DIFF.get("dmax", [])[d0.get("ndmax", 0):])
        if dm:
            print(f"[band]   logits max |d| per column: max {dm[-1]:.3f}, p99 {dm[int(0.99 * len(dm))]:.3f}, median {dm[len(dm) // 2]:.3f}", flush=True)
    bad += finish("teacher", cur)
    print(f"  teacher drafts: {nver} verify waves, {ncols} columns, {R} commits in {time.time() - t0:.1f}s -> "
          f"{'OK' if bad == 0 else f'{bad} mismatches'}", flush=True)
    if bad:
        fail(f"ctx {args.ctx} N {N} teacher-draft phase")

    # ---- 4. wrong drafts: the server's acceptance rule; committed tokens must be the greedy sequence
    for s in range(N):
        restore(s, p, snaps[s])
    cur = [0] * N
    bad = 0
    nacc_hist = [0] * (KMAX + 2)
    t0 = time.time()
    rounds = 0
    while all(cur[s] + KMAX + 1 <= S for s in range(N)):
        cols, segs, drafts = [], [], []
        for s in range(N):
            K = (s + rounds) % (KMAX + 1)
            off = len(cols)
            d = [t[s][cur[s]]]
            for j in range(1, K + 1):
                d.append(t[s][cur[s] + j] if rng.random() < 0.6 else rng.randrange(1000, 100000))
            for j in range(K + 1):
                cols.append((d[j], s, p + 1 + cur[s] + j))
            segs.append((s, off, K + 1)); drafts.append(d)
        am, lg = verify(cols, segs)
        acc = []
        for s, (slot, off, n) in enumerate(segs):
            d = drafts[s]
            j = 0
            while j + 1 < n and am[off + j] == d[j + 1]:
                j += 1
            n_acc = j + 1                                         # columns 0..j committed; bonus = am[off + j]
            committed = d[1:n_acc] + [am[off + j]]
            expect = t[s][cur[s] + 1: cur[s] + 1 + n_acc]
            if committed != expect:
                bad += 1
                if bad <= 3:
                    print(f"    wrong-draft round {rounds} slot {slot}: committed {committed} != greedy {expect}")
            # the accepted columns' logits are the reference's (their prefix is the teacher's)
            for jj in range(n_acc):
                if not torch.equal(lg[off + jj], ref_lg[slot][cur[slot] + jj]):
                    bad += 1
                    if bad <= 3:
                        print(f"    wrong-draft round {rounds} slot {slot} col {jj}: accepted column logits differ")
            acc.append(n_acc); nacc_hist[n_acc] += 1
        commit(list(range(N)), acc)
        for s in range(N):
            cur[s] += acc[s]
        rounds += 1
    bad += finish("wrong-draft", cur)
    print(f"  wrong drafts: {rounds} rounds, accepted-columns histogram {nacc_hist[1:]} in {time.time() - t0:.1f}s -> "
          f"{'OK' if bad == 0 else f'{bad} mismatches'}", flush=True)
    if bad:
        fail(f"ctx {args.ctx} N {N} wrong-draft phase")
    print(f"[spec-check] ctx {args.ctx} N {N}: {'PASS' if fails == f0 else 'FAIL'}", flush=True)


def bench():
    """verify-wave ms vs decode-step ms at T columns (32K default ctx), and the commit replay ms."""
    p = args.ctx - 1
    sink_on(False)
    ids = all_ids[:args.ctx]
    ck(L.qwn_paged_reset_slot(0), "reset")
    seed = paged_prefill_slot(L, 0, ids, 128)
    snap = snapshot(0, p)
    for s in range(1, Nmax):
        restore(s, p, snap)
    print(f"  bench: {Nmax} slots x {args.ctx}", flush=True)

    def time_it(fn, iters):
        fn(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return 1000.0 * (time.time() - t0) / iters

    out = (ctypes.c_int * 64)()
    res = {}
    for N in [n for n in (1, 2, 4, 8, 12, 16) if n <= Nmax]:
        def dec():
            ck(L.qwn_paged_decode_step(_ci([seed] * N), _ci(list(range(N))), _ci([p + 1] * N), N, out), "decode")
        ms = time_it(dec, args.bench_iters)
        res[("decode", N)] = ms
        print(f"  decode step N={N:2d}: {ms:7.2f} ms", flush=True)
    for (N, K) in [(1, 0), (1, 3), (2, 3), (4, 3), (8, 3), (8, 7), (12, 4), (16, 3)]:
        if N > Nmax:
            continue
        T = N * (K + 1)
        toks = [seed] * T
        cslot = [s for s in range(N) for _ in range(K + 1)]
        cpos = [p + 1 + j for _ in range(N) for j in range(K + 1)]
        segs = [(s, s * (K + 1), K + 1) for s in range(N)]
        def ver():
            ck(L.qwn_paged_verify(_ci(toks), _ci(cslot), _ci(cpos), _ci([x[0] for x in segs]), _ci([x[1] for x in segs]),
                                  _ci([x[2] for x in segs]), N, T, out), "verify")
        ms = time_it(ver, args.bench_iters)
        def ver_commit_all():
            ver(); ck(L.qwn_paged_commit(_ci(list(range(N))), _ci([K + 1] * N), N), "commit")
        ms_vc = time_it(ver_commit_all, args.bench_iters)
        def ver_commit_one():
            ver(); ck(L.qwn_paged_commit(_ci(list(range(N))), _ci([1] * N), N), "commit")
        ms_vc1 = time_it(ver_commit_one, args.bench_iters)
        print(f"  verify N={N:2d} K={K} T={T:2d}: {ms:7.2f} ms; + commit all {ms_vc - ms:6.2f} ms, "
              f"+ commit 1 col {ms_vc1 - ms:6.2f} ms  (decode step at N={N}: {res.get(('decode', N), float('nan')):.2f} ms)",
              flush=True)


if args.bench:
    bench()
else:
    prefill_base(min(Nmax, args.distinct))
    for N in Ns:
        run_case(N)
    if args.no_pin:
        print(f"[spec-check] unpinned band over {DIFF['cols']} verify columns: {DIFF['logit']} with different logits "
              f"(max |d| {DIFF['maxd']:.3e}), {DIFF['argmax']} argmax flips", flush=True)
    print(f"[spec-check] {'ALL PASS' if fails == 0 else f'{fails} FAILURES'} (ctx {args.ctx}, Ns {Ns})", flush=True)
L.qwn_paged_free(); L.qwn_free()
sys.exit(1 if fails else 0)
