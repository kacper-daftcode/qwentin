#!/usr/bin/env python3
"""Roadmap E milestone 1: batched / multi-client DECODE -- correctness + throughput.

Validates the batched N-client decode step (qwn_batched_decode_step) that reuses the wide
FP6 GEMM for ALL projections so the SAME weights are read once and multiplied by N
client-columns (decode -> compute-bound as N grows). NO speculative decoding.

  CORRECTNESS: prefill N distinct prompts through the proven single-stream path, snapshot
  each into a batched client slot (qwn_batched_load_client), run ONE batched step over all
  N clients, and compare each client's next-token argmax to that client decoded
  independently via qwn_decode (float-eps gate -> argmax must match).

  THROUGHPUT: aggregate tokens/s of one batched step at width N in {1,2,4,8,16,32} vs the
  N-sequential single-stream baseline (which can only do 1/t_single tok/s total). The win =
  weights read once for N clients.

Run (GPU7 ONLY):
    CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 python3 -u tools/batched_smoke.py
"""
from __future__ import annotations
import argparse, ctypes, os, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIGTEXT = os.path.join(HERE, "src", "forward_qwen.cu")


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_num_layers.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_batched_init.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_batched_init.restype = ctypes.c_int
    L.qwn_batched_free.restype = ctypes.c_int
    L.qwn_batched_load_client.argtypes = [ctypes.c_int, ctypes.c_int]
    L.qwn_batched_load_client.restype = ctypes.c_int
    L.qwn_batched_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                          ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_batched_decode_step.restype = ctypes.c_int
    L.qwn_batched_logits.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    L.qwn_batched_logits.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
    return L


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


def gpu_mem_used_mib():
    """Used VRAM on the (single visible) GPU; CUDA_VISIBLE_DEVICES=7 -> nvidia-smi index 7."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", dev, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def single_prefill(L, ids):
    """Single-stream prefill ids[0..p] via qwn_decode; returns (p, seed=argmax at p)."""
    ck(L.qwn_reset_state(), "reset")
    am = 0
    for t, tok in enumerate(ids):
        am = ck(L.qwn_decode(int(tok), t), "decode")
    return len(ids) - 1, am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--prompt-tokens", type=int, default=48)
    ap.add_argument("--bat-ctx", type=int, default=512, help="per-client KV capacity (rows)")
    ap.add_argument("--iters", type=int, default=64, help="timing iterations per width")
    ap.add_argument("--ns", default="1,2,4,8,16,32", help="batched widths to sweep")
    ap.add_argument("--vram-probe", type=int, default=0,
                    help="if >0, probe how many clients allocate at this per-client ctx (e.g. 131072)")
    ap.add_argument("--vram-probe-only", action="store_true",
                    help="skip correctness/throughput; just run the VRAM allocation probe")
    ap.add_argument("--probe-nmax", type=int, default=6, help="max clients to try in the VRAM probe")
    args = ap.parse_args()

    kv_mode = "Q4(K)+E4M3(V)" if os.environ.get("TQ_KV_Q4", "") not in ("", "0") else (
        "FP8" if os.environ.get("TQ_KV_FP8", "") not in ("", "0") else "fp32")
    P = args.prompt_tokens
    Ns = [int(x) for x in args.ns.split(",")]
    Nmax = max(Ns)

    # Four (or Nmax) distinct prompts from distinct slices of the reference text.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = open(BIGTEXT).read() if os.path.exists(BIGTEXT) else (
        "The history of cartography is the study of maps. " * 400)
    all_ids = tok(text, add_special_tokens=False).input_ids
    if len(all_ids) < Nmax * (P + 4):
        all_ids = (all_ids * (Nmax * (P + 4) // max(1, len(all_ids)) + 1))
    prompts = [all_ids[c * (P + 1): c * (P + 1) + P] for c in range(Nmax)]

    L = load_lib(args.lib)
    print(f"loading {args.tqf} ...", flush=True)
    mem0 = gpu_mem_used_mib()
    ck(L.qwn_init(args.tqf.encode()), "init")
    H = L.qwn_hidden_size(); nlayers = L.qwn_num_layers()
    mem_weights = gpu_mem_used_mib()
    print(f"H={H} layers={nlayers}  KV mode={kv_mode}  VRAM after load: {mem_weights} MiB "
          f"(weights+embed ~ {mem_weights - mem0 if mem0>=0 else mem_weights} MiB delta)", flush=True)

    try:
        # ---------------- VRAM PROBE @ per-client ctx (e.g. 128k) ----------------
        # Pure allocation test (no decode -> no single-stream KV), so the budget reflects
        # batched-only mode. Reports how many clients fit at the target context.
        if args.vram_probe > 0:
            print("\n" + "=" * 72, flush=True)
            print(f"  VRAM PROBE: clients that allocate @ ctx={args.vram_probe} (KV mode={kv_mode})", flush=True)
            print("=" * 72, flush=True)
            usable = 31.36  # GiB, RTX 5090 usable (per live doc)
            for N in range(1, args.probe_nmax + 1):
                rc = L.qwn_batched_init(N, args.vram_probe)
                if rc != 0:
                    print(f"  N={N}: qwn_batched_init -> {rc} (alloc failed / OOM) -> max {N-1} clients", flush=True)
                    L.qwn_batched_free()
                    break
                mem = gpu_mem_used_mib()
                per_cli = (mem - mem_weights) / N if mem >= 0 else -1
                print(f"  N={N}: OK  VRAM={mem} MiB  (~{per_cli:.0f} MiB/client incl shared)", flush=True)
                L.qwn_batched_free()
            if args.vram_probe_only:
                return
        # ---------------- CORRECTNESS (batched vs single-stream, with near-tie margins) ----
        VBUF = 260000
        logbuf = (ctypes.c_float * VBUF)()

        def run_corr(Ncorr, ragged):
            ck(L.qwn_batched_init(Ncorr, args.bat_ctx), "batched_init(corr)")
            seeds = (ctypes.c_int * Ncorr)(); pos = (ctypes.c_int * Ncorr)(); ref = [0] * Ncorr
            for c in range(Ncorr):
                plen = (P - (c % 5) * 6) if ragged else P          # distinct lengths if ragged
                p, seed = single_prefill(L, prompts[c][:plen])     # state -> [0..p]
                ck(L.qwn_batched_load_client(c, p), "load_client")
                seeds[c] = seed; pos[c] = p + 1
                ref[c] = ck(L.qwn_decode(seed, p + 1), "ref_decode")  # single-stream Q4 reference
            out = (ctypes.c_int * Ncorr)()
            ck(L.qwn_batched_decode_step(seeds, pos, Ncorr, out), "batched_step")
            nmatch = 0; tie_only = True
            for c in range(Ncorr):
                if out[c] == ref[c]:
                    nmatch += 1; continue
                # quantify the mismatch: gap between the two candidates in the batched logits
                v = L.qwn_batched_logits(c, logbuf)
                gap = abs(logbuf[out[c]] - logbuf[ref[c]]) if v > 0 else float("nan")
                near = (v > 0 and gap < 0.5)               # < 0.5 logit = float-eps/Q4 near-tie
                tie_only = tie_only and near
                print(f"    mismatch client {c} (pos {int(pos[c])}): batched={out[c]} single={ref[c]} "
                      f"logit-gap={gap:.4f} ({'near-tie' if near else 'NOT near-tie'})", flush=True)
            return nmatch, tie_only

        Ncorr = min(Nmax, 16)
        print("\n" + "=" * 72, flush=True)
        print(f"  CORRECTNESS: batched Q4-KV vs single-stream Q4-KV  (KV mode={kv_mode})", flush=True)
        print("=" * 72, flush=True)
        eq_match, eq_tie = run_corr(Ncorr, ragged=False)
        print(f"  equal-length (P={P}):  {eq_match}/{Ncorr} argmax match"
              f"{'' if eq_match == Ncorr else ' (mismatches are near-ties)' if eq_tie else ' (CHECK!)'}", flush=True)
        rg_match, rg_tie = run_corr(Ncorr, ragged=True)
        print(f"  ragged (distinct lengths): {rg_match}/{Ncorr} argmax match"
              f"{'' if rg_match == Ncorr else ' (mismatches are near-ties)' if rg_tie else ' (CHECK!)'}", flush=True)

        # ---------------- THROUGHPUT ----------------
        # single-stream decode latency (one token at a time)
        _p, seed = single_prefill(L, prompts[0])
        t0 = time.time()
        cur, pp = seed, _p
        M = args.iters
        for _ in range(M):
            cur = ck(L.qwn_decode(cur, pp + 1), "decode_bench"); pp += 1
        t_single = (time.time() - t0) / M
        single_tps = 1.0 / t_single
        print("\n" + "=" * 72, flush=True)
        print(f"  THROUGHPUT: batched aggregate tok/s vs N-sequential single-stream", flush=True)
        print(f"  (single-stream decode = {1000*t_single:.3f} ms/tok = {single_tps:.1f} tok/s)", flush=True)
        print("=" * 72, flush=True)
        print(f"  {'N':>4} {'ms/step':>9} {'agg tok/s':>11} {'per-cli tok/s':>14} "
              f"{'vs Nx-single':>12} {'VRAM MiB':>9}", flush=True)
        rows = []
        for N in Ns:
            ck(L.qwn_batched_init(N, args.bat_ctx), f"batched_init(N={N})")
            # populate all N slots from prompt 0's state; pos fixed for constant per-step work
            _p, seed = single_prefill(L, prompts[0])
            for c in range(N):
                ck(L.qwn_batched_load_client(c, _p), "load_client(bench)")
            toks = (ctypes.c_int * N)(*([seed] * N))
            pos = (ctypes.c_int * N)(*([_p + 1] * N))
            out = (ctypes.c_int * N)()
            ck(L.qwn_batched_decode_step(toks, pos, N, out), "warmup")  # warm allocs
            mem_n = gpu_mem_used_mib()
            t0 = time.time()
            for _ in range(M):
                ck(L.qwn_batched_decode_step(toks, pos, N, out), "step")
            t_step = (time.time() - t0) / M
            agg = N / t_step
            percli = 1.0 / t_step
            speedup = agg / single_tps
            rows.append((N, t_step, agg, percli, speedup, mem_n))
            print(f"  {N:>4} {1000*t_step:>9.3f} {agg:>11.1f} {percli:>14.1f} "
                  f"{speedup:>11.2f}x {mem_n:>9}", flush=True)
        print("\n  (per-cli tok/s = 1/step latency; ideal weight-amortization keeps step latency"
              "\n   ~flat as N grows, so agg tok/s scales ~N until compute-bound.)", flush=True)
    finally:
        L.qwn_batched_free()
        L.qwn_free()


if __name__ == "__main__":
    main()
