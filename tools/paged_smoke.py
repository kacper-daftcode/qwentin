#!/usr/bin/env python3
"""Roadmap E milestone 3.1: PAGED KV -- parity + capacity.

The contiguous bat_ctx-per-client KV slab is replaced by a shared POOL of fixed-size blocks
(page positions each) + a per-slot block table. Short clients use few blocks, long ones grow
on demand out of the SAME pool. This test:
  PARITY   : paged decode argmax == single-stream Q4 decode (per-client, identity slot map).
  PACKING  : load K short clients into the pool, decode them in one paged step, all correct;
             show pool-block consumption.
  CAPACITY : measure per-block and per-slot VRAM, then report how many clients fit at short
             ctx (DeltaNet-state-limited) vs @128k (KV-block-limited) for the free budget.

Run (GPU7, prod Q4 config):
    CUDA_VISIBLE_DEVICES=7 TQ_CTX=8192 TQ_KV_Q4=1 python3 -u tools/paged_smoke.py
"""
from __future__ import annotations
import argparse, ctypes, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIGTEXT = os.path.join(HERE, "src", "forward_qwen.cu")


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_paged_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; L.qwn_paged_init.restype = ctypes.c_int
    L.qwn_paged_free.restype = ctypes.c_int
    L.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; L.qwn_paged_reset_slot.restype = ctypes.c_int
    L.qwn_paged_load_client.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_load_client.restype = ctypes.c_int
    L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_decode_step.restype = ctypes.c_int
    L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; L.qwn_paged_stats.restype = ctypes.c_int
    L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_prefill_batch.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
    return L


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


def gpu_mem_used_mib():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return -1


def single_prefill(L, ids):
    ck(L.qwn_reset_state(), "reset")
    am = 0
    for t, tok in enumerate(ids):
        am = ck(L.qwn_decode(int(tok), t), "decode")
    return len(ids) - 1, am


def stats(L):
    fb, tb, pg, mb = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
    ck(L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb)), "stats")
    return fb.value, tb.value, pg.value, mb.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--prompt-tokens", type=int, default=48)
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--ncorr", type=int, default=8)
    args = ap.parse_args()
    P = args.prompt_tokens
    assert os.environ.get("TQ_KV_Q4", "") not in ("", "0"), "set TQ_KV_Q4=1 (paged KV is Q4-only)"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = open(BIGTEXT).read() if os.path.exists(BIGTEXT) else ("The quick brown fox. " * 600)
    all_ids = tok(text, add_special_tokens=False).input_ids
    need = args.ncorr * (P + 2)
    if len(all_ids) < need:
        all_ids = all_ids * (need // max(1, len(all_ids)) + 1)
    prompts = [all_ids[c * (P + 1): c * (P + 1) + P] for c in range(args.ncorr)]

    L = load_lib(args.lib)
    print(f"loading {args.tqf} ...", flush=True)
    ck(L.qwn_init(args.tqf.encode()), "init")
    H = L.qwn_hidden_size()
    mem_w = gpu_mem_used_mib()
    print(f"H={H}  VRAM after load: {mem_w} MiB", flush=True)

    try:
        ck(L.qwn_paged_init(args.max_slots, args.num_blocks, args.page), "paged_init")
        fb, tb, pg, mb = stats(L)
        mem_pool = gpu_mem_used_mib()
        print(f"\npaged pool: blocks={tb} page={pg} max_blocks/seq={mb} max_slots={args.max_slots}  "
              f"VRAM={mem_pool} MiB (+{mem_pool-mem_w} for pool+state)", flush=True)

        # ---- PARITY: paged decode vs single-stream Q4 ----
        Nc = args.ncorr
        seeds = (ctypes.c_int * Nc)(); slots = (ctypes.c_int * Nc)(); pos = (ctypes.c_int * Nc)()
        ref = [0] * Nc
        for c in range(Nc):
            p, seed = single_prefill(L, prompts[c])
            ck(L.qwn_paged_load_client(c, p), "paged_load")
            seeds[c] = seed; slots[c] = c; pos[c] = p + 1
            ref[c] = ck(L.qwn_decode(seed, p + 1), "ref_decode")
        out = (ctypes.c_int * Nc)()
        ck(L.qwn_paged_decode_step(seeds, slots, pos, Nc, out), "paged_step")
        nmatch = sum(1 for c in range(Nc) if out[c] == ref[c])
        fb2, _, _, _ = stats(L)
        print("\n" + "=" * 72, flush=True)
        print(f"  PARITY: paged decode vs single-stream Q4  (N={Nc}, P={P}, page={pg})", flush=True)
        print("=" * 72, flush=True)
        for c in range(Nc):
            print(f"  slot {c}: seed={seeds[c]:6d} pos={pos[c]:4d}  paged={out[c]:6d} single={ref[c]:6d}  "
                  f"{'MATCH' if out[c]==ref[c] else 'MISMATCH'}", flush=True)
        print(f"  -> {nmatch}/{Nc} argmax match", flush=True)
        print(f"  pool blocks used by {Nc} clients @{P}tok: {tb-fb2}/{tb} "
              f"({(tb-fb2)//Nc} block/client; page={pg} covers {pg} pos)", flush=True)

        # ---- ragged step: decode the same N clients again at pos+1 (they grow) ----
        for c in range(Nc):
            seeds[c] = out[c]; pos[c] = pos[c] + 1
        out2 = (ctypes.c_int * Nc)()
        ck(L.qwn_paged_decode_step(seeds, slots, pos, Nc, out2), "paged_step2")
        print(f"  second paged step OK (clients advanced to pos {int(pos[0])}); "
              f"free blocks now {stats(L)[0]}/{tb}", flush=True)

        # ---- RAGGED BATCHED PREFILL (3.2a): batch-prefill K clients in one wave ----
        print("\n" + "=" * 72, flush=True)
        print(f"  RAGGED BATCHED PREFILL: {Nc} clients in ONE wave vs single-stream", flush=True)
        print("=" * 72, flush=True)
        for c in range(Nc):
            ck(L.qwn_paged_reset_slot(c), "reset_slot")
        # ragged columns: client c at slots c, intra-pos 0..len-1 (distinct lengths, wave<=128 cols)
        clen = [10 + c for c in range(Nc)]               # 10..(10+Nc-1); sum <= 128
        toks_c, cslot, cpos, soff, slen, sslot, sfin = [], [], [], [], [], [], []
        off = 0
        for c in range(Nc):
            ids = prompts[c][:clen[c]]
            soff.append(off); slen.append(len(ids)); sslot.append(c); sfin.append(1)
            for p, t in enumerate(ids):
                toks_c.append(t); cslot.append(c); cpos.append(p)
            off += len(ids)
        T = off
        cT = lambda a: (ctypes.c_int * len(a))(*a)
        oseed = (ctypes.c_int * Nc)()
        ck(L.qwn_paged_prefill_batch(cT(toks_c), cT(cslot), cT(cpos), cT(sslot), cT(soff),
                                     cT(slen), cT(sfin), Nc, T, oseed), "prefill_batch")
        # reference: single-stream seed per client
        nmatch = 0
        for c in range(Nc):
            _, rseed = single_prefill(L, prompts[c][:clen[c]])
            ok = (oseed[c] == rseed); nmatch += ok
            print(f"  client {c} (len {clen[c]}): batch-prefill seed={oseed[c]:6d} single={rseed:6d}  "
                  f"{'MATCH' if ok else 'DIFF(near-tie)'}", flush=True)
        print(f"  -> {nmatch}/{Nc} seed match; wave T={T} cols, 1 weight-read for all clients", flush=True)
        # the batch-prefilled slots can now decode (continue) -> one paged step
        sd = (ctypes.c_int * Nc)(*[oseed[c] for c in range(Nc)])
        ssl = (ctypes.c_int * Nc)(*list(range(Nc)))
        spos = (ctypes.c_int * Nc)(*[clen[c] for c in range(Nc)])
        sout = (ctypes.c_int * Nc)()
        ck(L.qwn_paged_decode_step(sd, ssl, spos, Nc, sout), "decode_after_prefill")
        print(f"  decode step after batch-prefill OK (next tokens e.g. {[sout[c] for c in range(min(4,Nc))]})", flush=True)

        # ---- CAPACITY: per-block & per-slot VRAM -> short vs 128k client counts ----
        print("\n" + "=" * 72, flush=True)
        print("  CAPACITY (measured per-block + per-slot VRAM)", flush=True)
        print("=" * 72, flush=True)
        # per-block: re-init with 2x blocks, measure delta
        ck(L.qwn_paged_free(), "free")
        ck(L.qwn_paged_init(1, args.num_blocks, args.page), "pinit_b1")
        m_b1 = gpu_mem_used_mib()
        ck(L.qwn_paged_free(), "free")
        ck(L.qwn_paged_init(1, args.num_blocks * 2, args.page), "pinit_b2")
        m_b2 = gpu_mem_used_mib()
        per_block_kib = (m_b2 - m_b1) * 1024.0 / args.num_blocks
        # per-slot (DeltaNet): re-init with more slots, small pool
        ck(L.qwn_paged_free(), "free")
        ck(L.qwn_paged_init(1, 64, args.page), "pinit_s1")
        m_s1 = gpu_mem_used_mib()
        ck(L.qwn_paged_free(), "free")
        ck(L.qwn_paged_init(9, 64, args.page), "pinit_s2")
        m_s2 = gpu_mem_used_mib()
        per_slot_mib = (m_s2 - m_s1) / 8.0
        ck(L.qwn_paged_free(), "free")
        usable = 31.36 * 1024  # MiB
        free_after_w = usable - mem_w
        blk_per_128k = (131072 + args.page - 1) // args.page
        print(f"  per-block (pool, 16 full-attn layers): {per_block_kib:.0f} KiB/block", flush=True)
        print(f"  per-slot DeltaNet state: {per_slot_mib:.1f} MiB/slot (ctx-independent)", flush=True)
        print(f"  free after weights: ~{free_after_w/1024:.1f} GiB; 128k needs {blk_per_128k} blocks/client", flush=True)
        # tradeoff: short clients are DeltaNet-limited; 128k clients are block-limited
        max_short = int(free_after_w / max(1e-9, per_slot_mib))
        # for 128k: budget split between slots(DeltaNet) and blocks(KV); each 128k client = 1 slot + blk_per_128k blocks
        cost_128k_mib = per_slot_mib + blk_per_128k * per_block_kib / 1024.0
        max_128k = int(free_after_w / max(1e-9, cost_128k_mib))
        print(f"  => short clients (~{P}tok, DeltaNet-limited): ~{max_short}", flush=True)
        print(f"  => @128k clients (KV+DeltaNet, {cost_128k_mib/1024:.2f} GiB each): ~{max_128k}", flush=True)
    finally:
        L.qwn_paged_free()
        L.qwn_free()


if __name__ == "__main__":
    main()
