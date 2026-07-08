#!/usr/bin/env python3
"""Decode-only spec-round benchmark: prefill once (wide path if enabled), then
time M production C-rounds (qwn_spec_round). Reports ms/round, net tok/s and
accept-length without the dense baseline / divergence passes of mtp_spec_smoke.

Optionally brackets the timed rounds with cudaProfilerStart/Stop so
`nsys profile -c cudaProfilerApi` captures exactly the steady-state decode.

Run: CUDA_VISIBLE_DEVICES=3 TQ_KV_Q4=1 TQ_CTX=262144 python3 tools/bench_rounds.py \
        --prompt-tokens 1024 --rounds 200
"""
from __future__ import annotations
import argparse, ctypes, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck, BIGTEXT
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
ap.add_argument("--prompt-tokens", type=int, default=1024)
ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--tau", type=float, default=12.0)
ap.add_argument("--maxn", type=int, default=8)
ap.add_argument("--profile", action="store_true",
                help="cudaProfilerStart/Stop around the timed rounds (for nsys -c cudaProfilerApi)")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
text = open(BIGTEXT).read()
ids = tok(text, add_special_tokens=False).input_ids
P = args.prompt_tokens
assert len(ids) >= P + 8, f"bench text too short: {len(ids)}"

L = load_lib(args.lib)
print(f"loading {args.tqf} ...", flush=True)
ck(L.qwn_init(args.tqf.encode()), "init")
e = Eng(L)

t0 = time.time()
seed = prefill(e, ids, P - 1)
e.snapshot_root()
tp = time.time() - t0
print(f"prefill {P} tok in {tp:.2f}s = {P/tp:.0f} tok/s", flush=True)

chain_buf = (ctypes.c_int * (args.maxn + 2))()
state = (ctypes.c_int * 2)()
base_pos = P - 1

def round_once(sd, bp):
    cl = L.qwn_spec_round(int(sd), int(bp), args.depth, args.k,
                          ctypes.c_float(args.tau), args.maxn, chain_buf, state)
    ck(cl, "spec_round")
    return cl, state[0], state[1]

for _ in range(args.warmup):
    cl, seed, base_pos = round_once(seed, base_pos)

if args.profile:
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaProfilerStart()

times = []
lens = []
t0 = time.time()
for _ in range(args.rounds):
    t = time.time()
    cl, seed, base_pos = round_once(seed, base_pos)
    times.append(time.time() - t)
    lens.append(cl)
wall = time.time() - t0

if args.profile:
    rt.cudaProfilerStop()

times = np.array(times) * 1000.0
lens = np.array(lens)
net = lens - 1          # chain includes the previous round's bonus seed
print(f"rounds={args.rounds}  ms/round: mean={times.mean():.2f}  p50={np.percentile(times,50):.2f}  "
      f"p90={np.percentile(times,90):.2f}  min={times.min():.2f}", flush=True)
print(f"accept: mean_chain={lens.mean():.2f}  net_commit/round={net.mean():.2f}", flush=True)
print(f"decode-only tok/s = {net.sum()/wall:.2f}   (ctx now {base_pos})", flush=True)
L.qwn_free()
