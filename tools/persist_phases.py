#!/usr/bin/env python3
"""Phase-clock breakdown of the persistent spec kernel (part 0/1/2).

Reads the TQ_SPEC_PERSIST_PROF clock64 marks (qwn_spec_persist_clocks ABI)
after each of M rounds and prints per-phase work / barrier-tail time for the
LAST persistent launch of the round (prof=1 -> a part-1, prof=2 -> part-0,
prof=3 -> part-2; the kernel records CTA0's entry/exit marks per barrier).

Slot map (src tq_persist_bar_pf / tq_persist_exit):
  clk[0]      kernel start          clk[15]      kernel end
  clk[16+b]   barrier b entry       clk[b+1]     barrier b exit
Phase work = entry[b] - exit[prev b]; barrier tail = exit[b] - entry[b].

Run: CUDA_VISIBLE_DEVICES=4 TQ_KV_Q4=1 TQ_CTX=8192 TQ_SPEC_PERSIST_PROF=1 \
         python3 tools/persist_phases.py --prompt-tokens 1024 --rounds 30
Env:  SM_GHZ (cycles->us conversion, default 1.90)
"""
from __future__ import annotations
import argparse, ctypes, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck, BIGTEXT
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
ap.add_argument("--prompt-tokens", type=int, default=1024)
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
args = ap.parse_args()

PROF = int(os.environ.get("TQ_SPEC_PERSIST_PROF", "0"))
assert PROF in (1, 2, 3), "set TQ_SPEC_PERSIST_PROF=1 (part1) / 2 (part0) / 3 (part2)"
GHZ = float(os.environ.get("SM_GHZ", "1.90"))

# (label, barrier id) in flow order per part; None = phase runs to kernel end.
FLOWS = {
    1: [("quant1(post)", 0), ("gemv1(gate+up)", 1), ("silu*up", 4),
        ("quant2(mlp)", 5), ("gemv2(down)+reduce", None)],
    2: [("quant1(norm)", 0), ("gemv1(q/k/v)", 1), ("reduce q/k/v", 2),
        ("kvprep(rope)", 3), ("attn", 4), ("quant2", 5), ("gemv2(o)", None)],
    0: [("quant1(norm)", 0), ("gemv1(qkv/z/b/a)", 1), ("reduce qkv/b/a", 2),
        ("dnprep", 7), ("deltanet segs", 3), ("gated norm", 4),
        ("quant2(core)", 5), ("gemv2(out)", None)],
}
FLOW = FLOWS[{1: 1, 2: 0, 3: 2}[PROF]]

tok = AutoTokenizer.from_pretrained(os.path.dirname(args.tqf), trust_remote_code=True)
ids = tok(open(BIGTEXT).read(), add_special_tokens=False).input_ids
P = args.prompt_tokens
L = load_lib(args.lib)
ck(L.qwn_init(args.tqf.encode()), "init")
e = Eng(L)
seed = prefill(e, ids, P - 1)
e.snapshot_root()
L.qwn_spec_persist_clocks.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
L.qwn_spec_persist_clocks.restype = ctypes.c_int

chain_buf = (ctypes.c_int * 10)()
state = (ctypes.c_int * 2)()
base_pos = P - 1
rows = []
for r in range(args.warmup + args.rounds):
    cl = L.qwn_spec_round(int(seed), int(base_pos), 6, 3, ctypes.c_float(12.0), 8,
                          chain_buf, state)
    ck(cl, "spec_round")
    seed, base_pos = state[0], state[1]
    if r < args.warmup:
        continue
    clk = (ctypes.c_longlong * 32)()
    ck(L.qwn_spec_persist_clocks(clk), "clocks")
    start, end = clk[0], clk[15]
    prev_exit = start
    row = []
    ok = end > start
    for label, b in FLOW:
        if b is None:
            row.append((label, end - prev_exit, 0))
            break
        entry, exit_ = clk[16 + b], clk[b + 1]
        if not (entry >= prev_exit and exit_ >= entry):
            ok = False
            break
        row.append((label, entry - prev_exit, exit_ - entry))
        prev_exit = exit_
    if ok:
        rows.append((row, end - start))

assert rows, "no consistent clock snapshots (wrong prof/part flow?)"
n = len(rows)
total = np.mean([t for _, t in rows])
print(f"part={'1(MLP)' if PROF == 1 else '0(DeltaNet)' if PROF == 2 else '2(attn)'} "
      f"rounds={n}  total = {total:,.0f} cyc = {total / GHZ / 1000:.1f} us @ {GHZ} GHz")
print(f"{'phase':>20} {'work us':>9} {'bar us':>8} {'work %':>7}")
for i, (label, _) in enumerate(FLOW):
    w = np.mean([r[i][1] for r, _ in rows if len(r) > i])
    b = np.mean([r[i][2] for r, _ in rows if len(r) > i])
    print(f"{label:>20} {w / GHZ / 1000:9.1f} {b / GHZ / 1000:8.1f} {100 * w / total:6.1f}%")
bars = np.mean([sum(x[2] for x in r) for r, _ in rows])
print(f"{'ALL barriers':>20} {'':>9} {bars / GHZ / 1000:8.1f} {100 * bars / total:6.1f}%")
L.qwn_free()
