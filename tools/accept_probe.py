#!/usr/bin/env python3
"""Teacher-forced accept probe (degeneration-free A/B for draft quality).

Runs the REAL round machinery (tree build + batched verify + commit + trunk
advance) but teacher-forces the committed trajectory to a fixed reference text,
so self-generated loops can't inflate accept. Reports the real spec acceptance
(draft path matching the verify argmax) and kernel-side tok/s.

Port of tl-e tools/e2e_honest.py to the qwentin qwn_* ABI. Use for the
calibrated-MTP-head A/B (TQ_MTP_WEIGHTS=<dir>), tree-shape sweeps, etc.

Env: TQ_REF_TEXT (comma list; default internal/bench/ref_prose.txt),
     PROMPT (48), STEPS (384), DEPTH/K/MAXN/TAU (6/3/8/12).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, dfs_order, ck
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get("TQ_REF_TEXT", os.path.join(ROOT, "internal", "bench", "ref_prose.txt"))
P = int(os.environ.get("PROMPT", "48"))
STEPS = int(os.environ.get("STEPS", "384"))
DEPTH = int(os.environ.get("DEPTH", "6"))
K = int(os.environ.get("K", "3"))
TAU = float(os.environ.get("TAU", "12"))
MAXN = int(os.environ.get("MAXN", "8"))

tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.6-27B", trust_remote_code=True)
REF_LIST = [r for r in REF.split(",") if r]
REF_IDS = {}
for _r in REF_LIST:
    _ids = tok(open(_r).read(), add_special_tokens=False).input_ids
    assert len(_ids) >= P + STEPS + DEPTH + 4, f"reference too short: {_r} ({len(_ids)})"
    REF_IDS[_r] = _ids

L = load_lib(os.environ.get("TQ_LIB", "/workspace/qwentin/build-qwen/libforward_qwen.so"))
TQF = os.environ.get("TQ_MODEL_TQF", "/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
ck(L.qwn_init(TQF.encode()), "init")
e = Eng(L)


def run_ref(ref, ids):
    prefill(e, ids, P - 1)
    e.snapshot_root(); e.save(0)
    base_pos = P - 1
    idx = P
    seed = ids[idx]
    committed = [ids[idx]]
    rl = []
    prof = {"tree": 0.0, "verify": 0.0, "commit": 0.0, "mtp": 0.0}
    while len(committed) < STEPS:
        t = time.time()
        tt, tp, td = e.tree(seed, base_pos, DEPTH, K, TAU, MAXN - 1)
        prof["tree"] += time.time() - t
        dtok, dpar, ddep, dpos, dch = dfs_order(seed, tt, tp, td, base_pos)
        t = time.time()
        am = e.spec_forward(dtok, dpar, ddep, dpos)
        prof["verify"] += time.time() - t
        # real accept criterion: descend matching the verify argmax
        ap = [0]; a = am[0]
        while True:
            nxt = None
            for (ci, ctok) in dch.get(ap[-1], []):
                if ctok == a:
                    nxt = ci; break
            if nxt is None:
                break
            ap.append(nxt); a = am[nxt]
        rl.append(len(ap))
        # trajectory: teacher-forced walk along the reference continuation
        path = [0]; rp = idx + 1
        while True:
            nxt = None
            for (ci, ctok) in dch.get(path[-1], []):
                if ctok == ids[rp]:
                    nxt = ci; break
            if nxt is None:
                break
            path.append(nxt); rp += 1
        m = len(path)
        accepted = [ids[idx + 1 + j] for j in range(m)]
        path_pos = [base_pos + 1 + j for j in range(m)]
        t = time.time()
        e.commit(path, path_pos)
        prof["commit"] += time.time() - t
        t = time.time()
        e.set_root_from(path[-1])
        for j in range(m):
            e.advance_from_spec(path[j], accepted[j], base_pos + 1 + j)
        e.save(0)
        prof["mtp"] += time.time() - t
        committed.extend(accepted)
        base_pos += m
        idx += m
        seed = ids[idx]
    committed = committed[:STEPS]
    assert committed == ids[P:P + len(committed)], "teacher-forcing drifted"
    rl = np.array(rl)
    kt = sum(prof.values())
    mw = os.environ.get("TQ_MTP_WEIGHTS", "")
    print(f"ref={os.path.basename(ref)} steps={STEPS} shape=(d{DEPTH} k{K} mn{MAXN} tau{TAU}) "
          f"head={'CALIB:' + os.path.basename(mw) if mw else 'stock'}", flush=True)
    print(f"  accept/round = {rl.mean():.3f}  ({len(rl)} rounds)", flush=True)
    print(f"  kernel tok/s = {rl.sum()/kt:.1f}   (tree {1000*prof['tree']/len(rl):.2f} "
          f"verify {1000*prof['verify']/len(rl):.2f} commit {1000*prof['commit']/len(rl):.2f} "
          f"mtp {1000*prof['mtp']/len(rl):.2f} ms/round)", flush=True)


try:
    for _r in REF_LIST:
        run_ref(_r, REF_IDS[_r])
finally:
    L.qwn_free()
