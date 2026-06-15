#!/usr/bin/env python3
"""Calibration dataset dumper for the MTP draft head.

Walks a diverse corpus TEACHER-FORCED through the FP6 engine in 16-token chain
chunks (the prefill machinery) and dumps, per position t:
  tokens[t]   -- the corpus token committed at t
  argmax[t]   -- the engine's FP6-greedy NEXT token after processing t
  hidden[t]   -- PRE-final-norm trunk hidden (the MTP root input), fp16

Segments are contiguous (SEG tokens each) so the trainer can replicate the MTP
layer's causal attention over real context. Output: one .npz per segment in
OUT_DIR (bench/calib by default).

Env: CORPUS (file list, ':'-separated; default = bench refs + repo sources),
SEG (default 512), MAX_POS (default 120000), OUT_DIR.
"""
from __future__ import annotations
import ctypes, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT, "bench", "calib"))
SEG = int(os.environ.get("SEG", "512"))
MAX_POS = int(os.environ.get("MAX_POS", "120000"))
P0 = 48                                    # warm prefill per segment

def default_corpus():
    files = [os.path.join(ROOT, "bench", "ref_prose.txt")]
    for pat in ("tools/*.py", "src/*.cu", "docs/*.md", "*.md"):
        files += sorted(glob.glob(os.path.join(ROOT, pat)))
    return files

corpus = (os.environ.get("CORPUS", "").split(":") if os.environ.get("CORPUS")
          else default_corpus())
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.6-27B", trust_remote_code=True)
text = "\n\n".join(open(f, errors="ignore").read() for f in corpus if os.path.exists(f))
ids = tok(text, add_special_tokens=False).input_ids
print(f"corpus: {len(corpus)} files, {len(ids)} tokens", flush=True)

L = load_lib("/workspace/qwentin/build-qwen/libforward_qwen.so")
L.qwn_spec_copy_hidden_pre.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
L.qwn_spec_copy_hidden_pre.restype = ctypes.c_int
ck(L.qwn_init(b"/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf"), "init")
e = Eng(L)
H = e.H
os.makedirs(OUT_DIR, exist_ok=True)

CH = 16
hbuf = np.zeros((CH, H), dtype=np.float32)
try:
    n_seg = min(len(ids) // (P0 + SEG + 1), (MAX_POS + SEG - 1) // SEG)
    total = 0
    for s in range(n_seg):
        base = s * (P0 + SEG)
        seg_ids = ids[base:base + P0 + SEG + 1]
        prefill(e, seg_ids, P0 - 1, build_trunk=False)
        base_pos = P0 - 1
        toks, ams, hids = [], [], []
        t = P0
        while t + CH <= len(seg_ids) and len(toks) < SEG:
            chunk = seg_ids[t:t + CH]
            n = len(chunk)
            ct = (ctypes.c_int * n)(*chunk)
            cp = (ctypes.c_int * n)(*range(-1, n - 1))
            cd = (ctypes.c_int * n)(*range(n))
            cpos = (ctypes.c_int * n)(*[base_pos + 1 + j for j in range(n)])
            am = (ctypes.c_int * n)()
            ck(L.qwn_spec_forward_test(ct, cp, cd, cpos, n, am, None), "verify")
            ck(L.qwn_spec_copy_hidden_pre(hbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n), "hid")
            path = (ctypes.c_int * n)(*range(n))
            ck(L.qwn_spec_commit(path, cpos, n), "commit")
            toks.extend(chunk)
            ams.extend(am[:n])
            hids.append(hbuf[:n].astype(np.float16).copy())
            base_pos += n
            t += n
        np.savez_compressed(os.path.join(OUT_DIR, f"seg{s:05d}.npz"),
                            tokens=np.array(toks, dtype=np.int32),
                            argmax=np.array(ams, dtype=np.int32),
                            hidden=np.concatenate(hids, axis=0))
        total += len(toks)
        if (s + 1) % 10 == 0:
            print(f"  seg {s+1}/{n_seg}  positions={total}", flush=True)
    print(f"DONE: {total} positions in {n_seg} segments -> {OUT_DIR}", flush=True)
finally:
    L.qwn_free()
