#!/usr/bin/env python3
"""Validate the PyTorch MTP replication against the engine, position by position.

Teacher-forced walk (engine: decode + mtp_advance). At each position t:
  engine top-1 = qwn_mtp_tree_build(seed=x_{t+1}, pos=t, depth1 k1)
  torch  top-1 = argmax(lm_head(MTPHead(emb(x_{t+1}), h_t)))  over the walked seq
The torch attention covers positions [P0..t] while the engine's MTP trunk also
holds [0..P0) from the prefill -- expect near-100% top-1 agreement, not 100%.
"""
from __future__ import annotations
import ctypes, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck
from mtp_calib_train import MTPHead, load_tensors, MTP_NAMES, P0 as TRAIN_P0
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get("TQ_REF_TEXT", os.path.join(ROOT, "bench", "ref_prose.txt"))
STEPS = int(os.environ.get("STEPS", "96"))
P = 48
tok = AutoTokenizer.from_pretrained("/workspace/models/Qwen3.6-27B", trust_remote_code=True)
ids = tok(open(REF).read(), add_special_tokens=False).input_ids
L = load_lib("/workspace/qwentin/build-qwen/libforward_qwen.so")
L.qwn_copy_mtp_debug.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
ck(L.qwn_init(b"/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf"), "init")
e = Eng(L)
H = e.H
hbuf = np.zeros(H, dtype=np.float32)

freed = False
try:
    prefill(e, ids, P - 1)                  # builds MTP trunk for 0..P-1
    p = P - 1
    eng_top1, hiddens, pairs = [], [], []
    toks4 = (ctypes.c_int * 4)(); pars4 = (ctypes.c_int * 4)()
    deps4 = (ctypes.c_int * 4)(); mrg4 = (ctypes.c_float * 4)()
    for t in range(STEPS):
        # hidden after decoding ids[p] sits in d_debug_x from the PREVIOUS decode;
        # current loop: decode ids[t+P-1... keep engine pattern: decode token at p+1
        e.snapshot_root()
        n = ck(L.qwn_mtp_tree_build(int(ids[p + 1]), int(p), 1, 1, ctypes.c_float(1e30), 1,
                                    toks4, pars4, deps4, mrg4), "tree")
        eng_top1.append(int(toks4[0]) if n >= 1 else -1)
        ck(L.qwn_copy_mtp_debug(4, hbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), H), "hid")
        hiddens.append(hbuf.copy())
        pairs.append(ids[p + 1])
        e.decode(ids[p + 1], p + 1)
        e.mtp_advance(ids[p + 2], p + 1)
        p += 1
    L.qwn_free()                            # release the 22.6 GB engine before torch
    freed = True
    # torch side: full-sequence forward over the same walk
    t = load_tensors(MTP_NAMES + ["model.language_model.embed_tokens.weight", "lm_head.weight"])
    embed = t["model.language_model.embed_tokens.weight"].cuda().to(torch.bfloat16)
    head_w = t["lm_head.weight"].cuda().to(torch.bfloat16)
    model = MTPHead(t).cuda().eval()
    hid = torch.from_numpy(np.stack(hiddens)).unsqueeze(0).cuda()
    nxt = torch.tensor(pairs, dtype=torch.long).unsqueeze(0).cuda()
    pos = (torch.arange(STEPS, device="cuda") + (P - 1)).unsqueeze(0)
    with torch.no_grad():
        hn = model(embed[nxt].float(), hid, pos)
        tt = []
        for c0 in range(0, STEPS, 32):
            logits = hn[:, c0:c0 + 32].to(torch.bfloat16) @ head_w.T
            tt.append(logits.float().argmax(-1)[0])
        torch_top1 = torch.cat(tt).cpu().numpy()
    eng = np.array(eng_top1)
    agree = float(np.mean(eng == torch_top1))
    h = STEPS // 2
    print(f"top-1 agreement engine vs torch: {100*agree:.1f}%  ({STEPS} positions)  "
          f"[first half {100*np.mean(eng[:h]==torch_top1[:h]):.1f}%, "
          f"second half {100*np.mean(eng[h:]==torch_top1[h:]):.1f}%]")
    if agree < 0.9:
        bad = np.where(eng != torch_top1)[0][:8]
        for i in bad:
            print(f"  pos{i}: engine={eng[i]} torch={int(torch_top1[i])}")
finally:
    if not freed:
        L.qwn_free()
