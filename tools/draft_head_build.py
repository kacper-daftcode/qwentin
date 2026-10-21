#!/usr/bin/env python3
"""Build the DRAFT lm_head of a .tqf (convert_qwen_tqf.py --draft-head): the paged drafter's reduced vocabulary.

The engine's drafter proposes K tokens per round through the model's own MTP layer and the lm_head; with the full
248 320-row 6-bit lm_head that read is 0.95 GB per draft step -- 72 % of the step, 2.3 ms of a 21 ms round at one
stream. The verify decides every token with the full head, so the drafter may use a smaller one: this tool takes the
V' most frequent token ids (tools/draft_vocab/qwen3_8-27b-ids-64k.txt: 65 536 ids that cover 99.7 % of the drafts the
served engine accepted), dequantizes the EXL3 checkpoint's lm_head with exllamav3's own reconstruct, gathers those rows
and quantizes them as one EXL3 Linear of --bits (6 = the checkpoint's lm_head bitrate: the draft head's argmax equals
the full head's on 99.4 % of positions; the outputs never change). ~10 s on one GPU; needs exllamav3 importable.

  python3 tools/draft_head_build.py /path/to/Qwen3.8-27B-exl3-opt5.75 -o /path/to/draft_head_64k.safetensors
  python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-opt5.75 -o /path/to/model-mtp4.tqf --mtp-exl3 \
      --draft-head /path/to/draft_head_64k.safetensors

Output: safetensors {draft.d2t int32 [V'] (the token id of each row, ascending), draft.lm_head.{trellis, suh, svh, mul1}}.
Options: --ids <file> (another id list, one id per line, '#' comments), --bits 4|5|6|8, --exllamav3 <checkout> when the
package is not installed. The frequency table behind the shipped list came from the research repo's
analysis/draft_vocab_coverage.py (code + the model's own chat answers); re-rank for a different workload if its token
distribution is very different (the coverage of the accepted drafts is what matters, not the generated text's).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import safe_open, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("model", help="the EXL3 checkpoint directory (its lm_head.* tensors)")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--ids", default=os.path.join(HERE, "draft_vocab", "qwen3_8-27b-ids-64k.txt"))
ap.add_argument("--bits", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exllamav3", default=None, help="path of an exllamav3 checkout to import from")
a = ap.parse_args()
if a.exllamav3:
    sys.path.insert(0, a.exllamav3)
from exllamav3.modules.quant.exl3 import LinearEXL3  # noqa: E402
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3  # noqa: E402

dev = torch.device("cuda")
t0 = time.time()
ids = np.array(sorted({int(l.split("#")[0]) for l in open(a.ids) if l.split("#")[0].strip()}), dtype=np.int64)
if ids.size % 128:
    raise SystemExit(f"{ids.size} ids: the draft vocabulary must be a multiple of 128")
with open(os.path.join(a.model, "config.json")) as f:
    cfg = json.load(f)
V = int((cfg.get("text_config") or cfg)["vocab_size"])
if ids.min() < 0 or ids.max() >= V:
    raise SystemExit(f"ids outside the vocabulary [0, {V})")
idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(a.model, idx[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


sd = {k: get(f"lm_head.{k}") for k in ("suh", "svh", "trellis", "mul1", "mcg") if f"lm_head.{k}" in idx}
if "trellis" not in sd:
    raise SystemExit("the checkpoint's lm_head is not an EXL3 Linear (lm_head.trellis missing)")
H = sd["suh"].numel()
lin = LinearEXL3(None, in_features=H, out_features=V, suh=sd["suh"], svh=sd["svh"], trellis=sd["trellis"],
                 mcg=sd.get("mcg"), mul1=sd.get("mul1"))
w_full = lin.get_weight_tensor().float()                           # [H, V] (in_features, out_features)
w_draft = w_full[:, torch.from_numpy(ids).to(dev)].contiguous()    # [H, V']
del w_full
print(f"[draft-head] lm_head {int(sd['trellis'].shape[2]) // 16}-bit EXL3 [{V}, {H}] -> {ids.size} rows gathered in {time.time() - t0:.0f}s", flush=True)
H_data = {"H": torch.zeros(H, H, dtype=torch.float32, device="meta"), "first_key": "draft.lm_head", "count": 0,
          "finalized": False, "num_total": 0, "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
qa = {"seed": a.seed, "K": a.bits, "devices": [0], "device_ratios": None, "apply_out_scales": True,
      "debug_dir": "/tmp/draft-head-exl3-debug", "mul1": True}
t1 = time.time()
with torch.inference_mode():
    _, perr, ot = quantize_exl3(w_draft, H_data, qa, False, None, False, None)
out = {f"draft.lm_head.{k}": v.cpu().contiguous() for k, v in ot.items() if isinstance(v, torch.Tensor)}
out["draft.d2t"] = torch.from_numpy(ids.astype(np.int32))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
save_file(out, a.out, metadata={"vocab": str(ids.size), "bits": str(a.bits), "ids": os.path.abspath(a.ids),
                                 "model": os.path.abspath(a.model), "seed": str(a.seed)})
mib = out["draft.lm_head.trellis"].numel() * 2 / 2**20
print(f"[draft-head] {a.bits}-bit EXL3 of [{ids.size}, {H}]: proxy err {perr:.6f}, {mib:.0f} MiB, {time.time() - t1:.0f}s -> {a.out}", flush=True)
