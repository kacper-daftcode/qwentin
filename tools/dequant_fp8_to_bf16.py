#!/usr/bin/env python3
"""Dequantize a block-scaled FP8 HF checkpoint (weight + weight_scale_inv, e.g. Qwen/Qwen3.8-27B-FP8) to bf16
safetensors, so exllamav3 can load it as an unquantized model -- this is how the FP8 reference row of the quality
yardstick is produced (docs/QUALITY.md): tools/exl3_diff.py --ref <original bf16> --model <this output>.

The expansion is exact up to the bf16 rounding of (fp8 code x fp32 block scale); quantization_config is dropped from
the copied config.json, tokenizer / template files are copied along. CPU only, ~30 GB of output for the 27B.

  python3 tools/dequant_fp8_to_bf16.py /models/Qwen3.8-27B-FP8 /models/Qwen3.8-27B-FP8-as-bf16

Vendored from analysis/dequant_fp8_to_bf16.py of the research project (kacper-daftcode/qwentin-internal)."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="block-scaled FP8 HF checkpoint dir")
    p.add_argument("dst", help="output dir (bf16 safetensors + config / tokenizer files)")
    args = p.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    shards = sorted(set(wm.values()))
    new_map = {}
    for shard in shards:
        out = {}
        with safe_open(os.path.join(src, shard), "pt") as f:
            keys = list(f.keys())
            for k in keys:
                if k.endswith("_scale_inv"):
                    continue
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn and (k + "_scale_inv") in keys:
                    s = f.get_tensor(k + "_scale_inv").to(torch.float32)
                    w = t.to(torch.float32)
                    bo, bi = w.shape[0] // s.shape[0], w.shape[1] // s.shape[1]
                    w = w * s.repeat_interleave(bo, 0).repeat_interleave(bi, 1)
                    t = w.to(torch.bfloat16)
                out[k] = t.contiguous()
                new_map[k] = shard
        save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
        print(shard, len(out), flush=True)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f, indent=1)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    cfg.get("text_config", {}).pop("quantization_config", None)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    for fn in os.listdir(src):
        if fn.endswith((".json", ".jinja", ".txt", ".md")) and fn not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(src, fn), dst)
    print("done", dst)


if __name__ == "__main__":
    main()
