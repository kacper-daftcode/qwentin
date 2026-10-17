#!/usr/bin/env python3
"""The FP8-gate table: KL / top-1 of every run in a directory vs the original bf16 model, and the paired difference to a
reference run on the same tokens with a 95 % block-bootstrap confidence interval (docs/QUALITY.md).

Reads <e2e-dir>/<tag>.json + <tag>.pt written by tools/qwentin_e2e.py (the engine) or tools/exl3_diff.py
--per-token-out (weights through exllamav3), all scored against the original bf16 model on the same held-out rows,
and prints per tag: KL, top-1, and -- when a reference run of the same (seq, seed) exists -- dKL [CI] and dtop-1 pt
[CI] against it, resampling 4096-token blocks (2000 draws). The gate rule: a stack is "FP8 quality" when its KL is
below or equal to the FP8 checkpoint's within the CI and the top-1 CI crosses 0 (or is positive), at 4K and at 32K.

  python3 tools/e2e_frontier.py --e2e-dir runs/e2e                                 # vs the FP8 reference run(s) found by name
  python3 tools/e2e_frontier.py --e2e-dir runs/e2e --vs fp8_32k --min-pos 4096     # long-context view, explicit reference
  python3 tools/e2e_frontier.py --e2e-dir runs/e2e --tags a b --vs a               # A/B of two runs (e.g. a lever on vs off)

Engine rows carry KL on the engine's fp32 logits; --kl16 scores them on fp16-rounded logits instead, like the
exllamav3-side candidates (exllamav3's lm_head returns fp16).

Vendored from analysis/e2e_frontier.py of the research project (kacper-daftcode/qwentin-internal) without its
memory-model columns (pool GB / tokens per card, which read the recipe YAMLs and analysis/kv_fakequant.py of that
repo); the statistics are unchanged. CPU only (torch for the tensors).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch


def paired(tok_c, tok_r, min_pos: int, block: int = 4096, reps: int = 2000):
    """Mean difference candidate - reference (KL, top-1) over tokens with pos >= min_pos, with a block bootstrap CI:
    the tokens are grouped into (row, pos // block) blocks, the blocks are resampled with replacement `reps` times
    (seeded generator, so the table is reproducible) and the 2.5 / 97.5 percentiles of the resampled means are the CI."""
    m = tok_c["pos"] >= min_pos
    dkl = (tok_c["kl"] - tok_r["kl"])[m].double()
    dag = (tok_c["agree"].double() - tok_r["agree"].double())[m]
    blk = (tok_c["row"].long() * 1_000_000 + tok_c["pos"].long() // block)[m]
    ids, inv = torch.unique(blk, return_inverse=True)
    nb = len(ids)
    s_kl = torch.zeros(nb, dtype=torch.float64).index_add_(0, inv, dkl)
    s_ag = torch.zeros(nb, dtype=torch.float64).index_add_(0, inv, dag)
    cnt = torch.zeros(nb, dtype=torch.float64).index_add_(0, inv, torch.ones_like(dkl))
    g = torch.Generator().manual_seed(0)
    pick = torch.randint(0, nb, (reps, nb), generator=g)
    bkl = s_kl[pick].sum(1) / cnt[pick].sum(1)
    bag = s_ag[pick].sum(1) / cnt[pick].sum(1)
    q = torch.tensor([0.025, 0.975], dtype=torch.float64)
    return (dkl.mean().item(), *bkl.quantile(q).tolist()), (dag.mean().item(), *bag.quantile(q).tolist())


def is_reference_run(d: dict, pattern: str) -> bool:
    """A reference run = a plain exl3_diff run (no simulation fields) whose candidate dir name matches --ref-pattern:
    the published FP8 reference rows scored the FP8 checkpoint dequantized to bf16 (tools/dequant_fp8_to_bf16.py)."""
    return pattern in os.path.basename(str(d.get("model", ""))) and not (d.get("sim_kv") or d.get("sim_embed") or d.get("sim_w"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e2e-dir", required=True, help="directory of <tag>.json + <tag>.pt run files")
    p.add_argument("--min-pos", type=int, default=0, help="score only positions >= this (long-context view)")
    p.add_argument("--tags", nargs="*", default=None, help="tags to print (default: every .json in --e2e-dir)")
    p.add_argument("--vs", default=None, help="pair with this tag (same seq/seed) instead of the reference run")
    p.add_argument("--ref-pattern", default="Qwen3.8-27B-bf16",
                   help="substring of the candidate dir name that marks a reference run (the FP8 checkpoint as bf16)")
    p.add_argument("--kl16", action="store_true", help="engine rows on fp16-rounded logits (kl16/agree16)")
    p.add_argument("--block", type=int, default=4096, help="bootstrap block length in tokens")
    p.add_argument("--reps", type=int, default=2000, help="bootstrap resamples")
    args = p.parse_args()
    E2E = args.e2e_dir

    def load_tok(tag):
        tok = torch.load(os.path.join(E2E, f"{tag}.pt"))
        if args.kl16 and "kl16" in tok:
            tok = dict(tok, kl=tok["kl16"], agree=tok["agree16"])
        return tok

    tags = args.tags or sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(E2E, "*.json")))
    ref_runs = {}
    for f in glob.glob(os.path.join(E2E, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if is_reference_run(r, args.ref_pattern):
            ref_runs[(r["seq"], r["seed"])] = os.path.basename(f)[:-5]
    vs = json.load(open(os.path.join(E2E, f"{args.vs}.json"))) if args.vs else None
    rows = []
    for tag in tags:
        with open(os.path.join(E2E, f"{tag}.json")) as fh:
            d = json.load(fh)
        tok = load_tok(tag)
        if vs:
            ref_tag = args.vs if (d["seq"], d["seed"]) == (vs["seq"], vs["seed"]) else ""
        else:
            ref_tag = ref_runs.get((d["seq"], d["seed"]), "")
        m = tok["pos"] >= args.min_pos
        if not m.any():
            continue
        kl, top1 = tok["kl"][m].mean().item(), tok["agree"][m].float().mean().item()
        dkl = dag = None
        if tag != ref_tag and os.path.exists(os.path.join(E2E, f"{ref_tag}.pt")):
            tok_r = load_tok(ref_tag)
            if tok_r["kl"].shape == tok["kl"].shape:
                dkl, dag = paired(tok, tok_r, args.min_pos, args.block, args.reps)
        rows.append((tag, d["seq"], kl, top1, dkl, dag, ref_tag))
    print(f"positions >= {args.min_pos}; deltas vs {args.vs or 'the reference run of the same (seq, seed)'} on the same "
          f"tokens, 95 % block-bootstrap CI ({args.block}-token blocks, {args.reps} draws)"
          f"{'; engine rows on fp16-rounded logits' if args.kl16 else ''}")
    print(f"{'tag':34s} {'seq':>6s} {'KL':>8s} {'dKL [CI]':>30s} {'top1':>7s} {'dtop1 pt [CI]':>24s}  vs")
    for tag, seq, kl, top1, dkl, dag, ref_tag in rows:
        s_dkl = f"{dkl[0]:+.5f} [{dkl[1]:+.5f},{dkl[2]:+.5f}]" if dkl else ""
        s_dag = f"{100 * dag[0]:+.2f} [{100 * dag[1]:+.2f},{100 * dag[2]:+.2f}]" if dag else ""
        print(f"{tag:34s} {seq:6d} {kl:8.5f} {s_dkl:>30s} {100 * top1:7.2f} {s_dag:>24s}  {ref_tag if dkl else ''}")


if __name__ == "__main__":
    main()
