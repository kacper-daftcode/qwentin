#!/usr/bin/env python3
"""Quality of a checkpoint vs the original bf16 model, through exllamav3, on the held-out code tokens (docs/QUALITY.md).

Both models run through exllamav3 module by module (one layer of each model resident at a time, so the 27B bf16
reference and the candidate fit on ONE 32 GB card). The candidate is any directory exllamav3 can load: an EXL3
quantization (the weights-only rows of the quality ladder), or an unquantized bf16/fp16 export -- the FP8 reference
row (0.0037 / 98.7 %) is `Qwen/Qwen3.8-27B-FP8` dequantized to bf16 containers with tools/dequant_fp8_to_bf16.py and
scored here as the candidate against the original `Qwen/Qwen3.8-27B`.

Reports: NLL / ppl of both, teacher-forced top-1 agreement (candidate argmax == reference argmax), KL(reference ||
candidate) mean / median / p90 / p99 and bucketed by reference confidence and by position, top-k agreement, the
per-layer hidden-state error, and with --per-token-out the per-token tensors that tools/e2e_frontier.py pairs across
runs (block-bootstrap CIs). The same per-token layout is written by tools/qwentin_e2e.py for the engine, so engine
rows and exllamav3 rows pair on the same tokens.

Vendored from analysis/exl3_diff.py of the research project (kacper-daftcode/qwentin-internal) without its fake-quant
simulation options (--sim-kv / --sim-w / --sim-a / --sim-embed, which need analysis/kv_fakequant.py); the numerics of
the paths kept here are unchanged. Needs exllamav3 (upstream commit 02aef45 or newer; `pip install -e` of a checkout,
or --exllamav3 /path/to/checkout) and a CUDA GPU.

  CUDA_VISIBLE_DEVICES=0 python3 tools/exl3_diff.py --ref /models/Qwen3.8-27B --model /models/Qwen3.8-27B-FP8-as-bf16 \
      --corpus /path/to/vllm/vllm --out runs/e2e/fp8_4k.json --per-token-out runs/e2e/fp8_4k.pt
  # 32K rows: --seq 32768 (6 rows); 128K / 256K rows need --chunk 16384 --states-on-cpu (see forward_row_chunked)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness_common import calibration_ids, corpus_fingerprint, load_corpus  # noqa: E402


def add_exllamav3_path(path: str | None) -> None:
    """--exllamav3 /path/to/checkout puts a source checkout first on sys.path; else the installed package is used."""
    if path:
        sys.path.insert(0, path)
    try:
        import exllamav3  # noqa: F401
    except ImportError as e:
        raise SystemExit("exllamav3 is not importable: pip install -e /path/to/exllamav3 (upstream commit 02aef45 or "
                         "newer) or pass --exllamav3 /path/to/checkout") from e


def load_module(config, module, device):
    config.stc.begin_deferred_load()
    module.load(device if not module.caps.get("prefer_cpu") else "cpu")
    config.stc.end_deferred_load()


def unload_module(config, module):
    from exllamav3.util.memory import free_mem

    module.unload()
    config.stc.close()
    free_mem()


def hidden_error_sums(y_q: torch.Tensor, y_r: torch.Tensor, acc=(0.0, 0.0, 0.0, 0.0), slice_tokens: int = 8192):
    """Sums behind the per-layer rfn / cos_err of a (1, n, hidden) candidate vs reference output: float64 in
    8192-token slices (a full 128K-token row is 5 GB per copy), accumulated onto `acc` as Python floats in slice
    order (so a row fed in chunks of a multiple of 8192 tokens sums exactly as the whole row)."""
    yq2, yr2 = y_q[0].reshape(-1, y_q.shape[-1]), y_r[0].reshape(-1, y_r.shape[-1])
    a2, b2, ab, d2 = acc
    for c0 in range(0, yq2.shape[0], slice_tokens):
        a = yq2[c0 : c0 + slice_tokens].double()
        b = yr2[c0 : c0 + slice_tokens].double()
        a2 += a.square().sum().item()
        b2 += b.square().sum().item()
        ab += (a * b).sum().item()
        d2 += (a - b).square().sum().item()
        del a, b
    return a2, b2, ab, d2


class _RowSlices:
    """Whole-row output of a Linear, served per chunk through its .forward (see forward_row_chunked)."""

    def __init__(self, full: torch.Tensor):
        self.full, self.c0, self.c1 = full, 0, 0

    def forward(self, x: torch.Tensor, params: dict, out_dtype=None) -> torch.Tensor:
        assert x.shape[1] == self.c1 - self.c0, (x.shape, self.c0, self.c1)
        return self.full[:, self.c0 : self.c1]


def _precompute_row_linears(m, x_in, seq: int, chunk: int, device):
    """The narrow projections are the one place where chunking changes numerics: cuBLAS picks an M-dependent
    algorithm for a skinny GEMM, so e.g. GatedDeltaNet's beta / decay projections (in_proj_b / in_proj_a, 5120 -> 48,
    fp32 out) from a 2048-row call differ from the rows of a 4096-row call at the fp32 rounding level, and attention's
    k_proj / v_proj (5120 -> 1024, fp16) at M <= 4096 differ from M >= 8192 in 21 % of values by one fp16 ulp.
    Everything else (the wide GEMMs incl. the MLP, conv, fla chunk kernel, attention kernel) is bit-identical chunked
    vs whole. So in chunk mode these projections run over the whole row, exactly as the whole-row path does, and are
    served per chunk. Returns the overrides to undo after the row."""
    from exllamav3.modules import Attention, GatedDeltaNet

    g = getattr(m, "attn", None)
    if getattr(m, "attn_norm", None) is None:
        return []
    if isinstance(g, GatedDeltaNet) and g.b_proj is not None and g.a_proj is not None:
        lins = (g.b_proj, g.a_proj)
    elif isinstance(g, Attention) and g.num_kv_heads and g.k_proj is not None and g.v_proj is not None:
        lins = (g.k_proj, g.v_proj)
    else:
        return []
    xn = None
    for c0 in range(0, seq, chunk):
        c1 = min(c0 + chunk, seq)
        yn = m.attn_norm.forward(x_in[:, c0:c1].to(device, non_blocking=False), {}, out_dtype=torch.half)
        if xn is None:
            xn = torch.empty((1, seq, yn.shape[-1]), dtype=yn.dtype, device=device)
        xn[:, c0:c1] = yn
        del yn
    overrides = []
    for lin in lins:
        rs = _RowSlices(lin.forward(xn, {}))
        overrides.append((lin, lin.__dict__.get("forward"), rs))
        lin.forward = rs.forward
    del xn
    return overrides


def forward_row_chunked(side_r, side_q, ids_row: torch.Tensor, chunk: int, device, state_dev):
    """--chunk: run the reference and the candidate module over ONE row in sequence chunks of `chunk` tokens, each
    module's own state carried between chunks exactly as exllamav3 carries it at generation time:
      * attention keeps the K/V of the earlier chunks in a per-layer paged fp16 cache (allocated with the layer,
        freed with it); the kernel is the same _paged_attn_prefill_kernel the whole-row path runs on contiguous K/V
        (same tiles from kv position 0, same online-softmax order; num_splits stays 1 for chunks >= 4096, so the
        chunked output is bit-identical to the whole-row one);
      * GatedDeltaNet carries the fp32 recurrent state and the 3-token conv window: fla's chunk kernel starts from
        the carried fp32 state, so a chunk boundary on its 64-token grid changes nothing;
      * the narrow projections (GatedDeltaNet's beta / decay, attention's K / V) run over the whole row and are served
        per chunk (_precompute_row_linears), because cuBLAS's algorithm for a skinny GEMM depends on M.
    side_* = (model, module, cache, row params, row input state on `state_dev` or the input ids). Returns the two row
    outputs on `state_dev` and the (a2, b2, ab, d2) sums of hidden_error_sums."""
    from exllamav3.cache.recurrent_util import advance_recurrent_states

    seq = ids_row.shape[1]
    seq_pad = side_r[2].max_num_tokens
    states = [side[2].get_new_state() for side in (side_r, side_q)]  # fresh (zeroed) recurrent slot per row
    overrides = [_precompute_row_linears(side[1], side[4], seq, chunk, device) for side in (side_r, side_q)]
    y = [None, None]
    acc = (0.0, 0.0, 0.0, 0.0)
    for c0 in range(0, seq, chunk):
        c1 = min(c0 + chunk, seq)
        chunk_ids = ids_row[:, c0:c1]
        yc = []
        for i, (model, m, cache, prow, x_in) in enumerate((side_r, side_q)):
            for _, _, rs in overrides[i]:
                rs.c0, rs.c1 = c0, c1
            p = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, seq_pad), "past_len": c0,
                 "recurrent_states": [states[i]]}
            model.prepare_inputs(chunk_ids, p)  # block table, cache_seqlens = c0, rope position = c0, slot tensor
            x = m.prepare_for_device(x_in[:, c0:c1].to(device, non_blocking=False), p)
            out = m.forward(x, p)
            advance_recurrent_states(chunk_ids, p, model)  # state.position += c1 - c0
            if y[i] is None:
                y[i] = torch.empty((1, seq, out.shape[-1]), dtype=out.dtype, device=state_dev)
            y[i][:, c0:c1].copy_(out)
            yc.append(out)
            del x, p
        acc = hidden_error_sums(yc[1], yc[0], acc)
        del yc, out
    for st in states:
        st.free()
    for side in overrides:
        for lin, orig, _ in side:
            if orig is None:
                del lin.forward
            else:
                lin.forward = orig
    return y[0], y[1], acc


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="the original bf16 HF checkpoint dir (Qwen/Qwen3.8-27B)")
    p.add_argument("--model", required=True, help="candidate dir: an EXL3 quantization, or a bf16/fp16 HF dir")
    p.add_argument("--corpus", required=True, help="held-out code tree (docs/QUALITY.md: a vLLM checkout's vllm/ dir)")
    p.add_argument("--tokens", type=int, default=200_000)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--seed", type=int, default=12345, help="held-out shuffle seed (the published rows: 12345)")
    p.add_argument("--tokenizer", default=None, help="HF dir for the tokenizer (default: --ref)")
    p.add_argument("--exllamav3", default=None, help="path of an exllamav3 source checkout to import from")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--states-on-cpu", action="store_true", help="park the residual states in host RAM between layers")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--per-token-out", default=None, help="save per-token KL / top-1 agreement / position tensors (.pt)")
    p.add_argument("--chunk", type=int, default=0,
                   help="run every layer over each row in sequence chunks of this many tokens with the module's own "
                        "state carried between chunks (attention: per-layer paged fp16 KV cache; GatedDeltaNet: "
                        "recurrent + conv state), so 128K / 256K rows fit a 32 GB card; a multiple of 256, >= 4096 "
                        "keeps the attention kernel on its unsplit path (bit-identical to whole rows); 0 = whole rows")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.chunk:
        assert args.chunk % 256 == 0, "--chunk must be a multiple of the 256-token cache page"
    add_exllamav3_path(args.exllamav3)
    from exllamav3 import Config, Model
    from exllamav3.util.measures import compute_kl_div
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ref)
    corpus = load_corpus(args.corpus)
    ids = torch.tensor(calibration_ids(tok, corpus, args.tokens, seed=args.seed), dtype=torch.long)
    n_rows = ids.numel() // args.seq
    ids = ids[: n_rows * args.seq].view(n_rows, args.seq)
    print(f"held-out: {n_rows} x {args.seq} tokens from {args.corpus}", flush=True)

    cfg_r = Config.from_directory(args.ref)
    cfg_q = Config.from_directory(args.model)
    model_r = Model.from_config(cfg_r)
    model_q = Model.from_config(cfg_q)
    vocab = model_r.config.vocab_size
    assert len(model_r.modules) == len(model_q.modules), "module lists differ"
    print(f"{len(model_r.modules)} modules, vocab {vocab}", flush=True)

    cache_r = cache_q = None
    if args.chunk:
        # One cache per model, tied to the model: a layer's K/V pages (fp16, sized to the row) and GatedDeltaNet state
        # slot are allocated when the layer loads and freed when it unloads, so only the resident layer's state exists
        from exllamav3 import Cache

        seq_pad = -(-args.seq // 256) * 256
        cache_r = Cache(model_r, max_num_tokens=seq_pad, max_batch_size=1)
        cache_q = Cache(model_q, max_num_tokens=seq_pad, max_batch_size=1)
        print(f"chunk mode: {args.chunk}-token chunks, {-(-args.seq // args.chunk)} per row, per-layer KV cache of "
              f"{seq_pad} tokens ({seq_pad * 4 * 256 * 2 * 2 / 2**30:.2f} GiB fp16 per model)", flush=True)

    state_dev = torch.device("cpu") if args.states_on_cpu else device
    params_r = [{} for _ in range(n_rows)]
    params_q = [{} for _ in range(n_rows)]
    states_r, states_q = [], []
    for j in range(n_rows):
        row = ids[j : j + 1].clone()
        states_r.append(model_r.prepare_inputs(row.clone(), params_r[j]))
        states_q.append(model_q.prepare_inputs(row.clone(), params_q[j]))

    per_layer = []
    result = {
        "ref": args.ref, "model": args.model, "sim_kv": None, "sim_w": None, "sim_embed": None, "sim_kv_layer": [],
        "tokens": int(n_rows * args.seq), "seq": args.seq, "seed": args.seed,
        "corpus": corpus_fingerprint(args.corpus, corpus),
    }
    if args.chunk:
        result["chunk"] = args.chunk
    nll = [0.0, 0.0]
    n_tok = 0
    top1_agree = 0
    topk_agree = [0] * args.topk
    kl_sum = 0.0
    kl_rev_sum = 0.0
    kl_toks, conf_toks, agree_toks = [], [], []
    klh_toks, pos_toks, row_toks = [], [], []
    kl_head_sum = 0.0
    head_agree = 0
    label_top1 = [0, 0]
    t_start = time.time()

    for idx, (m_r, m_q) in enumerate(zip(model_r.modules, model_q.modules)):
        logits_layer = idx == len(model_r.modules) - 1
        t0 = time.time()
        load_module(cfg_r, m_r, device)
        load_module(cfg_q, m_q, device)
        rfn_sum, cos_sum = 0.0, 0.0
        for j in range(n_rows):
            pr, pq = params_r[j], params_q[j]
            pr["dev_cache"] = None
            pq["dev_cache"] = None
            if logits_layer:
                # lm_head: chunk the positions (full-vocab fp32 logits of a 4096 row are 4 GB per model)
                xr = m_r.prepare_for_device(states_r[j].to(device), pr)
                xq = m_q.prepare_for_device(states_q[j].to(device), pq)
                xh = m_q.prepare_for_device(states_r[j].to(device), pq)  # ref hidden states -> candidate head
                for s in range(0, xr.shape[1] - 1, 512):
                    e = min(s + 512, xr.shape[1] - 1)
                    lr = m_r.forward(xr[:, s:e].contiguous(), pr)[0, :, :vocab]
                    lq = m_q.forward(xq[:, s:e].contiguous(), pq)[0, :, :vocab]
                    tgt = ids[j, s + 1 : e + 1].to(device)
                    for i, l in enumerate((lr, lq)):
                        lp = torch.log_softmax(l.float(), -1)
                        nll[i] += -lp.gather(-1, tgt[:, None]).sum().item()
                        del lp
                        label_top1[i] += (l.argmax(-1) == tgt).sum().item()
                    n_tok += tgt.numel()
                    tk_r = lr.topk(args.topk, dim=-1).indices
                    tk_q = lq.topk(args.topk, dim=-1).indices
                    agree = tk_r[:, 0] == tk_q[:, 0]
                    top1_agree += agree.sum().item()
                    lh = m_q.forward(xh[:, s:e].contiguous(), pq)[0, :, :vocab]
                    klh = compute_kl_div(lh, lr, vocab)
                    kl_head_sum += klh.sum().item()
                    head_agree += (lh.argmax(-1) == tk_r[:, 0]).sum().item()
                    klh_toks.append(klh.flatten().float().cpu())
                    pos_toks.append(torch.arange(s, e, dtype=torch.int32))
                    row_toks.append(torch.full((e - s,), j, dtype=torch.int16))
                    del lh, klh
                    for t in range(args.topk):
                        topk_agree[t] += (tk_r[:, : t + 1] == tk_q[:, : t + 1]).all(-1).sum().item()
                    kl = compute_kl_div(lq, lr, vocab)  # KL(softmax(ref) || softmax(q)) per token
                    kl_rev = compute_kl_div(lr, lq, vocab)
                    kl_sum += kl.sum().item()
                    kl_rev_sum += kl_rev.sum().item()
                    kl_toks.append(kl.flatten().float().cpu())
                    agree_toks.append(agree.cpu())
                    cf = lr.float()
                    conf_toks.append((cf.max(-1).values - cf.logsumexp(-1)).exp().cpu())
                    del lr, lq, kl, kl_rev, cf, tk_r, tk_q
                del xr, xq, xh
                continue
            if args.chunk:
                y_r, y_q, (a2, b2, ab, d2) = forward_row_chunked(
                    (model_r, m_r, cache_r, pr, states_r[j]), (model_q, m_q, cache_q, pq, states_q[j]),
                    ids[j : j + 1], args.chunk, device, state_dev)
            else:
                x = m_r.prepare_for_device(states_r[j].to(device, non_blocking=False), pr)
                y_r = m_r.forward(x, pr).clone()
                x = m_q.prepare_for_device(states_q[j].to(device, non_blocking=False), pq)
                y_q = m_q.forward(x, pq).clone()
                a2, b2, ab, d2 = hidden_error_sums(y_q, y_r)
                del x
            cos_sum += 1.0 - ab / max(math.sqrt(a2 * b2), 1e-8)
            rfn_sum += math.sqrt(d2 / b2) if b2 > 0 else 0.0
            states_r[j] = y_r.to(state_dev)
            states_q[j] = y_q.to(state_dev)
            del y_r, y_q
        unload_module(cfg_r, m_r)
        unload_module(cfg_q, m_q)
        if not logits_layer:
            per_layer.append({"module": m_q.key, "rfn": rfn_sum / n_rows, "cos_err": cos_sum / n_rows})
            print(f"[{idx:3d}] {m_q.key:48s} rfn {rfn_sum / n_rows:.5f}  cos_err {cos_sum / n_rows:.6f}  ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"[{idx:3d}] {m_q.key} (logits) ({time.time() - t0:.0f}s)", flush=True)

    kl_all = torch.cat(kl_toks)
    conf_all = torch.cat(conf_toks)
    agree_all = torch.cat(agree_toks)
    klh_all = torch.cat(klh_toks)
    pos_all = torch.cat(pos_toks)
    ppl = [math.exp(nll[i] / n_tok) for i in (0, 1)]
    result.update({
        "ref_nll": nll[0] / n_tok, "ref_ppl": ppl[0], "ref_label_top1": label_top1[0] / n_tok,
        "q_nll": nll[1] / n_tok, "q_ppl": ppl[1], "q_label_top1": label_top1[1] / n_tok,
        "ppl_ratio": ppl[1] / ppl[0],
        "top1_agree": top1_agree / n_tok,
        "topk_agree": {t + 1: topk_agree[t] / n_tok for t in range(args.topk)},
        "kl_mean": kl_sum / n_tok, "kl_rev_mean": kl_rev_sum / n_tok,
        "kl_median": kl_all.median().item(), "kl_p90": kl_all.quantile(0.9).item(), "kl_p99": kl_all.quantile(0.99).item(),
        "kl_head_only_mean": kl_head_sum / n_tok, "top1_head_only": head_agree / n_tok,
        "by_position": {},
        "by_ref_confidence": {},
        "per_layer": per_layer,
        "seconds": time.time() - t_start,
    })
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)]:
        mask = (conf_all >= lo) & (conf_all < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        result["by_ref_confidence"][f"[{lo:.2f},{min(hi, 1.0):.2f})"] = {
            "frac_tokens": n / conf_all.numel(), "kl_mean": kl_all[mask].mean().item(),
            "top1_agree": agree_all[mask].float().mean().item(),
        }
    edges = [0, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 30]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pos_all >= lo) & (pos_all < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        result["by_position"][f"[{lo},{hi})"] = {
            "n": n, "kl_mean": kl_all[mask].mean().item(), "top1_agree": agree_all[mask].float().mean().item(),
            "kl_head_only_mean": klh_all[mask].mean().item(),
        }
    if args.per_token_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_token_out)), exist_ok=True)
        torch.save({"kl": kl_all, "kl_head_only": klh_all, "agree": agree_all, "conf": conf_all.half(), "pos": pos_all,
                    "row": torch.cat(row_toks), "seq": args.seq, "n_rows": n_rows}, args.per_token_out)
    print(json.dumps({k: v for k, v in result.items() if k != "per_layer"}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
