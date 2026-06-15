#!/usr/bin/env python3
"""Validate the engine's native MTP head against the PyTorch reference.

This drives the CUDA engine (libforward_qwen.so) with an MTP-enabled TQF and
compares its MTP top-k against the VALIDATED reference forward in
``tools/mtp_accept.py`` for the SAME prompt/positions.

Two checks:
  (A) MTP-kernel correctness (injected hidden): feed the reference's PRE-final-norm
      hidden + on-policy token into the engine via ``qwn_mtp_step_hidden`` and compare
      the engine MTP top-k against the reference MTP top-k slot-by-slot. Same inputs,
      same weights -> top-1 should agree on most positions (the only difference is the
      engine's E2M3/FP8 weight quant vs the reference's BF16).
  (B) End-to-end on-policy depth-1: drive ``qwn_decode`` + ``qwn_mtp_step`` (engine's
      own hidden) and measure how often the MTP top-1 equals the engine's own next-next
      greedy token (the realistic spec-decode accept, ~54% per the reference).

VALIDATION GATE (per docs/MTP_SPECDEC_PLAN.md):
  - engine MTP top-1 matches the reference MTP top-1 on MOST positions, and
  - on-policy depth-1 top-1 ~54% vs dense.

Run (GPU 6 only):
    CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python3 -u tools/mtp_engine_smoke.py \
        --model-dir /workspace/models/Qwen3.5-0.8B \
        --tqf /workspace/models/Qwen3.5-0.8B/qwen3_5-0_8b-e2m3-mtp.tqf
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_accept as ref  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.convert import float_to_e4m3, float_to_e2m3  # noqa: E402

GPU = "cuda:0"  # CUDA_VISIBLE_DEVICES=6 maps GPU 6 -> cuda:0

FALLBACK_TEXT = ref.FALLBACK_TEXT


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------- engine ABI
def load_lib(path: str):
    lib = ctypes.CDLL(path)
    lib.qwn_init.argtypes = [ctypes.c_char_p]
    lib.qwn_init.restype = ctypes.c_int
    lib.qwn_free.argtypes = []
    lib.qwn_hidden_size.restype = ctypes.c_int
    lib.qwn_vocab_size.restype = ctypes.c_int
    lib.qwn_num_layers.restype = ctypes.c_int
    lib.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.qwn_decode.restype = ctypes.c_int
    lib.qwn_reset_state.argtypes = []
    lib.qwn_reset_state.restype = ctypes.c_int
    lib.qwn_has_mtp.restype = ctypes.c_int
    lib.qwn_mtp_reset.restype = ctypes.c_int
    lib.qwn_mtp_step.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.qwn_mtp_step.restype = ctypes.c_int
    lib.qwn_mtp_step_hidden.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.qwn_mtp_step_hidden.restype = ctypes.c_int
    lib.qwn_copy_last_mtp_logits.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_copy_last_mtp_logits.restype = ctypes.c_int
    lib.qwn_copy_last_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.qwn_copy_last_norm.restype = ctypes.c_int
    return lib


# --------------------------------------------------------------- reference pass
def _e4m3_decode(codes: np.ndarray) -> np.ndarray:
    """Decode uint8 E4M3 codes to float32 (matches the engine's tq_e4m3_to_float)."""
    codes = codes.astype(np.uint8)
    sign = ((codes >> 7) & 1).astype(np.float32)
    exp = ((codes >> 3) & 0x0F).astype(np.int32)
    man = (codes & 0x07).astype(np.float32)
    out = np.where(exp == 0, np.ldexp(man / 8.0, -6), np.ldexp(1.0 + man / 8.0, exp - 7)).astype(np.float32)
    out = np.where((codes & 0x7F) == 0, 0.0, out)
    return np.where(sign > 0, -out, out).astype(np.float32)


@torch.no_grad()
def conv_dequant_weight(w: torch.Tensor, fmt: str) -> torch.Tensor:
    """128x128 block-scaled round-trip using the CONVERTER'S EXACT quantization
    (tools.convert.float_to_e4m3 / float_to_e2m3) + the engine's decode. Builds a
    reference head with the SAME quantized weights the engine actually uses, so the
    comparison isolates the MTP forward (kernel) from weight-quant quality."""
    from tools.convert import _e2m3_tables
    wf = w.detach().float().cpu().numpy()
    m, k = wf.shape
    out = wf.copy()
    target = 448.0 if fmt == "fp8" else 7.5
    abs_vals, code_vals, _ = _e2m3_tables()
    code_to_mag = np.zeros(32, dtype=np.float32)   # code (e<<3|m) -> magnitude
    code_to_mag[code_vals] = abs_vals
    for r0 in range(0, m, 128):
        for c0 in range(0, k, 128):
            blk = wf[r0:r0 + 128, c0:c0 + 128]
            amax = float(np.abs(blk).max())
            if amax <= 0:
                continue
            if fmt == "fp8":
                s = amax / target              # float policy (matches converter FP8)
                out[r0:r0 + 128, c0:c0 + 128] = _e4m3_decode(float_to_e4m3(blk / s)) * s
            else:                               # e2m3, pow2 scale (matches converter qmma-e2m3)
                s = float(2.0 ** np.ceil(np.log2(max(amax / target, 1e-30))))
                codes = float_to_e2m3(blk / s)
                dec = code_to_mag[codes & 0x1F] * np.where((codes >> 5) & 1, -1.0, 1.0)
                out[r0:r0 + 128, c0:c0 + 128] = dec.astype(np.float32) * s
    return torch.from_numpy(out).to(torch.bfloat16)


@torch.no_grad()
def build_head_dtype(weights: dict, template_layer, RMSNorm, dtype):
    """Like ref.build_head but at an arbitrary dtype. The engine computes the MTP
    forward in FP32 (dequantized weights, fp32 activations), so an FP32 reference is
    the apples-to-apples comparison; the canonical BF16 head differs only by BF16
    arithmetic, which flips near-tie top-1s on the flat MTP distribution."""
    import copy
    H = ref.H

    def rmsnorm(name):
        m = RMSNorm(H, eps=1e-6)
        m.weight.data = weights[name].to(torch.float32).cpu()
        return m.to(GPU, dtype).eval()

    fc = torch.nn.Linear(2 * H, H, bias=False)
    fc.weight.data = weights["mtp.fc.weight"].to(torch.float32).cpu()
    fc = fc.to(GPU, dtype).eval()

    layer = copy.deepcopy(template_layer)
    pref = "mtp.layers.0."
    layer_sd = {k[len(pref):]: v for k, v in weights.items() if k.startswith(pref)}
    missing, unexpected = layer.load_state_dict(layer_sd, strict=False)
    assert not missing and not unexpected, (list(missing), list(unexpected))
    layer = layer.to(GPU, dtype).eval()
    layer.self_attn.config._attn_implementation = "eager"
    return {
        "pe": rmsnorm("mtp.pre_fc_norm_embedding.weight"),
        "ph": rmsnorm("mtp.pre_fc_norm_hidden.weight"),
        "norm": rmsnorm("mtp.norm.weight"),
        "fc": fc,
        "layer": layer,
    }


@torch.no_grad()
def make_fp8_weights(w_bf16: dict, fmt: str = "fp8") -> dict:
    # "bf16" = full precision (MTP weights are stored unquantized in the TQF), so the
    # engine-matching reference is the plain BF16 head at FP32 compute.
    if fmt == "bf16":
        return dict(w_bf16)
    quant_suffix = ("fc.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight",
                    "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    out = {}
    for kk, v in w_bf16.items():
        if v.ndim == 2 and any(kk.endswith(s) for s in quant_suffix):
            out[kk] = conv_dequant_weight(v, fmt)
        else:
            out[kk] = v
    return out


@torch.no_grad()
def build_reference(model_dir: str, ids: list[int], topk: int, quant_fmt: str = "fp8",
                    ref_device: str = "cuda"):
    """Load the model, capture pre-final-norm hiddens + dense greedy, build the
    native-BF16 MTP head, and compute the reference MTP top-k (on-policy depth-1).

    The big main model loads on ``ref_device`` (use 'cpu' for 27B, which does not fit
    on the GPU); the small MTP head + captured tensors are moved to the GPU so the MTP
    forward stays fast. Returns numpy arrays so the model can be freed before the
    engine runs (avoids holding two large allocations at once)."""
    import copy
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5RMSNorm,
    )

    n = len(ids)
    log(f"loading reference model {model_dir} (BF16, eager, device={ref_device}) ...")
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True
    ).to(ref_device).eval()
    log(f"reference model loaded in {time.time() - t0:.1f}s")

    tm = model.model.language_model
    lm_head = model.lm_head
    final_norm = tm.norm
    rotary = tm.rotary_emb
    layer_types = model.config.text_config.layer_types
    full_idx = layer_types.index("full_attention")

    cap = {}

    def pre_hook(_mod, args):
        cap["h"] = args[0].detach()

    handle = final_norm.register_forward_pre_hook(pre_hook)
    input_ids = torch.tensor([ids], dtype=torch.long, device=ref_device)
    log("running main-model forward (slow on CPU for 27B) ...")
    tf = time.time()
    out = model(input_ids=input_ids, use_cache=False)
    handle.remove()
    log(f"main forward done in {time.time() - tf:.1f}s")

    logits = out.logits[0]
    h_pre = cap["h"][0]                       # [n,H] bf16 (pre-final-norm)
    dense_pred = logits.argmax(-1)            # [n]

    # SANITY: argmax(lm_head(final_norm(h_pre))) == model logits argmax (~100%).
    fn = final_norm(h_pre)
    rec = lm_head(fn).argmax(-1)
    sanity = (rec == dense_pred).float().mean().item() * 100
    ref_final_hidden = fn.float().cpu().numpy()        # [n,H] post-final-norm (lm_head input)

    # Move the MTP-relevant tensors to the GPU for a fast reference MTP forward.
    W_emb = tm.embed_tokens.weight.detach().to(GPU)
    W_lm = lm_head.weight.detach().to(GPU)
    h_pre = h_pre.to(GPU)
    dense_pred = dense_pred.to(GPU)
    rotary_g = copy.deepcopy(rotary).to(GPU)
    template_layer = copy.deepcopy(tm.layers[full_idx]).cpu()

    w_bf16 = ref.load_mtp_bf16()
    head = ref.build_head(w_bf16, template_layer, Qwen3_5RMSNorm)
    # Same quant as the engine (converter float_to_e4m3 / float_to_e2m3) + FP32 compute
    # -> matches the engine's effective weights AND precision, isolating the kernel.
    head_fp8 = build_head_dtype(make_fp8_weights(w_bf16, quant_fmt), template_layer, Qwen3_5RMSNorm, torch.float32)

    # Reference MTP forward (on-policy depth-1) over slots 0..n-2 in one causal pass.
    # slot t: hidden = h_pre[t], token = dense_pred[t] (model's greedy at pos t),
    # RoPE position = t (relative; a global shift is RoPE-invariant and matches the
    # engine's cache slot t). target = dense_pred[t+1] (the dense greedy at pos t+1).
    L = n - 1
    base_t = torch.arange(L, device=GPU)
    post = ref.mtp_step(head, rotary_g, W_emb, h_pre[0:L], dense_pred[base_t], base_t)
    ref_logits = (post @ W_lm.t()).float()    # [L,V]
    ref_topk = ref_logits.topk(topk, dim=-1).indices.cpu().numpy()   # [L,topk]
    ref_top2 = ref_logits.topk(2, dim=-1).values
    ref_margin = (ref_top2[:, 0] - ref_top2[:, 1]).cpu().numpy()     # [L] top1-top2 logit gap

    # FP8-quantized-weight reference: same kernel/math as BF16 ref, weights FP8 like
    # the engine. corr(engine, fp8_ref) ~ 1 proves the engine kernel is correct;
    # corr(fp8_ref, bf16_ref) shows the inherent FP8 quant effect.
    post8 = ref.mtp_step(head_fp8, rotary_g, W_emb.float(), h_pre[0:L].float(),
                         dense_pred[base_t], base_t)
    fp8_logits = (post8 @ W_lm.float().t()).float()   # [L,V]
    fp8_topk = fp8_logits.topk(topk, dim=-1).indices.cpu().numpy()

    op_tgt = dense_pred[(base_t + 1).clamp(max=n - 1)]
    ref_op_d1 = (ref_topk[:, 0] == op_tgt.cpu().numpy())[: L - 1].mean() * 100

    result = dict(
        h_pre=h_pre[0:L].float().cpu().numpy(),       # [L,H] f32
        dense_pred=dense_pred.cpu().numpy(),          # [n]
        ref_topk=ref_topk,                            # [L,topk]
        ref_logits=ref_logits.cpu().numpy(),          # [L,V] f32
        fp8_topk=fp8_topk,                            # [L,topk]
        fp8_logits=fp8_logits.cpu().numpy(),          # [L,V] f32
        ref_margin=ref_margin,                        # [L]
        ref_final_hidden=ref_final_hidden,            # [n,H] post-final-norm
        ref_op_d1=float(ref_op_d1),
        sanity=float(sanity),
        n=n,
        L=L,
    )
    del model, out, logits, h_pre, rec, post, ref_logits, post8, fp8_logits
    del head, head_fp8, rotary_g, template_layer, W_emb, W_lm, dense_pred, fn
    torch.cuda.empty_cache()
    return result


def mtp_step_hidden(lib, hidden_row: np.ndarray, token_id: int, pos: int, k: int):
    H = hidden_row.shape[0]
    hbuf = (ctypes.c_float * H)(*hidden_row.tolist())
    ids_buf = (ctypes.c_int * k)()
    vals_buf = (ctypes.c_float * k)()
    top1 = lib.qwn_mtp_step_hidden(hbuf, int(token_id), int(pos), ids_buf, vals_buf, k)
    return top1, list(ids_buf), list(vals_buf)


def mtp_step(lib, token_id: int, pos: int, k: int):
    ids_buf = (ctypes.c_int * k)()
    vals_buf = (ctypes.c_float * k)()
    top1 = lib.qwn_mtp_step(int(token_id), int(pos), ids_buf, vals_buf, k)
    return top1, list(ids_buf), list(vals_buf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate engine MTP head vs mtp_accept.py")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.5-0.8B")
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.5-0.8B/qwen3_5-0_8b-e2m3-mtp.tqf")
    ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
    ap.add_argument("--max-tokens", type=int, default=192, help="prompt tokens / MTP slots to test")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--prompt", default=None, help="override prompt text")
    ap.add_argument("--ref-quant", choices=("fp8", "e2m3", "bf16"), default="bf16",
                    help="weight quant for the engine-matching reference head; use 'bf16' "
                         "(default) now that MTP projections are stored full-precision")
    ap.add_argument("--ref-device", choices=("cuda", "cpu"), default="cuda",
                    help="device for the big reference model ('cpu' for 27B, which does not fit on GPU)")
    ap.add_argument("--gate-kernel-corr", type=float, default=0.99,
                    help="min engine-vs-matched-quant-reference MTP logit correlation to PASS (kernel fidelity)")
    ap.add_argument("--gate-op-d1-tol", type=float, default=6.0,
                    help="max |engine - reference| on-policy depth-1 top-1 %% gap to PASS")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU required (run with CUDA_VISIBLE_DEVICES=6)"
    log("device:", torch.cuda.get_device_name(0))

    # Point the reference module at the requested model.
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    tc = getattr(cfg, "text_config", cfg)
    ref.H = int(tc.hidden_size)
    ref.MODEL_DIR = args.model_dir
    ref.GPU = GPU

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    text = args.prompt
    if text is None:
        text = open(ref.BIGTEXT).read() if os.path.exists(ref.BIGTEXT) else FALLBACK_TEXT
    ids = tok(text, add_special_tokens=False).input_ids[: args.max_tokens]
    n = len(ids)
    log(f"prompt tokens: {n}  topk={args.topk}")

    rdata = build_reference(args.model_dir, ids, args.topk, args.ref_quant, args.ref_device)
    h_pre = rdata["h_pre"]            # [L,H] f32
    dense_pred = rdata["dense_pred"]  # [n]
    ref_topk = rdata["ref_topk"]      # [L,topk]
    L = rdata["L"]
    log(f"reference: hidden-extraction sanity={rdata['sanity']:.2f}%  "
        f"on-policy depth-1 top1={rdata['ref_op_d1']:.2f}%")

    # ---- engine -----------------------------------------------------------------
    lib = load_lib(args.lib)
    ret = lib.qwn_init(args.tqf.encode())
    if ret != 0:
        raise RuntimeError(f"qwn_init failed: {ret}")
    try:
        H = int(lib.qwn_hidden_size())
        V = int(lib.qwn_vocab_size())
        if H != h_pre.shape[1]:
            raise RuntimeError(f"engine H={H} != reference H={h_pre.shape[1]}")
        if not lib.qwn_has_mtp():
            raise RuntimeError("engine reports no MTP section (regenerate the TQF with TQ_EMIT_MTP=1)")
        log(f"engine loaded: H={H} V={V} has_mtp={bool(lib.qwn_has_mtp())}")

        K = args.topk

        # (A) MTP-kernel correctness: inject the reference hidden + on-policy token.
        if lib.qwn_mtp_reset() != 0:
            raise RuntimeError("qwn_mtp_reset failed")
        ref_logits = rdata["ref_logits"]   # [L,V]
        fp8_logits = rdata["fp8_logits"]   # [L,V]
        fp8_topk = rdata["fp8_topk"]       # [L,topk]
        ref_margin = rdata["ref_margin"]   # [L]
        eng_top1_inj = np.empty(L, dtype=np.int64)
        eng_topk_inj = np.empty((L, K), dtype=np.int64)
        logit_buf = (ctypes.c_float * V)()
        corrs = np.empty(L, dtype=np.float64)        # engine vs BF16 reference
        corrs8 = np.empty(L, dtype=np.float64)       # engine vs FP8 reference
        for t in range(L):
            top1, ids_k, _ = mtp_step_hidden(lib, h_pre[t], int(dense_pred[t]), t, K)
            if top1 < 0:
                raise RuntimeError(f"qwn_mtp_step_hidden failed at slot {t}: {top1}")
            eng_top1_inj[t] = top1
            eng_topk_inj[t] = ids_k
            ncopy = lib.qwn_copy_last_mtp_logits(logit_buf, V)
            eng_log = np.ctypeslib.as_array(logit_buf)[:ncopy]
            corrs[t] = np.corrcoef(eng_log, ref_logits[t][:ncopy])[0, 1]
            corrs8[t] = np.corrcoef(eng_log, fp8_logits[t][:ncopy])[0, 1]

        ref_top1 = ref_topk[:, 0]
        fp8_top1 = fp8_topk[:, 0]
        top1_agree = (eng_top1_inj == ref_top1).mean() * 100
        top1_agree_fp8 = (eng_top1_inj == fp8_top1).mean() * 100
        mean_logit_corr_fp8 = float(np.nanmean(corrs8)) * 100
        fp8_vs_bf16_corr = float(np.nanmean([
            np.corrcoef(fp8_logits[t], ref_logits[t])[0, 1] for t in range(L)
        ])) * 100
        # engine top-1 covered by reference top-k, and top-k set overlap (mean Jaccard).
        in_ref_topk = np.array([eng_top1_inj[t] in set(ref_topk[t]) for t in range(L)]).mean() * 100
        jacc = np.mean([
            len(set(eng_topk_inj[t]) & set(ref_topk[t])) / len(set(eng_topk_inj[t]) | set(ref_topk[t]))
            for t in range(L)
        ]) * 100
        mean_logit_corr = float(np.nanmean(corrs)) * 100
        # Stratify top-1 agreement by the reference's top1-top2 margin: a correct
        # kernel agrees ~always on CONFIDENT (high-margin) slots; disagreements
        # concentrate on near-tie (flat) slots that any rounding difference flips.
        med = float(np.median(ref_margin))
        conf = ref_margin >= med
        flat = ~conf
        agree_conf = (eng_top1_inj[conf] == ref_top1[conf]).mean() * 100 if conf.any() else 0.0
        agree_flat = (eng_top1_inj[flat] == ref_top1[flat]).mean() * 100 if flat.any() else 0.0

        # (B) End-to-end on-policy depth-1 using the engine's OWN hidden + predictions.
        if lib.qwn_reset_state() != 0:
            raise RuntimeError("qwn_reset_state failed")
        if lib.qwn_mtp_reset() != 0:
            raise RuntimeError("qwn_mtp_reset failed")
        eng_pred = np.empty(n, dtype=np.int64)     # engine main-model greedy at each pos
        eng_mtp1 = np.empty(L, dtype=np.int64)     # engine MTP top-1 (predicts pos t+2)
        ref_final_hidden = rdata["ref_final_hidden"]   # [n,H]
        norm_buf = (ctypes.c_float * H)()
        main_hid_corr = np.empty(n, dtype=np.float64)
        for t in range(n):
            pred = lib.qwn_decode(int(ids[t]), t)
            if pred < 0:
                raise RuntimeError(f"qwn_decode failed at pos {t}: {pred}")
            eng_pred[t] = pred
            # main-model fidelity baseline: engine vs reference post-final-norm hidden
            # (grab BEFORE qwn_mtp_step overwrites the shared norm scratch).
            nc = lib.qwn_copy_last_norm(norm_buf, H)
            eh = np.ctypeslib.as_array(norm_buf)[:nc]
            main_hid_corr[t] = np.corrcoef(eh, ref_final_hidden[t][:nc])[0, 1]
            if t < L:
                m1, _, _ = mtp_step(lib, pred, t, K)
                if m1 < 0:
                    raise RuntimeError(f"qwn_mtp_step failed at slot {t}: {m1}")
                eng_mtp1[t] = m1
        mean_main_hid_corr = float(np.nanmean(main_hid_corr)) * 100

        # main-model fidelity (engine greedy vs reference dense greedy).
        main_agree = (eng_pred[:n] == dense_pred[:n]).mean() * 100
        # engine on-policy depth-1: MTP top-1[t] == engine's own greedy at pos t+1.
        eng_op_d1 = (eng_mtp1[: L - 1] == eng_pred[1:L]).mean() * 100
        # cross-check vs the reference dense greedy target too.
        eng_op_d1_vs_ref = (eng_mtp1[: L - 1] == dense_pred[1:L]).mean() * 100
    finally:
        lib.qwn_free()

    # ---- report -----------------------------------------------------------------
    print("\n" + "=" * 74, flush=True)
    print("  Engine native MTP head — validation vs tools/mtp_accept.py", flush=True)
    print(f"  model={args.model_dir}", flush=True)
    print(f"  tqf={args.tqf}", flush=True)
    print(f"  slots L={L}  topk K={args.topk}", flush=True)
    print("=" * 74, flush=True)
    print(f"  reference hidden-extraction sanity     : {rdata['sanity']:.2f}%  (target ~100%)", flush=True)
    print(f"  engine main-model greedy == reference  : {main_agree:.2f}%  (main-model top-1 fidelity)", flush=True)
    print(f"  engine vs ref final-norm hidden corr   : {mean_main_hid_corr:.3f}%  (main-model quant baseline)", flush=True)
    print("  -- (A) MTP kernel correctness (inject reference hidden) --", flush=True)
    print("    [FP32-ref = FP8 weights + FP32 compute = matches the engine exactly]", flush=True)
    print(f"  engine vs FP32-ref logit correlation   : {mean_logit_corr_fp8:.3f}%  <-- KERNEL FIDELITY", flush=True)
    print(f"  engine MTP top-1 == FP32-ref MTP top-1 : {top1_agree_fp8:.2f}%", flush=True)
    print("    [BF16-ref = canonical mtp_accept.py head (BF16 compute)]", flush=True)
    print(f"  engine vs BF16-ref logit correlation   : {mean_logit_corr:.3f}%", flush=True)
    print(f"  FP32-ref vs BF16-ref logit correlation : {fp8_vs_bf16_corr:.3f}%  (BF16 vs FP32 compute gap)", flush=True)
    print(f"  engine MTP top-1 == BF16-ref MTP top-1 : {top1_agree:.2f}%  (all slots)", flush=True)
    print(f"    on CONFIDENT slots (margin>=median)  : {agree_conf:.2f}%", flush=True)
    print(f"    on near-tie slots  (margin< median)  : {agree_flat:.2f}%", flush=True)
    print(f"  engine MTP top-1 in BF16-ref top-{args.topk:<2d}     : {in_ref_topk:.2f}%", flush=True)
    print(f"  mean top-{args.topk} set Jaccard (engine,BF16ref): {jacc:.2f}%", flush=True)
    print("  -- (B) end-to-end on-policy depth-1 (engine's own hidden) --", flush=True)
    print(f"  reference on-policy depth-1 top-1       : {rdata['ref_op_d1']:.2f}%", flush=True)
    print(f"  engine    on-policy depth-1 top-1       : {eng_op_d1:.2f}%  (vs engine greedy)", flush=True)
    print(f"  engine    on-policy depth-1 top-1       : {eng_op_d1_vs_ref:.2f}%  (vs reference greedy)", flush=True)

    op_gap = abs(eng_op_d1 - rdata["ref_op_d1"])
    gate_a = mean_logit_corr_fp8 >= args.gate_kernel_corr * 100
    gate_b = op_gap <= args.gate_op_d1_tol
    verdict = "VALIDATED" if (gate_a and gate_b) else "FAILED"
    print("=" * 74, flush=True)
    print(f"  GATE (A) engine-vs-FP8ref kernel corr >= {args.gate_kernel_corr*100:.1f}% : "
          f"{'PASS' if gate_a else 'FAIL'} ({mean_logit_corr_fp8:.3f}%)", flush=True)
    print(f"  GATE (B) |engine-ref| on-policy d1 <= {args.gate_op_d1_tol:.0f}pt : "
          f"{'PASS' if gate_b else 'FAIL'} (gap {op_gap:.2f}pt; eng {eng_op_d1:.2f}% vs ref {rdata['ref_op_d1']:.2f}%)", flush=True)
    print(f"  RESULT: {verdict}", flush=True)
    print("=" * 74, flush=True)
    sys.exit(0 if verdict == "VALIDATED" else 1)


if __name__ == "__main__":
    main()
