#!/usr/bin/env python3
"""Native Qwen3.6-27B MTP (multi-token-prediction) head: reconstruct the exact
forward and measure teacher-forced draft acceptance vs the dense model.

What this does
--------------
1. Loads Qwen3.6-27B (HF `qwen3_5`, Qwen3_5ForConditionalGeneration) in BF16 on CPU.
   NOTE: `AutoModelForCausalLM` resolves qwen3_5 -> text-only `Qwen3_5ForCausalLM`,
   whose param names (`model.layers.*`) do NOT match this checkpoint's
   `model.language_model.*` keys -> it would silently load RANDOM weights. We load
   `Qwen3_5ForConditionalGeneration` (verified: 0 key mismatches).
2. Captures the PRE-final-norm last hidden state h_t for the whole sequence via a
   forward_pre_hook on `...language_model.norm`, plus the dense argmax per position.
   Sanity: argmax(lm_head(norm(h))) must equal model logits argmax (~100%).
3. Reconstructs the MTP head forward (confirmed against vLLM qwen3_5_mtp.py):
       e = pre_fc_norm_embedding(embed(tok_{t+1}))
       h = pre_fc_norm_hidden(hidden_t)
       x = fc(cat[e, h])
       x = MTP_full_attention_decoder_layer(x)         # reuse HF Qwen3_5DecoderLayer
       logits = lm_head(mtp.norm(x))
   The decoder layer is a deep-copy of a real full_attention layer with the
   dequantized mtp.layers.0.* weights loaded in, driven with the same RoPE
   (rotary_emb) and a causal mask, exactly like the model drives its own layers.
4. MTP weights come from the FP8 block-scaled checkpoint (dequant to BF16) AND, as a
   cross-check, the native BF16 copy embedded in the main checkpoint. Both heads are
   measured.
5. Reports depth d=1..4 top-1/4/8 accept (teacher-forced) and a greedy-rollout
   top-1 (part c).

Run (GPU 6 only):
    CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python3 -u tools/mtp_accept.py
"""

import copy
import json
import os
import time
from collections import defaultdict

import torch

# ----------------------------------------------------------------------------- config
MODEL_DIR = "/workspace/models/Qwen3.6-27B"
MTP_FP8 = "/workspace/qwen36-27b-fp8/mtp.safetensors"
BIGTEXT = "/tmp/bigtext.txt"
GPU = "cuda:0"                      # CUDA_VISIBLE_DEVICES=6 maps GPU 6 -> cuda:0
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
DEPTH = int(os.environ.get("DEPTH", "4"))
H = 5120

FALLBACK_TEXT = (
    "The history of science is full of surprising reversals. In 1905 a young clerk "
    "at the Swiss patent office published four papers that reshaped physics. Light, "
    "he argued, behaves as discrete quanta; matter and energy are interchangeable. "
    "Meanwhile, biologists were rediscovering Mendel, economists were debating gold, "
    "and engineers in Detroit were learning to build cars on a moving line. Consider "
    "a train that travels 60 kilometers in 45 minutes: its average speed is 80 km/h, "
    "because 60 divided by 0.75 equals 80. Programming offers similar clarity. Here is "
    "a function that returns the n-th Fibonacci number using fast doubling, which runs "
    "in O(log n) time rather than the naive linear scan. Markets, genomes, and source "
    "code all reward the same habit: state the invariant, then prove it holds."
)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- dequant
def dequant_fp8_block(w_fp8, scale_inv):
    """W[out,in] float8_e4m3 * S[ceil(out/128),ceil(in/128)] (per 128x128 block)."""
    w = w_fp8.to(GPU).to(torch.float32)
    s = scale_inv.to(GPU).to(torch.float32)
    out, inn = w.shape
    s = s.repeat_interleave(128, dim=0)[:out].repeat_interleave(128, dim=1)[:, :inn]
    return (w * s).to(torch.bfloat16)


def load_mtp_fp8():
    """Load mtp.safetensors; dequant the 7 FP8 block-scaled weights to BF16."""
    from safetensors.torch import load_file

    sd = load_file(MTP_FP8)
    out = {}
    for k, v in sd.items():
        if k.endswith("weight_scale_inv"):
            continue
        sk = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
        if sk is not None and sk in sd:
            out[k] = dequant_fp8_block(v, sd[sk])
        else:
            out[k] = v.to(torch.bfloat16).to(GPU)
    return out


def load_mtp_bf16():
    """Load native BF16 mtp.* tensors embedded in the main checkpoint shards."""
    from safetensors import safe_open

    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    groups = defaultdict(list)
    for k in idx:
        if k.startswith("mtp."):
            groups[idx[k]].append(k)
    out = {}
    for shard, keys in groups.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k).to(torch.bfloat16).to(GPU)
    return out


# ---------------------------------------------------------------- MTP head assembly
def build_head(weights, template_layer, Qwen3_5RMSNorm):
    """Assemble the MTP head modules (all on GPU, bf16) from a weight dict."""
    def rmsnorm(name):
        m = Qwen3_5RMSNorm(H, eps=1e-6)
        m.weight.data = weights[name].to(torch.float32).cpu()
        return m.to(GPU, torch.bfloat16).eval()

    fc = torch.nn.Linear(2 * H, H, bias=False)
    fc.weight.data = weights["mtp.fc.weight"].to(torch.float32).cpu()
    fc = fc.to(GPU, torch.bfloat16).eval()

    layer = copy.deepcopy(template_layer)
    pref = "mtp.layers.0."
    layer_sd = {k[len(pref):]: v for k, v in weights.items() if k.startswith(pref)}
    missing, unexpected = layer.load_state_dict(layer_sd, strict=False)
    assert not missing and not unexpected, (list(missing), list(unexpected))
    layer = layer.to(GPU, torch.bfloat16).eval()
    layer.self_attn.config._attn_implementation = "eager"

    return {
        "pe": rmsnorm("mtp.pre_fc_norm_embedding.weight"),
        "ph": rmsnorm("mtp.pre_fc_norm_hidden.weight"),
        "norm": rmsnorm("mtp.norm.weight"),
        "fc": fc,
        "layer": layer,
    }


@torch.no_grad()
def mtp_step(head, rotary_g, W_emb, hidden_in, token_ids, positions):
    """One MTP head application over a full causal sequence of slots.

    hidden_in [L,H] bf16, token_ids [L] long, positions [L] long  (all on GPU).
    Returns the POST-mtp.norm hidden [L,H] (this is what vLLM feeds back + lm_heads).
    """
    e = head["pe"](W_emb[token_ids])                  # norm(embed(tok))   [L,H]
    hh = head["ph"](hidden_in)                         # norm(hidden)       [L,H]
    x = head["fc"](torch.cat([e, hh], dim=-1)).unsqueeze(0)   # [1,L,H]
    cos, sin = rotary_g(x, positions.unsqueeze(0))     # [1,L,rot]
    L = x.shape[1]
    mask = torch.triu(
        torch.full((L, L), torch.finfo(torch.float32).min, device=GPU), diagonal=1
    )[None, None]                                      # additive causal [1,1,L,L]
    out = head["layer"](
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=mask,
        position_ids=None,
        past_key_values=None,
    )
    out = out[0] if isinstance(out, (tuple, list)) else out
    return head["norm"](out)[0]                        # post-norm [L,H]


@torch.no_grad()
def topk_metrics(logits, target, valid):
    """Return (top1, top4, top8) accept over valid slots. logits [L,V], target [L]."""
    top8 = logits.topk(8, dim=-1).indices              # [L,8]
    tgt = target.unsqueeze(1)
    in1 = top8[:, 0] == target
    in4 = (top8[:, :4] == tgt).any(1)
    in8 = (top8 == tgt).any(1)
    v = valid
    return (
        in1[v].float().mean().item() * 100,
        in4[v].float().mean().item() * 100,
        in8[v].float().mean().item() * 100,
    )


@torch.no_grad()
def run_chain(head, rotary_g, W_emb, W_lm, h_pre_g, input_ids_g, dense_pred_g, n, mode):
    """Autoregressive MTP rollout, depths 1..DEPTH (teacher-forced or greedy).

    slot t (t=0..n-2): predicts the token at sequence position t+d+1 at depth d.
      d=1 : hidden = h_main[t] (pre-final-norm), token = x[t+1]
      d>=2: hidden = post_{d-1}[t] (MTP's own post-norm), token = x[t+d] (teacher)
            or g_{d-1}[t] (the MTP's own previous greedy draft, mode='greedy')
      RoPE position = t+d (relative; a global shift is RoPE-invariant).
      target = dense_pred[t+d]  (= dense greedy token at position t+d+1).
    Cross-position attention at d>=2 uses the same-depth slots of earlier positions
    (the natural batched generalization of the depth-1 causal pass).
    """
    L = n - 1
    base_t = torch.arange(0, L, device=GPU)            # t = 0..n-2
    prev_post, prev_greedy = None, None
    rows = []
    for d in range(1, DEPTH + 1):
        # NOTE: the embedding lookup needs the TOKEN ID at a position, not the
        # position index. `base_t + d` is a position; the token id there is
        # `input_ids_g[base_t + d]`. (Feeding the position index as a token id is
        # a silent, catastrophic bug -> garbage embeddings -> ~8% accept.)
        if d == 1:
            hin = h_pre_g[0:L]
            token_ids = input_ids_g[(base_t + 1).clamp(max=n - 1)]   # true token x[t+1]
        else:
            hin = prev_post
            if mode == "teacher":
                token_ids = input_ids_g[(base_t + d).clamp(max=n - 1)]  # true token x[t+d]
            else:
                token_ids = prev_greedy   # MTP's own previous draft (already token ids)
        pos = base_t + d                  # RoPE position (relative; global shift invariant)
        post = mtp_step(head, rotary_g, W_emb, hin, token_ids, pos)
        logits = (post @ W_lm.t()).float()             # [L,V]
        tgt_idx = (base_t + d).clamp(max=n - 1)
        target = dense_pred_g[tgt_idx]
        valid = (base_t + d) <= (n - 1)
        rows.append((d, topk_metrics(logits, target, valid), int(valid.sum())))
        prev_post = post
        prev_greedy = logits.argmax(-1)
    return rows


def main():
    t0 = time.time()
    log("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    assert torch.cuda.is_available(), "GPU required (run with CUDA_VISIBLE_DEVICES=6)"
    log("device:", torch.cuda.get_device_name(0))

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5RMSNorm,
    )

    # ---- text -------------------------------------------------------------------
    text = open(BIGTEXT).read() if os.path.exists(BIGTEXT) else FALLBACK_TEXT
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok(text, add_special_tokens=False).input_ids[:MAX_TOKENS]
    n = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long)
    log(f"tokens: {n}")

    # ---- load model (CPU, bf16, eager) -----------------------------------------
    log("loading Qwen3.6-27B (BF16, CPU, eager attn) ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True
    ).eval()
    log(f"model loaded in {time.time() - t0:.1f}s")

    tm = model.model.language_model
    lm_head = model.lm_head
    final_norm = tm.norm
    rotary = tm.rotary_emb
    layer_types = model.config.text_config.layer_types
    full_idx = layer_types.index("full_attention")
    log(f"full_attention template layer idx = {full_idx}")

    # ---- capture pre-final-norm hidden + dense logits (one forward) -------------
    cap = {}

    def pre_hook(_mod, args):
        cap["h"] = args[0].detach()

    handle = final_norm.register_forward_pre_hook(pre_hook)
    log("running main-model forward on CPU (this is the slow step) ...")
    tf = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    handle.remove()
    log(f"main forward done in {time.time() - tf:.1f}s")

    logits_cpu = out.logits[0]                         # [n,V] bf16
    h_pre = cap["h"][0]                                # [n,H] bf16 (pre-final-norm)
    dense_pred = logits_cpu.argmax(-1)                 # [n]

    # ---- SANITY: hidden extraction ---------------------------------------------
    with torch.no_grad():
        rec = lm_head(final_norm(h_pre)).argmax(-1)
    sanity = (rec == dense_pred).float().mean().item() * 100
    log(f"SANITY  argmax(lm_head(norm(captured_h))) == model.logits.argmax : "
        f"{sanity:.2f}%  (target ~100%)")

    # ---- GPU tensors ------------------------------------------------------------
    W_lm = lm_head.weight.detach().to(GPU)             # [V,H]
    W_emb = tm.embed_tokens.weight.detach().to(GPU)    # [V,H]  (shared embedding)
    h_pre_g = h_pre.to(GPU)
    input_ids_g = input_ids[0].to(GPU)
    dense_pred_g = dense_pred.to(GPU)
    rotary_g = copy.deepcopy(rotary).to(GPU)
    template_layer = copy.deepcopy(tm.layers[full_idx]).cpu()

    # ---- load MTP weights (FP8 dequant + native BF16) + cross-check -------------
    log("loading MTP weights: FP8 (dequant) + native BF16 ...")
    w_fp8 = load_mtp_fp8()
    w_bf16 = load_mtp_bf16()
    fp8_layer_keys = [k for k in w_fp8 if k.startswith("mtp.layers.0.")
                      and any(s in k for s in ("q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"))]
    worst = 0.0
    for k in fp8_layer_keys:
        a, b = w_fp8[k].float(), w_bf16[k].float()
        rel = (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)
        worst = max(worst, rel)
    log(f"DEQUANT  max relative |fp8_dequant - bf16| over {len(fp8_layer_keys)} "
        f"FP8 weights = {worst:.4%}  (FP8 rounding, expect <~10%)")

    heads = {
        "FP8-dequant": build_head(w_fp8, template_layer, Qwen3_5RMSNorm),
        "native-BF16": build_head(w_bf16, template_layer, Qwen3_5RMSNorm),
    }

    # ---- measure ---------------------------------------------------------------
    summary = {}
    base_t = torch.arange(0, n - 1, device=GPU)
    for name, head in heads.items():
        tf = run_chain(head, rotary_g, W_emb, W_lm, h_pre_g, input_ids_g,
                       dense_pred_g, n, "teacher")
        gr = run_chain(head, rotary_g, W_emb, W_lm, h_pre_g, input_ids_g,
                       dense_pred_g, n, "greedy")
        # depth-1 "on-policy": feed the model's OWN greedy token dense_pred[t] (the
        # token the target model would actually produce) instead of the true text
        # token. This is the realistic spec-decode accept; brackets vLLM's 58%.
        with torch.no_grad():
            post = mtp_step(head, rotary_g, W_emb, h_pre_g[0:n - 1],
                            dense_pred_g[base_t], base_t + 1)
            op_pr = (post @ W_lm.t()).float().argmax(-1)
            op_tgt = dense_pred_g[(base_t + 1).clamp(max=n - 1)]
            op_valid = (base_t + 1) <= (n - 1)
            op_d1 = (op_pr == op_tgt)[op_valid].float().mean().item() * 100
        summary[name] = (tf, gr, op_d1)

    # ---- report ----------------------------------------------------------------
    print("\n" + "=" * 74, flush=True)
    print(f"  Qwen3.6-27B native MTP head — draft acceptance vs dense greedy", flush=True)
    print(f"  tokens={n}   hidden-extraction sanity={sanity:.2f}%   "
          f"dequant-rel-err={worst:.3%}", flush=True)
    print("=" * 74, flush=True)
    for name, (tf, gr, op_d1) in summary.items():
        d1 = tf[0][1][0]
        # Hard validation gate (per spec): 40-75% = valid impl; 50-62% = ideal band.
        valid = 40.0 <= d1 <= 75.0
        verdict = "VALIDATED" if valid else "FAILED"
        band = "" if 50.0 <= d1 <= 62.0 else "  [above 50-62 ideal: teacher-forced on coherent text]"
        print(f"\n[{name}]  depth-1 top-1 (teacher-forced) = {d1:.2f}%  vs ~58% target -> "
              f"{verdict}{band}", flush=True)
        print(f"           depth-1 top-1 (on-policy, feed model's greedy token) = {op_d1:.2f}%  "
              f"(realistic spec-decode accept)", flush=True)
        print("  teacher-forced (feed TRUE next token each step):", flush=True)
        print(f"    {'depth':>5} {'top1':>8} {'top4':>8} {'top8':>8} {'n':>7}", flush=True)
        for d, (t1, t4, t8), cnt in tf:
            print(f"    {d:>5} {t1:>7.2f}% {t4:>7.2f}% {t8:>7.2f}% {cnt:>7}", flush=True)
        print("  greedy rollout (feed MTP's OWN draft token each step):", flush=True)
        print(f"    {'depth':>5} {'top1':>8} {'top4':>8} {'top8':>8} {'n':>7}", flush=True)
        for d, (t1, t4, t8), cnt in gr:
            print(f"    {d:>5} {t1:>7.2f}% {t4:>7.2f}% {t8:>7.2f}% {cnt:>7}", flush=True)
    print("\n" + "=" * 74, flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
