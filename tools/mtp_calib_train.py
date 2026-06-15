#!/usr/bin/env python3
"""Calibrate the MTP draft head on FP6-engine hiddens/targets.

Replicates the ENGINE'S MTP forward exactly (gated attention, (1+w) rmsnorms,
per-head qk-norm, rotate-half rope on the first 64 dims, theta=1e7) and trains
ONLY the mtp.* tensors. embed_tokens and lm_head stay frozen: the head must
match the FP6-greedy argmax of the frozen verify, so moving the head would
shift the target itself.

Data: bench/calib/seg*.npz from mtp_calib_dump.py
  pair t: input (hidden[t], tokens[t+1]) -> target argmax[t+1]
Output: raw bf16 .bin per tensor in OUT (engine side-load format) + report.

v3 additions (default-off; behavior identical to the v2 trainer unless set):
  MTP_DEPTH=2|3    multi-depth self-distillation. Depth-2 unrolls the engine's
                   draft recurrence (mtp_batch_expand_wave): step-2 input pair =
                   (emb(argmax[t+1]) i.e. the depth-1 target, hn[t] the head's
                   own post-mtp.norm output hidden), rope/KV position t+1,
                   attention over the depth-1 k/v states at slots 0..t (the
                   committed-cache analog) plus its own k/v (ancestor archive
                   analog). Target argmax[t+2], loss weight MTP_D2_WEIGHT.
                   Depth-3 chains the same recurrence once more.
  MTP_D2_WEIGHT    depth-2 loss weight (default 0.5)
  MTP_D3_WEIGHT    depth-3 loss weight (default 0.25)
  CKPT_FRAC=0.15   save a mid-epoch checkpoint every ~15% of an epoch into
                   CKPT_PREFIX<N> (same .bin export as OUT) for gate-based
                   selection; val metrics per ckpt unless EVAL_AT_CKPT=0
  CKPT_PREFIX      checkpoint dir prefix (default: OUT + "_ckpt")
"""
from __future__ import annotations
import glob, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.environ.get("CAL_DIR", os.path.join(ROOT, "bench", "calib"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "bench", "calib", "mtp_ft"))
MODEL = "/workspace/models/Qwen3.6-27B"
DEV = "cuda"
H, NH, NKV, HD, I, V = 5120, 24, 4, 256, 17408, 248320
ROT, THETA, EPS = 64, 1.0e7, 1.0e-6
P0 = 48                                   # segment position offset (rope)
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "5e-5"))
BATCH = int(os.environ.get("BATCH", "2"))
CE_CHUNK = 64
DEPTH_SUP = int(os.environ.get("MTP_DEPTH", "1"))      # supervised draft depth
D2W = float(os.environ.get("MTP_D2_WEIGHT", "0.5"))
D3W = float(os.environ.get("MTP_D3_WEIGHT", "0.25"))
CKPT_FRAC = float(os.environ.get("CKPT_FRAC", "0"))    # >0: mid-epoch checkpoints
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", OUT + "_ckpt")
EVAL_AT_CKPT = int(os.environ.get("EVAL_AT_CKPT", "1"))

def load_tensors(names):
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    out = {}
    by_file = {}
    for n in names:
        by_file.setdefault(idx[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(os.path.join(MODEL, f), framework="pt") as sf:
            for n in ns:
                out[n] = sf.get_tensor(n)
    return out

MTP_NAMES = [
    "mtp.fc.weight", "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]

def rms(x, w):
    # engine convention: x * rsqrt(mean(x^2)+eps) * (1 + w)
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * (1.0 + w)

def rope(x, pos):
    # x: [B,T,h,HD]; rotate-half on the first ROT dims, pairs (i, i+ROT/2)
    half = ROT // 2
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    freq = THETA ** (-2.0 * idx / ROT)
    ang = pos[:, :, None].float() * freq[None, None, :]          # [B,T,half]
    c = torch.cos(ang)[:, :, None, :].to(x.dtype)
    s = torch.sin(ang)[:, :, None, :].to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:ROT]
    rx1 = x1 * c - x2 * s
    rx2 = x2 * c + x1 * s
    return torch.cat([rx1, rx2, x[..., ROT:]], dim=-1)

class MTPHead(nn.Module):
    def __init__(self, t):
        super().__init__()
        mk = lambda n: nn.Parameter(t[n].to(torch.float32))
        self.fc = mk("mtp.fc.weight")
        self.norm = mk("mtp.norm.weight")
        self.pre_emb = mk("mtp.pre_fc_norm_embedding.weight")
        self.pre_hid = mk("mtp.pre_fc_norm_hidden.weight")
        self.in_ln = mk("mtp.layers.0.input_layernorm.weight")
        self.post_ln = mk("mtp.layers.0.post_attention_layernorm.weight")
        self.qw = mk("mtp.layers.0.self_attn.q_proj.weight")
        self.kw = mk("mtp.layers.0.self_attn.k_proj.weight")
        self.vw = mk("mtp.layers.0.self_attn.v_proj.weight")
        self.ow = mk("mtp.layers.0.self_attn.o_proj.weight")
        self.qn = mk("mtp.layers.0.self_attn.q_norm.weight")
        self.kn = mk("mtp.layers.0.self_attn.k_norm.weight")
        self.gw = mk("mtp.layers.0.mlp.gate_proj.weight")
        self.uw = mk("mtp.layers.0.mlp.up_proj.weight")
        self.dw = mk("mtp.layers.0.mlp.down_proj.weight")

    def forward(self, emb, hid, pos, prev_kv=None, diag_kvs=(), return_kv=False):
        # emb: [B,T,H] frozen embedding of the input token; hid: [B,T,H] input
        # hidden (trunk h_t at depth-1, the head's own hn at depth>=2).
        # prev_kv: (k,v) [B,T,NKV,HD] depth-1 states attended CAUSALLY (the
        # engine's committed MTP cache analog). diag_kvs: list of (k,v) states
        # of intermediate-depth ancestors, attended at slot t only (ancestor
        # archive analog). Default (both unset) = the v2 depth-1 forward.
        B, T, _ = emb.shape
        fc_in = torch.cat([rms(emb, self.pre_emb), rms(hid, self.pre_hid)], dim=-1)
        x = fc_in @ self.fc.T                                     # [B,T,H]
        r = x
        xn = rms(x, self.in_ln)
        qg = (xn @ self.qw.T).view(B, T, NH, 2, HD)
        q, gate = qg[..., 0, :], qg[..., 1, :]
        k = (xn @ self.kw.T).view(B, T, NKV, HD)
        v = (xn @ self.vw.T).view(B, T, NKV, HD)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + self.qn)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + self.kn)
        q = rope(q, pos); k = rope(k, pos)
        if prev_kv is None and not diag_kvs:
            kr = k.repeat_interleave(NH // NKV, dim=2)
            vr = v.repeat_interleave(NH // NKV, dim=2)
            a = F.scaled_dot_product_attention(
                q.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2),
                is_causal=True, scale=1.0 / math.sqrt(HD))
        else:
            pk, pv = prev_kv
            assert pk.shape[1] == T, "slice prev_kv to the query length"
            kk = torch.cat([pk] + [dk for dk, _ in diag_kvs] + [k], dim=1)
            vv = torch.cat([pv] + [dv for _, dv in diag_kvs] + [v], dim=1)
            eye = torch.eye(T, dtype=torch.bool, device=emb.device)
            mask = torch.cat(
                [torch.ones(T, T, dtype=torch.bool, device=emb.device).tril_()]
                + [eye] * (len(diag_kvs) + 1), dim=1)             # [T, (2+ndiag)*T]
            kr = kk.repeat_interleave(NH // NKV, dim=2)
            vr = vv.repeat_interleave(NH // NKV, dim=2)
            a = F.scaled_dot_product_attention(
                q.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2),
                attn_mask=mask[None, None], scale=1.0 / math.sqrt(HD))
        a = a.transpose(1, 2) * torch.sigmoid(gate)               # gated attention
        x = r + a.reshape(B, T, NH * HD) @ self.ow.T
        r2 = x
        xn2 = rms(x, self.post_ln)
        x = r2 + (F.silu(xn2 @ self.gw.T) * (xn2 @ self.uw.T)) @ self.dw.T
        hn = rms(x, self.norm)                                    # pre-lm_head
        if return_kv:
            return hn, (k, v)
        return hn


def save_weights(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name_to_param = {
        "mtp.fc.weight": model.fc, "mtp.norm.weight": model.norm,
        "mtp.pre_fc_norm_embedding.weight": model.pre_emb,
        "mtp.pre_fc_norm_hidden.weight": model.pre_hid,
        "mtp.layers.0.input_layernorm.weight": model.in_ln,
        "mtp.layers.0.post_attention_layernorm.weight": model.post_ln,
        "mtp.layers.0.self_attn.q_proj.weight": model.qw,
        "mtp.layers.0.self_attn.k_proj.weight": model.kw,
        "mtp.layers.0.self_attn.v_proj.weight": model.vw,
        "mtp.layers.0.self_attn.o_proj.weight": model.ow,
        "mtp.layers.0.self_attn.q_norm.weight": model.qn,
        "mtp.layers.0.self_attn.k_norm.weight": model.kn,
        "mtp.layers.0.mlp.gate_proj.weight": model.gw,
        "mtp.layers.0.mlp.up_proj.weight": model.uw,
        "mtp.layers.0.mlp.down_proj.weight": model.dw,
    }
    for n, p in name_to_param.items():
        raw = p.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
        raw.tofile(os.path.join(out_dir, n.replace("/", "_") + ".bin"))
    return len(name_to_param)


def main():
    files = sorted(glob.glob(os.path.join(CAL, "seg*.npz")))
    assert files, f"no calib segments in {CAL}"
    segs = [np.load(f) for f in files]
    n_val = max(2, len(segs) // 20)
    train, val = segs[:-n_val], segs[-n_val:]
    print(f"segments: {len(train)} train / {len(val)} val", flush=True)
    if DEPTH_SUP > 1:
        print(f"multi-depth distillation: depth={DEPTH_SUP} w2={D2W} w3={D3W}", flush=True)

    t = load_tensors(MTP_NAMES + ["model.language_model.embed_tokens.weight", "lm_head.weight"])
    embed = t["model.language_model.embed_tokens.weight"].to(DEV, torch.bfloat16)
    head_w = t["lm_head.weight"].to(DEV, torch.bfloat16)
    model = MTPHead(t).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    nstep = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, nstep))

    def batch_tensors(group):
        T = min(int(s["tokens"].shape[0]) for s in group) - 1
        hid = torch.stack([torch.from_numpy(s["hidden"][:T].astype(np.float32)) for s in group]).to(DEV)
        nxt = torch.stack([torch.from_numpy(s["tokens"][1:T + 1].astype(np.int64)) for s in group]).to(DEV)
        tgt = torch.stack([torch.from_numpy(s["argmax"][1:T + 1].astype(np.int64)) for s in group]).to(DEV)
        pos = (torch.arange(T, device=DEV) + P0).unsqueeze(0).expand(len(group), T)
        return hid, nxt, tgt, pos

    def ce_and_top1(hn, tgt, train_mode, weight=1.0, retain=False):
        # chunked CE over positions (full-vocab logits would be 0.5 GB)
        B, T, _ = hn.shape
        tot, hits = 0.0, 0
        for c0 in range(0, T, CE_CHUNK):
            c1 = min(T, c0 + CE_CHUNK)
            logits = hn[:, c0:c1].to(torch.bfloat16) @ head_w.T
            ls = logits.float()
            l = F.cross_entropy(ls.reshape(-1, V), tgt[:, c0:c1].reshape(-1), reduction="sum")
            if train_mode:
                (weight * l / (B * T)).backward(retain_graph=(c1 < T) or retain)
            with torch.no_grad():
                hits += (ls.argmax(-1) == tgt[:, c0:c1]).sum().item()
            tot += l.item()
        return tot / (B * T), hits / (B * T)

    def depth_losses(hid, nxt, tgt, pos, train_mode):
        # one (loss, top1) per supervised depth; backward inside when training.
        # Depth d>=2 unrolls the engine recurrence: input token = the depth-(d-1)
        # TARGET argmax (teacher-forced surviving draft), input hidden = the
        # head's own post-norm output, rope/KV slot shifted by +1 per depth,
        # attention = depth-1 states 0..t (causal) + own-chain states at t.
        out = []
        res = model(embed[nxt].float(), hid, pos, return_kv=DEPTH_SUP >= 2)
        hn1 = res[0] if DEPTH_SUP >= 2 else res
        out.append(ce_and_top1(hn1, tgt, train_mode, 1.0, retain=DEPTH_SUP >= 2))
        if DEPTH_SUP >= 2:
            k1, v1 = res[1]
            Tq = tgt.shape[1] - 1
            res2 = model(embed[tgt[:, :Tq]].float(), hn1[:, :Tq], pos[:, :Tq] + 1,
                         prev_kv=(k1[:, :Tq], v1[:, :Tq]), return_kv=DEPTH_SUP >= 3)
            hn2 = res2[0] if DEPTH_SUP >= 3 else res2
            out.append(ce_and_top1(hn2, tgt[:, 1:], train_mode, D2W, retain=DEPTH_SUP >= 3))
        if DEPTH_SUP >= 3:
            k2, v2 = res2[1]
            T3 = tgt.shape[1] - 2
            hn3 = model(embed[tgt[:, 1:1 + T3]].float(), hn2[:, :T3], pos[:, :T3] + 2,
                        prev_kv=(k1[:, :T3], v1[:, :T3]),
                        diag_kvs=[(k2[:, :T3], v2[:, :T3])])
            out.append(ce_and_top1(hn3, tgt[:, 2:], train_mode, D3W, retain=False))
        return out

    def fmt_extra(evs):
        return "".join(f" d{i + 2}_loss={l:.4f} d{i + 2}_top1={100*h:.2f}%"
                       for i, (l, h) in enumerate(evs[1:]))

    @torch.no_grad()
    def evaluate():
        model.eval()
        acc, n = None, 0
        for i in range(0, len(val), BATCH):
            group = val[i:i + BATCH]
            hid, nxt, tgt, pos = batch_tensors(group)
            evs = depth_losses(hid, nxt, tgt, pos, False)
            if acc is None:
                acc = [[0.0, 0.0] for _ in evs]
            for j, (l, h) in enumerate(evs):
                acc[j][0] += l; acc[j][1] += h
            n += 1
        model.train()
        return [(a / n, b / n) for a, b in acc]

    ev0 = evaluate()
    l0, h0 = ev0[0]
    print(f"BASELINE  val_loss={l0:.4f}  top1={100*h0:.2f}%{fmt_extra(ev0)}", flush=True)
    spe = len(train) // BATCH
    ckpt_every = max(1, round(spe * CKPT_FRAC)) if CKPT_FRAC > 0 else 0
    nck = 0
    step = 0
    for ep in range(EPOCHS):
        order = np.random.RandomState(ep).permutation(len(train))
        for i in range(0, len(train) - BATCH + 1, BATCH):
            group = [train[j] for j in order[i:i + BATCH]]
            hid, nxt, tgt, pos = batch_tensors(group)
            opt.zero_grad(set_to_none=True)
            losses = depth_losses(hid, nxt, tgt, pos, True)
            loss, top1 = losses[0]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 25 == 0:
                extra = "".join(f" d{di + 2}_loss={l:.4f} d{di + 2}_top1={100*h:.2f}%"
                                for di, (l, h) in enumerate(losses[1:]))
                print(f"  ep{ep} step{step}/{nstep} loss={loss:.4f} top1={100*top1:.2f}%{extra}", flush=True)
            if ckpt_every and step % ckpt_every == 0 and step < nstep:
                nck += 1
                cdir = f"{CKPT_PREFIX}{nck}"
                save_weights(model, cdir)
                meta = {"step": step, "nstep": nstep, "frac": round(step / nstep, 4)}
                extra = ""
                if EVAL_AT_CKPT:
                    ev = evaluate()
                    meta["val"] = [[round(l, 4), round(h, 4)] for l, h in ev]
                    extra = f" val_loss={ev[0][0]:.4f} top1={100*ev[0][1]:.2f}%{fmt_extra(ev)}"
                json.dump(meta, open(os.path.join(cdir, "ckpt_meta.json"), "w"))
                print(f"CKPT {nck} @ step {step}/{nstep} -> {cdir}{extra}", flush=True)
        ev = evaluate()
        l, h1 = ev[0]
        print(f"epoch {ep}: val_loss={l:.4f} top1={100*h1:.2f}%{fmt_extra(ev)}  (baseline {l0:.4f}/{100*h0:.2f}%)", flush=True)

    nsaved = save_weights(model, OUT)
    print(f"saved {nsaved} tensors -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
