#!/usr/bin/env python3
"""The engine's own numerics end to end: KL / top-1 of libforward_qwen.so vs the original bf16 model on the held-out
code tokens (docs/QUALITY.md). This is the tool behind the README's "quality vs the bf16 original" cells.

  ref    reference final-norm outputs of the original bf16 model (exllamav3, layer by layer on one GPU) -> OUT_DIR/refstates/
  score  the bare engine (the same libforward_qwen.so, paged pool and TQ_* env as tools/serve_batched.py) on the same rows:
         positions < --decode-from go through paged prefill waves (<= 128 columns, one segment: how the server prefills a
         prompt), the rest through teacher-forced paged decode steps with --batch rows in lockstep (the served decode path:
         N-column GEMMs, recurrent DeltaNet, GQA decode attention). The engine's logits are scored against exllamav3's
         lm_head on the reference's final-norm outputs with exl3_diff's metrics (compute_kl_div, argmax agreement); the
         per-token .pt has tools/exl3_diff.py's (row, pos) layout, so tools/e2e_frontier.py pairs it with every run of
         the same (seq, seed) -- in particular with the FP8 reference row.

exllamav3's lm_head returns fp16 logits, so every exl3_diff candidate is scored on fp16-rounded logits; the engine's
are fp32. kl_mean is on the engine's fp32 logits; kl16_mean rounds them to fp16 first (like-for-like with exl3_diff).

  export CUDA_VISIBLE_DEVICES=0 TQ_KV_V4=1 TQ_KV_T4=1 TQ_PDL=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1   # the published rows' env
  python3 tools/qwentin_e2e.py ref   --ref-dir /models/Qwen3.8-27B --corpus /path/to/vllm/vllm --out-dir runs/e2e --seq 4096
  python3 tools/qwentin_e2e.py score --ref-dir /models/Qwen3.8-27B --corpus /path/to/vllm/vllm --out-dir runs/e2e --seq 4096 \
      --tqf /models/qwen3_8-27b-exl3-opt5.75-mtp.tqf --lib build-qwen/libforward_qwen.so --decode-from 4096 --tag opt575_4k_prefill
  python3 tools/qwentin_e2e.py score ... --seq 4096  --decode-from 0 --batch 8 --tag opt575_4k_dec8        # decode path, 8 rows lockstep
  python3 tools/qwentin_e2e.py score ... --seq 32768 --decode-from 32768 --tag opt575_32k_prefill
  python3 tools/e2e_frontier.py --e2e-dir runs/e2e --vs fp8_4k --tags opt575_4k_prefill                   # the paired CI

Vendored from analysis/qwentin_e2e.py of the research project (kacper-daftcode/qwentin-internal): the hard-coded box
paths became arguments (--corpus, --ref-dir, --out-dir, --tqf, --lib are required / repo-relative), nothing else
changed. Needs exllamav3 (upstream 02aef45 or newer) for the reference pass and the lm_head, and a CUDA GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exl3_diff import add_exllamav3_path, load_module, unload_module  # noqa: E402
from harness_common import calibration_ids, corpus_fingerprint, load_corpus  # noqa: E402

# serve_batched.py's environment; TQ_WIDE_ATTN_MMA / TQ_EMBED_HOST / TQ_NO_MTP are what its --wide-attn-mma /
# --embed-host / no --mtp defaults set (TQ_EMBED_HOST overrides TQ_EMBED_FP8). Explicit env wins (setdefault).
SERVED_ENV = {"TQ_CTX": "262144", "TQ_KV_Q4": "1", "TQ_EMBED_FP8": "2", "TQ_PAGED_ATTN_MMA": "1", "TQ_WIDE_ATTN_MMA": "1",
              "TQ_EMBED_HOST": "1", "TQ_NO_MTP": "1"}
WAVE_MAX = 128
PAGE = 128
TOPK = 5


def held_out(tokenizer_dir: str, corpus_dir: str, seq: int, seed: int, tokens: int) -> torch.Tensor:
    """exl3_diff.py's rows: calibration_ids(seed) cut into n_rows x seq."""
    from transformers import AutoTokenizer

    corpus = load_corpus(corpus_dir)
    ids = torch.tensor(calibration_ids(AutoTokenizer.from_pretrained(tokenizer_dir), corpus, tokens, seed=seed),
                       dtype=torch.long)
    n_rows = ids.numel() // seq
    return ids[: n_rows * seq].view(n_rows, seq), corpus


def refstates_path(out_dir: str, seq: int, seed: int, tokens: int) -> str:
    return os.path.join(out_dir, "refstates", f"ref-{seq}-s{seed}-t{tokens}.pt")


@torch.inference_mode()
def cmd_ref(args):
    from exllamav3 import Config, Model

    out = refstates_path(args.out_dir, args.seq, args.seed, args.tokens)
    if os.path.exists(out):
        print(f"exists: {out}")
        return
    device = torch.device("cuda:0")
    ids, _ = held_out(args.tokenizer or args.ref_dir, args.corpus, args.seq, args.seed, args.tokens)
    n_rows = ids.shape[0]
    print(f"held-out: {n_rows} x {args.seq} tokens", flush=True)
    cfg = Config.from_directory(args.ref_dir)
    model = Model.from_config(cfg)
    state_dev = torch.device("cpu") if args.states_on_cpu else device
    params = [{} for _ in range(n_rows)]
    states = [model.prepare_inputs(ids[j : j + 1].clone(), params[j]) for j in range(n_rows)]
    t_start = time.time()
    for idx, m in enumerate(model.modules[:-1]):     # through the final norm; the lm_head runs at scoring time
        t0 = time.time()
        load_module(cfg, m, device)
        for j in range(n_rows):
            params[j]["dev_cache"] = None
            x = m.prepare_for_device(states[j].to(device), params[j])
            states[j] = m.forward(x, params[j]).clone().to(state_dev)
            del x
        unload_module(cfg, m)
        print(f"[{idx:3d}] {m.key:48s} ({time.time() - t0:.0f}s)", flush=True)
    final = torch.cat([s.cpu() for s in states])
    assert final.dtype == torch.half and final.shape == (n_rows, args.seq, cfg.hidden_size), (final.dtype, final.shape)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"states": final, "ids": ids, "seq": args.seq, "seed": args.seed, "tokens": args.tokens, "ref": args.ref_dir,
                "module": model.modules[-2].key}, out)
    print(f"wrote {out} ({final.numel() * 2 / 1e9:.2f} GB) in {time.time() - t_start:.0f}s")


class Engine:
    """The bare engine through serve_batched.py's ctypes signatures, plus the qwn_paged_logits_sink test ABI."""

    def __init__(self, lib: str, tqf: str, slots: int, blocks: int):
        from serve_batched import load_lib

        self.L = load_lib(lib)
        self.L.qwn_vocab_size.restype = ctypes.c_int
        self.L.qwn_device_bytes.restype = ctypes.c_size_t
        self.L.qwn_paged_logits_sink.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.L.qwn_paged_logits_sink.restype = ctypes.c_int
        self._ck(self.L.qwn_init(tqf.encode()), "init")
        self.V = self.L.qwn_vocab_size()
        self._ck(self.L.qwn_paged_init(slots, blocks, PAGE), "paged_init")
        self.seed = (ctypes.c_int * 1)()
        self.argmax = (ctypes.c_int * slots)()

    def set_sink(self, sink: torch.Tensor, head_cols: int):
        assert sink.is_cuda and sink.dtype == torch.float32 and sink.is_contiguous() and sink.shape[1] == self.V
        self._ck(self.L.qwn_paged_logits_sink(ctypes.c_void_p(sink.data_ptr()), sink.shape[0], head_cols), "sink")

    @staticmethod
    def _ck(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} rc={rc}")

    @staticmethod
    def _ci(a):
        a = [int(x) for x in a]
        return (ctypes.c_int * len(a))(*a)

    def reset(self, slot: int):
        self._ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")

    def wave(self, slot: int, toks, pos0: int, final: bool) -> int:
        """One single-segment ragged prefill wave; the logits of its columns land in the sink's first rows."""
        torch.cuda.synchronize()          # the scorer is done reading the sink
        n = len(toks)
        self._ck(self.L.qwn_paged_prefill_batch(self._ci(toks), self._ci([slot] * n), self._ci(range(pos0, pos0 + n)),
                                                self._ci([slot]), self._ci([0]), self._ci([n]), self._ci([int(final)]),
                                                1, n, self.seed), f"prefill_batch pos {pos0}")
        return self.seed[0]

    def step(self, toks, slots, positions) -> list[int]:
        """One paged decode step; the logits of its N columns land in the sink's first N rows."""
        torch.cuda.synchronize()
        n = len(toks)
        self._ck(self.L.qwn_paged_decode_step(self._ci(toks), self._ci(slots), self._ci(positions), n, self.argmax),
                 f"decode_step pos {positions[0]}")
        return list(self.argmax[:n])


class Scorer:
    """exl3_diff.py's per-token metrics: KL(ref || engine), top-1/top-k agreement, NLL of both, ref confidence."""

    def __init__(self, ref_dir: str, ids: torch.Tensor, states: torch.Tensor, device):
        from exllamav3 import Config, Model

        self.ids, self.states, self.device = ids, states, device
        cfg = Config.from_directory(ref_dir)
        self.head = Model.from_config(cfg).modules[-1]
        load_module(cfg, self.head, device)
        self.vocab = cfg.vocab_size
        n_rows, seq = ids.shape
        shape = (n_rows, seq - 1)
        self.kl, self.kl16, self.conf = torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
        self.agree, self.agree16 = torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)
        self.decode, self.done = torch.zeros(shape, dtype=torch.bool), torch.zeros(shape, dtype=torch.bool)
        # nll ref/engine, label top-1 ref/engine, kl_rev, top-k agreement 1..TOPK
        self.sums = torch.zeros(5 + TOPK, dtype=torch.float64, device=device)
        self.n = 0
        self.ref_dtype = None

    @torch.inference_mode()
    def score(self, rows, poss, lq: torch.Tensor, decode: bool):
        from exllamav3.util.measures import compute_kl_div

        rows_t = torch.as_tensor(list(rows), dtype=torch.long)
        pos_t = torch.as_tensor(list(poss), dtype=torch.long)
        keep = pos_t < self.ids.shape[1] - 1             # a row's last position has no target
        if not keep.all():
            nk = int(keep.sum())
            assert keep[:nk].all(), "only a wave's trailing columns can be past the row"
            rows_t, pos_t, lq = rows_t[keep], pos_t[keep], lq[:nk]
        if rows_t.numel() == 0:
            return
        x = self.states[rows_t, pos_t].to(self.device)
        lr = self.head.forward(x[None].contiguous(), {})[0, :, : self.vocab]
        self.ref_dtype = str(lr.dtype)
        lq = lq[:, : self.vocab].contiguous()
        lq16 = lq.half()
        tgt = self.ids[rows_t, pos_t + 1].to(self.device)
        s = self.sums
        for i, l in enumerate((lr, lq)):
            lp = torch.log_softmax(l.float(), -1)
            s[i] -= lp.gather(-1, tgt[:, None]).sum()
            del lp
            s[2 + i] += (l.argmax(-1) == tgt).sum()
        tk_r = lr.topk(TOPK, dim=-1).indices
        tk_q = lq.topk(TOPK, dim=-1).indices
        for t in range(TOPK):
            s[5 + t] += (tk_r[:, : t + 1] == tk_q[:, : t + 1]).all(-1).sum()
        s[4] += compute_kl_div(lr, lq, self.vocab).sum()
        kl = compute_kl_div(lq, lr, self.vocab)          # KL(softmax(ref) || softmax(engine)) per token
        kl16 = compute_kl_div(lq16, lr, self.vocab)
        cf = lr.float()
        conf = (cf.max(-1).values - cf.logsumexp(-1)).exp()
        self.kl[rows_t, pos_t] = kl.flatten().float().cpu()
        self.kl16[rows_t, pos_t] = kl16.flatten().float().cpu()
        self.agree[rows_t, pos_t] = (tk_r[:, 0] == tk_q[:, 0]).cpu()
        self.agree16[rows_t, pos_t] = (tk_r[:, 0] == lq16.argmax(-1)).cpu()
        self.conf[rows_t, pos_t] = conf.cpu()
        self.decode[rows_t, pos_t] = decode
        self.done[rows_t, pos_t] = True
        self.n += rows_t.numel()

    def running_kl(self) -> float:
        return self.kl[self.done].double().mean().item()

    def result(self, n_rows: int, seq: int) -> tuple[dict, dict]:
        assert self.done.all(), f"{int((~self.done).sum())} positions not scored"
        kl_all, agree_all, conf_all = self.kl.flatten(), self.agree.flatten(), self.conf.flatten()
        pos_all = torch.arange(seq - 1, dtype=torch.int32).repeat(n_rows)
        row_all = torch.arange(n_rows, dtype=torch.int16).repeat_interleave(seq - 1)
        dec_all = self.decode.flatten()
        n = self.n
        s = self.sums.tolist()
        ppl = [math.exp(s[i] / n) for i in (0, 1)]
        res = {
            "ref_nll": s[0] / n, "ref_ppl": ppl[0], "ref_label_top1": s[2] / n,
            "q_nll": s[1] / n, "q_ppl": ppl[1], "q_label_top1": s[3] / n,
            "ppl_ratio": ppl[1] / ppl[0],
            "top1_agree": agree_all.float().mean().item(),
            "topk_agree": {t + 1: s[5 + t] / n for t in range(TOPK)},
            "kl_mean": kl_all.double().mean().item(), "kl_rev_mean": s[4] / n,
            "kl_median": kl_all.median().item(), "kl_p90": kl_all.quantile(0.9).item(),
            "kl_p99": kl_all.quantile(0.99).item(),
            "kl16_mean": self.kl16.double().mean().item(), "top1_agree16": self.agree16.float().mean().item(),
            "ref_logits_dtype": self.ref_dtype,
            "by_mode": {}, "by_position": {}, "by_ref_confidence": {},
        }
        for name, mask in (("prefill", ~dec_all), ("decode", dec_all)):
            if mask.any():
                res["by_mode"][name] = {"n": int(mask.sum()), "kl_mean": kl_all[mask].double().mean().item(),
                                        "top1_agree": agree_all[mask].float().mean().item()}
        for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)]:
            mask = (conf_all >= lo) & (conf_all < hi)
            if mask.any():
                res["by_ref_confidence"][f"[{lo:.2f},{min(hi, 1.0):.2f})"] = {
                    "frac_tokens": int(mask.sum()) / conf_all.numel(), "kl_mean": kl_all[mask].mean().item(),
                    "top1_agree": agree_all[mask].float().mean().item()}
        edges = [0, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 30]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (pos_all >= lo) & (pos_all < hi)
            if mask.any():
                res["by_position"][f"[{lo},{hi})"] = {"n": int(mask.sum()), "kl_mean": kl_all[mask].mean().item(),
                                                      "top1_agree": agree_all[mask].float().mean().item()}
        tok = {"kl": kl_all, "agree": agree_all, "kl16": self.kl16.flatten(), "agree16": self.agree16.flatten(),
               "conf": conf_all.half(), "pos": pos_all, "row": row_all, "decode": dec_all, "seq": seq, "n_rows": n_rows}
        return res, tok


def cmd_score(args):
    for k, v in SERVED_ENV.items():
        os.environ.setdefault(k, v)
    out = args.out or os.path.join(args.out_dir, f"{args.tag}.json")
    if os.path.exists(out) and not args.force:
        print(f"exists: {out}")
        return
    device = torch.device("cuda:0")
    ids, corpus = held_out(args.tokenizer or args.ref_dir, args.corpus, args.seq, args.seed, args.tokens)
    n_rows, seq = ids.shape
    if args.rows:
        n_rows = min(n_rows, args.rows)
        ids = ids[:n_rows]
    ref_path = refstates_path(args.out_dir, args.seq, args.seed, args.tokens)
    if not os.path.exists(ref_path):
        raise SystemExit(f"reference states {ref_path} missing: run `qwentin_e2e.py ref` with the same --seq/--seed/--tokens first")
    ref = torch.load(ref_path)
    assert torch.equal(ref["ids"][:n_rows], ids), "reference states belong to other rows (different corpus / tokenizer / seed?)"
    states = ref["states"][:n_rows]
    P = min(args.decode_from, seq)
    batch = 1 if P >= seq - 1 else min(args.batch, n_rows)
    blocks = batch * ((seq + PAGE - 1) // PAGE)
    wave_max = args.wave_max          # 128 = the server's waves; 256 needs TQ_WAVE_MAX=256 in the engine
    print(f"held-out: {n_rows} x {seq}; positions < {P} prefill waves of <= {wave_max} columns, >= {P} decode steps; "
          f"{batch} rows in lockstep; {blocks} blocks; prefill head in {args.head_cols}-column chunks", flush=True)

    t0 = time.time()
    eng = Engine(args.lib, args.tqf, batch, blocks)
    sink = torch.empty((max(wave_max, batch), eng.V), dtype=torch.float32, device=device)
    eng.set_sink(sink, args.head_cols)
    sc = Scorer(args.ref_dir, ids, states, device)
    free, total = torch.cuda.mem_get_info()
    print(f"engine + reference head up in {time.time() - t0:.0f}s: engine device bytes "
          f"{eng.L.qwn_device_bytes() / 1e9:.2f} GB, GPU free {free / 1e9:.2f} of {total / 1e9:.2f} GB", flush=True)
    env = {k: v for k, v in os.environ.items() if k.startswith("TQ_")}
    seed_mismatch = 0
    t_start = time.time()
    for b0 in range(0, n_rows, batch):
        rows = list(range(b0, min(b0 + batch, n_rows)))
        for s, j in enumerate(rows):
            eng.reset(s)
            for c0 in range(0, P, wave_max):
                c = min(wave_max, P - c0)
                final = c0 + c >= P
                seed = eng.wave(s, ids[j, c0 : c0 + c].tolist(), c0, final)
                if final and int(sink[c - 1].argmax()) != seed:
                    seed_mismatch += 1        # the seed's own Kf=1 head call vs the chunked one: near-ties only
                sc.score([j] * c, range(c0, c0 + c), sink[:c], decode=False)
        slots = list(range(len(rows)))
        n = len(rows)
        for t in range(P, seq - 1):
            am = torch.tensor(eng.step(ids[rows, t].tolist(), slots, [t] * n), device=device)
            lg = sink[:n]
            if (lg.gather(-1, am[:, None])[:, 0] != lg.max(-1).values).any():
                raise RuntimeError(f"the engine's argmax is not the max of the sink row at pos {t}")
            sc.score(rows, [t] * n, lg, decode=True)
            if (t - P) % 4096 == 4095:
                print(f"   decode pos {t + 1} ({time.time() - t_start:.0f}s): running KL {sc.running_kl():.5f}", flush=True)
        print(f"rows {rows[0]}-{rows[-1]} done ({time.time() - t_start:.0f}s): running KL {sc.running_kl():.5f}",
              flush=True)
    for s in range(batch):
        eng.reset(s)

    res, tok = sc.result(n_rows, seq)
    result = {
        "ref": args.ref_dir, "model": args.tqf, "sim_kv": None, "sim_w": None, "sim_embed": None, "sim_kv_layer": [],
        "engine": {"lib": args.lib, "tqf": args.tqf, "env": env, "decode_from": P, "batch": batch,
                   "head_cols": args.head_cols, "page": PAGE, "wave_max": wave_max, "seed_argmax_mismatch": seed_mismatch},
        "tokens": int(n_rows * seq), "seq": seq, "seed": args.seed, "rows": n_rows,
        "corpus": corpus_fingerprint(args.corpus, corpus),
    }
    result.update(res)
    result["seconds"] = time.time() - t_start
    print(json.dumps({k: v for k, v in result.items() if k != "by_ref_confidence"}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(tok, out[:-5] + ".pt")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ref", "score"):
        q = sub.add_parser(name)
        q.add_argument("--ref-dir", required=True, help="the original bf16 HF checkpoint dir (Qwen/Qwen3.8-27B)")
        q.add_argument("--corpus", required=True, help="held-out code tree (docs/QUALITY.md: a vLLM checkout's vllm/ dir)")
        q.add_argument("--out-dir", required=True, help="run directory: <tag>.json / .pt here, reference states in refstates/")
        q.add_argument("--tokenizer", default=None, help="HF dir for the tokenizer (default: --ref-dir)")
        q.add_argument("--exllamav3", default=None, help="path of an exllamav3 source checkout to import from")
        q.add_argument("--seq", type=int, default=4096)
        q.add_argument("--seed", type=int, default=12345)
        q.add_argument("--tokens", type=int, default=200_000)
        if name == "ref":
            q.add_argument("--states-on-cpu", action="store_true")
        else:
            q.add_argument("--tqf", required=True)
            q.add_argument("--lib", default=os.path.join(REPO, "build-qwen", "libforward_qwen.so"))
            q.add_argument("--decode-from", type=int, required=True,
                           help="positions >= this are teacher-forced decode steps (>= seq: prefill only, 0: decode only)")
            q.add_argument("--batch", type=int, default=8, help="rows decoded in lockstep (decode columns per step)")
            q.add_argument("--head-cols", type=int, default=8, help="columns per lm_head call on prefill waves")
            q.add_argument("--wave-max", type=int, default=WAVE_MAX, help="columns per prefill wave (256 needs TQ_WAVE_MAX=256)")
            q.add_argument("--rows", type=int, default=0, help="first N rows only (smoke tests)")
            q.add_argument("--tag", default=None)
            q.add_argument("--out", default=None)
            q.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.cmd == "score" and not (args.tag or args.out):
        p.error("score needs --tag or --out")
    add_exllamav3_path(args.exllamav3)
    (cmd_ref if args.cmd == "ref" else cmd_score)(args)


if __name__ == "__main__":
    main()
