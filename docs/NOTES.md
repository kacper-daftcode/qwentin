# Notes — the 2026-10 build: where the memory went, what was taken from DeepSeek-V4.1, how the served stack performs

These were the front pages of the 2026-10-17 README. At v0.10 the README became a release page (three lines of pitch,
one table, the reproduction) and the narrative moved here, **as written on 2026-10-17 for engine `2cf671c`** — the
2026-10 served stack: EXL3 opt-5.75 weights + the 4-bit trellis-coded KV + speculative decoding on the batched server.

One number changed after these notes were written: the gated-DeltaNet prefill kernels (`TQ_GDN_PREFILL=1`, engine
`b2e7a3f`, landed 2026-10-17 16:45 UTC) took the cold time to first token from 15.9 / 86.6 s to **13.3 / 77.0 s** at
32K / 128K (`docs/evidence/cand8-all.log` gate 3). Every TTFT quoted below is the earlier value; every other number is
unchanged — decode runs the same kernels bit for bit (gate 1 of that log). The table pinned at v0.10 is the one in
the top-level [README](../README.md); the cell-by-cell provenance is [`docs/evidence/README.md`](evidence/README.md).

Sections: the TL;DR (the memory accounting on one 32 GB card, the speed side, what was taken from DeepSeek-V4.1) ·
the summary block and its sources · why it's interesting (the full comparison table with the 2026-08 FP6 column, read
honestly) · the quality protocol in short · the performance of the served stack (the speed table, the mixed load, the
lever accounting).

## TL;DR — what is new in the 2026-10 build, and where the memory went

**A 27B model with 576K tokens of live context on one 32 GB consumer GPU, at FP8 quality, with
speculative decoding on every stream.** The official FP8 checkpoint of this model does not even
fit the card (it OOMs after loading 27.5 GiB of weights); stock vLLM with NVFP4 weights holds
147–179K tokens on it at an order of magnitude more KL. Same card, same model — this is how the
memory was found:

| | bf16 / stock | qwentin 2026-10 | how |
|---|---|---|---|
| Weights | 54 GB bf16 · 27.5 GiB FP8 (does not fit) | **18.5 GiB at FP8 quality** | EXL3 **trellis-coded weights** (5.75 bpw, exllamav3 checkpoint) decoded straight on the tensor cores; 6-bit lm_head; the embedding table lives in host RAM |
| KV cache | ~70 KiB/token bf16 (~35 fp8) | **16.5 KiB/token** | our own 4-bit **trellis-coded KV cache**: a 12-bit-state bitshift trellis with a Lloyd-trained codebook, encoded by a 5.8 M-tiles/s CUDA Viterbi during prefill, decoded inside the attention kernel — passes the FP8 gate where plain int4 / FP4 K+V does not |
| Recurrent state, per session (48 gated-DeltaNet layers) | 155 MiB fp32 (vLLM: 163 MB per request) | **80 MiB** | fp16 with deterministic stochastic rounding — plain fp16 drifts over a long decode, bf16 fails the gate; this one is indistinguishable from fp32 at 64K |
| Idle sessions | resident in VRAM | **parked in host RAM** at 37–45 GB/s | a pinned staging ring; a 32K session leaves or returns in ~16–19 ms, so a follow-up turn costs ~nothing |
| Speculative drafter | 0.85 GB bf16 MTP head | **0.25 GB** | the model's own multi-token-prediction head, EXL3-quantized at 4 bpw; the verified output is bit for bit the plain decode's |
| **Total** | | **29.9 GB for 12 sessions, each up to 256K** | |

<sub>Provenance: engine `2cf671c` (`src/forward_qwen.cu`; later commits are docs / tools), library `build-qwen/libforward_qwen.so` md5 `e454a29c3caf`, CUDA 13.1 (nvcc `cuda_13.1.r13.1/compiler.37061995_0`), driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0, measured 2026-10-17 — `docs/evidence/cand7-all.log` gate 3 (29 894 MiB; the startup line's 16.50 KiB/token and 80 MiB/slot); the bf16 / stock column: `docs/evidence/baselines-vllm-20261017.md`; the parking GB/s and the drafter's 0.25 GB: `ASSUMPTIONS.md` of the research repo. Cell-by-cell map: `docs/evidence/README.md`.</sub>

The speed side came from the same toolbox: speculative decoding through the paged batched server
(2.6× per stream), fp16 staged trellis attention, the decode step as one CUDA graph with concurrent
GEMM groups, 128-token prefill GEMM units — **167 tok/s per stream at 32K, 133 at 128K, 12 sessions
in flight.** The one place stock vLLM wins is cold prefill (3–5× lower time to first token, see the
table below) — that is the next target.

**What we took from DeepSeek-V4.1.** We read the V4.1-Flash report as a recipe for VRAM-cheap long
context and tested, with numbers, what transfers to a dense hybrid Qwen *without retraining it*:
hybrid linear / full attention (Qwen3.8 already has it — 48 gated-DeltaNet + 16 full-attention
layers — and the engine implements both), multi-token prediction (shipped, through the model's own
MTP head), a low-rank KV latent (MLA-style — measured: a zero-shot or distilled retrofit does not
reach FP8 quality below rank 1536, so it needs training), cross-layer KV reuse (same), and a trained
sparse-attention indexer (the page-selection kernels are in the engine; the indexer is a training
project). Every pilot is in the research repo with its numbers; the part that pays without training
is the trellis toolbox above.

## The 2026-10-17 summary block and its sources

(The TTFT line predates the DeltaNet prefill lever — v0.10 reads 13.3 s @32K · 77.0 s @128K, `docs/evidence/cand8-all.log`.)

```
27B at FP8 quality, 12 sessions        →  one 32 GB RTX 5090: 576K pooled tokens (4500 × 128), 29 894 MiB
decode per stream @32K, N = 1 / 4 / 8  →  167 / 109 / 81 tok/s  (65 / 56 / 47 without speculative decoding)
decode per stream @128K, N = 1 / 2     →  133 / 111 tok/s
time to first token, cold              →  15.9 s @32K · 86.6 s @128K   (~2000 / ~1500 prompt tok/s)
quality vs the bf16 original           →  KL 0.0022–0.0027, top-1 99.0–99.1 %  (the FP8 checkpoint: 0.0037 / 98.7 %)
16 GB card (4.25 bpw, 2 slots)         →  51K tokens at 16 006 MiB, 184 tok/s @32K, needle 3/3 to 53K
```

Sources for every number: the gate log of the exact served line (`logs/cand7-all.log`), the
speed table (`logs/final-speed.log`), the quality yardstick (`logs/final-e2e.log`), the mixed
prefill/decode load (`logs/final-t5.log`) and the 16 GB check (`logs/p16-check.log`) in the
research repository (`kacper-daftcode/qwentin-internal`, `ASSUMPTIONS.md` and `notes/`) — copied
verbatim into [`docs/evidence/`](evidence/README.md) of this repo together with the scripts
that produced them, the vLLM baseline note and a cell-by-cell map from each README number to its
log line. Every table below carries a one-line provenance footer (engine commit, library md5,
toolchain, date, log).

## Why it's interesting

The FP8 checkpoint of a dense 27B does not fit one 32 GB card at all (29 GB of weights before
any KV cache; our FP8 reference server — vLLM, the T1 numbers below — ran tensor-parallel on two
5090s). On one 5090 qwentin now
serves the 27B **at FP8 quality**, with **576K tokens of KV cache** shared by 12 sessions, and
with speculative decoding on every stream — where the 2026-08 build held 196K pooled tokens at
FP6 quality.

**One 32 GB RTX 5090, 12 sessions** — ours and stock vLLM measured on the same cards on
2026-10-17 (vLLM `0.32.0rc1.dev281`, nightly image of 2026-10-16, prefix caching on; the NVFP4
checkpoint is a llm-compressor W4A4 export of the bf16 model, 512 × 2048 calibration; the
official FP8 checkpoint **does not fit one 5090** — it OOMs after loading 27.5 GiB of weights even
at `--max-model-len 4096` — so its row is tensor-parallel on **two** cards; full tables, commands
and caveats in `notes/baselines-vllm-20261017.md` of the research repo):

| | **qwentin 2026-10** (EXL3 opt-5.75 + trellis KV + spec), 1 card | vLLM NVFP4, 1 card (bf16 KV / `--kv-cache-dtype fp8`) | vLLM FP8, **2 cards** (TP2, bf16 KV) | qwentin 2026-08 (FP6, previous README), 1 card |
|---|---|---|---|---|
| Weights on the device | EXL3 trellis 5.75 bpw, ~18.5 GiB | NVFP4 4.5 bpw, 17.6 GiB (4.7 GiB of it bf16 embed + lm_head) | FP8, 13.8 GiB per card | FP6 E2M3, ~20 GiB |
| KV cache | 4-bit trellis-coded K+V, **16.5 KiB/token** | bf16 K+V (832-token blocks) / fp8 | bf16 K+V | int4 K + E4M3 V, 26.2 KiB/token |
| Max context accepted | **262 144, 12 slots** | 131 072 (262 144 rejected: "est. max 147 264") / 262 144 | 262 144 | 131 072 pooled |
| Pooled KV capacity | **576K tokens** (`--num-blocks 4500`), 29 894 MiB total | **147–179K** (the same command gave both, cold vs warm compile cache) / 283K | 493K **across two cards** | 196K |
| Sessions @128K | 4.4 | 1.1–1.4 / ~2.1 | 3.7 (two cards) | ~1.5 |
| Quality vs the bf16 original (KL / top-1; protocol below) | **0.0022–0.0027 / 99.0–99.1 %** — the FP8 checkpoint's own 0.0037 / 98.7 % or better | NVFP4 PTQ: ~3.5× the KL of the 4.25-bpw EXL3 model in the project's fake-quant yardstick (≈ 0.04); 2026-08 tf-top1 85.8 on the previous ladder (FP8 95.9, FP6 91.3) | = the FP8 checkpoint (0.0037 / 98.7 %) | 0.0139 / 97.4 % |
| Decode per stream @32K, N = 1 / 4 / 8 | **167 / 109 / 81 tok/s** (65 / 56 / 47 with spec decode off) | 71.5 / 51.4 / — (N = 8 does not fit the pool) / 75.3 / 58.8 / — | 76.8 / 60.3 / 50.4 (vLLM's own MTP spec path: 56 / 55 / 50 — slower than without) | 142 (N = 1, single-stream server) |
| Decode per stream @128K, N = 1 | **133 tok/s** | 55.5 / 66.0 | 65.8 | 123 |
| Time to first token, cold, 32K / 128K | 15.9 s / 86.6 s (~2 000 / ~1 500 prompt tok/s) | **3.1 s / 25.2 s** / 3.3 s / 30.0 s (~10 000 prompt tok/s) | **4.4 s / 24.1 s** | 18.7 s / 116 s |
| Needle-in-a-repository (3 depths, 3 decoys) | 6/6 at 32K and 128K | 6/6 | 6/6 | 4/4 @239k |
| Decode under a concurrent long prefill (8-session mix) | 40 / 25 / 43 tok/s per client (32K / 96K / 160K classes) | — | — | — |

<sub>Provenance: qwentin column — engine `2cf671c`, library md5 `e454a29c3caf` (`docs/evidence/cand7-all.log`) and `c9d5269c2963` run with the `2cf671c` defaults as flags (`docs/evidence/final-speed.log` (R), `final-t5.log` g1, `final-e2e.log`; bitwise the same path, cand7 gate 1), CUDA 13.1 (nvcc `cuda_13.1.r13.1/compiler.37061995_0`), driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0, 2026-10-17; vLLM columns — `docs/evidence/baselines-vllm-20261017.md` (vLLM `0.32.0rc1.dev281`, image `nightly-49cb2bbf…`, same cards, same day); quality cells — `docs/evidence/e2e/*.json` via `tools/e2e_frontier.py`; the 2026-08 column — the 2026-08 README (no 2026-10 log). **Speed with speculative decoding depends on the text** (the accepted draft length — 3.4–3.6 tokens per round on the vLLM-repository corpus these numbers use, 2.7 on this repository's own code, where `bench/reproduce.sh` measured 129 / 72 tok/s at N = 1 / 8 with spec and the same 65 / 48 without); TTFT and spec-off decode do not depend on it. Run `bench/reproduce.sh` with `CORPUS=<a vllm checkout>` for like-for-like numbers.</sub>

Reading the table honestly: on one card qwentin holds **2–3.9× the context** of stock vLLM at
**FP8 quality instead of NVFP4**, decodes **2.2× faster per stream** with speculative decoding
(equal without it: the trellis GEMV is as memory-bound as the FP4 one), and is the only one of
the three that fits 12 sessions or the FP8-class quality on a 32 GB card at all. **vLLM's clear
win is prefill**: 3–5× lower time to first token (FlashInfer gated-DeltaNet prefill + CUTLASS NVFP4
GEMMs on the FP4 tensor cores vs our trellis-decode-to-fp16 GEMM, chunkwise DeltaNet and wide
attention kernels at ~2 000 prompt tok/s) — that is the next engineering target. Two caveats
from the measurement: vLLM's KV budget for this hybrid model was not deterministic across
startups (same command, 147K vs 179K tokens), and the NVFP4 export needed one repair (a shared
global scale for the fused gated-DeltaNet `in_proj_qkv` / `in_proj_z`, which vLLM fuses and
llm-compressor scales separately); needle 6/6 passed either way, so needle is not a sensitive
quality probe — the KL yardstick is.

Versus the 2026-08 build ([`legacy-fp6.md` at v0.11](https://github.com/kacper-daftcode/qwentin/blob/v0.11/docs/legacy-fp6.md)): pooled tokens 196K → 576K, quality FP6 → FP8-level,
single-stream decode 62 → 167 tok/s on the batched server through speculative decoding
(2.5–2.7×, outputs bit-for-bit plain decode's), and the pre-2026-10 stack is still one flag
away (`--no-kv-t4 --no-spec`).

### The quality protocol

(The full protocol — tokens, tools, gate rule, reference rows, what is not measured — is
[`docs/QUALITY.md`](QUALITY.md); the tools are `tools/qwentin_e2e.py`, `tools/exl3_diff.py`,
`tools/e2e_frontier.py`.) Every quality number above comes from one yardstick, run on the engine itself (not a
simulation): **197K held-out code tokens** (files of a large Python/C++ repository — the `vllm/` package of a
pinned vLLM checkout, 48 × 4096
rows for the "4K" cells, 6 × 32768 rows for the "32K" cells), scored teacher-forced against the
**original bf16 `Qwen/Qwen3.8-27B`**. **KL** = mean per-token KL(bf16 ‖ candidate) over the
next-token distributions; **top-1** = the share of positions where the candidate's argmax equals
the bf16 argmax. "FP8 quality" means: not worse than the official FP8 checkpoint
(`Qwen/Qwen3.8-27B-FP8`, as vLLM serves it with a bf16 KV cache) on both, within the 95 %
block-bootstrap confidence interval, at 4K and at 32–64K. The FP8 checkpoint itself scores
**0.0037 / 98.73 %** at 4K and 0.0030 / 98.92 % · 0.0038 / 98.81 % on the two 32K samples; the
served stack scores 0.0024 / 98.98 % (4K, prefill path), 0.0022 / 99.08 % (32K prefill),
0.0027 / 98.96 % (32K decode path) and 0.0030 at 4K on the decode path (`logs/final-e2e.log`,
`notes/session10-b2.md`). **T1** = EvalPlus HumanEval+ / MBPP+ greedy through the server:
HumanEval/+ 93.9 / 91.5 · MBPP/+ 89.7 / 77.5 for the trellis stack (12 slots, spec off — spec is
bitwise plain decode) vs 93.9 / 90.9 · 90.2 / 77.0 for the FP8 reference — equal within T1's ±1–2 pt noise (T1 is too coarse to separate
bitrates; KL / top-1 is the gate, T1 the sanity check). The "tf-top1" ladder of the previous
README (FP8 95.94 > FP6 91.30 > E2M1 86.46 ≈ NVFP4 85.78) is a different protocol on a different
token set and has not been re-run on the new stack. Speculative decoding is lossless by
construction here: the verify wave is the decode step's own kernels, and the committed tokens
are bitwise plain decode's at the same client count (`tools/spec_paged_check.py`,
`tools/spec_server_ab.py` 10/10 identical) — with one historical exception: from 2026-10-16 to
2026-10-21 a wave that straddled position 384 (the first key-split change) could NaN its first
column, after which the stream emitted token 0; the gates' page-aligned 4096 / 32768-token contexts
and 1.5K-token A/B prompts never crossed it, T1 with spec on did (see CHANGELOG, `cand23-all.log`).
T1 is now also run through the served line with speculative decoding on.

## Performance of the served stack (2026-10-17, main `2cf671c`, CUDA 13.1, driver 610, RTX 5090 at 550 W)

Per-stream decode tok/s (p50 over N streams, 300 tokens each, code prompts, spec on / off),
through the HTTP server with the exact line in the README's "Quick start" (`tools/spec_speed.py`;
`logs/final-speed.log` rows "(R)", `logs/cand7-all.log` gate 3):

| Context | N = 1 | N = 2 | N = 4 | N = 8 | aggregate @N = 8 |
|--------:|------:|------:|------:|------:|-----------------:|
| 32K, spec on  | **167** | **143** | **109** | **81** | 619–632 tok/s |
| 32K, spec off | 65 | 62 | 56 | 47 | 377 tok/s |
| 128K, spec on | **133** | **111** | | | |

<sub>Provenance: engine `2cf671c`, library md5 `e454a29c3caf` (`docs/evidence/cand7-all.log` gate 3: 166.9 / 80.6 tok/s @32K N = 1 / 8, 132.8 @128K — the N = 1, N = 8 and 128K N = 1 cells) and `c9d5269c2963` with the `2cf671c` defaults as flags (`docs/evidence/final-speed.log` block (R): every other cell, 165.3 / 143.0 / 109.0 / 78.8 on, 65.0 / 61.8 / 56.0 / 47.0 off, 131.7 / 111.4 @128K), CUDA 13.1 (nvcc `cuda_13.1.r13.1/compiler.37061995_0`), driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0, 2026-10-17.</sub>

~3.4 accepted tokens per speculative round at K = 4 (10 prompts × 300 tokens: 121 vs 59 tok/s
per request, outputs 10/10 identical with spec off). Cold prefill = time to first token through
the server: **15.9 s at 32K** (32 596 tokens) and **86.6 s at 128K** (130 905 tokens), needle
6/6 on the same requests. Steady-state process memory **29 894 MiB** with 4500 blocks = 576 000
pooled tokens; the startup line reads `trellis-4 K + V ... 16.50 KiB/token, 576000 pooled tokens;
DeltaNet state fp16sr 80 MiB/slot; speculative decoding ON (MTP drafter, K=auto)`.

Under a mixed load (`eval/loadgen.py` in the research repo: 8 sessions with 32K / 96K / 160K
contexts, 3 concurrent, 3 turns each, ~1M prompt tokens; `logs/final-t5.log` run g1): 24/24 requests OK,
cold TTFT p50 16 / 97 / 365 s for the 32K / 96K / 160K classes, follow-up TTFT 3–5 s (prefix cache
+ parking: 16/16 prefix hits, 5 sessions parked to host RAM), decode **40 / 25 / 43 tok/s per
client while a prefill was in flight** (95 % of the wall), 39 / 40 / 35 on follow-up turns — vs
13 / 8 / 11 with `--decode-steps-per-wave 0`.

Decode-step cost on the bare engine, for the lever accounting (ms per step at 4K / 32K, graph +
fork on, the 10-16 bf16 trellis attention kernels): 14.75 / 15.4 at N = 1, 16.9 / 21.5 at N = 8;
the fp16 staged attention alone measured 22.1 → 20.9 ms at 32K N = 8 on its branch (trellis
attention 335 → 277 µs per layer; the FP4 K+V kernels: 249) (`ASSUMPTIONS.md` "2c", "B2-eng,
sixth part").

What each landed lever bought on this line (each measured against the previous default on the
same binary; `ASSUMPTIONS.md` "The landed served library, measured for the decision list"):
trellis KV +39 % pooled tokens over the 10-13 served stack (opt-5.75 + trellis 589K vs
opt-6.25 + FP4 V 422K; at equal weights 576K vs 496K); `--spec` 2.5–2.7× per stream at N = 1;
`--dstate fp16sr` +400 blocks (525K → 576K) and −0.38 ms per step at 32K N = 8;
`TQ_T4_ATTN_F16=1` −1.0 ms per step at 32K N = 8 (the opt-in set, mostly this lever: +20–28 %
at 128K); `TQ_GRAPH=1 TQ_GEMM_FORK=1` −0.35 ms per step everywhere; prefill GEMM units −5 % per
wave.
