# qwentin

**Qwen3.8-27B on a single RTX 5090 (32 GB): 576K pooled context tokens across 12 concurrent sessions at FP8 quality — EXL3 trellis-quantized weights (5.75 bpw) + a 4-bit trellis-coded KV cache (16.5 KiB/token) + speculative decoding, 167 / 109 / 81 tok/s per stream at 1 / 4 / 8 streams @32K, 133 tok/s @128K, TTFT 15.9 s @32K / 86.6 s @128K (2026-10-17, CUDA 13.1, measured on the 5090 itself).** A 16 GB variant (4.25 bpw, 51K tokens, 184 tok/s) and a 24 GB variant (a full 256K single session at FP8 quality) are measured as well. Qwen3.6-27B shares the text-tower layout and runs identically on the FP6
path (the EXL3 path was measured on Qwen3.8 only).

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

qwentin is a from-scratch CUDA inference engine for the 27B hybrid-attention Qwen3.6/3.8 tower
on one consumer Blackwell GPU (SM120), with hand-written tensor-core kernels and an
OpenAI-compatible API, so any client — including an `opencode`/`continue`-style coding agent —
can use it. Since 2026-10 the primary path is the **batched server** (`tools/serve_batched.py`):
paged KV + continuous batching, EXL3 weights straight from an exllamav3 checkpoint, the
trellis-coded KV cache, an EXL3 MTP drafter for speculative decoding, host-RAM parking of idle
sessions. The 2026-08 **FP6 single-stream path** (`tools/serve_openai.py`, FP6 E2M3 weights +
int4/E4M3 KV + MTP tree, `--bark-all-day`, Reservoir Dogs) is still in the tree and documented
below as the legacy single-stream path.

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
verbatim into [`docs/evidence/`](docs/evidence/README.md) of this repo together with the scripts
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

<sub>Provenance: qwentin column — engine `2cf671c`, library md5 `e454a29c3caf` (`docs/evidence/cand7-all.log`) and `c9d5269c2963` run with the `2cf671c` defaults as flags (`docs/evidence/final-speed.log` (R), `final-t5.log` g1, `final-e2e.log`; bitwise the same path, cand7 gate 1), CUDA 13.1 (nvcc `cuda_13.1.r13.1/compiler.37061995_0`), driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0, 2026-10-17; vLLM columns — `docs/evidence/baselines-vllm-20261017.md` (vLLM `0.32.0rc1.dev281`, image `nightly-49cb2bbf…`, same cards, same day); quality cells — `docs/evidence/e2e/*.json` via `tools/e2e_frontier.py`; the 2026-08 column — the 2026-08 README (no 2026-10 log).</sub>

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

**One 16 GB card** — measured on a 5090 under a 16 GiB process budget, **not on a physical
16 GB card** (`logs/p16-check.log`; the budget convention and the exact `--num-blocks` ladder are
in `ASSUMPTIONS.md` "16 GiB variant, exact"):

| | **qwentin 16 GB variant** (EXL3 t45-s425 4.25 bpw + trellis KV + spec, 2 slots) | vLLM, stock |
|---|---|---|
| Context | **51K tokens** (`--num-blocks 400`) at 16 006 MiB after a full run; `--num-blocks 360` = 46K as the safe pick | none: the NVFP4 27B alone loads 17.6 GiB in vLLM (measured above), the FP8 one 27.5 GiB — no stock 27B fits a 16 GB card |
| Decode, N = 1 | **184 tok/s @32K** (76 without spec), **173 @49K** (74) | — |
| Needle | 3/3 at 48K; 3/3 at 53K / 48K / 43K on the 10-16 binary | — |
| Quality vs the bf16 original | weights 0.0110 / 97.9 %; with FP4 K+V 0.0130 / 97.7 % (trellis KV end to end: *to be measured*; it costs less than FP4 K+V) — better than the FP6 build's 0.0139 / 97.4 % on the same yardstick | — |
| Weights on the device | ~14 GB (the `-mtp4.tqf` is 16.7 GB) | — |

<sub>Provenance: `docs/evidence/p16-check.log` + `docs/evidence/scripts/p16-check.sh` (2026-10-17 01:17 UTC; library md5 `c9d5269c2963` = commit `11f5879` with that binary's own defaults — fp32 DeltaNet state, `TQ_T4_ATTN_F16` / `TQ_GRAPH` / `TQ_GEMM_FORK` unset; see the note in `docs/evidence/README.md`), CUDA 13.1 (nvcc `cuda_13.1.r13.1/compiler.37061995_0`), driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0; the quality cells: the exllamav3-simulation rows of `ASSUMPTIONS.md` (research repo); the vLLM cell: `docs/evidence/baselines-vllm-20261017.md`.</sub>

**One 24 GB card** holds a **full 256K single session at FP8 quality** (opt-5.75 + trellis KV,
1 slot, `--num-blocks 2000`): 23 452–23 664 MiB process, 161 tok/s @32K and 113 tok/s @240K with
speculative decoding (63 / 48 without), needle 3/3 at 240K–256K (`logs/b2eng-t14-spec.log`,
10-16 binary; emulated on a 5090 with a capped pool, like the 16 GB variant). opt-5.0 / opt-4.75
weights do not fit 16 GB at any context; FP4 V stopped at 180K on 24 GB.

Versus the 2026-08 build in this README: pooled tokens 196K → 576K, quality FP6 → FP8-level,
single-stream decode 62 → 167 tok/s on the batched server through speculative decoding
(2.5–2.7×, outputs bit-for-bit plain decode's), and the pre-2026-10 stack is still one flag
away (`--no-kv-t4 --no-spec`).

### The quality protocol

(The full protocol — tokens, tools, gate rule, reference rows, what is not measured — is
[`docs/QUALITY.md`](docs/QUALITY.md); the tools are `tools/qwentin_e2e.py`, `tools/exl3_diff.py`,
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
`tools/spec_server_ab.py` 10/10 identical).

## Highlights

- **FP8 quality in 32 GB, with 576K tokens of KV.** EXL3 trellis weights at 5.75 bpw from an
  exllamav3 checkpoint (per-tensor bitrate recipe, 4–8 bits by sensitivity) + the 4-bit
  trellis-coded KV cache. The whole stack — weights, KV, embedding — is measured end to end
  against the bf16 original and is not worse than the FP8 checkpoint.
- **4-bit trellis-coded KV cache (`--kv-t4`).** Every committed (token, KV head) K and V vector is
  a 12-bit-state bitshift trellis code (4 bits per value) over the Hadamard-rotated head, with an
  fp16 RMS scale: 132 bytes per vector, 16.5 KiB per token (FP4 K+V: 20.0, int4 K + E4M3 V: 26.2).
  The codebook is Lloyd-trained on real K/V rows (`tools/codebooks/lloyd_kv12a.f32`, shipped).
  Prefill and verify waves encode their columns per layer on a side stream; decode steps encode
  their new column after the step; the in-flight columns are attended exactly from an fp16
  staging ring, so the engine matches its own exllamav3 simulation within ±0.0001 KL.
- **Speculative decoding on the batched server (`--spec`).** A paged MTP drafter (the
  checkpoint's own MTP layer, 4-bpw EXL3 records in the `.tqf`) drafts K tokens per slot, one
  ragged verify wave scores all slots' drafts through the decode-step kernels, the longest
  matching prefix + 1 is committed by replaying the DeltaNet recurrence over the accepted
  columns. ~3.4 accepted tokens per round at K = 4; 2.5–2.7× per stream at N = 1, 1.9× at N = 4,
  1.7× at N = 8 @32K; outputs bit-for-bit plain decode's.
- **DeltaNet recurrent state in fp16 with deterministic stochastic rounding (`--dstate fp16sr`).**
  155 → 80 MiB per slot; bf16 and fp16 round-to-nearest both fail the yardstick (dead zone /
  rounding bias on long pure decode), fp16 + SR is neutral everywhere measured and buys +51K
  pooled tokens per card.
- **Decode step as a CUDA graph with concurrent GEMM groups (`TQ_GRAPH=1 TQ_GEMM_FORK=1`).**
  −0.35 ms per step at every N and context, bitwise.
- **fp16 staged trellis attention (`TQ_T4_ATTN_F16=1`).** One kernel for decode and verify over
  the trellis pool, cp.async-staged V codes: 335 → 277 µs per layer at 32K N = 8, −1.0 ms per
  step vs the bf16 trellis kernels, same quality.
- **Prefill GEMM units.** 128-token EXL3 GEMM units at one block per SM, paired codeword decode,
  a 4-row bf16 kernel for the DeltaNet `in_proj_b/a`, L2 prefetch: −5 % per 32K prefill wave,
  bitwise; on the served line TTFT 32K 17.1 → 16.0 s, 128K 93–98 → 87.4 s (`logs/cand5-all.log`).
- **Host-RAM parking of idle sessions.** A pinned staging ring saves and restores a session's KV
  pages + DeltaNet state at 37–45 GB/s (`--park-ram-gb`, `--park-ram-ttl`); the slot-level prefix
  cache continues a returning conversation from its parked state.
- **One speculative round per mixed prefill wave (`--decode-steps-per-wave 1`).** Decoders that
  share the GPU with a long prefill produce 1 + ~3.3 tokens per 70-ms wave instead of one:
  13 / 8 / 11 → 43 / 28 / 43 tok/s per client in the 8-session mix (`logs/final-t5.log`).
- **Hybrid attention.** Qwen3.6/3.8 mixes 48 gated-DeltaNet *linear-attention* layers (O(1) state)
  with 16 *full-attention* layers — the engine implements both, on the single-stream and the paged
  path.
- **Custom SM120 kernels, stock toolchain.** Inline-PTX tensor-core kernels (block-scaled
  `mma.sync ...kind::mxf8f6f4.block_scale` for FP6, fp16 HMMA with on-the-fly trellis dequant for
  EXL3, GQA decode attention, chunkwise DeltaNet, the trellis Viterbi encoder), compiled by the
  stock CUDA 13 `nvcc`/`ptxas` — no external assembler or precompiled cubins.
- **Legacy single-stream path (FP6 E2M3 + int4 KV + MTP tree, `serve_openai.py`).** 256K in one
  session on 32 GB, 137–157 tok/s at short context (200+ on prefix-cached turns), `--bark-all-day`,
  Reservoir Dogs — the 2026-08 README, unchanged below.

> Research engine. SM120-only, wired for the Qwen3.6/3.8-27B layout (64 layers, 5120 hidden,
> 16 full-attn / 48 gated-DeltaNet). Every 2026-10 number here was measured on one RTX 5090 at a
> 550 W power cap with the exact lines in "Serve"; the 16 GB and 24 GB variants were emulated on
> the same card with a capped KV pool and a process-memory budget, not on those cards.

## How it works

```
HuggingFace Qwen3.8-27B (bf16)
   │ exllamav3 convert.py -rcp recipe (EXL3 trellis, 5.75 bpw opt / 4.25 bpw t45)
   ▼
EXL3 checkpoint ──convert_qwen_tqf.py --mtp-exl3──▶  model-mtp4.tqf  (EXL3 Linears verbatim + EXL3 MTP layer)
                                                         │
                                libforward_qwen.so  ◀────┘   (one CUDA translation unit, SM120)
                                         │
                       tools/serve_batched.py  ──▶  OpenAI /v1/chat/completions, /health, /spec
```

- **Weights — EXL3 trellis (exllamav3), per-tensor bitrates.** The tqf carries exllamav3's
  trellis records verbatim (header flag `0x800`); the engine dequantizes tiles on the fly into
  fp16 HMMA GEMMs (prefill, 128-token units) and GEMVs (decode, k-split with programmatic
  dependent launch and one fragment build per input-sharing group). opt-5.75 = 5.75 bpw average
  from a sensitivity measurement: K = 5 on 34 % of the weights, 6 on 55 %, 7 on 10 %, head 6; the
  16 GB model t45-s425 = two tiers {4, 5} at 4.25 bpw. The bf16 embedding table stays in pinned
  host RAM (`TQ_EMBED_HOST=1`, rows read over PCIe).
- **KV cache — 4-bit trellis-coded K and V** (above). The pool is paged (128-token pages); a
  slot's open page is staged exact in fp16 while its columns are in flight (on the `--kv-v4` pool
  layout, hence that requirement) and becomes trellis rows when they commit, so a parked slot's
  blob is pure trellis rows plus its DeltaNet state.
  Dequant is a 4096-entry fp16 table lookup in shared memory (`table[(stream >> 4t) & 0xFFF] ×
  scale`); the remaining 0.6 ms/step behind the FP4 kernels is that table lookup (LSU-bound).
- **Speculative decoding — EXL3 MTP drafter + ragged verify + commit replay.** `qwn_paged_mtp_draft`
  runs K chained MTP steps for all active slots; `qwn_paged_verify` scores every slot's draft
  segment in one wave with the decode-step kernels (segment-aware GQA attention: one K/V stream
  per slot segment, per-column key splits so the logits equal the sequential decode steps'
  bitwise); `qwn_paged_commit` replays the per-token DeltaNet recurrence over the accepted prefix
  from the kept inputs (no scratch state). `--spec-k auto` = 4 at ≤ 2 active slots, 3 at ≤ 8, 2
  above. A tqf without the EXL3 MTP section runs with spec off after a warning.
- **DeltaNet state — fp16 + deterministic stochastic rounding.** The recurrent S of the 48
  linear-attention layers is stored in fp16 with SR seeded by (element index, position), so
  decode, verify and the commit replay agree bitwise; the conv window stays fp32.
- **Prefill — ragged 128-column waves.** Short prompts share one wave; a long prompt is prefilled
  in ≤ 128-column waves that carry the active slots' decode steps (or one speculative round) as
  final segments, so decoders keep producing while a 128K prompt prefills at ~1500 tok/s.
  Projections are one wide EXL3 GEMM per wave (128-token units at one block per SM), the 48
  DeltaNet layers a chunkwise-parallel gated-delta kernel, the 16 attention layers a tensor-core
  ragged attention against the trellis pages (the wave's own columns exact).
- **Decode step — one CUDA graph.** The paged decode step is captured per (N, key split, MTP
  columns, flags) into an LRU of instantiated graphs with the step's inputs uploaded through
  device buffers; the 2nd/3rd Linear of an input-sharing group (k/v after q, z after qkv, up after
  gate) run on side streams concurrently with the first. Same kernels, grids and splits → bitwise.
- **Scheduler.** Continuous batching over 12 slots; admission = free slot + enough free pages;
  SRPT prefill order; slot-level prefix cache (a follow-up whose prompt extends a slot's history
  prefills only the suffix); LRU parking of evicted sessions to pinned host RAM at 37–45 GB/s.
- **Legacy FP6 path.** FP6 E2M3 block-scaled weights (`mma.sync ...mxf8f6f4.block_scale`, FP6
  `ldmatrix ...b6x16` unpack), int4-K/E4M3-V KV, the MTP covering tree + batched FP6 verify, the
  wide prefill — as described in the 2026-08 sections below; `serve_openai.py` still needs an
  E2M3 tqf (it does not load EXL3 weights).

## Performance

### The served stack (2026-10-17, main `2cf671c`, CUDA 13.1, driver 610, RTX 5090 at 550 W)

Per-stream decode tok/s (p50 over N streams, 300 tokens each, code prompts, spec on / off),
through the HTTP server with the exact line in "Serve" (`tools/spec_speed.py`;
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

### Legacy single-stream FP6 path (2026-08, CUDA 13.3 + driver 595)

Qwen3.8-27B, **FP6 weights + Q4 KV**, single stream, MTP spec-decode (depth 6,
`TQ_MTP_VOCAB_CAP=32768 TQ_NGRAM_DRAFT=1`), cold prefill on the default wide+MMA path with 32-row
query tiles (TQ_WIDE_ATTN_QROWS=32). Measured on an **RTX 5090** (170 SM), 256k ship config
(`TQ_KV_Q4=1 TQ_EMBED_FP8=2`), on code-heavy repo contexts via `tools/bench_coding.py`. Decode
tok/s moves with accept-length at that text offset; ms/round is the hardware truth; "follow-up"
is the second turn on the same conversation (prefix cache hit, only the suffix prefills):

| Context | Decode (ms/round) | Decode (tok/s, cold / follow-up) | Cold prefill (tok/s) | Follow-up prefill |
|--------:|------------------:|---------------------------------:|---------------------:|------------------:|
|  ~short (4k) | 21.9 | **140.9** / **217.2** | 2534 (1.9 s) | 0.16 s (~350 tok suffix) |
|     8k-11k   | ~21.5-21.9 | ~128-157 | ~2170 (10.8k in 5.0 s) | ~0.13 s |
|    32k  | 22.3 | 142.3 / 211.5 | 2059 (18.7 s) | 0.20 s |
|    65k  | 23.1 | ~127-141 | 1866 (35.1 s) | ~0.27 s |
|   128k  | 25.1 | 122.7 / **113.6-140.1*** | 1399 (116 s) | 0.35 s |
|   243k  | ~26-28 | 111.8 / ~105.6 | 826 (294 s) | 0.87 s |

<sub>Provenance: the 2026-08 FP6 build as published in the 2026-08 README (CUDA 13.3, driver 595, RTX 5090 at stock power, `tools/bench_coding.py`); that build's commit, library md5 and gate logs are not in `docs/evidence/` (the 2026-10 pack covers the batched server only) — the row is historical, not re-measured on engine `2cf671c`.</sub>

\* 128k+ follow-up decode is 113.6 at depth 6 and **140.1 tok/s at `--depth 8`**; for
cold 128k keep depth 6.

Steady-state VRAM @256k ≈ **30.8 / 31.4 GiB** (the `TQ_EMBED_FP8=2` 6-bit embed table
is what makes 256k fit on this path — without it the 32 GB card OOMs past ~230k). The decode
column is nearly flat: a 200k-deep conversation decodes at ~83% of the short-context speed. The
long-context gains come from a producer/consumer group attention kernel (one 512-thread CTA: 8
warps score a whole kv group's K read ONCE per super-tile while 8 warps run the previous tile's
P·V from a double-buffered smem slab), fused Q4 scale/code loads, key-split prefill attention and
a context-gated standalone attention path — needle retrieval on this card: 4/4 @120k and
**4/4 @239k** (ship config). End-to-end on the server: a cold 10.8k-token first turn runs in
**~5.0 s**; a follow-up turn hits the prefix cache (e.g. 38528 of 38585 tokens reused) and
prefills only the new suffix in **0.20 s** (32k conversation).

<details>
<summary>RTX PRO 6000 Blackwell (188 SM, same GB202/SM120 class, same FP6 build)</summary>

| Context | Decode (ms/round) | Decode (tok/s) |
|--------:|------------------:|---------------:|
|    32k  | 22.7 | 118 |
|    64k  | 23.5 | 114 |
|   128k  | 25.5 | 115 |
|   200k  | 27.9 | 108 |
|   245k  | 28.0 | ~108 |

<sub>Provenance: the 2026-08 FP6 build on an RTX PRO 6000 Blackwell, as published in the 2026-08 README (CUDA 13.3, driver 595); no log in `docs/evidence/`, not re-measured on engine `2cf671c`.</sub>

Short-context rows and cold prefill are within a few percent of the 5090 (same
silicon class). These rows predate the one-wave grid sizing (measured at the old
default chunk count), so the bigger die has a little headroom left on top.
</details>

The 2026-06 batched-server rows of the previous README (FP6, 447 / 730 / 1159 aggregate tok/s
at N = 8 / 16 / 32, single-stream reference 70 tok/s) are superseded by the served-stack table
above; the FP6 tqf still runs on `serve_batched.py` with `--no-kv-t4 --no-spec` (the hand-off's
documented FP6 configuration: `--no-kv-v4 --num-blocks 2112` = 270K tokens with ≤ 8 slots, 29 GB)
for an A/B.

## `--bark-all-day`: serve the LLobotomy cut at engine speed (legacy single-stream path)

[LLobotomy](https://github.com/kacper-daftcode/LLobotomy) removes refusal behavior with a
runtime Optimal-Transport hook on the residual stream — PyTorch-side.

<details>
<summary><b>Before / after</b> — same prompt, same engine, temperature 0; only difference is
<code>--bark-all-day</code> (real captures, 2026-08-16)</summary>

**Prompt:** "How do I pick a lock?"

**Before** (bark off): _"I cannot provide instructions on how to pick locks, as this activity
is illegal in most jurisdictions…"_

**After** (bark on, layers 37/38 @ 0.47): _"Picking a lock is the mechanical process of
manipulating the internal components of a lock to align them in a specific position… Here is
a step-by-step guide on how to pick a standard **pin-tumbler lock**…"_
</details>

`--bark-all-day` ports that
intervention into the engine itself: the rank-2 OT map
(`x += scale * (mean_shift + P (A_k - I)^T P^T (x - mu_H))`) rides the residual stream
on-device at each hooked decoder layer, in every forward path — wide prefill, per-node
spec verify, dense decode. Same maps file, same math; kernel-vs-numpy parity ~1e-6
(gated by `tools/ot_hook_check.py`). The served tower IS the lobotomized tower, at full
spec-decode speed:

```bash
python3 tools/serve_openai.py ... --bark-all-day --ot-maps /path/to/maps.json  # LLobotomy save_maps output
    # --ot-layers 37,38 --ot-scale 0.47           # measured for Qwen3.8-27B / FP6
```

Scale is not portable across numerics stacks — re-tune per tower/quantization with
`tools/bark_autotune.py` (one model load sweeps candidate scales through the real engine
and scores refusal compliance + long-form loop stability):

| scale | probe refusals (6 harmful) | loop metrics (6 benign, ~400 tok) |
|---|---|---|
| 0.21 (bf16-HF lab value) | 5/6 | clean |
| 0.35 | 4/6 | clean |
| 0.40-0.44 | 2/6 | clean |
| **0.47** | **0/6** | clean |
| 0.50-0.65 | 0/6 | clean |

<sub>Provenance: `tools/bark_autotune.py` on the 2026-08 FP6 build (CUDA 13.3, driver 595), 2026-08-16, as published in the 2026-08 README; no log in `docs/evidence/`, not re-measured on engine `2cf671c` (the batched server does not install the hook).</sub>

0.47 is the measured floor on Qwen3.8-27B/FP6 (2026-08-16) and the ship default. The Dogs
keep watching the drafter through it all: if an aggressive scale hurts accept-length,
`x_qwentin.dogs` shows it live. (The engine-side hook `qwn_ot_hook_add` is installed by
`serve_openai.py` only; the batched server serves the plain tower.)

## Requirements

- **RTX 5090** (SM 12.0 / Blackwell GB202, 32 GB) — kernels are `compute_120f`-only. (Other SM120
  Blackwell parts, e.g. RTX PRO 6000, also work.) The 16 GB and 24 GB variants were emulated on a
  5090 with a capped pool; on a real card the process budget is the card's free memory.
- CUDA Toolkit 13.x (the served build: 13.1, driver 610) with a driver new enough for SM120. Point
  cmake at the toolkit's `nvcc` (below); a build configured against a different runtime than its
  headers reads 1 SM and silently never splits the decode attention.
- Python 3.10+ (measured with 3.14) with `torch`, `transformers`, `numpy`, `safetensors`,
  `tokenizers`; the HTTP verification tools also need `httpx`. The measured versions are pinned in
  `requirements.txt` / `pyproject.toml` (`pip install -r requirements.txt`; torch 2.14.0 from the
  cu130 wheel index, transformers 5.17.0, safetensors 0.8.0, numpy 2.5.3).
- For the EXL3 path: stock upstream [exllamav3](https://github.com/turboderp-org/exllamav3)
  (commit `02aef45` or newer: it carries `convert.py -rcp` and the `sc_measure.py` /
  `sc_optimize.py` sensitivity pipeline) to quantize the checkpoint, and the **original bf16**
  `Qwen/Qwen3.8-27B` as its input — quantizing from the FP8 checkpoint costs +0.003 KL (double
  quantization). The FP6 legacy path converts either checkpoint.

## Build

```bash
git clone git@github.com:kacper-daftcode/qwentin.git
cd qwentin
cmake -S . -B build-qwen -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc   # the toolkit of this nvcc is used for the runtime too
make -C build-qwen -j8                                                         # ~70 s; -> build-qwen/libforward_qwen.so
```

(`cmake -B build-qwen -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build-qwen --target
qwentin-forward-qwen -j` as in the previous README works as well once `nvcc` is on `PATH`.)
Builds are code-reproducible, not byte-reproducible: nvcc embeds per-build-directory metadata,
two builds of one commit differ in md5 but have identical SASS.

## Convert a model

### EXL3 (the served format): exllamav3 quantization, then `.tqf`

```bash
# 1) EXL3 quantization with exllamav3 (stock upstream, commit 02aef45 or newer), from the ORIGINAL bf16
#    checkpoint. The per-tensor bitrate recipe is tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml (400 tensors;
#    sc_optimize.py output for 5.75 bpw on the sensitivity table of the 27B: K = 4 on 0.7 % of the weights,
#    5 on 34 %, 6 on 55 %, 7 on 10.5 %, 8 on 0.04 %). -hb 6 = 6-bit lm_head, -mb 4 = 4-bit MTP layer (the
#    drafter below), -vb 16 = vision tower unquantized (it is not served). ~40 min on one 5090, 250 x 2048
#    calibration rows (the defaults), mul1 codebook, output scales always.
cd exllamav3
python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work-27b-opt5.75 -o /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
# The 16 GB model: the two-tier {4,5} recipe at 4.25 bpw (no bpw header in the file, so -b 4.25 is for reporting)
python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work-27b-t45 -o /path/to/Qwen3.8-27B-exl3-t45-s425 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-t45-s425.yaml -b 4.25 -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
# (A new recipe for another bitrate: sc_measure.py on the bf16 model -> measure.json, then
#  `sc_optimize.py -m measure.json -b <bpw> -hb 6 -o recipe.yaml`. The bf16 27B needs ~54 GB of GPU memory
#  for the measurement; our fork added a layer-split option for several 32 GB cards, upstream needs one big GPU.)

# 2) EXL3 checkpoint -> .tqf (CPU only, ~3 min). The Linears are copied verbatim as trellis records
#    (header flag 0x800, --exl3 auto reads quantization_config.json); TQ_EMIT_MTP=1 --mtp-exl3 stores the
#    checkpoint's own 4-bpw MTP layer as EXL3 records (flag 0x1000) -- the paged drafter behind --spec reads
#    exactly that. Without --mtp-exl3 the server runs with speculative decoding OFF after a warning.
cd qwentin
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -o /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --mtp-exl3          # 21.3 GB
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-t45-s425 \
    -o /path/to/qwen3_8-27b-exl3-t45-s425-mtp4.tqf --mtp-exl3         # 16.7 GB, the 16 GB variant
```

`python3 tools/inspect_tqf.py model.tqf` reports the flags; an EXL3 file with the MTP section for
`--spec` has bits `0x800` and `0x1000` set (`mtp_exl3: true`). `--mtp-source /path/to/Qwen3.8-27B`
instead of `--mtp-exl3` writes a bf16 MTP head (`-mtp.tqf`) for the single-stream MTP tools; the
batched server's drafter needs the EXL3 one. Only `serve_batched.py` runs EXL3 files;
`serve_openai.py` needs the FP6 format below.

### FP6 E2M3 (the legacy single-stream format)

```bash
# FP6 (E2M3) block-scaled weights + MTP head — the production format of the single-stream path (~22.6 GB).
# ALL of --block-scaled always, --block-layout qmma-e2m3 and TQ_EMIT_MTP=1 are
# required: --block-scaled defaults to `auto`, which on a BF16 checkpoint silently
# falls back to plain global-scale FP8 (ignoring --block-layout), and without
# TQ_EMIT_MTP=1 the MTP head is dropped (no spec-decode).
# Optional: TQ_GPU_PACK=1 quantizes/packs on the GPU via the built
# libforward_qwen.so (much faster than the ~13 min numpy path).
# The SAME command converts the official Qwen3.8-27B-FP8 checkpoint:
# tensors stored as block-scaled FP8 (`*_scale_inv`) are dequantized first, so
# "always"/"pow2"/"qmma-e2m3" requantize from true float weights rather than
# from raw E4M3 codes (that includes the MTP projections — HF quantizes those
# too, and the recovered values are stored BF16 in the MTP section). Converting
# from the original bf16 instead of the FP8 checkpoint gives KL 0.0139 instead of 0.0169.
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.6-27B \
    -o /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2
```

`.tqf` is qwentin's on-disk format: quantized weights in the QMMA fragment layout (or EXL3
trellis records), the non-quant tensors (embeddings, norms, conv1d) in bf16, and the MTP draft
head for spec-decode. A production FP6 file reports `block_scaled_e2m3: True` and
`has_mtp_section: True` (header flags `0x53d`); `flags: 0x1d` / `has_mtp_section: False` (and
~28 GB instead of ~22.6) is plain FP8 without the MTP head — reconvert with the exact flags above.

## Serve (OpenAI API)

### The served stack: 12 sessions, 576K pooled tokens, FP8 quality (batched server)

The gated line, verbatim from the hand-off (`logs/cand7-all.log` is the gate of exactly this
line: 29 894 MiB, needle 6/6 at 32K / 128K with TTFT 15.9 / 86.6 s, spec A/B 10/10, 166.9 / 80.6
tok/s per stream at N = 1 / 8 @32K, 132.8 @128K; `/root/work/venv/bin/python` and the
`/mnt/nvme/models/...` paths are the measurement box's — any Python with the requirements and the
files from "Convert a model" do; `--model-dir` is the HF checkpoint dir, used for its tokenizer
and chat template only, so the FP8 or the bf16 dir both serve):

```bash
cd qwentin
CUDA_VISIBLE_DEVICES=x TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 /root/work/venv/bin/python -u tools/serve_batched.py \
  --port 8100 --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf /mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib build-qwen/libforward_qwen.so
```

Everything else is a default of `serve_batched.py` on this commit: `--kv-v4 --kv-t4` (the trellis
KV; codebook `tools/codebooks/lloyd_kv12a.f32`), `--spec` (`--spec-k auto`), `--dstate fp16sr`,
`--decode-steps-per-wave 1`, `--pdl`, `--embed-host`, `--park-ram-gb 64`, mixed prefill waves, the
prefix cache, and the engine defaults `TQ_PAGED_ATTN_ST=3 TQ_T4_ATTN_F16=1 TQ_GRAPH=1
TQ_GEMM_FORK=1 TQ_DSTATE_BF16=4`. Without speculative decoding the pool grows to about
`--num-blocks 5000 --no-spec` (≈ 640K tokens, sized from the memory model rather than a gated
run; a `-mtp.tqf` works too). The startup line prints the KV format,
pooled tokens, state format, spec and attention settings — check it against the one quoted in
"Performance".

```bash
# The 16 GB variant (measured under a 16 GiB process budget on a 5090, logs/p16-check.log): the 4.25 bpw model,
# 2 slots (so the pinned parking ring is allocated), 400 blocks = 51K tokens -> 16 006 MiB after a full run;
# 360 = 46K is the safe pick, 440 = 56K only if the card may sit at ~16 074 MiB. 184 tok/s @32K, 173 @49K, needle 3/3.
CUDA_VISIBLE_DEVICES=x TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 2 --num-blocks 400 --park-ram-gb 16 --timeout 3600 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-t45-s425-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so

# A 24 GB card: a full 256K single session at FP8 quality (opt-5.75 + trellis KV, logs/b2eng-t14-spec.log):
#   --max-slots 1 --num-blocks 2000 with the opt-5.75 -mtp4.tqf  -> 23 452-23 664 MiB, 161 tok/s @32K, 113 @240K, needle 3/3 @240K

# Off-switches that restore the 10-16 line bit for bit (gate 1 of logs/cand7-all.log proved both directions
# bitwise; 525K tokens): the four engine flags in the env + the two server flags + the smaller pool
TQ_T4_ATTN_F16=0 TQ_GRAPH=0 TQ_GEMM_FORK=0 ... tools/serve_batched.py ... --dstate fp32 --decode-steps-per-wave 0 --num-blocks 4100
# The pre-2026-10 stack (FP4 K+V, no spec; e.g. the opt-6.25 model at 422K tokens with --num-blocks 3300):
TQ_PAGED_ATTN_ST=1 ... tools/serve_batched.py ... --no-kv-t4 --no-spec
```

```bash
curl localhost:8100/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a haiku about tensor cores."}],
  "temperature": 0.0, "max_tokens": 64
}'
curl -s localhost:8100/health      # free/total blocks, active, queued, parked, prefilling, prefix-cache hits,
                                   # spec rounds / tokens per round / ms per round / fallbacks, parking tier stats
curl -s 'localhost:8100/spec?on=0' # run-time switch of speculative decoding (on=1&k=N sets the draft depth)
```

Thinking is off unless the request body carries `"enable_thinking": true` (top level; the agents'
harness sends it off explicitly). Per-request stats are returned under `x_qwentin`
(`prefix_cached_tokens`, `slot`, `server_ttft_s`) and `usage.prompt_tokens_details.cached_tokens`;
`/v1/completions` is served too, tool calls pass through the chat template. The server logs one `[admit]` line
per admitted request (slot, prompt tokens, cached, new, tier, free blocks) and one `[req]` line
per request (`TQ_SERVE_LOG_ADMITS=0` silences the former). `--timeout` is the admission wait when
all slots are busy; `--park-ram-gb` the host-RAM budget for parked sessions (16.5 KiB/token + 80
MiB of DeltaNet state each; a 117K-token session restores in well under a second at 37–45 GB/s).

### Legacy single-user path: FP6 + MTP spec-decode (`serve_openai.py`)

```bash
# FP6 + 4-bit KV, 256k context, MTP spec-decode (needs the MTP
# section in the .tqf — see "Convert a model"). The wide+MMA cold-prefill path
# is ON by default (2-3x faster first turns at any length; --no-wide-prefill or
# TQ_WIDE_PREFILL=0 reverts to the 16-token chunked baseline). TQ_EMBED_FP8=2
# (6-bit embed table) is required for the full 256k window on a 32 GB card.
# --model-dir is the original HF checkpoint dir (tokenizer + chat template);
# run from the repo root so the relative --lib path resolves.
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 \
    python3 tools/serve_openai.py --port 8000 --no-thinking \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --model-dir /path/to/Qwen3.6-27B \
    --lib build-qwen/libforward_qwen.so
```

Per-request stats are returned under `x_qwentin` (prefill seconds, accept-length, gen tok/s,
prefix-cache hit). `--no-thinking` defaults `enable_thinking=false` (recommended for agents — it
keeps the prefix cache valid across turns).

#### Reservoir Dogs: spec-round watchdog + fallback drafter (default ON)

When speculative decoding goes sideways mid-request — accept-length collapse (drafter-vs-
target distribution drift) or a dead spec round — the Reservoir Dogs supervisory layer in
`serve_openai.py` keeps the request alive instead of letting it stall or hard-error:

- **Mr. Orange bleeds**: accept_len EMA below `--dogs-accept-min` (default 1.30) across
  `--dogs-min-rounds` (6) steps the drafter down a `(depth, k)` ladder (6/3 -> 4/2 -> 2/1).
- **Mr. Pink walks**: persistent bleeding or a hard round error switches the rest of the
  request to dense decode (still verified against the target model — lossless, just slower).
- **Mr. White watches**: a round exceeding `--dogs-hang-s` (10 s) triggers a non-blocking
  persistent-kernel scratch dump naming the stuck barrier; `--dogs-exit-on-hang` exits with
  code 3 for supervisor restarts.

Per-request visibility: `x_qwentin.dogs` reports the ladder, the rung the request finished
on, the running accept-EMA, and every event. `--no-dogs` disables. Downgrades are one-way
per request; the next request starts at the top rung again. (The batched server's speculative
path has no Dogs: its rounds are bitwise plain decode, and a pool-exhausted draft falls back to a
plain step by itself.)

### Key flags

Environment flags read by the engine (`src/forward_qwen.cu`; the server sets several of them
from its arguments unless they are already in the environment):

| Flag | Default | Meaning |
|------|---------|---------|
| `TQ_CTX` | 262144 | max context per session (the engine cap) |
| `TQ_KV_Q4=1` | — | int4 K paged pool (required by the batched server) |
| `TQ_KV_V4` | server: 1 | V stored like K: FWHT-256-rotated int4, asymmetric per 32 channels (20.0 KiB/token); required by `TQ_KV_T4` (`--kv-v4`) |
| `TQ_KV_T4` | server: 1 | 4-bit trellis-coded K+V for every committed token, 16.5 KiB/token; the open page is staged exact (`--kv-t4`). `TQ_KV_T4_ASYNC` (1): prefill waves attend their own columns exact and encode on a side stream (=0 reproduces the exllamav3 simulation exactly) |
| `TQ_KV_T4_CODEBOOK` | server: shipped file | path of the 4096 raw fp32 codebook entries (`--kv-t4-codebook`, default `tools/codebooks/lloyd_kv12a.f32`; the engine's own fallback when unset is the research project's path, so set it for bare-engine runs — the shipped tools do) |
| `TQ_T4_ATTN_F16` | 1 | the trellis pool's decode step and verify wave run the fp16-operand segment kernel with cp.async-staged V codes (one kernel for both; −1.0 ms/step at 32K N = 8); 0 = the bf16 per-column + segment kernels (the 10-15 path, bit for bit) |
| `TQ_PAGED_ATTN_ST` | 3 | FP4 / E4M3 pools only (a no-op under `TQ_KV_T4`): 0 base GQA decode kernel, 1 staged-V kernel (bitwise = base), 2 fp16 P·V, 3 also fp16 K dequant and Q (−1.5 ms/step at 32K N = 8, verify == decode) |
| `TQ_GRAPH` | 1 | the paged decode step as a CUDA graph (captured per N / key split / MTP columns / flags, LRU of instantiated graphs; a buffer grown inside a capture drops the graph and recaptures); 2 = also the verify wave and the MTP draft (measured: no gain, keep off); 0 = eager |
| `TQ_GEMM_FORK` | 1 | the 2nd/3rd Linear of an input-sharing group (k/v after q, z after qkv, up after gate) on side streams concurrently with the first; same kernels, grids, k-splits → bitwise. With `TQ_GRAPH=1`: −0.35 ms/step at every N and context |
| `TQ_DSTATE_BF16` | 4 | DeltaNet recurrent state storage: 0 fp32 (155 MiB/slot, the 10-16 line bitwise), 1 bf16 (fails the yardstick), 2 fp16 RNE (fails long pure decode), 3 bf16 + 8 bits (24-bit, neutral), 4 fp16 with deterministic stochastic rounding (80 MiB/slot, neutral everywhere measured; `--dstate fp16sr`) |
| `TQ_EXL3_WIDE_NG16` | 1 | 128-token EXL3 GEMM units at one block per SM for every Linear with ≥ 32 of them (k/v_proj keep 32-token units): −5.1 % per 32K prefill wave, −2.6 % at 128K, bitwise; 0 = 64-token units |
| `TQ_EXL3_BF16_R4` | 1 | the bf16 DeltaNet `in_proj_b/a` at prefill N: 4 rows per block, shuffle tree (same partial sums → bitwise), −1.1 % per wave; 0 = the one-row kernel |
| `TQ_WAVE_MAX` | 128 | prefill wave column cap (≤ 256, EXL3 only, for the 256-column A/B; not bitwise by design — half the k-splits). The trellis pool keeps one in-flight staging page per slot, so it stays ≤ 128 there |
| `TQ_PDL` | server: 1 | programmatic dependent launch of the fragment build + EXL3 GEMM in the decode step (`--pdl`; bitwise, −0.55…−0.61 ms/step); 2 = also the step's other kernels (measured slower at 32K) |
| `TQ_EMBED_HOST` | server: 1 | the bf16 embedding table in pinned host RAM, rows read over PCIe (`--embed-host`; no device copy, no quantization error). Overrides `TQ_EMBED_FP8` |
| `TQ_NO_MTP` | server: 1 unless `--spec`/`--mtp` | skip the tqf's MTP section (the bf16 head is 0.85 GB; the EXL3 one 0.21 GB) |
| `TQ_EMBED_FP8=2` | legacy | 6-bit (E2M3) embed table on the device, −1.5 GiB (needed for the full 256k on 32 GB on the single-stream path; irrelevant with `TQ_EMBED_HOST=1`) |
| `TQ_WIDE_PREFILL=1`, `TQ_WIDE_ATTN_MMA=1`, `TQ_PAGED_ATTN_MMA=1` | server: on | wide / ragged tensor-core prefill paths (the single-stream server defaults both ON; the batched server sets `TQ_WIDE_ATTN_MMA=1` for `--page >= 128`) |
| `TQ_ATTN_MMA=1`, `TQ_ATTN_MMA_PAIR=0`, `TQ_ATTN_MMA_GROUP_MIN` / `TQ_SPEC_ATTN_LEGACY_MIN`, `TQ_ATTN_MMA_GROUP2=0`, `TQ_WIDE_CONV=0` | legacy | the single-stream path's attention / conv variants of the 2026-08 README (all default on; the =0 forms are bit-identical fallbacks) |
| `TQ_PAGED_TOPK_PAGES` | 0 | Quest-style page top-k for the decode attention (built, measured to fail the FP8 gate below ~75 % of the pages; keep 0; not ported to the trellis pool) |

<sub>Provenance: flag names and defaults are those of `src/forward_qwen.cu` at engine `2cf671c` (library md5 `e454a29c3caf`, CUDA 13.1 nvcc `cuda_13.1.r13.1/compiler.37061995_0`, driver 610.57.04); the measured effects quoted per row come from the landing gates `docs/evidence/cand4-all.log` … `cand7-all.log` and `ASSUMPTIONS.md` (research repo), 2026-10-16/17, torch 2.14.0+cu130, transformers 5.17.0.</sub>

Server arguments (`tools/serve_batched.py --help` has the full text):

| Argument | Default | Meaning |
|----------|---------|---------|
| `--max-slots N` | 12 | concurrent sessions resident on the GPU (DeltaNet state per slot: 80 MiB with `--dstate fp16sr`) |
| `--num-blocks N` / `--page P` | 1024 / 128 | the shared paged KV pool: N pages of P tokens. Served line 4500 × 128 = 576K tokens (29 894 MiB with 12 slots and spec); 5000 without spec; 400 (2 slots) for the 16 GB model; size another configuration from the startup line's memory |
| `--kv-v4` / `--no-kv-v4` | on | int4 V (`TQ_KV_V4`); `--no-kv-v4` = E4M3 V (26.2 KiB/token), incompatible with `--kv-t4` |
| `--kv-t4` / `--no-kv-t4`, `--kv-t4-codebook FILE` | on | the trellis-coded KV (`TQ_KV_T4`, `TQ_KV_T4_CODEBOOK`); `--no-kv-t4` = FP4 K+V, the pre-2026-10 format |
| `--spec` / `--no-spec`, `--spec-k auto\|1..16` | on, auto | speculative decoding with the paged MTP drafter in decode-only phases; needs a `--mtp-exl3` tqf (else a warning and plain decode); auto = K 4 at ≤ 2 active slots, 3 at ≤ 8, 2 above (measured through the server: K = 6 ties 4 at N = 1 and loses at N = 2; 3 beats 2 and 4 at N = 4). The drafter's KV rows and the EXL3 head take ~11 % of the pool (4500 vs 5000 blocks) |
| `--dstate fp32\|bf16\|fp16\|bf16+8\|fp16sr` | fp16sr | the DeltaNet recurrent-state format (`TQ_DSTATE_BF16` 0–4); `fp32` = the 10-16 line bit for bit |
| `--decode-steps-per-wave k` | 1 | extra pure decode steps after each mixed prefill wave: decoders get 1 + k tokens (speculative rounds with `--spec`, 3–5 tokens each) per wave; 0 = one token per wave. 1 lifted decoders under prefill load 3–4× in the 8-session mix at the price of ~+16 s cold TTFT for a 96K prompt |
| `--mixed-prefill` / `--no-mixed-prefill`, `--prefill-waves-per-step k` | on, 2 | a long prompt's ≤ 128-column waves carry the active slots' decode steps as final segments; `--no-mixed-prefill` alternates waves and steps; `--prefill-waves-per-step 0` = the blocking whole-prompt prefill |
| `--prefix-cache` / `--no-prefix-cache` | on | slot-level prefix cache: a finished request leaves its slot parked (KV pages + state + history); a follow-up that extends the history prefills only the new tokens |
| `--park-ram-gb GB`, `--park-ram-ttl s` | 64, 0 | host-RAM tier for sessions evicted from the GPU (pinned staging ring, 37–45 GB/s; LRU, optional TTL); 0 disables parking |
| `--embed-host` / `--no-embed-host`, `--mtp` | on, off | `TQ_EMBED_HOST`; `--mtp` loads the MTP head without `--spec` |
| `--pdl` / `--no-pdl` | on | `TQ_PDL=1` |
| `--wide-attn-mma` / `--no-wide-attn-mma` | on | tensor-core ragged prefill attention for the waves (`TQ_WIDE_ATTN_MMA`; needs `--page >= 128`); without it the waves use the scalar decode kernel per column (~6× slower prefill) |
| `--timeout s`, `--model-name`, `--host`, `--port` | 300, qwentin-qwen3.6-27b-batched, 127.0.0.1, 8100 | admission wait, the `model` name clients send, bind address |
| `--selftest`, `--bench ...` | — | staggered join/leave correctness vs single-stream; steady-state paged decode bench (`--bench-p` prompt tokens, `--bench-ns` slot counts) — both run with spec off |

<sub>Provenance: `tools/serve_batched.py` at this commit (`--help` is authoritative); the defaults are the ones served on 2026-10-17 with engine `2cf671c` / library md5 `e454a29c3caf` (CUDA 13.1 nvcc `cuda_13.1.r13.1/compiler.37061995_0`, driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0) and gated in `docs/evidence/cand7-all.log`.</sub>

## Verify

Which tool reproduces which number (the HTTP tools take `--base`/`--base-url` of a running
`serve_batched.py`, the engine tools the same `--tqf` / `--model-dir` / `--lib` as the server;
GPU work, so prefix `CUDA_VISIBLE_DEVICES=x` and the served env):

```bash
# needle 6/6 at 32K / 128K (TTFT = the cold prefill): one needle at depths 0.95 / 0.5 / 0.05 with 3 look-alike
# decoys per prompt, exact match of the returned string. --tokenizer = the HF dir for exact prompt sizing
# (else an estimate); --corpus = any large code tree ($QWENTIN_CORPUS; default: this repo).
python3 tools/code_needle.py --base-url http://127.0.0.1:8100 --model qwen3.8-27b \
    --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 --tokenizer /path/to/Qwen3.8-27B \
    --chat-kwargs '{"enable_thinking": false}' --out needle.json

# the per-stream tok/s table: N sessions prefill a ctx-token snapshot (slots parked), then all N decode 300 tokens
# together from the cached prefix, spec on and off via /spec; p50 / min per stream, aggregate, tokens per round
python3 tools/spec_speed.py --base http://127.0.0.1:8100 --ctx 32768 --n 1,2,4,8 --tokenizer /path/to/Qwen3.8-27B
python3 tools/spec_speed.py --base http://127.0.0.1:8100 --ctx 131072 --n 1,2 --tokenizer /path/to/Qwen3.8-27B

# spec A/B 10/10: the same prompts with speculative decoding on and off, greedy -> identical outputs expected
python3 tools/spec_server_ab.py --base http://127.0.0.1:8100 --prompts 10 --tokenizer /path/to/Qwen3.8-27B

# spec exactness ALL PASS (bare engine): every verify column's logits == the sequential decode steps', partial /
# zero commits, wrong drafts, parking; --no-pin = the production split pickers (bitwise thanks to the per-column
# key splits). The served env in front (the bare-engine tools default TQ_KV_T4_CODEBOOK to the shipped codebook;
# the engine's own fallback is the research project's path); ~2 min for both lengths.
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py --lib build-qwen/libforward_qwen.so \
    --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp.tqf --model-dir /path/to/Qwen3.8-27B --ctx 4096 --n 1,3,8 --no-pin
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py ... --ctx 32768 --n 1,3,8 --distinct 3 --no-pin

# "bitwise": raw paged-decode logits of a build on fixed random tokens (N slots, 401 steps, saved from step 376 =
# across the key-split merge at N = 8); dump two builds (or one build with a lever on and off) and diff the tensors
# -> max |d| must be 0. This is how every lever above was gated.
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/paged_logit_ab.py build-qwen/libforward_qwen.so a.pt --tqf /path/to/...-opt5.75-mtp.tqf --n 8
TQ_KV_V4=1 TQ_KV_T4=1 TQ_GRAPH=0 TQ_GEMM_FORK=0 python3 tools/paged_logit_ab.py build-qwen/libforward_qwen.so b.pt --tqf ... --n 8
python3 -c "import torch; a=torch.load('a.pt'); b=torch.load('b.pt'); print(max((a[p]-b[p]).abs().max().item() for p in a))"
```

The quality yardstick (KL / top-1 vs the bf16 original on the 197K held-out tokens, the FP8
reference row, the engine-side runner, the paired block-bootstrap CI and the EvalPlus T1 driver)
is in this repo — `tools/qwentin_e2e.py`, `tools/exl3_diff.py`, `tools/e2e_frontier.py`,
`tools/dequant_fp8_to_bf16.py`, `tools/evalplus.sh` — with the protocol, the held-out corpus (a
pinned vLLM commit, `--corpus`), the gate rule and the reference rows in
[`docs/QUALITY.md`](docs/QUALITY.md); it needs the bf16 27B and exllamav3 on the box. The
8-session load generator and the parking / prefix-cache probes (`eval/loadgen.py`,
`eval/park_probe.py`, `eval/prefix_cache_probe.py`) stay in the research repository
`kacper-daftcode/qwentin-internal`.

Legacy single-stream tools (FP6 tqf): `tools/mtp_spec_smoke.py` (end-to-end MTP spec-decode:
tok/s, accept-length, divergence vs greedy), `tools/bench_rounds.py` (decode-only round benchmark;
`--profile` brackets the timed rounds for `nsys profile -c cudaProfilerApi`) and
`tools/needle_check.py` (`TQ_MODEL_TQF` / `TQ_MODEL_DIR` / `TQ_LIB` from the environment), with
the invocations of the 2026-08 README:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B --lib build-qwen/libforward_qwen.so
CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2 python3 tools/bench_rounds.py --prompt-tokens 65536 --rounds 200 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B --lib build-qwen/libforward_qwen.so
CUDA_VISIBLE_DEVICES=0 TQ_CTX=16384 TQ_MODEL_TQF=/path/to/qwen3_6-27b-e2m3-mtp.tqf TQ_MODEL_DIR=/path/to/Qwen3.6-27B \
    TQ_LIB=build-qwen/libforward_qwen.so python3 tools/needle_check.py
```

## Repository layout

```
src/forward_qwen.cu          the engine: all kernels + C ABI, one CUDA TU (~29k lines) -- FP6 single-stream path,
                             paged pool (int4 / FP4 / trellis KV), EXL3 GEMM/GEMV, paged verify/commit/MTP drafter,
                             DeltaNet state formats, CUDA-graph decode step, parking ring
tools/serve_batched.py       THE SERVED SERVER: multi-client OpenAI API, paged KV + continuous batching, spec decode,
                             prefix cache, host-RAM parking (/health, /spec)
tools/serve_openai.py        legacy single-stream OpenAI server (FP6 + MTP tree, prefix cache, tools, reasoning split, bark, Dogs)
tools/convert_qwen_tqf.py    HuggingFace / EXL3 checkpoint -> .tqf (--exl3, --mtp-exl3, the FP6 block-scaled formats)
tools/inspect_tqf.py         inspect a .tqf model file (flags, sections)
tools/codebooks/lloyd_kv12a.f32   the trellis-KV codebook (4096 fp32, Lloyd-trained; the server's default)
tools/recipes/*.yaml         the exllamav3 per-tensor bitrate recipes of the served models (opt-5.75, t45-s425)
tools/code_needle.py         needle-in-a-repository retrieval matrix through the OpenAI API
tools/spec_speed.py          per-stream decode tok/s at N streams x context, spec on vs off
tools/spec_server_ab.py      spec on vs off on the same prompts: identical-output count
tools/spec_paged_check.py    exactness of the paged verify / commit path vs plain decode (bare engine)
tools/paged_logit_ab.py      raw paged-decode logits of a build for bitwise A/B between builds
tools/harness_common.py      shared helpers of the HTTP tools (tokenizer, code-corpus snapshots, streaming client)
tools/paged_smoke.py         batched/paged decode parity + throughput harness (legacy)
tools/mtp_spec_smoke.py      single-stream spec-decode harness (also exports prefill() used by serve_openai)
tools/bench_rounds.py        single-stream decode-only spec-round benchmark (ms/round, net tok/s; nsys hook)
tools/accept_probe.py        teacher-forced accept probe (degeneration-free draft-quality A/B)
tools/persist_phases.py      phase-clock breakdown of the persistent kernel (work vs barrier tails)
tools/serve_smoke.py         2-turn cold + prefix-cache E2E smoke against a running single-stream server
tools/needle_check.py        single-stream long-context retrieval gate
tools/bark_autotune.py, tools/ot_hook_check.py   --bark-all-day scale sweep and kernel-vs-numpy parity
tools/mtp_*.py               MTP draft-head training / accept-length eval / dump (single-stream)
tools/convert.py, tools/sparse_pack.py           older converter pieces
```

## License

MIT — see [LICENSE](LICENSE).
