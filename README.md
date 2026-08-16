# qwentin

**Qwen3.8-27B at 256k context on a single RTX 5090 — ~137-157 tok/s at short context (200+ on prefix-cached turns), 113-123 at 128k and ~106-112 at 240k, vLLM-class throughput at higher quality, with cold prefill 2–4× faster by default (2026-08, CUDA 13.3, measured on the 5090 itself).** Qwen3.6-27B converts and runs identically (same text-tower layout).

qwentin is a from-scratch CUDA inference engine that runs the 27B hybrid-attention Qwen3.6
tower on one 32 GB consumer GPU (Blackwell / SM120), using hand-written tensor-core kernels
(FP6 weights, 4-bit KV cache) + MTP speculative decoding. It exposes an OpenAI-compatible API,
so any client — including an `opencode`/`continue`-style coding agent — can use it.

```
27B params + 256k context            →  one 32 GB RTX 5090   (steady-state ~30.8 GiB)
single-stream decode                 →  ~137-157 tok/s @ short · 113-123 @ 128k · ~106-112 @ 243k
cold prefill (wide+MMA, default ON)  →  2059 tok/s @ 32k · 1399 @ 128k · 826 @ 243k
quality (tf-top1 vs bf16)            →  91.3  (the highest that still fits 256k on 32 GB)
```

## Why it's interesting

On a single RTX 5090, qwentin is the only setup that serves **27B at the full 256k context
window** while keeping near-FP8 quality — and it does so at interactive speed.

| | **qwentin (FP6)** | vLLM (NVFP4, Qwen3.6-27B) |
|---|---|---|
| Decode, mean | **123 tok/s** | 142 tok/s |
| Quality (tf-top1 vs bf16) | **91.3** | 85.8 |
| Max context on one 32 GB 5090 | **256k (full)** | ~200k (OOMs at 256k) |
| Weights | 6-bit FP6 (E2M3) tensor-core | 4-bit NVFP4 |

vLLM's 4-bit path is ~15% faster on *raw* decode (fewer weight bytes) — but at **−5.5 points of
quality** and it cannot reach 256k on a 32 GB card. qwentin also ships a 4-bit weight mode
(E2M1) that matches NVFP4 quality (86.5) at ~134 tok/s with the full 256k window, if you want
those bytes. The interactive lever qwentin leans on instead is **speculative decoding** (MTP),
which is lossless w.r.t. quality. (Quality ladder, same protocol: FP8 95.94 > **FP6 91.30** >
E2M1 86.46 ≈ NVFP4 85.78.)

## `--bark-all-day`: serve the LLobotomy cut at engine speed

[LLobotomy](https://github.com/kacper-daftcode/LLobotomy) removes refusal behavior with a
runtime Optimal-Transport hook on the residual stream — PyTorch-side, which caps a hooked
27B tower at ~1 tok/s on a 32 GB card (lab gear, not serving). `--bark-all-day` ports that
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

0.47 is the measured floor on Qwen3.8-27B/FP6 (2026-08-16) and the ship default. The Dogs
keep watching the drafter through it all: if an aggressive scale hurts accept-length,
`x_qwentin.dogs` shows it live.


## Highlights

- **Fits 27B + 256k in 32 GB.** FP6 (E2M3) block-scaled tensor-core weights (~20 GiB) + a
  4-bit-K / E4M3-V KV cache. Needle-in-a-haystack retrieval: **4/4 @ 239k** (depths 2k–235k).
- **MTP speculative decoding.** A multi-token-prediction covering tree + batched FP6 verify
  (each weight read once for the whole tree) → real single-stream speedup at accept-length ~2.6–3.0.
- **Wide prefill.** A dedicated N-wide prefill path — wide FP6 GEMM + chunkwise-parallel
  gated-DeltaNet + tensor-core wide attention — makes cold prompts **2–4× faster** at any
  length (works at Q4-KV/256k; the old 16k cap is gone). ON by default in the server, fully
  integrated with spec-decode and the prefix cache.
- **Hybrid attention.** Qwen3.6 mixes 48 gated-DeltaNet *linear-attention* layers (O(1) state,
  context-independent) with 16 *full-attention* layers — the engine implements both.
- **Custom SM120 tensor-core kernels.** Hand-written inline-PTX Blackwell block-scaled MMA
  (`mma.sync ...kind::mxf8f6f4.block_scale`) + FP6 `ldmatrix ...b6x16` unpack, built with the stock
  CUDA 13 `nvcc`/`ptxas` toolchain — no external assembler or precompiled cubins.
- **OpenAI-compatible server** with cross-turn prefix/KV caching, tool-calling, and a
  reasoning/answer split.
- **Batched / multi-client throughput tier.** A paged-KV + continuous-batching server
  (`serve_batched.py`) serves N concurrent requests on one engine — **1159 tok/s @ N=32**, and
  with the opt-in 4-bit (E2M1) weights **~4 clients @128k** (beating vLLM's ~2-seq long-context cap
  on the same dense 27B/5090). Default-off; the FP6 single-stream path is untouched.

> Research engine. The target is single-stream latency/quality on one RTX 5090, not portability —
> it is SM120-only and wired for the Qwen3.6-27B layout. The Qwen3.8-27B text tower shares that
> exact layout (64 layers, 5120 hidden, 16 full-attn / 48 gated-DeltaNet), so it runs as-is;
> the official FP8 checkpoint converts through the same pipeline (see "Convert a model").

## Performance

**Current build (2026-08, CUDA 13.3 + driver 595)** — Qwen3.8-27B, **FP6 weights + Q4 KV**,
single stream, MTP spec-decode (depth 6, `TQ_MTP_VOCAB_CAP=32768 TQ_NGRAM_DRAFT=1`), cold
prefill on the default wide+MMA path with 32-row query tiles (TQ_WIDE_ATTN_QROWS=32).
Measured on an **RTX 5090** (170 SM), 256k ship config (`TQ_KV_Q4=1 TQ_EMBED_FP8=2`),
on code-heavy repo contexts via `tools/bench_coding.py`. Decode tok/s moves with
accept-length at that text offset; ms/round is the hardware truth; "follow-up" is the
second turn on the same conversation (prefix cache hit, only the suffix prefills):

| Context | Decode (ms/round) | Decode (tok/s, cold / follow-up) | Cold prefill (tok/s) | Follow-up prefill |
|--------:|------------------:|---------------------------------:|---------------------:|------------------:|
|  ~short (4k) | 21.9 | **140.9** / **217.2** | 2534 (1.9 s) | 0.16 s (~350 tok suffix) |
|     8k-11k   | ~21.5-21.9 | ~128-157 | ~2170 (10.8k in 5.0 s) | ~0.13 s |
|    32k  | 22.3 | 142.3 / 211.5 | 2059 (18.7 s) | 0.20 s |
|    65k  | 23.1 | ~127-141 | 1866 (35.1 s) | ~0.27 s |
|   128k  | 25.1 | 122.7 / **113.6-140.1*** | 1399 (116 s) | 0.35 s |
|   243k  | ~26-28 | 111.8 / ~105.6 | 826 (294 s) | 0.87 s |

\* 128k+ follow-up decode is 113.6 at depth 6 and **140.1 tok/s at `--depth 8`**; for
cold 128k keep depth 6 (see "Coding-agent workload tuning" for the trade-off).

*(2026-07 provenance: the previous table on this hardware read 21.7-28.4 ms/round,
119-131 tok/s @short to ~98 @245k, 2200-840 tok/s prefill — pre-QT2-attention,
pre-CUDA-13.3, pre-spec-tuning builds on Qwen3.6. The 2026-08 rows win everywhere.)*

Steady-state VRAM @256k ≈ **30.8 / 31.4 GiB** (the `TQ_EMBED_FP8=2` 6-bit embed table
is what makes 256k fit — without it the 32 GB card OOMs past ~230k).

The decode column is nearly flat: a 200k-deep conversation decodes at ~83% of the
short-context speed (~26-28 vs 21.9 ms/round). The long-context gains come from a
producer/consumer group attention kernel (one 512-thread CTA: 8 warps score a whole
attention kernel (one 512-thread CTA: 8 warps score a whole kv group's K read ONCE
per super-tile while 8 warps run the previous tile's P·V from a double-buffered
smem slab), fused Q4 scale/code loads, key-split prefill attention and a
context-gated standalone attention path — needle retrieval on this card: 4/4 @120k
and **4/4 @239k** (ship config; @24k 4/4 with the bf16 embed table). End-to-end on
the server: a cold 10.8k-token first turn runs in **~5.0 s**; a follow-up turn hits
the prefix cache (e.g. 38528 of 38585 tokens reused) and prefills only the new
suffix in **0.20 s** (32k conversation).

<details>
<summary>RTX PRO 6000 Blackwell (188 SM, same GB202/SM120 class, same build)</summary>

| Context | Decode (ms/round) | Decode (tok/s) |
|--------:|------------------:|---------------:|
|    32k  | 22.7 | 118 |
|    64k  | 23.5 | 114 |
|   128k  | 25.5 | 115 |
|   200k  | 27.9 | 108 |
|   245k  | 28.0 | ~108 |

Short-context rows and cold prefill are within a few percent of the 5090 (same
silicon class). These rows predate the one-wave grid sizing (measured at the old
default chunk count), so the bigger die has a little headroom left on top.
</details>

**Batched / multi-client** (`serve_batched.py` — paged KV + continuous batching). Aggregate decode
throughput scales with concurrency N (FP6; measured on the 2026-06 build, whose single-stream
reference was 70 tok/s):

| N | agg tok/s | vs single-stream |
|--:|----------:|-----------------:|
|  8 |  447 |  6.4× |
| 16 |  730 | 10.4× |
| 32 | 1159 | 16.5× |

The opt-in 4-bit (E2M1) weights push N=32 to **1351 tok/s** and raise capacity from ~2 to **~4
clients @128k** (~98 short clients). vLLM on the same dense 27B/5090 caps at ~2 concurrent @128k —
the hybrid architecture's O(1) DeltaNet state (~155 MiB/client) is the short-client floor (a
long-context advantage, a fixed tax at short ctx). Native ragged batched prefill keeps the
end-to-end HTTP path close to the steady-state ceiling.

## How it works

```
HuggingFace Qwen3.6  ──convert_qwen_tqf.py──▶  model.tqf  (FP6 E2M3, block-scaled, + MTP head)
                                                   │
                          libforward_qwen.so  ◀────┘   (one CUDA translation unit, SM120)
                                   │
                       tools/serve_openai.py  ──▶  OpenAI /v1/chat/completions
```

- **Weights — FP6 E2M3, block-scaled, QMMA.SF.** 6 bits/param on the tensor cores with 128-wide
  block scales; ~20 GiB for 27B.
- **KV cache — 4-bit K (rotated int4 + Hadamard) + E4M3 V.** This is what buys 256k in 32 GB. A
  fp32-KV mode also exists (caps ~32–40k); wide prefill runs against either.
- **Speculative decoding — MTP tree.** Covering-tree build → batched k-split FP6 verify over the
  tree → dense-argmax descent → single-path commit that advances the real
  DeltaNet/conv/full-attn-KV state and extends the draft trunk.
- **Wide prefill.** Projections become one wide FP6 GEMM (weight read once for N tokens); the 48
  DeltaNet layers run a chunkwise-parallel gated-delta kernel; the 16 full-attn layers run a
  tensor-core MMA wide attention that works against the Q4 KV cache at any length (up to 256k)
  — no length gate. The server enables the whole path by default.
- **SM120 kernels.** Block-scaled tensor-core MMA (`mma.sync ...mxf8f6f4.block_scale`), FP6
  `ldmatrix ...b6x16` unpack, fused DeltaNet chains, and the persistent-MLP GEMV are hand-written
  as inline PTX, compiled by the stock CUDA 13 `nvcc`/`ptxas`.

## Requirements

- **RTX 5090** (SM 12.0 / Blackwell GB202, 32 GB) — kernels are `compute_120f`-only. (Other SM120
  Blackwell parts, e.g. RTX PRO 6000, also work.)
- CUDA Toolkit 12.x/13.x with a driver new enough for SM120.
- Python 3.10+ with `torch`, `transformers`, `numpy`, `safetensors`.
- A Qwen3.6-27B / Qwen3.8-27B (or Qwen3.5) HuggingFace checkpoint to convert. Official FP8
  checkpoints (e.g. `Qwen/Qwen3.8-27B-FP8`) work directly — the converter dequantizes the
  block-scaled E4M3 tensors before re-quantizing to FP6, and decodes the MTP head back to BF16.

## Build

```bash
git clone git@github.com:kacper-daftcode/qwentin.git
cd qwentin
export PATH=/usr/local/cuda/bin:$PATH          # so cmake finds nvcc
cmake -B build-qwen -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-qwen --target qwentin-forward-qwen -j
# -> build-qwen/libforward_qwen.so
```

## Convert a model

```bash
# FP6 (E2M3) block-scaled weights + MTP head — the production format (~22.6 GB).
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
# too, and the recovered values are stored BF16 in the MTP section).
TQ_EMIT_MTP=1 python3 tools/convert_qwen_tqf.py /path/to/Qwen3.6-27B \
    -o /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2
```

`.tqf` is qwentin's on-disk format: quantized weights in the QMMA fragment layout, the
non-quant tensors (embeddings, norms, conv1d) in bf16, and the MTP draft head for spec-decode.
Verify with `python3 tools/inspect_tqf.py model.tqf` before serving: a production file
reports `block_scaled_e2m3: True` and `has_mtp_section: True` (header flags `0x53d`).
If you see `flags: 0x1d` / `has_mtp_section: False` (and ~28 GB instead of ~22.6), the
file is plain FP8 without the MTP head — reconvert with the exact flags above.

## Serve (OpenAI API)

### Single-user, latency-optimized (FP6 + MTP spec-decode)

```bash
# Production: FP6 + 4-bit KV, 256k context, MTP spec-decode (needs the MTP
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

```bash
curl localhost:8000/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Write a haiku about tensor cores."}],
  "temperature": 0.0, "max_tokens": 64
}'
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
per request; the next request starts at the top rung again.


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

### Batch-optimized: many concurrent clients (paged KV + continuous batching)

```bash
# Ship config for one 32 GB card: 12 slots over a shared paged pool
# (--num-blocks 1536 x --page 128 = 196k pooled KV tokens next to the FP6
# weights), each client up to TQ_CTX tokens. Without TQ_CTX the engine caps
# every client at 8192 tokens.
CUDA_VISIBLE_DEVICES=0 TQ_CTX=131072 TQ_KV_Q4=1 \
    python3 tools/serve_batched.py --port 8000 \
    --max-slots 12 --num-blocks 1536 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --model-dir /path/to/Qwen3.6-27B
```

Add `TQ_W_E2M1=1` in front for the opt-in 4-bit (E2M1) throughput tier — more clients
(~4 @128k) and ~17% more aggregate tok/s, at NVFP4-level quality (the FP6 default keeps
the quality ceiling).

`serve_batched.py` admits/decodes/detaches N streams against a shared paged KV pool;
`serve_openai.py` stays the latency-optimized single-stream path (FP6, the quality default).

### Key environment flags

| Flag | Meaning |
|------|---------|
| `TQ_CTX` | max context (default = engine cap, 262144) |
| `TQ_KV_Q4=1` | 4-bit-K + E4M3-V KV cache (needed for 256k) |
| `TQ_EMBED_FP8=2` | 6-bit (E2M3) embed table, −1.5 GiB (needed for the full 256k on 32 GB) |
| `TQ_WIDE_PREFILL=1` | wide prefill path (fp32 or Q4 KV; with `TQ_WIDE_ATTN_MMA=1` uncapped, else 16k gate; server defaults both ON) |
| `TQ_ATTN_MMA=1` | tensor-core MMA + online-softmax attention (default on) |
| `TQ_ATTN_MMA_PAIR=0` | disable GQA-paired attention items (default on; bit-identical either way) |
| `TQ_ATTN_MMA_GROUP_MIN` / `TQ_SPEC_ATTN_LEGACY_MIN` | context thresholds of the long-ctx attention auto-gates (default 8k for both; below them the persistent/pair path keeps short contexts bit-identical) |
| `TQ_ATTN_MMA_GROUP2=0` | revert the producer/consumer group-attention kernel to the 2-half variant (default on) |
| `TQ_WIDE_CONV=0` | revert the wide prefill's chunk-parallel conv update to the serial per-token loop (default on; bit-identical either way) |

### Coding-agent workload tuning (2026-08, measured on Qwen3.8-27B-FP8 / RTX 5090)

Build with CUDA 13.x (13.3 + driver 595 measured); 12.9 loses ~10% decode to worse
register allocation in the spec-decode persistent kernels.

```bash
# Recommended decode env for agentic coding loops (long contexts, many follow-up turns):
TQ_MTP_VOCAB_CAP=32768 TQ_NGRAM_DRAFT=1
```

- `TQ_MTP_VOCAB_CAP=32768` caps the MTP draft LM head to the first 32k vocab rows.
  Draft-side only (verify stays exact): ~+3-5% decode tok/s, accept-length unchanged.
- `TQ_NGRAM_DRAFT=1` grafts copy chains from the conversation history into the spec
  tree. Code generation repeats identifiers/indentation constantly, so follow-up
  turns jump to accept-length ~5 (+45-55% decode; 4k: 140.9->217 tok/s, 32k:
  142.3->211.5). Caveat: at 128k+ COLD context (no useful history) failed grafts
  waste verify nodes (~-17% decode there); prefix-cached turns are unaffected and
  dominate real agent loops.
- Keep `--depth 6` (default). `--depth 8` wins only for follow-up turns at 128k+
  (~+27%: 110.3->140.1) and loses ~10-15% at 4k-32k; not worth switching unless the
  deployment is uniform long-context.

### Prefill (2026-08, Qwen3.8 / RTX 5090)

`k_tq_wide_attn_mma<3>` (Q4-KV wide prefill attention) was **L2-bandwidth-bound**
(72% L2 vs 23% SM): each 16-query block re-streamed the K/V super-tiles. Two fixes:

- 128-bit loads in the int4-K dequant path (8 scalar LDG.U8 -> 1 LDG.128 per
  (key, channel-group); bit-identical values),
- `TQ_WIDE_ATTN_QROWS=32` (default): two query tiles per block share one K/V pass;
  numerics bit-identical per query row (verified: 48/48 fixed-point greedy argmax
  agreement vs the stock build over a 24k continuation probe). `=16` restores legacy.

Kernel: 2.17 ms -> 1.40 ms avg @65k (-35%). End-to-end cold prefill on code:
32k: 1865 -> 2059 tok/s; **128k: 1078 -> 1399 tok/s (149.9 s -> 116.4 s wall)**.

## Verify

The verify tools take the same `--tqf` / `--model-dir` / `--lib` paths as the servers
(needle_check reads them from `TQ_MODEL_TQF` / `TQ_MODEL_DIR` / `TQ_LIB`):

```bash
# End-to-end MTP spec-decode (tok/s, accept-length, divergence vs greedy)
CUDA_VISIBLE_DEVICES=0 python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B \
    --lib build-qwen/libforward_qwen.so

# Decode-only round benchmark: prefill once, time M production spec-rounds
# (ms/round, net tok/s, accept-length; --profile brackets the timed rounds
# for `nsys profile -c cudaProfilerApi`). TQ_EMBED_FP8=2 keeps TQ_CTX=262144
# inside 32 GB on a 5090.
CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2 \
    python3 tools/bench_rounds.py --prompt-tokens 65536 --rounds 200 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf --model-dir /path/to/Qwen3.6-27B \
    --lib build-qwen/libforward_qwen.so

# Needle-in-a-haystack retrieval quality at long context
CUDA_VISIBLE_DEVICES=0 TQ_CTX=16384 \
    TQ_MODEL_TQF=/path/to/qwen3_6-27b-e2m3-mtp.tqf \
    TQ_MODEL_DIR=/path/to/Qwen3.6-27B \
    TQ_LIB=build-qwen/libforward_qwen.so \
    python3 tools/needle_check.py
```

## Repository layout

```
src/forward_qwen.cu        the Qwen3.6 engine: all kernels + C ABI, one CUDA TU (~22k lines)
tools/serve_openai.py      single-stream OpenAI server (prefix cache, tools, reasoning split)
tools/serve_batched.py     multi-client OpenAI server (paged KV + continuous batching)
tools/mtp_spec_smoke.py    spec-decode harness (also exports prefill() used by the server)
tools/bench_rounds.py      decode-only spec-round benchmark (ms/round, net tok/s; nsys hook)
tools/accept_probe.py      teacher-forced accept probe (degeneration-free draft-quality A/B)
tools/persist_phases.py    phase-clock breakdown of the persistent kernel (work vs barrier tails)
tools/serve_smoke.py       2-turn cold + prefix-cache E2E smoke against a running server
tools/paged_smoke.py       batched/paged decode parity + throughput harness
tools/needle_check.py      long-context retrieval gate
tools/convert_qwen_tqf.py  HuggingFace Qwen -> .tqf converter (+ convert.py, sparse_pack.py)
tools/mtp_*.py             MTP draft-head training / accept-length eval / dump
tools/inspect_tqf.py       inspect a .tqf model file
```

## License

MIT — see [LICENSE](LICENSE).
