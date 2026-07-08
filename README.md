# qwentin

**Qwen3.6-27B at 256k context on a single RTX 5090 — ~120 tok/s at short context and 91+ deep into 200k, vLLM-class throughput at higher quality, with cold prefill 2–4× faster by default (2026-07).**

qwentin is a from-scratch CUDA inference engine that runs the 27B hybrid-attention Qwen3.6
tower on one 32 GB consumer GPU (Blackwell / SM120), using hand-written tensor-core kernels
(FP6 weights, 4-bit KV cache) + MTP speculative decoding. It exposes an OpenAI-compatible API,
so any client — including an `opencode`/`continue`-style coding agent — can use it.

```
27B params + 256k context            →  one 32 GB RTX 5090   (steady-state 30.9 GiB)
single-stream decode                 →  118-122 tok/s @ short · 112 @ 32k · 96-99 @ 128k · 91 @ 200k
cold prefill (wide+MMA, default ON)  →  2-4× faster at any length · 1626 tok/s @ 32k · 1210 @ 128k
quality (tf-top1 vs bf16)            →  91.3  (the highest that still fits 256k on 32 GB)

(long-context tok/s: 2026-07 build, RTX PRO 6000 — same GB202/SM120 class; details in Performance)
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

## Highlights

- **Fits 27B + 256k in 32 GB.** FP6 (E2M3) block-scaled tensor-core weights (~20 GiB) + a
  4-bit-K / E4M3-V KV cache. Needle-in-a-haystack retrieval: **3/3 @ 240k**.
- **MTP speculative decoding.** A multi-token-prediction covering tree + batched FP6 verify
  (each weight read once for the whole tree) → real single-stream speedup at accept-length ~2.6–3.0.
- **Wide prefill.** A dedicated N-wide prefill path — wide FP6 GEMM + chunkwise-parallel
  gated-DeltaNet + tensor-core wide attention — makes cold prompts **2–3× faster** at any
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
> it is SM120-only and wired for the Qwen3.6-27B layout.

## Performance

**Current build (2026-07)** — Qwen3.6-27B, **FP6 weights + Q4 KV**, single stream, MTP
spec-decode, cold prefill on the default wide+MMA path. Measured on an RTX PRO 6000
Blackwell workstation card (the same GB202/SM120 silicon class as the 5090, 188 vs 170
SMs; the 5090 rows below will be re-measured when a card frees up):

| Context | Decode (tok/s) | Cold prefill (tok/s) |
|--------:|---------------:|---------------------:|
|  ~short | **118-122** | — |
|    32k  | 112 | 1626 |
|    64k  | 107-114 | 1527 |
|   128k  | **96-99** | 1210 (full prompt in 108 s) |
|   200k  | **91** | 981 (204 s) |
|   245k  | 88 | |

Steady-state VRAM @256k ≈ **30.9 / 31.4 GiB**.

Short contexts are bit-identical to the 2026-06 build; the long-context gains
(128k +63%, 200k 2.1×) come from GQA-paired/grouped attention items (K/V read once
per head pair, and past ~196k once per whole kv group), fused Q4 scale/code loads,
key-split prefill attention and a context-gated standalone attention path — needle
retrieval 4/4 @24k, @120k and @239k on all of them. End-to-end on the server: a cold
10.8k-token first turn drops from ~15 s to **5.5 s**; a follow-up turn hits the prefix
cache (10752 tokens reused) and prefills only the new suffix in **0.074 s**.

<details>
<summary><b>Original 2026-06 build (RTX 5090)</b> — the numbers the engine launched with</summary>

| Context | Decode (tok/s) | Notes |
|--------:|---------------:|-------|
|  ~short | 118 | |
|     8k  | 110 | |
|    32k  |  85 | |
|    64k  |  70 | |
|   100k  |  77 | accept-len 2.66 |
|   200k  |  49 | accept-len 2.98 |

Cold prefill, original fp32-KV wide path vs the N=16 baseline (the wide+MMA successor
is uncapped, works at Q4-KV/256k and is ON by default in the server):

| Prompt | Baseline | Wide | Speedup |
|-------:|---------:|-----:|--------:|
| 2k | ~695 tok/s | **1582 tok/s** | 2.28× |
| 4k | ~688 tok/s | **1364 tok/s** | 1.98× |
| 8k | ~679 tok/s | **1062 tok/s** | 1.56× |

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
- A Qwen3.6-27B (or Qwen3.5) HuggingFace checkpoint to convert.

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
# FP6 (E2M3) block-scaled weights — the production format.
python tools/convert_qwen_tqf.py /path/to/Qwen3.6-27B \
    -o /path/to/qwen3_6-27b-e2m3-mtp.tqf \
    --block-layout qmma-e2m3 --block-scale-policy pow2
```

`.tqf` is qwentin's on-disk format: quantized weights in the QMMA fragment layout, the
non-quant tensors (embeddings, norms, conv1d) in bf16, and the MTP draft head for spec-decode.
Inspect a file with `python tools/inspect_tqf.py model.tqf`.

## Serve (OpenAI API)

```bash
# Production: FP6 + 4-bit KV, 256k context. The wide+MMA cold-prefill path is
# ON by default (2-3x faster first turns at any length; --no-wide-prefill or
# TQ_WIDE_PREFILL=0 reverts to the 16-token chunked baseline).
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 \
    python3 tools/serve_openai.py --port 8000 --no-thinking \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf
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

For **many concurrent clients**, use the batched server (paged KV + continuous batching); add
`TQ_W_E2M1=1` for the 4-bit throughput tier (more clients, faster):

```bash
CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_W_E2M1=1 \
    python3 tools/serve_batched.py --port 8000 \
    --tqf /path/to/qwen3_6-27b-e2m3-mtp.tqf
```

`serve_batched.py` admits/decodes/detaches N streams against a shared paged KV pool;
`serve_openai.py` stays the latency-optimized single-stream path (FP6, the quality default).

### Key environment flags

| Flag | Meaning |
|------|---------|
| `TQ_CTX` | max context (default = engine cap, 262144) |
| `TQ_KV_Q4=1` | 4-bit-K + E4M3-V KV cache (needed for 256k) |
| `TQ_WIDE_PREFILL=1` | wide prefill path (fp32 or Q4 KV; with `TQ_WIDE_ATTN_MMA=1` uncapped, else 16k gate; server defaults both ON) |
| `TQ_ATTN_MMA=1` | tensor-core MMA + online-softmax attention (default on) |
| `TQ_ATTN_MMA_PAIR=0` | disable GQA-paired attention items (default on; bit-identical either way) |

## Verify

```bash
# End-to-end MTP spec-decode (tok/s, accept-length, divergence vs greedy)
CUDA_VISIBLE_DEVICES=0 python3 tools/mtp_spec_smoke.py --prompt-tokens 1024 --gen 128

# Decode-only round benchmark: prefill once, time M production spec-rounds
# (ms/round, net tok/s, accept-length; --profile brackets the timed rounds
# for `nsys profile -c cudaProfilerApi`)
CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 python3 tools/bench_rounds.py \
    --prompt-tokens 65536 --rounds 200

# Needle-in-a-haystack retrieval quality at long context
CUDA_VISIBLE_DEVICES=0 TQ_CTX=16384 python3 tools/needle_check.py
```

## Repository layout

```
src/forward_qwen.cu        the Qwen3.6 engine: all kernels + C ABI, one CUDA TU (~18k lines)
tools/serve_openai.py      single-stream OpenAI server (prefix cache, tools, reasoning split)
tools/serve_batched.py     multi-client OpenAI server (paged KV + continuous batching)
tools/mtp_spec_smoke.py    spec-decode harness (also exports prefill() used by the server)
tools/bench_rounds.py      decode-only spec-round benchmark (ms/round, net tok/s; nsys hook)
tools/accept_probe.py      teacher-forced accept probe (degeneration-free draft-quality A/B)
tools/serve_smoke.py       2-turn cold + prefix-cache E2E smoke against a running server
tools/paged_smoke.py       batched/paged decode parity + throughput harness
tools/needle_check.py      long-context retrieval gate
tools/convert_qwen_tqf.py  HuggingFace Qwen -> .tqf converter (+ convert.py, sparse_pack.py)
tools/mtp_*.py             MTP draft-head training / accept-length eval / dump
tools/inspect_tqf.py       inspect a .tqf model file
```

## License

MIT — see [LICENSE](LICENSE).
