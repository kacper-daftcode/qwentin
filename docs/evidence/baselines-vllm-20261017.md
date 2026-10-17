# Stock vLLM baselines vs the qwentin served line — Qwen3.8-27B on RTX 5090, measured 2026-10-17

_All vLLM numbers below were **measured today (10-17, 07:51–08:50 UTC) on rtxdev2** with the stock vLLM nightly docker image
(`vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294` = vLLM `0.32.0rc1.dev281+g49cb2bbf7`, torch 2.14.0+cu130,
transformers 5.19.0, image built 2026-10-16). GPUs 0 / 1 / 2 are RTX 5090 (32 607 MiB, 31.4 GiB usable, 550 W cap, driver
610.57.04 / CUDA 13.3); every GPU job ran under `flock /tmp/exl3diff-gpu<N>.lock`. The qwentin row is **cited, not re-measured**
(`logs/cand7-all.log`, `logs/final-speed.log`, 10-17 02:44 / 07:15 UTC). Note: the project's earlier FP8 reference container
`qwen38-ref` ran an older image (`fc120ece0a38` = vLLM `0.1.dev20073+g8e685d198`, Aug 2026); today's rows use the nightly for all
cells so that the two vLLM checkpoints are compared on one engine version._

## (a) Summary table

Decode = per-stream tok/s, p50 over the streams, greedy, 300 output tokens, prompts of ~32.7K tokens of real code (the same
corpus / prompts as our `analysis/spec_speed.py`: phase 1 prefills each session with `max_tokens=1`, phase 2 all N sessions
decode together from the prefix cache; per-stream tok/s = (completion_tokens − 1) / (e2e − ttft)). TTFT = cold prefill (no
cache hit) of the needle prompt, streaming, first content token. Needle = `eval/code_needle.py`, 3 decoys, depths 95 / 50 / 5 %
at 32K and 128K (6 probes). "KV capacity" for vLLM is the engine's own `GPU KV cache size` line (pooled across the 4 KV-cache
groups of this hybrid model, in 832-token blocks — see caveat 1); for qwentin it is the pooled block count × 128.

| system | cards | weights on GPU | max context on the card(s) | KV capacity (tokens) | concurrency @128K | decode tok/s per stream N = 1 / 4 / 8 @32K | TTFT 32K / 128K | needle 32K+128K |
|---|---|---|---|---|---|---|---|---|
| **vLLM FP8** (official `Qwen3.8-27B-FP8`, block-FP8 W8A8, bf16 KV) | **1 × 5090** | 27.48 GiB loaded (language model only) | **does not fit** — OOM at `--max-model-len 4096 --gpu-memory-utilization 0.97`: after the weights 2.88 GiB were free and the engine's minimal KV cache for CUDA-graph profiling (vLLM's default `max_num_seqs` 256 × 16 attention layers × one 832-token page of 3.25 MiB = 13.0 GiB, exactly the size it tried to allocate) could not be allocated | — | — | — | — | — |
| **vLLM FP8 TP2** (same checkpoint, `--tensor-parallel-size 2`, bf16 KV) | **2 × 5090 (GPUs 0+2)** | 13.76 GiB per card | 262 144 accepted (first try) | **492 830** @262K-config (479 550 @131K-config), 15.52 GiB per card | **3.66×** (engine line at 131 072); 1.88× @256K | **76.8 / 60.3 / 50.4** (agg 77 / 241 / 397); @128K N=1 65.8, N=2 53.0 | **4.4 s / 24.1 s** | **6 / 6** |
| vLLM FP8 TP2 + MTP spec decode (`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`, extra row) | 2 × 5090 | 13.98 GiB per card | 262 144 | 433 210 | 1.65× @256K | 56.4 / 55.1 / 49.7 — **slower than without** although 2.4–2.7 tokens accepted per step (76–86 % draft acceptance); @128K N=1 **20.0** | 4.6 s / 20.9 s | not run |
| **vLLM NVFP4** (our llm-compressor NVFP4 W4A4 export of the bf16 model, bf16 KV) | **1 × 5090 (GPU 0)** | 17.56 GiB loaded (4.7 GiB of it = bf16 embed + lm_head, 248K vocab) | 262 144 **rejected** ("16.5 GiB KV cache needed, 9.49 GiB available, estimated maximum model length 147 264"); **131 072 accepted** | **146 769 – 178 948** (two startups of the same config: 9.54 vs 11.62 GiB available; vLLM's profiling counted 2.51 vs 0.43 GiB "peak activation") | **1.12× – 1.37×** (engine lines) | **71.5 / 51.4 / —** (agg 72 / 205); **N = 8 does not fit**: the pool holds 4–5 × 32K sessions, the 8-stream run executed as 5 + 3 (p50 19 tok/s = queueing); @128K N=1 **55.5** | **3.1 s / 25.2 s** | **6 / 6** |
| vLLM NVFP4 + `--kv-cache-dtype fp8` (extra row; vLLM switches to the FlashInfer attention backend, 1600-token blocks) | 1 × 5090 | 17.56 GiB | **262 144 accepted** (1.08×) | **283 357** (9.15 GiB; 2.9 GiB "peak activation" charged) | 1.08× @256K (engine line); 2.1× @128K derived (85 of 177 blocks per request) | **75.3 / 58.8 / —** (agg 75 / 231); N = 8 again does not fit (7 running + 1 waiting); @128K N=1 **66.0** | **3.3 s / 30.0 s** | **6 / 6** |
| **qwentin served line** (EXL3 opt-5.75 + 4-bit trellis K+V + MTP spec, `--max-slots 12 --num-blocks 4500`; cited) | **1 × 5090** | ≈18.5 GB device memory for the model (ASSUMPTIONS, opt-5.75 + mtp4), 29 894 MiB total at 576K | 262 144 (12 slots) | **576 000** pooled (4500 × 128) | 4.4 (576K / 131K) | **167 / 109 / 81** with spec (65 / 56 / 47 without); @128K N=1 133, N=2 111 | **15.9 s / 86.6 s** | **6 / 6** (`cand7-all.log`) |

Reading across the rows: on **one** 5090, stock vLLM cannot run the FP8 checkpoint at all, and runs the NVFP4 export at
**147–179K pooled tokens with ≤ 1.4 sessions at 128K** (and 4–5 at 32K) with its default bf16 KV, **283K / 2 at 128K / 7 at 32K**
with `--kv-cache-dtype fp8`, versus qwentin's 576K / 4.4 at 128K / 12 slots; the single-card decode speeds are 71.5 / 51.4 tok/s
(bf16 KV) or 75.3 / 58.8 (fp8 KV) at N = 1 / 4 with no N = 8 possible (vLLM NVFP4, no spec) vs 167 / 109 / 81 (qwentin with spec;
65 / 56 / 47 without).
vLLM's clear win is **prefill**: cold TTFT 3.1 s @32K and 25 s @128K on one card (FlashInfer GDN prefill + CUTLASS/FlashInfer
NVFP4 GEMMs, ~10K prompt tok/s) versus our 15.9 s / 86.6 s. The two-card FP8 TP2 server is a different budget (2 cards, 2× the
HBM) and is listed only because it is the configuration that actually serves the official FP8 checkpoint.

## (b) Exact commands, versions, logs

Launcher used for every server (holds the per-GPU locks for the container's lifetime, logs to `logs/baseline-<name>.log`, copy in
`runs/baseline/run_vllm.sh`; originals under `/root/work/baseline-tools/`):

```
docker run -d --init --name baseline-<name> --gpus '"device=<GPUS>"' --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 -p 127.0.0.1:<PORT>:<PORT> \
  -v /mnt/nvme/models:/models:ro -v /mnt/nvme/vllm-cache-baseline-20261017:/root/.cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HUB_OFFLINE=1 -e NCCL_P2P_LEVEL=SYS \
  vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294 /models/<MODEL> --host 0.0.0.0 --port <PORT> <args>
```
(the same mounts / env / ulimits / `--ipc host` as the old `qwen38-ref` container, taken from `docker inspect qwen38-ref`; new,
separate compile-cache dir so nothing of the old container was touched.)

### 1. FP8 on one 5090 — `logs/baseline-fp8-tp1.log` (full container log `logs/baseline-fp8-tp1.container.log`)

```
run_vllm.sh fp8-tp1 0 8411 Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b-fp8 --max-model-len 4096 --gpu-memory-utilization 0.97 \
  --language-model-only --enable-prefix-caching --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
```
Log lines: `Model loading took 27.48 GiB memory and 13.29 seconds` → `Setting attention block size to 832 tokens to ensure that
attention page size is >= mamba page size` → `torch.compile took 34.57 s` → `torch.OutOfMemoryError: CUDA out of memory. Tried to
allocate 13.00 GiB. GPU 0 has a total capacity of 31.40 GiB of which 2.88 GiB is free. Including non-PyTorch memory, this process
has 28.51 GiB memory in use` (raised in `cudagraph_utils._init_minimal_kv_cache_for_profiling`, i.e. before the real KV budget is
even computed). Container exit code 1 at 07:53:04 UTC. One attempt, as ordered. (With `--max-num-seqs 1` the profiling cache
would be 52 MiB and the server might come up with ~2.5 GiB of KV ≈ 2 × 4K sessions — not a serving configuration; not tried.)

### 2. FP8 TP2 on GPUs 0+2 — `logs/baseline-fp8-tp2-262k.log`, `logs/baseline-fp8-tp2-131k.log`, `logs/baseline-fp8-tp2-mtp.log`

```
run_vllm.sh fp8-tp2-262k 0,2 8412 Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b-fp8 --tensor-parallel-size 2 --max-model-len 262144 \
  --gpu-memory-utilization 0.95 --max-num-seqs 32 --language-model-only --enable-prefix-caching --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
# same with --max-model-len 131072 (name fp8-tp2-131k, started only to record the engine's concurrency line, then stopped)
# same + --speculative-config '{"method":"mtp","num_speculative_tokens":2}' (name fp8-tp2-mtp)
```
Startup lines (262144): `Model loading took 13.76 GiB memory` per worker · `Using FlashInfer GDN prefill kernel` · `Using FLASH_ATTN
attention backend` (FlashAttention 2) · `Setting attention block size to 832 tokens …` · `Padding mamba page size by 6.26% …` ·
`Available KV cache memory: 15.52 GiB` · **`GPU KV cache size: 492,830 tokens, Maximum concurrency for 262,144 tokens per request:
1.88x`** · `kv cache group sizes (832, 832, 832, 832)` · up after 151 s (35 s torch.compile + 53 s CUDA-graph capture, cold cache).
131072: **`GPU KV cache size: 479,550 tokens, Maximum concurrency for 131,072 tokens per request: 3.66x`**. MTP: `Resolved
architecture: Qwen3_5MTP`, `GPU KV cache size: 433,210 tokens, Maximum concurrency for 262,144 tokens per request: 1.65x`,
`SpecDecoding metrics: Mean acceptance length: 2.44–2.73 … Per-position acceptance rate: 0.81–0.91, 0.63–0.85`.

Measurements (client = `runs/baseline/vllm_speed.py`, run with the project's `venv-eval/bin/python`):
```
vllm_speed.py --base http://127.0.0.1:8412 --model qwen3.8-27b-fp8 --ctx 32768 --n 1,4,8 --max-tokens 300 --out runs/baseline/speed-fp8-tp2-32k.json
  -> N=1 76.7 (cold 32K prefill 4.47 s) · N=4 60.3 (min 59.1, agg 240.6) · N=8 50.5 (min 48.1, agg 398.9)      logs/baseline-speed-fp8-tp2-32k.log
  re-run with GPU 1 idle (the first run overlapped the NVFP4 quantization on GPU 1): 76.8 / 60.3 / 50.4           …-32k-rerun.{json,log}
vllm_speed.py … --ctx 131072 --n 1,2 --out runs/baseline/speed-fp8-tp2-128k.json
  -> N=1 65.8 (cold 128K prefill 24.2 s) · N=2 53.0 (agg 104.6)
eval/code_needle.py --base-url http://127.0.0.1:8412 --model qwen3.8-27b-fp8 --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/baseline/needle-fp8-tp2.json
  -> 6/6 PASS; ttft 4.401 / 4.387 / 4.389 s @32K (32.6K prompt tokens), 24.124 / 24.146 / 23.763 s @128K (130.9K)   logs/baseline-needle-fp8-tp2.log
MTP variant: 32K N=1/4/8 = 56.4 / 55.1 / 49.7 (runs/baseline/speed-fp8-tp2-mtp-32k.json); 128K N=1 = 20.0 (…-mtp-128k.json)
```

### 3. NVFP4 on one 5090 — checkpoint `/mnt/nvme/models/Qwen3.8-27B-NVFP4-llmc` (new, 19.7 GB + 0.85 GB `model_mtp.safetensors`)

There was no NVFP4 checkpoint on disk; it was produced today with **llm-compressor 0.14.0** (compressed-tensors 0.19.0,
transformers 5.17.0, torch 2.14.0+cu130 — a venv `/root/work/venv-llmc` created *inside* the vLLM nightly image with
`--system-site-packages`, script `runs/baseline/setup_llmc.sh`; `/root/work/venv` untouched):

```
run_quant.sh 1          # -> docker run … --gpus device=1 … /venv/bin/python /tools/quant_nvfp4.py   (GPU 1 was free; lock held)
quant_nvfp4.py: Qwen3_5ForConditionalGeneration.from_pretrained(/mnt/nvme/models/Qwen3.8-27B, bf16) on CPU;
  recipe = QuantizationModifier(targets="Linear", scheme="NVFP4",
             ignore=["lm_head", "re:.*embed_tokens.*", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*mtp.*"])
  oneshot(pipeline="sequential", sequential_targets=["Qwen3_5DecoderLayer"], num_calibration_samples=512, max_seq_length=2048,
          propagate_error=True, shuffle_calibration_samples=False, tracing_ignore=defaults + ["get_rope_index", "get_placeholder_mask"])
  calibration = 512 windows × 2048 tokens: 384 code (/mnt/nvme/vllm/{tests,csrc,examples,benchmarks}) + 128 markdown (/mnt/nvme/vllm/docs),
                i.e. outside the eval corpus subtree /mnt/nvme/vllm/vllm; the exact token windows: runs/baseline/calib-512x2048.json
  -> 65 subgraphs (64 decoder layers + head), ~7 s per calibration pass and per propagation pass per layer on the 5090,
     oneshot 835 s, save 8 s; log logs/baseline-nvfp4-quant.log; recipe copies: <ckpt>/recipe.yaml, <ckpt>/QUANT-RECIPE.json
```
Scheme as written to `config.json`: `format: nvfp4-pack-quantized`; weights `num_bits 4, type float, strategy tensor_group,
group_size 16, scale_dtype float8_e4m3fn, observer memoryless_minmax` (+ fp32 per-tensor global scale); input activations
`num_bits 4, float, tensor_group 16, dynamic: local, scale_dtype float8_e4m3fn, observer static_minmax` (the calibrated part).
Kept bf16, mirroring the official FP8 checkpoint's `modules_to_not_convert`: lm_head, embeddings, the vision tower, the
48-wide GDN `in_proj_a` / `in_proj_b`, norms/conv/A_log/dt_bias (not Linear). MTP head copied in bf16 by llm-compressor.

**One change was necessary to serve it correctly (`fix_fused_gs.py`, log `logs/baseline-nvfp4-fixgs.log`):** vLLM fuses the GDN
projections `in_proj_qkv` + `in_proj_z` into one `in_proj_qkvz` linear and, for NVFP4, uses a single `weight_global_scale` =
max over the shards (`compressed_tensors_w4a4_nvfp4.py: "the weight global scale is different for parallel layers … will likely
result in reduced accuracy"`). llm-compressor shares the global scale for the fusions it knows (q/k/v, gate/up — verified equal
in the checkpoint) but not for `in_proj_qkv` / `in_proj_z`, whose scales differed by up to 1.8× (e.g. layer 0: 6176 vs 11264),
so vLLM would have silently shrunk one projection of every linear-attention layer by that ratio. The fix re-quantizes exactly
those two modules of the 48 GDN layers from the bf16 source with one shared global scale, using the same compressed-tensors
functions (`calculate_qparams` / `quantize` / `pack_fp4_to_uint8`, bf16 statistics) — the procedure was first verified to
reproduce the stored packed weights and scales **bit-exactly** when given each module's own global scale. 288 tensors replaced,
28 s; the unfixed file is kept as `<ckpt>/model.safetensors.unfused-gs.bak`, per-layer scales in `<ckpt>/FUSED-GS-FIX.json`.
Needle happens to pass 6/6 on the unfixed file too (`runs/baseline/needle-nvfp4-unfixed.json`), so the retrieval probe is not
sensitive enough to see the damage; the warning disappears with the fixed file. All NVFP4 table numbers are from the fixed file
(speed is identical by construction: same kernels and bytes — `speed-nvfp4-32k-unfixed.json` 71.5 / 51.4 confirms).

Serving (`logs/baseline-nvfp4-262k.log` = rejected 262144, `logs/baseline-nvfp4-131k.log` = unfixed file, **`logs/baseline-nvfp4-131k-fixed.log`**):
```
run_vllm.sh nvfp4-131k-fixed 0 8413 Qwen3.8-27B-NVFP4-llmc --served-model-name qwen3.8-27b-nvfp4 --max-model-len 131072 \
  --gpu-memory-utilization 0.95 --max-num-seqs 12 --language-model-only --enable-prefix-caching --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'      # quantization auto-detected: compressed-tensors
```
Startup lines: `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` (real W4A4 on SM120, not the Marlin W4A16 emulation) ·
`Using FlashInfer GDN prefill kernel` · `Using FLASH_ATTN attention backend` · `Model loading took 17.56 GiB memory` · `Setting
attention block size to 832 tokens …` · `Padding mamba page size by 6.26% …` · 262144: `ValueError: To serve at least one request
with the model's max seq len (262144), (16.5 GiB KV cache is needed, which is larger than the available KV cache memory (9.49
GiB). Based on the available memory, the estimated maximum model length is 147264` · 131072 (unfixed file, cold compile cache):
`Available KV cache memory: 9.54 GiB` / **`GPU KV cache size: 146,769 tokens, Maximum concurrency for 131,072 tokens per request:
1.12x`** (`2.51 GiB for peak activation`) · 131072 (fixed file, warm compile cache): `Available KV cache memory: 11.62 GiB` /
**`GPU KV cache size: 178,948 tokens, Maximum concurrency … 1.37x`** (`0.43 GiB for peak activation`).

Measurements:
```
vllm_speed.py --base http://127.0.0.1:8413 --model qwen3.8-27b-nvfp4 --ctx 32768 --n 1,4,8 --max-tokens 300 --out runs/baseline/speed-nvfp4-32k.json
  -> N=1 71.5 (cold 32K prefill 3.06 s) · N=4 51.4 (min 50.8, agg 204.6) · N=8: 5 running + 3 waiting (server log
     "Running: 5 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.5%"), p50 19.1 / max 54.9, follow-up ttft p50 16.6 s = queueing
vllm_speed.py … --ctx 130600 --n 1 --max-tokens 300 --extra-body '{"min_tokens": 300}' --out runs/baseline/speed-nvfp4-128k-min300.json
  -> N=1 55.5 tok/s, cold 128K prefill 22.2 s (130 550 prompt tokens; a 131 022-token prompt leaves only 50 output tokens at
     max_model_len 131072 — speed-nvfp4-128k.json is that 50-token run: 56.9)
eval/code_needle.py --base-url http://127.0.0.1:8413 --model qwen3.8-27b-nvfp4 --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 … --out runs/baseline/needle-nvfp4.json
  -> 6/6 PASS; ttft 3.335 / 3.050 / 3.305 s @32K, 25.279 / 25.266 / 25.083 s @128K                      logs/baseline-needle-nvfp4.log
```

### 4. Extra: NVFP4 + `--kv-cache-dtype fp8` — `logs/baseline-nvfp4-kvfp8-262k.log`

Not asked for, added because "max context / capacity on the card" with bf16 KV undersells what a stock user would do; FP8 KV is
one flag. Same command as 3 with `--max-model-len 262144 --kv-cache-dtype fp8`. Startup: `Using FLASHINFER attention backend out
of potential backends: ['FLASHINFER', 'TRITON_ATTN']` (FlashAttention 2 has no fp8-KV path) · `Setting attention block size to
1600 tokens …` · `Padding mamba page size by 2.17% …` · `Available KV cache memory: 9.15 GiB` (the profiling now charges 2.9 GiB of
peak activation — FlashInfer workspace) · **`GPU KV cache size: 283,357 tokens, Maximum concurrency for 262,144 tokens per
request: 1.08x`** · up after 96 s.
```
vllm_speed.py … --ctx 32768 --n 1,4,8 --out runs/baseline/speed-nvfp4-kvfp8-32k.json
  -> N=1 75.3 (cold 32K prefill 3.3 s) · N=4 58.8 (min 56.1, agg 231.4) · N=8: "Running: 7 reqs, Waiting: 1" (24 blocks of 1600
     per 32K request, 177 in the pool) -> p50 20.9 = queueing
vllm_speed.py … --ctx 131072 --n 1 --extra-body '{"min_tokens": 300}' --out runs/baseline/speed-nvfp4-kvfp8-128k.json
  -> N=1 66.0 tok/s (131 022 prompt tokens), cold 128K prefill 30.1 s
eval/code_needle.py … --out runs/baseline/needle-nvfp4-kvfp8.json -> 6/6; ttft 3.34 / 3.33 / 3.32 s @32K, 29.96 / 29.97 / 29.75 s @128K
```
So with fp8 KV the one-card NVFP4 server reaches 256K context and ~283K pooled tokens (≈ 2 sessions at 128K, 7 at 32K), decodes
slightly faster (FlashInfer decode: 75 / 59 at 32K, 66 at 128K) and prefills 128K slower (30 s vs 25 s with FlashAttention 2).

### DeltaNet / mamba state handling in vLLM (what limits the hybrid model's short-context concurrency)

Qwen3.8-27B = 48 GatedDeltaNet layers + 16 full-attention layers (GQA 4 KV heads × 256). vLLM V1 puts both in one pooled
allocator ("hybrid KV cache manager"): `kv cache group sizes (832, 832, 832, 832)` = 4 groups of 16 layers (1 attention group,
3 GDN groups), one page size for all. Because a GDN layer's recurrent state is 48 heads × 128 × 128 fp32 (+ conv state) =
3.2 MiB per request per layer, vLLM **raises the attention block size from 16 to 832 tokens** so that an attention page (832
tokens × 4 heads × 256 × 2 × 2 B = 3.4 MB) equals a GDN state page, and pads the mamba page by 6.26 % to match exactly. Consequences:
(i) every request costs 3 × 16 state pages = 48 × 3.4 MB = **163 MB** (TP2: 82 MB per card) on top of its KV, independent of
length — ~0.5 % of the pool each; (ii) attention KV is allocated in 832-token (3.4 MB × 16 layers = 54 MB) blocks, so a 32K
request takes 40 + 3 = 43 blocks (35.8K "tokens") — the NVFP4 pool of 176–215 blocks holds 4–5 × 32K sessions, the engine's
1.12×–1.37× at 128K; (iii) `--max-num-seqs` sizes the CUDA-graph profiling allocation (`max_num_seqs` × 16 × 3.4 MB: 13.0 GiB
at the default 256 — the FP8 one-card OOM above; 0.6 GiB at 12) and the graph capture, not the pool; (iv) with prefix caching
the "Mamba cache mode is set to 'align'" (states are cached only at block boundaries; the follow-up turns in the speed probe
still hit the cache: follow-up TTFT 0.13–0.5 s at 32K). The state is fp32 (`mamba_ssm_dtype float32` in the config; vLLM offers
`--mamba-ssm-cache-dtype float16` + stochastic rounding, not used here = stock). Our engine keeps the same 48-layer state in fp16
with stochastic rounding (`--dstate fp16sr`, 76 MB per slot) and 128-token KV blocks.

## (c) Caveats

1. **"KV capacity (tokens)" means different things.** vLLM's `GPU KV cache size: N tokens` is the pooled block count × 832 for
   one KV group; a request also consumes 3 GDN state blocks and whole 832-token blocks, so the usable token count is ~7 % lower
   at 32K and the per-128K concurrency is the engine's own 1.12×–1.37× (NVFP4) / 3.66× (FP8 TP2, two cards). qwentin's 576K is
   4500 × 128-token blocks shared by 12 slots with the DeltaNet state outside the block pool.
2. **vLLM's KV budget is not deterministic across startups**: the same NVFP4 / 131072 command gave 9.54 GiB (146,769 tokens,
   1.12×) on a cold compile cache and 11.62 GiB (178,948 tokens, 1.37×) on a warm one, because the profiling run's "peak
   activation" was 2.51 vs 0.43 GiB. Both are reported; the FP8 TP2 budget was stable (15.52 GiB in three startups). vLLM prints a
   `--kv-cache-memory=…` hint to pin it.
3. **Prefix caching ON in every vLLM run** (`--enable-prefix-caching`, as in the old reference and as in qwentin). The decode
   probe's phase 2 therefore starts from a cache hit (follow-up TTFT 0.13–0.6 s); the "cold" TTFTs are the first-ever prefill of
   each prompt (prefix cache hit rate for that request 0 %). When the pool is smaller than N × 32K (NVFP4, N = 8) the cached
   prefixes are evicted and re-prefilled, which is part of the N = 8 degradation.
4. **Warm-up / compile:** torch.compile, CUDA-graph capture and the Triton GDN warm-up all run before `Application startup
   complete`; the first client request reached each server 20–100 s after that, and that first request (the 32K depth-0.95
   needle, or the N = 1 phase-1 prefill) is included in the tables — it was within 0.3 s of the following probes (e.g. 3.335 vs
   3.050 / 3.305 s; 4.401 vs 4.387 / 4.389 s), so no separate warm-up pass was made. Speed runs were repeated (FP8 TP2 32K:
   76.7 / 60.3 / 50.5 then 76.8 / 60.3 / 50.4; NVFP4 32K: 71.5 / 51.4 twice, on two server instances) — stable to ±0.5 tok/s.
5. **Spec decode:** qwentin's 167 / 109 / 81 use its MTP speculative decoder; its no-spec numbers (65 / 56 / 47) are the
   like-for-like comparison with the plain vLLM rows (76.8 / 60.3 / 50.4 on two cards, 71.5 / 51.4 on one). vLLM's own MTP path
   on this hybrid model (`method: mtp`, k = 2) accepted 2.4–2.7 tokens per step yet was slower per generated token (56 vs 77
   tok/s at 32K, 20 vs 66 at 128K) — the verify step for the GDN layers is expensive in this nightly; not tuned further (stock).
6. **NVFP4 quality:** the needle probe (6 / 6) is a retrieval check, not the project's KL / top-1 yardstick. The project's
   fake-quant measurement of 10-14 (ASSUMPTIONS "NVFP4 / MXFP4 vs EXL3 4.25 bpw": NVFP4 W4A4 RTN = KL 0.066 vs FP8 0.0037 /
   top-1 −4.3 pt) is the quality statement for this format; today's checkpoint is the same scheme (RTN weights, max-calibrated
   activation scales), so it is a **speed/capacity** baseline, not an FP8-quality one. The fused-global-scale repair was
   required for vLLM to even apply the intended scales to the GDN projections.
7. **Different vLLM version from the old reference** (`qwen38-ref` = Aug build `0.1.dev20073`; today = `0.32.0rc1.dev281`, Oct 16).
   Today's nightly is what gives SM120 the FlashInfer NVFP4 GEMM and the FlashInfer GDN prefill kernel; the old image could not
   have served NVFP4.
8. **Two-card rows are two cards.** FP8 TP2 numbers (and the MTP variant) use GPUs 0+2 with NCCL over host memory
   (`NCCL_P2P_LEVEL=SYS`, no P2P on GeForce); per card that is 246K pooled tokens and half the aggregate throughput
   (38 / 120 / 199 tok/s per card at N = 1 / 4 / 8) — listed because it is the only stock way to serve the official FP8
   checkpoint, not as a single-card peer.
9. **Overlap:** the first FP8 TP2 32K speed run overlapped the NVFP4 quantization on GPU 1 (different root port); the identical
   re-run with GPU 1 idle is the number in the table. Nothing else ran on the box during the measurements (GPU 1 idle, 2 MiB).
10. `--language-model-only` in every vLLM run (skips the vision tower, as the old reference did); `--max-num-seqs 32` for TP2 (as
    the reference), `12` for the one-card NVFP4 server (= qwentin's 12 slots); `--gpu-memory-utilization 0.95` for both
    (the reference used 0.90; 0.95 is the more favourable, still stable setting for vLLM).

## Raw files

- Logs: `logs/baseline-fp8-tp1.log` (+ `.container.log`), `logs/baseline-fp8-tp2-262k.log`, `logs/baseline-fp8-tp2-131k.log`,
  `logs/baseline-fp8-tp2-mtp.log`, `logs/baseline-nvfp4-quant.log`, `logs/baseline-nvfp4-fixgs.log`, `logs/baseline-nvfp4-262k.log`
  (+ `.container.log`), `logs/baseline-nvfp4-131k.log`, `logs/baseline-nvfp4-131k-fixed.log`, `logs/baseline-nvfp4-kvfp8-262k.log`,
  `logs/baseline-speed-*.log`, `logs/baseline-needle-*.log`.
- Results: `runs/baseline/speed-{fp8-tp2-32k,fp8-tp2-32k-rerun,fp8-tp2-128k,fp8-tp2-mtp-32k,fp8-tp2-mtp-128k,nvfp4-32k,
  nvfp4-32k-unfixed,nvfp4-128k,nvfp4-128k-min300,nvfp4-kvfp8-32k,nvfp4-kvfp8-128k}.json` (per-stream tok/s, TTFTs, token counts),
  `runs/baseline/needle-{fp8-tp2,nvfp4,nvfp4-unfixed,nvfp4-kvfp8}.json`.
- Tools (copies; originals in `/root/work/baseline-tools/`): `runs/baseline/run_vllm.sh`, `vllm_speed.py`, `quant_nvfp4.py`,
  `fix_fused_gs.py`, `run_quant.sh`, `setup_llmc.sh`; calibration windows `runs/baseline/calib-512x2048.json`.
- Checkpoint: `/mnt/nvme/models/Qwen3.8-27B-NVFP4-llmc` (19.7 GB `model.safetensors` fixed, `.unfused-gs.bak` original,
  `model_mtp.safetensors`, `recipe.yaml`, `QUANT-RECIPE.json`, `FUSED-GS-FIX.json`). Compile cache `/mnt/nvme/vllm-cache-baseline-20261017`.
- Containers `baseline-*` were stopped and removed (`docker rm`); no GPU process left running, all three locks free; `qwen38-ref` / `ds41-w2` untouched.
