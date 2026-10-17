# qwentin

**Qwen3.8-27B on one RTX 5090 (32 GB): 576K pooled context tokens across 12 sessions at FP8 quality, with speculative
decoding on every stream — 167 / 109 / 81 tok/s per stream at 1 / 4 / 8 streams @32K, 133 tok/s @128K, cold TTFT
13.3 s @32K / 77.0 s @128K.** A from-scratch CUDA inference engine for the 27B hybrid-attention Qwen3.6/3.8 tower on
one consumer Blackwell GPU (SM120), hand-written tensor-core kernels behind an OpenAI-compatible API with paged
continuous batching. The memory comes from EXL3 trellis-quantized weights at 5.75 bpw (quantized with turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3), whose EXL3 quantizer the headline quality numbers stand on),
our own 4-bit trellis-coded KV cache (16.5 KiB/token) and the model's own MTP head as an EXL3 drafter — stock vLLM
holds 147–179K tokens on the same card at NVFP4 quality, and wins on cold prefill (3–5× lower TTFT, the next target).
The engine and the measurements were built by AI coding agents (Cursor) directed by the maintainer; every number
below is reproduced by `bench/reproduce.sh` and backed by a log in `docs/evidence/`.

## One 32 GB card, measured 2026-10-17

| | **qwentin v0.10**, 1 × RTX 5090 | vLLM NVFP4, 1 × RTX 5090 (bf16 KV / fp8 KV) | vLLM FP8, **2 × RTX 5090** (TP2) |
|---|---|---|---|
| Weights on the device | EXL3 trellis 5.75 bpw, ~18.5 GiB | NVFP4 4.5 bpw, 17.6 GiB | FP8, 13.8 GiB per card |
| KV cache | 4-bit trellis-coded K+V, **16.5 KiB/token** | bf16 K+V / fp8 | bf16 K+V |
| Pooled KV capacity | **576K tokens** (12 slots × up to 256K; 29 894 MiB) | 147–179K / 283K | 493K across two cards |
| Quality vs the bf16 original (KL / top-1) | **0.0022–0.0027 / 99.0–99.1 %**; T1 EvalPlus = the FP8 reference | NVFP4 PTQ ≈ 0.04 KL (fake-quant yardstick) | = the FP8 checkpoint: 0.0037 / 98.7 % |
| Decode per stream @32K, N = 1 / 4 / 8 | **167 / 109 / 81 tok/s** with spec (65 / 56 / 47 without) | 71.5 / 51.4 / — ; 75.3 / 58.8 / — | 76.8 / 60.3 / 50.4 |
| Decode per stream @128K, N = 1 | **133 tok/s** | 55.5 / 66.0 | 65.8 |
| Time to first token, cold, 32K / 128K | 13.3 s / 77.0 s (~2 450 / ~1 700 prompt tok/s) | **3.1 s / 25.2 s** ; 3.3 / 30.0 s | **4.4 s / 24.1 s** |
| Needle-in-a-repository (3 depths × 3 decoys), 32K and 128K | 6/6 | 6/6 | 6/6 |
| Decode under a concurrent long prefill (8 sessions, 32K / 96K / 160K) | 40 / 25 / 43 tok/s per client | — | — |

<sub>**Numbers pinned at v0.10.** qwentin: engine `b2e7a3f` (`src/forward_qwen.cu`; the DeltaNet prefill kernels on by default, otherwise the kernels of `2cf671c`), library md5 `484bdb825c36`, served line `--max-slots 12 --num-blocks 4500` with no flags, one RTX 5090 at 550 W, CUDA 13.1, driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0, 2026-10-17. Capacity, needle, TTFT: `docs/evidence/cand8-all.log` gate 3 (15.9 / 86.6 s before the DeltaNet prefill lever, `cand7-all.log`). Decode: `docs/evidence/final-speed.log` (R) and `cand7-all.log` gate 3 on `2cf671c`, whose decode path `b2e7a3f` reproduces bit for bit (`cand8-all.log` gate 1); across the gate runs the spec-on cells read 157–169 tok/s at N = 1 and 74–81 at N = 8 (run-to-run ±2 % on one card; the accepted draft length also depends on the text — 3.4–3.6 tokens per round on the vLLM-repository corpus of these numbers, 2.7 on this repository's own code, where `bench/reproduce.sh` measured 129 / 72); spec-off decode and TTFT do not move. Quality: `final-e2e.log`, `e2e/*.json`, `evalplus/` (protocol: `docs/QUALITY.md`); the mixed load: `final-t5.log`. vLLM: `docs/evidence/baselines-vllm-20261017.md` (vLLM `0.32.0rc1.dev281`, same cards, same day; the official FP8 checkpoint OOMs on one 5090 after loading 27.5 GiB of weights, hence two cards). Cell-by-cell map and the full environment: [`docs/evidence/README.md`](docs/evidence/README.md); the full table with the 2026-08 column, read honestly: [`docs/NOTES.md`](docs/NOTES.md).</sub>

**16 GB card** (4.25 bpw model, 2 slots): 51K tokens at 16 006 MiB, 184 tok/s @32K, KL 0.0110 (weights) — [docs/VARIANTS.md](docs/VARIANTS.md).

**24 GB card**: a full 256K single session at FP8 quality, 161 tok/s @32K, 113 @240K — [docs/VARIANTS.md](docs/VARIANTS.md).

## Build

Requirements: an SM120 Blackwell GPU (RTX 5090; the kernels are `compute_120f`-only; the RTX PRO 6000 works too), CUDA
Toolkit 13.x (served: 13.1, driver 610), Python ≥ 3.10 with the pins of `requirements.txt` (torch 2.14.0 from the cu130
wheel index, transformers 5.17.0), and for the EXL3 path stock upstream exllamav3 (commit `02aef45` or newer) plus the
**original bf16** `Qwen/Qwen3.8-27B` (quantizing from the FP8 checkpoint costs +0.003 KL).

```bash
git clone git@github.com:kacper-daftcode/qwentin.git && cd qwentin
cmake -S . -B build-qwen -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc   # that toolkit's runtime is linked too
make -C build-qwen -j8                                                         # ~70 s -> build-qwen/libforward_qwen.so
pip install -r requirements.txt                                                # torch first, from the cu130 index
```

Point cmake at the toolkit's `nvcc`: a build linked against a different runtime than its headers reads 1 SM and
silently never splits the decode attention (`cmake -B build-qwen -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build
build-qwen --target qwentin-forward-qwen -j` works too once `nvcc` is on `PATH`). Builds are code-reproducible, not
byte-reproducible (per-build-directory metadata; identical SASS). CI (`.github/workflows/ci.yml`, CPU only): compile
check of the engine TU, `ruff`, `pytest tests/`, `bench/reproduce.sh --dry-run`; the GPU gate (`nightly-5090.yml`) is
disabled until a self-hosted runner exists (repository variable `QWENTIN_NIGHTLY_RUNNER`).

## Convert

```bash
# 1) EXL3 quantization with exllamav3 from the ORIGINAL bf16 checkpoint, with the per-tensor bitrate recipe of the served
#    model (~40 min on one 5090). -hb 6 = 6-bit lm_head, -mb 4 = the 4-bit MTP layer (the drafter), -vb 16 = vision tower as is
cd exllamav3 && python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work -o /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
# 2) EXL3 checkpoint -> .tqf (CPU only, ~3 min, 21.3 GB); --mtp-exl3 stores the checkpoint's own MTP layer = what --spec drafts with
cd qwentin && python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-opt5.75 -o /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --mtp-exl3
python3 tools/inspect_tqf.py /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --expect exl3-mtp     # "verdict EXL3 + MTP ... PASS"
```

The 16 GB model is the same with `tools/recipes/qwen3_8-27b-exl3-t45-s425.yaml -b 4.25` ([docs/VARIANTS.md](docs/VARIANTS.md)).
The annotated recipe, new bitrates and the file format: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/TQF_FORMAT.md](docs/TQF_FORMAT.md); the FP6 format of the single-stream path: [docs/legacy-fp6.md](docs/legacy-fp6.md).

## Serve

The served line — exactly what `docs/evidence/cand8-all.log` gate 3 measured (29 894 MiB, needle 6/6 at 32K / 128K
with TTFT 13.3 / 77.0 s, spec A/B 10/10 identical). `--model-dir` is the HF checkpoint dir, used for its tokenizer and
chat template only (the FP8 or the bf16 dir both serve):

```bash
CUDA_VISIBLE_DEVICES=0 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so
```

Everything else is a default of `serve_batched.py` at this tag: `--kv-v4 --kv-t4` (the trellis KV; codebook
`tools/codebooks/lloyd_kv12a.f32`), `--spec` (`--spec-k auto`), `--dstate fp16sr`, `--decode-steps-per-wave 1`,
`--gdn-prefill`, `--pdl`, `--embed-host`, `--park-ram-gb 64`, mixed prefill waves, the prefix cache, and the engine
defaults `TQ_PAGED_ATTN_ST=3 TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 TQ_DSTATE_BF16=4 TQ_GDN_PREFILL=1`. The startup
line prints the KV format, the pooled tokens (`576000`), the state format and the spec / attention settings. Without
spec the pool grows to `--num-blocks 5000 --no-spec` (≈ 640K tokens, from the memory model, not a gated run).

```bash
curl localhost:8100/v1/chat/completions -d '{"model": "qwen3.8-27b", "temperature": 0.0, "max_tokens": 64,
  "messages": [{"role": "user", "content": "Write a haiku about tensor cores."}]}'
curl -s localhost:8100/health          # free/total blocks, active, queued, parked, prefilling, prefix-cache hits, spec rounds /
                                       # tokens per round / ms per round / fallbacks, parking tier stats
curl -s 'localhost:8100/spec?on=0'     # run-time switch of speculative decoding (on=1&k=N sets the draft depth)
```

Both servers bind `127.0.0.1`; serve on `--host 0.0.0.0` only with `--api-key KEY` (and a TLS proxy). Thinking is off
unless the request carries `"enable_thinking": true`. Per-request stats, logging, the hardening guards and every
argument: [docs/FLAGS.md](docs/FLAGS.md).

## Verify

```bash
TQF=/path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf MODEL_DIR=/path/to/Qwen3.8-27B bench/reproduce.sh    # ~5 min after the build
```

Build → `.tqf` check → the served line → needle (32K × 3 depths; TTFT = the cold prefill) → spec A/B (10 prompts,
identical outputs expected) → per-stream speed (32K, N = 1 / 8, spec on / off) → `bench/results-<date>-<commit>.json`
with the environment block (commit, library md5, driver, nvcc, torch). `--quality` adds the 4K / 32K KL / top-1 cells
against the bf16 original ([docs/QUALITY.md](docs/QUALITY.md)); `--dry-run` prints every command; `CORPUS=<a vllm
checkout>` gives like-for-like spec-on speed. The individual tools and the exact command behind each log line:
[docs/evidence/README.md](docs/evidence/README.md).

## Key flags

The dozen that matter at this tag (the full tables, measured effects and the off-switches that restore each previous
line bit for bit: [docs/FLAGS.md](docs/FLAGS.md)):

- `--max-slots 12 --num-blocks 4500` — 12 sessions, 4500 × 128-token pages = 576K pooled tokens; 5000 without spec; 400 (2 slots) on 16 GB.
- `--kv-t4` / `--no-kv-t4` (on) — the 4-bit trellis-coded KV (`TQ_KV_T4`, 16.5 KiB/token); off = FP4 K+V, the pre-2026-10 pool.
- `--spec` / `--no-spec`, `--spec-k auto` (on) — speculative decoding with the EXL3 MTP drafter; outputs bitwise plain decode's.
- `--dstate fp16sr` (`TQ_DSTATE_BF16=4`) — DeltaNet state in fp16 + deterministic stochastic rounding, 80 MiB/slot; `fp32` = the 10-16 line.
- `--gdn-prefill` / `--no-gdn-prefill` (`TQ_GDN_PREFILL=1`) — the register-resident DeltaNet prefill kernels: −12 ms per wave, bitwise.
- `TQ_T4_ATTN_F16` (1) — fp16 staged trellis attention for decode and verify (−1.0 ms/step at 32K N = 8).
- `TQ_GRAPH` (1), `TQ_GEMM_FORK` (1) — the decode step as one CUDA graph with concurrent GEMM groups (−0.35 ms/step, bitwise).
- `--decode-steps-per-wave 1` — one speculative round between mixed prefill waves: decoders under a long prefill 13 / 8 / 11 → 43 / 28 / 43 tok/s.
- `--park-ram-gb 64`, `--park-ram-ttl` — host-RAM parking of evicted sessions through a pinned ring (37–45 GB/s); 0 disables.
- `--embed-host` (`TQ_EMBED_HOST=1`) — the bf16 embedding table in pinned host RAM.
- `--pdl` (`TQ_PDL=1`) — programmatic dependent launch in the decode step (−0.55 ms/step, bitwise).
- `--api-key`, `--max-body-bytes` (32 MiB), `--max-tokens-cap` (16384), `--socket-timeout` (600 s) — request-side hardening, both servers.

## Layout

```
src/forward_qwen.cu        the engine: all kernels + C ABI, one CUDA translation unit (SM120)
tools/serve_batched.py     the served server: OpenAI API, paged KV + continuous batching, spec decode, prefix cache, parking
tools/serve_openai.py      the legacy single-stream server (FP6 + MTP tree, residual-map hook, Dogs) -- docs/legacy-fp6.md
tools/convert_qwen_tqf.py  HF / EXL3 checkpoint -> .tqf;  tools/inspect_tqf.py  inspects one (flags, payload walk, verdicts)
tools/codebooks/, tools/recipes/   the trellis-KV codebook; the exllamav3 bitrate recipes of the served models
tools/*.py                 the verification tools (needle, spec speed / A-B / exactness, bitwise logit dump), the quality yardstick
bench/reproduce.sh         build -> model check -> served line -> needle / A-B / speed -> results JSON
tests/                     CPU pytest: the .tqf parser, the request guards, the residual-map flags
docs/                      NOTES.md, ARCHITECTURE.md (how it works, the full layout), FLAGS.md, VARIANTS.md, legacy-fp6.md,
                           QUALITY.md, TQF_FORMAT.md, evidence/ (the gate logs, their scripts, the cell-by-cell map)
CHANGELOG.md               what landed in each release
```

## License

MIT — see [LICENSE](LICENSE).
