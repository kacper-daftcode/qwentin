# qwentin

**Qwen3.8-27B on one RTX 5090 (32 GB): 576K pooled context tokens across 12 sessions at FP8 quality, with speculative
decoding on every stream — 179 / 114 / 82 tok/s per stream at 1 / 4 / 8 streams @32K. On a 16 GB card: 51K tokens,
198 tok/s @32K.**

A from-scratch CUDA inference engine for the 27B hybrid-attention Qwen3.6/3.8 model on one consumer Blackwell GPU
(SM120): hand-written tensor-core kernels behind an OpenAI-compatible server with paged continuous batching. The memory
comes from EXL3 trellis-quantized weights at 5.75 bpw (quantized with turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3), whose EXL3 quantizer the quality numbers stand on), our own
4-bit trellis-coded KV cache (16.5 KiB/token) and the model's own MTP head as the drafter. Built by AI coding agents
(Cursor) directed by the maintainer; every number below is backed by a log in [`docs/evidence/`](docs/evidence/README.md),
and `bench/reproduce.sh` re-measures the served line on your card.

## One 32 GB card (v0.12, 2026-10-21)

| | **qwentin**, 1 × RTX 5090 | vLLM NVFP4, 1 × RTX 5090 (bf16 KV ; fp8 KV) | vLLM FP8, 2 × RTX 5090 (TP2) |
|---|---|---|---|
| Weights, KV cache | EXL3 5.75 bpw, 4-bit trellis K+V (16.5 KiB/token) | NVFP4 4.5 bpw, bf16 ; fp8 K+V | FP8, bf16 K+V |
| Pooled KV capacity | **576K tokens** (12 slots, 29 898 MiB) | 147–179K ; 283K | 493K over two cards |
| Quality vs bf16 (KL / top-1) | **0.0021–0.0027 / 99.0–99.1 %**, EvalPlus = FP8 (93.9 / 91.5 · 90.7 / 77.8 with speculative decoding on) | ≈ 0.04 KL (fake-quant estimate) | 0.0037 / 98.7 % |
| Decode per stream @32K, N = 1 / 4 / 8 | **179 / 114 / 82 tok/s** (66 / 57 / 48 without spec) | 71.5 / 51.4 / — ; 75.3 / 58.8 / — | 76.8 / 60.3 / 50.4 |
| Decode per stream @128K, N = 1 | **114 tok/s** (58 without spec) | 55.5 ; 66.0 | 65.8 |
| Cold TTFT, 32K / 128K | 11.6 / 60.1 s | **3.1 / 25.2 s** ; 3.3 / 30.0 s | **4.4 / 24.1 s** |
| Needle in a repository, 32K + 128K | 6/6 | 6/6 | 6/6 |

<sub>qwentin: engine `40892ff`, library md5 `5f9cee32da7c`, the serve line below, 550 W, CUDA 13.1; every cell measured
twice in one run (`docs/evidence/rel12-table.log`), the two readings within 2 % except the N = 2 cells (one prompt per
stream, 153 / 147 tok/s). Since v0.11: the drafter reads the trunk hidden after the final norm (+0.17 tokens per
round) and a NaN of speculative decoding at position 384 is fixed (CHANGELOG) — the decode kernels are otherwise v0.10's
bit for bit. Each spec-on cell follows the accepted draft length of its one prompt: 3.83 tokens per round @32K N = 1,
2.82 @128K at ~21 / 25 ms per round. vLLM `0.32.0rc1.dev281`, same cards, 2026-10-17; the FP8 checkpoint does not fit
one 5090. Cell-by-cell sources: [`docs/evidence/README.md`](docs/evidence/README.md); the story behind the numbers:
[`docs/NOTES.md`](docs/NOTES.md).</sub>

## One 16 GB card (v0.12, 2026-10-21)

The 4.25 bpw model `t45v4` (−19 % KL at the same size as the previous `t45-s425`) with the same engine, two sessions,
**emulated on a 5090 under a 16 GiB process budget** — not measured on a physical 16 GB card
([`docs/VARIANTS.md`](docs/VARIANTS.md) has the budget convention, the `--num-blocks` ladder and the previous model's cells).

| | **qwentin**, 16 GB | vLLM, stock |
|---|---|---|
| Weights, KV cache | EXL3 4.25 bpw (t45v4), 4-bit trellis K+V | no stock 27B fits: the NVFP4 model alone loads 17.6 GiB, the FP8 one 27.5 GiB |
| Context | **51K tokens** (2 slots, `--num-blocks 400`): 15 604 MiB at start, 15 922 MiB after a full run | — |
| Quality vs bf16 (KL / top-1) | **0.0075–0.0100 / 98.0–98.3 %** (the engine end to end, 4K and 32K), EvalPlus HumanEval/+ 93.9 / 89.6 | — |
| Decode per stream, N = 1 | **198 tok/s @32K** (78 without spec), **182 @49K** (77); 2 streams @32K: 350 tok/s aggregate | — |
| Cold TTFT, 32K / 49K | 12.0 / 18.8 s | — |
| Needle in a repository | 3/3 at 32K, 3/3 at 49K | — |

<sub>Sources: `docs/evidence/cand24-all.log` gate 3 (the line, the 49K cells, T1), `rel12-p16.log` (the 32K cells),
`e2e/qw-t45v4_*.json` (quality); the previous model on the same line and run: 0.0093–0.0128 / 97.8–98.2 %, 193 / 178
tok/s. A 24 GB card holds a full 256K session at FP8 quality with the 32 GB model and one slot (161 tok/s @32K;
[`docs/VARIANTS.md`](docs/VARIANTS.md)).</sub>

## Quick start

Needs an SM120 GPU (RTX 5090 or RTX PRO 6000), CUDA 13.x (13.1 served), Python ≥ 3.10 with `requirements.txt`,
upstream exllamav3 (`02aef45` or newer) and the original bf16 `Qwen/Qwen3.8-27B` (quantizing from the FP8 checkpoint
costs +0.003 KL).

```bash
# build (~70 s); point cmake at the toolkit's nvcc -- a runtime that does not match the headers silently reads 1 SM
git clone git@github.com:kacper-daftcode/qwentin.git && cd qwentin
cmake -S . -B build-qwen -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc && make -C build-qwen -j8
pip install -r requirements.txt                         # torch from the cu130 wheel index first

# convert: exllamav3 with the served recipe (in the exllamav3 checkout, ~40 min on one 5090), then the .tqf (CPU, ~3 min)
python convert.py -i /path/to/Qwen3.8-27B -w /path/to/work -o /path/to/Qwen3.8-27B-exl3-opt5.75 \
    -rcp /path/to/qwentin/tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
python3 tools/convert_qwen_tqf.py /path/to/Qwen3.8-27B-exl3-opt5.75 -o /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --mtp-exl3

# serve: the served line -- every other setting is a default of serve_batched.py
CUDA_VISIBLE_DEVICES=0 python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so
curl localhost:8100/v1/chat/completions -d '{"model": "qwen3.8-27b", "temperature": 0.0, "max_tokens": 64,
  "messages": [{"role": "user", "content": "Write a haiku about tensor cores."}]}'

# verify: build, model check, the served line, needle, spec A/B, speed -> bench/results-<date>-<commit>.json (~5 min)
TQF=/path/to/qwen3_8-27b-exl3-opt5.75-mtp4.tqf MODEL_DIR=/path/to/Qwen3.8-27B bench/reproduce.sh

# a 16 GB card: the t45v4 model (its conversion needs the calibration set and the exllamav3 patch of docs/ARCHITECTURE.md),
# 2 slots, 400 blocks = 51K tokens -> docs/VARIANTS.md "Serve lines"
```

The server binds 127.0.0.1; expose it only with `--api-key` behind a TLS proxy. `/health` reports the pool, the sessions
and the spec-decode stats; thinking is off unless a request sets `"enable_thinking": true`.

## Documentation

- [`docs/FLAGS.md`](docs/FLAGS.md) — every server argument and engine flag, their measured effects, the off-switches
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how it works, the annotated conversion recipe, the repository layout
- [`docs/VARIANTS.md`](docs/VARIANTS.md) — the 16 GB and 24 GB configurations
- [`docs/QUALITY.md`](docs/QUALITY.md) — the quality yardstick (KL / top-1 vs the bf16 original) and its protocol
- [`docs/evidence/`](docs/evidence/README.md) — the gate logs behind every number and how to re-run each gate
- [`docs/NOTES.md`](docs/NOTES.md) — the narrative; [`docs/TQF_FORMAT.md`](docs/TQF_FORMAT.md) — the model file format;
  [`docs/legacy-fp6.md`](docs/legacy-fp6.md) — FP6 models (the single-stream server was retired after v0.11);
  [`CHANGELOG.md`](CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE).
