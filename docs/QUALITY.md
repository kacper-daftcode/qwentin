# The quality protocol — KL / top-1 vs the bf16 original, the FP8 gate, T1, and what is not measured yet

Every "quality" number in the README comes from one yardstick, run on the engine itself (not a simulation of it)
and scored against the **original bf16 `Qwen/Qwen3.8-27B`**. This document is the protocol: what is measured, on
which tokens, with which tools (all in `tools/`), the rule that decides "FP8 quality", the reference rows, the
EvalPlus sanity check, and — honestly — what has not been measured (a standard perplexity eval).

## 1. What KL and top-1 mean here

For every position \(t\) of every held-out row the reference model (bf16, through exllamav3) and the candidate (the
engine, or a checkpoint through exllamav3) each produce a next-token distribution over the 248 320-entry vocabulary,
teacher-forced on the same tokens.

- **KL** = mean over positions of \( \mathrm{KL}(p_{\text{bf16}} \,\|\, p_{\text{candidate}}) =
  \sum_v p_{\text{bf16}}(v) \log \frac{p_{\text{bf16}}(v)}{p_{\text{candidate}}(v)} \), computed by exllamav3's
  `compute_kl_div(candidate_logits, reference_logits, vocab)` in fp32 (`kl_mean` in the run files; `kl_rev_mean` is
  the other direction, `kl_median` / `kl_p90` / `kl_p99` the distribution). The direction matters: it weights the
  candidate's errors by what the *reference* puts probability on.
- **top-1** = the share of positions where \( \arg\max p_{\text{candidate}} = \arg\max p_{\text{bf16}} \)
  (`top1_agree`; `topk_agree` for k = 1…5). This is the greedy-decoding agreement.
- also recorded: `ppl_ratio` (candidate perplexity / reference perplexity on the true next tokens), `by_position`
  buckets (0–512, …, 131072–262144), `by_ref_confidence` buckets, and per-token tensors (`<tag>.pt`) for the paired
  statistics below.

exllamav3's `lm_head` returns fp16 logits, so every checkpoint candidate is scored on fp16-rounded logits; the engine
returns fp32 logits. Engine rows carry both `kl_mean` (fp32) and `kl16_mean` (rounded to fp16 first, like-for-like
with the exllamav3-side rows). The two differ by < 0.00001 on every published row.

**Two paths of the engine are scored separately**, because they are different kernels:
*prefill* (positions go through ≤ 128-column ragged prefill waves, one segment per row — how `serve_batched.py`
prefills a prompt) and *decode* (teacher-forced paged decode steps with 8 or 6 rows in lockstep — the served decode
path: N-column GEMMs, recurrent DeltaNet, GQA decode attention over the trellis pool). `tools/qwentin_e2e.py
--decode-from P` puts positions < P on the prefill path and ≥ P on the decode path.

## 2. The held-out tokens

- **Corpus**: the `vllm/` package directory of a checkout of
  [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — real Python / CUDA / C++ / Markdown, none
  of it in any calibration set used by the quantizers (exllamav3's calibration is its own built-in mix; the NVFP4
  baseline was calibrated on `tests/`, `csrc/`, `examples/`, `benchmarks/`, `docs/` of the same checkout — outside
  `vllm/`).
- **Files**: `load_corpus()` in `tools/harness_common.py` — every file under the root with one of the extensions
  `.py .cu .cuh .h .cpp .rs .ts .js .go .md`, at least 512 characters, sorted by path.
- **Sampling**: `calibration_ids(tokenizer, corpus, 200_000, seed)` — shuffle the file list with
  `random.Random(seed)`, tokenize `### FILE: <relpath>\n<text>\n\n` per file with the Qwen3.8 tokenizer (no special
  tokens), concatenate until 200 000 tokens. The stream is cut into rows: **48 × 4096** ("4K", 196 608 tokens) or
  **6 × 32768** ("32K", 196 608 tokens). Seed **12345** for every 4K row and the first 32K sample ("s1"), seed **2**
  for the second 32K sample ("s2"). The 36 files of the seed-12345 sample are the same for 4K and 32K rows (same
  stream, different cut).
- **Views**: 4K cells score all positions (0 … 4095 of each row). The published **32K cells of the FP8 reference are
  scored at positions ≥ 4096** (`tools/e2e_frontier.py --min-pos 4096`, the long-context view: the first 4K of a 32K
  row is the 4K regime again); the README's served-stack 32K cells quote the all-positions view (see
  `docs/evidence/README.md` for both views of every cell — the gate passes in both).

**Pinned commit and an honest caveat.** The measurement box's checkout is at local commit `6b7448f5b`, which is
upstream commit **`e862c2f45bc81c72bb84f3807d88d60cfb93d694`** (`[Docs] Add example for Renderer.render_cmpl()
usage (#54265)`, 2026-09-03) plus 10 local commits that are not public (an overlay of a vLLM docker image's installed
package and MoE work: 116 files changed under `vllm/`, 47 of them added). The qualifying file count is 2137 locally vs
2095 at the public commit, so the seeded shuffle — and therefore the exact 36-file, 200 000-token sample — **cannot be
regenerated from the public commit**. Six of the 36 files of the seed-12345 sample come from the local commits. A
reproduction from the public commit draws a different sample of the same distribution; expect the same ordering of
stacks and the same ±0.0003 agreement with the FP8 row, not identical digits. The exact streams used are identified
by their hashes (`sha256` of the 200 000 ids as little-endian int32): seed 12345
`e892fda496eb2414e59b463ae6b5a87d88c7ea58f1109539978ba5faae2de627`, seed 2
`2e87b6a8b1f7153e97855aff1928e5a4a92b6a347a12992b72a05fa5a4a4159b`. Every run file records the corpus path, file
count and git HEAD (`corpus` block). **TODO (GPU):** re-run the reference row and the served-stack rows on the public
commit so that the published digits come from public inputs.

```bash
git clone --no-checkout https://github.com/vllm-project/vllm /path/to/vllm
git -C /path/to/vllm checkout e862c2f45bc81c72bb84f3807d88d60cfb93d694
export CORPUS=/path/to/vllm/vllm          # the package directory, not the repo root
# the hash of your sample (should differ from the ones above unless you have the private checkout):
python3 -c "import sys, hashlib, struct; sys.path.insert(0, 'tools'); from harness_common import *; from transformers import AutoTokenizer
ids = calibration_ids(AutoTokenizer.from_pretrained('/models/Qwen3.8-27B'), load_corpus('$CORPUS'), 200000, 12345)
print(len(ids), hashlib.sha256(struct.pack(f'<{len(ids)}i', *ids)).hexdigest())"
```

## 3. The FP8 gate

"FP8 quality" means **not worse than the official FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`, as vLLM serves it with a
bf16 KV cache) on both metrics, within the 95 % confidence interval, at 4K and at 32K**. Concretely, with
`tools/e2e_frontier.py` pairing the candidate run with the FP8 reference run **on the same tokens**:

1. **KL**: the paired mean difference \( \Delta\mathrm{KL} = \mathrm{KL}_{\text{cand}} - \mathrm{KL}_{\text{FP8}} \)
   must be ≤ 0, or its 95 % interval must contain 0 ("equal within the CI").
2. **top-1**: the 95 % interval of \( \Delta\text{top-1} \) must cross 0 or lie above it.
3. at the 4K cell (48 × 4096) **and** at the 32K cells (6 × 32768, positions ≥ 4096, both samples); a stack that passes
   at 4K and fails at 32K fails (this is what eliminated FP4 K+V at opt-5.75 and the plain fp16 DeltaNet state).

The interval is a **block bootstrap**: tokens are grouped into (row, position // 4096) blocks, the blocks are resampled
with replacement 2000 times (fixed seed, so the table is reproducible), and the 2.5 / 97.5 percentiles of the
resampled mean differences are the CI. Blocks rather than tokens because neighbouring positions in one file are not
independent. The same statistic with `--vs <tag>` decides whether a lever is quality-neutral (e.g. `fp16sr` DeltaNet
state vs fp32: `dKL −0.00001 [−0.00005, +0.00003]`).

## 4. Reference rows (the numbers the README quotes)

All vs the original bf16 `Qwen/Qwen3.8-27B`, 2026-10. Run files in `docs/evidence/e2e/` (the `.pt` per-token files
stay in the research repo; the tables below are what `tools/e2e_frontier.py` prints from them).

| row | 4K: KL / top-1 | 32K s1 (≥ 4K) | 32K s2 (≥ 4K) | file |
|---|---|---|---|---|
| **FP8 checkpoint** (`Qwen3.8-27B-FP8` → bf16 via `tools/dequant_fp8_to_bf16.py`, through exllamav3) | **0.00372 / 98.73 %** | **0.00302 / 98.92 %** | **0.00375 / 98.81 %** | `fp8_4k.json`, `fp8_32k.json`, `fp8_32k_s2.json` |
| served stack, prefill path (engine: EXL3 opt-5.75 + trellis-4 K+V, `fp16sr` state) | 0.00235 / 98.98 % (ΔKL −0.00137 [−0.00168, −0.00109], Δtop-1 +0.25 pt [+0.20, +0.31]) | 0.00209 / 99.09 % (−0.00093 [−0.00118, −0.00071], +0.16 [+0.10, +0.23]); all positions 0.00220 / 99.08 % | not run on s2 | `qw-final-def_4k_prefill.json`, `qw-final-def_32k_prefill.json` |
| served stack, decode path (6 or 8 rows in lockstep) | 0.00295 / 98.83 % on 48 rows (−0.00077 [−0.00100, −0.00056], +0.10 [+0.04, +0.16]; 2026-10-15 library); the 8-row sample of 2026-10-17 reads 0.00406 / 98.58 % | 0.00255 / 98.97 % (−0.00047 [−0.00066, −0.00029], +0.05 [−0.01, +0.11]); all positions 0.00269 / 98.96 % | not run on s2 | `qw-o5.75+kvT4L12fin2+e4m3_4k_dec8.json`, `qw-final-def_4k_dec8.json`, `qw-final-def_32k_dec6.json` |
| the four 2026-10-17 opt-ins (`TQ_T4_ATTN_F16` `TQ_GRAPH` `TQ_GEMM_FORK` `fp16sr`) vs the same binary without them | −0.00001 … +0.00001 KL, −0.02 … 0.00 pt, every CI across 0 | −0.00001 [−0.00005, +0.00003] | | `qw-final-all_*.json` vs `qw-final-def_*.json` (`--vs`) |

Weights-only rows (checkpoint through exllamav3, `tools/exl3_diff.py`, 4K, all vs the bf16 original; from the research
repo's `ASSUMPTIONS.md` "A1/A2 re-measured vs the ORIGINAL bf16", run files `runs/vsorig/*.json` there):

| candidate | bpw | ppl ratio | top-1 | KL | KL p99 |
|---|---:|---:|---:|---:|---:|
| FP8 checkpoint | 8 | 1.0016 | 98.73 % | 0.0037 | 0.061 |
| EXL3 opt-6.0 (recipe) | 6.0 | | | 0.0012 | |
| EXL3 opt-5.75 (recipe; **the served weights**) | 5.75 | | | 0.0016 | |
| EXL3 opt-5.5 | 5.5 | | | 0.0023 | |
| EXL3 opt-5.25 | 5.25 | 1.0016 | 98.88 % | 0.0032 | 0.049 |
| EXL3 opt-5.0 | 5.0 | 1.0015 | 98.71 % | 0.0040 | 0.062 |
| EXL3 opt-4.75 | 4.75 | 1.0022 | 98.51 % | 0.0054 | 0.086 |
| EXL3 t45-s425 (two tiers {4, 5}; **the 16 GB weights**) | 4.25 | 1.0032 | 97.92 % | 0.0110 | 0.178 |
| FP6 E2M3 weights (qwentin's legacy format, simulated bit-exactly) | ~6 | 1.0056 | 97.51 % | 0.0126 | 0.195 |
| EXL3 opt-4.0 | 4.0 | 1.0049 | 97.62 % | 0.0138 | 0.221 |
| FP6 + int4-K / E4M3-V KV = the legacy production `.tqf` (from bf16) | ~6 | 1.0062 | 97.36 % | 0.0139 | 0.211 |
| the same converted from the FP8 checkpoint instead of bf16 | ~6 | 1.0090 | 97.09 % | 0.0169 | 0.264 |

The weights-only KL roughly halves per +0.5 bpw; the KV cache adds ~0.0006–0.0008 on top of the weights (trellis-4
K+V on opt-5.75: 0.0016 → 0.0022–0.0024 end to end); quantizing from the FP8 checkpoint instead of the bf16 original
costs ~+0.003 at every bitrate (double quantization) — that is why the README insists on the bf16 input.

## 5. Running the yardstick

Needs: one CUDA GPU with ≥ 32 GB, the original bf16 `Qwen/Qwen3.8-27B` (55.6 GB; the tokenizer is taken from the
same directory), exllamav3 (upstream commit `02aef45` or newer: `pip install -e /path/to/exllamav3`, or
`--exllamav3 /path/to/checkout`), the corpus of § 2, the `.tqf` and `libforward_qwen.so` under test, and the
requirements of `pyproject.toml`. Times are for one RTX 5090.

```bash
export CUDA_VISIBLE_DEVICES=0
export CORPUS=/path/to/vllm/vllm REF=/models/Qwen3.8-27B OUT=runs/e2e

# (a) the FP8 reference row, once: expand the FP8 checkpoint to bf16 containers (CPU, ~30 GB), then score it
python3 tools/dequant_fp8_to_bf16.py /models/Qwen3.8-27B-FP8 /models/Qwen3.8-27B-FP8-as-bf16     # drop "-bf16"/"FP8" naming at will,
python3 tools/exl3_diff.py --ref $REF --model /models/Qwen3.8-27B-FP8-as-bf16 --corpus $CORPUS \  # but keep "Qwen3.8-27B-bf16" in the
    --seq 4096  --out $OUT/fp8_4k.json  --per-token-out $OUT/fp8_4k.pt                            # dir name or pass --ref-pattern
python3 tools/exl3_diff.py --ref $REF --model /models/Qwen3.8-27B-FP8-as-bf16 --corpus $CORPUS \  # to e2e_frontier.py
    --seq 32768 --out $OUT/fp8_32k.json --per-token-out $OUT/fp8_32k.pt                            # ~4 min / ~8 min each
python3 tools/exl3_diff.py ... --seq 32768 --seed 2 --out $OUT/fp8_32k_s2.json --per-token-out $OUT/fp8_32k_s2.pt

# (b) a checkpoint's weights-only row (any EXL3 model dir), same way:
python3 tools/exl3_diff.py --ref $REF --model /models/Qwen3.8-27B-exl3-opt5.75 --corpus $CORPUS --seq 4096 \
    --out $OUT/opt575_w_4k.json --per-token-out $OUT/opt575_w_4k.pt

# (c) the engine end to end: the reference final-norm states once per (seq, seed), then score the engine.
#     The env is the served one (see the README's "Serve"); the published rows used the opt-5.75 -mtp.tqf with
#     TQ_KV_V4=1 TQ_KV_T4=1 TQ_PDL=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1 (docs/evidence/scripts/final-e2e.sh).
python3 tools/qwentin_e2e.py ref --ref-dir $REF --corpus $CORPUS --out-dir $OUT --seq 4096                    # ~3 min, 2 GB .pt
python3 tools/qwentin_e2e.py ref --ref-dir $REF --corpus $CORPUS --out-dir $OUT --seq 32768 --states-on-cpu
export TQ_KV_V4=1 TQ_KV_T4=1 TQ_PDL=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1
E="--ref-dir $REF --corpus $CORPUS --out-dir $OUT --tqf /models/qwen3_8-27b-exl3-opt5.75-mtp.tqf --lib build-qwen/libforward_qwen.so"
python3 tools/qwentin_e2e.py score $E --seq 4096  --decode-from 4096  --tag opt575_4k_prefill    # 48 rows, ~2 min
python3 tools/qwentin_e2e.py score $E --seq 4096  --decode-from 0 --batch 8 --tag opt575_4k_dec8 # decode path, 8 rows lockstep (~2 min; all 48 rows: ~12 min)
python3 tools/qwentin_e2e.py score $E --seq 32768 --decode-from 32768 --tag opt575_32k_prefill   # ~2 min
python3 tools/qwentin_e2e.py score $E --seq 32768 --decode-from 0 --batch 6 --tag opt575_32k_dec6 # ~12 min

# (d) the gate table
python3 tools/e2e_frontier.py --e2e-dir $OUT                      # every run vs the FP8 row of its (seq, seed), all positions
python3 tools/e2e_frontier.py --e2e-dir $OUT --min-pos 4096        # the 32K convention
python3 tools/e2e_frontier.py --e2e-dir $OUT --tags a b --vs a     # lever A/B on the same tokens
```

`bench/reproduce.sh --quality` runs (c)+(d) for the served `.tqf` against an existing `$OUT` (it does not redo the
FP8 row).

## 6. T1 — EvalPlus HumanEval+ / MBPP+ (the sanity check, not the gate)

pass@1, greedy, through the OpenAI endpoint (`tools/evalplus.sh <base-url> <model-name> <out-dir> humaneval|mbpp`;
`pip install evalplus`). 164 HumanEval and 378 MBPP tasks, thinking off. Logs of the two rows the README quotes:
`docs/evidence/evalplus/`.

| stack | HumanEval / HumanEval+ | MBPP / MBPP+ | where |
|---|---|---|---|
| **FP8 reference** — the FP8 checkpoint through vLLM, TP2 on two 5090s, bf16 KV | **93.9 / 90.9** | **90.2 / 77.0** | `docs/evidence/evalplus/fp8-vllm-ref-*.log` |
| **served stack** — EXL3 opt-5.75 + trellis-4 K+V through `serve_batched.py`, 12 slots, 4500 blocks, spec off (spec is bitwise plain decode) | **93.9 / 91.5** | **89.7 / 77.5** | `docs/evidence/evalplus/b2eng-t4-*.log` (2026-10-15) |
| EXL3 opt-5.75 + int4 K / E4M3 V (batched server, 2026-10-02) | 92.1 / 89.6 | 89.9 / 77.5 | research repo `runs/evalplus-exl3o575-batched/` |
| EXL3 opt-6.25 + FP4 K+V (the 2026-10-13 served line) | 93.3 / 89.6 | 90.7 / 77.2 | `runs/evalplus-exl3o6-v4-batched/` |
| legacy FP6 + int4/E4M3 KV, single-stream spec server (from bf16) | 93.3 / 89.6 | 91.3 / 78.0 | `runs/evalplus-fp6q4-bf16src-spec/` |
| legacy FP6, same, converted from the FP8 checkpoint | 93.3 / 90.9 | 90.7 / 78.0 | `runs/evalplus-fp6q4-mmafix-spec/` |
| EXL3 ladder through exllamav3's own generator (weights only, 2026-09, models quantized from the FP8 checkpoint's weights — before the bf16 original was on the box): flat 3.0 bpw | 92.7 / 89.6 | 89.9 / 76.7 | `runs/evalplus-exl3-b3.0/` |
| flat 3.5 bpw | 92.7 / 89.6 | 91.0 / 77.2 | `runs/evalplus-exl3-b3.5/` |
| flat 4.0 bpw | 92.7 / 90.2 | 91.3 / 77.8 | `runs/evalplus-exl3-b4.0/` |
| opt-4.0 recipe | 92.7 / 89.6 | 89.9 / 77.5 | `runs/evalplus-exl3-opt4.0/` |
| hand recipe a6g4m3 (3.43 bpw) | 93.3 / 90.2 | 90.5 / 77.2 | `runs/evalplus-exl3-a6g4m3/` |

Reading: every row from 3.0 bpw up scores within 1–2 points of the reference on all four columns while the KL spans
more than an order of magnitude across the ladder (flat 3.0 bpw: 0.056 vs the FP8 checkpoint; opt-5.75: 0.0016 vs the
original). **T1 is too coarse to separate
bitrates; KL / top-1 is the gate, T1 the sanity check** that nothing is grossly broken (tool-call formatting, chat
template, stop tokens). The ladder rows were produced by the research repo's `eval/t1_exl3.sh`, which starts
`eval/serve_exl3_openai.py` (an OpenAI server on exllamav3's generator) — not vendored here because it is not
self-contained; `tools/evalplus.sh` is the part that scores any endpoint.

## 7. Not measured yet: a standard perplexity eval (wikitext-2 / C4)

No wikitext-2 or C4 perplexity has been measured for any qwentin stack. The yardstick above *is* a perplexity-class
measurement (`ppl_ratio` is reported: 1.0013 for the served stack at 4K, 1.0016 for FP8) but on the code corpus and
against the bf16 model's own distribution, not the standard word-level perplexity that papers quote.

How it would be run, and why it cannot be yet:

```bash
pip install lm-eval
lm_eval --model local-completions --tasks wikitext \
    --model_args model=qwen3.8-27b,base_url=http://127.0.0.1:8100/v1/completions,tokenizer=/models/Qwen3.8-27B,tokenizer_backend=huggingface,num_concurrent=1,max_retries=3
```

`wikitext` (and every loglikelihood task) needs the completions endpoint to return the **prompt's own token logprobs**:
`lm-eval` sends `{"prompt": ..., "max_tokens": 0, "echo": true, "logprobs": 1}` and reads `choices[0].logprobs.token_logprobs`
for the prompt tokens (vLLM's `prompt_logprobs`). **Neither server implements this today**: `tools/serve_batched.py`
ignores `logprobs` / `echo` (it returns text only), `tools/serve_openai.py` answers 400 to `logprobs`. The engine side
is ready — `tools/qwentin_e2e.py` already reads the fp32 logits of every prefill column through the
`qwn_paged_logits_sink` ABI — so this is a server feature, not an engine one.

**TODO**: (1) `serve_batched.py`: `logprobs` + `echo` on `/v1/completions` (prompt logprobs from the prefill waves'
logits sink; `top_logprobs` optional), gated behind a flag because it costs a 248K-wide fp32 row per prompt token;
(2) run `lm_eval` wikitext-2 and C4 (`--tasks wikitext,c4` or the `pile_*`/`c4` configs available in the installed
lm-eval) against the served line and against the FP8 checkpoint through vLLM on the same box; (3) publish both next to
the KL table with the same provenance footer. Until then the README does not claim a wikitext / C4 number.

## 8. Needle, speed and the rest

The needle matrix (`tools/code_needle.py`), the spec A/B (`tools/spec_server_ab.py`), the per-stream speed table
(`tools/spec_speed.py`) and the bitwise checks (`tools/paged_logit_ab.py`, `tools/spec_paged_check.py`) are not
quality yardsticks in the sense above — the needle is a retrieval probe that passed 6/6 even for a damaged NVFP4
export (`docs/evidence/baselines-vllm-20261017.md`, § 3) — and are documented in the README's "Verify" section.
