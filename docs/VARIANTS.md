# The 16 GB and 24 GB variants

Both variants were **emulated on an RTX 5090** with a capped KV pool and a process-memory budget — not measured on a
physical 16 GB or 24 GB card; on a real card the process budget is the card's free memory. The 16 GB model is the
4.25 bpw EXL3 checkpoint **t45v4** (since 2026-10-21: `tools/recipes/qwen3_8-27b-exl3-t45v4.yaml` with the calibration
set `tools/recipes/cal250.safetensors` and the token-weighted Hessian, the conversion block in
[`ARCHITECTURE.md`](ARCHITECTURE.md); −19 % KL on the code yardstick at the same size as the previous two-tier `t45-s425`
recipe, whose numbers are kept below for comparison — but **+7 % on wikitext-2 prose** (`QUALITY.md` § 7): the allocation
is tuned for code, and a prose workload should keep `t45-s425`); the 24 GB card runs the served opt-5.75 model with one slot. Provenance of
every cell: [`evidence/README.md`](evidence/README.md).

**One 16 GB card** — measured on a 5090 under a 16 GiB process budget, **not on a physical
16 GB card** (`docs/evidence/cand24-all.log` gate 3, 2026-10-21, the served library `5f9cee32`; the budget convention
and the exact `--num-blocks` ladder are in `ASSUMPTIONS.md` "16 GiB variant, exact" of the research repo):

| | **qwentin 16 GB variant** (EXL3 t45v4 4.25 bpw + trellis KV + spec, 2 slots) | the same line on the previous model t45-s425 (same library, same run) | vLLM, stock |
|---|---|---|---|
| Context | **51K tokens** (`--num-blocks 400`): 15 604 MiB at start, 15 922 MiB after the parking ring, a 49K needle, speed and T1; `--num-blocks 360` = 46K as the safe pick | 15 654 → 15 954 MiB | none: the NVFP4 27B alone loads 17.6 GiB in vLLM (measured in the README's 32 GB table), the FP8 one 27.5 GiB — no stock 27B fits a 16 GB card |
| Decode, N = 1 | **198 tok/s @32K** (78 without spec), **182 @49K** (77); N = 2 @32K: 350 tok/s aggregate (151 off) | 193 / 178 (79 / 77 off; the spec-on cells are one prompt's acceptance, ±3 %) | — |
| Needle | 3/3 at 32K (TTFT 12.0 s) and at 49K (TTFT 18.8 s) | 3/3 at 49K | — |
| Quality vs the bf16 original, the engine end to end (KL / top-1; `tools/qwentin_e2e.py`, prefill mode, E4M3 embedding) | **0.00921 / 97.99 %** (4K), **0.00750 / 98.32 %** (32K s1), **0.00999 / 98.09 %** (32K s2) | 0.01154 / 97.83 %, 0.00925 / 98.15 %, 0.01281 / 97.82 % — t45v4 on the same tokens: ΔKL −0.00234 [−0.00311, −0.00169], −0.00175 [−0.00234, −0.00128], −0.00282 [−0.00427, −0.00156]; Δtop-1 +0.16 / +0.17 / +0.26 pt | — |
| T1 HumanEval/+ through this line, speculative decoding on | 93.9 / 89.6 (the 32 GB stack: 93.9 / 91.5) | — | — |
| Weights on the device | ~14 GB (the `-mtp4.tqf` is 16.7 GB) | the same | — |

<sub>Provenance: `docs/evidence/cand24-all.log` + `docs/evidence/scripts/cand24-all.sh` (2026-10-21 10:10–10:35 UTC) and the 32K
cells of `docs/evidence/rel12-p16.log` (`scripts/rel12-p16.sh`, 10:53–10:57 UTC; the served
library md5 `5f9cee32da7c` = main `52b4974`, every switch at its default: fp16 + SR DeltaNet state, the fp16 T4 attention, the
decode graph), CUDA 13.1, driver 610.57.04, torch 2.14.0+cu130, transformers 5.17.0; the quality cells: the e2e JSONs
`docs/evidence/e2e/qw-t45v4_*.json` and `qw-t45s425_*.json` (`tools/e2e_frontier.py --vs`); the simulated full-stack rows
of the two models (exllamav3 + the trellis-KV and E4M3-embedding simulation: t45-s425 0.01245 / 0.01342 / 0.00991, t45v4
0.01005 / 0.01069 / 0.00822 at 4K / 32K s2 / 32K s1) are in the research repo; the 2026-10-17 numbers of the previous
model on the previous library (184 / 173 tok/s, 16 006 MiB after a full run) are in `docs/evidence/p16-check.log`; the
vLLM cell: `docs/evidence/baselines-vllm-20261017.md`.</sub>

**One 24 GB card** holds a **full 256K single session at FP8 quality** (opt-5.75 + trellis KV,
1 slot, `--num-blocks 2000`): 23 452–23 664 MiB process, 161 tok/s @32K and 113 tok/s @240K with
speculative decoding (63 / 48 without), needle 3/3 at 240K–256K (`logs/b2eng-t14-spec.log`,
10-16 binary; emulated on a 5090 with a capped pool, like the 16 GB variant). opt-5.0 / opt-4.75
weights do not fit 16 GB at any context; FP4 V stopped at 180K on 24 GB.

## Serve lines

```bash
# The 16 GB variant (measured under a 16 GiB process budget on a 5090, docs/evidence/cand24-all.log): the 4.25 bpw
# t45v4 model, 2 slots (so the pinned parking ring is allocated), 400 blocks = 51K tokens -> 15 922 MiB after a full
# run; 360 = 46K is the safe pick, 440 = 56K only if the card may sit at ~16 074 MiB. 198 tok/s @32K, 182 @49K, needle 3/3.
CUDA_VISIBLE_DEVICES=x python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 2 --num-blocks 400 --park-ram-gb 16 --timeout 3600 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-t45v4-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so

# A 24 GB card: a full 256K single session at FP8 quality (opt-5.75 + trellis KV, logs/b2eng-t14-spec.log):
#   --max-slots 1 --num-blocks 2000 with the opt-5.75 -mtp4.tqf  -> 23 452-23 664 MiB, 161 tok/s @32K, 113 @240K, needle 3/3 @240K
```
