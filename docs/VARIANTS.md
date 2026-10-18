# The 16 GB and 24 GB variants

Both variants were **emulated on an RTX 5090** with a capped KV pool and a process-memory budget — not measured on a
physical 16 GB or 24 GB card; on a real card the process budget is the card's free memory. The 16 GB model is the
4.25 bpw EXL3 checkpoint (`tools/recipes/qwen3_8-27b-exl3-t45-s425.yaml`, converted as in the README's "Quick start"
with `-b 4.25`); the 24 GB card runs the served opt-5.75 model with one slot. Moved from the 2026-10-17 README at
v0.10, numbers unchanged (the DeltaNet prefill lever of `b2e7a3f` shortens their TTFT too but was not re-measured on
these configurations). Provenance of every cell: [`evidence/README.md`](evidence/README.md).

**One 16 GB card** — measured on a 5090 under a 16 GiB process budget, **not on a physical
16 GB card** (`logs/p16-check.log`; the budget convention and the exact `--num-blocks` ladder are
in `ASSUMPTIONS.md` "16 GiB variant, exact"):

| | **qwentin 16 GB variant** (EXL3 t45-s425 4.25 bpw + trellis KV + spec, 2 slots) | vLLM, stock |
|---|---|---|
| Context | **51K tokens** (`--num-blocks 400`) at 16 006 MiB after a full run; `--num-blocks 360` = 46K as the safe pick | none: the NVFP4 27B alone loads 17.6 GiB in vLLM (measured in the README's 32 GB table), the FP8 one 27.5 GiB — no stock 27B fits a 16 GB card |
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

## Serve lines

```bash
# The 16 GB variant (measured under a 16 GiB process budget on a 5090, logs/p16-check.log): the 4.25 bpw model,
# 2 slots (so the pinned parking ring is allocated), 400 blocks = 51K tokens -> 16 006 MiB after a full run;
# 360 = 46K is the safe pick, 440 = 56K only if the card may sit at ~16 074 MiB. 184 tok/s @32K, 173 @49K, needle 3/3.
CUDA_VISIBLE_DEVICES=x TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 python3 -u tools/serve_batched.py \
  --port 8100 --max-slots 2 --num-blocks 400 --park-ram-gb 16 --timeout 3600 --model-name qwen3.8-27b \
  --tqf /path/to/qwen3_8-27b-exl3-t45-s425-mtp4.tqf --model-dir /path/to/Qwen3.8-27B --lib build-qwen/libforward_qwen.so

# A 24 GB card: a full 256K single session at FP8 quality (opt-5.75 + trellis KV, logs/b2eng-t14-spec.log):
#   --max-slots 1 --num-blocks 2000 with the opt-5.75 -mtp4.tqf  -> 23 452-23 664 MiB, 161 tok/s @32K, 113 @240K, needle 3/3 @240K
```
