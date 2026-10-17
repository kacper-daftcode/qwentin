# Evidence pack — the logs behind the README's numbers

Every headline number in the top-level `README.md` is a line in one of the files in this directory. The files are
verbatim copies of the research project's (`kacper-daftcode/qwentin-internal`) `logs/`, `notes/` and `runs/` —
nothing was edited or trimmed (the paths `/mnt/nvme/...` and `/root/work/venv/bin/python` are the measurement
box's). `scripts/` holds the exact shell scripts that produced each log, so every flag of every gate is on record.

## Environment of the 2026-10-17 measurements

| item | value |
|---|---|
| engine source | `src/forward_qwen.cu` at commit `2cf671c` (`2cf671cc1b4437647e6012da2c6a773491b6b7c0`, 2026-10-17 06:59 UTC); every commit after it on `main` is documentation / tools only |
| served library | `build-qwen/libforward_qwen.so`, md5 `e454a29c3caf7d131ba46e8d9d9bdf07` (the logs print the first 12 hex digits: `e454a29c3caf`), built from `2cf671c` with the CMakeLists.txt of this repo |
| CUDA toolkit | 13.1 — `nvcc` `Cuda compilation tools, release 13.1, V13.1.115`, `Build cuda_13.1.r13.1/compiler.37061995_0` (`/usr/local/cuda-13.1/bin/nvcc`) |
| driver | 610.57.04 |
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB, 550 W power cap (`nvidia-smi` power.limit 550.00 W); one card per server |
| Python | 3.14.4 (`/root/work/venv`); torch 2.14.0+cu130, transformers 5.17.0, safetensors 0.8.0, numpy 2.5.3 |
| models | `Qwen/Qwen3.8-27B` (original bf16, the quality reference), `Qwen/Qwen3.8-27B-FP8` (tokenizer / chat template of the served line, and the FP8 reference row), EXL3 checkpoints quantized with upstream exllamav3 from the bf16 original (`tools/recipes/*.yaml`), `.tqf` files written by `tools/convert_qwen_tqf.py --mtp-exl3` |
| held-out corpus | the `vllm/` package directory of a vLLM checkout — see `docs/QUALITY.md` for the exact commit and the caveat |
| date | 2026-10-17 (UTC times in the logs); the vLLM baselines 07:51–08:50 UTC the same day |

Earlier libraries that appear in the logs (all on `main`, each the gated predecessor of the next):

| md5 (12) | commit | what |
|---|---|---|
| `3ff196fece3f` | `9da2d3b` | the 10-16 served line (`--kv-t4 --spec` defaults) |
| `2bb8e6d81d9e` | `0f74130` | + cp.async-staged fp16 trellis attention behind `TQ_T4_ATTN_F16` (landed by `cand4-all.log`) |
| `e388864ae443` | `14be568` | + CUDA-graph decode step, concurrent GEMM groups, 128-token prefill GEMM units (landed by `cand5-all.log`) |
| `c9d5269c2963` | `11f5879` | + DeltaNet state formats `--dstate` (landed by `cand6-all.log`); `final-*.log` ran on this library with the four opt-ins as flags |
| `e454a29c3caf` | `2cf671c` | the opt-ins as defaults — the served line (landed by `cand7-all.log`); gate 1 of that log proves it bitwise equal to `c9d5269c2963` + flags |

## Which file backs which README number

| README statement | file : line | note |
|---|---|---|
| 29 894 MiB process memory, 12 slots, 4500 blocks = 576 000 pooled tokens, startup line `trellis-4 K + V ... 16.50 KiB/token ... fp16sr 80 MiB/slot ... speculative decoding ON` | `cand7-all.log` gate 3 (`server up ... 29894 MiB`, `batched server: pool blocks=4500 ...`) | library `e454a29c3caf` = `2cf671c`, pure defaults; command in `scripts/cand7-all.sh` (gate 3) |
| needle 6/6 at 32K / 128K, TTFT 15.9 s @32K (32 596 tokens) / 86.6 s @128K (130 905 tokens) | `cand7-all.log` gate 3, the six `[ 32k/128k depth=... ] PASS prompt=... ttft=...` lines | `eval/code_needle.py` = `tools/code_needle.py`, 3 depths × 3 decoys |
| spec A/B 10/10 identical, 121 vs 59 tok/s, 3.39 tokens per round | `cand7-all.log` gate 3, `[ab] ...` lines | `tools/spec_server_ab.py --prompts 10 --max-tokens 300` |
| 167 / 81 tok/s per stream @32K N = 1 / 8, 133 @128K N = 1 (the gate's own run: 166.9 / 80.6 / 132.8) | `cand7-all.log` gate 3, `[speed] ...` lines | `tools/spec_speed.py` |
| the full speed table: 32K N = 1 / 2 / 4 / 8 spec on 167 / 143 / 109 / 81 (log: 165.3 / 143.0 / 109.0 / 78.8), spec off 65 / 62 / 56 / 47 (65.0 / 61.8 / 56.0 / 47.0), 128K N = 1 / 2 spec on 133 / 111 (131.7 / 111.4), aggregate @N = 8 619–632 | `final-speed.log` block `(R)` (the `(D)` block is the same binary's old defaults, 4100 blocks) | library `c9d5269c2963` + `TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 --dstate fp16sr --num-blocks 4500` = the `2cf671c` defaults (bitwise, `cand7-all.log` gate 1). The README rounds the N = 1 cell to the `cand7-all.log` run (166.9 → 167) and N = 8 to 81 (80.6); the 128K cell 133 = 132.8 (`cand7`) / 131.7 (`final-speed`) |
| quality: 0.0024 / 98.98 % (4K prefill), 0.0022 / 99.08 % (32K prefill), 0.0027 / 98.96 % (32K decode) | `final-e2e.log` rows `qw-final-def_4k_prefill` (0.00235 / 98.98), `qw-final-def_32k_prefill` (0.00220 / 99.08), `qw-final-def_32k_dec6` (0.00269 / 98.96) in the first table of each block (positions ≥ 0); the run files are `e2e/qw-final-def_*.json` | the engine-side yardstick `tools/qwentin_e2e.py` on library `c9d5269c2963`, old defaults; the `qw-final-all_*` rows are the same binary with the four opt-ins = the served defaults, within ±0.00003 KL of the `def` rows (the `[CI]` columns); `def` is bitwise the `qw-dstate-f32_*` reference rows (`[e2e] ... bitwise ... True`) |
| 0.0030 at 4K on the decode path | `e2e/qw-o5.75+kvT4L12fin2+e4m3_4k_dec8.json` (`kl_mean` 0.00295, `top1_agree` 98.83 %) — the 48-row × 4096 decode-only run of 2026-10-15 on the B2 branch library (`notes/session10-b2.md` in the research repo, table "final (v4, rebased) 4K decode B=8") | NOT the 4K decode row of `final-e2e.log`: that one (`qw-final-def_4k_dec8`, 0.00406 / 98.58 %) scores only the first 8 rows (32 760 tokens, a different and smaller sample; its `[CI]` column pairs it with the dstate-f32 run on the same 8 rows, bitwise). The 48-row decode measurement was not repeated on the landed library; the prefill rows and the 8-row decode rows were (bitwise = the 10-16 reference rows) |
| FP8 checkpoint: 0.0037 / 98.73 % at 4K; 0.0030 / 98.92 % · 0.0038 / 98.81 % on the two 32K samples | `e2e/fp8_4k.json` (`kl_mean` 0.00372, `top1_agree` 98.73 %); `e2e/fp8_32k.json`, `e2e/fp8_32k_s2.json` scored on positions ≥ 4096 (`tools/e2e_frontier.py --min-pos 4096`: 0.00302 / 98.92 and 0.00375 / 98.81); over all positions they read 0.00324 / 98.90 and 0.00357 / 98.83 | the two 32K FP8 cells in the README are the `--min-pos 4096` view, the served stack's 32K cells the all-positions view (≥ 4096 the served stack reads 0.00209 / 99.09 prefill and 0.00255 / 98.97 decode) — the FP8 gate passes in either view (paired dKL −0.00093 [−0.00118, −0.00071] prefill, −0.00047 [−0.00066, −0.00029] decode at ≥ 4096) |
| T1: HumanEval/+ 93.9 / 91.5 · MBPP/+ 89.7 / 77.5 (trellis stack) vs 93.9 / 90.9 · 90.2 / 77.0 (FP8 reference) | `evalplus/b2eng-t4-{humaneval,mbpp}.log` (`pass@1: 0.939 / 0.915`, `0.897 / 0.775`); `evalplus/fp8-vllm-ref-{humaneval,mbpp}.log` (`0.939 / 0.909`, `0.902 / 0.770`) | served stack: `serve_batched.py` on the opt-5.75 + trellis-4 line, 12 slots, 4500 blocks, spec off (2026-10-15 21:47 / 22:05 UTC, B2 branch library); reference: the FP8 checkpoint through vLLM TP2 (two 5090s, `qwen38-ref`) |
| 8-session mix: 24/24 OK, cold TTFT p50 16 / 97 / 365 s, follow-up 3–5 s, 16/16 prefix hits, 5 parked, decode 40 / 25 / 43 tok/s per client under prefill, 39 / 40 / 35 on follow-ups, vs 13 / 8 / 11 with `--decode-steps-per-wave 0` | `final-t5.log` block `g1` (the 40.3 / 24.7 / 43.1 and 38.6 / 40.3 / 34.5 rows); `e1` = `--decode-steps-per-wave 0` (12.6 / 8.4 / 11.1) | `eval/loadgen.py` of the research repo; library `c9d5269c2963` + the opt-in flags (= `2cf671c` defaults) |
| 16 GB variant: 51K tokens (`--num-blocks 400`) at 16 006 MiB after a full run, 184 tok/s @32K (76 without spec), 173 @49K (74), needle 3/3 at 48K | `p16-check.log` (`process 16006 MiB`, `[speed] ctx=32768 ... 184.1 / 75.9`, `ctx=49152 ... 172.8 / 73.7`, three `48k ... PASS`) | 2 slots, t45-s425 4.25 bpw `-mtp4.tqf`, 16 GiB process budget on a 5090 (not a 16 GB card). The script header names `14be568`; by the log's start time (01:17:46 UTC) `build-qwen/libforward_qwen.so` had already been swapped to `c9d5269c2963` (`cand6-all.log` LANDED 01:14:04); the log itself does not print the md5. Flags: the `export` line of `scripts/p16-check.sh` (the then-defaults of that binary: fp32 DeltaNet state, `TQ_T4_ATTN_F16` / `TQ_GRAPH` / `TQ_GEMM_FORK` unset) |
| 24 GB variant: a full 256K single session, 23 452–23 664 MiB, 161 tok/s @32K and 113 @240K with spec (63 / 48 without), needle 3/3 at 240K | `b2eng-t14-spec.log` block `p24 + spec` (`process 23452 MiB at start`, `[speed] ctx=32768 ... 161.3 / 62.2`, `ctx=240000 ... 112.8 / 48.0`, `240000 1/1 1/1 1/1 3/3`, `23664 MiB after`) | 2026-10-16 01:18 UTC, B2 branch library `build-qwen-b2t` (the 10-16 binary), 1 slot, 2000 blocks, emulated on a 5090 with a capped pool; `scripts/b2eng-t14-spec.sh` |
| the landing gates of patches 0045–0060 (TTFT 32K 17.1 → 16.0 s, 128K 93–98 → 87.4 s; each lever bitwise against its predecessor) | `cand4-all.log` (0045–0047), `cand5-all.log` (0048–0057), `cand6-all.log` (0058–0060); `cand7-all.log` (0061, the defaults flip) | each log: gate 1 `paged_logit_ab` bitwise (max |d| 0), gate 2 `spec_paged_check` ALL PASS, gate 3 the served line (needle / A-B / speed) |
| the vLLM columns: FP8 does not fit one 5090 (OOM after 27.48 GiB), NVFP4 147–179K tokens / 71.5 / 51.4 tok/s / TTFT 3.1 / 25.2 s, FP8 TP2 493K / 76.8 / 60.3 / 50.4 / 4.4 / 24.1 s, needle 6/6 | `baselines-vllm-20261017.md` — the summary table (a), the exact `docker run` / `vllm serve` commands, image digest `nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294` = vLLM `0.32.0rc1.dev281+g49cb2bbf7`, the NVFP4 checkpoint's provenance (llm-compressor 0.14.0, recipe, the fused-global-scale repair), the caveats (c) | measured 2026-10-17 on the same cards; the per-run vLLM logs (`logs/baseline-*.log`) and JSON results (`runs/baseline/*.json`) stay in the research repo (listed at the end of the note) |

Not in this pack (no 2026-10 gate log exists for them): the legacy FP6 single-stream table (2026-08 build, CUDA 13.3 /
driver 595), the RTX PRO 6000 rows and the 2026-06 batched-server rows — they are carried over from the 2026-08 README
unchanged and are labelled as such in their footers. (The 2026-08 scale sweep of the residual-stream steering hook
moved out of this repository together with the steering maps it was measured for.)

## Reading the logs

- `[plab] ... bitwise identical ... True (max |d| 0)` — `tools/paged_logit_ab.py` dumped 25 paged-decode steps of raw
  logits for N = 3 and N = 8 slots with two libraries (or one library with a lever on and off) and the tensors are
  equal element for element. This is the "bitwise" of the README.
- `[spec-check] ... ALL PASS` — `tools/spec_paged_check.py`: every verify column's logits equal the sequential
  decode steps' (0 different, max |d| 0), partial / zero commits, wrong drafts and parking included.
- `[ab] identical outputs: 10/10` — `tools/spec_server_ab.py`: the same 10 prompts greedy with speculative decoding
  on and off through the HTTP server give identical texts.
- `[speed] ctx=... N=... spec=on: per-stream tok/s p50 ...` — `tools/spec_speed.py`: N sessions prefill a
  ctx-token code snapshot, then all N decode 300 tokens together from the cached prefix; the per-stream number is the
  median over the N streams of (completion_tokens − 1) / (e2e − ttft).
- `[ 32k depth=0.95 rep=0] PASS prompt=32596 ttft=15.953s` — `tools/code_needle.py`: one needle at the given depth
  with 3 look-alike decoys, exact string match; `ttft` is the cold prefill through the server.
- `KL 0.00235 top1 0.9898 ppl x1.0013 kl16 0.00235` — `tools/qwentin_e2e.py score`: mean KL(bf16 ‖ engine) over the
  rows' next-token distributions, share of positions with the same argmax, perplexity ratio, and the KL on
  fp16-rounded engine logits (like-for-like with the exllamav3-side runs). The `tag ... dKL [CI] ... dtop1 pt [CI]`
  tables are `tools/e2e_frontier.py --vs`: paired differences with a 95 % block-bootstrap confidence interval
  (4096-token blocks, 2000 resamples).
