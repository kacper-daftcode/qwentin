# Evidence pack — the logs behind the README's numbers

Every headline number in the top-level `README.md` (pinned at **v0.14**) is a line in one of the files in this
directory. The files are verbatim copies of the research project's (`kacper-daftcode/qwentin-internal`) `logs/`,
`notes/` and `runs/` — nothing was edited or trimmed (the paths `/mnt/nvme/...` and `/root/work/venv/bin/python` are
the measurement box's). `scripts/` holds the exact shell scripts that produced each log, so every flag of every gate is
on record. The narrative around the numbers is [`docs/NOTES.md`](../NOTES.md); the release notes, [`CHANGELOG.md`](../../CHANGELOG.md).

## v0.14 (2026-10-23): what was re-measured

v0.14 re-measured every served-line cell of the README's 32 GB table on the tagged library itself (`8f42ffdde443` = main
`5ab8417`, engine `49bc602`; the served reduced-vocabulary file), each speed cell twice in one run (`rel14-table.log`,
`scripts/rel14-table.sh`): 576 000 pooled tokens at 30 138 MiB (30 586 after everything), needle 6/6 (TTFT 10.94–10.97 s @32K,
56.76–56.89 s @128K), A/B 10/10 with `cand21`'s texts, **216.8 / 164.0–161.6 / 127.2–123.4 / 89.1–89.3 / 69.4–68.3 tok/s per
stream @32K spec on at N = 1 / 2 / 4 / 8 / 12** (66.0 / 62.9 / 57.7 / 49.1 / 42.5 off; 830 / 829 aggregate at 12), **125.5–125.9 /
103.3 @128K at N = 1 / 2** (58.5 / 53.1 off). Everything since v0.13 is bitwise (the three levers' gates `cand31-all.log` and
`cand32-all.log`: paged logits max |d| 0, the prefill wave's 4-wave logits = the reference dumps, T1 93.9 / 91.5, the A/B texts
= `cand21`'s), so the quality rows, the 24 GB variant, the 16 GB table (v0.12's) and the vLLM columns of 2026-10-17 carry over
unchanged.

## v0.13 (2026-10-22): what was re-measured

v0.13 re-measured every served-line cell of the README's 32 GB table on one library (`9be3c8e7513d` = main `fbf26a6`,
engine `fec7211` — the tagged library `b2e93405f8a6` differs only in the norm grid of waves with ≥ 16 rows, bitwise; the
served reduced-vocabulary file), each speed cell twice in one run (`rel13-table.log`, `scripts/rel13-table.sh`): 576 000 pooled
tokens at 30 138 MiB (30 586 after everything), needle 6/6 (TTFT 11.36–11.38 s @32K, 58.4–58.5 s @128K), A/B 10/10 with
`cand21`'s texts, **212.6 / 165.8 / 119.8 / 88.4 / 64.3 tok/s per stream @32K spec on at N = 1 / 2 / 4 / 8 / 12** (65.8 /
62.5 / 57.2 / 48.4 / 41.7 off; 774 / 788 aggregate at 12), **123.6 / 102.2 @128K at N = 1 / 2** (57.9 / 52.7 off). The later
`cand-1218` (one norm block per row from 16 rows up) touches only the waves of ≥ 16 rows, bitwise and speed-neutral on the
verify wave at 8 streams; its gate read the TTFT at 11.35 / 58.5–58.7 s on the tagged library. The 16 GB table is v0.12's
(nothing of v0.13 reaches a 2-slot line but the bitwise-faster prefill norm). What carries over from v0.12 unchanged: the
quality rows (the kernels compute what v0.12's did), the 24 GB variant, the vLLM columns of 2026-10-17.

## v0.12 (2026-10-21): what was re-measured

v0.12 re-measured every served-line cell of the README on its own library (`5f9cee32da7c` = main `a0834e5`, engine
`40892ff`) and the version-2 opt-5.75 `.tqf`, each speed cell twice in one run (`rel12-table.log`, `scripts/rel12-table.sh`),
and measured the new "One 16 GB card" table on the `t45v4` model (`cand24-all.log` gate 3 + `rel12-p16.log`). The quality
rows of the 32 GB table were re-run on this library at 4K and 32K (bitwise the 2026-10-17 rows) with a new 32K seed-2 cell
(`cand22-all.log` gate 1, `e2e/qw-cand1211_*.json`); T1 was run with speculative decoding on for the first time
(`cand23-all.log` gate 3). The vLLM columns carry over from 2026-10-17.

| item | value |
|---|---|
| engine source | `src/forward_qwen.cu` + `src/engine/*.cuh` at commit `40892ff` (2026-10-21 09:26 UTC; patches 0109–0127 of the research repo since v0.11's `dd6d21e`); `a0834e5` adds docs and the t45v4 recipe only |
| served library | `build-qwen/libforward_qwen.so`, md5 `5f9cee32da7c`, built from `40892ff`; landed by `cand23-all.log` (199 / 211 kernels SASS-identical to the previous library `0f4f517442ca`, the 12 that differ are the three segment-attention kernels of the fix; paged-decode logits bitwise) |
| model files | the 32 GB line: the v0.11 version-2 opt-5.75 `-mtp4.tqf` (unchanged); the 16 GB line: `qwen3_8-27b-exl3-t45v4-mtp4.tqf` (16.70 GB, sealed v2, `cand24-all.log` gate 0) |
| toolkit, driver, GPU, Python | unchanged from v0.10: nvcc 13.1 V13.1.115, driver 610.57.04, RTX 5090 at 550 W, Python 3.14.4 with torch 2.14.0+cu130, transformers 5.17.0 |

| README statement | file : line | note |
|---|---|---|
| 576 000 pooled tokens at 29 898 MiB (30 300 after a 128K prefill and the 8-stream runs) | `rel12-table.log` (`server up ... 29898 MiB`, `... MiB after everything`) | +4 MiB vs v0.11: the drafter's normalized-row buffer |
| needle 6/6, **TTFT 11.6 s @32K** (11.646 / 11.629 / 11.644 s) **/ 60.1 s @128K** (59.909 / 60.126 / 60.253 s) | `rel12-table.log`, the six `[ 32k/128k depth=... ] PASS` lines | within 0.4 s of v0.11 (prefill kernels unchanged) |
| spec A/B 10/10 identical, the same texts as `cand21` (and every gate since `cand12`) | `rel12-table.log`, `[ab] ...`, `[ab-vs-cand21] ...` | |
| 179 / 153 / 114 / 82 tok/s per stream @32K spec on (N = 1 / 2 / 4 / 8; the mean of two readings), 66 / 62 / 57 / 48 off, 114 / 100 @128K (N = 1 / 2; 58 / 52 off) | `rel12-table.log`, the `[speed] ...` lines (`--repeats 2`) | 3.83 tokens per round @32K N = 1 (v0.11: 3.78), 2.82 @128K |
| T1 EvalPlus with speculative decoding on: HumanEval/+ 93.9 / 91.5, MBPP/+ 90.7 / 77.8 | `cand23-all.log` gate 3 (`[t1] ...` lines); the bug's before / after: `cand23-t1-probe-served-spec.log` (53.7 / 53.0, 75 / 164 degenerate) and `cand23-t1-probe-served-nospec.log` (93.9 / 91.5) | |
| quality 0.0021–0.0027 / 99.0–99.1 % | `e2e/qw-cand1211_4k_prefill.json` (0.00235 / 98.97 %), `qw-cand1211_32k_prefill.json` (0.00209 / 99.05 %), `qw-cand1211_32k_s2_prefill.json` (0.00256 / 98.96 %), the v0.10 decode-path rows | the first two are bitwise the 2026-10-17 rows (`cand22-all.log` gate 1, `[e2e] ... bitwise True`) |
| **16 GB**: 51K tokens at 15 604 → 15 922 MiB, needle 3/3 at 49K (TTFT 18.8 s), 198 / 182 tok/s @32K / 49K (78 / 77 off), T1 HumanEval/+ 93.9 / 89.6 | `cand24-all.log` gate 3 (`-- new:` block) | the previous model `t45-s425` on the same line and run: 15 654 → 15 954 MiB, 3/3, 193 / 178 (`-- old:` block) |
| **16 GB**: needle 3/3 at 32K (TTFT 12.0 s), 198 tok/s @32K N = 1 (twice), 350 aggregate at N = 2, 15 894 MiB after | `rel12-p16.log` (`scripts/rel12-p16.sh`) | |
| **16 GB** quality 0.0075–0.0100 / 98.0–98.3 % | `e2e/qw-t45v4_4k_prefill.json` (0.00921 / 97.99 %), `qw-t45v4_32k_prefill.json` (0.00750 / 98.32 %), `qw-t45v4_32k_s2_prefill.json` (0.00999 / 98.09 %); `qw-t45s425_*.json` for the previous model | `cand24-all.log` gate 1 (`[gate1] ... better`) |

Carried over from v0.10 / v0.11 without re-measurement: the decode-path quality rows (the decode kernels are v0.10's bit
for bit except the fixed NaN write), the 8-session mix, the 24 GB variant and the vLLM columns (2026-10-17).

## v0.11 (2026-10-20): what was re-measured

v0.11 re-measured every served-line cell of the README on its own library and model file — capacity, needle, TTFT,
spec A/B and the per-stream speed table with the commands of `scripts/final-speed.sh` (`cand16-all.log` gate 2). The
rows it did not re-measure carry over from v0.10 for the reasons at the end of this section; the sections after it are
the v0.10 record.

| item | value |
|---|---|
| engine source | `src/forward_qwen.cu` + `src/engine/*.cuh` at commit `dd6d21e` (2026-10-20 07:14 UTC; patches 0084–0108 of the research repo since v0.10's `b2e7a3f`) |
| served library | `build-qwen/libforward_qwen.so`, md5 `158edd06753376d9e4671fb96d86a803` (the logs print `158edd067533`), built from `dd6d21e`; landed by `cand15-all.log` (all 398 kernels' SASS identical to the previous library, paged-decode logits bitwise) |
| model file | the v0.10 opt-5.75 `-mtp4.tqf` re-converted to format version 2 — the same payload byte for byte plus the 80-byte extension (`cand16-all.log` gate 0: `inspect_tqf.py --verify`, `cmp` of the payloads); the engine's paged-decode logits are bitwise equal on both files (gate 1) and the served line's A/B texts equal the run on the version-1 file (gate 2) |
| toolkit, driver, GPU, Python | unchanged from v0.10 (next section): nvcc 13.1 V13.1.115, driver 610.57.04, RTX 5090 at 550 W, Python 3.14.4 with torch 2.14.0+cu130, transformers 5.17.0 |

| README statement | file : line | note |
|---|---|---|
| 576 000 pooled tokens at 29 894 MiB, the startup line | `cand16-all.log` gate 2 (`server up ... 29894 MiB`, `batched server: pool blocks=4500 ...`) | unchanged since v0.10 |
| needle 6/6, **TTFT 11.6 s @32K** (11.651 / 11.62 / 11.632 s) **/ 59.7 s @128K** (59.651 / 59.721 / 59.8 s) | `cand16-all.log` gate 2, the six `[ 32k/128k depth=... ] PASS` lines | within 0.2 s of `cand14-all.log` / `cand15-all.log` |
| spec A/B 10/10 identical, 128 vs 61 tok/s, 3.39 tokens per round | `cand16-all.log` gate 2, `[ab] ...` lines; `[ab-vs-v1]` = the texts of the version-1 file's run (`cand15`) | |
| 177 / 153 / 106 / 79 tok/s per stream @32K spec on (N = 1 / 2 / 4 / 8), 66 / 62 / 57 / 48 spec off, 115 / 99 @128K (N = 1 / 2) | `cand16-all.log` gate 2, `[speed] ...` lines | `tools/spec_speed.py` on the vLLM-checkout corpus, one prompt per cell; N = 1 / 8 across the gates since v0.10: 175–177 / 78–82 (`cand9`–`cand12`, `cand14`–`cand16`) |
| the decode path is v0.10's bit for bit | `rel11-v010-bitwise.log` | `tools/paged_logit_ab.py` (401 decode steps, N = 3 / 8): the v0.11 library = the v0.10 library (`484bdb825c36`), max \|d\| 0, with the default flags and with `TQ_PATTN_GQA=0` (the tool decodes token by token, so the prefill attention is not in it) |
| why the spec-on cells moved: 3.78 vs 3.56 tokens per round @32K N = 1, 2.85 vs 3.29 @128K | `diag-128k-pattn.log` | the same library and line with `TQ_PATTN_GQA=0` (v0.10's prefill attention, bitwise per `cand9-all.log` gate 1): 166.9 tok/s in 84 rounds @32K and 129.9 in 91 rounds @128K — v0.10's values (`cand7-all.log`: 166.9 and 132.8 tok/s, 91 rounds @128K). Per round: ~21 ms @32K, ~25 ms @128K with either kernel |
| `bench/reproduce.sh`: 176 / 79 tok/s (N = 1 / 8, spec on; 65 / 48 off) with `CORPUS` = the vLLM checkout, ~115 / 75 on this repository's own code (2.45 tokens per round); needle 3/3, A/B 10/10 in both | `rel11-reproduce.log`, `results-rel11-own.json` (run 1, the script's defaults), `results-rel11-vllm.json` (run 2) | the release worktree (`dd6d21e` + the release docs) on GPU 1, the version-2 file; the library the script built (`build-qwen-repro/`, md5 `45cb462773de`) has all 398 kernels' SASS identical to the served one; run 2 reads the same 79 rounds @32K N = 1 as `cand16` |

Carried over from v0.10 without re-measurement: the quality rows (the decode path is the one they were measured on, and
the new prefill attention's prefill rows are within the 95 % CI of v0.10's, `cand9-all.log` gate 2), T1 EvalPlus, the
8-session mix, the 16 GB / 24 GB variants and the vLLM columns (2026-10-17).

## v0.10 (2026-10-17): environment of the measurements

Two engine commits appear in this pack, and the README's table footer names both. **`2cf671c`** (06:59 UTC) is the
served stack of the morning: the kernels every decode, quality and capacity number was measured with
(`cand7-all.log`, `final-*.log`). **`b2e7a3f`** (16:45 UTC, v0.10) adds one lever, the gated-DeltaNet prefill kernels
(`TQ_GDN_PREFILL=1`, default on), which changes only the paged prefill waves: gate 1 of `cand8-all.log` proves the new
library bitwise equal to `e454a29c3caf` on the paged-decode logits (N = 3 / 8, max |d| 0) with the flag on *and* off,
and gate 2 the spec exactness on the new defaults, so **decode is bitwise unchanged** and the decode / quality rows
measured on `2cf671c` carry over to v0.10 unchanged. The TTFT rows are the one thing the lever moves (15.9 → 13.3 s
@32K, 86.6 → 77.0 s @128K) and are therefore quoted from `cand8-all.log`.

| item | value |
|---|---|
| engine source, v0.10 (TTFT rows, capacity re-confirmed) | `src/forward_qwen.cu` at commit `b2e7a3f` (`TQ_GDN_PREFILL` default 1 / `--gdn-prefill` default on, 2026-10-17 16:45 UTC; patches 0079–0083 of the research repo); the commits after it on `main` are documentation / tools / tests only |
| served library, v0.10 | `build-qwen/libforward_qwen.so`, md5 `484bdb825c3652a7df51c508e89af5b2` (the logs print `484bdb825c36`), built from `b2e7a3f` with the CMakeLists.txt of this repo; `cand8-all.log` gate 3 is this library on the served line |
| engine source of the decode / quality rows | `src/forward_qwen.cu` at commit `2cf671c` (`2cf671cc1b4437647e6012da2c6a773491b6b7c0`, 2026-10-17 06:59 UTC); the kernels of `b2e7a3f` outside the prefill waves are these, bit for bit (`cand8-all.log` gate 1) |
| served library of the decode / quality rows | `build-qwen/libforward_qwen.so`, md5 `e454a29c3caf7d131ba46e8d9d9bdf07` (the logs print the first 12 hex digits: `e454a29c3caf`), built from `2cf671c` with the CMakeLists.txt of this repo |
| CUDA toolkit | 13.1 — `nvcc` `Cuda compilation tools, release 13.1, V13.1.115`, `Build cuda_13.1.r13.1/compiler.37061995_0` (`/usr/local/cuda-13.1/bin/nvcc`) |
| driver | 610.57.04 |
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB, 550 W power cap (`nvidia-smi` power.limit 550.00 W); one card per server |
| Python | 3.14.4 (`/root/work/venv`); torch 2.14.0+cu130, transformers 5.17.0, safetensors 0.8.0, numpy 2.5.3 |
| models | `Qwen/Qwen3.8-27B` (original bf16, the quality reference), `Qwen/Qwen3.8-27B-FP8` (tokenizer / chat template of the served line, and the FP8 reference row), EXL3 checkpoints quantized with upstream exllamav3 from the bf16 original (`tools/recipes/*.yaml`), `.tqf` files written by `tools/convert_qwen_tqf.py --mtp-exl3` |
| held-out corpus | the `vllm/` package directory of a vLLM checkout — see `docs/QUALITY.md` for the exact commit and the caveat |
| date | 2026-10-17 (UTC times in the logs); the vLLM baselines 07:51–08:50 UTC the same day |

Builds are code-reproducible, not byte-reproducible: nvcc embeds per-build-directory metadata, so a rebuild of the
same commit gets a different md5 with identical SASS; the md5s here identify the binaries the gates ran.

Earlier libraries that appear in the logs (all on `main`, each the gated predecessor of the next):

| md5 (12) | commit | what |
|---|---|---|
| `3ff196fece3f` | `9da2d3b` | the 10-16 served line (`--kv-t4 --spec` defaults) |
| `2bb8e6d81d9e` | `0f74130` | + cp.async-staged fp16 trellis attention behind `TQ_T4_ATTN_F16` (landed by `cand4-all.log`) |
| `e388864ae443` | `14be568` | + CUDA-graph decode step, concurrent GEMM groups, 128-token prefill GEMM units (landed by `cand5-all.log`) |
| `c9d5269c2963` | `11f5879` | + DeltaNet state formats `--dstate` (landed by `cand6-all.log`); `final-*.log` ran on this library with the four opt-ins as flags |
| `e454a29c3caf` | `2cf671c` | the opt-ins as defaults — the served line of the morning (landed by `cand7-all.log`); gate 1 of that log proves it bitwise equal to `c9d5269c2963` + flags |
| `484bdb825c36` | `b2e7a3f` | + the gated-DeltaNet prefill kernels, `TQ_GDN_PREFILL` default 1 — **the v0.10 served line** (landed by `cand8-all.log`); gate 1 of that log proves it bitwise equal to `e454a29c3caf` on the paged-decode logits with the flag on and with `TQ_GDN_PREFILL=0` (the off-switch) |

## v0.10: which file backs which number of its README

(The README of the `v0.10` tag; v0.11's re-measured cells are in the first section.)

| README statement | file : line | note |
|---|---|---|
| 29 894 MiB process memory, 12 slots, 4500 blocks = 576 000 pooled tokens, startup line `trellis-4 K + V ... 16.50 KiB/token ... fp16sr 80 MiB/slot ... speculative decoding ON` | `cand8-all.log` gate 3 (`server up ... 29894 MiB`, `batched server: pool blocks=4500 ...`); the same two lines in `cand7-all.log` gate 3 | library `484bdb825c36` = `b2e7a3f`, pure defaults; command in `scripts/cand8-all.sh` (gate 3). The lever adds no memory: `cand7` (`e454a29c3caf` = `2cf671c`) read the same 29 894 MiB |
| needle 6/6 at 32K / 128K, **TTFT 13.3 s @32K** (32 596–32 602 tokens: 13.332 / 13.303 / 13.31 s) **/ 77.0 s @128K** (130 903–130 905 tokens: 76.936 / 77.048 / 77.156 s) | `cand8-all.log` gate 3, the six `[ 32k/128k depth=... ] PASS prompt=... ttft=...` lines | `eval/code_needle.py` = `tools/code_needle.py`, 3 depths × 3 decoys; library `484bdb825c36` = `b2e7a3f`. **Before the DeltaNet prefill lever:** 15.9 s @32K / 86.6 s @128K, the same six lines in `cand7-all.log` gate 3 (library `e454a29c3caf` = `2cf671c`) — the README quotes both |
| spec A/B 10/10 identical, 126 vs 60 tok/s, 3.39 tokens per round (`cand7`: 121 vs 59, 3.39) | `cand8-all.log` gate 3, `[ab] ...` lines; `cand7-all.log` gate 3 | `tools/spec_server_ab.py --prompts 10 --max-tokens 300` |
| 167 / 81 tok/s per stream @32K N = 1 / 8, 133 @128K N = 1 (the gate's own run: 166.9 / 80.6 / 132.8) | `cand7-all.log` gate 3, `[speed] ...` lines | `tools/spec_speed.py`; library `e454a29c3caf` = `2cf671c`. The v0.10 gate (`cand8-all.log` gate 3, library `484bdb825c36`) ran the same tool on the same line and read **157.2 / 73.8** — decode is bitwise the same path (gate 1), so the gap is run-to-run: the README keeps 167 / 81 (`final-speed.log` (R) / `cand7`) as the table values and states the spread of the gate runs — **N = 1: 157–169** (`cand8` 157.2; `final-speed` (R) 165.3, (D) 168.2; `cand7` 166.9; `cand4` 165.5, `cand5` 168.7 / 166.3, `cand6` 168.6 on the earlier default sets, which barely matter at N = 1), **N = 8: 74–81 on the served defaults** (`cand8` 73.8, `final-speed` (R) 78.8, `cand7` 80.6; the earlier default sets read 70–78 at N = 8 — `cand4` 77.8, `cand5` 71.2 / 75.9, `cand6` 69.9, `final-speed` (D) 75.3 — because the opt-in levers are worth ~+5 % there). The spec-off cells do not move (`final-speed` (R) 65.0 / 47.0) |
| the full speed table: 32K N = 1 / 2 / 4 / 8 spec on 167 / 143 / 109 / 81 (log: 165.3 / 143.0 / 109.0 / 78.8), spec off 65 / 62 / 56 / 47 (65.0 / 61.8 / 56.0 / 47.0), 128K N = 1 / 2 spec on 133 / 111 (131.7 / 111.4), aggregate @N = 8 619–632 | `final-speed.log` block `(R)` (the `(D)` block is the same binary's old defaults, 4100 blocks) | library `c9d5269c2963` + `TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 --dstate fp16sr --num-blocks 4500` = the `2cf671c` defaults (bitwise, `cand7-all.log` gate 1). The README rounds the N = 1 cell to the `cand7-all.log` run (166.9 → 167) and N = 8 to 81 (80.6); the 128K cell 133 = 132.8 (`cand7`) / 131.7 (`final-speed`) |
| quality: 0.0024 / 98.98 % (4K prefill), 0.0022 / 99.08 % (32K prefill), 0.0027 / 98.96 % (32K decode) | `final-e2e.log` rows `qw-final-def_4k_prefill` (0.00235 / 98.98), `qw-final-def_32k_prefill` (0.00220 / 99.08), `qw-final-def_32k_dec6` (0.00269 / 98.96) in the first table of each block (positions ≥ 0); the run files are `e2e/qw-final-def_*.json` | the engine-side yardstick `tools/qwentin_e2e.py` on library `c9d5269c2963`, old defaults; the `qw-final-all_*` rows are the same binary with the four opt-ins = the served defaults, within ±0.00003 KL of the `def` rows (the `[CI]` columns); `def` is bitwise the `qw-dstate-f32_*` reference rows (`[e2e] ... bitwise ... True`) |
| 0.0030 at 4K on the decode path | `e2e/qw-o5.75+kvT4L12fin2+e4m3_4k_dec8.json` (`kl_mean` 0.00295, `top1_agree` 98.83 %) — the 48-row × 4096 decode-only run of 2026-10-15 on the B2 branch library (`notes/session10-b2.md` in the research repo, table "final (v4, rebased) 4K decode B=8") | NOT the 4K decode row of `final-e2e.log`: that one (`qw-final-def_4k_dec8`, 0.00406 / 98.58 %) scores only the first 8 rows (32 760 tokens, a different and smaller sample; its `[CI]` column pairs it with the dstate-f32 run on the same 8 rows, bitwise). The 48-row decode measurement was not repeated on the landed library; the prefill rows and the 8-row decode rows were (bitwise = the 10-16 reference rows) |
| FP8 checkpoint: 0.0037 / 98.73 % at 4K; 0.0030 / 98.92 % · 0.0038 / 98.81 % on the two 32K samples | `e2e/fp8_4k.json` (`kl_mean` 0.00372, `top1_agree` 98.73 %); `e2e/fp8_32k.json`, `e2e/fp8_32k_s2.json` scored on positions ≥ 4096 (`tools/e2e_frontier.py --min-pos 4096`: 0.00302 / 98.92 and 0.00375 / 98.81); over all positions they read 0.00324 / 98.90 and 0.00357 / 98.83 | the two 32K FP8 cells in the README are the `--min-pos 4096` view, the served stack's 32K cells the all-positions view (≥ 4096 the served stack reads 0.00209 / 99.09 prefill and 0.00255 / 98.97 decode) — the FP8 gate passes in either view (paired dKL −0.00093 [−0.00118, −0.00071] prefill, −0.00047 [−0.00066, −0.00029] decode at ≥ 4096) |
| T1: HumanEval/+ 93.9 / 91.5 · MBPP/+ 89.7 / 77.5 (trellis stack) vs 93.9 / 90.9 · 90.2 / 77.0 (FP8 reference) | `evalplus/b2eng-t4-{humaneval,mbpp}.log` (`pass@1: 0.939 / 0.915`, `0.897 / 0.775`); `evalplus/fp8-vllm-ref-{humaneval,mbpp}.log` (`0.939 / 0.909`, `0.902 / 0.770`) | served stack: `serve_batched.py` on the opt-5.75 + trellis-4 line, 12 slots, 4500 blocks, spec off (2026-10-15 21:47 / 22:05 UTC, B2 branch library); reference: the FP8 checkpoint through vLLM TP2 (two 5090s, `qwen38-ref`) |
| 8-session mix: 24/24 OK, cold TTFT p50 16 / 97 / 365 s, follow-up 3–5 s, 16/16 prefix hits, 5 parked, decode 40 / 25 / 43 tok/s per client under prefill, 39 / 40 / 35 on follow-ups, vs 13 / 8 / 11 with `--decode-steps-per-wave 0` | `final-t5.log` block `g1` (the 40.3 / 24.7 / 43.1 and 38.6 / 40.3 / 34.5 rows); `e1` = `--decode-steps-per-wave 0` (12.6 / 8.4 / 11.1) | `eval/loadgen.py` of the research repo; library `c9d5269c2963` + the opt-in flags (= `2cf671c` defaults) |
| 16 GB variant: 51K tokens (`--num-blocks 400`) at 16 006 MiB after a full run, 184 tok/s @32K (76 without spec), 173 @49K (74), needle 3/3 at 48K | `p16-check.log` (`process 16006 MiB`, `[speed] ctx=32768 ... 184.1 / 75.9`, `ctx=49152 ... 172.8 / 73.7`, three `48k ... PASS`) | 2 slots, t45-s425 4.25 bpw `-mtp4.tqf`, 16 GiB process budget on a 5090 (not a 16 GB card). The script header names `14be568`; by the log's start time (01:17:46 UTC) `build-qwen/libforward_qwen.so` had already been swapped to `c9d5269c2963` (`cand6-all.log` LANDED 01:14:04); the log itself does not print the md5. Flags: the `export` line of `scripts/p16-check.sh` (the then-defaults of that binary: fp32 DeltaNet state, `TQ_T4_ATTN_F16` / `TQ_GRAPH` / `TQ_GEMM_FORK` unset) |
| 24 GB variant: a full 256K single session, 23 452–23 664 MiB, 161 tok/s @32K and 113 @240K with spec (63 / 48 without), needle 3/3 at 240K | `b2eng-t14-spec.log` block `p24 + spec` (`process 23452 MiB at start`, `[speed] ctx=32768 ... 161.3 / 62.2`, `ctx=240000 ... 112.8 / 48.0`, `240000 1/1 1/1 1/1 3/3`, `23664 MiB after`) | 2026-10-16 01:18 UTC, B2 branch library `build-qwen-b2t` (the 10-16 binary), 1 slot, 2000 blocks, emulated on a 5090 with a capped pool; `scripts/b2eng-t14-spec.sh` |
| the landing gates of patches 0045–0060 (TTFT 32K 17.1 → 16.0 s, 128K 93–98 → 87.4 s; each lever bitwise against its predecessor) | `cand4-all.log` (0045–0047), `cand5-all.log` (0048–0057), `cand6-all.log` (0058–0060); `cand7-all.log` (0061, the defaults flip) | each log: gate 1 `paged_logit_ab` bitwise (max |d| 0), gate 2 `spec_paged_check` ALL PASS, gate 3 the served line (needle / A-B / speed) |
| the landing gate of patches 0079–0083 — the gated-DeltaNet prefill kernels (`TQ_GDN_PREFILL=1` / `--gdn-prefill`): **TTFT 15.9 → 13.3 s @32K, 86.6 → 77.0 s @128K; decode bitwise unchanged** | `cand8-all.log` (2026-10-17 16:34–16:44 UTC; `scripts/cand8-all.sh`) | gate 1: `[plab] N=3/8: new defaults == served: True (max |d| 0); new + TQ_GDN_PREFILL=0 == served: True (max |d| 0)` — the v0.10 library against `e454a29c3caf` with the lever on and off; gate 2: spec exactness 4K + 32K ALL PASS on the new defaults; gate 3: the served line, 29 894 MiB, needle 6/6 with the TTFT above, A/B 10/10, 157.2 / 73.8 tok/s, 0 error lines; `LANDED ... = 484bdb825c36 ... main = b2e7a3f`. The bare-engine wave numbers behind it (DeltaNet 14.95 → 2.76 ms per 128-column wave; 32K wave 65.7 → 55.0 ms, 128K 116.5 → 106.0 ms; one-slot prefill 16.0 → 13.2 s / 87.5 → 76.7 s) are in `notes/session11-gdn.md` of the research repo |
| the vLLM columns: FP8 does not fit one 5090 (OOM after 27.48 GiB), NVFP4 147–179K tokens / 71.5 / 51.4 tok/s / TTFT 3.1 / 25.2 s, FP8 TP2 493K / 76.8 / 60.3 / 50.4 / 4.4 / 24.1 s, needle 6/6 | `baselines-vllm-20261017.md` — the summary table (a), the exact `docker run` / `vllm serve` commands, image digest `nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294` = vLLM `0.32.0rc1.dev281+g49cb2bbf7`, the NVFP4 checkpoint's provenance (llm-compressor 0.14.0, recipe, the fused-global-scale repair), the caveats (c) | measured 2026-10-17 on the same cards; the per-run vLLM logs (`logs/baseline-*.log`) and JSON results (`runs/baseline/*.json`) stay in the research repo (listed at the end of the note) |

Not in this pack (no 2026-10 gate log exists for them): the legacy FP6 single-stream table (2026-08 build, CUDA 13.3 /
driver 595), the RTX PRO 6000 rows and the 2026-06 batched-server rows — they are carried over from the 2026-08 README
unchanged and are labelled as such in their footers. (The 2026-08 scale sweep of the residual-stream steering hook
moved out of this repository together with the steering maps it was measured for.)

## From v0.10 to v0.11: the landing gates

| change | file | what the gate proved |
|---|---|---|
| `TQ_PATTN_GQA=1` default — the GQA-grouped prefill attention on the trellis pool (main `9c0a046`, library md5 `c40173fefe9a`) | `cand9-all.log` (2026-10-17 17:19–18:06 UTC; `scripts/cand9-all.sh`) | gate 1: `TQ_PATTN_GQA=0` bitwise = the v0.10 library (N = 3 / 8, max \|d\| 0); gate 2: the e2e prefill rows within the 95 % CI at 4K and 32K (dKL −0.00000 … +0.00003, top-1 CIs across 0, also on positions ≥ 16K / 24K); gate 3: spec exactness 4K + 32K ALL PASS; gate 4: the served line, 29 894 MiB, needle 6/6 with **TTFT 11.8 s @32K / 60.4 s @128K**, A/B 10/10, 175 / 79 tok/s at N = 1 / 8 |
| the trellis-KV codebook compiled into the engine (`src/kv_t4_codebook.inc`; `TQ_KV_T4_CODEBOOK` only overrides it; engine `81b8453`, library md5 `5734b39678c6`) | `cand10-all.log` (2026-10-18 08:56–09:06 UTC; `scripts/cand10-all.sh`) | gate 0: `--kv-t4-codebook` with a missing file is refused; gate 1: the built-in table and the shipped file each bitwise = the `9c0a046` library reading the measurement box's file (N = 3 / 8, max \|d\| 0), and `TQ_KV_T4_CODEBOOK=random12.f32` is still read (= the old library's rows, ≠ the lloyd_kv12a rows); gate 2: spec exactness 4K + 32K ALL PASS; gate 3: the served line with no codebook flag — `codebook built-in lloyd_kv12a`, 576 000 pooled tokens, 29 894 MiB, needle 6/6 (TTFT 11.8 / 60.4 s), A/B 10/10, 176 / 78 tok/s at N = 1 / 8 |
| `TQ_GEMM_FORK_MAXN=64`, stale comments and dead code, `qwn_ot_*` → `qwn_rmap_*` (main `364aa9c`, library md5 `4a42c129072d`) | `cand11-all.log` (2026-10-18 16:26–16:36 UTC) | gate 1: bitwise = the `d842e36` library at N = 3 / 8, with `TQ_GEMM_FORK_MAXN=1000000` (the previous behaviour) and at 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the served line, needle 6/6 (TTFT 11.6 / 59.7 s), A/B 10/10, 177 / 81 tok/s; gate 4: the hook through the new names, `residual_map_check.py` PARITY OK |
| the engine in 26 parts `src/engine/*.cuh`; tool defaults without development-box paths | `scripts/cand12-all.sh` (16:40–17:06 UTC); its log stays in the research repository — its first line records another project's job on the shared GPUs | gate 1: all 398 kernels' SASS identical to the unsplit file (`analysis/sass_diff.py` of the research repo), bitwise as `cand11`; gates 2–3 as `cand11` (177 / 81 tok/s); gate 4 re-run after a wrong maps path in the script (recorded in the log), PARITY OK |
| `src/qwentin.h` (the C ABI as a generated header) | `cand13-all.log` (17:06–17:09 UTC) | declarations only: SASS identical, bitwise at N = 3, the served line's needle at 32K and A/B 10/10 |
| `.tqf` format version 2 (main `0d647f9`, library md5 `a3731a9854bd`) | `cand14-all.log` (17:09–17:18 UTC) | gate 0: the engine's BLAKE2b = `hashlib`, each refusal code; gate 1: SASS identical, bitwise on the version-1 file and version 2 = version 1; gate 2: spec exactness 4K; gate 3: the served line on the version-1 file (needle 6/6, A/B 10/10, 177 / 82 tok/s); gate 4: the server on the version-2 file, `qwn_probe` with `TQ_VERIFY_TQF=1` |
| ten experiment switches of the served path removed, their defaults hard-coded (main `dd6d21e`, library md5 `158edd067533` — **v0.11**) | `cand15-all.log` (2026-10-20 07:15–07:26 UTC) | gate 1: SASS identical, bitwise at N = 3 / 8 and 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the served line, needle 6/6 (TTFT 11.6–11.7 / 59.7–59.8 s), A/B 10/10, 177 / 80 tok/s |
| the served line on the version-2 `.tqf`; the v0.11 numbers | `cand16-all.log` (07:37–07:54 UTC) | see the first section |

## Since v0.11 (CHANGELOG "Unreleased")

| change | file | what the gate proved |
|---|---|---|
| the single-stream path retired: 138 → 55 entry points, 398 → 211 kernels, 119 → 34 `TQ_*` switches (main `cand-1207`) | `cand17-all.log` (2026-10-20; `scripts/cand17-all.sh`) | gate 0: the library exports exactly `src/qwentin.h`'s 55 entry points; gate 1: 187 kernels gone, 210 of the 211 left with identical SASS, the one that differs is the single-stream decoder's `k_tq_full_attn_decode`; paged-decode logits bitwise at N = 3 / 8, with the MTP drafter's trunk hooks on, and at 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the decoder (`qwn_decode`, FP6 file) bitwise — teacher-forced argmax, greedy continuations and the paged logits after loading its KV (`analysis/oracle_ab.py` of the research repo) — and `tools/paged_smoke.py` line for line; gate 4: an FP6 file on the batched path bitwise (paged logits) and served (needle 3/3, the same answers); gate 5: the served line as `cand16` (needle 6/6, A/B 10/10 with the same texts) |
| the `TQ_*` config struct — every switch parsed once in `src/engine/config.cuh` with one parse rule, the `[config]` line; a bf16 MTP head read past instead of loaded (the FP6 file: 21.07 → 20.28 GiB on the device); four parts renamed, the spec path's last leftovers removed (main `cand-1208`) | `cand18-all.log`, `cand18-matrix.log` (2026-10-20 10:39–10:54 UTC; `scripts/cand18-all.sh`, `scripts/cand18-matrix.sh`) | gate 0: the new parse rule against the old `getenv` expressions on 2050 (switch, input) pairs — no documented value and no unset switch parses differently; the 115 pairs that do are words, leading zeros, empty strings and the two presence-only debug switches (`analysis/config_parse_ab.py` of the research repo); gate 1: all 211 kernels' SASS identical, paged-decode logits bitwise at N = 3 / 8, with the drafter's trunk hooks on, and at 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the test decoder bitwise and `tools/paged_smoke.py` line for line, the served library loading the FP6 file's bf16 MTP head and the candidate reading past it; gate 4: the FP6 file on the batched path bitwise and served (needle 3/3, the same answers); gate 5: the served line's `[config]` line as in `docs/FLAGS.md`, needle 6/6, A/B 10/10 with `cand17`'s texts, 176 / 81 tok/s at N = 1 / 8; gate 6: 25 cells with one or two switches off their defaults (the off-switches, the two pins, the FP4 pool's four attention modes and page top-k, the embedding modes, `TQ_VERIFY_TQF`, a codebook file, the FP6 file with `TQ_NO_MTP=0`) bitwise against the served library under the same environment, each cell's `[config]` line naming exactly its switches |
| the last placeholder constants of bank 3 folded: `gc_use_ldsm`, `gc_use_p2p`, and `gc_kv_q4_fp32s` with the dead fp32 branch of the `q4s_*` helpers (main `cand-1209`) | `cand19-all.log`, `cand19-matrix.log`, `cand19-bank3.log` (2026-10-20 14:11–14:27 UTC; `scripts/cand19-all.sh`, `scripts/cand19-matrix.sh`) | gate 1: 184 of the 211 kernels keep their SASS bit for bit and each of the 27 that change is explained (`cand19-bank3.log`, written by `analysis/bank3_check.py` of the research repo): the two embedding lookups read `gc_embed_*` at the moved offsets, the other 25 are exactly the kernels that read the removed constant; paged-decode logits bitwise at N = 3 / 8, with the drafter's trunk hooks on, and at 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the test decoder (its attention is one of the 25) bitwise, `tools/paged_smoke.py` line for line; gate 4: the FP6 file bitwise and served (needle 3/3, the same answers); gate 5: the served line as `cand18` (needle 6/6, A/B 10/10 with its texts, 177 / 81 tok/s); gate 6: 28 cells bitwise against the previous library — `cand18`'s plus the FP4 pool's prefill at the default query tile and the E4M3-V pool's decode and prefill |
| the engine's defaults are the served line's; `qwn_config` in the C ABI (`qwn_init_with`), the server starts the engine through it (main `cand-1210`) | `cand20-all.log`, `cand20-matrix.log` (2026-10-20 17:50–18:06 UTC; `scripts/cand20-all.sh`, `scripts/cand20-matrix.sh`) | gate 0: the struct's layout against ctypes, the new defaults and the field rule (g++), the server's precedence on the real library, the 59 exports = `src/qwentin.h`; gate 1: all 211 kernels' SASS identical, and the candidate WITHOUT `TQ_PDL` / `TQ_KV_V4` / `TQ_KV_T4` bitwise = the previous library WITH them (paged-decode logits at N = 3 / 8, with the drafter's trunk hooks on, at 32K × 2 slots); gate 2: spec exactness 4K + 32K without `TQ_KV_T4` in the environment; gate 3: the test decoder bitwise, `tools/paged_smoke.py` line for line (the candidate's tool pinning the pool the previous library got by default), `--selftest` through `qwn_init_with`; gate 4: the FP6 file bitwise and served without any environment (needle 3/3, the same answers); gate 5: the served line with no `TQ_*` variable at all — `[config] every TQ_* switch at its default`, 576 000 tokens, 29 894 MiB, needle 6/6, A/B 10/10 with `cand19`'s texts, 176 / 81 tok/s; gate 6: `TQ_KV_T4=0` in the environment beats the server's default `--kv-t4`; gate 7: 28 cells with a fully explicit pool environment bitwise against the previous library |
| the paged MTP drafter reads the trunk hidden after the final norm — what the head was trained on — instead of the pre-norm residual (main `cand-1211`) | `cand21-all.log` (2026-10-21 02:46–02:57 UTC; `scripts/cand21-all.sh`) | gate 0: ruff, the CPU tests with and without the library, the 59 exports = `src/qwentin.h`; gate 1: all 211 kernels' SASS identical (host code only), paged-decode logits bitwise at N = 3 / 8, with the drafter's trunk hooks on (`--mtp-on`) and at 32K × 2 slots; gate 2: spec exactness 4K + 32K; gate 3: the served line with no `TQ_*` environment — 576 000 tokens at 29 898 MiB (+4 MiB: the normalized-row buffer), needle 6/6 (TTFT 11.7 / 60.0 s), A/B 10/10 with `cand20`'s texts (the outputs are unchanged by construction: the verify decides every token), **179.1 / 657.1 aggregate tok/s at N = 1 / 8 @32K (cand20: 176.3 / 644.3) — 3.85 / 18.32 tokens per round (3.80 / 17.65)** |
| **the segment-attention NaN of mixed key-split waves (position 384) fixed** — only split 0 writes an S_t = 1 column's final rows in `k_tq_paged_attn_q4_gqa_seg` / `_f16` / `_seg_t4` (main `cand-1212`) | `cand23-all.log` (2026-10-21 09:27–09:51 UTC; `scripts/cand23-all.sh`), the bug's before / after `cand23-t1-probe-served-{spec,nospec}.log` (`scripts/t1-probe.sh`), the 6-prompt repro `scripts/t1_repro.py` | gate 0: ruff, the CPU tests with and without the library, the 59 exports; gate 1: 199 / 211 kernels SASS-identical, the 12 that differ are the three kernels' instances, paged-decode logits bitwise at N = 3 / 8, with the drafter's hooks on and at 32K; gate 2: spec exactness at 4K and 32K as before AND at the unaligned contexts 300 / 100 (N = 1, 3; 40 / 60 rounds) on the trellis pool in both attention modes, the int4-V fp16 pool and the E4M3-V pool — the candidate passes every cell bitwise, the served library fails the trellis cells with NaN logits at the wave [383, 384, 385] (4 argmax flips); gate 3: the served line (576 000 tokens, needle 6/6, A/B 10/10 with `cand21`'s texts, 178.6 / 645.7 aggregate tok/s at N = 1 / 8 = 3.85 / 18.46 tokens per round), the 6-prompt repro clean, **T1 with speculative decoding on: HumanEval/+ 93.9 / 91.5, MBPP/+ 90.7 / 77.8, 0 degenerate answers** (the served library before the fix: 53.7 / 53.0 with 75 / 164 degenerate; `--no-spec` 93.9 / 91.5) |
| **the drafter's reduced vocabulary** — a DRAFT lm_head section in the `.tqf` (65 536 ids, 6-bit EXL3, 0.25 GB), `TQ_DRAFT_VOCAB` (main `cand-1214`) | `cand25-all.log` (2026-10-21 21:17–21:48 UTC; `scripts/cand25-all.sh`) | gate 0: the CPU tests, the exports, `inspect_tqf --verify` of the new file, its payload byte-identical to the served file's up to the end of the MTP section; gate 1: 211 / 211 kernels SASS-identical + the new `k_tq_batched_argmax_map`, paged-decode logits bitwise at N = 3 / 8, with the drafter's hooks on, on the new file with the section on and off, at 32K; gate 2: spec exactness 4K / 32K / 300; gate 3: the served line on the new file — 576 000 tokens at 30 138 MiB, needle 6/6, A/B 10/10 with `cand21`'s texts, the 6-prompt repro clean, **190.8 / 192.3 tok/s at N = 1 @32K (two readings; 178.6 before), 670.4 / 670.9 aggregate at N = 8 (645.7), 121.7 @128K (113.6)** at the same 78 / 130 rounds per 300 tokens, T1 HumanEval+ with spec on 93.9 / 91.5; gate 4: `TQ_DRAFT_VOCAB=0` on the new file = the stock drafts round for round; 60 traced prompts served vs new file: outputs 60/60 identical, 3.425 → 3.408 tokens per round (−0.5 %), 163.2 → 177.8 tok/s (+8.9 %) |
| **`--spec-k auto` = 6 at ≤ 2 streams and ≤ 64K with the reduced vocabulary** (`qwn_draft_vocab_size`, 60 entry points; main `cand-1215`) | `cand26-all.log` (2026-10-21 22:55–23:08 UTC; `scripts/cand26-all.sh`) | gate 0: the CPU tests, 60 exports = `qwentin.h`; gate 1: 212 / 212 kernels SASS-identical, paged logits bitwise at N = 3 / 8 and with the drafter's hooks on the served file; gate 2: the served line — the startup line `K=auto = 6/3/2 by active slots; draft vocabulary 65536 ids`, `/health` `draft_vocab` 65536, needle 6/6, A/B 10/10 with `cand21`'s texts, the 6-prompt repro clean, **209.6 / 209.5 tok/s at N = 1 @32K (4.62 tokens per round; 191 at K = 4), 673 / 685 aggregate at N = 8 (K = 3 unchanged), 122.6 @128K (K = 4 past 64K)**, T1 HumanEval+ with spec on 93.9 / 91.5; the served v2 file (no section) on the same server line stays at `4/3/2 ... draft vocabulary 0` and its 78 rounds |
| **the verify wave runs 33..64 columns in ONE pass over the weights** (`k_tq_exl3_wide_gemm<KB, 8, F32>`, the fp32 64-token units; `qwn_verify_onepass_cols`, 61 entry points; `--spec-k auto` = 3 above 8 slots; main `cand-1216`) | `cand27-all.log` (2026-10-22 00:57–01:26 UTC; `scripts/cand27-all.sh`) | gate 0: the CPU tests (84 with the library), 61 exports = `qwentin.h`; gate 1: 212 / 212 kernels SASS-identical + the 7 F32 instantiations, paged logits bitwise at N = 3 / 8 and with the drafter's hooks on the served file; gate 2: spec exactness vs plain decode on the waves the new units run — 40 / 36 / 60 columns at 4K, 48 at 32K, ctx 300 at 40 columns — and the standard 4K / 32K / 300 cells, 0 differing logits everywhere; gate 3: the served line — the startup line `K=auto = 6/3/3 ...; one-pass verify wave 64 columns`, `/health` `verify_onepass` 64, needle 6/6 (TTFT 11.5 / 59.0 s), A/B 10/10 with `cand21`'s texts, the 6-prompt repro clean, 212 tok/s at N = 1 @32K, 700 / 709 aggregate at N = 8 (unchanged paths), **800 / 800 aggregate at N = 12 @32K (67 tok/s per stream; 136 / 145 rounds per 300 tokens) vs 666 / 680 (55–57 per stream, K = 2 with the second pass) on the served library in the same gate**, 123.9 @128K, T1 HumanEval+ with spec on 93.9 / 91.5; the served library on the new server line reads `6/3/2 ... one-pass verify wave 32 columns` (the fallback) |
| **`--spec-k auto` adapts the draft depth to the recent acceptance** (`spec_p_hat`; 6 at ≤ 4 slots when ≥ 0.82, 4 at ≤ 6 when ≥ 0.78; server only, main `cand-1217`) | `cand28-all.log` (2026-10-22 02:49–03:24 UTC; `scripts/cand28-all.sh`) | gate 0: the CPU tests (84 with the library), 61 exports; gate 1: the served line (the served library `9be3c8e7`) — the startup line `K=auto = 6/3/3 ...; adaptive: 6 at <= 4 slots when p_hat >= 0.82 / 4 at <= 6 slots when p_hat >= 0.78`, `/health` `spec_p_hat`, needle 6/6, A/B 10/10 with `cand21`'s texts, the 6-prompt repro clean, the unchanged paths round for round as `cand27` (212 tok/s at N = 1, 4.62 / 6.52 / 18.6 tokens per round at N = 1 / 2 / 8, N = 12 and 128K), **the design-note answers at 4 / 6 streams with `k=table` vs `k=auto` 468 / 487 and 584 / 598 tok/s aggregate (a tie: p̂ 0.70–0.79), the code continuation 586 → 659 (+12.4 %) at 4 streams and 735 → 803 (+9.3 %) at 6**, T1 HumanEval+ with spec on 93.9 / 91.5 (its concurrency runs the adaptive K on code) |
| **the batched norms run one block per row from 16 rows up** (`tq_norm_grid`; host code only, main `cand-1218`) | `cand29-all.log` (2026-10-22 03:51–04:08 UTC; `scripts/cand29-all.sh`) + `cand29-wave-ab.log` | gate 0: the CPU tests (84 with the library), 61 exports; gate 1: 219 / 219 kernels SASS-identical, paged logits bitwise at N = 3 / 8 and with the drafter's hooks; gate 2: spec exactness at 40 / 36 columns and the standard 4K / 32K / 300 cells, 0 differing logits; gate 3: the served line — needle 6/6 (TTFT 11.35 / 58.5–58.7 s), A/B 10/10 with `cand21`'s texts, repro clean, 212 tok/s at N = 1 @32K and 4.62 / 6.52 / 18.6 tokens per round at N = 1 / 2 / 8 as before, 123.6 @128K, T1 with spec on 93.9 / 91.5. The speed evidence is the wave A/B/A on one GPU (`cand29-wave-ab.log`, 8 waves each from the slot blobs): **47.86 / 47.91 → 47.13 ms per 128-column prefill wave at 32K (−0.75 ms), 72.96 / 73.00 → 72.27 at 128K**, the 4 dumped waves' logits bitwise = the reference dumps at both contexts; e2e TTFT moves within the run-to-run band (the served library read 11.36–11.51 s at 32K across the night's runs) |
| **the DeltaNet recurrence's q / k norms without per-column block reductions** (`tq_warp_sumsq128` in the decode core and the verify's sequential core, the sequential core's 8-column prologue and barrier-free column loop; bitwise; main `cand-1221`) | `cand31-all.log` (2026-10-22 15:40–16:03 UTC; `scripts/cand31-all.sh`) + `cand31-round-ab.log` | gate 0: the CPU tests (84 with the library), 61 exports; gate 1: 204 / 219 kernels SASS-identical, the 15 that differ are the three DeltaNet-core families × 5 state formats, paged logits bitwise at N = 3 / 8 and with the drafter's hooks on the dv64k file (max |d| 0); gate 2: spec exactness vs plain decode at the 40 / 28 / 18-column fixed-K waves and the standard 4K / 32K / 300 cells, 0 differing logits; gate 3: the served line — needle 6/6 (TTFT 11.2 / 57.8 s), A/B 10/10 with `cand21`'s texts, repro clean, **215.0 / 215.0 tok/s at N = 1 @32K (212 before; 4.62 tokens per round as before), 709 / 727 aggregate at N = 8, 833 / 839 at N = 12, 125.0 @128K (123.6)**, T1 HumanEval+ with spec on 93.9 / 91.5. The speed evidence is the spec round on the frozen corpus at 32K, candidate vs the served library back to back (`cand31-round-ab.log`, 60 rounds, two A/B pairs): **21.46 / 21.50 → 21.22 / 21.22 ms at N = 1 K = 6 (−1.2 %), 32.91 / 32.93 → 32.03 / 32.05 at N = 8 K = 3 (−2.7 %), 45.02 / 45.07 → 43.73 / 43.76 at N = 12 K = 3 (−2.9 %)**; the commit replay 2.66 → 2.26 ms at N = 8. |
| **the trellis page encoder's step loop without the per-step shuffle chain + the prefill wave on the verify wave's fused launch set** (`k_t4_encode` 235 → 178 µs per 1024-tile launch; 208 fewer launches per wave; both bitwise; main `cand-1222`) | `cand32-all.log` (2026-10-23 09:49–10:19 UTC; `scripts/cand32-all.sh`) + `cand32-wave-ab.log` (`scripts/s17-fuse-ab.sh`) + `cand32-encoder-ab.log` (`scripts/s17-enc-ab.sh`, `tools/t4_encode_ab.py`) | gate 0: the CPU tests (84 with the library), 61 exports; gate 1: 217 / 219 kernels SASS-identical, the 2 that differ are `k_t4_encode` / `k_t4_encode_dev`, paged logits bitwise at N = 3 / 8 and with the drafter's hooks (max |d| 0); gate 2: spec exactness at the 40 / 28 / 18-column fixed-K waves and the standard 4K / 32K / 300 cells, 0 differing logits; gate 3: the served line — **needle 6/6 with TTFT 10.92–10.94 s at 32K (11.22 before) and 56.65–56.71 s at 128K (57.8)**, A/B 10/10 with `cand21`'s texts, repro clean, 216.8 / 216.9 tok/s at N = 1 @32K (4.62 tokens per round as before), 733 / 732 aggregate at N = 8, 834 / 843 at N = 12, 126.0 @128K, T1 HumanEval+ with spec on 93.9 / 91.5. The wave evidence: 8 waves per library, A/B/A on one GPU — the encoder alone 46.91 → 46.22 ms at 32K / 71.99 → 71.31 at 128K, encoder + fused launches **46.87 → 45.87 / 71.85 → 70.89**, the 4-wave logits bitwise = the reference dumps at both contexts; the encoder's codes on 8192 fixed rows identical to the previous library's. |
| the wikitext-2 KL table (weights only, exllamav3's `model_diff.py`) | `wiki2/*.log` (`scripts/s16-wiki2-kl.sh`, 2026-10-21 21:22–21:46 UTC) | docs/QUALITY.md § 7 |
| **the 16 GB model := t45v4** (docs only: the recipe, calibration set and the exllamav3 patch in `tools/`, VARIANTS / README / QUALITY; main `cand-1213`) | `cand24-all.log` (2026-10-21 10:10–10:35 UTC; `scripts/cand24-all.sh`), the e2e JSONs `e2e/qw-t45v4_*.json`, `e2e/qw-t45s425_*.json` | gate 0: `inspect_tqf --verify` of the new file, flags = the previous file's; gate 1: engine e2e rows (prefill mode) of t45v4 and t45-s425 on the same library at 4K / 32K s1 / 32K s2 — t45v4 better in every cell (ΔKL −0.00234 [−0.00311, −0.00169] / −0.00175 [−0.00234, −0.00128] / −0.00282 [−0.00427, −0.00156], Δtop-1 +0.16 / +0.17 / +0.26 pt); gate 2: spec exactness on the new file at ctx 4096 and 300 (N = 1, 2) bitwise; gate 3: the 16 GB line (2 slots, 400 blocks, `--park-ram-gb 16`, no `TQ_*` environment) on both files — t45v4 15 604 / 15 896 / 15 922 MiB (start / after the 49K needle / after everything), needle 3/3 at 49K, 198.4 / 181.6 tok/s @32K / 49K spec on (78.9 / 77.1 off), T1 HumanEval/+ 93.9 / 89.6 with spec on and 0 degenerate answers; t45-s425 15 654 / 15 946 / 15 954 MiB, 3/3, 192.5 / 178.2 (78.9 / 77.1) |

## Reading the logs

- `[plab] ... bitwise identical ... True (max |d| 0)` — `tools/paged_logit_ab.py` dumped 25 paged-decode steps of raw
  logits for N = 3 and N = 8 slots with two libraries (or one library with a lever on and off) and the tensors are
  equal element for element. This is the "bitwise" of the README.
- `[spec-check] ... ALL PASS` — `tools/spec_paged_check.py`: every verify column's logits equal the sequential
  decode steps' (0 different, max |d| 0), partial / zero commits, wrong drafts and parking included. Since
  `cand23` the gate runs it at `--ctx 300` and `--ctx 100` too (rounds crossing position 384, where the
  per-column key-split count first changes), and T1 through the served line with speculative decoding ON.
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

## Re-running each gate by hand

`bench/reproduce.sh` (README "Quick start") runs the served line's gates in one go: build → `.tqf` check → the served
line → needle (32K × 3 depths; TTFT = the cold prefill) → spec A/B (10 prompts, identical outputs expected) →
per-stream speed (32K, N = 1 / 8, spec on / off) → `bench/results-<date>-<commit>.json` with the environment block
(commit, library md5, driver, nvcc, torch). `--quality` adds the 4K / 32K KL / top-1 cells against the bf16 original
([`docs/QUALITY.md`](../QUALITY.md)), `--dry-run` prints every command, `CORPUS=<a vllm checkout>` gives like-for-like
spec-on speed. The individual commands behind each log line — moved here from the 2026-10-17 README's "Verify" section. Which tool reproduces which number (the HTTP tools take `--base`/`--base-url` of a running
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
# key splits). The served env in front (the trellis pool uses the engine's built-in codebook unless
# TQ_KV_T4_CODEBOOK names a file); ~2 min for both lengths.
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py --lib build-qwen/libforward_qwen.so \
    --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp.tqf --model-dir /path/to/Qwen3.8-27B --ctx 4096 --n 1,3,8 --no-pin
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py ... --ctx 32768 --n 1,3,8 --distinct 3 --no-pin
# and the unaligned short contexts (since cand23): the rounds cross position 384, the first change of a column's
# key-split count (nst / 2, nst = (pos + 128) >> 7), where a wave mixes S_t = 1 and S_t = 2 columns -- the cell the
# 2026-10-16..21 NaN hid in (the page-aligned contexts above never cross it with S_t = 1)
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py ... --ctx 300 --n 1,3 --rounds 40 --no-pin
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py ... --ctx 100 --n 1,3 --rounds 60 --no-pin

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
[`docs/QUALITY.md`](../QUALITY.md); it needs the bf16 27B and exllamav3 on the box. The
8-session load generator and the parking / prefix-cache probes (`eval/loadgen.py`,
`eval/park_probe.py`, `eval/prefix_cache_probe.py`) stay in the research repository
`kacper-daftcode/qwentin-internal`.
