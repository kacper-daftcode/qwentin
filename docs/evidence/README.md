# Evidence pack — the logs behind the README's numbers

Every headline number in the top-level `README.md` (pinned at **v0.10**) is a line in one of the files in this
directory. The files are verbatim copies of the research project's (`kacper-daftcode/qwentin-internal`) `logs/`,
`notes/` and `runs/` — nothing was edited or trimmed (the paths `/mnt/nvme/...` and `/root/work/venv/bin/python` are
the measurement box's). `scripts/` holds the exact shell scripts that produced each log, so every flag of every gate is
on record. The narrative around the numbers is [`docs/NOTES.md`](../NOTES.md); the release notes, [`CHANGELOG.md`](../../CHANGELOG.md).

## Environment of the 2026-10-17 measurements

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

Earlier libraries that appear in the logs (all on `main`, each the gated predecessor of the next):

| md5 (12) | commit | what |
|---|---|---|
| `3ff196fece3f` | `9da2d3b` | the 10-16 served line (`--kv-t4 --spec` defaults) |
| `2bb8e6d81d9e` | `0f74130` | + cp.async-staged fp16 trellis attention behind `TQ_T4_ATTN_F16` (landed by `cand4-all.log`) |
| `e388864ae443` | `14be568` | + CUDA-graph decode step, concurrent GEMM groups, 128-token prefill GEMM units (landed by `cand5-all.log`) |
| `c9d5269c2963` | `11f5879` | + DeltaNet state formats `--dstate` (landed by `cand6-all.log`); `final-*.log` ran on this library with the four opt-ins as flags |
| `e454a29c3caf` | `2cf671c` | the opt-ins as defaults — the served line of the morning (landed by `cand7-all.log`); gate 1 of that log proves it bitwise equal to `c9d5269c2963` + flags |
| `484bdb825c36` | `b2e7a3f` | + the gated-DeltaNet prefill kernels, `TQ_GDN_PREFILL` default 1 — **the v0.10 served line** (landed by `cand8-all.log`); gate 1 of that log proves it bitwise equal to `e454a29c3caf` on the paged-decode logits with the flag on and with `TQ_GDN_PREFILL=0` (the off-switch) |

## Which file backs which README number

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

## Re-running each gate by hand

`bench/reproduce.sh` (README "Verify") runs the served line's gates in one go. The individual commands behind each
log line — moved here from the 2026-10-17 README's "Verify" section. Which tool reproduces which number (the HTTP tools take `--base`/`--base-url` of a running
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
# key splits). The served env in front (the bare-engine tools default TQ_KV_T4_CODEBOOK to the shipped codebook;
# the engine's own fallback is the research project's path); ~2 min for both lengths.
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py --lib build-qwen/libforward_qwen.so \
    --tqf /path/to/qwen3_8-27b-exl3-opt5.75-mtp.tqf --model-dir /path/to/Qwen3.8-27B --ctx 4096 --n 1,3,8 --no-pin
TQ_KV_V4=1 TQ_KV_T4=1 python3 tools/spec_paged_check.py ... --ctx 32768 --n 1,3,8 --distinct 3 --no-pin

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
