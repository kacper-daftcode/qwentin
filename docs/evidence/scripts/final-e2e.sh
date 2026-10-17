#!/usr/bin/env bash
# The LANDED served library (main 11f5879, build-qwen/ c9d5269c) on the engine-side e2e yardstick (analysis/qwentin_e2e.py via
# eval/qw_e2e_run.sh = per-GPU flock per run, skip if the JSON exists), GPU 0:
#  (D) pure defaults -> must be BITWISE the dstate thread's fp32 reference rows (qw-dstate-f32_*, made by a flag-off build that was
#      bitwise main 9da2d3b): proves the served path of the landed binary is numerically what every session-10 gate measured.
#  (R) the three opt-in decisions at once: TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 TQ_DSTATE_BF16=4 (fp16 + SR state)
#      -> dKL [95 % CI] and dtop-1 vs (D) per cell (e2e_frontier.py --vs); the FP8-gate rule: KL below / equal within CI, top-1 CI across 0.
# Cells (the dstate gate's): 4K prefill (48 x 4096), 4K decode (8 rows lockstep), 32K prefill (6 x 32768), 32K decode (6 rows).
# Env = the B2 / cand3 / dstate e2e rows: opt-5.75 -mtp tqf, TQ_KV_V4=1 TQ_KV_T4=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1 TQ_PDL=1, ST default 3.
# usage: nohup logs/final-e2e.sh > logs/final-e2e.log 2>&1 &        (~2 h)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
TQF575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
GPU=0
cd $Q
unset TQ_KV_T4 TQ_PAGED_ATTN_ST TQ_KV_V4 TQ_NO_MTP TQ_PDL TQ_EMBED_HOST TQ_EMBED_FP8 TQ_KV_T4_ASYNC TQ_DSTATE_BF16 TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK
export TQ_KV_T4_CODEBOOK=$CB TQ_KV_V4=1 TQ_PDL=1 TQ_KV_T4=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1
echo "[final-e2e] start $(date -u +%T)  lib $(md5sum $LIB | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-50)"
D() { local tag=$1; shift; eval/qw_e2e_run.sh $GPU $tag --tqf $TQF575 --lib $LIB "$@"; }
R() { local tag=$1; shift; TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 TQ_DSTATE_BF16=4 eval/qw_e2e_run.sh $GPU $tag --tqf $TQF575 --lib $LIB "$@"; }
cmp_bitwise() { $PY - "$1" "$2" <<'PY'
import sys, torch
Q = "/mnt/nvme/qwen256k/runs/vsorig/e2e"
new, ref = sys.argv[1], sys.argv[2]
try:
    a = torch.load(f"{Q}/{ref}.pt"); b = torch.load(f"{Q}/{new}.pt")
    n = min(len(a["kl"]), len(b["kl"]))
    print(f"[e2e] {new}: per-token KL bitwise vs {ref}: {torch.equal(a['kl'][:n], b['kl'][:n])}  top1 identical: {torch.equal(a['agree'][:n], b['agree'][:n])}  (n {len(b['kl'])} / {len(a['kl'])})")
except Exception as e:
    print("[e2e]", new, "compare failed:", e)
PY
}
echo "== 4K: prefill mode (48 x 4096) and decode mode (8 rows)   $(date -u +%T)"
D qw-final-def_4k_prefill --seq 4096 --decode-from 4096
D qw-final-def_4k_dec8    --seq 4096 --decode-from 0 --batch 8 --rows 8
cmp_bitwise qw-final-def_4k_prefill qw-dstate-f32_4k_prefill
cmp_bitwise qw-final-def_4k_dec8    qw-dstate-f32_4k_dec8
R qw-final-all_4k_prefill --seq 4096 --decode-from 4096
R qw-final-all_4k_dec8    --seq 4096 --decode-from 0 --batch 8 --rows 8
$PY analysis/e2e_frontier.py --tags qw-final-def_4k_prefill qw-final-all_4k_prefill --vs qw-final-def_4k_prefill 2>/dev/null | tail -3
$PY analysis/e2e_frontier.py --tags qw-final-def_4k_dec8 qw-final-all_4k_dec8 --vs qw-final-def_4k_dec8 2>/dev/null | tail -3
echo "== 32K: prefill mode (6 x 32768) and decode mode (6 rows)   $(date -u +%T)"
D qw-final-def_32k_prefill --seq 32768 --decode-from 32768
D qw-final-def_32k_dec6    --seq 32768 --decode-from 0 --batch 6
cmp_bitwise qw-final-def_32k_prefill qw-dstate-f32_32k_prefill
cmp_bitwise qw-final-def_32k_dec6    qw-dstate-f32_32k_dec6
R qw-final-all_32k_prefill --seq 32768 --decode-from 32768
R qw-final-all_32k_dec6    --seq 32768 --decode-from 0 --batch 6
for mp in 0 4096 16384 24576; do
  $PY analysis/e2e_frontier.py --min-pos $mp --tags qw-final-def_32k_prefill qw-final-all_32k_prefill --vs qw-final-def_32k_prefill 2>/dev/null | tail -3
  $PY analysis/e2e_frontier.py --min-pos $mp --tags qw-final-def_32k_dec6 qw-final-all_32k_dec6 --vs qw-final-def_32k_dec6 2>/dev/null | tail -3
done
echo "[final-e2e] done $(date -u +%T)"
