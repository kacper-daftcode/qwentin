#!/usr/bin/env bash
# cand-1208, the off-switch matrix: the served library (main 07897ac, getenv at each site) and the candidate (one parse
# in src/engine/config.cuh) run the same harness under the same non-default switch values; their logits must be equal
# bit for bit in every cell. Decode-path switches through tools/paged_logit_ab.py (N = 3 slots, 401 steps; trellis pool,
# FP4 pool, FP6 file), prefill-path switches through analysis/paged_logit_ab_ctx.py (2 slots x 8K + 16 steps). The
# candidate's [config] line must show the cell's switches (or, "!NAME=", leave out one set to its default). GPU 1.
# usage: nohup logs/cand18-matrix.sh > logs/cand18-matrix.log 2>&1 &        (~15 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
W=$Q/qwentin-cfg
LIB=$W/build-qwen-cfg/libforward_qwen.so.cand1208
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
FP6=/mnt/nvme/models/qwen3_8-27b-e2m3-mtp.tqf
GPU=1; LOCK=/tmp/exl3diff-gpu$GPU.lock
R=$Q/runs/cand18/matrix; mkdir -p $R $Q/logs/cand18-matrix
cd $Q
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
export CUDA_VISIBLE_DEVICES=$GPU
echo "[matrix] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12), served $(md5sum < $OLD | cut -c1-12)"
T4="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
FP4="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=0 TQ_NO_MTP=1"
# cell | harness (plab / ctx) | model | expected in the candidate's [config] line | env
CELLS=(
  "t4-attn-f16-0|plab|$T575|TQ_T4_ATTN_F16=0|TQ_T4_ATTN_F16=0 $T4"
  "graph-0|plab|$T575|TQ_GRAPH=0|TQ_GRAPH=0 $T4"
  "gemm-fork-0|plab|$T575|TQ_GEMM_FORK=0|TQ_GEMM_FORK=0 $T4"
  "dstate-0|plab|$T575|TQ_DSTATE_BF16=0|TQ_DSTATE_BF16=0 $T4"
  "pdl-0|plab|$T575|!TQ_PDL=|TQ_PDL=0 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
  "pdl-2|plab|$T575|TQ_PDL=2|TQ_PDL=2 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
  "pins|plab|$T575|TQ_QMMA_SF_K_SPLIT=4 TQ_PAGED_GQA_SPLIT=8|TQ_QMMA_SF_K_SPLIT=4 TQ_PAGED_GQA_SPLIT=8 $T4"
  "embed-fp8-2|plab|$T575|TQ_EMBED_FP8=2|TQ_EMBED_FP8=2 TQ_EMBED_HOST=0 $T4"
  "embed-fp8-1|plab|$T575|TQ_EMBED_FP8=1|TQ_EMBED_FP8=1 TQ_EMBED_HOST=0 $T4"
  "verify-tqf|plab|$T575|TQ_VERIFY_TQF=1|TQ_VERIFY_TQF=1 $T4"
  "codebook-file|plab|$T575|TQ_KV_T4_CODEBOOK=$W/tools/codebooks/lloyd_kv12a.f32|TQ_KV_T4_CODEBOOK=$W/tools/codebooks/lloyd_kv12a.f32 $T4"
  "fp4-st3|plab|$T575|!TQ_KV_T4=|$FP4"
  "fp4-st1|plab|$T575|TQ_PAGED_ATTN_ST=1|TQ_PAGED_ATTN_ST=1 $FP4"
  "fp4-st0|plab|$T575|TQ_PAGED_ATTN_ST=0|TQ_PAGED_ATTN_ST=0 $FP4"
  "fp4-st2|plab|$T575|TQ_PAGED_ATTN_ST=2|TQ_PAGED_ATTN_ST=2 $FP4"
  "fp4-mma-0|plab|$T575|TQ_PAGED_ATTN_MMA=0|TQ_PAGED_ATTN_MMA=0 $FP4"
  "fp4-topk|plab|$T575|TQ_PAGED_TOPK_PAGES=2 TQ_PAGED_TOPK_MIN_CTX=0|TQ_PAGED_TOPK_PAGES=2 TQ_PAGED_TOPK_MIN_CTX=0 $FP4"
  "fp6-mtp-head|plab|$FP6|!TQ_NO_MTP=|TQ_NO_MTP=0 TQ_PDL=1"
  "gdn-prefill-0|ctx|$T575|TQ_GDN_PREFILL=0|TQ_GDN_PREFILL=0 $T4"
  "pattn-gqa-0-qrows-16|ctx|$T575|TQ_PATTN_GQA=0 TQ_WIDE_ATTN_QROWS=16|TQ_PATTN_GQA=0 TQ_WIDE_ATTN_QROWS=16 $T4"
  "t4-async-0|ctx|$T575|TQ_KV_T4_ASYNC=0|TQ_KV_T4_ASYNC=0 $T4"
  "exl3-ng16-0|ctx|$T575|TQ_EXL3_WIDE_NG16=0|TQ_EXL3_WIDE_NG16=0 $T4"
  "exl3-bf16-r4-0|ctx|$T575|TQ_EXL3_BF16_R4=0|TQ_EXL3_BF16_R4=0 $T4"
  "gemm-fork-maxn|ctx|$T575|TQ_GEMM_FORK_MAXN=1000000|TQ_GEMM_FORK_MAXN=1000000 $T4"
  "fp4-ctx-qrows-16|ctx|$T575|TQ_WIDE_ATTN_QROWS=16|TQ_WIDE_ATTN_QROWS=16 $FP4"
)
run() { lib=$1; tools=$2; harness=$3; tqf=$4; envs=$5; out=$6
  if [ "$harness" = plab ]; then
    env $envs flock $LOCK timeout 1200 $PY -u $tools/paged_logit_ab.py $lib $R/$out.pt --n 3 --tqf $tqf > logs/cand18-matrix/$out.log 2>&1
  else
    env $envs QWENTIN_TOOLS=$tools flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib $R/$out.pt --n 2 --ctx 8192 --steps 16 \
      --tqf $tqf > logs/cand18-matrix/$out.log 2>&1
  fi; }
fails=0
for c in "${CELLS[@]}"; do
  IFS='|' read -r name harness tqf expect envs <<< "$c"
  run $OLD $Q/qwentin/tools $harness $tqf "$envs" $name-old || { echo "[matrix] $name: the served library's run failed"; fails=$((fails + 1)); continue; }
  run $LIB $W/tools $harness $tqf "$envs" $name-new || { echo "[matrix] $name: the candidate's run failed"; fails=$((fails + 1)); continue; }
  cfg=$(grep -m1 '^\[config\]' logs/cand18-matrix/$name-new.log)
  verdict=$($PY - "$R/$name-old.pt" "$R/$name-new.pt" <<'PY'
import sys, torch
A = torch.load(sys.argv[1]); B = torch.load(sys.argv[2])
same = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
print(("bitwise" if same else "DIFFERENT") + f" ({len(A)} tensors)")
PY
)
  if [ "${expect:0:1}" = "!" ]; then case "$cfg" in *"${expect:1}"*) cfgok=0 ;; *) cfgok=1 ;; esac
  else case "$cfg" in *" $expect "*) cfgok=1 ;; *) cfgok=0 ;; esac; fi
  [ -n "$cfg" ] || cfgok=0
  echo "[matrix] $name ($harness; $envs): $verdict; [config] line as expected ($expect): $cfgok"
  [ "${verdict%% *}" = bitwise ] && [ $cfgok -eq 1 ] || fails=$((fails + 1))
done
echo "[matrix] example: $(grep -m1 '^\[config\]' logs/cand18-matrix/fp4-topk-new.log)"
echo "[matrix] bf16 MTP head (FP6 file, TQ_NO_MTP=0): served '$(grep -m1 'MTP head section' logs/cand18-matrix/fp6-mtp-head-old.log | sed 's/^ *//')', candidate '$(grep -m1 'MTP head section' logs/cand18-matrix/fp6-mtp-head-new.log | sed 's/^ *//')'"
echo "[matrix] ${#CELLS[@]} cells, $fails failed   $(date -u +%T)"
[ $fails -eq 0 ] && echo "[matrix] PASS" || echo "[matrix] FAIL"
