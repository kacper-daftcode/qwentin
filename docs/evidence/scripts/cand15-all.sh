#!/usr/bin/env bash
# cand-1206 = main 0d647f9 + 0db2bd3 (docs: TQ_WIDE_PREFILL, debug switches, a stale comment) + dd6d21e (ten experiment
# switches of the served path removed, defaults hard-coded: TQ_PATTN_S, TQ_WIDE_ATTN_SPLIT, TQ_EXL3_WIDE_NG, TQ_PAGED_SPLIT,
# TQ_BFRAG_BDIM, TQ_VF_NG8, TQ_VF_SPLIT_BY_SLOTS, TQ_VF_COMMIT_FUSED, TQ_VF_SEG_ATTN, TQ_FUSE_NORM). Host code only ->
# SASS identity + bitwise. Candidate = the snapshot qwentin-hyg5/build-qwen-hyg5/libforward_qwen.so.cand1206. GPU 0, port 8145.
#  Gate 0  CPU: ruff + pytest (and with the candidate library).
#  Gate 1  per-kernel SASS identical to the served library; plab N=3/8 (prefill waves + decode) and the 32K variant bitwise.
#  Gate 2  spec exactness 4K + 32K (the verify / commit code lost its switches).
#  Gate 3  THE SERVED LINE, pure defaults: needle 32K + 128K 6/6, spec A/B 10/10, speed N=1/8.
#  NOT landed by this script: the owner approves the removal batch first; then build-qwen/ := candidate (old kept as
#  .0d647f9), main := cand-1206 (ff), patches 0107-.
# usage: nohup logs/cand15-all.sh > logs/cand15-all.log 2>&1 &        (~15 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-hyg5
LIB=$W/build-qwen-hyg5/libforward_qwen.so.cand1206
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
PORT=8145; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand15
fail() { echo "[cand15-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA TQ_GEMM_FORK_MAXN TQ_GEMM_FORK TQ_GRAPH TQ_VERIFY_TQF
echo "[cand15-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C $W log --oneline -1 cand-1206 | cut -c1-70); served $(md5sum $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest   $(date -u +%T)"
git -C $W diff --quiet cand-1206 || fail "the worktree differs from cand-1206"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand15-pytest-lib.log
grep -q "43 passed" logs/cand15-pytest-lib.log && ! grep -q "failed\|error" logs/cand15-pytest-lib.log || fail "gate 0 (with the library)"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise vs the served library   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand15-sass.log; sassrc=$?; head -n 3 logs/cand15-sass.log; [ $sassrc -eq 0 ] || fail "SASS differs"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; n=$3
  env $B flock $LOCK timeout 900 $PY -u $W/tools/paged_logit_ab.py $lib runs/cand15/$out.pt --n $n --tqf $T575 > logs/cand15-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $N; P $LIB plab-new-n$N $N; done
C() { lib=$1; out=$2
  env $B flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib runs/cand15/$out.pt --n 2 --ctx 32768 --steps 16 --tqf $T575 > logs/cand15-$out.log 2>&1 || fail "ctx $out"; }
C $OLD ctx32k-old; C $LIB ctx32k-new
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand15"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1206 vs served")
cmp("ctx32k-new", "ctx32k-old", "32K x 2 slots + 16 steps, cand-1206 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  env TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand15-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand15-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand15-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE, pure defaults   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir $MD --lib $LIB > logs/cand15-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand15-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -oE "[0-9]+ pooled tokens" logs/cand15-server.log | sed 's/^/   /'
grep -q "576000 pooled tokens" logs/cand15-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand15/needle.json > logs/cand15-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand15-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand15-needle.log); NF=$(grep -c "\] FAIL" logs/cand15-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand15/spec-ab.json > logs/cand15-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs" logs/cand15-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand15-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand15/spec-speed-32k.json > logs/cand15-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand15-spec-speed-32k.log | cut -c1-110
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand15-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing waits for the owner's approval of the removal batch)"
