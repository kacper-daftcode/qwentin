#!/usr/bin/env bash
# cand-1204 = cand-1203 + 6283654 (src/qwentin.h generated from the engine's extern "C" definitions and included first by
# the TU; tools/qwentin_abi.py; qwn_free.restype = None in three tools). Declarations only -> SASS identity + bitwise.
# Candidate = qwentin-hyg3/build-qwen-hyg3/libforward_qwen.so.cand1204. GPU 0 under its lock, port 8143. Runs after cand12.
#  Gate 0  CPU: ruff + pytest (incl. tests/test_qwentin_abi.py).
#  Gate 1  per-kernel SASS identical to the cand-1203 snapshot; plab N=3 bitwise vs the served binary.
#  Gate 3  the served line: startup, needle 32K x 3, spec A/B 10/10, clean shutdown (qwn_free).
#  Landing by hand after ALL PASS (with cand12): build-qwen/ := this candidate (old kept as .364aa9c), main := cand-1204.
# usage: nohup logs/cand13-all.sh > logs/cand13-all.log 2>&1 &        (~6 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
W=$Q/qwentin-hyg3
LIB=$W/build-qwen-hyg3/libforward_qwen.so.cand1204
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
TFP6=/mnt/nvme/models/qwen3_8-27b-e2m3-mtp-bf16src.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
PORT=8143; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand13
fail() { echo "[cand13-all] FAIL: $*   $(date -u +%T)"; echo "[cand13-all] nothing swapped; build-qwen/ and main stay at 364aa9c"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA TQ_GEMM_FORK_MAXN TQ_GEMM_FORK TQ_GRAPH
echo "[cand13-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C $W log --oneline -1 cand-1204 | cut -c1-70); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-40)"

echo "== gate 0: ruff + pytest   $(date -u +%T)"
git -C $W diff --quiet cand-1204 || fail "the worktree differs from cand-1204"
(cd $W && /tmp/ci-venv/bin/ruff check tools/ tests/ && /tmp/ci-venv/bin/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise vs the served binary   $(date -u +%T)"
$PY analysis/sass_diff.py $Q/qwentin-hyg2/build-qwen-hyg2/libforward_qwen.so.cand1203 $LIB > logs/cand13-sass.log; sassrc=$?; head -n 3 logs/cand13-sass.log; [ $sassrc -eq 0 ] || fail "SASS differs"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; n=$3; shift 3
  env $B "$@" flock $LOCK timeout 900 $PY -u $W/tools/paged_logit_ab.py $lib runs/cand13/$out.pt --n $n --tqf $T575 > logs/cand13-$out.log 2>&1 || fail "plab $out"; }
P $OLD plab-old-n3 3; P $LIB plab-new-n3 3
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand13"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    md = max((A[k] - B[k]).abs().max().item() for k in A) if A.keys() == B.keys() else float("nan")
    ok &= s; print(f"[plab] {label}: bitwise {s} (max |d| {md:.3g}, {len(A)} tensors)")
cmp("plab-new-n3", "plab-old-n3", "N=3 cand-1204 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 3: THE SERVED LINE, pure defaults   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir $MD --lib $LIB > logs/cand13-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand13-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -oE "codebook [^)]*|[0-9]+ pooled tokens" logs/cand13-server.log | sed 's/^/   /'
grep -q "576000 pooled tokens" logs/cand13-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand13/needle.json > logs/cand13-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand13-needle.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand13-needle.log); NF=$(grep -c "\] FAIL" logs/cand13-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand13/spec-ab.json > logs/cand13-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand13-spec-ab.log | cut -c1-150; AB=$(grep -c "identical outputs: 10/10" logs/cand13-spec-ab.log)
src=0
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand13-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT

echo "== ALL PASS   $(date -u +%T)"
echo "[cand13-all] land by hand: [ -f $OLD.364aa9c ] || cp -p $OLD $OLD.364aa9c; cp -p $LIB $OLD; git -C $Q/qwentin merge --ff-only cand-1204;"
echo "[cand13-all]   touch $OLD (after the merge); (cd $Q/qwentin && git format-patch -q --start-number 102 364aa9c..HEAD -o ../patches/)"
