#!/usr/bin/env bash
# Integration candidate cand-1018 = main 0f74130 + the CUDA-graph / GEMM-fork thread (5 commits, TQ_GRAPH / TQ_GEMM_FORK default
# off) + the prefill-GEMM thread (5 commits: 128-token EXL3 units TQ_EXL3_WIDE_NG16 default 1, paired codeword decode, bf16
# in_proj_b/a r4 default 1, TQ_WAVE_MAX default 128 = unchanged). Candidate library qwentin-cand/build-qwen-cand/libforward_qwen.so.
# GPU 1 (PCI c1:00) under /tmp/exl3diff-gpu1.lock, port 8121. The owner's GLM chain holds GPU 0 / 2 and may also take the gpu1
# lock between our jobs; every GPU job here runs under the lock and tolerates that.
#  Gate 1  defaults (prefill levers on), trellis pool, paged_logit_ab N=3/8: candidate bitwise == the served library (0f74130).
#  Gate 1b the same with TQ_GRAPH=1 TQ_GEMM_FORK=1 (the graph thread's pair) -- bitwise too.
#  Gate 2  TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1: spec exactness (verify == decode, --no-pin, parking inside), 4K + 32K.
#  Gate 3  THE SERVED LINE with pure defaults (= what lands): needle 32K + 128K (TTFT = the prefill levers), spec_speed @32K N=1/8.
#  Gate 4  the served line + the recommended flags (TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1): needle 32K, spec A/B 10, speed N=1/8.
#  On ALL PASS: build-qwen/libforward_qwen.so := the gated candidate (old kept as .0f74130), main := cand-1018 (ff), patches/0048-0057.
# usage: nohup logs/cand5-all.sh > logs/cand5-all.log 2>&1 &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8121; GPU=1; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
REC="TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1"
cd $Q; mkdir -p runs/cand5
fail() { echo "[cand5-all] FAIL: $*   $(date -u +%T)"; echo "[cand5-all] nothing swapped; build-qwen/ and main stay at 0f74130"; exit 1; }
cmp -s $Q/qwentin-cand/tools/serve_batched.py $Q/qwentin/tools/serve_batched.py || fail "serve_batched.py differs between cand and main (unexpected)"
export CUDA_VISIBLE_DEVICES=$GPU TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_T4_ATTN_F16 TQ_T4_VSTAGE TQ_GRAPH TQ_GEMM_FORK TQ_EXL3_WIDE_NG16 TQ_EXL3_BF16_R4 TQ_WAVE_MAX
echo "[cand5-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand log --oneline -1 | cut -c1-60); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-50)"

echo "== gate 1 / 1b: trellis pool, paged_logit_ab N=3/8 -- candidate (defaults) and candidate (graph+fork) bitwise == served   $(date -u +%T)"
for N in 3 8; do
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand5/plab-served-n$N.pt --n $N --tqf $T575 > logs/cand5-plab-served-n$N.log 2>&1 || fail "plab served N=$N"
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand5/plab-cand-n$N.pt --n $N --tqf $T575 > logs/cand5-plab-cand-n$N.log 2>&1 || fail "plab cand N=$N"
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand5/plab-candgf-n$N.pt --n $N --tqf $T575 > logs/cand5-plab-candgf-n$N.log 2>&1 || fail "plab cand graph+fork N=$N"
done
$PY - <<'PY' || fail "gate 1 / 1b (bitwise)"
import torch, sys
Q = "/mnt/nvme/qwen256k"; ok = True
for N in (3, 8):
    a = torch.load(f"{Q}/runs/cand5/plab-served-n{N}.pt")
    for tag in ("cand", "candgf"):
        b = torch.load(f"{Q}/runs/cand5/plab-{tag}-n{N}.pt")
        same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
        ok &= same
        print(f"[plab] N={N} {tag:6s}: {len(a)} steps, bitwise identical to the served library: {same} (max |d| {md:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: $REC -- spec exactness (verify == decode, --no-pin), trellis pool, 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  env TQ_KV_T4=1 $REC flock $LOCK timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand5-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand5-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand5-spec-$tag.log || fail "spec_paged_check $tag (logs/cand5-spec-$tag.log)"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
SPID=""
start_server() {   # $1 = log, $2 = extra env string, then serve args
  local log=$1 envs=$2; shift 2
  env $envs nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4100 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  SPID=""
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; return 1; }
  echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
  grep -E "^batched server" $log | head -1 | cut -c1-200
}
stop_server() { kill $SPID 2>/dev/null; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; SPID=""; }

exec 9>$LOCK; flock 9
echo "== gate 3: THE SERVED LINE, pure defaults (what lands): needle 32K + 128K, speed N=1/8   $(date -u +%T)"
start_server logs/cand5-server-D.log "" || fail "server D did not come up (logs/cand5-server-D.log)"
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand5/needle-D.json > logs/cand5-needle-D.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand5-needle-D.log | cut -c1-110
echo "   (cand3 10-16: 32K TTFT 18.1 s, 128K 97.6-98.2 s; cand4: 32K 17.1 s -- the prefill levers should show here)"
NP=$(grep -c "\] PASS" logs/cand5-needle-D.log); NF=$(grep -c "\] FAIL" logs/cand5-needle-D.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand5/spec-speed-D.json > logs/cand5-spec-speed-D.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand5-spec-speed-D.log | cut -c1-150
echo "   (served line 10-16: 163.4 / 72.1 tok/s per stream at N=1/8)"
stop_server
ERR=$(grep -ciE "xid|error|traceback" logs/cand5-server-D.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, speed rc=$src, server error lines=$ERR)"
echo "[gate3] PASS"

echo "== gate 4: the served line + $REC: needle 32K, spec A/B 10, speed N=1/8   $(date -u +%T)"
start_server logs/cand5-server-R.log "$REC" || fail "server R did not come up (logs/cand5-server-R.log)"
timeout 2400 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand5/needle-R.json > logs/cand5-needle-R.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand5-needle-R.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand5-needle-R.log); NF=$(grep -c "\] FAIL" logs/cand5-needle-R.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand5/spec-ab-R.json > logs/cand5-spec-ab-R.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand5-spec-ab-R.log | cut -c1-150
AB=$(grep -c "identical outputs: 10/10" logs/cand5-spec-ab-R.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand5/spec-speed-R.json > logs/cand5-spec-speed-R.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand5-spec-speed-R.log | cut -c1-150
echo "   (cand4 with TQ_T4_ATTN_F16=1 alone: 165.5 / 77.8; graph thread's pair alone @32K N=1: 169.5)"
stop_server
ERR=$(grep -ciE "xid|error|traceback" logs/cand5-server-R.log); GC=$(grep -c "graph" logs/cand5-server-R.log)
echo "   server log error lines: $ERR (graph lines $GC); needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 4 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, server error lines=$ERR)"
echo "[gate4] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT

echo "== ALL PASS -> landing cand-1018 on main   $(date -u +%T)"
[ -f $OLD.0f74130 ] || cp -p $OLD $OLD.0f74130
cp -p $LIB $OLD && touch $OLD
git -C qwentin merge -q --ff-only cand-1018 || fail "ff-merge (build-qwen/ already holds the gated library; old at .0f74130)"
(cd qwentin && git format-patch -q --start-number 48 0f74130..HEAD -o ../patches/) && ls patches/00[45][0-9]*.patch | tail -10
echo "[cand5-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .0f74130); main = $(git -C qwentin log --oneline -1 | cut -c1-60)"
echo "[cand5-all] rollback: cp $OLD.0f74130 $OLD && git -C $Q/qwentin reset --hard 0f74130"
