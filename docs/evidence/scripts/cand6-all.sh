#!/usr/bin/env bash
# Integration candidate cand-1019 = cand-1018 + the DeltaNet-state thread (3 commits: TQ_DSTATE_BF16 0..4, default 0 = fp32 =
# bitwise today; serve_batched --dstate {fp32,bf16,fp16,bf16+8,fp16sr}). Worktree qwentin-cand2, library build-qwen-cand2/.
# GPU 1 (PCI c1:00) under /tmp/exl3diff-gpu1.lock, port 8121 (moved from GPU 2 at 01:00 UTC: the owner's GLM chain took GPU 2). Lands only AFTER cand-1018 has landed (polls logs/cand5-all.log).
#  Gate 1  defaults (dstate off), trellis pool, paged_logit_ab N=3/8: candidate bitwise == the cand-1018 binary (what lands first).
#  Gate 2  TQ_DSTATE_BF16=4 (fp16 + SR) under TQ_KV_T4=1: spec exactness (verify == decode incl. the SR seeds), 4K + 32K.
#  Gate 3  the served line + --dstate fp16sr --num-blocks 4500 (the +51K-token configuration): needle 32K + 128K, spec A/B 10, speed N=1/8.
#  Gate 3b the served line with pure defaults on this binary + tools (dstate off): needle 32K, speed N=1 -- the tools change is inert.
#  On ALL PASS and cand-1018 LANDED: build-qwen/ := this binary (old kept as .<cand-1018 sha>), main := cand-1019 (ff), patches/.
# usage: nohup logs/cand6-all.sh > logs/cand6-all.log 2>&1 &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand2/build-qwen-cand2/libforward_qwen.so
BASE=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so     # cand-1018's gated binary
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand2/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8121; GPU=1; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand6
fail() { echo "[cand6-all] FAIL: $*   $(date -u +%T)"; echo "[cand6-all] nothing swapped"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK TQ_DSTATE_BF16
echo "[cand6-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand2 log --oneline -1 | cut -c1-60); base (cand-1018) $(md5sum $BASE | cut -c1-12)"

echo "== gate 1: trellis pool, dstate off, paged_logit_ab N=3/8 -- candidate bitwise == cand-1018   $(date -u +%T)"
for N in 3 8; do
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $BASE runs/cand6/plab-base-n$N.pt --n $N --tqf $T575 > logs/cand6-plab-base-n$N.log 2>&1 || fail "plab base N=$N"
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB  runs/cand6/plab-cand-n$N.pt --n $N --tqf $T575 > logs/cand6-plab-cand-n$N.log 2>&1 || fail "plab cand N=$N"
done
$PY - <<'PY' || fail "gate 1 (bitwise)"
import torch, sys
Q = "/mnt/nvme/qwen256k"; ok = True
for N in (3, 8):
    a = torch.load(f"{Q}/runs/cand6/plab-base-n{N}.pt"); b = torch.load(f"{Q}/runs/cand6/plab-cand-n{N}.pt")
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
    ok &= same
    print(f"[plab] N={N}: {len(a)} steps, candidate (dstate off) bitwise identical to cand-1018: {same} (max |d| {md:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: TQ_DSTATE_BF16=4 (fp16 + SR), TQ_KV_T4=1 -- spec exactness (verify == decode, --no-pin), 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  TQ_KV_T4=1 TQ_DSTATE_BF16=4 flock $LOCK timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand6-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand6-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand6-spec-$tag.log || fail "spec_paged_check $tag (logs/cand6-spec-$tag.log)"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
SPID=""
start_server() {   # $1 = log, then serve args
  local log=$1; shift
  nohup $PY -u $SERVE --port $PORT --max-slots 12 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  SPID=""
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; return 1; }
  echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
  grep -E "^batched server" $log | head -1 | cut -c1-220
}
stop_server() { kill $SPID 2>/dev/null; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; SPID=""; }

exec 9>$LOCK; flock 9
echo "== gate 3: the served line + --dstate fp16sr --num-blocks 4500: needle 32K + 128K, spec A/B 10, speed N=1/8   $(date -u +%T)"
start_server logs/cand6-server-S.log --num-blocks 4500 --dstate fp16sr || fail "server S did not come up (logs/cand6-server-S.log)"
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand6/needle-S.json > logs/cand6-needle-S.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand6-needle-S.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand6-needle-S.log); NF=$(grep -c "\] FAIL" logs/cand6-needle-S.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand6/spec-ab-S.json > logs/cand6-spec-ab-S.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand6-spec-ab-S.log | cut -c1-150
AB=$(grep -c "identical outputs: 10/10" logs/cand6-spec-ab-S.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand6/spec-speed-S.json > logs/cand6-spec-speed-S.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand6-spec-speed-S.log | cut -c1-150
echo "   (served line: 163.4 / 72.1 tok/s per stream at N=1/8; the dstate thread measured -0.38 ms per step at N=8 with fp16sr)"
stop_server
ERR=$(grep -ciE "xid|error|traceback" logs/cand6-server-S.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, server error lines=$ERR)"
echo "[gate3] PASS"

echo "== gate 3b: the served line, pure defaults on this binary + tools (dstate off): needle 32K, speed N=1   $(date -u +%T)"
start_server logs/cand6-server-D.log --num-blocks 4100 || fail "server D did not come up (logs/cand6-server-D.log)"
timeout 2400 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand6/needle-D.json > logs/cand6-needle-D.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand6-needle-D.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand6-needle-D.log); NF=$(grep -c "\] FAIL" logs/cand6-needle-D.log)
timeout 1800 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on --max-tokens 300 --out runs/cand6/spec-speed-D.json > logs/cand6-spec-speed-D.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand6-spec-speed-D.log | cut -c1-150
stop_server
ERR=$(grep -ciE "xid|error|traceback" logs/cand6-server-D.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3b (needle rc=$nrc PASS=$NP FAIL=$NF, speed rc=$src, server error lines=$ERR)"
echo "[gate3b] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT

echo "== ALL PASS; waiting for cand-1018 to land (logs/cand5-all.log)   $(date -u +%T)"
while ! grep -qE "^\[cand5-all\] (LANDED|FAIL)" logs/cand5-all.log 2>/dev/null; do sleep 60; done
grep -q "^\[cand5-all\] LANDED" logs/cand5-all.log || fail "cand-1018 did not land -- cand-1019 is gated but stays unmerged (rebase it onto whatever main becomes)"
TIP=$(git -C qwentin-cand log -1 --format=%h cand-1018); MAIN=$(git -C qwentin log -1 --format=%h)
[ "$TIP" = "$MAIN" ] || fail "main ($MAIN) != cand-1018 tip ($TIP) -- not landing on an unexpected main"
echo "== landing cand-1019 on main $MAIN   $(date -u +%T)"
[ -f $OLD.$MAIN ] || cp -p $OLD $OLD.$MAIN
cp -p $LIB $OLD && touch $OLD
git -C qwentin merge -q --ff-only cand-1019 || fail "ff-merge (build-qwen/ already holds the cand-1019 library; old at .$MAIN)"
START=$(( $(ls patches/0*.patch | wc -l) + 1 ))
(cd qwentin && git format-patch -q --start-number $START $MAIN..HEAD -o ../patches/) && ls patches/0*.patch | tail -3
echo "[cand6-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .$MAIN); main = $(git -C qwentin log --oneline -1 | cut -c1-60); --dstate default fp32 (decision (c): fp16sr)"
echo "[cand6-all] rollback: cp $OLD.$MAIN $OLD && git -C $Q/qwentin reset --hard $MAIN"
