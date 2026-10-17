#!/usr/bin/env bash
# Integration candidate cand-1017 = main 9da2d3b + the B2 attention levers (9c74f7c TQ_T4_ATTN_F16, d5191d6 TQ_T4_VSTAGE;
# patches/b2/0001-0002) + 0f74130 (TQ_T4_ATTN_F16 default 0 on main until the owner's decision (b); =1 is the B2
# recommendation). Candidate library qwentin-cand/build-qwen-cand/libforward_qwen.so. GPU 0 under the flock, port 8120.
#  Gate 1 (flag off = what the served line runs, bitwise): paged_logit_ab on the TRELLIS pool N=3/8, candidate vs the served
#          library (build-qwen/, main 9da2d3b) -- the new code must be a no-op when off.
#  Gate 2 (flag on, spec exactness, verify == decode, --no-pin = parking inside): spec_paged_check 4K + 32K N=1,3,8 under TQ_KV_T4=1.
#  Gate 3 (flag on, the served line): serve_batched defaults (--kv-t4 --spec, 12 slots, 4100 blocks) + TQ_T4_ATTN_F16=1:
#          needle 32K x 3 depths (3 decoys), spec_speed @32K N=1/8 (the served line today: 163.4 / 72.1 tok/s per stream).
#  On ALL PASS: build-qwen/libforward_qwen.so := the gated candidate (old kept as .9da2d3b), main := cand-1017 (ff),
#          patches/0045-0047, then a reproducibility rebuild of build-qwen/ in place (md5 compared, the gated binary kept).
# usage: nohup logs/cand4-all.sh > logs/cand4-all.log 2>&1 &
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8120
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand4
fail() { echo "[cand4-all] FAIL: $*   $(date -u +%T)"; echo "[cand4-all] nothing swapped; build-qwen/ and main stay at 9da2d3b"; exit 1; }
cmp -s $Q/qwentin-cand/tools/serve_batched.py $Q/qwentin/tools/serve_batched.py || fail "serve_batched.py differs between cand and main (unexpected)"
exec 9>/tmp/exl3diff-gpu0.lock; flock 9
export CUDA_VISIBLE_DEVICES=0 TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_T4_ATTN_F16 TQ_T4_VSTAGE TQ_T4_TABREP
echo "[cand4-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand log --oneline -1 | cut -c1-60); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-50)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep "^0,"

echo "== gate 1: trellis pool, TQ_T4_ATTN_F16 unset (= the served line's path): candidate bitwise == served library   $(date -u +%T)"
for N in 3 8; do
  [ -f runs/cand4/plab-t4-served-n$N.pt ] || TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand4/plab-t4-served-n$N.pt --n $N --tqf $T575 > logs/cand4-plab-served-n$N.log 2>&1 || fail "paged_logit_ab served N=$N (logs/cand4-plab-served-n$N.log)"
  TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1 timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand4/plab-t4-cand-n$N.pt --n $N --tqf $T575 > logs/cand4-plab-cand-n$N.log 2>&1 || fail "paged_logit_ab cand N=$N (logs/cand4-plab-cand-n$N.log)"
done
$PY - <<'PY' || fail "gate 1 (flag-off bitwise)"
import torch, sys
Q = "/mnt/nvme/qwen256k"; ok = True
for N in (3, 8):
    a = torch.load(f"{Q}/runs/cand4/plab-t4-served-n{N}.pt"); b = torch.load(f"{Q}/runs/cand4/plab-t4-cand-n{N}.pt")
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
    ok &= same
    print(f"[plab] N={N} trellis pool, flag off: {len(a)} steps, candidate bitwise identical to the served library: {same} (max |d| {md:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: TQ_T4_ATTN_F16=1 -- spec exactness (verify == decode, --no-pin, parking inside), trellis pool, 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  TQ_KV_T4=1 TQ_T4_ATTN_F16=1 timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand4-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand4-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand4-spec-$tag.log || fail "spec_paged_check $tag (logs/cand4-spec-$tag.log)"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: the served line (defaults --kv-t4 --spec, 12 slots, 4100 blocks) + TQ_T4_ATTN_F16=1   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
TQ_T4_ATTN_F16=1 nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4100 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB > logs/cand4-server.log 2>&1 &
SPID=""
for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand4-server.log; fail "server did not come up (logs/cand4-server.log)"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -E "KV format|kv format|\[paged\] init|spec" logs/cand4-server.log | head -3
timeout 2400 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand4/needle.json > logs/cand4-needle.log 2>&1; nrc=$?
grep -E "PASS|FAIL|^wall" logs/cand4-needle.log | tail -4 | cut -c1-160
NP=$(grep -c "\] PASS" logs/cand4-needle.log); NF=$(grep -c "\] FAIL" logs/cand4-needle.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand4/spec-speed-32k.json > logs/cand4-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand4-spec-speed-32k.log | cut -c1-150
echo "   (the served line today, cand3: N=1 163.4, N=8 72.1 tok/s per stream)"
kill $SPID; for i in $(seq 1 20); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand4-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; spec_speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 3 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, speed rc=$src, server error lines=$ERR)"
echo "[gate3] PASS"
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT

echo "== ALL PASS -> landing cand-1017 on main   $(date -u +%T)"
[ -f $OLD.9da2d3b ] || cp -p $OLD $OLD.9da2d3b
cp -p $LIB $OLD
git -C qwentin merge -q --ff-only cand-1017 || fail "ff-merge (build-qwen/ already holds the gated library; old at .9da2d3b)"
(cd qwentin && git format-patch -q --start-number 45 9da2d3b..HEAD -o ../patches/) && ls patches/004[5-9]*.patch
echo "-- reproducibility rebuild of build-qwen/ from the merged main (the gated binary is kept whatever the md5 says)"
make -C qwentin/build-qwen -j8 > logs/cand4-rebuild.log 2>&1 && RB=$(md5sum $OLD | cut -c1-12) || RB="(build failed, see logs/cand4-rebuild.log)"
cp -p $LIB $OLD
echo "   rebuild md5 $RB vs gated $(md5sum $LIB | cut -c1-12)"
echo "[cand4-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .9da2d3b); main = $(git -C qwentin log --oneline -1 | cut -c1-60)"
echo "[cand4-all] rollback: cp $OLD.9da2d3b $OLD && git -C $Q/qwentin reset --hard 9da2d3b"
