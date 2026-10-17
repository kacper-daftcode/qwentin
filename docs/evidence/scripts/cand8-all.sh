#!/usr/bin/env bash
# cand-1101 = main 92ff977 + the session-11 DeltaNet-prefill thread (4 commits, gdn-prefill) + b2e7a3f (TQ_GDN_PREFILL default 1,
# --gdn-prefill default on). Candidate qwentin-cand/build-qwen-cand/libforward_qwen.so. GPU 1 under its lock, port 8128.
#  Gate 1  the flip is only a flip + the lever is bitwise: candidate BARE == served binary (e454a29c) on the trellis pool (plab N=3/8)
#          -- the prefill of the plab rows runs through the new kernels, decode through the untouched ones; AND candidate with
#          TQ_GDN_PREFILL=0 == served binary (off-switch).
#  Gate 2  spec exactness on the bare new defaults (--no-pin, 4K + 32K; the prefill writes the state the verify reads).
#  Gate 3  THE SERVED LINE, pure defaults (--max-slots 12 --num-blocks 4500): needle 32K + 128K (TTFT = the headline: expect
#          ~13.2 s / ~76.7 s vs 15.9 / 86.6), spec A/B 10, spec_speed @32K N=1/8 (unchanged: 165-169 / 78-81).
#  On ALL PASS: build-qwen/ := candidate (old kept as .2cf671c-engine ... named .92ff977), main := cand-1101 (ff), patches/0079-0083.
# usage: nohup logs/cand8-all.sh > logs/cand8-all.log 2>&1 &        (~30 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8128; GPU=1; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand8
fail() { echo "[cand8-all] FAIL: $*   $(date -u +%T)"; echo "[cand8-all] nothing swapped; build-qwen/ and main stay at 92ff977"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_GDN_PREFILL
echo "[cand8-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand log --oneline -1 | cut -c1-50); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-40)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
echo "== gate 1: plab trellis pool N=3/8: new bare == served; new+TQ_GDN_PREFILL=0 == served   $(date -u +%T)"
for N in 3 8; do
  env $B flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand8/plab-old-n$N.pt --n $N --tqf $T575 > logs/cand8-plab-old-n$N.log 2>&1 || fail "plab old N=$N"
  env $B flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand8/plab-new-n$N.pt --n $N --tqf $T575 > logs/cand8-plab-new-n$N.log 2>&1 || fail "plab new N=$N"
  env $B TQ_GDN_PREFILL=0 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand8/plab-newoff-n$N.pt --n $N --tqf $T575 > logs/cand8-plab-newoff-n$N.log 2>&1 || fail "plab new+off N=$N"
done
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand8"; ok = True
def same(a, b):
    a = torch.load(f"{Q}/{a}.pt"); b = torch.load(f"{Q}/{b}.pt")
    s = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
    return s, md
for N in (3, 8):
    s1, m1 = same(f"plab-new-n{N}", f"plab-old-n{N}"); s2, m2 = same(f"plab-newoff-n{N}", f"plab-old-n{N}")
    ok &= s1 and s2
    print(f"[plab] N={N}: new defaults == served: {s1} (max |d| {m1:.3g}); new + TQ_GDN_PREFILL=0 == served: {s2} (max |d| {m2:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
echo "== gate 2: spec exactness, bare new defaults, 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand8-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand8-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand8-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"
echo "== gate 3: THE SERVED LINE, pure defaults   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB > logs/cand8-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand8-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -E "^batched server" logs/cand8-server.log | head -1 | cut -c1-300
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand8/needle.json > logs/cand8-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand8-needle.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand8-needle.log); NF=$(grep -c "\] FAIL" logs/cand8-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand8/spec-ab.json > logs/cand8-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand8-spec-ab.log | cut -c1-150; AB=$(grep -c "identical outputs: 10/10" logs/cand8-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand8/spec-speed-32k.json > logs/cand8-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand8-spec-speed-32k.log | cut -c1-130
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand8-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT
echo "== ALL PASS -> landing cand-1101 on main   $(date -u +%T)"
[ -f $OLD.92ff977 ] || cp -p $OLD $OLD.92ff977
cp -p $LIB $OLD && touch $OLD
git -C qwentin merge -q --ff-only cand-1101 || fail "ff-merge (build-qwen/ already holds the gated library; old at .92ff977)"
(cd qwentin && git format-patch -q --start-number 79 92ff977..HEAD -o ../patches/) && ls patches/008*.patch patches/0079*.patch 2>/dev/null | tail -5
echo "[cand8-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .92ff977); main = $(git -C qwentin log --oneline -1 | cut -c1-60)"
echo "[cand8-all] rollback: cp $OLD.92ff977 $OLD && git -C $Q/qwentin reset --hard 92ff977"
