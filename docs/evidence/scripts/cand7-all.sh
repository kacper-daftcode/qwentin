#!/usr/bin/env bash
# cand-1020 = main 11f5879 + 2cf671c: the owner's decisions (a)(b)(c)(g) as defaults (TQ_T4_ATTN_F16=1, TQ_GRAPH=1, TQ_GEMM_FORK=1,
# TQ_DSTATE_BF16=4, serve_batched --decode-steps-per-wave 1 / --dstate fp16sr). Candidate qwentin-cand/build-qwen-cand/, GPU 0, port 8123.
#  Gate 1  the flip is only a flip: (i) candidate BARE == served binary (c9d5269c) WITH the four flags; (ii) candidate with the four
#          off-switches == served binary BARE -- paged_logit_ab on the trellis pool, N=3/8, all bitwise.
#  Gate 2  spec exactness on the candidate's bare defaults (verify == decode, --no-pin), 4K + 32K.
#  Gate 3  THE NEW SERVED LINE, pure defaults: `--max-slots 12 --num-blocks 4500` -- startup line must say fp16sr + 576000 tokens +
#          decode steps per mixed wave 1; needle 32K + 128K, spec A/B 10, spec_speed @32K N=1/8 + @128K N=1.
#  On ALL PASS: build-qwen/ := the candidate (old kept as .11f5879), main := cand-1020 (ff), patches/0061.
# usage: nohup logs/cand7-all.sh > logs/cand7-all.log 2>&1 &        (~30 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8123; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
ON="TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 TQ_DSTATE_BF16=4"
OFF="TQ_T4_ATTN_F16=0 TQ_GRAPH=0 TQ_GEMM_FORK=0 TQ_DSTATE_BF16=0"
cd $Q; mkdir -p runs/cand7
fail() { echo "[cand7-all] FAIL: $*   $(date -u +%T)"; echo "[cand7-all] nothing swapped; build-qwen/ and main stay at 11f5879"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK TQ_DSTATE_BF16
echo "[cand7-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand log --oneline -1 | cut -c1-50); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-40)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
echo "== gate 1: the flip is only a flip (plab trellis pool N=3/8)   $(date -u +%T)"
for N in 3 8; do
  env $B $ON  flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand7/plab-old-on-n$N.pt   --n $N --tqf $T575 > logs/cand7-plab-old-on-n$N.log 2>&1   || fail "plab old+on N=$N"
  env $B      flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand7/plab-new-bare-n$N.pt --n $N --tqf $T575 > logs/cand7-plab-new-bare-n$N.log 2>&1 || fail "plab new bare N=$N"
  env $B      flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand7/plab-old-bare-n$N.pt --n $N --tqf $T575 > logs/cand7-plab-old-bare-n$N.log 2>&1 || fail "plab old bare N=$N"
  env $B $OFF flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand7/plab-new-off-n$N.pt  --n $N --tqf $T575 > logs/cand7-plab-new-off-n$N.log 2>&1  || fail "plab new+off N=$N"
done
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand7"; ok = True
def same(a, b):
    a = torch.load(f"{Q}/{a}.pt"); b = torch.load(f"{Q}/{b}.pt")
    s = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
    return s, md, len(a)
for N in (3, 8):
    s1, m1, n1 = same(f"plab-new-bare-n{N}", f"plab-old-on-n{N}")
    s2, m2, n2 = same(f"plab-new-off-n{N}", f"plab-old-bare-n{N}")
    s3, m3, _  = same(f"plab-new-bare-n{N}", f"plab-old-bare-n{N}")
    ok &= s1 and s2
    print(f"[plab] N={N}: new defaults == old+flags: {s1} (max |d| {m1:.3g}, {n1} steps); new+off-switches == old defaults: {s2} (max |d| {m2:.3g}); (new defaults vs old defaults differ as expected: {not s3}, max |d| {m3:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness on the bare new defaults (verify == decode, --no-pin), 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand7-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand7-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand7-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE NEW SERVED LINE, pure defaults: --max-slots 12 --num-blocks 4500   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB > logs/cand7-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand7-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
SL=$(grep -E "^batched server" logs/cand7-server.log | head -1); echo "   $SL" | cut -c1-300
echo "$SL" | grep -q "576000 pooled tokens" || fail "startup line: not 576000 pooled tokens"
echo "$SL" | grep -qi "fp16sr" || fail "startup line: state format is not fp16sr"
echo "$SL" | grep -q "decode steps per mixed wave 1" || fail "startup line: decode steps per mixed wave != 1"
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand7/needle.json > logs/cand7-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand7-needle.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand7-needle.log); NF=$(grep -c "\] FAIL" logs/cand7-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand7/spec-ab.json > logs/cand7-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand7-spec-ab.log | cut -c1-150
AB=$(grep -c "identical outputs: 10/10" logs/cand7-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand7/spec-speed-32k.json > logs/cand7-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand7-spec-speed-32k.log | cut -c1-130
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand7/spec-speed-128k.json > logs/cand7-spec-speed-128k.log 2>&1; src2=$?
grep -E "\[speed\] ctx" logs/cand7-spec-speed-128k.log | cut -c1-130
echo "   (final-speed on the same binary with the flags: 165.3 / 78.8 @32K N=1/8, 131.7 @128K N=1)"
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand7-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src / $src2"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$src2" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src/$src2, server error lines=$ERR)"
echo "[gate3] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT

echo "== ALL PASS -> landing cand-1020 on main   $(date -u +%T)"
[ -f $OLD.11f5879 ] || cp -p $OLD $OLD.11f5879
cp -p $LIB $OLD && touch $OLD
git -C qwentin merge -q --ff-only cand-1020 || fail "ff-merge (build-qwen/ already holds the gated library; old at .11f5879)"
(cd qwentin && git format-patch -q --start-number 61 11f5879..HEAD -o ../patches/) && ls patches/0061*.patch
echo "[cand7-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .11f5879); main = $(git -C qwentin log --oneline -1 | cut -c1-60)"
echo "[cand7-all] rollback: cp $OLD.11f5879 $OLD && git -C $Q/qwentin reset --hard 11f5879"
