#!/usr/bin/env bash
# cand-1201 = main 9c0a046 + 81b8453 (the trellis-KV codebook compiled into the engine, src/kv_t4_codebook.inc; TQ_KV_T4_CODEBOOK
# only overrides it; serve_batched.py / paged_logit_ab.py / spec_paged_check.py / reproduce.sh stop passing the file) + docs.
# Same 4096 codebook values -> the gate is bitwise. Candidate qwentin-hyg/build-qwen-hyg/libforward_qwen.so. GPU 0, port 8140.
#  Gate 0  CPU: serve_batched.py --kv-t4-codebook <missing file> is refused before any GPU work.
#  Gate 1  bitwise (plab N=3/8, trellis pool): the served binary (c40173fe) reading the dev-box codebook file == the candidate
#          with TQ_KV_T4_CODEBOOK unset (built-in; its log says "codebook built-in lloyd_kv12a") == the candidate reading the
#          shipped tools/codebooks/lloyd_kv12a.f32; the override is still read: candidate == served with random12.f32 (N=3),
#          and those rows differ from the lloyd_kv12a rows.
#  Gate 2  spec exactness on the candidate with the built-in codebook, 4K + 32K (--no-pin), the candidate's spec_paged_check.py.
#  Gate 3  THE SERVED LINE through the candidate's serve_batched.py, pure defaults (no codebook flag, env unset): the engine
#          reports the built-in codebook, 576000 pooled tokens, needle 32K + 128K 6/6, spec A/B 10/10, speed N=1/8.
#  Landing by hand after ALL PASS (the README / CHANGELOG commits of cand-1201 land with it): build-qwen/ := candidate (old
#  kept as .9c0a046), main := cand-1201 (ff), patches/0095-.
# usage: nohup logs/cand10-all.sh > logs/cand10-all.log 2>&1 &        (~35 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
W=$Q/qwentin-hyg
LIB=$W/build-qwen-hyg/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$W/tools/serve_batched.py
PLAB=$W/tools/paged_logit_ab.py
SPECCHK=$W/tools/spec_paged_check.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
SHIP=$W/tools/codebooks/lloyd_kv12a.f32
RND=$Q/analysis/kv_trellis_cuda/codebooks/random12.f32
PORT=8140; GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand10
fail() { echo "[cand10-all] FAIL: $*   $(date -u +%T)"; echo "[cand10-all] nothing swapped; build-qwen/ and main stay at 9c0a046"; exit 1; }
unset TQ_KV_T4_CODEBOOK TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL \
      TQ_CTX TQ_KV_Q4 TQ_PAGED_ATTN_MMA
echo "[cand10-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-70); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-40)"

echo "== gate 0: --kv-t4-codebook with a missing file is refused   $(date -u +%T)"
out=$(CUDA_VISIBLE_DEVICES= timeout 120 $PY $SERVE --tqf $T575M --kv-t4-codebook /nonexistent/cb.f32 2>&1); rc=$?
echo "   rc=$rc: $(echo "$out" | tail -1 | cut -c1-150)"
{ [ $rc -ne 0 ] && echo "$out" | grep -q "codebook /nonexistent/cb.f32 not found"; } || fail "gate 0"
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: bitwise, built-in vs the dev-box file vs the shipped file (plab N=3/8)   $(date -u +%T)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
P() { lib=$1; out=$2; cb=$3; n=$4
  env $B TQ_KV_T4_CODEBOOK="$cb" flock $LOCK timeout 900 $PY -u $PLAB $lib runs/cand10/$out.pt --n $n --tqf $T575 > logs/cand10-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do
  P $OLD plab-old-n$N "$CB" $N
  P $LIB plab-new-n$N "" $N
  P $LIB plab-newfile-n$N "$SHIP" $N
done
P $OLD plab-oldrnd-n3 "$RND" 3
P $LIB plab-newrnd-n3 "$RND" 3
for f in old-n3:"codebook $CB" new-n3:"codebook built-in lloyd_kv12a" newfile-n3:"codebook $SHIP" newrnd-n3:"codebook $RND"; do
  log=logs/cand10-plab-${f%%:*}.log; pat=${f#*:}
  grep -q "$pat" $log || fail "$log does not report '$pat'"
  echo "   ${f%%:*}: $(grep -o 'codebook [^)]*' $log | head -1)"
done
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand10"; ok = True
ld = lambda n: torch.load(f"{Q}/plab-{n}.pt")
eq = lambda a, b: a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
md = lambda a, b: max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
for N in (3, 8):
    o, n, f = ld(f"old-n{N}"), ld(f"new-n{N}"), ld(f"newfile-n{N}")
    s1, s2 = eq(o, n), eq(o, f); ok &= s1 and s2
    print(f"[plab] N={N}: built-in == served (dev-box file): {s1} (max |d| {md(o, n):.3g}); shipped file == served: {s2} (max |d| {md(o, f):.3g})")
o, n, l = ld("oldrnd-n3"), ld("newrnd-n3"), ld("new-n3")
s3, s4 = eq(o, n), not eq(n, l); ok &= s3 and s4
print(f"[plab] N=3 TQ_KV_T4_CODEBOOK=random12.f32: candidate == served: {s3} (max |d| {md(o, n):.3g}); differs from lloyd_kv12a: {s4} (max |d| {md(n, l):.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness, candidate, built-in codebook, 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  env TQ_KV_T4=1 TQ_KV_T4_CODEBOOK= flock $LOCK timeout 3600 $PY -u $SPECCHK --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand10-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand10-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand10-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE, pure defaults (no codebook flag, TQ_KV_T4_CODEBOOK unset)   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir $MD --lib $LIB > logs/cand10-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand10-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -oE "codebook [^)]*|[0-9]+ pooled tokens" logs/cand10-server.log | sed 's/^/   /'
grep -q "codebook built-in lloyd_kv12a" logs/cand10-server.log || { kill $SPID; fail "the server's engine did not report the built-in codebook"; }
grep -q "576000 pooled tokens" logs/cand10-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand10/needle.json > logs/cand10-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand10-needle.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand10-needle.log); NF=$(grep -c "\] FAIL" logs/cand10-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand10/spec-ab.json > logs/cand10-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand10-spec-ab.log | cut -c1-150; AB=$(grep -c "identical outputs: 10/10" logs/cand10-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand10/spec-speed-32k.json > logs/cand10-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand10-spec-speed-32k.log | cut -c1-130
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand10-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT
echo "== ALL PASS   $(date -u +%T)"
echo "[cand10-all] land by hand: [ -f $OLD.9c0a046 ] || cp -p $OLD $OLD.9c0a046; cp -p $LIB $OLD && touch $OLD;"
echo "[cand10-all]   git -C $Q/qwentin merge --ff-only cand-1201; (cd $Q/qwentin && git format-patch -q --start-number 95 9c0a046..HEAD -o ../patches/)"
