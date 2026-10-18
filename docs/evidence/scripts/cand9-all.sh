#!/usr/bin/env bash
# cand-1102 = main 50b7e3d + the session-11 prefill-attention thread (6ee1751, a03975a) + b2bdd08 (TQ_PATTN_GQA default 1).
# The lever is NOT bitwise (a different tiling of the same fp32 online softmax) -> the gate is the e2e yardstick within CI.
# Candidate qwentin-cand/build-qwen-cand/libforward_qwen.so (c40173fe). GPU 2 under its lock, port 8130.
#  Gate 1  off-switch: candidate with TQ_PATTN_GQA=0 == served binary (484bdb82) on the trellis pool (plab N=3/8), bitwise.
#  Gate 2  quality: e2e prefill rows on the bare new defaults at 4K (48 x 4096) and 32K (6 x 32768) vs the served library's rows
#          (qw-final-def_4k_prefill / qw-final-def_32k_prefill; the GDN lever is bitwise, so those rows are the served binary's):
#          FP8-gate rule = dKL within the 95 % CI (|dKL| <= 0.0001 with the CI covering 0) and top-1 CI across 0, incl. --min-pos 16384.
#  Gate 3  spec exactness, bare defaults, 4K + 32K (the verify wave does not take this kernel; prove it).
#  Gate 4  THE SERVED LINE, pure defaults: needle 32K + 128K (TTFT: expect ~12-13 s / ~62-70 s), spec A/B 10, speed N=1/8.
#  On ALL PASS: build-qwen/ := candidate (old kept as .50b7e3d), main := cand-1102 (ff), patches/0088-0090.
# usage: nohup logs/cand9-all.sh > logs/cand9-all.log 2>&1 &        (~60 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-cand/build-qwen-cand/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
SERVE=$Q/qwentin-cand/tools/serve_batched.py
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
T575M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8130; GPU=2; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand9
fail() { echo "[cand9-all] FAIL: $*   $(date -u +%T)"; echo "[cand9-all] nothing swapped; build-qwen/ and main stay at 50b7e3d"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU TQ_KV_T4_CODEBOOK=$CB
unset TQ_KV_T4 TQ_NO_MTP TQ_KV_V4 TQ_PDL TQ_PAGED_ATTN_ST TQ_EMBED_HOST TQ_EMBED_FP8 TQ_PATTN_GQA TQ_GDN_PREFILL
echo "[cand9-all] start $(date -u +%T); cand $(md5sum $LIB | cut -c1-12) = $(git -C qwentin-cand log --oneline -1 | cut -c1-50); served $(md5sum $OLD | cut -c1-12) = $(git -C qwentin log --oneline -1 | cut -c1-40)"
B="TQ_PDL=1 TQ_KV_V4=1 TQ_KV_T4=1 TQ_NO_MTP=1"
echo "== gate 1: off-switch bitwise (plab N=3/8, TQ_PATTN_GQA=0 vs served)   $(date -u +%T)"
for N in 3 8; do
  env $B flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $OLD runs/cand9/plab-old-n$N.pt --n $N --tqf $T575 > logs/cand9-plab-old-n$N.log 2>&1 || fail "plab old N=$N"
  env $B TQ_PATTN_GQA=0 flock $LOCK timeout 900 $PY -u analysis/paged_logit_ab.py $LIB runs/cand9/plab-newoff-n$N.pt --n $N --tqf $T575 > logs/cand9-plab-newoff-n$N.log 2>&1 || fail "plab new+off N=$N"
done
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand9"; ok = True
for N in (3, 8):
    a = torch.load(f"{Q}/plab-old-n{N}.pt"); b = torch.load(f"{Q}/plab-newoff-n{N}.pt")
    s = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    md = max((a[k] - b[k]).abs().max().item() for k in a) if a.keys() == b.keys() else float("nan")
    ok &= s; print(f"[plab] N={N}: new + TQ_PATTN_GQA=0 == served: {s} (max |d| {md:.3g})")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
echo "== gate 2: e2e prefill rows, bare new defaults, vs the served rows   $(date -u +%T)"
export TQ_KV_V4=1 TQ_PDL=1 TQ_KV_T4=1 TQ_EMBED_HOST=0 TQ_EMBED_FP8=1
eval/qw_e2e_run.sh $GPU qw-cand1102_4k_prefill  --tqf $T575 --lib $LIB --seq 4096  --decode-from 4096
eval/qw_e2e_run.sh $GPU qw-cand1102_32k_prefill --tqf $T575 --lib $LIB --seq 32768 --decode-from 32768
unset TQ_KV_V4 TQ_PDL TQ_KV_T4 TQ_EMBED_HOST TQ_EMBED_FP8
$PY analysis/e2e_frontier.py --tags qw-final-def_4k_prefill qw-cand1102_4k_prefill --vs qw-final-def_4k_prefill 2>/dev/null | tail -3 | tee logs/cand9-frontier.txt
for mp in 0 16384 24576; do $PY analysis/e2e_frontier.py --min-pos $mp --tags qw-final-def_32k_prefill qw-cand1102_32k_prefill --vs qw-final-def_32k_prefill 2>/dev/null | tail -2 | tee -a logs/cand9-frontier.txt; done
$PY - <<'PY' || fail "gate 2 (quality)"
import re, sys
rows = [l for l in open("/mnt/nvme/qwen256k/logs/cand9-frontier.txt") if l.startswith("qw-cand1102")]
ok = True
for l in rows:
    m = re.search(r"([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\]\s+[\d.]+\s+([+-]\d\.\d+) \[([+-]\d\.\d+),([+-]\d\.\d+)\]", l)
    if not m: print("[gate2] cannot parse:", l.strip()); ok = False; continue
    d, lo, hi, t, tlo, thi = map(float, m.groups())
    kl_ok = lo <= 0.0 and abs(d) <= 0.0001; top_ok = tlo <= 0.0 <= thi
    ok &= kl_ok and top_ok
    print(f"[gate2] {l.split()[0]}: dKL {d:+.5f} [{lo:+.5f},{hi:+.5f}] {'ok' if kl_ok else 'FAIL'}; dtop1 {t:+.2f} [{tlo:+.2f},{thi:+.2f}] {'ok' if top_ok else 'FAIL'}")
print("[gate2]", "PASS" if ok and rows else "FAIL"); sys.exit(0 if ok and rows else 1)
PY
echo "== gate 3: spec exactness, bare defaults, 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  TQ_KV_T4=1 flock $LOCK timeout 3600 $PY -u analysis/spec_paged_check.py --lib $LIB --tqf $T575 "$@" 2>&1 | grep -Ev "$FILT" > logs/cand9-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|FAIL)' logs/cand9-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand9-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate3] PASS"
echo "== gate 4: THE SERVED LINE, pure defaults   $(date -u +%T)"
export TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
nohup $PY -u $SERVE --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $T575M --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB > logs/cand9-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand9-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/cand9/needle.json > logs/cand9-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)|^wall" logs/cand9-needle.log | cut -c1-110
NP=$(grep -c "\] PASS" logs/cand9-needle.log); NF=$(grep -c "\] FAIL" logs/cand9-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand9/spec-ab.json > logs/cand9-spec-ab.log 2>&1; abrc=$?
grep -E "^\[ab\]" logs/cand9-spec-ab.log | cut -c1-150; AB=$(grep -c "identical outputs: 10/10" logs/cand9-spec-ab.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand9/spec-speed-32k.json > logs/cand9-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand9-spec-speed-32k.log | cut -c1-130
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand9-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB; speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] || fail "gate 4 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB, speed rc=$src, err=$ERR)"
echo "[gate4] PASS"
exec 9>&-
unset TQ_CTX TQ_KV_Q4 TQ_EMBED_FP8 TQ_PAGED_ATTN_MMA PROBE_PORT
echo "== ALL PASS -> landing cand-1102 on main   $(date -u +%T)"
[ -f $OLD.50b7e3d ] || cp -p $OLD $OLD.50b7e3d
cp -p $LIB $OLD && touch $OLD
git -C qwentin merge -q --ff-only cand-1102 || fail "ff-merge (build-qwen/ already holds the gated library; old at .50b7e3d)"
(cd qwentin && git format-patch -q --start-number 88 50b7e3d..HEAD -o ../patches/) && ls patches/008[89]*.patch patches/009*.patch 2>/dev/null | tail -3
echo "[cand9-all] LANDED $(date -u +%T): build-qwen/libforward_qwen.so = $(md5sum $OLD | cut -c1-12) (old .50b7e3d); main = $(git -C qwentin log --oneline -1 | cut -c1-60)"
echo "[cand9-all] rollback: cp $OLD.50b7e3d $OLD && git -C $Q/qwentin reset --hard 50b7e3d"
