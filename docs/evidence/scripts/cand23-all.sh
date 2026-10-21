#!/usr/bin/env bash
# cand-1212 = main ec37faf + the segment-attention fix: a verify wave that mixes S_t = 1 and S_t > 1 columns (the first
# key-split change, at position 384) had its S_t = 1 rows' final output overwritten with NaN by the split >= 1 blocks of
# k_tq_paged_attn_q4_gqa_seg / _f16 / _seg_t4 (only split 0 may write final rows). Served since 10-16: every spec-on
# request with a prompt < 384 tokens whose generation crossed 384 with a straddling wave turned into token 0 ("!!!!").
# Candidate = qwentin-segfix/build-segfix/libforward_qwen.so. GPU 0, port 8202.
#  Gate 0  CPU: ruff + pytest (and with the candidate library); exports = qwentin.h (59).
#  Gate 1  per-kernel SASS: 199 / 211 identical, the 12 instances of the three segment kernels differ and nothing else;
#          plab N=3/8, --mtp-on, 32K bitwise vs the served library (decode steps have no mixed-S_t wave).
#  Gate 2  spec exactness: the README cells 4K + 32K, AND the new unaligned short contexts that cross position 384 --
#          ctx 300 / 100 at N = 1, 3 (40 / 60 rounds) -- on the candidate (PASS) and on the served library (the evidence:
#          FAIL expected at ctx 300 N 1); the same ctx-300 cell with TQ_T4_ATTN_F16=0, on the int4-V fp16 pool
#          (TQ_KV_T4=0) and on the E4M3-V pool (TQ_KV_T4=0 TQ_KV_V4=0): the three kernels.
#  Gate 3  THE SERVED LINE, no TQ_* environment: 576000 tokens, needle 32K + 128K 6/6, spec A/B 10/10 with texts ==
#          cand21's, speed N=1/8 @32K (tokens per round vs cand21's), the 6-task repro clean, and T1 HumanEval+ / MBPP+
#          WITH SPEC ON == the spec-off reference (93.9 / 91.5 . 89.7 / 77.5) within 1 pt.
#  NOT landed by this script.
# usage: nohup logs/cand23-all.sh > logs/cand23-all.log 2>&1 < /dev/null &        (~55 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; PYE=$Q/venv-eval/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-segfix
LIB=$W/build-segfix/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8202
FILT='^\[paged\]|^\[config\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand23
fail() { echo "[cand23-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand23-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand23-pytest-lib.log
grep -q "84 passed" logs/cand23-pytest-lib.log && ! grep -q "failed\|error" logs/cand23-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand23/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand23/header.txt
cmp -s runs/cand23/exported.txt runs/cand23/header.txt || fail "the exports differ from qwentin.h"
echo "[gate0] PASS ($(wc -l < runs/cand23/exported.txt) exports)"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise decode vs the served library   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand23-sass.log; head -n 1 logs/cand23-sass.log
grep -q "^kernels: old 211, new 211, common 211; identical 199, different 12$" logs/cand23-sass.log || fail "SASS summary"
ND=$(grep "DIFFERENT:" logs/cand23-sass.log | grep -cE "k_tq_paged_attn_q4_gqa_seg|k_tq_paged_attn_q4_gqa_f16|k_tq_paged_attn_q4_gqa_seg_t4")
echo "   differing kernels that are instances of the three segment kernels: $ND / 12"
[ "$ND" -eq 12 ] || { grep "DIFFERENT:" logs/cand23-sass.log; fail "a kernel outside the three changed"; }
BN="TQ_NO_MTP=1"
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand23/$out.pt "$@" > logs/cand23-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "$BN" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "$BN" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
C() { lib=$1; out=$2; envs=$4
  env $envs QWENTIN_TOOLS=$3 flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib runs/cand23/$out.pt --n 2 --ctx 32768 --steps 16 --tqf $T575 > logs/cand23-$out.log 2>&1 || fail "ctx $out"; }
C $OLD ctx32k-old $Q/qwentin/tools "$BN"; C $LIB ctx32k-new $W/tools "$BN"
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand23"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1212 vs served")
cmp("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (v2 file), cand-1212 vs served")
cmp("ctx32k-new", "ctx32k-old", "32K x 2 slots + 16 steps, cand-1212 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness -- the README cells and the unaligned short contexts across position 384   $(date -u +%T)"
R() { lib=$1; tools=$2; tag=$3; envs=$4; expect=$5; shift 5
  env $envs flock $LOCK timeout 3600 $PY -u $tools/spec_paged_check.py --lib $lib --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand23-spec-$tag.log
  v=$(grep -E '^\[spec-check\] (ALL PASS|[0-9]+ FAIL)' logs/cand23-spec-$tag.log | head -1 | sed -E 's/^\[spec-check\] //; s/ \(.*//')
  echo "   $tag [$envs]: $v   (expected $expect)   $(grep -E '^\[spec-check\] unpinned' logs/cand23-spec-$tag.log | sed -E 's/.*columns: //' | cut -c1-70)"
  case "$expect" in PASS) grep -q "ALL PASS" logs/cand23-spec-$tag.log || fail "spec_paged_check $tag";;
                    FAIL) grep -q "ALL PASS" logs/cand23-spec-$tag.log && echo "   NB: the served library passed $tag (the NaN race did not bite in this cell; informational)";; esac; }
R $LIB $W/tools new-4k  "NOENV=1" PASS --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R $LIB $W/tools new-32k "NOENV=1" PASS --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
R $LIB $W/tools new-300 "NOENV=1" PASS --ctx 300 --n 1,3 --rounds 40 --no-pin
R $LIB $W/tools new-100 "NOENV=1" PASS --ctx 100 --n 1,3 --rounds 60 --no-pin
R $OLD $Q/qwentin/tools old-300 "NOENV=1" FAIL --ctx 300 --n 1 --rounds 40 --no-pin
R $LIB $W/tools new-300-f16off "TQ_T4_ATTN_F16=0" PASS --ctx 300 --n 1,3 --rounds 40 --no-pin
R $OLD $Q/qwentin/tools old-300-f16off "TQ_T4_ATTN_F16=0" FAIL --ctx 300 --n 1 --rounds 40 --no-pin
R $LIB $W/tools new-300-v4fp16 "TQ_KV_T4=0 TQ_KV_V4=1" PASS --ctx 300 --n 1,3 --rounds 40 --no-pin
R $OLD $Q/qwentin/tools old-300-v4fp16 "TQ_KV_T4=0 TQ_KV_V4=1" FAIL --ctx 300 --n 1 --rounds 40 --no-pin
R $LIB $W/tools new-300-e4m3 "TQ_KV_T4=0 TQ_KV_V4=0" PASS --ctx 300 --n 1,3 --rounds 40 --no-pin
R $OLD $Q/qwentin/tools old-300-e4m3 "TQ_KV_T4=0 TQ_KV_V4=0" FAIL --ctx 300 --n 1 --rounds 40 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE, pure defaults, with T1 with spec on   $(date -u +%T)"
export PROBE_PORT=$PORT
env | grep -q '^TQ_' && fail "a TQ_* variable is set"
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/cand23-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand23-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB; $(grep -oE '[0-9]+ pooled tokens' logs/cand23-server.log); $(grep -m1 '^\[config\]' logs/cand23-server.log)"
grep -q "576000 pooled tokens" logs/cand23-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
grep -qxF '[config] every TQ_* switch at its default' logs/cand23-server.log || { kill $SPID; fail "the [config] line"; }
$PYE analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand23/repro.json > logs/cand23-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand23-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand23-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand23/needle.json > logs/cand23-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand23-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand23-needle.log); NF=$(grep -c "\] FAIL" logs/cand23-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand23/spec-ab.json > logs/cand23-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs" logs/cand23-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand23-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand23/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21 (the served library, same file): on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand23/spec-speed-32k.json > logs/cand23-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand23-spec-speed-32k.log | cut -c1-150
for t in cand21 cand23; do grep -E "\[speed\] ctx" logs/$t-spec-speed-32k.log | awk -v t=$t '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $3, tok / r, tok, r }'; done
echo "== T1 (HumanEval+ / MBPP+) WITH SPEC ON   $(date -u +%T)"
for ds in humaneval mbpp; do
  ./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand23-spec $ds > logs/cand23-t1-$ds.log 2>&1; echo "   $ds exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand23-t1-$ds.log | tr -s '\t ' ' ' | tr '\n' ' ')"
done
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
ok = True
ref = {"humaneval": [0.939, 0.915], "mbpp": [0.897, 0.775]}   # docs/QUALITY.md § 6: the served stack with spec OFF (2026-10-15; re-read 10-21 09:04, logs/t1-probe-served-nospec.log)
for ds in ref:
    v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open(f"/mnt/nvme/qwen256k/logs/cand23-t1-{ds}.log").read())]
    raw = [json.loads(l) for l in open(glob.glob(f"/mnt/nvme/qwen256k/runs/evalplus-cand23-spec/{ds}/*.raw.jsonl")[0])]
    g = sum("!!!!!!!!" in r["solution"] for r in raw)
    good = len(v) == 2 and all(abs(a - b) <= 0.01 + 1e-9 for a, b in zip(v, ref[ds])) and g == 0
    ok &= good
    print(f"   [t1] {ds} spec on: {v} vs spec off {ref[ds]}; garbage generations {g} / {len(raw)} -> {'ok' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand23-server.log)
echo "   server log error lines: $ERR; repro clean: $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $src; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 3 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src, T1 rc=$t1rc, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-

echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1212 (ff), patches 0127-0128)"
