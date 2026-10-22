#!/usr/bin/env bash
# cand-1216 = main 77996fa + the fp32 64-token verify GEMM units (k_tq_exl3_wide_gemm<KB, 8, F32>): a verify wave of 33..64
# columns runs as ONE pass over the weights instead of a 32-column pass + a tail pass (N = 12 at K = 2 -- the served
# rule at full occupancy -- paid a second read + decode of every Linear per round: 49.5 vs 31.3 ms at K = 1); the engine
# reports it (qwn_verify_onepass_cols, the 61st entry point) and --spec-k auto drafts 3 deep above 8 slots (2 before).
# Candidate qwentin-rag/build-rag/. GPU 0, port 8213.
#  Gate 0  CPU tests (and with the library); exports = qwentin.h (61).
#  Gate 1  SASS 212 / 212 identical + 7 new kernels (the F32 instantiations); plab N=3/8 + --mtp-on bitwise vs the served
#          library (decode is untouched).
#  Gate 2  spec exactness vs plain decode: the fixed-K waves the new units run (N=8 K=4 = 40 columns, N=12 K=2 = 36,
#          N=12 K=4 = 60, ctx 300 at 40 columns, N=12 K=3 at 32K = 48) + the standard mixed cells 4K / 32K / 300.
#  Gate 3  THE SERVED LINE (dv64k file), no TQ_* env: the startup line says K=auto = 6/3/3 and a one-pass wave of 64
#          columns; /health draft_vocab 65536 + verify_onepass 64; the 6-prompt repro clean; needle 6/6; A/B 10/10 with
#          texts == cand21's (outputs are the verify's); speed N = 1 / 2 / 8 / 12 @32K + N = 1 @128K (tokens per round vs
#          cand26's at N = 1 / 8; N = 12 is new); T1 HumanEval+ with spec on == 93.9 / 91.5. Then the SERVED library on
#          the same server line: the startup line must say 6/3/2 (no getter -> the old table) and its N = 12 cell is the
#          baseline of the new one.
#  NOT landed by this script.
# usage: nohup logs/cand27-all.sh > logs/cand27-all.log 2>&1 < /dev/null &        (~45 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-rag
LIB=$W/build-rag/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
DV=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8213
FILT="Token indices|indexing errors|^\[config\]|^\[tqf\]|^\[spec\]|^\[graph\]|^\[kv\]|^\[embed\]|^\[mtp\]|^\[paged\]|^\[exl3\]|^$"
cd $Q; mkdir -p runs/cand27
fail() { echo "[cand27-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand27-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && CUDA_VISIBLE_DEVICES=$GPU QWENTIN_LIB=$LIB flock $LOCK $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand27-pytest-lib.log
grep -q "84 passed" logs/cand27-pytest-lib.log && ! grep -q "failed\|error" logs/cand27-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand27/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand27/header.txt
cmp -s runs/cand27/exported.txt runs/cand27/header.txt || fail "the exports differ from qwentin.h"
[ "$(wc -l < runs/cand27/exported.txt)" = 61 ] || fail "expected 61 exports"
echo "[gate0] PASS (61 exports)"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: SASS + bitwise   $(date -u +%T)"
REN=""; for ng in 1 2 4 8 16; do REN="$REN --rename ELi${ng}EEvPfS0_mPKtPKjS2_iiiiiiiiPi=ELi${ng}ELb0EEvPfS0_mPKtPKjS2_iiiiiiiiPi"; done
$PY analysis/sass_diff.py $OLD $LIB $REN > logs/cand27-sass.log; head -n 1 logs/cand27-sass.log
grep -q "^kernels: old 212, new 219, common 212; identical 212, different 0$" logs/cand27-sass.log || fail "SASS summary"
[ "$(grep -c 'only in new: _Z19k_tq_exl3_wide_gemmILi[2-8]ELi8ELb1E' logs/cand27-sass.log)" = 7 ] || fail "the 7 new kernels are not the F32 units"
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand27/$out.pt "$@" > logs/cand27-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "TQ_NO_MTP=1" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "TQ_NO_MTP=1" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $DV --mtp-on
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand27"; ok = True
for a, b, label in (("plab-new-n3", "plab-old-n3", "N=3"), ("plab-new-n8", "plab-old-n8", "N=8"), ("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (dv64k file)")):
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A); ok &= s
    print(f"[plab] {label} cand-1216 vs served: bitwise {s} ({len(A)} tensors)")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness -- the 33..64-column waves + the standard cells   $(date -u +%T)"
R() { tag=$1; shift
  flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand27-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand27-spec-$tag.log | tr '\n' ' ' | cut -c1-200)"
  grep -q "ALL PASS" logs/cand27-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k-n8k4   --ctx 4096  --n 8  --kmax 4 --k-fixed --rounds 6 --no-pin
R 4k-n12k2  --ctx 4096  --n 12 --kmax 2 --k-fixed --rounds 6 --no-pin
R 4k-n12k4  --ctx 4096  --n 12 --kmax 4 --k-fixed --rounds 6 --no-pin
R 300-n8k4  --ctx 300   --n 8  --kmax 4 --k-fixed --rounds 40 --no-pin
R 32k-n12k3 --ctx 32768 --n 12 --kmax 3 --k-fixed --rounds 6 --distinct 3 --no-pin
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
R 300 --ctx 300 --n 1,3 --rounds 40 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE (dv64k file), K auto = 6/3/3, one-pass wave 64   $(date -u +%T)"
export PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
serve() { lib=$1; log=$2
  nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $DV --model-dir $MD --lib $lib > $log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; fail "server did not come up ($log)"; }
  MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}')
  echo "   server up after $((i * 5)) s, pid $SPID, $MIB MiB; $(grep -oE '[0-9]+ pooled tokens' $log); $(grep -oE 'speculative decoding ON \([^)]*\)' $log | head -1)"; }
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
tpr() { grep -E "\[speed\] ctx" $1 | awk -v t=$2 '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $2, $3, tok / r, tok, r }'; }
serve $LIB logs/cand27-server.log
grep -q "K=auto = 6/3/3 by active slots; draft vocabulary 65536 ids; one-pass verify wave 64 columns" logs/cand27-server.log || { stop; fail "the startup line does not say 6/3/3 with a 64-column one-pass wave"; }
curl -s http://127.0.0.1:$PORT/health | grep -oE '"(draft_vocab|verify_onepass)": ?[0-9]+' | sed 's/^/   \/health /'
curl -s http://127.0.0.1:$PORT/health | grep -qE '"verify_onepass": ?64' || { stop; fail "/health verify_onepass"; }
$Q/venv-eval/bin/python analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand27/repro.json > logs/cand27-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand27-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand27-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand27/needle.json > logs/cand27-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand27-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand27-needle.log); NF=$(grep -c "\] FAIL" logs/cand27-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand27/spec-ab.json > logs/cand27-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs|tokens/round" logs/cand27-spec-ab.log | cut -c1-160; AB=$(grep -c "identical outputs: 10/10" logs/cand27-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand27/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21: on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,2,8 --modes on --max-tokens 300 --repeats 2 --out runs/cand27/spec-speed-32k.json > logs/cand27-spec-speed-32k.log 2>&1; src=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 12 --modes on --max-tokens 300 --repeats 2 --out runs/cand27/spec-speed-32k-n12.json > logs/cand27-spec-speed-32k-n12.log 2>&1; src3=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand27/spec-speed-128k.json > logs/cand27-spec-speed-128k.log 2>&1; src2=$?
grep -hE "\[speed\] ctx" logs/cand27-spec-speed-32k.log logs/cand27-spec-speed-32k-n12.log logs/cand27-spec-speed-128k.log | cut -c1-150
tpr logs/cand26-spec-speed-32k.log cand26-served; tpr logs/cand27-spec-speed-32k.log cand27; tpr logs/cand27-spec-speed-32k-n12.log cand27-n12
echo "   -- T1 HumanEval+ with spec on   $(date -u +%T)"
./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand27 humaneval > logs/cand27-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand27-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand27-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand27/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = v == [0.939, 0.915] and g == 0
print(f"   [t1] humaneval spec on, K auto 6/3/3: {v} (must equal 93.9 / 91.5); degenerate {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
ERR=$(grep -ciE "xid|error|traceback" logs/cand27-server.log)
stop
echo "   server log error lines: $ERR; repro clean $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $src/$src3/$src2; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$src3" -eq 0 ] && [ "$src2" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 3 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src/$src3/$src2, T1 rc=$t1rc, err=$ERR)"
echo "   -- the SERVED library on the same server line: K auto must read 6/3/2 (no getter), and its N = 12 cell is the baseline"
serve $OLD logs/cand27-server-old.log
grep -q "K=auto = 6/3/2 by active slots; draft vocabulary 65536 ids; one-pass verify wave 32 columns" logs/cand27-server-old.log || { stop; fail "the served library's startup line is not 6/3/2 with a 32-column wave"; }
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 12 --modes on --max-tokens 300 --repeats 2 --out runs/cand27/spec-speed-32k-n12-old.json > logs/cand27-spec-speed-32k-n12-old.log 2>&1; src4=$?
grep -hE "\[speed\] ctx" logs/cand27-spec-speed-32k-n12-old.log | cut -c1-150
tpr logs/cand27-spec-speed-32k-n12-old.log served-n12
stop
[ "$src4" -eq 0 ] || fail "the served library's N = 12 cell"
echo "[gate3] PASS"
exec 9>&-
echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1216 (ff), patches 0138-0139)"
