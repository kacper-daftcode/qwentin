#!/usr/bin/env bash
# cand-1211 = main df6c70a + e4ee81a (engine: the paged MTP drafter reads the trunk hidden AFTER the final norm -- what
# the head was trained on; the drafter thread's finding, notes/session14-drafter.md). Host code only; it changes the
# drafts, never the outputs (the verify decides every token). Candidate = qwentin-drafter/build-drafter/
# libforward_qwen.so.cand1211. GPU 0, port 8192. The change is confined to the drafter (mtp_fc_forward), so the FP6 /
# test-decoder / off-switch-matrix gates of cand21 do not apply.
#  Gate 0  CPU: ruff + pytest (and with the candidate library); exports = qwentin.h (59).
#  Gate 1  per-kernel SASS 211 / 211 identical; plab N=3/8, the 32K variant and plab --mtp-on (the drafter's trunk hooks
#          in every step, now normalizing) bitwise against the served library.
#  Gate 2  spec exactness 4K + 32K.
#  Gate 3  THE SERVED LINE (v2 file), no TQ_* environment: 576000 tokens, needle 32K + 128K 6/6, spec A/B 10/10 and its
#          texts == cand20's (outputs unchanged), speed N=1/8 with the tokens per round against cand20's.
#  NOT landed by this script.
# usage: nohup logs/cand21-all.sh > logs/cand21-all.log 2>&1 &        (~15 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-drafter
LIB=$W/build-drafter/libforward_qwen.so.cand1211
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
FP6=/mnt/nvme/models/qwen3_8-27b-e2m3-mtp.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock
FILT='^\[paged\]|^\[config\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer'
cd $Q; mkdir -p runs/cand21
fail() { echo "[cand21-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand21-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 cand-1211 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12)"

echo "== gate 0: ruff + pytest, the exported ABI   $(date -u +%T)"
git -C $W diff --quiet cand-1211 || fail "the worktree differs from cand-1211"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand21-pytest-lib.log
grep -q "84 passed" logs/cand21-pytest-lib.log && ! grep -q "failed\|error" logs/cand21-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand21/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand21/header.txt
echo "   exported qwn_*: $(wc -l < runs/cand21/exported.txt), qwentin.h: $(wc -l < runs/cand21/header.txt)"
cmp -s runs/cand21/exported.txt runs/cand21/header.txt || { diff runs/cand21/exported.txt runs/cand21/header.txt | head; fail "the exports differ from qwentin.h"; }
echo "[gate0] PASS"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise vs the served library   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand21-sass.log; head -n 1 logs/cand21-sass.log
grep -q "^kernels: old 211, new 211, common 211; identical 211, different 0$" logs/cand21-sass.log || fail "SASS summary"
BN="TQ_NO_MTP=1"   # both libraries have the served defaults
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand21/$out.pt "$@" > logs/cand21-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "$BN" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "$BN" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
C() { lib=$1; out=$2; envs=$4
  env $envs QWENTIN_TOOLS=$3 flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib runs/cand21/$out.pt --n 2 --ctx 32768 --steps 16 --tqf $T575 > logs/cand21-$out.log 2>&1 || fail "ctx $out"; }
C $OLD ctx32k-old $Q/qwentin/tools "$BN"; C $LIB ctx32k-new $W/tools "$BN"
echo "   candidate's [config] in plab: $(grep -m1 '^\[config\]' logs/cand21-plab-new-n3.log)"
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand21"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1211 vs served")
cmp("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (v2 file), cand-1211 vs served")
cmp("ctx32k-new", "ctx32k-old", "32K x 2 slots + 16 steps, cand-1211 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness 4K + 32K   $(date -u +%T)"
R() { tag=$1; shift
  flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand21-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand21-spec-$tag.log | tr '\n' ' ')"
  grep -q "ALL PASS" logs/cand21-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
echo "[gate2] PASS"

echo "== gate 3: THE SERVED LINE (v2 file), pure defaults   $(date -u +%T)"
PORT=8192
export PROBE_PORT=$PORT
env | grep -q '^TQ_' && fail "a TQ_* variable is set: $(env | grep '^TQ_' | tr '\\n' ' ')"
exec 9>$LOCK; flock 9
nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $V2 --model-dir $MD --lib $LIB > logs/cand21-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/cand21-server.log; fail "server did not come up"; }
echo "   server up after $((i * 5)) s, pid $SPID, $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}') MiB"
grep -oE "[0-9]+ pooled tokens" logs/cand21-server.log | sed 's/^/   /'
grep -q "576000 pooled tokens" logs/cand21-server.log || { kill $SPID; fail "the pool is not 576000 tokens"; }
CFG='[config] every TQ_* switch at its default'
grep -m1 '^\[config\]' logs/cand21-server.log | sed 's/^/   /'
grep -qxF "$CFG" logs/cand21-server.log || { kill $SPID; fail "the [config] line differs from docs/FLAGS.md's"; }
tr '\n' ' ' < $W/docs/FLAGS.md | tr -s ' ' | grep -qF "$CFG" || { kill $SPID; fail "FLAGS.md does not quote the served [config] line"; }
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand21/needle.json > logs/cand21-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand21-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand21-needle.log); NF=$(grep -c "\] FAIL" logs/cand21-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand21/spec-ab.json > logs/cand21-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs" logs/cand21-spec-ab.log; AB=$(grep -c "identical outputs: 10/10" logs/cand21-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand21/spec-ab.json")); b = json.load(open(f"{R}/cand20/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand20] prompts identical {same_p}; texts identical to cand20 (the served library, same file): on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand21/spec-speed-32k.json > logs/cand21-spec-speed-32k.log 2>&1; src=$?
grep -E "\[speed\] ctx" logs/cand21-spec-speed-32k.log | cut -c1-150
for t in cand20 cand21; do grep -E "\[speed\] ctx" logs/$t-spec-speed-32k.log | awk -v t=$t '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $3, tok / r, tok, r }'; done
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
ERR=$(grep -ciE "xid|error|traceback" logs/cand21-server.log)
echo "   server log error lines: $ERR; needle PASS $NP FAIL $NF; A/B identical 10/10: $AB, texts = cand20: $((1 - ABT)); speed rc $src"
[ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 3 (needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src, err=$ERR)"
echo "[gate3] PASS"
exec 9>&-

echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1211 (ff), patches 0125-0126)"
