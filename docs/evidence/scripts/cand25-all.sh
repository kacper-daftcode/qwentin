#!/usr/bin/env bash
# cand-1214 = main 0b41c54 (v0.12) + 918abf2: the DRAFT lm_head section (the drafter argmaxes over the 65 536 most frequent
# token ids through a 6-bit EXL3 head, 0.25 GB per draft step instead of 0.95; the verify keeps the full head). Two things
# to gate: the LIBRARY (bitwise everywhere the drafter is not involved; a file without the section = the served behaviour)
# and the FILE /mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf (the served checkpoint reconverted with
# --mtp-exl3 --draft-head: its payload must be the served v2 file's byte for byte up to the end of the MTP section).
# Candidate library qwentin-dv/build-dv/libforward_qwen.so. GPU 0, port 8209.
#  Gate 0  CPU: ruff + pytest (and with the candidate library); exports = qwentin.h (59); inspect_tqf --verify of the new
#          file (EXL3 + MTP PASS, draft_head V' = 65536); the payload prefix cmp-identical to the served v2 file's.
#  Gate 1  per-kernel SASS 211 / 211 identical + 1 new kernel; plab N=3/8, 32K and --mtp-on (served v2 file) bitwise vs
#          the served library; the candidate on the NEW file with TQ_DRAFT_VOCAB=0 and =1 bitwise on the main logits too
#          (--mtp-on: the trunk hooks are unchanged; the drafts never touch the main path).
#  Gate 2  spec exactness 4K + 32K + ctx 300 (the verify is unchanged).
#  Gate 3  THE SERVED LINE with the new file, no TQ_* environment (TQ_DRAFT_VOCAB default 1): 576000 tokens, footprint
#          <= 29898 + 300 MiB, needle 32K + 128K 6/6, spec A/B 10/10 with texts == cand21's, the 6-prompt repro clean,
#          speed N=1/8 @32K + N=1 @128K with tokens per round vs cand23's, T1 HumanEval+ with spec on == 93.9 / 91.5.
#  Gate 4  the switch off: the same server with TQ_DRAFT_VOCAB=0 -> the drafts are the stock head's: tokens per round
#          == cand23's (3.85 / 18.46), A/B texts == cand21's. Then the 38-prompt engine traces (analysis/drafter/
#          engine_trace.py) with the new file vs the served file: outputs 38/38 identical, tokens per round and tok/s.
#  NOT landed by this script.
# usage: nohup logs/cand25-all.sh > logs/cand25-all.log 2>&1 < /dev/null &        (~45 min)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; CIV=/root/work/ci-venv/bin
W=$Q/qwentin-dv
LIB=$W/build-dv/libforward_qwen.so
OLD=$Q/qwentin/build-qwen/libforward_qwen.so
T575=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp.tqf
V2=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf
DV=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-dv64k.tqf
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
GPU=0; LOCK=/tmp/exl3diff-gpu$GPU.lock; PORT=8209
FILT='^\[paged\]|^\[config\]|Warning|warn|layer .*loaded|^TQ_|^TQF1|^KV prefix|^attention|^context|MTP head|transformers|mtp_layer|DRAFT'
cd $Q; mkdir -p runs/cand25
fail() { echo "[cand25-all] FAIL: $*   $(date -u +%T)"; exit 1; }
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
echo "[cand25-all] start $(date -u +%T); cand $(md5sum < $LIB | cut -c1-12) = $(git -C $W log --oneline -1 | cut -c1-60); served $(md5sum < $OLD | cut -c1-12); file $DV ($(stat -c %s $DV) bytes)"

echo "== gate 0: ruff + pytest, the exported ABI, the file   $(date -u +%T)"
git -C $W diff --quiet HEAD || fail "the worktree has uncommitted changes"
(cd $W && $CIV/ruff check tools/ tests/ && $CIV/python -m pytest -q tests/ 2>&1 | tail -n 1) || fail "gate 0 (CPU)"
(cd $W && QWENTIN_LIB=$LIB $CIV/python -m pytest -q tests/ 2>&1 | grep -E "passed|failed|error" | tail -n 1) | tee logs/cand25-pytest-lib.log
grep -q "84 passed" logs/cand25-pytest-lib.log && ! grep -q "failed\|error" logs/cand25-pytest-lib.log || fail "gate 0 (with the library)"
nm -D --defined-only $LIB | awk '$2 == "T" && $3 ~ /^qwn_/ {print $3}' | sort > runs/cand25/exported.txt
grep -oE "\bqwn_[a-z0-9_]+\(" $W/src/qwentin.h | tr -d '(' | sort -u > runs/cand25/header.txt
cmp -s runs/cand25/exported.txt runs/cand25/header.txt || fail "the exports differ from qwentin.h"
$PY $W/tools/inspect_tqf.py --verify --expect exl3-mtp $DV > logs/cand25-inspect.log 2>&1 || { tail -5 logs/cand25-inspect.log; fail "inspect_tqf --verify"; }
grep -E "^flags|draft_head|payload hash|verdict EXL3" logs/cand25-inspect.log | cut -c1-150 | sed 's/^/   /'
grep -q "draft_head: V' = 65536" logs/cand25-inspect.log || fail "inspect_tqf does not see the draft section"
PV2=$($PY $W/tools/inspect_tqf.py $V2 2>/dev/null | grep -oE "^expected_payload_bytes: [0-9]+" | grep -oE "[0-9]+$")
HB=$($PY $W/tools/inspect_tqf.py $V2 2>/dev/null | grep -oE "^header_bytes: [0-9]+" | grep -oE "[0-9]+$")
HB2=$($PY $W/tools/inspect_tqf.py $DV 2>/dev/null | grep -oE "^header_bytes: [0-9]+" | grep -oE "[0-9]+$")
[ "$HB" = "$HB2" ] || fail "header sizes differ ($HB vs $HB2)"
cmp -s <(tail -c +$((HB + 1)) $V2 | head -c $PV2) <(tail -c +$((HB2 + 1)) $DV | head -c $PV2) && echo "   payload prefix ($PV2 bytes = the served file's whole payload) identical to the served v2 file" || fail "the new file's payload prefix differs from the served file's"
echo "[gate0] PASS ($(wc -l < runs/cand25/exported.txt) exports)"

export CUDA_VISIBLE_DEVICES=$GPU
echo "== gate 1: per-kernel SASS + bitwise decode vs the served library; the new file's main logits with the draft head on / off   $(date -u +%T)"
$PY analysis/sass_diff.py $OLD $LIB > logs/cand25-sass.log; head -n 2 logs/cand25-sass.log
grep -q "^kernels: old 211, new 212, common 211; identical 211, different 0$" logs/cand25-sass.log || fail "SASS summary"
grep -q "only in new: .*k_tq_batched_argmax_map" logs/cand25-sass.log || fail "the new kernel is not the mapped argmax"
BN="TQ_NO_MTP=1"
P() { lib=$1; out=$2; tools=$3; envs=$4; shift 4
  env $envs flock $LOCK timeout 900 $PY -u $tools/paged_logit_ab.py $lib runs/cand25/$out.pt "$@" > logs/cand25-$out.log 2>&1 || fail "plab $out"; }
for N in 3 8; do P $OLD plab-old-n$N $Q/qwentin/tools "$BN" --n $N --tqf $T575; P $LIB plab-new-n$N $W/tools "$BN" --n $N --tqf $T575; done
P $OLD plab-old-mtp $Q/qwentin/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
P $LIB plab-new-mtp $W/tools "NOENV=1" --n 3 --tqf $V2 --mtp-on
P $LIB plab-dv-mtp-on  $W/tools "TQ_DRAFT_VOCAB=1" --n 3 --tqf $DV --mtp-on
P $LIB plab-dv-mtp-off $W/tools "TQ_DRAFT_VOCAB=0" --n 3 --tqf $DV --mtp-on
echo "   loader lines on the new file: $(grep -E 'DRAFT lm_head section' logs/cand25-plab-dv-mtp-on.log | head -1 | cut -c1-120) | $(grep -E 'DRAFT lm_head section' logs/cand25-plab-dv-mtp-off.log | head -1 | cut -c1-80)"
C() { lib=$1; out=$2; envs=$4
  env $envs QWENTIN_TOOLS=$3 flock $LOCK timeout 1800 $PY -u analysis/paged_logit_ab_ctx.py $lib runs/cand25/$out.pt --n 2 --ctx 32768 --steps 16 --tqf $T575 > logs/cand25-$out.log 2>&1 || fail "ctx $out"; }
C $OLD ctx32k-old $Q/qwentin/tools "$BN"; C $LIB ctx32k-new $W/tools "$BN"
$PY - <<'PY' || fail "gate 1"
import torch, sys
Q = "/mnt/nvme/qwen256k/runs/cand25"; ok = True
def cmp(a, b, label):
    global ok
    A = torch.load(f"{Q}/{a}.pt"); B = torch.load(f"{Q}/{b}.pt")
    s = A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A)
    ok &= s; print(f"[plab] {label}: bitwise {s} ({len(A)} tensors)")
for N in (3, 8): cmp(f"plab-new-n{N}", f"plab-old-n{N}", f"N={N} cand-1214 vs served")
cmp("plab-new-mtp", "plab-old-mtp", "N=3 --mtp-on (served v2 file), cand-1214 vs served")
cmp("plab-dv-mtp-on", "plab-old-mtp", "N=3 --mtp-on, the NEW file with the draft head ON vs the served library on the served file")
cmp("plab-dv-mtp-off", "plab-old-mtp", "N=3 --mtp-on, the NEW file with TQ_DRAFT_VOCAB=0 vs the served library on the served file")
cmp("ctx32k-new", "ctx32k-old", "32K x 2 slots + 16 steps, cand-1214 vs served")
print("[gate1]", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY

echo "== gate 2: spec exactness 4K + 32K + ctx 300   $(date -u +%T)"
R() { tag=$1; shift
  flock $LOCK timeout 3600 $PY -u $W/tools/spec_paged_check.py --lib $LIB --tqf $T575 --model-dir $MD "$@" 2>&1 | grep -Ev "$FILT" > logs/cand25-spec-$tag.log
  echo "   $tag: $(grep -E '^\[spec-check\] (unpinned|ALL|[0-9]+ FAIL)' logs/cand25-spec-$tag.log | tr '\n' ' ' | cut -c1-200)"
  grep -q "ALL PASS" logs/cand25-spec-$tag.log || fail "spec_paged_check $tag"; }
R 4k  --ctx 4096  --n 1,3,8 --rounds 6 --no-pin
R 32k --ctx 32768 --n 1,3,8 --rounds 6 --distinct 3 --no-pin
R 300 --ctx 300 --n 1,3 --rounds 40 --no-pin
echo "[gate2] PASS"

serve() { envs=$1; log=$2
  env $envs nohup $PY -u $W/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
    --tqf $DV --model-dir $MD --lib $LIB > $log 2>&1 &
  SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
  [ -n "$SPID" ] || { tail -5 $log; fail "server did not come up ($log)"; }
  MIB=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$SPID -F', ' '$1==p {print $2}')
  echo "   server up after $((i * 5)) s, pid $SPID, $MIB MiB (cand23: 29898); $(grep -oE '[0-9]+ pooled tokens' $log); $(grep -m1 '^\[config\]' $log | cut -c1-90); $(grep -oE 'DRAFT lm_head section (loaded|skipped)[^(]*' $log | head -1)"
  grep -q "576000 pooled tokens" $log || { kill $SPID; fail "the pool is not 576000 tokens"; }; }
stop() { kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done; }
tpr() { grep -E "\[speed\] ctx" $1 | awk -v t=$2 '{for (i = 1; i <= NF; i++) { if ($i == "tokens" && $(i+1) == "in") tok = $(i-1); if ($i == "rounds") r = $(i+1) } printf "   %s %s %s: %.2f tokens per round (%s tokens / %s rounds)\n", t, $2, $3, tok / r, tok, r }'; }

echo "== gate 3: THE SERVED LINE with the new file (TQ_DRAFT_VOCAB default 1)   $(date -u +%T)"
export PROBE_PORT=$PORT
exec 9>$LOCK; flock 9
serve "NOENV=1" logs/cand25-server.log
grep -q "DRAFT lm_head section loaded" logs/cand25-server.log || { stop; fail "the draft head was not loaded"; }
[ "$MIB" -le 30198 ] || { stop; fail "footprint $MIB MiB > 30198"; }
$Q/venv-eval/bin/python analysis/t1_repro.py --base http://127.0.0.1:$PORT --out runs/cand25/repro.json > logs/cand25-repro.log 2>&1; rrc=$?
grep "garbage" logs/cand25-repro.log | tail -1 | sed 's/^/   /'; RG=$(grep -c "garbage 0 / 6" logs/cand25-repro.log)
timeout 3000 $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths 32768,131072 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer $MD --chat-kwargs '{"enable_thinking": false}' --out runs/cand25/needle.json > logs/cand25-needle.log 2>&1; nrc=$?
grep -E "\] (PASS|FAIL)" logs/cand25-needle.log | cut -c1-100
NP=$(grep -c "\] PASS" logs/cand25-needle.log); NF=$(grep -c "\] FAIL" logs/cand25-needle.log)
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand25/spec-ab.json > logs/cand25-spec-ab.log 2>&1; abrc=$?
grep -E "identical outputs|tokens/round" logs/cand25-spec-ab.log | cut -c1-160; AB=$(grep -c "identical outputs: 10/10" logs/cand25-spec-ab.log)
$PY - <<'PY'; ABT=$?
import json, sys
R = "/mnt/nvme/qwen256k/runs"
a = json.load(open(f"{R}/cand25/spec-ab.json")); b = json.load(open(f"{R}/cand21/spec-ab.json"))
same_p = a["prompts"] == b["prompts"]
same = {m: [x["text"] for x in a["res"][m]] == [x["text"] for x in b["res"][m]] for m in ("on", "off")}
print(f"[ab-vs-cand21] prompts identical {same_p}; texts identical to cand21 (the served library + file): on {same['on']}, off {same['off']}")
sys.exit(0 if same_p and all(same.values()) else 1)
PY
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --repeats 2 --out runs/cand25/spec-speed-32k.json > logs/cand25-spec-speed-32k.log 2>&1; src=$?
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 131072 --n 1 --modes on --max-tokens 300 --out runs/cand25/spec-speed-128k.json > logs/cand25-spec-speed-128k.log 2>&1; src2=$?
grep -hE "\[speed\] ctx" logs/cand25-spec-speed-32k.log logs/cand25-spec-speed-128k.log | cut -c1-150
tpr logs/cand23-spec-speed-32k.log cand23-stock; tpr logs/cand25-spec-speed-32k.log cand25-draft64k
echo "   -- T1 HumanEval+ with spec on   $(date -u +%T)"
./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b cand25-dv humaneval > logs/cand25-t1-humaneval.log 2>&1; echo "   humaneval exit $? $(date -u +%T): $(grep -E 'pass@1' logs/cand25-t1-humaneval.log | tr -s '\t ' ' ' | tr '\n' ' ')"
$PY - <<'PY'; t1rc=$?
import re, sys, json, glob
v = [float(x) for x in re.findall(r"pass@1:\s*([\d.]+)", open("/mnt/nvme/qwen256k/logs/cand25-t1-humaneval.log").read())]
raw = [json.loads(l) for l in open(glob.glob("/mnt/nvme/qwen256k/runs/evalplus-cand25-dv/humaneval/*.raw.jsonl")[0])]
g = sum("!!!!!!!!" in r["solution"] for r in raw)
ok = v == [0.939, 0.915] and g == 0
print(f"   [t1] humaneval spec on, draft head: {v} (the spec-off reference 93.9 / 91.5 -- must be equal: the outputs are the same tokens); degenerate {g} / {len(raw)} -> {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
ERR=$(grep -ciE "xid|error|traceback" logs/cand25-server.log)
stop
echo "   server log error lines: $ERR; repro clean $RG; needle PASS $NP FAIL $NF; A/B 10/10: $AB, texts = cand21: $((1 - ABT)); speed rc $src/$src2; T1 rc $t1rc"
[ "$rrc" -eq 0 ] && [ "$RG" -eq 1 ] && [ "$nrc" -eq 0 ] && [ "$NF" -eq 0 ] && [ "$NP" -ge 6 ] && [ "$abrc" -eq 0 ] && [ "$AB" -eq 1 ] && [ "$ABT" -eq 0 ] && [ "$src" -eq 0 ] && [ "$src2" -eq 0 ] && [ "$t1rc" -eq 0 ] && [ "$ERR" -eq 0 ] \
  || fail "gate 3 (repro rc=$rrc clean=$RG, needle rc=$nrc PASS=$NP FAIL=$NF, ab rc=$abrc identical=$AB texts=$ABT, speed rc=$src/$src2, T1 rc=$t1rc, err=$ERR)"
echo "[gate3] PASS"

echo "== gate 4: the switch off (TQ_DRAFT_VOCAB=0 on the new file = the stock drafts), then the 38-prompt engine traces   $(date -u +%T)"
serve "TQ_DRAFT_VOCAB=0" logs/cand25-server-off.log
grep -q "DRAFT lm_head section skipped (TQ_DRAFT_VOCAB=0)" logs/cand25-server-off.log || { stop; fail "the off switch did not skip the draft head"; }
timeout 1800 $PY analysis/spec_server_ab.py --base http://127.0.0.1:$PORT --prompts 10 --max-tokens 300 --out runs/cand25/spec-ab-off.json > logs/cand25-spec-ab-off.log 2>&1
grep -E "identical outputs|tokens/round" logs/cand25-spec-ab-off.log | cut -c1-160; ABO=$(grep -c "identical outputs: 10/10" logs/cand25-spec-ab-off.log)
timeout 2400 $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1,8 --modes on --max-tokens 300 --out runs/cand25/spec-speed-32k-off.json > logs/cand25-spec-speed-32k-off.log 2>&1
grep -hE "\[speed\] ctx" logs/cand25-spec-speed-32k-off.log | cut -c1-150
tpr logs/cand25-spec-speed-32k-off.log cand25-switch-off
$PY - <<'PY'; OFFT=$?
import re, sys
def rounds(path):
    out = {}
    for l in open(path):
        m = re.search(r"ctx=\d+ N=(\d+) spec=on.*?rounds (\d+) steps", l)
        if m: out.setdefault(m.group(1), m.group(2))
    return out
a, b = rounds("/mnt/nvme/qwen256k/logs/cand25-spec-speed-32k-off.log"), rounds("/mnt/nvme/qwen256k/logs/cand23-spec-speed-32k.log")
print(f"[switch-off] rounds per 300 tokens at N=1 / 8: new file + TQ_DRAFT_VOCAB=0 {a} vs cand23 (stock) {b} -> {'the stock drafts' if a == b else 'DIFFERENT'}")
sys.exit(0 if a == b else 1)
PY
stop
[ "$ABO" -eq 1 ] && [ "$OFFT" -eq 0 ] || fail "gate 4 (switch off: ab=$ABO rounds-equal=$((1 - OFFT)))"
exec 9>&-
for cfg in "served $V2 $OLD $Q/qwentin/tools" "dv64k $DV $LIB $W/tools"; do set -- $cfg
  flock $LOCK timeout 3600 $PY -u analysis/drafter/engine_trace.py --prompts analysis/drafter/data/eval_prompts.json --out runs/cand25/trace-$1.jsonl \
    --sets vllmab,qwentinab,heldout,vllm32k --max-new 300 --tqf $2 --lib $3 --tag cand25-$1 > logs/cand25-trace-$1.log 2>&1 || fail "engine_trace $1"
done
$PY - <<'PY' || fail "gate 4 (traces)"
import json, sys
R = "/mnt/nvme/qwen256k/runs/cand25"
def load(t):
    return {(r["set"], r["idx"]): r for r in map(json.loads, open(f"{R}/trace-{t}.jsonl"))}
a, b = load("served"), load("dv64k")
keys = sorted(set(a) & set(b)); same = sum(a[k]["gen"] == b[k]["gen"] for k in keys)
def agg(d):
    tok = sum(len(d[k]["gen"]) for k in keys); rounds = sum(len(d[k]["rounds"]) for k in keys); sec = sum(d[k]["decode_s"] for k in keys)
    return tok / rounds, tok / sec
ta, sa = agg(a); tb, sb = agg(b)
print(f"[traces] {len(keys)} prompts: outputs identical {same}/{len(keys)}; tokens per round served {ta:.3f} -> draft64k {tb:.3f} ({(tb / ta - 1) * 100:+.1f} %); tok/s {sa:.1f} -> {sb:.1f} ({(sb / sa - 1) * 100:+.1f} %)")
for s in sorted({k[0] for k in keys}):
    ks = [k for k in keys if k[0] == s]
    ra = sum(len(a[k]["gen"]) for k in ks) / sum(len(a[k]["rounds"]) for k in ks); rb = sum(len(b[k]["gen"]) for k in ks) / sum(len(b[k]["rounds"]) for k in ks)
    va = sum(len(a[k]["gen"]) for k in ks) / sum(a[k]["decode_s"] for k in ks); vb = sum(len(b[k]["gen"]) for k in ks) / sum(b[k]["decode_s"] for k in ks)
    print(f"   {s:10s} {len(ks):2d} prompts: {ra:.3f} -> {rb:.3f} tok/round, {va:.1f} -> {vb:.1f} tok/s")
sys.exit(0 if same == len(keys) else 1)
PY
echo "[gate4] PASS"

echo "== ALL PASS   $(date -u +%T)  (landing: build-qwen/ := the candidate, main := cand-1214 (ff), the served --tqf := $DV, patches 0131-0132)"
