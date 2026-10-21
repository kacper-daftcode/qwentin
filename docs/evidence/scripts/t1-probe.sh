#!/usr/bin/env bash
# T1 probe: a fresh batched server (12 slots, the served line unless extra args say otherwise), EvalPlus HumanEval(+)
# greedy through it, then the pass@1 and the number of generations that degenerate into "!!!!" (token 0 = a NaN/garbage
# logit row). Written 10-21 09:00 UTC when the v4-5.6 served-line gate read HumanEval 0.494 with 85 / 164 garbage
# generations -- to tell weights / library / pool size / spec apart. Holds the GPU's flock for the run.
# usage: logs/t1-probe.sh GPU PORT LIB TQF TAG [extra serve_batched.py args, e.g. --no-spec --num-blocks 4707]
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
GPU=$1; PORT=$2; LIB=$3; TQF=$4; TAG=$5; shift 5
MD=/mnt/nvme/models/Qwen3.8-27B-FP8
cd $Q
unset $(env | grep -oE '^TQ_[A-Z0-9_]+')
export CUDA_VISIBLE_DEVICES=$GPU
exec 9>/tmp/exl3diff-gpu$GPU.lock; flock 9
echo "[t1-probe $TAG] start $(date -u +%T) GPU $GPU lib $(md5sum < $LIB | cut -c1-12) tqf $(basename $TQF) args: $*"
rm -rf runs/evalplus-t1probe-$TAG
nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks 4500 --timeout 1800 --model-name qwen3.8-27b \
  --tqf $TQF --model-dir $MD --lib $LIB "$@" > logs/t1-probe-$TAG-server.log 2>&1 &
SPID=""; for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { SPID=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); break; }; done
[ -n "$SPID" ] || { tail -5 logs/t1-probe-$TAG-server.log; echo "[t1-probe $TAG] FAIL: server"; exit 1; }
echo "   server up, pid $SPID, $(grep -oE '[0-9]+ pooled tokens' logs/t1-probe-$TAG-server.log), $(grep -m1 -oE 'spec[^,;]*' logs/t1-probe-$TAG-server.log | head -1)"
rc=0
for ds in ${T1_DATASETS:-humaneval}; do
  ./eval/evalplus.sh http://127.0.0.1:$PORT qwen3.8-27b t1probe-$TAG $ds > logs/t1-probe-$TAG-$ds.log 2>&1 || rc=$?
done
kill $SPID; for i in $(seq 1 30); do sleep 2; kill -0 $SPID 2>/dev/null || break; done
for ds in ${T1_DATASETS:-humaneval}; do $PY - $TAG $ds <<'PY'
import json, sys, glob, re
T, ds = sys.argv[1], sys.argv[2]; d = f"/mnt/nvme/qwen256k/runs/evalplus-t1probe-{T}/{ds}"
raw = [json.loads(l) for l in open(glob.glob(f"{d}/*.raw.jsonl")[0])]
g = [r for r in raw if "!!!!!!!!" in r["solution"]]
log = open(f"/mnt/nvme/qwen256k/logs/t1-probe-{T}-{ds}.log").read()
print(f"   [{T}] {ds} pass@1 {re.findall(r'pass@1:\s*([\d.]+)', log)}; garbage generations {len(g)} / {len(raw)}: {[r['task_id'].split('/')[1] for r in g][:30]}")
PY
done
echo "   server log error lines: $(grep -ciE 'xid|error|traceback|nan' logs/t1-probe-$TAG-server.log)"
echo "[t1-probe $TAG] done $(date -u +%T) rc $rc"
