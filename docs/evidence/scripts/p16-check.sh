#!/usr/bin/env bash
# 16 GiB re-check on the LANDED served library (10-17, main 14be568 after cand-1018, the prefill GEMM levers on): same protocol as
# logs/p16b.sh (B2 thread 10-16: 400 blocks -> 16 006 MiB after a full run, needle 3/3 at 49K, 183 tok/s @32K). Item 5: the 16 GiB variant on the served stack (t45-s425 -mtp4 + trellis-4 + --spec, 2 slots so the pinned parking
# ring is allocated): process MiB at start / after park_probe 16K (ring + a host save) / after a needle at the max
# context, for --num-blocks 440 / 400 / 360; spec_speed at 32K and the max; needle at the max. Budget = the 10-14
# convention (process <= ~15 750 MiB on a 16 384 MiB card). GPU 2, port 8133.
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
T4=/mnt/nvme/models/qwen3_8-27b-exl3-t45-s425-mtp4.tqf
PORT=8134
cd $Q; exec 9>/tmp/exl3diff-gpu1.lock; flock 9
export CUDA_VISIBLE_DEVICES=1 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 TQ_EMBED_HOST=1 TQ_NO_MTP=0 \
  TQ_KV_V4=1 TQ_KV_T4=1 TQ_KV_T4_CODEBOOK=$CB TQ_PDL=1 PROBE_PORT=$PORT
start_server() {   # slots blocks log extra...
  local slots=$1 blocks=$2 log=$3; shift 3
  nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots $slots --num-blocks $blocks --kv-v4 --kv-t4 --spec --park-ram-gb 16 \
    --timeout 3600 --model-name qwen3.8-27b --tqf $T4 --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  for i in $(seq 1 60); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { echo "   server up after $((i * 5))s"; return 0; }
    pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" > /dev/null || { echo "   server died"; tail -3 $log; return 1; }; done
  echo "   server did not come up"; return 1
}
stop_server() { local pid; pid=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); [ -n "$pid" ] && { kill "$pid"; sleep 8; }; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p="$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1)" -F', ' '$1==p {print $2}'; }
needle() { $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths $1 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/needle-$2.json > logs/needle-$2.log 2>&1; echo "   needle $1 exit $? $(date -u +%T): $(grep -E '^ +[0-9]+k +[0-9]+/[0-9]+' logs/needle-$2.log | tr -s ' ' | tr '\n' ';')"; grep "prompt=" logs/needle-$2.log | cut -c1-120 | tail -3; }
echo "[p16b] start $(date -u +%T)"
for B in 400; do
  echo "== t45-s425 -mtp4 + trellis + spec, 2 slots, $B blocks = $((B * 128)) tokens $(date -u +%T)"
  start_server 2 $B logs/p16b-server-$B.log || continue
  echo "   process $(gpu_mib) MiB at start"
  $PY eval/park_probe.py 16000 > logs/park-p16b-$B.log 2>&1; echo "   park_probe exit $? ($(grep -c PASS logs/park-p16b-$B.log)/3 PASS)"
  echo "   process $(gpu_mib) MiB after the park probe (ring allocated)"
  MAXCTX=$(( B * 128 - 2048 ))
  needle $MAXCTX p16b-$B
  echo "   process $(gpu_mib) MiB after the needle at $MAXCTX"
  if [ $B = 400 ]; then
    $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on,off --max-tokens 300 --out runs/p16b-speed-32k.json > logs/p16b-speed-32k.log 2>&1; grep -F "[speed] ctx" logs/p16b-speed-32k.log | cut -c1-160
    $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx $MAXCTX --n 1 --modes on,off --max-tokens 300 --out runs/p16b-speed-max.json > logs/p16b-speed-max.log 2>&1; grep -F "[speed] ctx" logs/p16b-speed-max.log | cut -c1-160
    echo "   process $(gpu_mib) MiB after spec_speed"
  fi
  stop_server
done
echo "[p16b] done $(date -u +%T)"
