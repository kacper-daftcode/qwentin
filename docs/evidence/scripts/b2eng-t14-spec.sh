#!/usr/bin/env bash
# Task 14 (d) on the trellis stack (build-qwen-b2t), GPU 3 under the lock, servers killed by PID.
#  p24: opt-5.75 + trellis-4, 1 slot, 2100 blocks = 268.8K tokens (23 210 MiB; the 10-14 p24 emulation ran at 23 820):
#       decode N=1 @32K and @256K (paged_decode_ab.py), needle 32K / 128K / 256K; then --spec (-mtp4, 1900 blocks =
#       243K) : spec_speed N=1 @32K and @240K, needle 240K.
#  p16: t45-s425 + trellis-4, 1 slot, 540 blocks = 69K tokens (~15 700 MiB; 10-14: 420 blocks = 54K at 15 614):
#       decode N=1 @32K / @64K, needle 32K / 48K / 64K; --spec with 500 blocks (64K): spec_speed N=1 @32K / @60K.
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin-b2/build-qwen-b2t/libforward_qwen.so
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8112
cd $Q; exec 9>/tmp/exl3diff-gpu3.lock; flock 9
export CUDA_VISIBLE_DEVICES=3 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 TQ_EMBED_HOST=1 TQ_NO_MTP=0 \
  TQ_KV_V4=1 TQ_KV_T4=1 TQ_KV_T4_CODEBOOK=$CB TQ_PDL=1 PROBE_PORT=$PORT
start_server() {   # tqf slots blocks log extra...
  local tqf=$1 slots=$2 blocks=$3 log=$4; shift 4
  nohup $PY -u qwentin-b2/tools/serve_batched.py --port $PORT --max-slots $slots --num-blocks $blocks --kv-v4 --kv-t4 \
    --timeout 3600 --model-name qwen3.8-27b --tqf $tqf --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { echo "   server up after $((i * 5))s"; return 0; }
    pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" > /dev/null || { echo "   server died"; tail -3 $log; return 1; }; done
  echo "   server did not come up"; return 1
}
stop_server() { local pid; pid=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); [ -n "$pid" ] && { kill "$pid"; sleep 8; }; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | rg "^3,"; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p="$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1)" -F', ' '$1==p {print $2}'; }
needle() { $PY eval/code_needle.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --lengths $1 --depths 0.95,0.5,0.05 --decoys 3 \
  --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --out runs/needle-$2.json > logs/needle-$2.log 2>&1; echo "   needle exit $? $(date -u +%T)"; tail -n $(( $(echo $1 | tr ',' '\n' | wc -l) + 1 )) logs/needle-$2.log; }

echo "== p24 + spec: opt-5.75 -mtp4 + trellis, 1 slot, 2000 blocks $(date -u +%T)"
T4=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
start_server $T4 1 2000 logs/b2eng-t14-p24s-server.log --spec && {
  echo "   process $(gpu_mib) MiB at start"
  $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on,off --max-tokens 300 --out runs/b2eng/t14-p24s-speed-32k.json > logs/b2eng-t14-p24s-speed-32k.log 2>&1; rg "\[speed\] ctx" logs/b2eng-t14-p24s-speed-32k.log | cut -c1-160
  $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 240000 --n 1 --modes on,off --max-tokens 300 --out runs/b2eng/t14-p24s-speed-240k.json > logs/b2eng-t14-p24s-speed-240k.log 2>&1; rg "\[speed\] ctx" logs/b2eng-t14-p24s-speed-240k.log | cut -c1-160
  needle 32768,240000 b2eng-t14-p24s
  echo "   process $(gpu_mib) MiB after"
  stop_server
}
echo "== p16 + spec: t45-s425 -mtp4 + trellis, 1 slot, 500 blocks $(date -u +%T)"
T4=/mnt/nvme/models/qwen3_8-27b-exl3-t45-s425-mtp4.tqf
start_server $T4 1 500 logs/b2eng-t14-p16s-server.log --spec && {
  echo "   process $(gpu_mib) MiB at start"
  $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 32768 --n 1 --modes on,off --max-tokens 300 --out runs/b2eng/t14-p16s-speed-32k.json > logs/b2eng-t14-p16s-speed-32k.log 2>&1; rg "\[speed\] ctx" logs/b2eng-t14-p16s-speed-32k.log | cut -c1-160
  $PY analysis/spec_speed.py --base http://127.0.0.1:$PORT --ctx 60000 --n 1 --modes on,off --max-tokens 300 --out runs/b2eng/t14-p16s-speed-60k.json > logs/b2eng-t14-p16s-speed-60k.log 2>&1; rg "\[speed\] ctx" logs/b2eng-t14-p16s-speed-60k.log | cut -c1-160
  needle 32768,60000 b2eng-t14-p16s
  echo "   process $(gpu_mib) MiB after"
  stop_server
}
echo "[t14s] done $(date -u +%T)"
