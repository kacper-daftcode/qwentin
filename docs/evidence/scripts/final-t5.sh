#!/usr/bin/env bash
# T5 mix on THE LANDED SERVED LINE (10-17 01:30 UTC, main 11f5879, build-qwen/ c9d5269c; GPU 1, port 8135): the same loadgen as
# logs/b2eng-t5v2.sh (8 sessions, mix 32K/96K/160K/256K, concurrency 3, 3 turns, +3000 tokens per turn, 400 out) in three lines:
#   (e) pure defaults (--max-slots 12 --num-blocks 4100: --kv-t4 --spec on)                      -> tag e1
#   (f) defaults + --decode-steps-per-wave 1 (decision (a))                                       -> tag f1
#   (g) (f) + the opt-in levers TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 + --dstate fp16sr --num-blocks 4500 (decisions b, c, g) -> tag g1
# Reference (10-16, b2eng-t5v2): c3 = defaults 906 s wall, d3 = + decode-steps-per-wave 1 (39.7 / 33.7 / 33.9 tok/s per client under prefill load).
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python
LIB=$Q/qwentin/build-qwen/libforward_qwen.so
CB=$Q/analysis/kv_trellis_cuda/codebooks/lloyd_kv12a.f32
PORT=8135
cd $Q; exec 9>/tmp/exl3diff-gpu1.lock; flock 9
export CUDA_VISIBLE_DEVICES=1 TQ_CTX=262144 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_PAGED_ATTN_MMA=1 TQ_KV_T4_CODEBOOK=$CB
start_server() {   # tqf blocks log extra...
  local tqf=$1 blocks=$2 log=$3; shift 3
  nohup $PY -u qwentin/tools/serve_batched.py --port $PORT --max-slots 12 --num-blocks $blocks --timeout 3600 \
    --model-name qwen3.8-27b --tqf $tqf --model-dir /mnt/nvme/models/Qwen3.8-27B-FP8 --lib $LIB "$@" > $log 2>&1 &
  for i in $(seq 1 120); do sleep 5; curl -s -m 2 http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { echo "   server up after $((i * 5))s"; return 0; }
    pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" > /dev/null || { echo "   server died"; tail -3 $log; return 1; }; done
  echo "   server did not come up"; return 1
}
stop_server() { local pid; pid=$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1); [ -n "$pid" ] && { kill "$pid"; sleep 8; }; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | rg "^2,"; }
gpu_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p="$(pgrep -f "serve_batched.py --port $PORT --max-slots [0-9]" | head -1)" -F', ' '$1==p {print $2}'; }
poller() {   # $1 = out jsonl; runs until killed
  while true; do curl -s -m 2 http://127.0.0.1:$PORT/health | $PY -c "import sys,json,time; d=json.load(sys.stdin); d['t']=time.time(); print(json.dumps(d))" >> $1 2>/dev/null; sleep 2; done
}
run_t5() {   # tag
  local tag=$1
  echo "   process $(gpu_mib) MiB before the load"
  rm -f runs/b2eng/health-$tag.jsonl
  poller runs/b2eng/health-$tag.jsonl & local PP=$!
  $PY eval/loadgen.py --base-url http://127.0.0.1:$PORT --model qwen3.8-27b --sessions 8 --concurrency 3 \
    --context-mix 32768:0.3,98304:0.3,163840:0.25,262144:0.15 --turns 3 --append-tokens 3000 --max-tokens 400 --gap 10 --stagger 30 \
    --tokenizer /mnt/nvme/models/Qwen3.8-27B-FP8 --chat-kwargs '{"enable_thinking": false}' --seed 1 \
    --out runs/load-b2eng-$tag.jsonl --verbose > logs/loadgen-b2eng-$tag.log 2>&1
  echo "   loadgen exit $? $(date -u +%T)"
  kill $PP 2>/dev/null; wait $PP 2>/dev/null
  curl -s http://127.0.0.1:$PORT/health | $PY -c "import sys,json; d=json.load(sys.stdin); print('   health:', {k: d[k] for k in ('prefix_cache_hits','prefix_cache_tokens','host_parked','host_hits','host_saves','host_save_s','host_restore_s','prefilled_tokens','decoded_tokens','steps','rounds','spec_tokens','tokens_per_round','spec_fallbacks') if k in d})"
  echo "   process $(gpu_mib) MiB after the load"
  rg "wall|aggregate|errors" logs/loadgen-b2eng-$tag.log | head -3
}
T4M=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4.tqf
echo "== (e) pure defaults, 4100 blocks $(date -u +%T)"
[ -f runs/load-b2eng-e1.summary.json ] && echo "   (e1) done already" || { start_server $T4M 4100 logs/final-t5-server-e1.log && { run_t5 e1; stop_server; } }
echo "== (f) defaults + --decode-steps-per-wave 1 $(date -u +%T)"
[ -f runs/load-b2eng-f1.summary.json ] && echo "   (f1) done already" || { start_server $T4M 4100 logs/final-t5-server-f1.log --decode-steps-per-wave 1 && { run_t5 f1; stop_server; } }
echo "== (g) (f) + TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1 + --dstate fp16sr --num-blocks 4500 $(date -u +%T)"
export TQ_T4_ATTN_F16=1 TQ_GRAPH=1 TQ_GEMM_FORK=1
[ -f runs/load-b2eng-g1.summary.json ] && echo "   (g1) done already" || { start_server $T4M 4500 logs/final-t5-server-g1.log --decode-steps-per-wave 1 --dstate fp16sr && { run_t5 g1; stop_server; } }
unset TQ_T4_ATTN_F16 TQ_GRAPH TQ_GEMM_FORK
$PY analysis/t5_report.py e1 f1 g1 2>&1 | tail -40
echo "[t5] done $(date -u +%T)"
