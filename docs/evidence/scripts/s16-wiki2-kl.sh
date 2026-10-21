#!/usr/bin/env bash
# Session 16 thread C (5): the public yardstick -- exllamav3's eval/model_diff.py on wikitext-2 test (100 rows x 2048), each
# EXL3 model (A) vs the bf16 original (B): KL(A,B), KL(B,A), perplexities, label-in-top-K, top-K agreement -- the numbers
# HF model cards quote (mratsim's GLM-4.7, Chungulus's Qwen3.8-27B-EXL3 ...). The same tool and dataset for every row, so
# the table is comparable across the community (our code-token yardstick is not). GPU 1 under its flock; the shared
# exllamav3 checkout is used read-only (its venv: /root/work/venv has exllamav3 installed from it).
# usage: nohup logs/s16-wiki2-kl.sh > logs/s16-wiki2-kl.log 2>&1 < /dev/null &   (~10 min per model)
set -u
Q=/mnt/nvme/qwen256k; PY=/root/work/venv/bin/python; X=/root/work/exllamav3
cd $Q; mkdir -p runs/session16/wiki2
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=0
exec 9>/tmp/exl3diff-gpu1.lock; flock 9
REF=/mnt/nvme/models/Qwen3.8-27B
for m in opt5.75-bf16src t45v4 t45-s425-bf16src opt5.5-bf16src opt5.5-R2 opt5.6-v4 opt6.0-bf16src opt5.0-bf16src opt4.0-bf16src; do
  M=/mnt/nvme/models/Qwen3.8-27B-exl3-$m
  [ -f runs/session16/wiki2/$m.log ] && { echo "skip $m"; continue; }
  echo "== $m   $(date -u +%T)"
  (cd $X && $PY eval/model_diff.py -ma $M -mb $REF -r 100 -l 2048 -d 0 > $Q/runs/session16/wiki2/$m.log 2>&1); rc=$?
  grep -E "perplexity|KL|top-K|K = [1-5]:" runs/session16/wiki2/$m.log | head -20 | sed 's/^/   /'
  echo "   exit $rc $(date -u +%T)"
done
echo "[wiki2] done $(date -u +%T)"
