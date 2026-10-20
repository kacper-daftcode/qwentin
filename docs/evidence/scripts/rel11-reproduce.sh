#!/usr/bin/env bash
# v0.11 release check: bench/reproduce.sh end to end on the release worktree qwentin-rel11 (= main dd6d21e + the release
# docs; its own build in build-qwen-repro/), the v2 file, GPU 1 (flock), port 8148. Run 1: the script's defaults (corpus =
# the repository itself). Run 2: the same library with CORPUS = the vLLM checkout of the README table.
# usage: nohup logs/rel11-reproduce.sh > logs/rel11-reproduce.log 2>&1 &        (~12 min)
set -u
W=/mnt/nvme/qwen256k/qwentin-rel11
cd $W
export TQF=/mnt/nvme/models/qwen3_8-27b-exl3-opt5.75-mtp4-v2.tqf MODEL_DIR=/mnt/nvme/models/Qwen3.8-27B-FP8 \
       PYTHON=/root/work/venv/bin/python GPU=1 PORT=8148 OUT_DIR=/mnt/nvme/qwen256k/runs/rel11
mkdir -p $OUT_DIR
echo "[rel11-reproduce] start $(date -u +%T); worktree $(git log --oneline -1 | cut -c1-60)"
flock /tmp/exl3diff-gpu1.lock bench/reproduce.sh --tag rel11-own; rc1=$?
echo "[rel11-reproduce] run 1 (corpus = this repository) rc $rc1 $(date -u +%T)"
LIB=$W/build-qwen-repro/libforward_qwen.so CORPUS=/mnt/nvme/vllm/vllm flock /tmp/exl3diff-gpu1.lock bench/reproduce.sh --no-build --no-check --tag rel11-vllm; rc2=$?
echo "[rel11-reproduce] run 2 (corpus = the vLLM checkout) rc $rc2 $(date -u +%T)"
