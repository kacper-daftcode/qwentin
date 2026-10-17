#!/usr/bin/env bash
# T1 of the quality protocol (docs/QUALITY.md): EvalPlus HumanEval+ / MBPP+ pass@1, greedy, against an OpenAI-compatible
# endpoint -- tools/serve_batched.py (the served stack), tools/serve_openai.py, vLLM (the FP8 reference row), ...
#
# Usage: tools/evalplus.sh <base-url> <served-model-name> <out-dir> [humaneval|mbpp]
#   EVALPLUS=/path/to/venv/bin/evalplus.evaluate   (default: `evalplus.evaluate` on PATH; `pip install evalplus`)
#   OPENAI_API_KEY                                 (default "none"; set it to the server's --api-key when one is configured)
# Results land in <out-dir>/<dataset>.log (the two `pass@1:` lines at the end) and <out-dir>/<dataset>/ (samples, eval json).
# The published rows (docs/evidence/evalplus/): 164 HumanEval / 378 MBPP tasks, temperature 0, n = 1, thinking off on the
# server side (serve_batched.py defaults; the request carries no enable_thinking), ~40 min per dataset pair on one 5090.
#
# Vendored from eval/evalplus.sh of the research project (kacper-daftcode/qwentin-internal); the venv path became $EVALPLUS,
# the result root an argument.
set -euo pipefail
BASE_URL=${1:?base url, e.g. http://127.0.0.1:8100}
MODEL=${2:?served model name (the --model-name of the server)}
OUT=${3:?output directory}
DATASET=${4:-humaneval}
EVALPLUS=${EVALPLUS:-evalplus.evaluate}
mkdir -p "$OUT"
export OPENAI_API_KEY=${OPENAI_API_KEY:-none}
# --greedy: temperature 0, n=1. The endpoint must accept /v1/chat/completions.
"$EVALPLUS" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --backend openai \
  --base-url "$BASE_URL/v1" \
  --greedy \
  --root "$OUT" 2>&1 | tee "$OUT/$DATASET.log"
echo "results: $OUT ($(grep -E '^pass@1' "$OUT/$DATASET.log" | tr '\n' ' '))"
