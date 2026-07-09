#!/usr/bin/env python3
"""Mini needle-in-a-haystack gate for the FP8 KV cache (TQ_KV_FP8).

Builds a ~12k-token haystack from the in-repo engine source (src/forward_qwen.cu),
injects a passphrase sentence ("Sekretny kod projektu to ZETA-<n>.") at a given
token depth, appends a question at the end, and greedy-decodes the answer
through the DENSE per-token path (qwn_decode reads the live prefix KV cache, so
with TQ_KV_FP8=1 every prefix row the answer attends to is E4M3). One process =
one KV mode; run twice (TQ_KV_FP8=0/1) and compare the per-depth columns.

Env: TQ_KV_FP8 (engine flag, also used for labels), TQ_CTX (>= ~12k),
     DEPTHS (default "1000,4000,8000,11000"), GEN (answer tokens, default 24),
     TQ_MODEL_TQF / TQ_LIB / TQ_MODEL_DIR (.tqf path, engine .so, HF checkpoint
     dir for the tokenizer).

Usage:
    CUDA_VISIBLE_DEVICES=7 TQ_CTX=16384 python3 tools/needle_check.py
    CUDA_VISIBLE_DEVICES=7 TQ_CTX=16384 TQ_KV_FP8=1 python3 tools/needle_check.py
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TQF = os.environ.get("TQ_MODEL_TQF", "/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
LIB = os.environ.get("TQ_LIB", os.path.join(REPO, "build-qwen", "libforward_qwen.so"))
MODEL_DIR = os.environ.get("TQ_MODEL_DIR", "/workspace/models/Qwen3.6-27B")
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "1000,4000,8000,11000").split(",")]
GEN = int(os.environ.get("GEN", "24"))
HAYSTACK_TOKENS = int(os.environ.get("HAYSTACK_TOKENS", "11500"))
CORPUS = os.environ.get("CORPUS",
    os.path.join(REPO, "src", "forward_qwen.cu")).split(",")
# long-context haystacks (>= ~12k tokens): widen the corpus with repo docs +
# tools sources until HAYSTACK_TOKENS is covered (deterministic file order);
# the engine source itself (~210k tokens) unlocks 240k+ haystacks
_EXTRA = (sorted(__import__("glob").glob(os.path.join(REPO, "tools", "*.py"))) +
          [os.path.join(REPO, "src", "forward_qwen.cu")])
# distinct code per depth so a stale/wrong retrieval cannot pass by accident
CODES = {1000: 471, 4000: 832, 8000: 159, 11000: 604}

QUESTION = ("\n\nPytanie: Jaki jest sekretny kod projektu wspomniany wczesniej "
            "w tekscie? Odpowiedz dokladnie jednym kodem.\nOdpowiedz: "
            "Sekretny kod projektu to")

mode = "fp8" if os.environ.get("TQ_KV_FP8", "0") not in ("", "0") else "fp32"
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
_texts = [open(f).read() for f in CORPUS if os.path.exists(f)]
corpus_ids = tok("\n\n".join(_texts), add_special_tokens=False).input_ids
for f in _EXTRA:
    if len(corpus_ids) >= HAYSTACK_TOKENS + 64:
        break
    if f in CORPUS or not os.path.exists(f):
        continue
    _texts.append(open(f).read())
    corpus_ids = tok("\n\n".join(_texts), add_special_tokens=False).input_ids
assert len(corpus_ids) >= HAYSTACK_TOKENS, f"corpus too short: {len(corpus_ids)}"

L = load_lib(LIB)
ck(L.qwn_init(TQF.encode()), "init")
e = Eng(L)
max_seq = L.qwn_max_seq()
print(f"mode={mode} lib={os.path.basename(LIB)} max_seq={max_seq} "
      f"haystack={HAYSTACK_TOKENS} depths={DEPTHS}", flush=True)

results = {}
try:
    for d in DEPTHS:
        code = CODES.get(d, 700 + d % 300)
        needle = (f"\n\nUWAGA, wazna informacja: Sekretny kod projektu to "
                  f"ZETA-{code}. Zapamietaj dokladnie ten kod.\n\n")
        text = (tok.decode(corpus_ids[:d]) + needle +
                tok.decode(corpus_ids[d:HAYSTACK_TOKENS]) + QUESTION)
        ids = tok(text, add_special_tokens=False).input_ids
        assert len(ids) + GEN + 2 < max_seq, f"prompt {len(ids)} too long for ctx {max_seq}"
        P = len(ids)
        cur = prefill(e, ids, P - 1, build_trunk=False)  # greedy argmax at P-1
        out, p = [], P - 1
        for _ in range(GEN):
            out.append(cur)
            cur = e.decode(cur, p + 1)
            p += 1
        ans = tok.decode(out)
        m = re.search(r"ZETA[\s-]*(\d+)", ans)
        got = m.group(1) if m else None
        ok = (got == str(code))
        results[d] = (ok, code, ans.splitlines()[0] if ans else "")
        print(f"  depth {d:6d}: expect ZETA-{code}  got={got}  "
              f"{'RECOVERED' if ok else 'MISS'}   answer={ans[:60]!r}", flush=True)
finally:
    L.qwn_free()

n_ok = sum(1 for ok, _, _ in results.values() if ok)
print(f"SUMMARY mode={mode}: {n_ok}/{len(DEPTHS)} needles recovered "
      f"({' '.join(str(d) + ':' + ('OK' if results[d][0] else 'MISS') for d in DEPTHS)})",
      flush=True)
