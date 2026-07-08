#!/usr/bin/env python3
"""Cold-prefill + prefix-cache smoke against a running serve_openai.py.

Usage: python3 tools/serve_smoke.py [port] [doc_chars]
Turn 1 = cold long prompt; turn 2 = same conversation + one short user turn
(the prefix-cache anchor path should prefill only the suffix)."""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
DOC_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
src = open("/root/workspace/qwentin/src/forward_qwen.cu").read()
doc = src[:DOC_CHARS]

def chat(messages, max_tokens=48):
    body = json.dumps({"messages": messages, "temperature": 0.0,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    x = out.get("x_qwentin") or out.get("x_turbollama") or {}
    txt = out["choices"][0]["message"]["content"]
    return wall, x, txt

m1 = [{"role": "system", "content": "Jestes pomocnym asystentem. Dokument:\n" + doc},
      {"role": "user", "content": "Odpowiedz jednym zdaniem: co robi plik z dokumentu?"}]
w, x, txt = chat(m1)
print(f"TURN1 (cold): wall={w:.2f}s x={json.dumps(x)}")
print(f"  text: {txt[:140]!r}")

m2 = m1 + [{"role": "assistant", "content": txt},
           {"role": "user", "content": "A jaki sprzet jest wymagany? Jedno zdanie."}]
w, x, txt = chat(m2)
print(f"TURN2 (cache): wall={w:.2f}s x={json.dumps(x)}")
print(f"  text: {txt[:140]!r}")
