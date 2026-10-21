#!/usr/bin/env python
"""Minimal repro of the T1 garbage (10-21): EvalPlus's exact HumanEval chat prompt for a few tasks, greedy, max_tokens 768,
one request at a time against a batched server; reports whether the answer degenerates into "!!!!" (token 0 = a garbage
logit row) and where. Run with the eval venv (it has evalplus + openai): /mnt/nvme/qwen256k/venv-eval/bin/python.
usage: t1_repro.py --base http://127.0.0.1:8100 [--tasks 10,32,45,53] [--max-tokens 768] [--repeat 1] [--pad N]
  --pad N prepends N copies of a neutral line to the instruction (longer prompt, same task) to test prompt-length triggers.
"""
import argparse, time, json
import openai
from evalplus.data import get_human_eval_plus

PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8100")
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--tasks", default="10,32,45,53,0,2")
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--pad", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
cl = openai.OpenAI(api_key="none", base_url=a.base + "/v1")
ds = get_human_eval_plus()
res = []
for rep in range(a.repeat):
    for t in a.tasks.split(","):
        task = ds[f"HumanEval/{t}"]
        pad = ("# (padding line to lengthen the prompt without changing the task)\n" * a.pad) if a.pad else ""
        msg = pad + PREFIX + f"\n```python\n{task['prompt'].strip()}\n```"
        t0 = time.time()
        r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": msg}],
                                       max_tokens=a.max_tokens, temperature=0.0, n=1)
        dt = time.time() - t0
        txt = r.choices[0].message.content or ""
        i = txt.find("!!!!!!!!")
        u = r.usage
        row = {"task": t, "rep": rep, "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None,
               "garbage_at_char": i, "chars": len(txt), "finish": r.choices[0].finish_reason, "s": round(dt, 2), "text": txt}
        res.append(row)
        print(f"[repro] task {t:>3} rep {rep}: prompt {row['prompt_tokens']} tok, out {row['completion_tokens']} tok / {len(txt)} chars, "
              f"finish {row['finish']}, {dt:.1f}s -> {'GARBAGE at char ' + str(i) if i >= 0 else 'clean'}", flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
n = sum(r["garbage_at_char"] >= 0 for r in res)
print(f"[repro] garbage {n} / {len(res)}")
