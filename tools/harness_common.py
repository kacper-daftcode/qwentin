"""Shared helpers for the verification tools that talk to a running server (code_needle.py, spec_speed.py,
spec_server_ab.py). Vendored from the qwen256k research harness (eval/common.py) so the tools run from this repo alone.

Model-agnostic: everything talks to an OpenAI-compatible /v1/chat/completions endpoint (qwentin's serve_batched.py or
serve_openai.py, vLLM, ...). Token accounting uses the served model's tokenizer.json when given (--tokenizer), else a
chars/token estimate.

The "code corpus" is a directory tree of source files that the tools concatenate into repository-snapshot prompts
('### FILE: <path>' headers). Default: QWENTIN_CORPUS if set, else this repository itself (src/ + tools/ hold ~1 MB of
code, enough for 256K-token prompts). The published numbers were measured with a vLLM checkout as the corpus
(/mnt/nvme/vllm/vllm on the measurement box); any large code tree gives comparable pass/fail matrices.
"""

from __future__ import annotations

import glob
import json
import os
import random
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_CORPUS = os.environ.get("QWENTIN_CORPUS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_EXTS = (".py", ".cu", ".cuh", ".h", ".cpp", ".rs", ".ts", ".js", ".go", ".md")

AGENT_SYSTEM_PROMPT = (
    "You are a coding agent operating inside a software repository. You are "
    "given a snapshot of repository files (each prefixed with '### FILE: <path>') "
    "followed by tool outputs and user requests. Answer precisely and concisely; "
    "when asked for a value, answer with the value only."
)


class Tok:
    """Token counter/truncator backed by a HF tokenizer.json, with a fallback."""

    def __init__(self, path: str | None = None, chars_per_token: float = 3.6):
        self.tk = None
        self.cpt = chars_per_token
        if path:
            p = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
            if os.path.exists(p):
                from tokenizers import Tokenizer

                self.tk = Tokenizer.from_file(p)
                self.source = p
        if self.tk is None:
            self.source = f"estimate ({chars_per_token} chars/token)"

    def count(self, text: str) -> int:
        if self.tk is None:
            return int(len(text) / self.cpt)
        return len(self.tk.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, n_tokens: int) -> str:
        if self.tk is None:
            return text[: int(n_tokens * self.cpt)]
        ids = self.tk.encode(text, add_special_tokens=False).ids
        if len(ids) <= n_tokens:
            return text
        return self.tk.decode(ids[:n_tokens])


def load_corpus(root: str = DEFAULT_CORPUS, exts=CODE_EXTS, min_bytes: int = 512):
    """Deterministically ordered list of (relpath, text) source files."""
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    files = sorted(set(files))
    out = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        if len(text) >= min_bytes:
            out.append((os.path.relpath(f, root), text))
    if not out:
        raise SystemExit(f"no corpus files under {root}")
    return out


def build_context(corpus, target_tokens: int, seed: int, tok: Tok) -> str:
    """Concatenate shuffled files with '### FILE:' headers up to target_tokens."""
    rng = random.Random(seed)
    order = list(range(len(corpus)))
    rng.shuffle(order)
    parts, total = [], 0
    for i in order:
        rel, text = corpus[i]
        chunk = f"### FILE: {rel}\n{text}\n\n"
        n = tok.count(chunk)
        if total + n > target_tokens:
            remaining = target_tokens - total
            if remaining > 64:
                parts.append(tok.truncate(chunk, remaining))
                total += remaining
            break
        parts.append(chunk)
        total += n
    if total < target_tokens * 0.9:
        # Corpus exhausted: cycle again with a different offset.
        rng2 = random.Random(seed + 7919)
        while total < target_tokens:
            rel, text = corpus[rng2.randrange(len(corpus))]
            chunk = f"### FILE: {rel}\n{text}\n\n"
            n = tok.count(chunk)
            if total + n > target_tokens:
                parts.append(tok.truncate(chunk, target_tokens - total))
                break
            parts.append(chunk)
            total += n
    return "".join(parts)


@dataclass
class ChatResult:
    ok: bool
    status: int
    text: str = ""
    ttft: float = 0.0  # seconds to first content token
    e2e: float = 0.0  # seconds total
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunks: int = 0
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def decode_tps(self) -> float:
        n = self.completion_tokens if self.completion_tokens else self.chunks
        dt = self.e2e - self.ttft
        return (n - 1) / dt if n and n > 1 and dt > 0 else 0.0


async def chat_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 0.0,
    chat_kwargs: dict | None = None,
    extra_body: dict | None = None,
    timeout: float = 3600.0,
) -> ChatResult:
    """Stream one chat completion, measuring TTFT and decode rate."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_kwargs:
        body["chat_template_kwargs"] = chat_kwargs
    if extra_body:
        body.update(extra_body)
    res = ChatResult(ok=False, status=0)
    t0 = time.perf_counter()
    text_parts: list[str] = []
    try:
        async with client.stream(
            "POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=body, timeout=timeout
        ) as r:
            res.status = r.status_code
            if r.status_code != 200:
                res.error = (await r.aread()).decode(errors="ignore")[:500]
                res.e2e = time.perf_counter() - t0
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    # vLLM streams engine-side failures as an error object.
                    err = obj["error"]
                    res.error = (err.get("message") if isinstance(err, dict) else str(err))[:500]
                    res.e2e = time.perf_counter() - t0
                    return res
                usage = obj.get("usage")
                if usage:
                    res.prompt_tokens = usage.get("prompt_tokens", res.prompt_tokens)
                    res.completion_tokens = usage.get("completion_tokens", res.completion_tokens)
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if content:
                        if res.ttft == 0.0:
                            res.ttft = time.perf_counter() - t0
                        res.chunks += 1
                        if delta.get("content"):
                            text_parts.append(content)
                for k in ("x_qwentin",):
                    if k in obj:
                        res.extra[k] = obj[k]
        res.e2e = time.perf_counter() - t0
        res.text = "".join(text_parts)
        if res.chunks == 0 and not res.completion_tokens:
            res.error = "empty completion (no content deltas, 0 completion tokens)"
            return res
        res.ok = True
    except Exception as e:  # noqa: BLE001
        res.e2e = time.perf_counter() - t0
        res.error = f"{type(e).__name__}: {e}"[:500]
    return res


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[k]


def fmt_tokens(n: int) -> str:
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)
