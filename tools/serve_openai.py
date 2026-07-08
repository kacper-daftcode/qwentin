#!/usr/bin/env python3
"""OpenAI-compatible API server for the qwentin FP6 spec-decode engine.

Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /health.
Features: SSE streaming, temperature (lossless tree speculative sampling,
TQ_TEMP kernel path; temp=0 = bit-exact greedy), per-request seed, stop
strings + EOS, max_tokens, usage accounting + speculative stats
(x_qwentin: accept_len, rounds, tok/s).

Single-stream engine: requests are serialized with a lock (queue waits).
top_p: only 1.0 supported (v1 sampler); other values are clamped with a
warning field. n>1, logprobs, tools: unsupported -> 400.

Prefix/KV cache between requests (--prefix-cache, default ON): every prefill
drops a DeltaNet snapshot (slot 0) at the last 16-aligned chunk boundary
inside the prompt (the "anchor"); the positional KV/trunk state below the
anchor stays in place. A new request whose token ids match the previous
prompt through the anchor restores the snapshot and prefills ONLY from the
anchor on -- through the exact same 16-token chunking a fresh prefill would
use, so the cache path is BIT-IDENTICAL to full reset+prefill (gate-tested;
that is also why the anchor must be 16-aligned and why reuse stops at the
previous PROMPT rather than at generated tokens: generation writes its state
through different verify seams, which is equally-valid-forward but not
bit-equal). Anything else falls back to a full reset+prefill.
--prefix-cache-live (default OFF, eps-equivalent NOT token-identical):
additionally continue straight from the live committed state when the new
prompt extends prompt+generation, prefilling only the strict suffix.
x_qwentin reports prefix/reused_tokens/prefilled_tokens per request.

Run:  CUDA_VISIBLE_DEVICES=6 python3 tools/serve_openai.py --port 8000
Test: curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user",
      "content":"Hej!"}],"temperature":0.8,"max_tokens":64,"stream":true}'
"""
from __future__ import annotations
import argparse, ctypes, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, Eng, prefill, ck  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF",
                "/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf"))
ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
ap.add_argument("--lib", default="/workspace/qwentin/build-qwen/libforward_qwen.so")
ap.add_argument("--model-name", default="qwentin-qwen3.6-27b-fp6")
ap.add_argument("--ctx", type=int, default=0, help="max prompt+gen tokens (0 = engine limit)")
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--tau", type=float, default=12.0)
ap.add_argument("--maxn", type=int, default=8)
ap.add_argument("--prefix-cache", dest="prefix_cache", action="store_true", default=True,
                help="reuse engine state across requests sharing a token prefix (default ON)")
ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
ap.add_argument("--prefix-cache-min", type=int, default=256,
                help="min reusable prefix tokens (also gated at 25%% of the new prompt)")
ap.add_argument("--prefix-cache-live", action="store_true", default=False,
                help="also continue from the live post-generation state (float-eps "
                     "equivalent, NOT token-identical to a full prefill)")
ap.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                default=(os.environ.get("TQ_NO_THINK", "") == "1"),
                help="default enable_thinking=false (reasoning off) unless a request "
                     "explicitly sets enable_thinking. Recommended for agent/tool use: "
                     "keeps the prefix cache valid across turns (reasoning otherwise "
                     "forces a full re-prefill each turn). Env: TQ_NO_THINK=1")
ap.add_argument("--wide-prefill", dest="wide_prefill", action="store_true", default=True,
                help="wide (N=128) tensor-core prefill path for cold prompts (default ON; "
                     "~2-3x faster first-turn prefill, works at Q4-KV/256k; validated "
                     "needle 4/4 @24k/@120k). Sets TQ_WIDE_PREFILL=1 + TQ_WIDE_ATTN_MMA=1 "
                     "unless those env vars are set explicitly (env always wins).")
ap.add_argument("--no-wide-prefill", dest="wide_prefill", action="store_false")
args = ap.parse_args()

# The engine lib and prefill() read these lazily; explicit env overrides the flag.
os.environ.setdefault("TQ_WIDE_PREFILL", "1" if args.wide_prefill else "0")
os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1" if args.wide_prefill else "0")

print(f"[serve] loading {args.tqf} ...", flush=True)
TOK = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
LIB = load_lib(args.lib)
LIB.qwn_set_sampling.argtypes = [ctypes.c_float, ctypes.c_ulonglong]
LIB.qwn_set_sampling.restype = ctypes.c_int
ck(LIB.qwn_init(args.tqf.encode()), "init")
try:
    eng_max = int(LIB.qwn_max_seq())
except Exception:
    eng_max = 2048
if args.ctx <= 0 or args.ctx > eng_max - args.depth - 4:
    args.ctx = eng_max - args.depth - 4
ENG = Eng(LIB)
ENG_LOCK = threading.Lock()
EOS_IDS = set(int(t) for t in [TOK.eos_token_id] if t is not None)
for name in ("<|im_end|>", "<|endoftext|>"):
    try:
        t = TOK.convert_tokens_to_ids(name)
        if t is not None and t >= 0:
            EOS_IDS.add(int(t))
    except Exception:
        pass

# Prefix/KV cache between requests (single slot: the engine holds ONE sequence).
#   prompt : token ids of the request that produced the engine state. The
#            positional KV/trunk rows for prompt[0..anchor-1] are live in the
#            engine; DeltaNet snapshot slot 0 holds the state at the anchor.
#   anchor : 16-aligned chunk boundary inside that prompt where slot 0 was
#            taken. Restoring slot 0 + prefilling from `anchor` replays the
#            EXACT chunking of a fresh full prefill -> bit-identical state.
#   tokens : (--prefix-cache-live only) every token that went THROUGH the
#            engine and is committed in its live state (prompt + chain commits,
#            WITHOUT the pending bonus seed). This is the engine sequence, not
#            the (possibly stop-string/max_tokens-cut) emitted text.
#   pending/pending_greedy: bonus token predicted right after `tokens`, and
#            whether it was an argmax (temp==0 or straight prefill return).
#   high   : number of positions the engine committed (prompt + generation).
#            Reuse REQUIRES the new prompt to reach past it (P >= high): a
#            shrunken prompt under a longer old state was measured to flip
#            near-tie argmaxes from ~100 tokens in even with the rows above P
#            zeroed (drafter-side eps leak, mechanism not pinned down) -- those
#            requests take the full reset+prefill path instead.
PC = {"valid": False, "prompt": [], "anchor": 0, "high": 0,
      "tokens": [], "pending": -1, "pending_greedy": False}
LIB.qwn_kv_clear_rows.argtypes = [ctypes.c_int, ctypes.c_int]
LIB.qwn_kv_clear_rows.restype = ctypes.c_int
if args.prefix_cache_live and EOS_IDS:
    # keep spec rounds from committing tokens PAST an EOS (the truncated tail
    # was never emitted; this keeps `tokens` a clean continuation point)
    LIB.qwn_set_commit_stop.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    LIB.qwn_set_commit_stop.restype = ctypes.c_int
    _stop = sorted(EOS_IDS)
    ck(LIB.qwn_set_commit_stop((ctypes.c_int * len(_stop))(*_stop), len(_stop)),
       "set_commit_stop")
print(f"[serve] ready on :{args.port} (eos={sorted(EOS_IDS)}, "
      f"prefix_cache={'on' if args.prefix_cache else 'off'}"
      f"{'+live' if args.prefix_cache_live else ''}, "
      f"thinking={'off (default)' if args.no_thinking else 'template-default (on)'}, "
      f"wide_prefill={'on' if os.environ.get('TQ_WIDE_PREFILL', '0') not in ('', '0') else 'off'}"
      f"+mma={'on' if os.environ.get('TQ_WIDE_ATTN_MMA', '0') not in ('', '0') else 'off'})", flush=True)


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _anchor_of(P):
    """Anchor boundary for a P-token prompt: the last 128-aligned chunk boundary
    at least 8 tokens inside it.

    Margin 8: the chat template's generation tail (e.g. the closed empty
    `<think>` block, 4 tokens) is NOT part of the next turn's prompt, so an
    anchor inside it can never match -- the old `last 16-aligned boundary`
    put the anchor past the shared prefix for P % 16 in {1,2,3} (~19% of
    turns = guaranteed full re-prefill).

    128-alignment: the wide prefill path chunks at 128; a replay from a
    128-aligned anchor reproduces the exact chunk seams of a fresh prefill
    (16-aligned-only anchors kept the 16-token path bit-identical but made
    wide-path replays eps-equivalent instead). 128 is a multiple of 16, so
    the baseline chunked path keeps its bit-identity guarantee too."""
    A = ((P - 8) // 128) * 128
    return A if A > 0 else 0


def _pc_store(prompt_ids, anchor, eng_extra, temp, rounds):
    """Record the engine state contents after a finished request. eng_extra =
    [first seed] + every FULL round chain tail (uncut by the EOS/max_tokens
    emission rules); its last element is the pending bonus seed."""
    seq = list(prompt_ids) + list(eng_extra)
    PC["prompt"] = list(prompt_ids)
    PC["anchor"] = anchor
    PC["tokens"] = seq[:-1]
    PC["pending"] = seq[-1]
    PC["pending_greedy"] = (temp == 0.0 or rounds == 0)
    PC["high"] = len(seq) - 1          # committed positions 0..high-1
    PC["valid"] = True


def generate(prompt_ids, max_new, temp, seed, on_tokens, no_cache=False, dbg=False):
    """Run spec rounds; call on_tokens(new_token_list) as chunks commit.
    Returns (gen_count, finish_reason, stats)."""
    P = len(prompt_ids)
    t0 = time.time()
    with ENG_LOCK:
        LIB.qwn_set_sampling(ctypes.c_float(temp), ctypes.c_ulonglong(seed))
        # ---- prefix-cache decision (token-level, against the LAST request) ----
        mode, reused = "full", 0
        if args.prefix_cache and not no_cache and PC["valid"]:
            need = max(args.prefix_cache_min, P // 4)
            if args.prefix_cache_live:
                ct = PC["tokens"]
                C = _common_prefix(prompt_ids, ct)
                if C == len(ct) and C == P and PC["pending_greedy"]:
                    mode, reused = "exact", C      # same sequence: state is ready
                elif C == len(ct) and C < P and C >= need:
                    mode, reused = "live", C       # committed state continues at C
            if mode == "full":
                A = PC["anchor"]
                if (0 < A < P and A >= need and P >= PC["high"]
                        and prompt_ids[:A] == PC["prompt"][:A]):
                    mode, reused = "anchor", A     # slot-0 restore, bit-identical
        PC["valid"] = False        # re-validated when this request finishes
        if mode != "full":
            # zero positional rows past this prompt: a previous request whose
            # state ran longer must not leak into rounds (reset-equivalence;
            # rows below P are rewritten by the continuation prefill)
            ck(LIB.qwn_kv_clear_rows(P, -1), "kv_clear_rows")
        A_new = _anchor_of(P)
        if mode == "exact":
            seed_tok = PC["pending"]   # bonus argmax; root hidden already live
            anchor_in_slot = PC["anchor"]
        else:
            if mode == "anchor":
                # DeltaNet back to the previous prompt's anchor; positional
                # KV/trunk rows below it are live, the rest gets re-prefilled
                # through the same 16-token chunk grid a full prefill uses
                ENG.restore(0)
            start = reused if mode != "full" else 0
            run_anchor = A_new if A_new > start else 0
            seed_tok = prefill(ENG, prompt_ids, P - 1, start=start, anchor=run_anchor)
            ENG.snapshot_root()
            # slot 0 holds the new anchor if this prefill crossed it; on a reuse
            # path that didn't cross it, the previous snapshot stays valid (the
            # new prompt extends the old one through the old anchor). After a
            # full reset, an uncrossed anchor means no usable snapshot.
            if run_anchor:
                anchor_in_slot = run_anchor
            else:
                anchor_in_slot = PC["anchor"] if mode != "full" else 0
        t_prefill = time.time() - t0
        chain_buf = (ctypes.c_int * (args.maxn + 2))()
        st_buf = (ctypes.c_int * 2)()
        cur_seed, cur_pos = seed_tok, P - 1
        out, rounds, finish = [], 0, "length"
        eng_extra = [seed_tok]     # engine-side continuation (incl. pending seed)
        # the prefill return IS the first generated token (greedy at v1 even for
        # temp>0 -- single-token asymmetry, documented); emit + EOS-check it
        if seed_tok in EOS_IDS:
            stats0 = {"prefill_s": round(t_prefill, 3), "gen_s": 0.0, "rounds": 0,
                      "accept_len": 0.0, "gen_tok_s": None,
                      "prefix": mode, "reused_tokens": reused,
                      "prefilled_tokens": P - reused}
            if dbg:
                stats0["gen_ids"] = []
                stats0["eng_tail_ids"] = list(eng_extra)
            _pc_store(prompt_ids, anchor_in_slot, eng_extra, temp, 0)
            return 0, "stop", stats0
        out.append(seed_tok)
        on_tokens([seed_tok])
        t1 = time.time()
        while len(out) < max_new:
            cl = LIB.qwn_spec_round(int(cur_seed), int(cur_pos), args.depth, args.k,
                                    ctypes.c_float(args.tau), args.maxn, chain_buf, st_buf)
            if cl < 0:
                finish = "error"
                break
            chunk = list(chain_buf[1:cl])
            rounds += 1
            cur_seed, cur_pos = st_buf[0], st_buf[1]
            eng_extra.extend(chunk)
            cut = None
            for i, t in enumerate(chunk):
                if t in EOS_IDS:
                    cut = i
                    break
            if cut is not None:
                out.extend(chunk[:cut])
                on_tokens(chunk[:cut])
                finish = "stop"
                break
            room = max_new - len(out)
            emit = chunk[:room]
            out.extend(emit)
            on_tokens(emit)
            if cur_pos + args.depth + 2 >= args.ctx:
                finish = "length"
                break
        dt = time.time() - t1
        if finish != "error":
            _pc_store(prompt_ids, anchor_in_slot, eng_extra, temp, rounds)
    stats = {"prefill_s": round(t_prefill, 3), "gen_s": round(dt, 3),
             "rounds": rounds, "accept_len": round(len(out) / max(1, rounds), 2),
             "gen_tok_s": round(len(out) / dt, 1) if dt > 0 else None,
             "prefix": mode, "reused_tokens": reused,
             "prefilled_tokens": P - reused}
    if dbg:
        stats["gen_ids"] = list(out)
        stats["eng_tail_ids"] = list(eng_extra)
    return len(out), finish, stats


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def split_think(full, primed):
    """Qwen3.6 reasoning split. The chat template opens `<think>\\n` in the PROMPT
    (when thinking is on), so the model emits `<reasoning></think><answer>` -- only
    the close tag is in the output. Returns (reasoning, answer). When thinking is
    off the template emits a closed empty block, primed is False, and everything
    is the answer. If the close never arrives (length cut mid-think) it is all
    reasoning. Raw (no strip) so streaming char counters stay monotonic."""
    if not primed:
        return "", full
    i = full.find("</think>")
    if i < 0:
        return full, ""
    return full[:i], full[i + len("</think>"):]

def _coerce_arg(val, typ):
    """Coerce an XML-ish parameter STRING to its JSON-schema type. The Qwen3.6
    template serializes every <parameter> value as text, so numbers/bools/arrays
    arrive as strings and fail strict client schemas (e.g. zod number). With the
    declared type we coerce exactly; without it we try a JSON literal and keep the
    string on failure (so paths like "client.py" stay strings)."""
    s = val.strip() if isinstance(val, str) else val
    if typ in ("integer", "number"):
        try:
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return val
    if typ == "boolean":
        if isinstance(s, str) and s.lower() in ("true", "false"):
            return s.lower() == "true"
        return val
    if typ in ("array", "object"):
        try:
            return json.loads(s)
        except Exception:
            return val
    if typ == "string":
        return val
    # type unknown: best-effort JSON literal (numbers/bools/null/array/object)
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats, both supported:
    (a) JSON:   <tool_call>{"name":..., "arguments": {...}}</tool_call>
    (b) XML-ish (Qwen3.6 template): <tool_call><function=NAME>
        <parameter=KEY>VALUE</parameter>...</function></tool_call>
    XML values are type-coerced against the tool's declared parameter schema.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    # name -> {param: json-schema-type} from the request's tools
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []
    def _emit(name, args):
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:16],
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
    def _take(m):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            _emit(obj.get("name", ""), obj.get("arguments", {}))
            return ""
        except Exception:
            pass
        fm = _re.search(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", raw, _re.S)
        if fm:
            name = fm.group(1)
            ptypes = types.get(name, {})
            args = {}
            for pm in _re.finditer(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", fm.group(2), _re.S):
                key = pm.group(1)
                args[key] = _coerce_arg(pm.group(2), ptypes.get(key))
            _emit(name, args)
            return ""
        return m.group(0)   # unparseable: leave verbatim in content
    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE), _take,
                    text, flags=_re.S)
    return clean.strip(), (calls or None)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": args.model_name, "object": "model", "owned_by": "qwentin"}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        chat = self.path.rstrip("/").endswith("/chat/completions")
        comp = self.path.rstrip("/").endswith("/completions") and not chat
        if not (chat or comp):
            return self._json(404, {"error": {"message": "not found"}})
        if req.get("n", 1) != 1 or req.get("logprobs"):
            return self._json(400, {"error": {"message": "n>1/logprobs unsupported"}})
        tools = req.get("tools") or None
        warn = None
        top_p = float(req.get("top_p", 1.0))
        if top_p != 1.0:
            warn = f"top_p={top_p} unsupported, clamped to 1.0"
        temp = float(req.get("temperature", 1.0 if chat else 1.0))
        seed = int(req.get("seed", int.from_bytes(os.urandom(4), "little")))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        # non-OpenAI test knobs: tl_no_cache forces the full reset+prefill path,
        # tl_debug echoes emitted/engine token ids in x_qwentin (gate tooling)
        no_cache = bool(req.get("tl_no_cache", False))
        dbg = bool(req.get("tl_debug", False))

        thinking_primed = False
        if chat:
            msgs = req.get("messages", [])
            # OpenAI wire format carries tool_call arguments as a JSON STRING;
            # the Qwen template iterates them as a mapping -- parse in place.
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            pass
            ckw = dict(req.get("chat_template_kwargs") or {})
            # Resolve thinking with explicit precedence (high -> low):
            #   1. top-level enable_thinking            (request is explicit)
            #   2. chat_template_kwargs.enable_thinking  (honored as-is)
            #   3. reasoning_effort                      (none/minimal/off -> off, else on;
            #                                            OpenAI-style knob, vLLM #43401 parity)
            #   4. server --no-thinking default          -> off
            #   5. otherwise leave to the chat template  (Qwen default = on)
            if "enable_thinking" in req:
                ckw["enable_thinking"] = bool(req["enable_thinking"])
            elif "enable_thinking" in ckw:
                pass
            elif req.get("reasoning_effort") is not None:
                eff = str(req["reasoning_effort"]).strip().lower()
                ckw["enable_thinking"] = eff not in ("none", "minimal", "off", "0", "false", "")
            elif args.no_thinking:
                ckw["enable_thinking"] = False
            try:
                text = TOK.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                               tokenize=False, **ckw)
                ids = TOK(text, add_special_tokens=False).input_ids
            except Exception as e:
                return self._json(400, {"error": {"message": f"chat template: {e}"}})
            # template opened an unclosed <think> -> model output carries reasoning
            # up to </think>; split it out of content into reasoning_content.
            thinking_primed = text.rstrip().endswith("<think>")
        else:
            ids = TOK(req.get("prompt", ""), add_special_tokens=False).input_ids
        if len(ids) < 2:
            ids = (TOK("\n", add_special_tokens=False).input_ids + list(ids))[-2:]
        if len(ids) + max_new > args.ctx:
            max_new = max(1, args.ctx - len(ids))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        stream = bool(req.get("stream", False))
        obj_t = "chat.completion" if chat else "text_completion"

        text_parts, stop_hit = [], threading.Event()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def sse(obj):
                data = f"data: {json.dumps(obj)}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            if chat:
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": args.model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})

            dec_state = {"buf": []}

            def on_tokens(toks):
                if stop_hit.is_set():
                    return
                dec_state["buf"].extend(toks)
                txt = TOK.decode(dec_state["buf"])
                if txt.endswith("\ufffd"):   # incomplete utf-8: hold the tail
                    return
                # tool-call block: stop streaming content, buffer to the end
                # (v1: a detected opening tag mutes the stream; the parsed calls
                # go out in the final chunk as a tool_calls delta)
                if tools and (TOOL_OPEN in ("".join(text_parts) + txt) or dec_state.get("tool")):
                    dec_state["tool"] = True
                    dec_state["buf"] = []
                    text_parts.append(txt)
                    return
                dec_state["buf"] = []
                text_parts.append(txt)
                full = "".join(text_parts)
                for st in stops:
                    if st and st in full:
                        stop_hit.set()
                        return
                if chat:
                    # route deltas: reasoning_content before </think>, content after.
                    # recompute the split from the full text each chunk so a close
                    # tag straddling a chunk boundary still lands correctly.
                    r_full, c_full = split_think(full, thinking_primed)
                    r_new = r_full[dec_state.get("r_emit", 0):]
                    c_new = c_full[dec_state.get("c_emit", 0):]
                    dec_state["r_emit"] = len(r_full)
                    dec_state["c_emit"] = len(c_full)
                    delta = {}
                    if r_new:
                        delta["reasoning_content"] = r_new
                    if c_new:
                        delta["content"] = c_new
                    if not delta:
                        return
                    ch = {"index": 0, "delta": delta}
                else:
                    ch = {"index": 0, "text": txt}
                sse({"id": rid, "object": obj_t + (".chunk" if chat else ""),
                     "created": created, "model": args.model_name, "choices": [ch]})

            ngen, finish, stats = generate(list(ids), max_new, temp, seed, on_tokens,
                                           no_cache=no_cache, dbg=dbg)
            if stop_hit.is_set():
                finish = "stop"
            final_delta = {}
            if chat and tools and dec_state.get("tool"):
                _, tcs = parse_tool_calls("".join(text_parts), tools)
                if tcs:
                    for i, tc in enumerate(tcs):
                        tc["index"] = i
                    final_delta = {"tool_calls": tcs}
                    finish = "tool_calls"
            ch = ({"index": 0, "delta": final_delta, "finish_reason": finish} if chat
                  else {"index": 0, "text": "", "finish_reason": finish})
            sse({"id": rid, "object": obj_t + (".chunk" if chat else ""), "created": created,
                 "model": args.model_name, "choices": [ch],
                 "usage": {"prompt_tokens": len(ids), "completion_tokens": ngen,
                           "total_tokens": len(ids) + ngen},
                 "x_qwentin": stats})
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        def on_tokens_ns(toks):
            text_parts.append(TOK.decode(toks))

        ngen, finish, stats = generate(list(ids), max_new, temp, seed, on_tokens_ns,
                                       no_cache=no_cache, dbg=dbg)
        text = "".join(text_parts)
        for st in stops:
            if st and st in text:
                text = text.split(st)[0]
                finish = "stop"
        reasoning = ""
        if chat:
            reasoning, text = split_think(text, thinking_primed)
            reasoning, text = reasoning.strip("\n"), text.lstrip("\n")
        tool_calls = None
        if chat and TOOL_OPEN in text:
            text, tool_calls = parse_tool_calls(text, tools)
        msg = {"role": "assistant", "content": text or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
            finish = "tool_calls"
        choice = ({"index": 0, "message": msg, "finish_reason": finish} if chat
                  else {"index": 0, "text": text, "finish_reason": finish})
        out = {"id": rid, "object": obj_t, "created": created, "model": args.model_name,
               "choices": [choice],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": ngen,
                         "total_tokens": len(ids) + ngen},
               "x_qwentin": stats}
        if warn:
            out["x_qwentin"]["warning"] = warn
        self._json(200, out)


if __name__ == "__main__":
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
