#!/usr/bin/env python3
"""Roadmap E milestone 3.2 + 3.3: continuous-batching scheduler + multi-client OpenAI server.

A single engine thread owns the CUDA engine (paged Q4 KV pool). HTTP handler threads only
tokenize, submit a Request, and wait on its completion event. The engine thread runs a
continuous-batching loop:
  - ADMIT queued requests into free slots (prefill the prompt single-stream, snapshot into the
    slot's paged blocks via qwn_paged_load_client = the validated bring-up path), subject to
    admission control (free slot + enough free pool blocks).
  - DECODE one paged step over ALL active slots (qwn_paged_decode_step), push each slot's new
    token to its request, advance positions, and DETACH finished requests (EOS / max_tokens),
    returning their blocks to the pool.

Decodes are batched (the throughput win); prefills are serialized in the engine thread (MVP --
chunked/in-batch prefill is a later optimization). Default-off relative to the single-stream
serve_openai.py; this is a SEPARATE server.

Modes:
  --selftest     : staggered join/leave correctness vs single-stream + aggregate tok/s (no HTTP)
  (default)      : run the OpenAI server on --port

Run (GPU7, prod Q4):
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=8192 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --selftest
  CUDA_VISIBLE_DEVICES=7 TQ_CTX=131072 TQ_KV_Q4=1 python3 -u tools/serve_batched.py --port 8100
"""
from __future__ import annotations
import argparse, ctypes, json, os, threading, time, queue, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVE_MAX = 128   # prefill wave column cap = the fast activation quantizer's smem limit


def load_lib(path):
    L = ctypes.CDLL(path)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_decode.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_decode.restype = ctypes.c_int
    L.qwn_paged_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; L.qwn_paged_init.restype = ctypes.c_int
    L.qwn_paged_free.restype = ctypes.c_int
    L.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; L.qwn_paged_reset_slot.restype = ctypes.c_int
    L.qwn_paged_load_client.argtypes = [ctypes.c_int, ctypes.c_int]; L.qwn_paged_load_client.restype = ctypes.c_int
    L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_decode_step.restype = ctypes.c_int
    L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; L.qwn_paged_stats.restype = ctypes.c_int
    L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_paged_prefill_batch.restype = ctypes.c_int
    L.qwn_prefill_chunk.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_prefill_chunk.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
    return L


def fast_prefill(L, ids, chunk=16):
    """Single-stream prefill in <=16-token spec chunks (qwn_prefill_chunk) so long
    contexts (e.g. 64k) prefill in seconds instead of token-by-token minutes.
    Returns (final_argmax, last_pos). Timing-grade: the committed KV is what
    qwn_paged_load_client snapshots into each paged slot."""
    L.qwn_reset_state()
    n = len(ids); pos = 0; am = 0
    out = (ctypes.c_int * 1)()
    while pos < n:
        c = min(chunk, n - pos)
        buf = (ctypes.c_int * c)(*[int(x) for x in ids[pos:pos + c]])
        rc = L.qwn_prefill_chunk(buf, pos - 1, c, out)
        if rc != 0:
            raise RuntimeError(f"qwn_prefill_chunk rc={rc} at pos {pos}")
        am = out[0]; pos += c
    return am, n - 1


def paged_prefill_slot(L, slot, ids, page, wave_max=128):
    """Prefill one paged slot to len(ids) tokens via qwn_paged_prefill_batch in
    <=wave_max-col chunks (the server's _prefill_long path; allocates blocks from the
    pool, no single-stream/spec scratch). Returns the seed (argmax of the final chunk).
    Raises on -4 (pool exhausted = this worker count does not fit)."""
    n = len(ids); pos = 0; seed = 0
    oseed = (ctypes.c_int * 1)()
    while pos < n:
        c = min(wave_max, n - pos)
        final = 1 if pos + c >= n else 0
        rc = L.qwn_paged_prefill_batch(
            _ci([int(x) for x in ids[pos:pos + c]]), _ci([slot] * c),
            _ci(list(range(pos, pos + c))), _ci([slot]), _ci([0]), _ci([c]),
            _ci([final]), 1, c, oseed)
        if rc != 0:
            raise RuntimeError(f"qwn_paged_prefill_batch rc={rc} at pos {pos} slot {slot}")
        seed = oseed[0]; pos += c
    return seed


def _ci(a):
    return (ctypes.c_int * len(a))(*a)


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


class Request:
    __slots__ = ("ids", "max_new", "eos", "out", "done", "slot", "pos", "next_tok",
                 "started", "t_admit", "t_first", "t_done", "n_prompt", "err")

    def __init__(self, ids, max_new, eos):
        self.ids = ids; self.max_new = max_new; self.eos = set(eos)
        self.out = []; self.done = threading.Event(); self.slot = -1
        self.pos = 0; self.next_tok = 0; self.started = False
        self.t_admit = self.t_first = self.t_done = 0.0; self.n_prompt = len(ids); self.err = None


class BatchedEngine:
    def __init__(self, lib, tqf, max_slots, num_blocks, page):
        self.L = lib
        ck(self.L.qwn_init(tqf.encode()), "init")
        self.page = page; self.max_slots = max_slots
        ck(self.L.qwn_paged_init(max_slots, num_blocks, page), "paged_init")
        fb, tb, pg, mb = self._stats()
        self.num_blocks = tb; self.max_blocks_per_seq = mb
        self.free_slots = list(range(max_slots))
        self.active = {}            # slot -> Request
        self.q = []                 # pending Requests (FIFO)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.running = True
        self.steps = 0; self.decoded_tokens = 0
        self.prefill_waves = 0; self.prefilled_tokens = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _stats(self):
        fb, tb, pg, mb = (ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int())
        ck(self.L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb)), "stats")
        return fb.value, tb.value, pg.value, mb.value

    def submit(self, ids, max_new, eos):
        req = Request(ids, max_new, eos)
        with self.cv:
            self.q.append(req)
            self.cv.notify()
        return req

    def _free_blocks(self):
        return self._stats()[0]

    def _activate(self, req, slot, seed):
        req.slot = slot; req.pos = req.n_prompt; req.next_tok = seed
        req.out.append(seed); req.started = True; req.t_admit = time.time(); req.t_first = req.t_admit
        self.active[slot] = req
        if seed in req.eos or len(req.out) >= req.max_new:
            self._detach(slot)

    def _prefill_long(self, req, slot):
        # prompt > WAVE_MAX: prefill alone in <=WAVE_MAX-column chunks (continuation; the slot
        # state + paged blocks carry across chunks). seed computed on the final chunk.
        ids = req.ids; n = req.n_prompt; pos = 0; seed = 0
        while pos < n:
            c = min(WAVE_MAX, n - pos)
            final = 1 if pos + c >= n else 0
            oseed = (ctypes.c_int * 1)()
            rc = self.L.qwn_paged_prefill_batch(
                _ci(ids[pos:pos + c]), _ci([slot] * c), _ci(list(range(pos, pos + c))),
                _ci([slot]), _ci([0]), _ci([c]), _ci([final]), 1, c, oseed)
            if rc != 0:
                self.free_slots.append(slot); req.err = f"prefill_long chunk rc={rc}"; req.done.set(); return
            seed = oseed[0]; pos += c
        self._activate(req, slot, seed)

    def _prefill_wave(self, wave):
        toks = []; cslot = []; cpos = []; soff = []; slen = []; sslot = []; sfin = []
        off = 0
        for req, slot in wave:
            soff.append(off); slen.append(req.n_prompt); sslot.append(slot); sfin.append(1)
            toks += req.ids; cslot += [slot] * req.n_prompt; cpos += list(range(req.n_prompt))
            off += req.n_prompt
        K = len(wave)
        oseed = (ctypes.c_int * K)()
        rc = self.L.qwn_paged_prefill_batch(_ci(toks), _ci(cslot), _ci(cpos), _ci(sslot),
                                            _ci(soff), _ci(slen), _ci(sfin), K, off, oseed)
        if rc != 0:
            for req, slot in wave:
                self.free_slots.append(slot); req.err = f"prefill wave rc={rc}"; req.done.set()
            return
        self.prefill_waves += 1; self.prefilled_tokens += off
        for i, (req, slot) in enumerate(wave):
            self._activate(req, slot, oseed[i])

    def _admit(self):
        # batch short queued requests into one <=WAVE_MAX-column prefill wave (1 weight read for
        # all clients); long prompts are prefilled alone in chunks.
        free_blk = self._free_blocks()
        wave = []; cols = 0
        while self.q and self.free_slots:
            req = self.q[0]
            if req.n_prompt < 1:
                self.q.pop(0); req.err = "empty prompt"; req.done.set(); continue
            need = (req.n_prompt + self.page - 1) // self.page
            if need > self.max_blocks_per_seq:
                self.q.pop(0); req.err = "prompt exceeds context"; req.done.set(); continue
            if need > free_blk:
                break                                   # pool full -> wait (admission control)
            if req.n_prompt > WAVE_MAX:                 # long: flush any pending wave, prefill alone
                if wave:
                    break
                self.q.pop(0); slot = self.free_slots.pop()
                ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
                free_blk -= need
                self._prefill_long(req, slot)
                continue
            if cols + req.n_prompt > WAVE_MAX:
                break                                   # wave full -> process it, rest waits
            self.q.pop(0); slot = self.free_slots.pop()
            ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
            free_blk -= need; cols += req.n_prompt
            wave.append((req, slot))
        if wave:
            self._prefill_wave(wave)

    def _detach(self, slot):
        req = self.active.pop(slot)
        ck(self.L.qwn_paged_reset_slot(slot), "reset_slot")
        self.free_slots.append(slot)
        req.t_done = time.time()
        req.done.set()

    def _step(self):
        slots = list(self.active.keys())
        n = len(slots)
        toks = (ctypes.c_int * n)(*[self.active[s].next_tok for s in slots])
        sid = (ctypes.c_int * n)(*slots)
        pos = (ctypes.c_int * n)(*[self.active[s].pos for s in slots])
        out = (ctypes.c_int * n)()
        ck(self.L.qwn_paged_decode_step(toks, sid, pos, n, out), "paged_step")
        self.steps += 1; self.decoded_tokens += n
        finished = []
        for j, s in enumerate(slots):
            req = self.active[s]
            o = out[j]
            req.out.append(o); req.pos += 1; req.next_tok = o
            if o in req.eos or len(req.out) >= req.max_new:
                finished.append(s)
        for s in finished:
            self._detach(s)

    def _loop(self):
        while True:
            with self.cv:
                while self.running and not self.q and not self.active:
                    self.cv.wait(timeout=0.5)
                if not self.running and not self.active and not self.q:
                    return
                try:
                    self._admit()
                except Exception as e:
                    print(f"[engine] admit error: {e}", flush=True)
                have = bool(self.active)
            if have:
                try:
                    self._step()
                except Exception as e:                  # never let the engine thread die
                    print(f"[engine] step error: {e}", flush=True)
                    with self.cv:
                        for s in list(self.active.keys()):
                            r = self.active.pop(s)
                            try:
                                self.L.qwn_paged_reset_slot(s)
                            except Exception:
                                pass
                            self.free_slots.append(s)
                            r.err = f"step failed: {e}"; r.done.set()

    def shutdown(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
        self.thread.join(timeout=5)
        self.L.qwn_paged_free(); self.L.qwn_free()


# ----------------------------- self test (milestone 3.2) -----------------------------
def selftest(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    prompts_txt = [
        "The history of cartography is the study of how maps",
        "In quantum mechanics, the wave function describes",
        "def fibonacci(n):\n    # return the nth Fibonacci number\n",
        "The recipe for a classic margherita pizza starts with",
        "Climate scientists measure global temperature using",
        "The French Revolution began in 1789 when",
        "To train a neural network you first need to",
        "The mitochondria is the powerhouse of",
    ][:args.clients]
    ids_list = [tok(t, add_special_tokens=False).input_ids for t in prompts_txt]
    MAXNEW = args.gen

    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page)
    fb, tb, pg, mb = eng._stats()
    print(f"engine up: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)

    # single-stream reference (greedy) for each prompt
    refs = []
    for ids in ids_list:
        eng.L.qwn_reset_state()
        am = 0
        for t, tk in enumerate(ids):
            am = ck(eng.L.qwn_decode(int(tk), t), "ref")
        out = [am]; p = len(ids) - 1
        for _ in range(MAXNEW - 1):
            am = ck(eng.L.qwn_decode(am, p + 1), "ref"); p += 1
            out.append(am)
            if am in set(eos):
                break
        refs.append(out)

    # staggered submission: client i joins after i*stagger seconds
    reqs = []
    t0 = time.time()

    def submit_staggered():
        for i, ids in enumerate(ids_list):
            time.sleep(args.stagger)
            reqs.append((i, eng.submit(ids, MAXNEW, eos)))

    th = threading.Thread(target=submit_staggered, daemon=True); th.start()
    th.join()
    for _, r in reqs:
        r.done.wait(timeout=120)
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f"  3.2 CONTINUOUS-BATCHING SELFTEST: {len(ids_list)} clients, stagger={args.stagger}s, gen={MAXNEW}", flush=True)
    print("=" * 72, flush=True)
    total_tok = 0
    for i, r in reqs:
        ref = refs[i]
        n = min(len(ref), len(r.out))
        match = 0
        for a, b in zip(r.out[:n], ref[:n]):
            if a == b:
                match += 1
            else:
                break
        total_tok += len(r.out)
        cont = tok.decode(r.out[1:]) if len(r.out) > 1 else ""
        print(f"  client {i}: gen={len(r.out)} leading-match-vs-single={match}/{n}  "
              f"text={cont[:54]!r}", flush=True)
    agg = total_tok / elapsed
    print(f"\n  aggregate: {total_tok} tokens / {elapsed:.2f}s = {agg:.1f} tok/s across "
          f"{len(ids_list)} staggered clients", flush=True)
    print(f"  engine steps={eng.steps} batched-decode-tokens={eng.decoded_tokens}", flush=True)
    eng.shutdown()


# ----------------------------- tool-calling (ported from serve_openai.py) -----------------
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def _coerce_arg(val, typ):
    """Coerce an XML-ish <parameter> string to its JSON-schema type (Qwen3.6 template
    serializes every value as text). With no declared type, best-effort JSON literal."""
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
    try:
        j = json.loads(s)
        return j if isinstance(j, (int, float, bool, list, dict)) or j is None else val
    except Exception:
        return val


def parse_tool_calls(text, tools=None):
    """Qwen tool-call formats: (a) JSON <tool_call>{"name":..,"arguments":{..}}</tool_call>
    (b) XML-ish <tool_call><function=NAME><parameter=KEY>VALUE</parameter>..</function></tool_call>.
    Returns (clean_text, tool_calls_list_or_None)."""
    import re as _re
    types = {}
    for t in (tools or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = (((fn.get("parameters") or {}).get("properties")) or {})
        types[fn.get("name", "")] = {k: (v or {}).get("type") for k, v in props.items()}
    calls = []

    def _emit(name, args):
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})

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
                args[pm.group(1)] = _coerce_arg(pm.group(2), ptypes.get(pm.group(1)))
            _emit(name, args)
            return ""
        return m.group(0)

    clean = _re.sub(_re.escape(TOOL_OPEN) + r"(.*?)" + _re.escape(TOOL_CLOSE), _take, text, flags=_re.S)
    return clean.strip(), (calls or None)


# ----------------------------- OpenAI server (milestone 3.3) -----------------------------
def make_handler(eng, tok, args):
    # EOS: eos_token_id + im_end/endoftext so tool-call turns terminate cleanly
    eos = set(int(t) for t in [tok.eos_token_id] if t is not None)
    for _name in ("<|im_end|>", "<|endoftext|>"):
        try:
            _t = tok.convert_tokens_to_ids(_name)
            if _t is not None and _t >= 0:
                eos.add(int(_t))
        except Exception:
            pass
    eos = list(eos)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._json(200, {"object": "list", "data": [{"id": args.model_name, "object": "model"}]})
            elif self.path.startswith("/health"):
                fb, tb, _, _ = eng._stats()
                self._json(200, {"status": "ok", "free_blocks": fb, "total_blocks": tb,
                                 "active": len(eng.active), "queued": len(eng.q)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions") and not self.path.startswith("/v1/completions"):
                self._json(404, {"error": "not found"}); return
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            max_new = int(body.get("max_tokens", 128))
            stream = bool(body.get("stream", False))
            is_chat = self.path.startswith("/v1/chat")
            tools = body.get("tools") or None
            if is_chat:
                msgs = body.get("messages", [])
                # OpenAI carries tool_call arguments as a JSON string; the Qwen
                # template iterates them as a mapping -> parse in place.
                for m in msgs:
                    for tc in (m.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except Exception:
                                pass
                think = bool(body.get("enable_thinking", False))
                try:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=think)
                except TypeError:
                    tmpl = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                   tokenize=False)
                ids = tok(tmpl, add_special_tokens=False).input_ids
            else:
                ids = tok(body.get("prompt", ""), add_special_tokens=False).input_ids
            req = eng.submit(list(ids), max_new, eos)
            if not req.done.wait(timeout=args.timeout):
                self._json(504, {"error": "generation timeout"}); return
            if getattr(req, "err", None):
                self._json(500, {"error": req.err}); return
            # include the full committed continuation (out[0] is the first generated token)
            text = tok.decode(req.out if len(req.out) else [], skip_special_tokens=True)
            gen = len(req.out)
            tool_calls = None
            if is_chat and tools and TOOL_OPEN in text:
                text, tool_calls = parse_tool_calls(text, tools)
            finish = "tool_calls" if tool_calls else ("length" if gen >= max_new else "stop")
            cid = f"chatcmpl-{int(time.time()*1000)}"
            if stream:
                # framed OpenAI SSE: role delta, then content OR tool_calls delta, finish,
                # [DONE]. (generation is already complete; framed so streaming clients like
                # opencode accept it. true token-streaming is a later step.)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                base = {"id": cid, "object": "chat.completion.chunk", "model": args.model_name}

                def _sse(o):
                    self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode()); self.wfile.flush()

                _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tc["index"] = i
                    _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]})
                elif text:
                    _sse({**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                return
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            resp = {
                "id": cid, "object": "chat.completion",
                "model": args.model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": req.n_prompt, "completion_tokens": gen,
                          "total_tokens": req.n_prompt + gen},
            }
            self._json(200, resp)

    return H


def serve(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eng = BatchedEngine(load_lib(args.lib), args.tqf, args.max_slots, args.num_blocks, args.page)
    fb, tb, pg, mb = eng._stats()
    print(f"batched server: pool blocks={tb} page={pg} max_slots={eng.max_slots} free={fb}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(eng, tok, args))
    print(f"listening on {args.host}:{args.port}  (model={args.model_name})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.shutdown()


# ----------------------------- steady-state throughput bench -----------------------------
def bench(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    L = load_lib(args.lib)
    ck(L.qwn_init(args.tqf.encode()), "init")
    text = "The history of cartography is the study of maps and "
    base_ids = tok(text, add_special_tokens=False).input_ids
    # tile the base text up to bench_p so we can measure decode at long contexts (e.g. 64k)
    reps = (args.bench_p + len(base_ids) - 1) // max(1, len(base_ids))
    ids = (base_ids * reps)[: args.bench_p]
    Ns = [int(x) for x in args.bench_ns.split(",")]
    Nmax = max(Ns)
    ck(L.qwn_paged_init(max(args.max_slots, Nmax), args.num_blocks, args.page), "paged_init")
    # Prefill all Nmax slots DIRECTLY via the paged path (qwn_paged_prefill_batch,
    # <=128-col chunks) -- the real server prefill. No single-stream spec scratch, so the
    # KV pool can be sized to the actual worker count at long contexts (e.g. 64k). A slot
    # prefill that runs the pool dry returns -4 (= that worker count does not fit).
    seed = 0
    for s in range(Nmax):
        ck(L.qwn_paged_reset_slot(s), "reset")
        seed = paged_prefill_slot(L, s, ids, args.page)
    p = len(ids) - 1
    M = args.bench_iters
    fb = ctypes.c_int(); tb = ctypes.c_int(); pgv = ctypes.c_int(); mb = ctypes.c_int()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pgv), ctypes.byref(mb))
    print("\n" + "=" * 72, flush=True)
    print(f"  3.x STEADY-STATE THROUGHPUT (paged decode) @ ctx={len(ids)} tok, "
          f"{Nmax} slots prefilled, KV blocks used={tb.value - fb.value}/{tb.value}", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'N':>4} {'ms/step':>9} {'agg tok/s':>11} {'per-cli':>9}", flush=True)
    for N in Ns:
        toks = (ctypes.c_int * N)(*([seed] * N))
        sid = (ctypes.c_int * N)(*list(range(N)))
        pos = (ctypes.c_int * N)(*([p + 1] * N))
        out = (ctypes.c_int * N)()
        ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "warm")
        t0 = time.time()
        for _ in range(M):
            ck(L.qwn_paged_decode_step(toks, sid, pos, N, out), "step")
        t_step = (time.time() - t0) / M
        agg = N / t_step
        print(f"  {N:>4} {1000*t_step:>9.3f} {agg:>11.1f} {1.0/t_step:>9.1f}", flush=True)
    L.qwn_paged_free(); L.qwn_free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default="/workspace/models/Qwen3.6-27B/qwen3_6-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default="/workspace/models/Qwen3.6-27B")
    ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--model-name", default="qwentin-qwen3.6-27b-batched")
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=12)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-ns", default="1,2,4,8,16,32")
    ap.add_argument("--bench-p", type=int, default=48)
    ap.add_argument("--bench-iters", type=int, default=64)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--stagger", type=float, default=0.15)
    args = ap.parse_args()
    if args.bench:
        bench(args)
    elif args.selftest:
        selftest(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
