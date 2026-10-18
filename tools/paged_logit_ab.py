"""Raw paged-decode logits of one library on fixed tokens: N slots decode a seeded random token stream each, positions
0..steps-1 in lockstep (no prefill waves), and the logits of every step from --from-pos are saved for a bitwise A/B
between builds. N = 8 splits the GQA decode attention from position 384 on (the merge path).
usage: paged_logit_ab.py LIB OUT.pt --tqf MODEL.tqf [--n 8] [--steps 401] [--from-pos 376]
Compare two runs with torch: `a = torch.load(A); b = torch.load(B); max((a[p] - b[p]).abs().max() for p in a)` -- the
README's "bitwise" claims are max |d| = 0 on these dumps (the served env: TQ_KV_V4=1 TQ_KV_T4=1 in front for the trellis
pool; the off-switches of a lever must give 0 against the previous build). Vendored from the qwen256k research harness
(analysis/paged_logit_ab.py)."""
import argparse, ctypes, os, sys
import torch

for k, v in {"TQ_CTX": "262144", "TQ_KV_Q4": "1", "TQ_EMBED_FP8": "2", "TQ_PAGED_ATTN_MMA": "1",
             "TQ_WIDE_ATTN_MMA": "1", "TQ_EMBED_HOST": "1", "TQ_NO_MTP": "1"}.items():
    os.environ.setdefault(k, v)
ap = argparse.ArgumentParser()
ap.add_argument("lib")
ap.add_argument("out")
ap.add_argument("--tqf", required=True, help="the model file (e.g. an EXL3 opt-5.75 tqf)")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--steps", type=int, default=401)
ap.add_argument("--from-pos", type=int, default=376)
ap.add_argument("--mtp-on", action="store_true", help="load the MTP section and turn the paged drafter on (its trunk hooks "
                "run in every decode step; the logits must stay bitwise identical) -- needs a --mtp-exl3 tqf")
a = ap.parse_args()
if a.mtp_on:
    os.environ["TQ_NO_MTP"] = "0"

L = ctypes.CDLL(a.lib)
L.qwn_init.argtypes = [ctypes.c_char_p]
L.qwn_vocab_size.restype = ctypes.c_int
L.qwn_paged_logits_sink.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ci = lambda xs: (ctypes.c_int * len(xs))(*xs)
assert L.qwn_init(a.tqf.encode()) == 0
V = L.qwn_vocab_size()
assert L.qwn_paged_init(a.n, a.n * ((a.steps + 127) // 128 + 1), 128) == 0
if a.mtp_on:
    L.qwn_paged_mtp_init.argtypes = [ctypes.c_int]
    rc = L.qwn_paged_mtp_init(1)
    assert rc == 0, f"qwn_paged_mtp_init rc={rc}"
sink = torch.zeros(a.n, V, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
assert L.qwn_paged_logits_sink(ctypes.c_void_p(sink.data_ptr()), a.n, 8) == 0
for s in range(a.n):
    assert L.qwn_paged_reset_slot(s) == 0
g = torch.Generator().manual_seed(0)
toks = torch.randint(1000, 100000, (a.n, a.steps), generator=g).tolist()
argmax = (ctypes.c_int * a.n)()
out = {}
for p in range(a.steps):
    rc = L.qwn_paged_decode_step(ci([toks[s][p] for s in range(a.n)]), ci(list(range(a.n))), ci([p] * a.n), a.n, argmax)
    assert rc == 0, (p, rc)
    if p >= a.from_pos:
        torch.cuda.synchronize()
        out[p] = sink.cpu().clone()
torch.save(out, a.out)
print("saved", len(out), "steps", file=sys.stderr)
