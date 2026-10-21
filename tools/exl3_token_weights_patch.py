#!/usr/bin/env python
"""Patch exllamav3's Linear.capture_H to accumulate a TOKEN-WEIGHTED Hessian (code-level QAT, approach 1).

Applies to <exllamav3>/exllamav3/modules/linear.py (idempotent). With the env EXL3_TOKEN_WEIGHTS=<fisher.safetensors>
(written by analysis/exl3_fisher.py) set during `convert.py`, every capture does
    H += (sqrt(w) x)^T (sqrt(w) x),   w_t = (1 - lambda) + lambda * (s_t / mean(s))^alpha,
with s_t the per-token output sensitivity of this Linear's shared-input group and the calibration row matched by a
hash of its token ids (params["input_ids"]). EXL3_TW_ALPHA (default 1), EXL3_TW_LAMBDA (default 1) temper / mix with
uniform. Without the env variable the capture is exactly the upstream one. Rows the file does not know fall back to
uniform (counted; printed at exit).
usage: exl3_token_weights_patch.py /root/qat/exllamav3
"""
import sys, os, re

MARK = "# --- token-weighted Hessian (maxing-qwentin, 2026-10-17) ---"
HELPER = '''
''' + MARK + '''
import os as _tw_os
_TW = None
def _tw_rowhash(ids):
    ids = ids.reshape(-1)[:128].to(torch.int64).cpu()
    w = torch.arange(1, ids.numel() + 1, dtype = torch.int64)
    return int(((ids * w).sum().item() * 1000003 + ids.numel()) % (1 << 62))
class _TokenWeights:
    def __init__(self, path):
        from safetensors.torch import load_file
        d = load_file(path)
        self.alpha = float(_tw_os.environ.get("EXL3_TW_ALPHA", "1"))
        self.lam = float(_tw_os.environ.get("EXL3_TW_LAMBDA", "1"))
        self.rowhash = {int(h): i for i, h in enumerate(d.pop("__rowhash__").tolist())}
        self.w = {}
        for k, v in d.items():
            s = v.float().clamp_min(0)
            if s.sum().item() <= 0:        # no sensitivity recorded (e.g. the lm_head, hooked on 2-D chunks): uniform
                continue
            if self.alpha != 1.0:
                s = s ** self.alpha
            s = s / s.mean().clamp_min(1e-30)
            self.w[k] = (1.0 - self.lam) + self.lam * s
        self.hits = 0; self.misses = 0; self.nokey = set()
        print(f" -- token-weighted Hessian capture: {path} ({len(self.w)} keys, {len(self.rowhash)} rows, alpha {self.alpha}, lambda {self.lam})", flush = True)
    def get(self, key, input_ids):
        k = key
        if k not in self.w:
            k = key.replace("model.layers.", "model.language_model.layers.") if "language_model" not in key else key.replace("model.language_model.layers.", "model.layers.")
        if k not in self.w:
            self.nokey.add(key); return None
        if input_ids is None:
            self.misses += 1; return None
        i = self.rowhash.get(_tw_rowhash(input_ids))
        if i is None:
            self.misses += 1; return None
        self.hits += 1
        return self.w[k][i]
def _token_weights():
    global _TW
    if _TW is None:
        p = _tw_os.environ.get("EXL3_TOKEN_WEIGHTS")
        _TW = _TokenWeights(p) if p else False
        if _TW:
            import atexit
            atexit.register(lambda: print(f" -- token-weighted Hessian: {_TW.hits} weighted captures, {_TW.misses} unmatched rows, keys without weights: {sorted(_TW.nokey)[:6]} ({len(_TW.nokey)})", flush = True))
    return _TW
# --- end token-weighted Hessian ---
'''

OLD = '''            finite = torch.isfinite(x).all(dim = 1)
            if not finite.all():
                x = x[finite]
                rows = x.shape[0]

            params["capture"][self.qmap]["H"].addmm_(x.T, x)'''
NEW = '''            finite = torch.isfinite(x).all(dim = 1)
            tw = _token_weights()
            w = tw.get(self.key, params.get("input_ids")) if tw else None
            if w is not None:
                w = w.to(x.device)
                w = w[: x.shape[0]] if w.numel() >= x.shape[0] else torch.cat([w, w.new_ones(x.shape[0] - w.numel())])
            if not finite.all():
                x = x[finite]
                if w is not None:
                    w = w[finite]
                rows = x.shape[0]
            if w is not None:
                x = x * w.sqrt().unsqueeze(1)

            params["capture"][self.qmap]["H"].addmm_(x.T, x)'''


def main():
    root = sys.argv[1]
    p = os.path.join(root, "exllamav3", "modules", "linear.py")
    s = open(p).read()
    if MARK in s:
        print("already patched:", p); return
    assert s.count(OLD) == 1, "capture_H body not found verbatim; upstream changed -- patch by hand"
    s = s.replace(OLD, NEW)
    # insert the helper after the imports (before the first top-level class/def)
    m = re.search(r"^(class |def )", s, flags = re.M)
    s = s[: m.start()] + HELPER + "\n\n" + s[m.start():]
    open(p, "w").write(s)
    print("patched:", p)


if __name__ == "__main__":
    main()
