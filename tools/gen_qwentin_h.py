#!/usr/bin/env python3
"""Generate the C ABI description of libforward_qwen.so from the engine source: src/qwentin.h (every extern "C" qwn_*
entry point, grouped by the src/engine/ part that defines it) and tools/qwentin_abi.py (the same signatures as ctypes
types, with bind(lib)). src/forward_qwen.cu includes qwentin.h first, so a definition that drifts from its declaration
does not compile; tests/test_qwentin_abi.py fails when either file is stale.

  python3 tools/gen_qwentin_h.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_text  # noqa: E402

ROOT = engine_text.REPO
HEADER = "src/qwentin.h"
ABI = "tools/qwentin_abi.py"

_DEF = re.compile(r'extern\s+"C"\s+([^;{()]*?)\b(qwn_\w+)\s*\(([^)]*)\)\s*\{')
_SCALARS = {"int": "C.c_int", "float": "C.c_float", "double": "C.c_double", "size_t": "C.c_size_t",
            "unsigned long long": "C.c_ulonglong", "long long": "C.c_longlong", "long": "C.c_long"}


def definitions(root: str = ROOT) -> list[tuple[str, str, str, str]]:
    """(part, return type, name, parameter list) of every extern "C" qwn_* definition, in translation-unit order."""
    out = []
    for path in engine_text.engine_source_files(root):
        with open(path) as f:
            src = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", f.read()), flags=re.S)
        for m in _DEF.finditer(src):
            params = " ".join(m.group(3).split()) or "void"
            out.append((os.path.basename(path), " ".join(m.group(1).split()), m.group(2), params))
    return out


def _param_type(p: str) -> str:
    p = " ".join(p.split())
    return p if p.endswith("*") else re.sub(r"\s*\b\w+$", "", p)


def _ctype(t: str) -> str:
    t = t.replace("const ", "").strip()
    if t == "void": return "None"
    if t == "char *": return "C.c_char_p"
    if t.endswith("*"): return "C.c_void_p"
    return _SCALARS[t]


def render_header(defs) -> str:
    lines = ["/* qwentin.h -- the C ABI of libforward_qwen.so, generated from src/engine/ by tools/gen_qwentin_h.py; do not",
             " * edit. The int entry points return 0 or a negative error code, except the getters (sizes, counts, modes);",
             " * the definitions in src/engine/ say which. */",
             "#ifndef QWENTIN_H", "#define QWENTIN_H", "", "#include <stddef.h>", "#include <stdint.h>", "",
             "#ifdef __cplusplus", 'extern "C" {', "#endif"]
    part = None
    for p, ret, name, params in defs:
        if p != part:
            lines += ["", f"/* engine/{p} */"]
            part = p
        sep = "" if ret.endswith("*") else " "
        lines.append(f"{ret}{sep}{name}({params});")
    lines += ["", "#ifdef __cplusplus", "}", "#endif", "#endif", ""]
    return "\n".join(lines)


def render_abi(defs) -> str:
    lines = ['"""The C ABI of libforward_qwen.so as ctypes signatures, generated from the engine source by',
             "tools/gen_qwentin_h.py -- do not edit. bind(lib) sets restype / argtypes of every qwn_* entry point the library",
             'exports; pointer parameters are c_void_p (pass ctypes pointers, arrays, c_void_p(addr) or None)."""',
             "", "import ctypes as C", "", "SIGNATURES = {"]
    for _, ret, name, params in defs:
        args = [] if params == "void" else [_ctype(_param_type(p)) for p in params.split(",")]
        lines.append(f'    "{name}": ({_ctype(ret)}, [{", ".join(args)}]),')
    lines += ["}", "", "", "def bind(lib, names=None):",
              "    for name in names or SIGNATURES:",
              "        fn = getattr(lib, name, None)",
              "        if fn is not None:",
              "            fn.restype, fn.argtypes = SIGNATURES[name]",
              "    return lib", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    d = definitions()
    for rel, text in ((HEADER, render_header(d)), (ABI, render_abi(d))):
        with open(os.path.join(ROOT, rel), "w") as f:
            f.write(text)
    print(f"wrote {HEADER} and {ABI} ({len(d)} entry points)")
