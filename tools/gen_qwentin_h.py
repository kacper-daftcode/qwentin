#!/usr/bin/env python3
"""Generate the C ABI description of libforward_qwen.so from the engine source: src/qwentin.h (the qwn_config struct, one
field per row of src/engine/config.cuh's switch table, and every extern "C" qwn_* entry point, grouped by the
src/engine/ part that defines it) and tools/qwentin_abi.py (the same as ctypes types: Config, CONFIG_FIELDS and the
signatures, with bind(lib)). src/forward_qwen.cu includes qwentin.h first, so a definition that drifts from its
declaration does not compile; tests/test_qwentin_abi.py fails when either file is stale.

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
CONFIG = "src/engine/config.cuh"

_DEF = re.compile(r'extern\s+"C"\s+([^;{()]*?)\b(qwn_\w+)\s*\(([^)]*)\)\s*\{')
_ROW = re.compile(r'\{"(TQ_[A-Z0-9_]+)",\s*&qwn_config::(\w+),')
_GROUP = re.compile(r"^\s*// group: (.+)$")
_CODEBOOK = ("kv_t4_codebook", "TQ_KV_T4_CODEBOOK", "a file of 4096 fp32 entries; NULL = the built-in codebook")
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


def config_fields(root: str = ROOT) -> list[tuple[str | None, str, str]]:
    """(group heading or None, field, TQ_* name) of every int switch in config.cuh's table, in order."""
    with open(os.path.join(root, CONFIG)) as f:
        src = f.read()
    table = src[src.index("k_tq_switches[] = {"):]
    table = table[:table.index("\n};")]
    out, group = [], None
    for line in table.splitlines():
        g = _GROUP.match(line)
        if g:
            group = g.group(1)
            continue
        m = _ROW.search(line)
        if m:
            out.append((group, m.group(2), m.group(1)))
            group = None
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


def render_header(defs, fields) -> str:
    lines = ["/* qwentin.h -- the C ABI of libforward_qwen.so, generated from src/engine/ by tools/gen_qwentin_h.py; do not",
             " * edit. The int entry points return 0 or a negative error code, except the getters (sizes, counts, modes);",
             " * the definitions in src/engine/ say which. */",
             "#ifndef QWENTIN_H", "#define QWENTIN_H", "", "#include <stddef.h>", "#include <stdint.h>", "",
             "#ifdef __cplusplus", 'extern "C" {', "#endif", "",
             "/* engine/config.cuh: the engine's configuration, one field per TQ_* switch (named in each comment; docs/FLAGS.md).",
             " * qwn_config_from_env fills it the way qwn_init reads the environment; qwn_init_with takes it instead. */",
             "typedef struct qwn_config {"]
    for group, field, name in fields:
        if group:
            lines.append(f"    /* {group} */")
        lines.append(f"    int {(field + ';').ljust(24)}/* {name} */")
    field, name, what = _CODEBOOK
    lines += [f"    const char *{field};   /* {name}: {what} */", "} qwn_config;"]
    part = None
    for p, ret, name, params in defs:
        if p != part:
            lines += ["", f"/* engine/{p} */"]
            part = p
        sep = "" if ret.endswith("*") else " "
        lines.append(f"{ret}{sep}{name}({params});")
    lines += ["", "#ifdef __cplusplus", "}", "#endif", "#endif", ""]
    return "\n".join(lines)


def render_abi(defs, fields) -> str:
    lines = ['"""The C ABI of libforward_qwen.so as ctypes types, generated from the engine source by tools/gen_qwentin_h.py --',
             "do not edit. Config is qwn_config (CONFIG_FIELDS maps each TQ_* name to its field); bind(lib) sets restype /",
             "argtypes of every qwn_* entry point the library exports; pointer parameters are c_void_p (pass ctypes pointers,",
             'arrays, ctypes.byref(config), c_void_p(addr) or None)."""',
             "", "import ctypes as C", "", "",
             "class Config(C.Structure):", "    _fields_ = ["]
    for _, field, _name in fields:
        lines.append(f'        ("{field}", C.c_int),')
    lines += [f'        ("{_CODEBOOK[0]}", C.c_char_p),', "    ]", "", "", "CONFIG_FIELDS = {"]
    for _, field, name in fields:
        lines.append(f'    "{name}": "{field}",')
    lines += [f'    "{_CODEBOOK[1]}": "{_CODEBOOK[0]}",', "}", "", "SIGNATURES = {"]
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
    d, cf = definitions(), config_fields()
    for rel, text in ((HEADER, render_header(d, cf)), (ABI, render_abi(d, cf))):
        with open(os.path.join(ROOT, rel), "w") as f:
            f.write(text)
    print(f"wrote {HEADER} and {ABI} ({len(d)} entry points, {len(cf) + 1} config fields)")
