"""The engine source as prompt material: the src/engine/ parts in the include order of src/forward_qwen.cu (the unity
build), as a file list or as one text (~1.6 MB of CUDA). Needle haystacks, prefill prompts and the spec-decode benches
draw their long code documents from it."""

from __future__ import annotations

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def engine_source_files(repo: str = REPO) -> list[str]:
    src = os.path.join(repo, "src")
    with open(os.path.join(src, "forward_qwen.cu")) as f:
        return [os.path.join(src, m.group(1)) for m in (re.match(r'#include "(engine/[\w.]+)"', ln) for ln in f) if m]


def engine_source_text(repo: str = REPO) -> str:
    parts = []
    for p in engine_source_files(repo):
        with open(p) as f:
            parts.append(f.read())
    return "".join(parts)
