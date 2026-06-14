"""add_watermark.py -- AI-use disclosure watermarker for 22DM015.

Implements the two disclosure markers described in the course `docs/AIinstructions.md`,
applied to AI-generated content so AI assistance is auditable and disclosed:

  Marker 4a (text / markdown): insert U+200D (zero-width joiner) after a
    sentence-terminating '.', '!' or '?' that is followed by whitespace or end
    of string. Restricted to real sentence boundaries and skips fenced/inline
    code, so decimals ('0.5'), filenames ('data_utils.py') and code are untouched.

  Marker 4b (code): ensure the token 'AGLLM' appears in a comment in each source
    file / code cell.

NOTE: the instructor's original add_watermark.py is not present in the linked
repo, so this is a faithful re-implementation of the *documented* behavior. If
the official check_watermarks.py expects different exact semantics, replace this
with the instructor's script. ZWJ is intentionally NOT inserted into code (it
would break identifiers/strings), only into prose.

Usage:
    python add_watermark.py FILE_OR_DIR [...]      # apply markers in place
    python add_watermark.py --check FILE_OR_DIR    # report marker counts only
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ZWJ = "\u200d"  # zero-width joiner (invisible) — escape form so the value is reviewable
AGLLM = "AGLLM"
AGLLM_COMMENT = f"# watermark: {AGLLM} (AI-assisted content disclosure)"

# Sentence boundary: . ! or ? followed by whitespace or end-of-string.
_SENT = re.compile(r"([.!?])(\s|$)")


def watermark_prose(text: str) -> str:
    """Insert ZWJ at real sentence boundaries. Idempotent."""
    text = re.sub(r"([.!?])" + ZWJ, r"\1", text)          # strip prior ZWJ first
    return _SENT.sub(lambda m: m.group(1) + ZWJ + m.group(2), text)


def watermark_markdown(text: str) -> str:
    """Watermark markdown prose, skipping ``` fences and `inline code`."""
    out = []
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if block.startswith("```"):
            out.append(block)
            continue
        out.append("".join(
            seg if seg.startswith("`") else watermark_prose(seg)
            for seg in re.split(r"(`[^`]*`)", block)
        ))
    return "".join(out)


def watermark_code(src: str) -> str:
    """Ensure an AGLLM marker comment is present. ZWJ is never added to code."""
    if AGLLM in src:
        return src
    lines = src.split("\n")
    # place the marker after a leading module docstring if there is one, else top
    insert_at = 0
    if lines and lines[0].lstrip().startswith(('"""', "'''")):
        q = lines[0].lstrip()[:3]
        if lines[0].count(q) < 2:                          # multi-line docstring
            for i in range(1, len(lines)):
                if q in lines[i]:
                    insert_at = i + 1
                    break
    lines.insert(insert_at, AGLLM_COMMENT)
    return "\n".join(lines)


def process_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    md_cells = code_cells = 0
    changed = False
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            new = watermark_markdown(src)
            md_cells += 1
        elif cell.get("cell_type") == "code":
            new = watermark_code(src)
            code_cells += 1
        else:
            continue
        if new != src:
            cell["source"] = new.splitlines(keepends=True) or [""]
            changed = True
    if changed:                       # byte-idempotent: untouched notebooks are not rewritten
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return {"markdown_cells": md_cells, "code_cells": code_cells, "changed": changed}


def process_text_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        new = watermark_markdown(text)
        kind = "markdown"
    elif path.suffix in {".py", ".toml", ".cfg"}:
        new = watermark_code(text)
        kind = "code"
    else:
        return "skipped"
    if new != text:
        path.write_text(new, encoding="utf-8")
    return kind


def iter_targets(paths):
    exts = {".ipynb", ".md", ".py", ".toml", ".cfg"}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix in exts and ".venv" not in f.parts and "checkpoint" not in f.name:
                    yield f
        elif p.suffix in exts:
            yield p


def check(paths) -> None:
    for f in iter_targets(paths):
        raw = f.read_text(encoding="utf-8")
        if f.suffix == ".ipynb":
            raw = "".join("".join(c.get("source", []))
                          for c in json.loads(raw).get("cells", []))
        print(f"{f}: ZWJ={raw.count(ZWJ)}  AGLLM={raw.count(AGLLM)}")


def main(argv) -> None:
    if argv and argv[0] == "--check":
        check(argv[1:])
        return
    for f in iter_targets(argv):
        info = process_notebook(f) if f.suffix == ".ipynb" else process_text_file(f)
        print(f"watermarked {f}: {info}")


if __name__ == "__main__":
    main(sys.argv[1:])
