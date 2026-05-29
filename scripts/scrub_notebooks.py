#!/usr/bin/env python3
"""Scrub secrets / PII from Jupyter notebooks before committing to GitHub.

What it does (idempotent):
  * Replaces a hardcoded Semantic Scholar key assignment
    `os.environ['S2_API_KEY'] = '<literal>'` with the `userdata.get(...)`
    pattern already used by the other keys in the corpus notebook.
  * Redacts a known personal email everywhere it appears (cell source AND
    output) so results are preserved but PII is not.
  * Sweeps every cell source and output for high-confidence secret patterns
    (hf_*, sk-*, nvapi-*, ghp_/gho_*, AWS AKIA*, generic `KEY = "literal"`)
    and redacts them.

Code and outputs are otherwise left untouched — only sensitive substrings
and the one S2 assignment line change.

Usage:
    python scripts/scrub_notebooks.py            # scrub notebooks/*.ipynb in place
    python scripts/scrub_notebooks.py --check     # report only, exit 1 if findings
    python scripts/scrub_notebooks.py a.ipynb b.ipynb [--check]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PII_EMAIL = "you@example.com"
EMAIL_PLACEHOLDER = "you@example.com"

# Replace `os.environ['S2_API_KEY'] = '<literal>'` (single/double quotes) with userdata.get.
S2_ASSIGN = re.compile(
    r"""os\.environ\[\s*['"]S2_API_KEY['"]\s*\]\s*=\s*['"][^'"\n]+['"]"""
)
S2_REPLACEMENT = "os.environ['S2_API_KEY'] = userdata.get('S2_API_KEY')"

# Bare `S2_API_KEY = '<literal>'` style assignments.
BARE_KEY_ASSIGN = re.compile(
    r"""(?P<name>\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)\b)\s*=\s*['"](?P<val>[^'"\n]{8,})['"]"""
)

# High-confidence secret token shapes (redacted wherever they appear).
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                 # HuggingFace token
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                 # OpenAI-style
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),           # NVIDIA build key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),          # GitHub PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key id
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),       # Slack token
]
REDACTED = "<REDACTED>"

# Don't flag obvious placeholders / env lookups as findings.
SAFE_HINTS = (
    "userdata.get", "os.environ.get", "getenv", "getpass", "your-key",
    "YOUR_", "<your", "xxxx", "placeholder", "example",
)


def scrub_text(text: str, findings: list[str], where: str) -> str:
    if not text:
        return text

    # 1) S2 assignment -> userdata.get
    if S2_ASSIGN.search(text):
        findings.append(f"{where}: hardcoded S2_API_KEY assignment")
        text = S2_ASSIGN.sub(S2_REPLACEMENT, text)

    # 2) Personal email -> placeholder
    if PII_EMAIL in text:
        findings.append(f"{where}: personal email ({text.count(PII_EMAIL)}x)")
        text = text.replace(PII_EMAIL, EMAIL_PLACEHOLDER)

    # 3) Known secret token shapes
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            findings.append(f"{where}: secret token matching /{pat.pattern}/")
            text = pat.sub(REDACTED, text)

    # 4) Generic KEY = "literal" assignments (skip safe lookups / placeholders)
    def _bare(m: re.Match) -> str:
        val = m.group("val")
        line = m.group(0)
        if any(h in line for h in SAFE_HINTS) or any(h in val for h in SAFE_HINTS):
            return line
        findings.append(f"{where}: hardcoded {m.group('name')}")
        return f"{m.group('name')} = userdata.get('{m.group('name')}')"

    text = BARE_KEY_ASSIGN.sub(_bare, text)
    return text


def output_text(out: dict) -> str:
    """Concatenate the textual parts of a notebook output for scanning."""
    parts: list[str] = []
    parts += out.get("text", []) if isinstance(out.get("text"), list) else [out.get("text", "")]
    data = out.get("data", {})
    for k, v in data.items():
        if k.startswith("image") or k == "application/vnd.jupyter.widget-view+json":
            continue
        parts += v if isinstance(v, list) else [v if isinstance(v, str) else ""]
    return "".join(p for p in parts if isinstance(p, str))


def scrub_output(out: dict, findings: list[str], where: str) -> None:
    # Stream text
    if isinstance(out.get("text"), list):
        out["text"] = [scrub_text(t, findings, where) for t in out["text"]]
    elif isinstance(out.get("text"), str):
        out["text"] = scrub_text(out["text"], findings, where)
    # Rich data (text/plain, text/markdown, text/html, ...), skip images/widgets
    data = out.get("data", {})
    for k in list(data.keys()):
        if k.startswith("image") or k == "application/vnd.jupyter.widget-view+json":
            continue
        v = data[k]
        if isinstance(v, list):
            data[k] = [scrub_text(t, findings, where) if isinstance(t, str) else t for t in v]
        elif isinstance(v, str):
            data[k] = scrub_text(v, findings, where)


def scrub_notebook(path: Path, check_only: bool) -> list[str]:
    nb = json.loads(path.read_text())
    findings: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", [])
        joined = "".join(src) if isinstance(src, list) else src
        new = scrub_text(joined, findings, f"{path.name} cell[{i}] source")
        if new != joined and not check_only:
            cell["source"] = new.splitlines(keepends=True)
        for j, out in enumerate(cell.get("outputs", []) or []):
            if check_only:
                scrub_text(output_text(out), findings, f"{path.name} cell[{i}] out[{j}]")
            else:
                scrub_output(out, findings, f"{path.name} cell[{i}] out[{j}]")
    if findings and not check_only:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return findings


def main() -> int:
    args = sys.argv[1:]
    check_only = "--check" in args
    paths = [Path(a) for a in args if a.endswith(".ipynb")]
    if not paths:
        paths = sorted(Path("notebooks").glob("*.ipynb"))
    if not paths:
        print("No notebooks found.")
        return 0

    total = 0
    for p in paths:
        findings = scrub_notebook(p, check_only)
        total += len(findings)
        tag = "FOUND" if findings else "clean"
        print(f"[{tag}] {p}")
        for f in findings:
            print(f"    - {f}")

    if check_only:
        if total:
            print(f"\n✗ {total} sensitive item(s) still present.")
            return 1
        print("\n✓ No secrets/PII detected.")
        return 0
    print(f"\n✓ Scrub complete ({total} item(s) redacted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
