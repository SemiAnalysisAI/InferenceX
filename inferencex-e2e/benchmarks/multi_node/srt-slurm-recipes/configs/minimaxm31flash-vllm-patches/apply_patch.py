"""Apply a git-style unified diff (new and modified text files) under a root
directory, with the Python standard library only: the vLLM images ship neither
git nor patch(1). Every hunk must match exactly; nothing is written unless the
whole patch applies."""

import re
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse(text):
    files, cur = [], None
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": []}
            files.append(cur)
        elif line.startswith("--- ") and cur is not None and not cur["hunks"]:
            path = line[4:].strip()
            cur["old"] = None if path == "/dev/null" else path[2:]
        elif line.startswith("+++ ") and cur is not None and not cur["hunks"]:
            cur["new"] = line[4:].strip()[2:]
        elif line.startswith("@@"):
            m = HUNK.match(line)
            old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
            new_len = int(m.group(4) or 1)
            body, o, n = [], 0, 0
            i += 1
            while o < old_len or n < new_len:
                l = lines[i]
                tag, content = l[0], l[1:]
                if l.startswith("\\"):
                    i += 1
                    continue
                body.append((tag, content))
                if tag in " -":
                    o += 1
                if tag in " +":
                    n += 1
                i += 1
            cur["hunks"].append((old_start, body))
            continue
        i += 1
    return files


def apply_hunks(src_lines, hunks, name):
    """Apply hunks in order. Like git apply, a hunk whose context matches exactly
    but at another line (the file moved upstream) applies at the nearest match."""
    out, pos, offset = [], 0, 0
    for old_start, body in hunks:
        old = [c for t, c in body if t in " -"]
        new = [c for t, c in body if t in " +"]
        want = (old_start - 1 if old else old_start) + offset
        start = None
        for delta in range(len(src_lines) + 1):
            for cand in (want - delta, want + delta):
                if pos <= cand <= len(src_lines) - len(old) and src_lines[cand:cand + len(old)] == old:
                    start = cand
                    break
            if start is not None:
                break
        if start is None:
            raise SystemExit(f"hunk at line {old_start} does not apply to {name}")
        offset = start - (old_start - 1 if old else old_start)
        out += src_lines[pos:start] + new
        pos = start + len(old)
    return out + src_lines[pos:]


def main(patch_file, root):
    root = Path(root)
    results = {}
    for f in parse(Path(patch_file).read_text()):
        target = root / f["new"]
        if f["old"] is None:
            if target.exists():
                raise SystemExit(f"{target} already exists")
            results[target] = apply_hunks([], f["hunks"], f["new"])
        else:
            src = (root / f["old"]).read_text().splitlines(keepends=True)
            results[target] = apply_hunks(src, f["hunks"], f["old"])
    for target, lines in results.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines))
    print(f"applied {len(results)} files under {root}")


if __name__ == "__main__":
    main(*sys.argv[1:])
