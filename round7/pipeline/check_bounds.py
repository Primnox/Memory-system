"""Did each Claude auditor stay inside its own folder?   python check_bounds.py FOLDER AGENT_ID [AGENT_ID ...]

Reads each subagent's tool calls from its transcript and prints file names and a
verdict only, never content. Allowed: the folder's brief (*.md) and its pNN_unlabelled /
pNN_labels files, by Read/Write/Edit, and Bash commands that touch nothing else.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

import os
D = Path(os.environ["CLAUDE_SUBAGENTS_DIR"])  # the Claude Code session's subagents/ transcript folder
FOLDER = sys.argv[1].lower()
OK_NAME = re.compile(r"^([A-Z_]+\.md|p\d\d_unlabelled\.json|p\d\d_labels\.json)$", re.I)
PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"')]+|/c/[^\s\"')]+")


def allowed(p: str) -> bool:
    w = PureWindowsPath(p.replace("/", "\\"))
    return w.parent.name.lower() == FOLDER and bool(OK_NAME.match(w.name))


worst = 0
for aid in sys.argv[2:]:
    bad, files = [], Counter()
    f = D / f"agent-{aid}.jsonl"
    if not f.exists():
        print(f"{aid}: no transcript"); worst = 1; continue
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("message")
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for c in m["content"]:
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            name, inp = c["name"], c.get("input", {})
            if name in ("Read", "Write", "Edit", "NotebookEdit"):
                p = inp.get("file_path", "")
                (files.update([PureWindowsPath(p).name]) if allowed(p) else bad.append(f"{name} {p}"))
            elif name in ("Bash", "PowerShell"):
                cmd = inp.get("command", "")
                for path in PATH.findall(cmd):
                    w = PureWindowsPath(path.replace("/", "\\"))
                    if not (allowed(path) or w.name.lower() == FOLDER):
                        bad.append(f"{name} touches {path[:110]}")
                if re.search(r"\b(ls|dir|find|tree|Get-ChildItem|grep|rg)\b|\.\.[\\/]", cmd):
                    bad.append(f"{name} lists/searches/climbs")
            elif name not in ("SubagentHandback", "TodoWrite"):
                bad.append(f"{name} {json.dumps(inp)[:110]}")
    worst |= bool(bad)
    print(f"{aid[:8]}: {len(files)} file touches inside {FOLDER}; "
          + ("OUT OF BOUNDS: " + "; ".join(bad[:3]) if bad else "stayed in bounds"))
sys.exit(worst)
