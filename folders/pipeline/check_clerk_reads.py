"""Any sign a filing subagent read something other than clerk_step.py's output?   python check_clerk_reads.py AGENT_ID ...

Looks in every Bash command for file-reading words (cat, ls, grep, find, head/tail on a file,
open(, Path(, a .json other than reply.json, data folders) and for python reading anything but
stdin. Prints the suspicious commands' tails, or "clean".
"""
import json
import re
import sys
from pathlib import Path

import os
D = Path(os.environ["CLAUDE_SUBAGENTS_DIR"])  # the Claude Code session's subagents/ transcript folder
SUSPECT = re.compile(r"\b(cat|ls|dir|grep|rg|find|type|more|less|stat)\b|open\(|Path\(|read_text|glob|"
                     r"state\.json|voted|test\.json|kaggle|realtalk|folders[\\/]|folders_b1|\.out\b|"
                     r"(head|tail) +(-\S+ +)?[\w./\\-]+\.(json|py|txt)", re.I)

for aid in sys.argv[1:]:
    hits = []
    for line in open(D / f"agent-{aid}.jsonl", encoding="utf-8"):
        c = json.loads(line).get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for x in c:
            if x.get("type") == "tool_use" and x["name"] == "Bash":
                cmd = x["input"].get("command", "")
                body = cmd.split("clerk_step.py", 1)[-1]          # ignore the cd + the tool's own path
                if SUSPECT.search(body):
                    hits.append(body[-200:])
    print(aid[:6], "clean" if not hits else f"{len(hits)} suspicious: {hits[:4]}")
