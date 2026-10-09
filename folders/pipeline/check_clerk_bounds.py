"""Did each filing subagent touch only clerk_step.py and its own reply file?   python check_clerk_bounds.py AGENT_ID ...

Prints tool counts and anything outside: a Bash command that is not a clerk_step.py call, or a
Read/Write/Edit of any file other than clerk_tmp/aN/reply.json. Names only, never content.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import os
D = Path(os.environ["CLAUDE_SUBAGENTS_DIR"])  # the Claude Code session's subagents/ transcript folder
REPLY = re.compile(r"clerk_tmp[\\/]a\d[\\/]reply\.json$")
STEP = re.compile(r"^(cd \S+ && )?(PYTHONIOENCODING=utf-8 )?python clerk_step\.py "
                  r"(spec|next (batch|single) \w+ [\w\-]+|submit (batch|single) \w+ [\w\-]+ clerk_tmp/a\d/reply\.json)"
                  r"( 2>&1)?( \| head -\d+)?$")

for aid in sys.argv[1:]:
    tools, bad = Counter(), []
    for line in open(D / f"agent-{aid}.jsonl", encoding="utf-8"):
        c = json.loads(line).get("message", {}).get("content")
        if not isinstance(c, list):
            continue
        for x in c:
            if x.get("type") != "tool_use":
                continue
            name, inp = x["name"], x.get("input", {})
            tools[name] += 1
            if name == "Bash":
                cmd = inp.get("command", "").strip().replace('"', "")
                if not STEP.match(cmd):
                    bad.append("Bash: ..." + cmd[-160:])
            elif name in ("Write", "Read", "Edit"):
                if not REPLY.search(inp.get("file_path", "")):
                    bad.append(f"{name}: {inp.get('file_path', '')[-100:]}")
            elif name != "SubagentHandback":
                bad.append(f"{name}")
    print(aid[:6], dict(tools), "-> all in bounds" if not bad else f"-> {len(bad)} to look at: {bad[:5]}")
