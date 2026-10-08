"""Why does the folder index miss? Dev set only (chat people).   python index_misses.py

For every question the index (T+N) does not rank first, sort the miss:
  now   - top-1 is an OLDER fact in the same slot as an answer (should have been ranked down)
        - top-1 is an older fact replaced by an answer but filed in ANOTHER slot
        - the answer itself was ranked down as stale (wrongly)
        - other
  past  - no time words understood / top-1 dated after the span / answer outside span & not
          the slot's standing value / other
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folder_eval import ALPHA, filed_text, named_folders, states   # noqa: E402
from folder_index import STALE, TIME, build_index, walk, when      # noqa: E402
from folders import load                                           # noqa: E402
from graph_search import Encoder                                   # noqa: E402

enc = Encoder()
sts = states("chat")
why = Counter()
examples = {}
for sc in load("chat"):
    st = sts[sc["scenario"]]
    facts = sc["statements"]
    n = len(facts)
    fid = {f["id"]: f for f in facts}
    replaced_by = {x: f["id"] for f in facts for x in f.get("replaces") or []}
    idx = build_index(sc, st)
    ff = enc([filed_text(f, st) for f in facts])
    ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
    df = Counter(x for s in ffold for x in s)
    w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
    qs = [q for q in sc["questions"] if q.get("answer_ids") and q["type"] != "absent"]
    qv = enc([q["text"] for q in qs])
    for qi, q in enumerate(qs):
        want = set(q["answer_ids"])
        qf = named_folders(q["text"], st)
        tb, sb, past = walk(q, sc, st, idx, facts)
        s = [x + ALPHA * sum(w[y] for y in qf & fo) + TIME * t - STALE * z
             for x, fo, t, z in zip((ff @ qv[qi]).tolist(), ffold, tb, sb)]
        order = sorted(range(n), key=lambda i: -s[i])
        top = facts[order[0]]["id"]
        if top in want:
            continue
        slot = lambda i: st["facts"][i]["slot"]
        kind = "now" if q["type"] in ("current", "multi") else "past"
        if kind == "now":
            if any(slot(top) == slot(a) and fid[top]["date"] < fid[a]["date"] for a in want):
                r = "now: older fact in the SAME slot ranked first"
            elif replaced_by.get(top) in want or (replaced_by.get(top) and replaced_by.get(top) not in want):
                r = "now: top-1 was replaced (label) but filed in ANOTHER slot / many-value slot"
            elif any(sb[[f['id'] for f in facts].index(a)] for a in want):
                r = "now: the answer itself was ranked down as stale"
            else:
                r = "now: other (similarity picked a different fact)"
        else:
            span = when.resolve(q["text"], date.fromisoformat(sc["today"]))
            if not past:
                r = "past: time words not understood"
            elif fid[top]["date"] > span.end.isoformat():
                r = "past: top-1 dated after the span"
            elif not any(tb[[f['id'] for f in facts].index(a)] for a in want):
                r = "past: answer got no time bonus (outside span, not the slot's standing value)"
            else:
                r = "past: other"
        why[r] += 1
        examples.setdefault(r, []).append((q["text"], fid[top]["text"], [fid[a]["text"] for a in want][:2],
                                           slot(top), [slot(a) for a in want][:2]))
total = sum(why.values())
print(f"index misses at top-1 on dev: {total}")
for r, c in why.most_common():
    print(f"  {c:4d}  {r}")
for r, ex in examples.items():
    print(f"\n== {r}")
    for e in ex[:3]:
        print(f"  Q: {e[0]}\n    top1 [{e[3]}]: {e[1]}\n    want {e[4]}: {e[2]}")
