"""Retired memories are kept; the QUESTION decides whether they count.   python question_modes.py <set> <retired.json>

  dated    the question names a time the app's when.py can read ("in March 2025") -> as things
           stood then (stated by then, not yet retired by then)
  history  past tense or "ever / how many times / when did" and no readable time -> every fact,
           retired or not ("what pasta did she make", "has she ever been to Bali")
  now      everything else -> standing facts only (today's app)
Rule fixed here, before any run. Plain similarity ranks (gte-small); compared with today's app,
which drops retired facts for every undated question. top-1 / top-3 / R@10, %.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folder_index import when      # noqa: E402
from folders import load           # noqa: E402
from graph_search import Encoder   # noqa: E402

HISTORY = re.compile(r"\b(did|was|were|had|used to|ever|how many times|how often has|in the past|before|previously|"
                     r"has \w+ (?:been|gone|visited|tried|made|done|had)|have \w+ (?:been|gone|visited|tried))\b", re.I)


def mode(q: str, today: date) -> tuple[str, str | None]:
    m = when.as_of_for(query=q, today=today, pick="end")
    if m:
        return "dated", m
    return ("history", None) if HISTORY.search(q) else ("now", None)


def main(name: str, src: str) -> None:
    enc = Encoder()
    ret = json.loads(Path(src).read_text(encoding="utf-8"))
    rows, modes = [], {}
    for sc in load(name):
        facts = sc["statements"]
        day = {f["id"]: f["date"] for f in facts}
        by = {o: n for o, n, _ in ret.get(sc["scenario"], [])}
        qs = [q for q in sc["questions"] if q.get("answer_ids") and q["type"] != "absent"]
        if not qs:
            continue
        today = date.fromisoformat(sc["today"])
        fa = enc([f["text"] for f in facts])
        qv = enc([q["text"] for q in qs])
        for qi, q in enumerate(qs):
            md, moment = mode(q["text"], today)
            modes[(q["type"], md)] = modes.get((q["type"], md), 0) + 1
            app_m = moment or sc["today"]
            app = [day[f["id"]] <= app_m and not (f["id"] in by and day[by[f["id"]]] <= app_m) for f in facts]
            if md == "history":
                new = [True] * len(facts)
            else:
                new = app
            sa = (fa @ qv[qi]).tolist()
            r = {"type": q["type"], "want": set(q["answer_ids"]), "ids": [f["id"] for f in facts]}
            r["APP"] = [s if a else -9 for s, a in zip(sa, app)]
            r["MODES"] = [s if a else -9 for s, a in zip(sa, new)]
            r["PLAIN"] = sa
            rows.append(r)

    def score(rs, key):
        t1 = t3 = r10 = 0.0
        for r in rs:
            order = [r["ids"][i] for i in sorted(range(len(r[key])), key=lambda i: -r[key][i])]
            t1 += order[0] in r["want"]
            t3 += bool(set(order[:3]) & r["want"])
            r10 += len(set(order[:10]) & r["want"]) / len(r["want"])
        k = max(len(rs), 1)
        return f"{100 * t1 / k:5.1f} / {100 * t3 / k:5.1f} / {100 * r10 / k:5.1f}"

    print(f"{name}, retirements from {Path(src).name}; question modes by type: {modes}")
    print(f"{'questions':16s} {'PLAIN (no head)':>20s} {'APP (head, today)':>20s} {'MODES':>20s}")
    for t in ["all"] + sorted({r["type"] for r in rows}):
        rs = rows if t == "all" else [r for r in rows if r["type"] == t]
        print(f"{t + f' ({len(rs)})':16s} " + " ".join(f"{score(rs, k):>20s}" for k in ("PLAIN", "APP", "MODES")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
