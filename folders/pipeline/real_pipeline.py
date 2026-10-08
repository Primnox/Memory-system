"""Search the way the app would, with a real head's retirements, and with the folder design on top.

    python real_pipeline.py <set> <retired.json | oracle> [<retired.json> ...]

retired.json: {scenario: [[old id, new id, p], ...]} from primnox-head-sweep (or "oracle":
the labels' replacements, as a ceiling). Nothing else uses labels. For each question:
  moment   the app's own when.py reads the time words in the QUESTION ("back in March") and
           gives the day to search as of (memory/when.py as_of_for, the middle of the span);
           no time words, or a span reaching today -> as things stand now
  live     facts stated by that moment and not yet retired by then (a retirement counts from
           the date of the fact that caused it), the app's search(as_of=...) rule
Rankings over the live facts:
  APP      plain similarity (gte-small), what the app's semantic search does
  CARDS    the folder cards (folder_eval.py F+R)
  CARDS+E  CARDS, and when the question dates itself by an EVENT the code cannot read as a
           date ("before I bought the iPhone 14", "back when I was at LAAS", "after I retired
           the Trek"): the fact closest to that clause is the anchor and its date the moment
           (before -> the day before it; when/while/after -> its own date)
top-1 / top-3 / R@10, %. Settings: ALPHA from the dev set; the event rule is fixed here.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folder_eval import ALPHA, filed_text, named_folders, states   # noqa: E402
from folder_index import when                                      # noqa: E402
from folders import load                                           # noqa: E402
from graph_search import Encoder                                   # noqa: E402

PICK = __import__("os").environ.get("PICK", "end")   # where in a named period to search as of: "end" beat "mid" on the dev set (past top-1 68.6 vs 65.5 with a perfect head)
EVENT = re.compile(r"\b(before|after|back when|when|while)\s+(?:i|she|he|they|we|you|her|his|my)\b(.+?)(?:[?,.]|$)", re.I)


def retirements(name: str, src: str) -> dict[str, dict[str, str]]:
    """{scenario: {retired id: id of the fact that retired it}}"""
    if src == "oracle":
        out = {}
        for sc in load(name):
            out[sc["scenario"]] = {x: f["id"] for f in sc["statements"] for x in f.get("replaces") or []}
        return out
    raw = json.loads(Path(src).read_text(encoding="utf-8"))
    return {k: {o: n for o, n, _ in v} for k, v in raw.items()}


def main(name: str, sources: list[str]) -> None:
    enc = Encoder()
    sts = states(name)
    for src in sources:
        ret = retirements(name, src)
        rows = []
        for sc in load(name):
            st = sts.get(sc["scenario"])
            qs = [q for q in sc["questions"] if q.get("answer_ids") and q["type"] != "absent"]
            if st is None or not qs or sc["scenario"] not in ret:
                continue
            facts = sc["statements"]
            n = len(facts)
            day = {f["id"]: f["date"] for f in facts}
            by = ret[sc["scenario"]]
            today = date.fromisoformat(sc["today"])
            fa = enc([f["text"] for f in facts])
            ff = enc([filed_text(f, st) for f in facts])
            ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
            df = Counter(x for s in ffold for x in s)
            w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
            qv = enc([q["text"] for q in qs])

            def alive(moment: str) -> list[bool]:
                return [day[f["id"]] <= moment and not (f["id"] in by and day[by[f["id"]]] <= moment) for f in facts]

            for qi, q in enumerate(qs):
                moment = when.as_of_for(query=q["text"], today=today, pick=PICK) or sc["today"]
                moment_e = moment
                if moment == sc["today"]:
                    m = EVENT.search(q["text"])
                    if m:
                        av = enc([m.group(2).strip()])[0]
                        anchor = max(range(n), key=lambda i: float(fa[i] @ av))
                        d = date.fromisoformat(day[facts[anchor]["id"]])
                        moment_e = (d - timedelta(days=1) if m.group(1).lower() == "before" else d).isoformat()
                qf = named_folders(q["text"], st)
                sa = (fa @ qv[qi]).tolist()
                cards = [s + ALPHA * sum(w[x] for x in qf & fo) for s, fo in zip((ff @ qv[qi]).tolist(), ffold)]
                ok, ok_e = alive(moment), alive(moment_e)
                rows.append({"type": q["type"], "want": set(q["answer_ids"]), "ids": [f["id"] for f in facts],
                             "APP": [s if a else -9 for s, a in zip(sa, ok)],
                             "CARDS": [s if a else -9 for s, a in zip(cards, ok)],
                             "CARDS+E": [s if a else -9 for s, a in zip(cards, ok_e)]})

        def score(rs, key):
            t1 = t3 = r10 = 0.0
            for r in rs:
                order = [r["ids"][i] for i in sorted(range(len(r[key])), key=lambda i: -r[key][i])]
                t1 += order[0] in r["want"]
                t3 += bool(set(order[:3]) & r["want"])
                r10 += len(set(order[:10]) & r["want"]) / len(r["want"])
            k = max(len(rs), 1)
            return f"{100 * t1 / k:5.1f} / {100 * t3 / k:5.1f} / {100 * r10 / k:5.1f}"

        print(f"\n{name}, retirements from {Path(src).name}: {len(rows)} questions (top-1 / top-3 / R@10, %)")
        print(f"{'questions':16s} {'APP':>20s} {'CARDS':>20s} {'CARDS+E':>20s}")
        for t in ["all"] + sorted({r["type"] for r in rows}):
            rs = rows if t == "all" else [r for r in rows if r["type"] == t]
            print(f"{t + f' ({len(rs)})':16s} " + " ".join(f"{score(rs, k):>20s}" for k in ("APP", "CARDS", "CARDS+E")))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
