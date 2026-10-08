"""The app's own approach at its ceiling, for comparison with the folder index.   python oracle_app.py <set>

The app answers "now" by retiring replaced memories (the head) and "back then" by searching
as of a date (when.py -> search(as_of=...)). Here both are PERFECT, taken from the labels:
  now   questions: facts replaced on or before `today` are removed
  past  questions: facts stated after the question's labelled `as_of` are removed, and so
                   are facts already replaced by then
then plain similarity ranks what is left (APP*). Beside it: the folder index (T+N, from the
clerk's filing, no labels), and APP* ranked on folder cards instead of bare facts (the head
and the folders together). top-1 / top-3 / R@10, %.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folder_eval import ALPHA, filed_text, named_folders, states   # noqa: E402
from folder_index import STALE, TIME, build_index, walk            # noqa: E402
from folders import load                                           # noqa: E402
from graph_search import Encoder                                   # noqa: E402


def main(name: str) -> None:
    enc = Encoder()
    sts = states(name)
    rows = []
    for sc in load(name):
        st = sts.get(sc["scenario"])
        qs = [q for q in sc["questions"] if q.get("answer_ids") and q["type"] != "absent"]
        if st is None or not qs:
            continue
        facts = sc["statements"]
        n = len(facts)
        date_of = {f["id"]: f["date"] for f in facts}
        replaced_on = {}
        for f in facts:
            for x in f.get("replaces") or []:
                replaced_on.setdefault(x, f["date"])
        idx = build_index(sc, st)
        fa = enc([f["text"] for f in facts])
        ff = enc([filed_text(f, st) for f in facts])
        ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
        df = Counter(x for s in ffold for x in s)
        w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
        qv = enc([q["text"] for q in qs])
        for qi, q in enumerate(qs):
            moment = q.get("as_of") or sc["today"]
            alive = [date_of[f["id"]] <= moment and not (f["id"] in replaced_on and replaced_on[f["id"]] <= moment)
                     for f in facts]
            sa = (fa @ qv[qi]).tolist()
            app = [s if ok else -9 for s, ok in zip(sa, alive)]
            qf = named_folders(q["text"], st)
            tb, sb, _ = walk(q, sc, st, idx, facts)
            ind = [s + ALPHA * sum(w[x] for x in qf & fo) + TIME * t - STALE * z
                   for s, fo, t, z in zip((ff @ qv[qi]).tolist(), ffold, tb, sb)]
            cards = [s + ALPHA * sum(w[x] for x in qf & fo) for s, fo in zip((ff @ qv[qi]).tolist(), ffold)]
            both = [c if ok else -9 for c, ok in zip(cards, alive)]
            rows.append({"type": q["type"], "want": set(q["answer_ids"]), "ids": [f["id"] for f in facts],
                         "PLAIN": sa, "APP*": app, "INDEX": ind, "APP*+cards": both})

    def score(rs, key):
        t1 = t3 = r10 = 0.0
        for r in rs:
            order = [r["ids"][i] for i in sorted(range(len(r[key])), key=lambda i: -r[key][i])]
            t1 += order[0] in r["want"]
            t3 += bool(set(order[:3]) & r["want"])
            r10 += len(set(order[:10]) & r["want"]) / len(r["want"])
        k = max(len(rs), 1)
        return f"{100 * t1 / k:5.1f} / {100 * t3 / k:5.1f} / {100 * r10 / k:5.1f}"

    print(f"{name}: {len(rows)} questions; APP* = perfect head + perfect date filter (from labels); INDEX = folders, no labels")
    print(f"{'questions':16s} {'PLAIN':>20s} {'APP* (ceiling)':>20s} {'INDEX':>20s} {'APP* + cards':>20s}")
    for t in ["all"] + sorted({r["type"] for r in rows}):
        rs = rows if t == "all" else [r for r in rows if r["type"] == t]
        print(f"{t + f' ({len(rs)})':16s} " + " ".join(f"{score(rs, k):>20s}" for k in ("PLAIN", "APP*", "INDEX", "APP*+cards")))


if __name__ == "__main__":
    main(sys.argv[1])
