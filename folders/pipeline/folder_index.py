"""The index layer over the folders: a main index, a days index and a card per folder, and
search that walks it.   python folder_index.py <set>

Built from the filing (folders.py) and each fact's date; no labels. A question is walked:
  time words  -> the app's own when.py (memory/when.py) gives the span of days they mean
                 ("yesterday", "back in November", "last summer"), relative to the set's `today`
  names       -> folders whose name or alias the question uses ("my sister" -> Sarah/)
  slots       -> a ONE-value slot (lives_in, employer, phone) holds one fact at a time
Rankings (all on the folder-card text, F, from folder_eval.py):
  F+R   folder cards + named-folder bonus (the last test's design)
  T     F+R, and for a question about a past span: + TIME for every fact dated inside the
        span, and for each one-value slot the fact that was standing at the span's end
        ("where was I living in November" -> lives_in as it stood then)
  T+N   T, and for a question about now: - STALE for a fact whose one-value slot has a later
        fact (it no longer stands); nothing is removed, only ranked down
TIME and STALE are chosen on the dev set (chat) only, then fixed for the proof sets.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folder_eval import ALPHA, filed_text, named_folders, states   # noqa: E402
from folders import load                                           # noqa: E402
from graph_search import Encoder                                   # noqa: E402

# the memory's own date reader, backend/primnox2/memory/when.py in this repo
_spec = importlib.util.spec_from_file_location(
    "when", Path(__file__).resolve().parents[2] / "backend" / "primnox2" / "memory" / "when.py")
when = importlib.util.module_from_spec(_spec)
sys.modules["when"] = when
_spec.loader.exec_module(when)

GRID = (0.05, 0.1, 0.2)
TIME = 0.05          # chosen on the dev set (chat, 24 people, 586 questions): best top-3 on all; fixed before proof runs
STALE = 0.05


def build_index(sc: dict, st: dict) -> dict:
    """days -> facts; folder -> slot -> facts (date order); which slots hold one value."""
    days, folders = {}, {}
    for f in sorted(sc["statements"], key=lambda f: f["date"]):
        e = st["facts"][f["id"]]
        days.setdefault(f["date"], []).append(f["id"])
        for fid in e["folders"]:
            folders.setdefault(fid, {}).setdefault(e["slot"], []).append(f["id"])
    one = {k for k, v in st["slots"].items() if v == "one"}
    return {"days": days, "folders": folders, "one": one}


def walk(q: dict, sc: dict, st: dict, idx: dict, facts: list[dict]) -> tuple[list[float], list[float], bool]:
    """Per fact: the time bonus (0/1), the stale mark (0/1), and whether the question is about the past."""
    today = date.fromisoformat(sc["today"])
    span = when.resolve(q["text"], today)
    past = span is not None and span.end < today
    by_id = {f["id"]: f for f in facts}
    slot_of = {f["id"]: st["facts"][f["id"]]["slot"] for f in facts}
    by_slot = {}
    for f in sorted(facts, key=lambda f: f["date"]):
        by_slot.setdefault(slot_of[f["id"]], []).append(f["id"])
    time_hit, stale = set(), set()
    if past:
        lo, hi = span.start.isoformat(), span.end.isoformat()
        for d, ids in idx["days"].items():
            if lo <= d <= hi:
                time_hit.update(ids)
        for slot in idx["one"]:
            before = [i for i in by_slot.get(slot, []) if by_id[i]["date"] <= hi]
            if before:
                time_hit.add(before[-1])
    else:
        for slot in idx["one"]:
            ids = by_slot.get(slot, [])
            last = ids[-1] if ids else None
            stale.update(i for i in ids if i != last and by_id[i]["date"] < by_id[last]["date"])
    return ([1.0 if f["id"] in time_hit else 0.0 for f in facts],
            [1.0 if f["id"] in stale else 0.0 for f in facts], past)


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
        idx = build_index(sc, st)
        ff = enc([filed_text(f, st) for f in facts])
        ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
        df = Counter(x for s in ffold for x in s)
        w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
        qv = enc([q["text"] for q in qs])
        for qi, q in enumerate(qs):
            qf = named_folders(q["text"], st)
            base = [s + ALPHA * sum(w[x] for x in qf & fo) for s, fo in zip((ff @ qv[qi]).tolist(), ffold)]
            tb, sb, past = walk(q, sc, st, idx, facts)
            rows.append({"type": q["type"], "want": set(q["answer_ids"]), "ids": [f["id"] for f in facts],
                         "base": base, "time": tb, "stale": sb, "past": past})
    n_past = sum(r["past"] for r in rows)
    print(f"{name}: {len(rows)} questions in {len(sts)} people; {n_past} name a past time the code understood")

    def score(rs, t, s):
        t1 = t3 = r10 = 0.0
        for r in rs:
            sc_ = [b + t * x - s * y for b, x, y in zip(r["base"], r["time"], r["stale"])]
            order = [r["ids"][i] for i in sorted(range(len(sc_)), key=lambda i: -sc_[i])]
            t1 += order[0] in r["want"]
            t3 += bool(set(order[:3]) & r["want"])
            r10 += len(set(order[:10]) & r["want"]) / len(r["want"])
        k = max(len(rs), 1)
        return f"{100 * t1 / k:5.1f} / {100 * t3 / k:5.1f} / {100 * r10 / k:5.1f}"

    settings = [(t, s) for t in GRID for s in GRID] if TIME is None else [(TIME, STALE)]
    groups = ["all"] + sorted({r["type"] for r in rows}) + ["past-time words"]
    for t, s in settings:
        print(f"\nTIME = {t}, STALE = {s}   (top-1 / top-3 / R@10, %)")
        print(f"{'questions':22s} {'F+R':>20s} {'T':>20s} {'T+N':>20s}")
        for g in groups:
            rs = rows if g == "all" else [r for r in rows if r["past"]] if g == "past-time words" else \
                [r for r in rows if r["type"] == g]
            print(f"{g + f' ({len(rs)})':22s} {score(rs, 0, 0):>20s} {score(rs, t, 0):>20s} {score(rs, t, s):>20s}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
