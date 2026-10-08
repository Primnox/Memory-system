"""Walk the web of folders to build what the model reads.   python folder_walk.py <set>

Everything is linked: a fact sits in every folder it touches and on its day; folders link
through the facts they share. For each question three bundles are built to the same size
(tokens ~ words x 1.3), and scored on how many of the question's answer facts they carry:
  TOP    plain similarity, best facts until the budget is spent
  INDEX  the index ranking (folder_index.py T+N), best facts until the budget is spent
  WALK   the index's top 3 facts in full; then the card of each folder they touch; then the
         cards of folders one link away; then more index-ranked facts to fill the budget.
         A card: "[Sarah | Bridget's sister]" + the value of each one-value slot as it stood
         at the question's time + the 2 facts of the folder closest to the question. The
         speaker's own card only lists slots among the index's top 10. One-link-away cards
         are short: name, link and up to 2 one-value slot values.
covered: share of a question's answer facts inside the bundle; all: every answer inside.
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

BUDGETS = (150, 300, 600)


def toks(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


def main(name: str) -> None:
    enc = Encoder()
    sts = states(name)
    res = {m: {b: [0.0, 0, 0] for b in BUDGETS} for m in ("TOP", "INDEX", "WALK")}
    by_type = {}
    for sc in load(name):
        st = sts.get(sc["scenario"])
        qs = [q for q in sc["questions"] if q.get("answer_ids") and q["type"] != "absent"]
        if st is None or not qs:
            continue
        facts = sc["statements"]
        n = len(facts)
        fid = {f["id"]: f for f in facts}
        idx = build_index(sc, st)
        fa = enc([f["text"] for f in facts])
        ff = enc([filed_text(f, st) for f in facts])
        ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
        df = Counter(x for s in ffold for x in s)
        w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
        speaker = {k for k, v in st["folders"].items() if v.get("type") == "speaker"}
        # folder -> folders it shares facts with, most shared first
        near = {}
        for s in ffold:
            for a in s:
                for b in s - {a}:
                    near.setdefault(a, Counter())[b] += 1
        qv = enc([q["text"] for q in qs])
        today = date.fromisoformat(sc["today"])
        for qi, q in enumerate(qs):
            want = set(q["answer_ids"])
            qf = named_folders(q["text"], st)
            sim_a = (fa @ qv[qi]).tolist()
            sim_f = (ff @ qv[qi]).tolist()
            tb, sb, past = walk(q, sc, st, idx, facts)
            sc_t = [s + ALPHA * sum(w[x] for x in qf & fo) + TIME * t - STALE * z
                    for s, fo, t, z in zip(sim_f, ffold, tb, sb)]
            top = [facts[i]["id"] for i in sorted(range(n), key=lambda i: -sim_a[i])]
            ind = [facts[i]["id"] for i in sorted(range(n), key=lambda i: -sc_t[i])]
            rank = {f: r for r, f in enumerate(ind)}
            span = when.resolve(q["text"], today) if past else None
            cutoff = span.end.isoformat() if span else "9999"

            def slot_value(slot: str) -> str | None:
                ids = [i for i in idx_by_slot.get(slot, []) if fid[i]["date"] <= cutoff]
                return ids[-1] if ids else None

            idx_by_slot = {}
            for f in sorted(facts, key=lambda f: f["date"]):
                idx_by_slot.setdefault(st["facts"][f["id"]]["slot"], []).append(f["id"])

            def card(folder: str, full: bool, top10_slots: set[str]) -> list[str]:
                f = st["folders"][folder]
                lines = [f"[{f['name']} | {f.get('link') or f.get('type')}]"]
                slots = [s for s in idx["folders"].get(folder, {}) if s in idx["one"]]
                if folder in speaker:
                    slots = [s for s in slots if s in top10_slots]
                ids = [v for v in (slot_value(s) for s in slots) if v][: (None if full else 2)]
                if full:
                    mine = [i for i in ind if folder in set(st["facts"][i]["folders"]) and i not in ids]
                    ids += mine[:2]
                return [lines[0]] + ids

            def fill(order: list[str], b: int, start: list = ()) -> set[str]:
                used, got = 0, set()
                for item in list(start) + order:
                    text = fid[item]["text"] if item in fid else item
                    if item in got:
                        continue
                    if used + toks(text) > b:
                        if item in fid:
                            continue
                        break
                    used += toks(text)
                    got.add(item)
                return {g for g in got if g in fid}

            seeds = ind[:3]
            top10_slots = {st["facts"][i]["slot"] for i in ind[:10]}
            plan = list(seeds)
            touched = []
            for i in seeds:
                for fo in st["facts"][i]["folders"]:
                    if fo not in touched:
                        touched.append(fo)
            for fo in touched:
                plan += card(fo, True, top10_slots)
            hops = Counter()
            for fo in touched:
                hops.update({k: v for k, v in near.get(fo, {}).items() if k not in touched and k not in speaker})
            for fo, _ in hops.most_common(3):
                plan += card(fo, False, top10_slots)
            plan += ind
            for b in BUDGETS:
                for m, order in (("TOP", top), ("INDEX", ind), ("WALK", plan)):
                    got = fill(order, b)
                    c = len(got & want) / len(want)
                    res[m][b][0] += c
                    res[m][b][1] += c == 1.0
                    res[m][b][2] += 1
                    by_type.setdefault((q["type"], m, b), []).append(c)
    k = res["TOP"][BUDGETS[0]][2]
    print(f"{name}: {k} questions; answer facts carried to the model (covered % / all answers %)")
    print(f"{'budget':>8s}" + "".join(f"{m:>18s}" for m in ("TOP", "INDEX", "WALK")))
    for b in BUDGETS:
        print(f"{b:>8d}" + "".join(f"{100 * res[m][b][0] / k:9.1f} / {100 * res[m][b][1] / k:5.1f}" for m in ("TOP", "INDEX", "WALK")))
    print("\nby question type, covered %:")
    types = sorted({t for t, _, _ in by_type})
    for t in types:
        cells = []
        for b in BUDGETS:
            cells.append(" ".join(f"{100 * sum(by_type[(t, m, b)]) / len(by_type[(t, m, b)]):5.1f}" for m in ("TOP", "INDEX", "WALK")))
        print(f"  {t:12s} ({len(by_type[(t, 'TOP', BUDGETS[0])])})  " + "   |   ".join(cells))
    print("  (each budget: TOP INDEX WALK)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
