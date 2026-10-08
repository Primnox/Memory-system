"""Does organising memory into folders and slots help?   python folder_eval.py search <set> | changes <set>

Uses the filing from folders.py (folders/<set>/*/state.json). No labels go into the filing.

search   every question with an answer (absent questions left out), ranked over ALL of the
         person's facts (nothing retired), gte-small cosine (the app's search encoder):
           A    plain similarity on the fact
           F    similarity on the fact filed with its folders: "Ana (Maya's sister) | job | <fact>"
           A+R  A, plus ALPHA * rarity weight of folders the question names (name or alias,
                so "my sister" reaches Ana's facts that never say "sister")
           F+R  both
         top-1 / top-3: an answer fact ranked first / in the top 3; R@10: share of the
         question's answer facts in the top 10. ALPHA is chosen on the dev set (chat) only.
changes  every labelled replacement (new fact -> the older fact it replaced). Pool = facts
         stated before it and not yet replaced. Is the replaced fact among the first B
         candidates?
           SIM   most similar first (today's head shortlist, without its name links)
           SLOT  same slot first (newest first), then facts sharing a folder other than the
                 speaker's, most similar first, then the rest most similar first
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folders import OUT, load                     # noqa: E402
from graph_search import Encoder                  # noqa: E402

ALPHA_GRID = (0.05, 0.1, 0.2, 0.3)
ALPHA = 0.05        # chosen on the dev set (chat, 23 people, 561 questions): best F+R top-3; fixed before any proof run
BUDGETS = (1, 3, 5, 10, 15)
PRONOUNS = {"i", "me", "my", "the user", "mine", "myself"}


def states(name: str) -> dict[str, dict]:
    out = {}
    for sc in load(name):
        f = OUT / name / re.sub(r"[^\w\-]", "_", sc["scenario"]) / "state.json"
        if f.exists():
            st = json.loads(f.read_text(encoding="utf-8"))
            if len(st["facts"]) == len(sc["statements"]):
                out[sc["scenario"]] = st
    return out


def filed_text(fact: dict, st: dict) -> str:
    e = st["facts"][fact["id"]]
    heads = []
    for fid in e["folders"]:
        f = st["folders"].get(fid)
        if f:
            heads.append(f"{f['name']} ({f['link']})" if f.get("link") else f["name"])
    slot = e["slot"].split(".", 1)[-1].replace("_", " ")
    return f"{'; '.join(heads)} | {slot} | {fact['text']}"


def named_folders(question: str, st: dict) -> set[str]:
    q = " " + re.sub(r"[^\w']+", " ", question.lower()) + " "
    hit = set()
    for fid, f in st["folders"].items():
        for a in [f["name"]] + list(f.get("aliases") or []):
            a = re.sub(r"[^\w']+", " ", str(a).lower()).strip()
            if len(a) >= 3 and a not in PRONOUNS and f" {a} " in q:
                hit.add(fid)
                break
    return hit


def search(name: str) -> None:
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
        fa = enc([f["text"] for f in facts])
        ff = enc([filed_text(f, st) for f in facts])
        ffold = [set(st["facts"][f["id"]]["folders"]) for f in facts]
        df = Counter(x for s in ffold for x in s)
        w = {x: math.log(n / c) / math.log(n) for x, c in df.items()}
        qv = enc([q["text"] for q in qs])
        for qi, q in enumerate(qs):
            want = set(q["answer_ids"])
            qf = named_folders(q["text"], st)
            route = [sum(w[x] for x in qf & s) for s in ffold]
            sa, sf = (fa @ qv[qi]).tolist(), (ff @ qv[qi]).tolist()
            row = {"type": q["type"], "want": want, "ids": [f["id"] for f in facts],
                   "A": sa, "F": sf, "route": route}
            rows.append(row)
    print(f"{name}: {len(rows)} questions in {len(sts)} people")

    def score(rs, key, alpha):
        t1 = t3 = r10 = 0.0
        for r in rs:
            base = r["A" if key in ("A", "A+R") else "F"]
            s = [b + (alpha * x if key.endswith("+R") else 0.0) for b, x in zip(base, r["route"])]
            order = [r["ids"][i] for i in sorted(range(len(s)), key=lambda i: -s[i])]
            t1 += order[0] in r["want"]
            t3 += bool(set(order[:3]) & r["want"])
            r10 += len(set(order[:10]) & r["want"]) / len(r["want"])
        k = len(rs)
        return f"{100 * t1 / k:5.1f} / {100 * t3 / k:5.1f} / {100 * r10 / k:5.1f}"

    alphas = ALPHA_GRID if ALPHA is None else (ALPHA,)
    for alpha in alphas:
        print(f"\nALPHA = {alpha}   (top-1 / top-3 / R@10, %)")
        print(f"{'questions':18s} {'A':>20s} {'F':>20s} {'A+R':>20s} {'F+R':>20s}")
        for t in ["all"] + sorted({r["type"] for r in rows}):
            rs = rows if t == "all" else [r for r in rows if r["type"] == t]
            print(f"{t + f' ({len(rs)})':18s} " + " ".join(f"{score(rs, k, alpha):>20s}" for k in ("A", "F", "A+R", "F+R")))


def changes(name: str) -> None:
    enc = Encoder()
    sts = states(name)
    hits = {m: Counter() for m in ("SIM", "SLOT")}
    total = same_slot = share_folder = pools = 0
    for sc in load(name):
        st = sts.get(sc["scenario"])
        if st is None:
            continue
        facts = sc["statements"]
        vec = enc([f["text"] for f in facts])
        speaker = {k for k, v in st["folders"].items() if v.get("type") == "speaker"}
        replaced_at = {}
        for i, f in enumerate(facts):
            for x in f.get("replaces") or []:
                replaced_at.setdefault(x, i)
        pos = {f["id"]: i for i, f in enumerate(facts)}
        for i, f in enumerate(facts):
            for x in f.get("replaces") or []:
                if x not in pos or pos[x] >= i:
                    continue
                pool = [j for j in range(i) if replaced_at.get(facts[j]["id"], i) >= i]
                if pos[x] not in pool:
                    continue
                total += 1
                pools += len(pool)
                sims = {j: float(vec[i] @ vec[j]) for j in pool}
                me = st["facts"][f["id"]]
                old = st["facts"][x]
                same_slot += me["slot"] == old["slot"]
                share_folder += bool((set(me["folders"]) & set(old["folders"])) - speaker)
                by_sim = sorted(pool, key=lambda j: -sims[j])
                slot = [j for j in pool if st["facts"][facts[j]["id"]]["slot"] == me["slot"]][::-1]
                mine = set(me["folders"]) - speaker
                near = [j for j in by_sim if j not in slot and set(st["facts"][facts[j]["id"]]["folders"]) & mine]
                rest = [j for j in by_sim if j not in slot and j not in near]
                for m, order in (("SIM", by_sim), ("SLOT", slot + near + rest)):
                    for b in BUDGETS:
                        hits[m][b] += pos[x] in order[:b]
    print(f"{name}: {total} replacements, average pool {pools / max(total, 1):.0f} earlier facts")
    print(f"  replaced fact filed in the SAME SLOT as the new one: {same_slot}/{total} ({100 * same_slot / max(total, 1):.0f}%)")
    print(f"  shares a folder other than the speaker's:           {share_folder}/{total}")
    print(f"  replaced fact among the first B candidates:")
    print("  " + f"{'B':>10s}" + "".join(f"{b:>8d}" for b in BUDGETS))
    for m in ("SIM", "SLOT"):
        print("  " + f"{m:>10s}" + "".join(f"{100 * hits[m][b] / max(total, 1):7.0f}%" for b in BUDGETS))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("search", "changes"):
        raise SystemExit(__doc__)
    {"search": search, "changes": changes}[sys.argv[1]](sys.argv[2])
