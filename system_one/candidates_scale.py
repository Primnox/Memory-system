"""How many true replacements can a pairwise detector even see, as memory grows?

For every statement that replaces earlier ones, the candidates are drawn from
the facts still live at that moment (true changes applied from the labels, so
the candidate step is measured alone). --pad N adds other people's non-safety
facts as older noise until the store holds N facts. Strategies:

  similar        top-k most similar live facts (MiniLM)
  +names         + live facts sharing a name / kin word (today's v1)
  +aliases       names and kin words joined when a statement ties them
                 ("my boyfriend Tom", "Tom, my boyfriend")
  +topic         + the most similar live facts with the same topic (Primnox's
                 topic lexicon), up to --topic-k

No model is run; this is the ceiling the detector works under.
    system_one/.venv/Scripts/python system_one/candidates_scale.py --pad 200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "graph"))
from build_train import KIN, embed, entities, names_in   # noqa: E402
from graph_index import topics                           # noqa: E402

SPLITS = {"train": sorted(str(p) for p in (HERE / "data").glob("train_writer_*.json")),
          "dev": [str(ROOT / "scripts/blind_memory/dev.json"), str(ROOT / "scripts/blind_memory/change.json")],
          "v2": [str(ROOT / "scripts/blind_memory/v2/test.json")]}
TIE = re.compile(r"\bmy (\w+),? ([A-Z][a-z]+)\b|\b([A-Z][a-z]+), my (\w+)\b")


def aliases(sts: list[dict], names: set[str]) -> dict[str, str]:
    """kin word <-> name ties stated anywhere earlier, as a union-find root map."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for s in sts:
        for m in TIE.finditer(s["text"]):
            kin, name = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            k, n = "my " + kin.lower(), name
            if n in names and KIN.search("my " + kin):
                parent[find(k)] = find(n)
    return {x: find(x) for x in parent}


def canon(ents: set[str], amap: dict[str, str]) -> set[str]:
    return {amap.get(e, e) for e in ents} | ents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="train")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--topic-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    scenarios = [sc for f in SPLITS[args.split] for sc in json.loads(Path(f).read_text(encoding="utf-8"))]
    others_pool = [(sc["scenario"], s) for sc in scenarios for s in sc["statements"]
                   if not topics.is_safety(s["text"])]
    vec = embed(sorted({s["text"] for sc in scenarios for s in sc["statements"]}))
    topic = {t: topics.topic_of(t) for t in vec}
    rng = random.Random(args.seed)
    strategies = ["similar", "+names", "+aliases", "+aliases+topic"]
    reach = {s: 0 for s in strategies}
    size = {s: 0 for s in strategies}
    total = asked = 0
    for sc in scenarios:
        sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
        noise = []
        if args.pad:
            pool = [s for name, s in others_pool if name != sc["scenario"]]
            noise = [{"id": f"noise:{i}", "text": s["text"]}
                     for i, s in enumerate(rng.sample(pool, min(len(pool), max(0, args.pad - len(sts)))))]
        live = list(noise)
        for i, new in enumerate(sts):
            seen = sts[:i + 1]
            names = names_in(seen)
            amap = aliases(seen, names)
            e_new = canon(entities(new["text"], names), amap)
            gone = set(new.get("replaces") or [])
            if gone:
                ranked = sorted(live, key=lambda o: -float(vec[o["text"]] @ vec[new["text"]]))
                sim = {o["id"] for o in ranked[:args.k]}
                rest = ranked[args.k:]
                named = {o["id"] for o in rest if entities(o["text"], names) & entities(new["text"], names)}
                aliased = {o["id"] for o in rest if canon(entities(o["text"], names), amap) & e_new}
                t_new = topic.get(new["text"])
                same_topic = [o["id"] for o in rest if t_new and topic.get(o["text"]) == t_new][:args.topic_k]
                sets = {"similar": sim, "+names": sim | named, "+aliases": sim | aliased,
                        "+aliases+topic": sim | aliased | set(same_topic)}
                total += len(gone)
                asked += 1
                for s, c in sets.items():
                    reach[s] += len(gone & c)
                    size[s] += len(c)
            live = [o for o in live if o["id"] not in gone] + [{"id": new["id"], "text": new["text"]}]
    print(f"split={args.split} pad={args.pad} k={args.k}: {total} true replacements")
    for s in strategies:
        print(f"  {s:<16} reach {reach[s]}/{total} = {reach[s] / total:.0%}   candidates per new fact {size[s] / max(1, asked):.1f}")
    (HERE / "results" / f"candidates_{args.split}_pad{args.pad}.json").write_text(json.dumps(
        {"split": args.split, "pad": args.pad, "k": args.k, "true": total,
         "rows": {s: {"reach": reach[s], "candidates": size[s] / max(1, asked)} for s in strategies}}, indent=1))


if __name__ == "__main__":
    main()
