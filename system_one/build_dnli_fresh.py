"""A fresh external blind sample from Dialogue NLI's unverified test split
(crowdworkers, MIT), excluding every pair already in pairs_dnli.jsonl (sampled
from the verified test split), so a model shaped after seeing those results is
scored on items it has not been looked at with. contradiction -> should replace;
entailment and neutral -> should not.

    python build_dnli_fresh.py --per-label 500
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

from build_dnli import tidy

HERE = Path(__file__).resolve().parent
EXT = HERE / "data" / "external"
OUT = HERE / "data" / "blind_external" / "pairs_dnli_fresh.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    used = set()
    with open(HERE / "data" / "blind_external" / "pairs_dnli.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            used.add((p["old_text"], p["new_text"]))
    z = zipfile.ZipFile(EXT / "dnli.zip")
    verified = {(tidy(r["sentence2"]), tidy(r["sentence1"]))
                for r in json.loads(z.read("dnli/dialogue_nli/dialogue_nli_verified_test.jsonl"))}
    rows = json.loads(z.read("dnli/dialogue_nli/dialogue_nli_test.jsonl"))
    random.Random(args.seed).shuffle(rows)
    taken, out = {"negative": 0, "positive": 0, "neutral": 0}, []
    for ex in rows:
        old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
        lab = ex["label"]
        if (old, new) in used or (old, new) in verified or len(old) < 8 or len(new) < 8:
            continue
        if taken.get(lab, args.per_label) >= args.per_label:
            continue
        taken[lab] += 1
        out.append({"id": ex["id"], "source": f"dnli-fresh:{lab}", "old_text": old, "new_text": new,
                    "old_date": "2025-03-01", "new_date": "2025-06-01", "label": int(lab == "negative")})
    with open(OUT, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{OUT}: {taken}, none from the verified split or the earlier sample")


if __name__ == "__main__":
    main()
