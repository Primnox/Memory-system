"""Dialogue NLI (Welleck et al. 2019, MIT) as extra memory-head rows.

Pairs of a persona fact and a later chat utterance, labelled by people:
contradiction -> `replaces` (the newer statement wins in a memory),
entailment -> `detail`, neutral -> `other`. Capped at --per-label rows each so
our own scenarios stay the main training signal. Source zip:
huggingface.co/datasets/xksteven/dialogue_nli (dnli.zip), kept out of git.

    system_one/.venv/Scripts/python system_one/build_dnli.py --per-label 300
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path

from build_train import QUESTIONS
from screen import RELATION, state_text

HERE = Path(__file__).resolve().parent
ZIP = HERE / "data" / "external" / "dnli.zip"
MAP = {"negative": "replaces", "positive": "detail", "neutral": "other"}


def tidy(s: str) -> str:
    s = re.sub(r"\s+([.,!?;:'])", r"\1", s.strip())
    s = re.sub(r"\bi\b", "I", s)
    s = s.replace(" n't", "n't")
    return s[:1].upper() + s[1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    data = json.loads(zipfile.ZipFile(ZIP).read("dnli/dialogue_nli/dialogue_nli_train.jsonl"))
    rng = random.Random(args.seed)
    rng.shuffle(data)
    taken = {k: 0 for k in MAP}
    out = HERE / "data" / "dnli_rows.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for ex in data:
            lab = ex["label"]
            if taken[lab] >= args.per_label:
                continue
            earlier, later = tidy(ex["sentence2"]), tidy(ex["sentence1"])
            if len(earlier) < 8 or len(later) < 8:
                continue
            taken[lab] += 1
            rel = MAP[lab]
            pair = {"old_text": earlier, "new_text": later, "old_date": "2025-03-01", "new_date": "2025-06-01"}
            f.write(json.dumps({"state": json.dumps(state_text(pair)),
                                "questions": json.dumps(QUESTIONS),
                                "gold": json.dumps({"relation": {"probabilities": {k: float(k == rel) for k in RELATION}}}),
                                "meta": {"source": "dnli", "kind": "dnli_" + rel}}, ensure_ascii=False) + "\n")
            if all(v >= args.per_label for v in taken.values()):
                break
    print(out, taken)


if __name__ == "__main__":
    main()
