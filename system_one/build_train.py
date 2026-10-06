"""Build Laya training rows for the memory head (see SPEC_memory_head.md).

For every statement, the candidates are the SHORTLIST most similar earlier
statements (MiniLM) plus every earlier statement sharing a named person or
thing with it. Each (earlier, later) candidate pair becomes one row in Laya's
training format, with the same question and wording the zero-shot screen used:

  replaces in `replaces`       -> {"replaces": 1}
  labelled trap in `relations` -> {<relation>: 1}
  anything else                -> {"replaces": 0, the other four 0.25 each}
                                  ("not a replacement, kind unknown")

Train rows come from system_one/data/train_*.json (generated scenarios).
Validation rows come from dev.json + change.json, never trained on. The
frozen test sets are never read.

    system_one/.venv/Scripts/python system_one/build_train.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from screen import RELATION, state_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
VAL_FILES = [ROOT / "scripts/blind_memory/dev.json", ROOT / "scripts/blind_memory/change.json"]
SHORTLIST = 10        # top-5 reached only 72% of replacements in chain-reaction-heavy data
MAX_ENTITY_EXTRA = 12

QUESTIONS = {"relation": {"type": "choice",
                          "instructions": "How does the later message relate to the earlier statement?",
                          "criteria": RELATION}}
NOT_REPLACE = {"replaces": 0.0, "adds": 0.25, "detail": 0.25, "other": 0.25, "event": 0.25}
KIN = re.compile(r"\bmy (wife|husband|partner|boyfriend|girlfriend|fianc[eé]e?|son|daughter|kids?|mum|mom|"
                 r"mother|dad|father|brother|sister|grandma|grandmother|grandpa|grandfather|aunt|uncle|"
                 r"cousin|niece|nephew|dog|cat|puppy|kitten|boss|manager|landlord|flatmate|roommate|"
                 r"car|bike|flat|apartment|house|phone|laptop|gym|doctor|therapist)\b", re.I)
WORD = re.compile(r"[A-Z][a-zA-Z'\-]+")


def names_in(scenario: list[dict]) -> set[str]:
    """Capitalised words seen somewhere other than the start of a sentence: names, places, brands."""
    seen = set()
    for s in scenario:
        for m in WORD.finditer(s["text"]):
            before = s["text"][:m.start()].rstrip()
            if before and before[-1] not in ".!?\"“(":
                w = m.group().split("'")[0]
                if w not in ("I", "I'm", "I've", "I'd", "I'll"):
                    seen.add(w)
    return seen


def entities(text: str, names: set[str]) -> set[str]:
    found = {w.split("'")[0] for w in WORD.findall(text)} & names
    found |= {"my " + m.group(1).lower() for m in KIN.finditer(text)}
    return found


def embed(texts: list[str]) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc = tok(chunk, padding=True, truncation=True, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.update(zip(chunk, torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)))
    return out


def rows_for(scenarios: list[dict], vec: dict) -> tuple[list[dict], dict]:
    rows, stats = [], {"pairs": 0, "replaces": 0, "labelled_traps": 0, "soft": 0, "skipped": 0,
                       "true_replacements": 0, "found_by_similarity": 0, "found_with_entities": 0}
    for sc in scenarios:
        # pairs whose label a checker of another family disputed (gen_people.py):
        # neither trained as the writer said nor as "not a replacement"
        skip = {tuple(p) for p in sc.get("skip") or []}
        sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
        names = names_in(sts)
        ents = {s["id"]: entities(s["text"], names) for s in sts}
        for i, new in enumerate(sts):
            earlier = sts[:i]
            if not earlier:
                continue
            ranked = sorted(earlier, key=lambda o: -float(vec[o["text"]] @ vec[new["text"]]))
            similar = {o["id"] for o in ranked[:SHORTLIST]}
            linked = [o["id"] for o in ranked[SHORTLIST:] if ents[o["id"]] & ents[new["id"]]][:MAX_ENTITY_EXTRA]
            cands = similar | set(linked)
            gone = set(new.get("replaces") or [])
            rel = {r["to"]: r["relation"] for r in new.get("relations") or []}
            stats["true_replacements"] += len(gone)
            stats["found_by_similarity"] += len(gone & similar)
            stats["found_with_entities"] += len(gone & cands)
            for old in earlier:
                if old["id"] not in cands:
                    continue
                if (old["id"], new["id"]) in skip:
                    stats["skipped"] += 1
                    continue
                if old["id"] in gone:
                    probs, kind = {k: float(k == "replaces") for k in RELATION}, "replaces"
                elif old["id"] in rel:
                    probs, kind = {k: float(k == rel[old["id"]]) for k in RELATION}, "labelled_traps"
                else:
                    probs, kind = dict(NOT_REPLACE), "soft"
                stats[kind] += 1
                stats["pairs"] += 1
                pair = {"old_text": old["text"], "new_text": new["text"],
                        "old_date": old["date"], "new_date": new["date"]}
                rows.append({"state": json.dumps(state_text(pair)),
                             "questions": json.dumps(QUESTIONS),
                             "gold": json.dumps({"relation": {"probabilities": probs}}),
                             "meta": {"scenario": sc["scenario"], "old": old["id"], "new": new["id"],
                                      "kind": kind, "via": "similar" if old["id"] in similar else "entity"}})
    return rows, stats


def main() -> None:
    train = [sc for f in sorted(DATA.glob("train_*.json")) for sc in json.loads(f.read_text(encoding="utf-8"))]
    val = [sc for f in VAL_FILES for sc in json.loads(f.read_text(encoding="utf-8"))]
    texts = sorted({s["text"] for sc in train + val for s in sc["statements"]})
    vec = embed(texts)
    for name, scs in (("train", train), ("val", val)):
        rows, stats = rows_for(scs, vec)
        with open(DATA / f"{name}_rows.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, f"{len(scs)} scenarios", json.dumps(stats))


if __name__ == "__main__":
    main()
