"""Does linking facts by people, places and things help search on real chats?   python graph_search.py

REALTALK (proof set, never tuned on): the Gemini-extracted facts and the 505 questions every input
can answer (../realtalk/e2e_llm_common.json). Three rankings over each chat's facts:

  A  similarity     cosine with gte-small (the app's search encoder)
  B  + entities     A + ALPHA * (rarity-weighted overlap of names / kin and thing words with the question)
  C  + one hop      B, then entities of B's top HOP_K facts (minus the question's own) boost facts
                    that share them: + BETA * overlap  (Emi -> sister -> Lisbon)

Entity weight = log(N / document frequency) within the chat, so a speaker named in every fact counts
for ~nothing (the "giant folder" problem). ALPHA, BETA and HOP_K are fixed here, before any run, and
not tuned on these questions. A hit = a fact from an evidence message ranked 1st (top-1) / in the top 3.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "realtalk" / "e2e_llm_common.json"
ALPHA, BETA, HOP_K = 0.10, 0.05, 5
KIN = {"sister", "brother", "mom", "mum", "mother", "dad", "father", "parents", "girlfriend", "boyfriend",
       "wife", "husband", "partner", "roommate", "friend", "friends", "cousin", "aunt", "uncle", "grandma",
       "grandmother", "grandpa", "grandfather", "daughter", "son", "baby", "dog", "cat", "puppy", "kitten",
       "boss", "manager", "coworker", "coworkers", "professor", "teacher", "class", "job", "school", "team",
       "car", "apartment", "house", "phone", "game", "show", "movie", "book", "concert", "trip", "birthday"}
SKIP = {"I", "The", "A", "An", "On", "In", "At", "And", "But", "My", "He", "She", "They", "We", "It", "This",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
        "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def entities(text: str) -> set[str]:
    out = set()
    for w in WORD.findall(text):
        base = w.split("'")[0]
        if base[:1].isupper() and base not in SKIP and len(base) > 1:
            out.add(base.lower())
        elif base.lower() in KIN:
            out.add(base.lower())
    return out


class Encoder:
    def __init__(self, name: str = "thenlper/gte-small"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def __call__(self, texts: list[str]) -> torch.Tensor:
        vecs = []
        for i in range(0, len(texts), 64):
            enc = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            vecs.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1))
        return torch.cat(vecs)


def overlap(a: set[str], b: set[str], w: dict[str, float]) -> float:
    return sum(w.get(e, 0.0) for e in a & b)


def main() -> None:
    enc = Encoder()
    scenarios = json.loads(DATA.read_text(encoding="utf-8"))
    rows = []
    for sc in scenarios:
        facts = sc["statements"]
        if not sc["questions"]:
            continue
        fv = enc([f["text"] for f in facts])
        fe = [entities(f["text"]) for f in facts]
        df = Counter(e for s in fe for e in s)
        n = len(facts)
        w = {e: math.log(n / c) / math.log(n) for e, c in df.items()}          # 0 (everywhere) .. 1 (unique)
        qv = enc([q["text"] for q in sc["questions"]])
        for qi, q in enumerate(sc["questions"]):
            want = set(q["answer_ids"])
            sim = (fv @ qv[qi]).tolist()
            qe = entities(q["text"])
            a = sim
            b = [s + ALPHA * overlap(qe, e, w) for s, e in zip(sim, fe)]
            top = sorted(range(n), key=lambda i: -b[i])[:HOP_K]
            hop = set().union(*(fe[i] for i in top)) - qe
            c = [s + BETA * overlap(hop, e, w) for s, e in zip(b, fe)]
            row = {"type": q["type"]}
            for name, score in (("A", a), ("B", b), ("C", c)):
                order = sorted(range(n), key=lambda i: -score[i])
                ids = [facts[i]["id"] for i in order[:3]]
                row[name + "1"] = ids[0] in want
                row[name + "3"] = bool(set(ids) & want)
            rows.append(row)
    print(f"REALTALK, Gemini facts, {len(rows)} questions; ALPHA={ALPHA} BETA={BETA} HOP_K={HOP_K} (fixed in advance)\n")
    print(f"{'questions':22s} {'A similarity':>16s} {'B +entities':>16s} {'C +one hop':>16s}   (top-1 / top-3)")
    for t in ("all", "multi-hop", "temporal", "other"):
        rs = rows if t == "all" else [r for r in rows if r["type"] == t]
        cells = [f"{100 * sum(r[k + '1'] for r in rs) / len(rs):5.1f}% / {100 * sum(r[k + '3'] for r in rs) / len(rs):5.1f}%"
                 for k in "ABC"]
        print(f"{t + f' ({len(rs)})':22s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")


if __name__ == "__main__":
    main()
