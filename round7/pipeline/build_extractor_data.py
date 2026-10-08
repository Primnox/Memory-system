"""Training data for a local fact extractor.   python build_extractor_data.py

From the 24 audited chat people: every USER turn becomes one example —
  input   the note-taker instruction, the message's date, the two turns before it in the
          same chat (context, to resolve "it", "there", "she"), and the message itself
  output  JSON list of the facts the audit kept for that turn ([] for task-only turns)
plus noisy copies (noisy.py: typos, shorthand, fragments...) of the user's words, same facts.

Split by PERSON, never by turn: 4 random people are validation, the rest train.
LongMemEval is never used here — it stays the exam.
Output: ../kaggle_ext_ds/{train,val}.jsonl  ({"prompt": ..., "completion": ...})
"""
from __future__ import annotations

import json
import random
import zlib
from collections import Counter
from pathlib import Path

from noisy import mess

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle_ext_ds"
NOISY_COPIES = 2

INSTRUCTION = (
    "You are the note-taker for a personal assistant's long-term memory. Read the user's latest "
    "message and write down every fact it states about the user's life (or the people, pets and "
    "things in it) worth remembering. One self-contained fact per item, first person, 5-25 words, "
    "with every reference and relative date resolved using the earlier turns and the date. "
    "Requests, questions, thanks and hypotheticals are not facts. Reply with a JSON list of "
    "strings; [] if the message states no fact.")


def prompt(date: str, context: list[dict], message: str) -> str:
    ctx = "\n".join(f"{t['role']}: {t['text']}" for t in context) or "(start of the chat)"
    return f"{INSTRUCTION}\n\nDate: {date}\nEarlier turns:\n{ctx}\n\nUser's latest message:\n{message}\n\nFacts:"


def examples(sc: dict, r: random.Random | None) -> list[dict]:
    kept = {(f["session"], f["turn"]): [] for f in sc["facts"]}
    for f in sc["facts"]:
        kept[(f["session"], f["turn"])].append(f["text"])
    out = []
    for s in sc["sessions"]:
        turns = s["turns"]
        for i, t in enumerate(turns):
            if t["role"] != "user":
                continue
            msg = t["text"] if r is None else mess(t["text"], r)
            ctx = [{"role": c["role"], "text": c["text"] if (r is None or c["role"] != "user") else mess(c["text"], r)}
                   for c in turns[max(0, i - 2):i]]
            facts = kept.get((s["id"], i), [])
            out.append({"prompt": prompt(s["date"], ctx, msg), "completion": json.dumps(facts, ensure_ascii=False)})
    return out


def main() -> None:
    people = [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "chat").glob("p*/voted.json"))]
    names = sorted(p["scenario"] for p in people)
    val_names = set(random.Random(7).sample(names, 4))
    OUT.mkdir(exist_ok=True)
    stats = Counter()
    for split in ("train", "val"):
        rows = []
        for sc in people:
            if (sc["scenario"] in val_names) != (split == "val"):
                continue
            rows += examples(sc, None)
            if split == "train":
                for c in range(NOISY_COPIES):
                    rows += examples(sc, random.Random(zlib.crc32(f"{sc['scenario']}/{c}".encode())))
        random.Random(11).shuffle(rows)
        (OUT / f"{split}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                                            encoding="utf-8")
        stats[f"{split} examples"] = len(rows)
        stats[f"{split} with no fact"] = sum(1 for x in rows if x["completion"] == "[]")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-extractor-train", "id": "anikethpani/primnox-extractor-train",
         "licenses": [{"name": "other"}]}), encoding="utf-8")
    print(dict(stats), "| validation people:", sorted(val_names))


if __name__ == "__main__":
    main()
