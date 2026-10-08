"""Extractor v2 training data: fix the measured "short message = no fact" shortcut.   python build_extractor_data_v2.py

v1 learned that short messages hold no fact (41% of its examples were [] and most of those were
short task-only turns); on LongMemEval it then wrote nothing for 59% of short messages that did
state a fact. v2:
  chat turns      the 20 training chat people's user turns, but "no fact" turns kept at ~20%
                  of the chat examples (randomly down-sampled); one noisy copy of the rest
  short facts     single short messages that state a fact, from the statement-format people
                  (writers A-D + DeepSeek, mixed-language, v3): message -> [the statement],
                  plus a noisy copy (messy in, clean fact out)
Validation: the same 4 held-out chat people as v1 (clean). LongMemEval is never used.
Output: ../kaggle_ext2_ds/{train,val}.jsonl
"""
from __future__ import annotations

import json
import random
import zlib
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round7" / "pipeline"))
from build_extractor_data import INSTRUCTION, examples, prompt
from noisy import mess

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle_ext2_ds"
MS_DATA = Path(__file__).resolve().parents[2] / "system_one" / "data"
V3 = HERE.parent / "kaggle_ds"   # v3 training people (kept private; not in this repo)

SHORT_FACTS = 2400


def statement_people() -> list[dict]:
    scs = [sc for f in sorted(MS_DATA.glob("train_writer_*.json")) for sc in json.loads(f.read_text(encoding="utf-8"))]
    scs += [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "cases" / "multilang").glob("p*/voted.json"))]
    scs += [sc for f in sorted(V3.glob("train_v3_*.json")) for sc in json.loads(f.read_text(encoding="utf-8"))]
    return scs


def main() -> None:
    r = random.Random(2027)
    chat = [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "chat").glob("p*/voted.json"))]
    val_names = set(random.Random(7).sample(sorted(p["scenario"] for p in chat), 4))   # same split as v1
    train, val = [], []
    for sc in chat:
        if sc["scenario"] in val_names:
            val += examples(sc, None)
            continue
        # every task-only turn stays (clean + one noisy copy): with the short-fact examples
        # added below, "no fact" ends up ~15% of the whole set (v1: 41%)
        train += examples(sc, None)
        train += examples(sc, random.Random(zlib.crc32(f"{sc['scenario']}/v2".encode())))
    sts = [(s["date"], s["text"]) for sc in statement_people() for s in sc["statements"]
           if 3 <= len(s["text"].split()) <= 40]
    r.shuffle(sts)
    for date, text in sts[:SHORT_FACTS]:
        train.append({"prompt": prompt(date, [], text), "completion": json.dumps([text], ensure_ascii=False)})
    for date, text in sts[:SHORT_FACTS // 2]:
        train.append({"prompt": prompt(date, [], mess(text, r)), "completion": json.dumps([text], ensure_ascii=False)})
    r.shuffle(train)
    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                                          encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-extractor-train-v2", "id": "anikethpani/primnox-extractor-train-v2",
         "licenses": [{"name": "other"}]}), encoding="utf-8")
    c = Counter("empty" if x["completion"] == "[]" else "fact" for x in train)
    short = sum(1 for x in train if len(x["prompt"].split("User's latest message:\n")[1].split()) < 35)
    print(f"train {len(train)} ({dict(c)}, {100 * c['empty'] / len(train):.0f}% empty; {short} with a short message); "
          f"val {len(val)}; statement pool {len(sts)}")


if __name__ == "__main__":
    main()
