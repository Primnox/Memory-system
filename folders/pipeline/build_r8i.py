"""Round 8 "index" training mix: round 7's people, each statement tagged with where it is filed.   python build_r8i.py

Every statement of round 7's mix (../kaggle_r7_ds, clean and noisy copies) gets a `filed` field
from the Gemini clerk's filing of its clean person (folders/<set>/<scenario>/state.json):
    "about Sarah · lives in · also: Leeds"
(the slot's folder, the slot, up to 2 other folders, by name only; the speaker is "the user";
links such as "former employer" are left out: they can carry timebleed (the filing saw later facts) and leak). One statement
in five, drawn per copy, gets none, so the head still works without an index. People whose
filing is missing keep no tags. Labels are unchanged.
Also writes blind v2 tagged from the one-fact-at-a-time filing (test only) for the evaluation.
Output: ../kaggle_r8_ds/
"""
from __future__ import annotations

import json
import random
import re
import shutil
import zlib
from pathlib import Path

import folders

HERE = Path(__file__).resolve().parent
R7 = HERE.parent / "kaggle_r7_ds"
OUT = HERE.parent / "kaggle_r8_ds"
DROP = 0.2
SET_OF = {"existing": ["writers"], "v3": ["v3short", "v3long", "v3dev"], "chat": ["chat"], "multilang": ["multilang"]}


def filings(names: list[str]) -> dict[str, dict]:
    out = {}
    for name in names:
        for f in (folders.OUT / name).glob("*/state.json"):
            out[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def tag(st: dict, fid: str) -> str | None:
    e = st["facts"].get(fid)
    if not e:
        return None
    def who(k: str) -> str:
        f = st["folders"].get(k, {})
        if f.get("type") == "speaker" or k == "F1":
            return "the user"
        # name only: a folder's link can carry timebleed ("former employer") and would leak changes
        return f.get("name", k)
    subject, key = e["slot"].split(".", 1) if "." in e["slot"] else (e["folders"][0], e["slot"])
    others = [who(k) for k in e["folders"] if k != subject and who(k) != "the user"][:2]
    return f"about {who(subject)} · {key.replace('_', ' ')}" + (f" · also: {', '.join(others)}" if others else "")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tagged = total = 0
    for f in sorted(R7.glob("train_r7_*.json")):
        group = f.stem.split("_")[2]
        sts = filings(SET_OF[group])
        people = json.loads(f.read_text(encoding="utf-8"))
        for sc in people:
            base = re.sub(r"-n\d+$", "", sc["scenario"])
            st = sts.get(re.sub(r"[^\w\-]", "_", base))
            r = random.Random(zlib.crc32(sc["scenario"].encode()))
            for s in sc["statements"]:
                total += 1
                t = tag(st, s["id"]) if st else None
                if t and r.random() >= DROP:
                    s["filed"] = t
                    tagged += 1
        (OUT / f.name.replace("r7", "r8i")).write_text(json.dumps(people, ensure_ascii=False) + "\n", encoding="utf-8")
    blind = json.loads(folders.BLIND.read_text(encoding="utf-8"))
    # blind v2 tags from the filing made ONE FACT AT A TIME (folders_b1): bleed-free
    bst = {f.parent.name: json.loads(f.read_text(encoding="utf-8")) for f in (HERE / "folders_b1" / "blind").glob("*/state.json")}
    for sc in blind:
        st = bst[re.sub(r"[^\w\-]", "_", sc["scenario"])]
        for s in sc["statements"]:
            s["filed"] = tag(st, s["id"])
    (OUT / "blind_v2_filed.json").write_text(json.dumps(blind, ensure_ascii=False), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-memory-r8i-train", "id": "anikethpani/primnox-memory-r8i-train",
         "licenses": [{"name": "other"}]}), encoding="utf-8")
    print(f"tagged {tagged} of {total} training statements ({100 * tagged / total:.0f}%); "
          f"example: {next(s['filed'] for s in blind[0]['statements'] if s.get('filed'))!r}")


if __name__ == "__main__":
    main()
