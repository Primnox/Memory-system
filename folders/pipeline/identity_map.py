"""Who-is-who coverage map: IDENTITY_CASES.md -> identity_map.json.   python identity_map.py

Every numbered case gets: its section, the layer that must handle it (model / code / answer /
test), the 500M model decisions it exercises, and whether its whole section is HELD OUT of
training (drawn once with a fixed seed, before any data exists, from the model-layer sections
that are not core; core = the four sections that define the basic decisions).

Model decisions (all choices):
  folder   which folder: the speaker / an existing folder / new / ASK the user
  type     what kind of thing a new folder is (person, pet, place, organisation, group, thing, activity, event)
  kind     fact, plan, hypothetical, question, quote, joke, rumour, correction, fiction
  about    who it is about: speaker / person / group / thing
  change   which earlier statements it makes no longer true
  merge    two existing folders are one / one folder is really two (split)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "IDENTITY_CASES.md"
OUT = HERE / "identity_map.json"

# Section -> (layer, decisions). Layers follow the "Who handles which section" table of
# IDENTITY_CASES.md; section 21 is not in that table and is assigned to the model here (person
# vs avatar is a same-or-different folder decision).
SECTION = {
    1: ("model", ["folder", "merge"]), 2: ("model", ["folder", "about"]), 3: ("model", ["about", "kind", "folder"]),
    4: ("model", ["change"]), 5: ("model", ["merge", "folder"]), 6: ("model", ["about", "kind"]),
    7: ("model", ["folder", "type"]), 8: ("model", ["folder", "merge"]),
    21: ("model", ["folder", "type"]), 22: ("code", []), 23: ("code", []), 24: ("model", ["folder", "type", "change"]),
    25: ("code", []), 26: ("code", []), 27: ("code", []), 28: ("test", []), 29: ("code", []),
    30: ("model", ["kind", "change"]), 31: ("model", ["kind", "change"]), 32: ("code", []), 33: ("model", ["kind", "change"]),
    34: ("code", []), 35: ("code", []), 36: ("code", []), 37: ("code", []), 38: ("test", []),
    39: ("model", ["kind", "folder"]), 40: ("model", ["folder", "about"]), 41: ("model", ["kind"]),
    42: ("model", ["folder", "merge"]), 43: ("model", ["folder"]), 44: ("code", []), 45: ("model", ["folder", "change"]),
    46: ("code", []), 47: ("answer", []), 48: ("test", []),
}
# Cases inside mixed sections ("part of 4, 5, 23, 24, 45") that belong to another layer.
CASE_LAYER = {
    43: "code", 47: "code",                                   # history kept; dates read by code (when.py)
    64: "code",                                               # inverse relations follow from one stored link
    84: "code",                                               # forgetting is executed by code
    113: "code", 114: "code", 115: "code", 117: "code", 123: "code",   # channel / account / corruption / access
    245: "model", 246: "model", 247: "model", 252: "model", 253: "model", 254: "model",  # 23: objects
    255: "code", 257: "code", 258: "code", 259: "code",       # 24: bookkeeping by transaction ids
    472: "code", 473: "code", 474: "code",                    # 45: version lineage, legal restructuring
}
CASE_DECISIONS = {
    6: ["folder"], 22: ["about", "folder"], 48: ["folder"], 55: ["folder"], 56: ["folder"], 57: ["change", "folder"],
    59: ["change"], 60: ["about"], 61: ["about"], 63: ["change"], 74: ["kind"], 75: ["kind"], 76: ["kind"],
    77: ["kind"], 80: ["kind", "change"], 81: ["kind", "change"], 82: ["kind"], 101: ["folder"], 102: ["folder"],
    245: ["folder", "type"], 246: ["folder", "type"], 247: ["folder", "type"], 252: ["folder", "change"],
    253: ["change"], 254: ["folder"], 260: ["folder", "type"], 261: ["change"],
}
CORE = {1, 2, 6, 8}
HOLD_OUT_N, SEED = 4, 20261011


def parse() -> tuple[dict, dict]:
    text = SRC.read_text(encoding="utf-8")
    sections, cases = {}, {}
    for m in re.finditer(r"^## (\d+)\. (.+?) \((\d+)–(\d+)\)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        sec, title, lo, hi, body = int(m[1]), m[2], int(m[3]), int(m[4]), m[5]
        body = re.sub(r"\n\((Speaker|Merges)[^)]*\)\.?", "", body).replace("\n", " ")
        sections[sec] = {"title": title, "first": lo, "last": hi}
        for part in body.split(" · "):
            cm = re.match(r"\s*(\d+) (.+?)\s*$", part)
            if cm:
                cases[int(cm[1])] = {"name": cm[2].rstrip(".").strip(), "section": sec}
    return sections, cases


def main() -> None:
    sections, cases = parse()
    assert len(cases) == 404, len(cases)
    full_model = sorted(s for s, (layer, _) in SECTION.items() if layer == "model" and s not in CORE
                        and s not in (4, 5, 24, 45))
    held = sorted(random.Random(SEED).sample(full_model, HOLD_OUT_N))
    for n, c in cases.items():
        layer, dec = SECTION[c["section"]]
        layer = CASE_LAYER.get(n, layer)
        c["layer"] = layer
        c["decisions"] = CASE_DECISIONS.get(n, dec) if layer == "model" else []
        c["split"] = ("held_out" if c["section"] in held else "train") if layer == "model" else "n/a"
    for s, v in sections.items():
        v["layer"] = SECTION[s][0]
        v["held_out"] = s in held
    out = {"source": "IDENTITY_CASES.md", "seed": SEED, "core_sections": sorted(CORE),
           "held_out_sections": held, "drawn_from": full_model,
           "kinds": ["fact", "plan", "hypothetical", "question", "quote", "joke", "rumour", "correction", "fiction"],
           "about": ["speaker", "person", "group", "thing"],
           "types": ["person", "pet", "place", "organisation", "group", "thing", "activity", "event"],
           "sections": {str(k): v for k, v in sorted(sections.items())},
           "cases": {str(k): v for k, v in sorted(cases.items())}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by = Counter((c["layer"], c["split"]) for c in cases.values())
    dec = Counter(d for c in cases.values() for d in c["decisions"])
    print(f"{len(cases)} cases in {len(sections)} sections; held-out sections {held} "
          f"({', '.join(sections[s]['title'] for s in held)})")
    print("layer/split:", dict(by))
    print("model decisions exercised:", dict(dec))


if __name__ == "__main__":
    main()
