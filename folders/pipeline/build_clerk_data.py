"""Training data for a local filing model (the clerk), from the Gemini clerk's own work.   python build_clerk_data.py

Each Gemini filing call is replayed exactly: the prompt is the folders and slots filed so far
plus the batch of new facts (in a short instruction, since a fine-tuned model learns the rules
from the examples); the completion is what Gemini filed, as compact JSON. Replaying uses the
same merge as folders.py and must end at the saved state.json, or the person is left out.
Only training people are used (never blind v2, LongMemEval or REALTALK). Held out for
validation: the same 4 chat people as the extractor's validation, and 4 writer people.
Output: ../kaggle_clerk_ds/{train,val}.jsonl  {"prompt", "completion", "person"}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import folders
from drive import parse

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle_clerk_ds"
SETS = ("chat", "writers", "multilang", "v3short", "v3dev", "v3long")

INSTRUCTION = """You file facts into a personal assistant's memory, kept as folders: one per person, pet, place, organisation, thing or activity in someone's life (types: person, pet, place, organisation, thing, activity, speaker), each with a name, aliases and a link to the speaker. Every fact goes in every folder it is about and under one slot "<folder id>.<key>" (lives_in, job, employer, diet, pet, hobby, events, ...): reuse an existing key whenever the fact is about the same attribute; a slot holds "one" value at a time or "many". One folder per real thing: add new names as aliases with folder_updates. Reply with compact JSON only:
{"new_folders":[...],"folder_updates":[...],"new_slots":[...],"facts":{"<fact id>":{"folders":[...],"slot":"..."}}}"""


def prompt(state: dict, batch: list[dict]) -> str:
    fs = "\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in state["folders"].items()) or "(none yet)"
    ss = "\n".join(f"{k} ({v})" for k, v in state["slots"].items()) or "(none yet)"
    facts = "\n".join(json.dumps({"id": f["id"], "date": f["date"], "text": f["text"]}, ensure_ascii=False) for f in batch)
    return f"{INSTRUCTION}\n\n## Folders so far\n{fs}\n\n## Slots so far\n{ss}\n\n## New facts\n{facts}"


def teacher_reply(cwd: Path, batch: list[dict], state: dict) -> dict | None:
    """The teacher's accepted reply for this batch: a Claude subagent's (clerk_step.py), or the
    last agy response that passes the checks."""
    if (cwd / "clerk_reply.json").exists():
        got = json.loads((cwd / "clerk_reply.json").read_text(encoding="utf-8"))
        return got if not folders.check(got, batch, state) else None
    good = None
    for raw in sorted(cwd.glob("raw_*.out")):
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == "result" and e["result"].get("status") == "SUCCESS":
                try:
                    got = parse(e["result"]["response"])
                except ValueError:
                    continue
                if not folders.check(got, batch, state):
                    good = got
    return good


def compact(got: dict, batch: list[dict]) -> str:
    out = {"new_folders": [{k: f.get(k, [] if k == "aliases" else "") for k in ("id", "name", "type", "aliases", "link")}
                           for f in got.get("new_folders", []) or [] if isinstance(f, dict)],
           "folder_updates": [u for u in got.get("folder_updates", []) or [] if isinstance(u, dict)],
           "new_slots": [{"key": s["key"], "holds": s.get("holds", "many")} for s in got.get("new_slots", []) or []
                         if isinstance(s, dict) and isinstance(s.get("key"), str)],
           "facts": {f["id"]: {"folders": got["facts"][f["id"]]["folders"], "slot": got["facts"][f["id"]]["slot"]}
                     for f in batch}}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def person(name: str, sc: dict) -> list[dict] | None:
    home = folders.OUT / name / re.sub(r"[^\w\-]", "_", sc["scenario"])
    if not (home / "state.json").exists():
        return None
    final = json.loads((home / "state.json").read_text(encoding="utf-8"))
    if len(final["facts"]) != len(sc["statements"]):
        return None
    state = folders.first_state(name, sc)
    facts = sorted(sc["statements"], key=lambda f: f["date"])
    rows = []
    for k in range(0, len(facts), folders.BATCH):
        batch = facts[k:k + folders.BATCH]
        got = teacher_reply(home / f"b{k // folders.BATCH:03d}", batch, state)
        if got is None:
            return None
        rows.append({"prompt": prompt(state, batch), "completion": compact(got, batch)})
        folders.merge(state, got, batch)
    if state["facts"] != final["facts"]:
        return None
    return rows


def main() -> None:
    chat_names = sorted(sc["scenario"] for sc in folders.load("chat"))
    held = set(random.Random(7).sample(chat_names, 4))                  # the extractor's validation people
    held |= set(random.Random(8).sample(sorted(sc["scenario"] for sc in folders.load("writers")), 4))
    train, val, missing = [], [], []
    for name in SETS:
        for sc in folders.load(name):
            rows = person(name, sc)
            if rows is None:
                missing.append(f"{name}/{sc['scenario']}")
                continue
            for r in rows:
                r["person"] = sc["scenario"]
            (val if sc["scenario"] in held else train).extend(rows)
    random.Random(2028).shuffle(train)
    OUT.mkdir(exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "held_out.json").write_text(json.dumps(sorted(held)), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-clerk-train", "id": "anikethpani/primnox-clerk-train", "licenses": [{"name": "other"}]}),
        encoding="utf-8")
    words = sum(len((r["prompt"] + r["completion"]).split()) for r in train)
    print(f"train {len(train)} examples ({words * 1.4 / 1e6:.1f}M tokens est.), val {len(val)}; "
          f"people left out (not fully filed or replay mismatch): {len(missing)} {missing[:6]}")


if __name__ == "__main__":
    main()
