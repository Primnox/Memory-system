"""Take ChatGPT writer replies into the GPT part.  python ingest.py [part]

Reads every gpt/replies/*.json|*.txt (a pasted reply: any text holding one or more
{"scenarios": [...]} objects), checks each scenario with the writers' validator, and
writes the passing ones to <part>/final.json (default part: test_c). Problems are
listed per reply so a fix-up message can be sent back to the same chat.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from drive import validate

HERE = Path(__file__).resolve().parent
ALL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def objects(text: str) -> list[dict]:
    dec = json.JSONDecoder()
    found, i = [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and "scenarios" in obj:
            found.append(obj)
        i = end
    return found


def taken(exclude: str) -> tuple[set[str], set[str]]:
    names, slugs = set(), set()
    for f in HERE.glob("*/final.json"):
        if f.parent.name == exclude:
            continue
        for sc in json.loads(f.read_text(encoding="utf-8")):
            names.add(sc["scenario"].split("-")[0])
            slugs.add(sc["scenario"])
    return names, slugs


def masked(text: str) -> str:
    return re.sub(r"\b[A-Z]\w+", "X", text).strip().lower()


def template_problems(sc: dict, others: list[dict]) -> list[str]:
    """A person copied from another with the names swapped is one data point, not two."""
    errs = []
    rep = [tuple(s["replaces"]) for s in sc["statements"]]
    stm = {masked(s["text"]) for s in sc["statements"]}
    qs = {q["text"].strip().lower() for q in sc["questions"]}
    for o in others:
        if rep == [tuple(s["replaces"]) for s in o["statements"]]:
            errs.append(f"[{sc['scenario']}] has exactly the same update pattern as {o['scenario']}")
        shared = len(stm & {masked(s["text"]) for s in o["statements"]})
        if shared > 2:
            errs.append(f"[{sc['scenario']}] reuses {shared} statements from {o['scenario']} with names swapped")
        sq = len(qs & {q["text"].strip().lower() for q in o["questions"]})
        if sq > 3:
            errs.append(f"[{sc['scenario']}] reuses {sq} question texts from {o['scenario']}")
    return errs


def everyone_else(exclude: str) -> list[dict]:
    out = []
    for f in HERE.glob("*/final.json"):
        if f.parent.name != exclude:
            out += json.loads(f.read_text(encoding="utf-8"))
    return out


def main(part: str) -> None:
    names, _ = taken(part)
    others = everyone_else(part)
    good, seen = [], set()
    for f in sorted((HERE / "gpt" / "replies").glob("*")):
        if f.suffix not in (".json", ".txt"):
            continue
        objs = objects(f.read_text(encoding="utf-8"))
        if not objs:
            print(f"{f.name}: no {{\"scenarios\": ...}} JSON found")
            continue
        for sc in (s for o in objs for s in o["scenarios"]):
            errs = validate({"scenarios": [sc]}, 1, ALL_LETTERS)
            first = sc.get("scenario", "?").split("-")[0]
            if first in names or first in seen:
                errs.append(f"[{sc.get('scenario')}] first name '{first}' is already used by another person")
            if not errs:
                errs += template_problems(sc, others + good)
            if errs:
                print(f"{f.name}: {sc.get('scenario')}: {len(errs)} problems")
                for e in errs:
                    print("   - " + e)
                continue
            seen.add(first)
            good.append(sc)
            print(f"{f.name}: {sc['scenario']}: ok")
    out = HERE / part
    out.mkdir(exist_ok=True)
    (out / "final.json").write_text(json.dumps(good, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{part}: {len(good)} scenarios pass, written to {part}/final.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_c")
