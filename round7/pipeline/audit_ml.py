"""Audit the mixed-language people (statement format, v3's AUDIT.md), vote 2 of 3.

    python audit_ml.py prep | gemini | collect | vote
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drive
from drive import AUDIT, log, parse, strip_labels
from long import strayed, vote as vote_labels

HERE = Path(__file__).resolve().parent
ROOT = HERE / "cases" / "multilang"
CLAUDE_DIR = HERE / "claude_audit_ml"
OTHER = {"gemini-3.1-pro-high": "gemini-3.8-flash-medium", "gemini-3.8-flash-high": "gemini-3.1-pro-high"}
CLAUDE_TAG = "labels_claude-sonnet-a.json"   # slot 1: a Sonnet run (no Opus subagents)
SONNET_DIR = CLAUDE_DIR.parent / (CLAUDE_DIR.name + "_sonnet")   # separate: never sees the Opus labels
SONNET_TAG = "labels_claude-sonnet-b.json"   # slot 2: a second, separate Sonnet run


def people() -> list[Path]:
    return sorted(d for d in ROOT.glob("p*") if (d / "final.json").exists())


def complete(lab: dict, sc: dict) -> bool:
    return (lab.get("scenario") == sc["scenario"]
            and {s["id"] for s in sc["statements"]} <= set(lab.get("replaces", {}))
            and {q["id"] for q in sc["questions"]} <= set(lab.get("answers", {})))


def prep() -> None:
    CLAUDE_DIR.mkdir(exist_ok=True)
    (CLAUDE_DIR / "AUDIT.md").write_text(AUDIT, encoding="utf-8")
    for d in people():
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))
        (CLAUDE_DIR / f"{d.name}_unlabelled.json").write_text(
            json.dumps(strip_labels(sc), indent=1, ensure_ascii=False), encoding="utf-8")
    SONNET_DIR.mkdir(exist_ok=True)
    for f in list(CLAUDE_DIR.glob("*_unlabelled.json")) + [CLAUDE_DIR / "AUDIT.md"]:
        (SONNET_DIR / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    leaks = [f.name for f in CLAUDE_DIR.glob("*_unlabelled.json")
             if any(k in f.read_text(encoding="utf-8") for k in ('"replaces"', '"answer_ids"', '"type"'))]
    print(f"{len(people())} label-free copies; files containing labels: {leaks or 'none'}")


def gemini_one(d: Path) -> None:
    model = OTHER[(d / "writer.txt").read_text(encoding="utf-8").strip()]
    if (d / f"labels_{model}.json").exists():
        return
    sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
    body = json.dumps(strip_labels([sc]), indent=1, ensure_ascii=False)
    for m in (model, "gemini-3.8-flash-low") if "flash" in model else (model,):
        for attempt in range(2):
            try:
                text, _ = drive.agy(AUDIT + "\nWork only from the data below. Do not use any tools.\n" + body,
                                    m, d / f"audit_{m}")
                lab = parse(text)["labels"][0]
                if not complete(lab, sc):
                    raise ValueError("labels do not cover every statement and question")
                (d / f"labels_{model}.json").write_text(json.dumps({"labels": [lab]}, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
                log(f"audit_ml {d.name}: {m} done")
                return
            except Exception as e:  # noqa: BLE001
                log(f"audit_ml {d.name}: {m} attempt {attempt} failed: {str(e)[:150]}")


def gemini() -> None:
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(gemini_one, people()))
    print(f"gemini audits: {sum(1 for d in people() if list(d.glob('labels_gemini*.json')))}/{len(people())}")


def collect() -> None:
    for src_dir, tag in ((CLAUDE_DIR, CLAUDE_TAG), (SONNET_DIR, SONNET_TAG)):
        collect_slot(src_dir, tag)


def collect_slot(src_dir: Path, tag: str) -> None:
    ok = 0
    for d in people():
        src = src_dir / f"{d.name}_labels.json"
        if src.exists():
            sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
            lab = json.loads(src.read_text(encoding="utf-8"))["labels"][0]
            if complete(lab, sc):
                (d / tag).write_text(json.dumps({"labels": [lab]}, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
                ok += 1
    print(f"{tag}: {ok}/{len(people())} in the vote")


def vote() -> None:
    total, n, agree = Counter(), 0, Counter()
    for d in people():
        stray = strayed(d) + [x for a in d.glob("audit_*") for x in strayed(a)]
        labs = [json.loads(f.read_text(encoding="utf-8"))["labels"][0] for f in sorted(d.glob("labels_*.json"))]
        if stray or len(labs) < 2:
            print(f"{d.name}: {'DISQUALIFIED (strayed)' if stray else f'{len(labs)}/2 audits'}, not voted")
            continue
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        w = {(o, s["id"]) for s in sc["statements"] for o in s["replaces"]}
        votes = Counter(w)
        for lab in labs:
            r = {(o, sid) for sid, os_ in lab["replaces"].items() for o in os_ or []}
            votes.update(r)
            agree["w"] += len(w); agree["both"] += len(w & r); agree["q"] += len(sc["questions"])
            agree["qa"] += sum(1 for q in sc["questions"] if set(lab["answers"].get(q["id"], ["?"])) == set(q["answer_ids"]))
        out, ch = vote_labels(sc, labs)
        out["skip"] = sorted([list(p) for p, v in votes.items() if 0 < v < 3], key=lambda p: (int(p[1][1:]), int(p[0][1:])))
        (d / "voted.json").write_text(json.dumps([out], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        total.update(ch); n += 1
    print(f"voted {n}/{len(people())}")
    if agree["w"]:
        print(f"auditors vs writer: updates {agree['both']}/{agree['w']} ({100 * agree['both'] / agree['w']:.0f}%); "
              f"answer sets {agree['qa']}/{agree['q']}")
    print(f"vote changes: {dict(total)}")


if __name__ == "__main__":
    {"prep": prep, "gemini": gemini, "collect": collect, "vote": vote}[sys.argv[1]]()
