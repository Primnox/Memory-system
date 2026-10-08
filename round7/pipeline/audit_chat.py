"""Audit the chat-format people, then settle every label 2 of 3.

    python audit_chat.py prep      # label-free copies -> claude_audit_chat/ (for the Claude subagents)
    python audit_chat.py gemini    # the other Gemini audits each person through agy
    python audit_chat.py collect   # Claude subagents' labels -> chat/pNN/labels_claude-opus-5.5.json
    python audit_chat.py vote      # -> chat/pNN/voted.json + report

Vote: writer, Claude, other Gemini. A fact stays unless both auditors say its message
does not state it (dropped facts leave every label and answer). `replaces` and answer
sets are 2 of 3, then made consistent; a non-absent question left with no majority
answer is dropped. Pairs not voted unanimously go into `skip` for training.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drive
from drive import log, parse
from long import strayed

HERE = Path(__file__).resolve().parent
ROOT = HERE / "chat"
CLAUDE_DIR = HERE / "claude_audit_chat"
BRIEF = (HERE / "AUDIT_CHAT.md").read_text(encoding="utf-8")
OTHER = {"gemini-3.1-pro-high": "gemini-3.8-flash-medium", "gemini-3.8-flash-high": "gemini-3.1-pro-high"}
CLAUDE_TAG = "labels_claude-opus-5.5.json"
SONNET_DIR = CLAUDE_DIR.parent / (CLAUDE_DIR.name + "_sonnet")   # separate: never sees the Opus labels
SONNET_TAG = "labels_claude-sonnet.json"


def people() -> list[Path]:
    return sorted(d for d in ROOT.glob("p*") if (d / "final.json").exists())


def bare(sc: dict) -> dict:
    return {"scenario": sc["scenario"], "today": sc["today"], "sessions": sc["sessions"],
            "facts": [{k: f[k] for k in ("id", "session", "turn", "text")} for f in sc["facts"]],
            "questions": [{"id": q["id"], "text": q["text"], "as_of": q.get("as_of")} for q in sc["questions"]]}


def complete(lab: dict, sc: dict) -> bool:
    fids = {f["id"] for f in sc["facts"]}
    qids = {q["id"] for q in sc["questions"]}
    return (lab.get("scenario") == sc["scenario"] and fids <= set(lab.get("supported", {}))
            and fids <= set(lab.get("replaces", {})) and qids <= set(lab.get("answers", {})))


def prep() -> None:
    CLAUDE_DIR.mkdir(exist_ok=True)
    (CLAUDE_DIR / "AUDIT_CHAT.md").write_text(BRIEF, encoding="utf-8")
    for d in people():
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        (CLAUDE_DIR / f"{d.name}_unlabelled.json").write_text(json.dumps(bare(sc), indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    SONNET_DIR.mkdir(exist_ok=True)
    for f in list(CLAUDE_DIR.glob("*_unlabelled.json")) + [CLAUDE_DIR / "AUDIT_CHAT.md"]:
        (SONNET_DIR / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    leaks = [f.name for f in CLAUDE_DIR.glob("*_unlabelled.json")
             if any(k in f.read_text(encoding="utf-8") for k in ('"replaces"', '"answer_ids"', '"type"'))]
    print(f"{len(people())} label-free copies in {CLAUDE_DIR.name}; files containing labels: {leaks or 'none'}")


def gemini_one(d: Path) -> None:
    writer = (d / "writer.txt").read_text(encoding="utf-8").strip()
    model = OTHER[writer]
    out = d / f"labels_{model}.json"
    if out.exists():
        return
    sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
    body = json.dumps(bare(sc), indent=1, ensure_ascii=False)
    for m in (model, "gemini-3.8-flash-low") if "flash" in model else (model,):
        for attempt in range(2):
            try:
                text, _ = drive.agy(BRIEF + "\n" + body, m, d / f"audit_{m}")
                lab = parse(text)["labels"][0]
                if not complete(lab, sc):
                    raise ValueError("labels do not cover every fact and question")
                (d / f"labels_{model}.json").write_text(json.dumps({"labels": [lab]}, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
                log(f"audit_chat {d.name}: {m} done")
                return
            except Exception as e:  # noqa: BLE001
                log(f"audit_chat {d.name}: {m} attempt {attempt} failed: {str(e)[:150]}")


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
        if not src.exists():
            continue
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        lab = json.loads(src.read_text(encoding="utf-8"))["labels"][0]
        if complete(lab, sc):
            (d / tag).write_text(json.dumps({"labels": [lab]}, indent=1, ensure_ascii=False), encoding="utf-8")
            ok += 1
        else:
            print(f"{d.name}: Claude labels incomplete, not used")
    print(f"{tag}: {ok}/{len(people())} in the vote")


def vote_one(sc: dict, labs: list[dict]) -> tuple[dict, Counter]:
    ch = Counter()
    keep = {f["id"] for f in sc["facts"] if sum(1 for lab in labs if lab["supported"].get(f["id"]) is False) < 2}
    ch["facts dropped (both auditors: not stated)"] = len(sc["facts"]) - len(keep)
    facts, votes_all = [], Counter()
    for f in sc["facts"]:
        if f["id"] not in keep:
            continue
        v = Counter(o for o in f.get("replaces") or [] if o in keep)
        for lab in labs:
            v.update(o for o in lab["replaces"].get(f["id"]) or [] if o in keep)
        new = sorted((o for o, n in v.items() if n >= 2), key=lambda x: int(x[1:]))
        ch["pairs dropped"] += len(set(f.get("replaces") or []) - set(new))
        ch["pairs added"] += len(set(new) - set(f.get("replaces") or []))
        votes_all.update({(o, f["id"]): n for o, n in v.items()})
        facts.append({**f, "replaces": new})
    sess_date = {s["id"]: s["date"] for s in sc["sessions"]}
    stmt = [{"id": f["id"], "date": sess_date[f["session"]], "text": f["text"], "replaces": f["replaces"]}
            for f in facts]
    qs = []
    for q in sc["questions"]:
        v = Counter(a for a in q.get("answer_ids") or [] if a in keep)
        for lab in labs:
            v.update(a for a in lab["answers"].get(q["id"]) or [] if a in keep)
        new = sorted((a for a, n in v.items() if n >= 2), key=lambda x: int(x[1:]))
        if set(new) != set(q.get("answer_ids") or []):
            ch[f"{q['type']} answer sets changed"] += 1
        qs.append({**q, "answer_ids": new})
    out = {"scenario": sc["scenario"], "today": sc["today"], "statements": stmt, "questions": qs}
    ch["answer ids dropped to match the voted updates"] += drive.reconcile(out)
    kept_q = [q for q in out["questions"] if q["type"] == "absent" or q["answer_ids"]]
    ch["questions dropped, no majority answer"] += len(out["questions"]) - len(kept_q)
    out["questions"] = kept_q
    out["skip"] = sorted([list(p) for p, n in votes_all.items() if 0 < n < 3], key=lambda p: (int(p[1][1:]), int(p[0][1:])))
    out["sessions"] = sc["sessions"]   # kept for training the extractor
    out["facts"] = facts
    return out, ch


def vote() -> None:
    total, n, agree = Counter(), 0, Counter()
    for d in people():
        labs = [json.loads(f.read_text(encoding="utf-8"))["labels"][0] for f in sorted(d.glob("labels_*.json"))]
        stray = strayed(d) + [x for a in d.glob("audit_*") for x in strayed(a)]
        if stray:
            print(f"{d.name}: DISQUALIFIED, a run reached outside its folder: {stray[:2]}")
            continue
        if len(labs) < 2:
            print(f"{d.name}: {len(labs)}/2 audits, not voted")
            continue
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        w = {(o, f["id"]) for f in sc["facts"] for o in f.get("replaces") or []}
        for lab in labs:
            r = {(o, fid) for fid, os_ in lab["replaces"].items() for o in os_ or []}
            agree["pairs writer"] += len(w); agree["pairs both"] += len(w & r)
            agree["facts"] += len(sc["facts"]); agree["supported"] += sum(1 for v in lab["supported"].values() if v)
            agree["q"] += len(sc["questions"])
            agree["q agree"] += sum(1 for q in sc["questions"] if set(lab["answers"].get(q["id"], ["?"])) == set(q["answer_ids"]))
        out, ch = vote_one(sc, labs)
        (d / "voted.json").write_text(json.dumps([out], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        total.update(ch); n += 1
    print(f"voted {n}/{len(people())} people")
    if agree["pairs writer"]:
        print(f"auditors vs writer: updates {agree['pairs both']}/{agree['pairs writer']} "
              f"({100 * agree['pairs both'] / agree['pairs writer']:.0f}%); facts judged stated "
              f"{agree['supported']}/{agree['facts']}; answer sets {agree['q agree']}/{agree['q']}")
    print(f"vote changes: {dict(total)}")


if __name__ == "__main__":
    {"prep": prep, "gemini": gemini, "collect": collect, "vote": vote}[sys.argv[1]]()
