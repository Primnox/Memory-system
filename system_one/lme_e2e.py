"""LongMemEval end to end: chats from another model family, run through the
whole memory, scored on whether search returns what is true now.

LongMemEval (Wu et al., MIT) gives each question its own chat history; the
oracle variant keeps only the sessions holding the evidence. User turns marked
`has_answer` are the evidence. Here each question becomes one scenario in the
blind-set schema:

  statements  every statement about the user in its user turns, split by the
              rule-based extractor (extract.py), dated by its session, in order;
              statements over the memory's 240-character limit are dropped and
              counted (the memory refuses them anyway)
  answer_ids  for knowledge-update questions, the evidence in the LATEST evidence
              session (the current value); for single-session-user, all evidence
  stale_ids   knowledge-update only: evidence from earlier sessions (the old value)

`build` writes the scenarios. `blind_pairs.py --data <file> --save-pairs` (on a
GPU) gives the memory head's retirements. `score` loads every scenario through
the application's own remember(), sets the "replaced by" links to the rules' own
decisions or to the head's, and scores search top-1 / top-3 with the question's
date as "today", plus how often a stale value comes first.

    python system_one/lme_e2e.py build
    PYTHONIOENCODING=utf-8 <primnox>/backend/venv/Scripts/python system_one/lme_e2e.py score \\
        --backend <primnox>/backend --decisions rules --tag rules
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE / "data" / "external"
OUT = HERE / "data" / "lme_e2e" / "lme_e2e.json"
TYPES = ("knowledge-update", "single-session-user")
MAX_CHARS = 240   # memory.max_chars


def iso(lme_date: str) -> str:
    return datetime.strptime(lme_date.split(" (")[0], "%Y/%m/%d").date().isoformat()


def build(args) -> None:
    from extract import statements
    data = json.loads((EXT / "longmemeval_oracle.json").read_text(encoding="utf-8"))
    scenarios, counts = [], {"questions": 0, "statements": 0, "too_long": 0, "answer_lost": 0}
    for item in data:
        if item["question_type"] not in TYPES:
            continue
        sessions = sorted(zip(item["haystack_dates"], item["haystack_session_ids"], item["haystack_sessions"]),
                          key=lambda s: iso(s[0]))
        evidence_sessions = [i for i, (_, _, turns) in enumerate(sessions)
                             if any(t.get("has_answer") and t["role"] == "user" for t in turns)]
        latest = evidence_sessions[-1] if evidence_sessions else None
        sts, answer, stale = [], [], []
        for si, (date, _, turns) in enumerate(sessions):
            user = [t for t in turns if t["role"] == "user"]
            for ti, turn in enumerate(user):
                kept = 0
                for k, text in enumerate(statements(turn["content"])):
                    if len(text) > MAX_CHARS:
                        counts["too_long"] += 1
                        continue
                    sid = f"s{si:02d}t{ti:03d}f{k:02d}"
                    sts.append({"id": sid, "date": iso(date), "text": text})
                    kept += 1
                    if turn.get("has_answer"):
                        if item["question_type"] == "single-session-user" or si == latest:
                            answer.append(sid)
                        else:
                            stale.append(sid)
                if turn.get("has_answer") and not kept:
                    counts["answer_lost"] += 1
        if not answer:
            continue
        counts["questions"] += 1
        counts["statements"] += len(sts)
        scenarios.append({"scenario": item["question_id"], "today": iso(item["question_date"]),
                          "statements": sts,
                          "questions": [{"id": "q", "type": item["question_type"], "text": item["question"],
                                         "answer_ids": answer, "stale_ids": stale}]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scenarios, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT}: {counts}")


def score(args) -> None:
    sys.path.insert(0, str(Path(args.backend).resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from answers_with_decisions import ms
    from primnox2.settings import tunables
    if tunables.get("memory.embeddings") or tunables.get("memory.embeddings_supersede"):
        from primnox2.memory import embeddings as enc
        if not enc.prepare(300):
            raise SystemExit("sentence encoder failed to load")
        if tunables.get("memory.search_encoder") and hasattr(enc, "prepare_search") and not enc.prepare_search(300):
            raise SystemExit("search encoder failed to load")
    from primnox2.storage import db
    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    model = None
    if args.decisions != "rules":
        model = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["per_scenario"]
    rows, refused = [], 0
    for sc in scenarios:
        db.configure(Path(tempfile.mkdtemp(prefix="lme-")) / "primnox.db")
        db.init()
        from primnox2.memory import service as mem
        sid, mid, real_now = {}, {}, mem.now_ms
        try:
            for s in sorted(sc["statements"], key=lambda s: (s["date"], s["id"])):
                mem.now_ms = lambda stamp=ms(s["date"]): stamp
                try:
                    out = mem.remember(s["text"])
                except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                    refused += 1
                    continue
                if out.get("duplicate_of") is None:
                    sid[out["id"]], mid[s["id"]] = s["id"], out["id"]
        finally:
            mem.now_ms = real_now
        if model is not None:
            con = db.connect()
            con.execute("UPDATE memories SET superseded_by = NULL")
            for old, new in model[sc["scenario"]]["retired_pairs"]:
                if old in mid and new in mid:
                    con.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (mid[new], mid[old]))
            con.commit()
        today = ms(sc["today"], end_of_day=True)
        for q in sc["questions"]:
            hits = [h for h in mem.search(q["text"], limit=10) if h["created_at"] <= today]
            ids = [sid.get(h["id"]) for h in hits]
            want, stale = set(q["answer_ids"]), set(q.get("stale_ids") or [])
            rows.append({"id": sc["scenario"], "type": q["type"],
                         "top1": bool(ids) and ids[0] in want,
                         "top3": bool(set(ids[:3]) & want),
                         "stale_first": bool(ids) and ids[0] in stale})
    out = {"data": Path(args.data).name, "decisions": args.decisions if model is None else Path(args.decisions).name,
           "refused": refused, "by_type": {}, "hits": {r["id"]: r["top1"] for r in rows}}
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        out["by_type"][t] = {k: [sum(r[k] for r in rs), len(rs)] for k in ("top1", "top3", "stale_first")}
        b = out["by_type"][t]
        print(f"{args.tag} {t}: top-1 {b['top1'][0]}/{len(rs)} = {b['top1'][0] / len(rs):.1%}, "
              f"top-3 {b['top3'][0] / len(rs):.1%}, stale value first {b['stale_first'][0]}/{len(rs)}")
    print(f"statements refused by the memory: {refused}")
    (HERE / "results" / f"lme_e2e_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--backend", required=True)
    s.add_argument("--data", default=str(OUT))
    s.add_argument("--decisions", required=True, help="rules | blind_pairs --save-pairs json")
    s.add_argument("--tag", required=True)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
