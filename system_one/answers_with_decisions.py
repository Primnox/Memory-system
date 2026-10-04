"""Answer accuracy (top-1) on a blind set under a chosen set of change decisions.

Each scenario is loaded through the real `remember()` (as the blind runner
does, with the clock set to each statement's date), so topics, embeddings and
duplicates are the application's own. Then the "replaced by" links are set to:
  rules   - left as the application's rules decided (the current system),
  labels  - the true changes from the labels (perfect change detection: a ceiling),
  <file>  - the pairs a model retired (blind_pairs.py --save-pairs output).
Answerable questions are scored top-1 with the true date for "back then"
questions (oracle), exactly as scripts/bench_memory_blind.py does. Prints and
saves totals, plus a per-question hit flag keyed by question id (no text) so
two conditions can be compared with a paired exact McNemar test.

Runs in the Primnox backend environment:
    PYTHONIOENCODING=utf-8 <primnox>/backend/venv/Scripts/python system_one/answers_with_decisions.py \\
        --backend <primnox>/backend --decisions rules --tag rules
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def ms(day: str, end_of_day: bool = False) -> int:
    d = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000) + (86_400_000 - 1 if end_of_day else 12 * 3_600_000)


def run(sc: dict, decided: set[tuple[str, str]] | None) -> list[dict]:
    from primnox2.storage import db
    db.configure(Path(tempfile.mkdtemp(prefix="s1-ans-")) / "primnox.db")
    db.init()
    from primnox2.memory import service as mem
    sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
    sid, mid, real_now = {}, {}, mem.now_ms
    try:
        for s in sts:
            stamp = ms(s["date"])
            mem.now_ms = lambda stamp=stamp: stamp
            try:
                out = mem.remember(s["text"])
            except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                continue
            if out.get("duplicate_of") is None:
                sid[out["id"]], mid[s["id"]] = s["id"], out["id"]
    finally:
        mem.now_ms = real_now
    if decided is not None:
        con = db.connect()
        con.execute("UPDATE memories SET superseded_by = NULL")
        for old, new in decided:
            if old in mid and new in mid:
                con.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (mid[new], mid[old]))
        con.commit()
    today = ms(sc["today"], end_of_day=True)
    out = []
    for q in sc["questions"]:
        want = set(q.get("answer_ids") or [])
        if not want:
            continue
        as_of = q.get("as_of") or None
        hits = mem.search(q["text"], limit=10, as_of=as_of)
        if as_of is None:
            hits = [h for h in hits if h["created_at"] <= today]
        ids = [sid.get(h["id"]) for h in hits]
        out.append({"id": f'{sc["scenario"]}:{q["id"]}', "type": q["type"], "top1": bool(ids) and ids[0] in want})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--data", default=str(ROOT / "scripts/blind_memory/v2/test.json"))
    ap.add_argument("--decisions", required=True, help="rules | labels | path to blind_pairs --save-pairs json")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from primnox2.settings import tunables
    if tunables.get("memory.embeddings") or tunables.get("memory.embeddings_supersede"):
        from primnox2.memory import embeddings as enc
        if not enc.prepare(300):
            raise SystemExit("sentence encoder failed to load")
    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    model = None
    if args.decisions not in ("rules", "labels"):
        model = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["per_scenario"]
    qs = []
    for sc in scenarios:
        if args.decisions == "rules":
            decided = None
        elif args.decisions == "labels":
            decided = {(o, s["id"]) for s in sc["statements"] for o in s.get("replaces", [])}
        else:
            decided = {tuple(p) for p in model[sc["scenario"]]["retired_pairs"]}
        qs += run(sc, decided)
    by_type = {}
    for q in qs:
        h, n = by_type.get(q["type"], (0, 0))
        by_type[q["type"]] = (h + q["top1"], n + 1)
    hit = sum(q["top1"] for q in qs)
    print(f"{args.tag}: top-1 {hit}/{len(qs)} = {hit / len(qs):.1%}  " +
          "  ".join(f"{t} {h}/{n}" for t, (h, n) in sorted(by_type.items())))
    out = {"data": Path(args.data).name, "decisions": args.decisions if model is None else Path(args.decisions).name,
           "top1": hit, "answerable": len(qs), "by_type": {t: list(v) for t, v in by_type.items()},
           "hits": {q["id"]: q["top1"] for q in qs}}
    (HERE / "results" / f"answers_v2_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
