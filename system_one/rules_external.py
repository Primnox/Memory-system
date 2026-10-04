"""The current rule-based change detection on the external blind pairs.

Each pair is a two-statement history in a fresh scratch database: the earlier
statement through `remember()` on its date, then the later one; the pair counts
as "replaces" when the earlier memory ends up superseded by the later one.
Same files and metrics as external_eval.py. Runs in the Primnox backend
environment (needs the memory code):

    PYTHONIOENCODING=utf-8 <primnox>/backend/venv/Scripts/python system_one/rules_external.py \\
        --backend <primnox>/backend
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from rules_pairs import ms

HERE = Path(__file__).resolve().parent
FILES = [HERE / "data" / "blind_external" / "pairs_dnli.jsonl", HERE / "data" / "blind_external" / "pairs_lme_ku.jsonl"]


def replaced(old: dict, new: dict) -> bool:
    from primnox2.storage import db
    db.configure(Path(tempfile.mkdtemp(prefix="s1-rx-")) / "primnox.db")
    db.init()
    from primnox2.memory import service as mem
    real_now, ids = mem.now_ms, []
    try:
        for s in (old, new):
            stamp = ms(s["date"])
            mem.now_ms = lambda stamp=stamp: stamp
            try:
                out = mem.remember(s["text"])
            except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                return False
            if out.get("duplicate_of") is not None:
                return False
            ids.append(out["id"])
    finally:
        mem.now_ms = real_now
    row = db.connect().execute("SELECT superseded_by FROM memories WHERE id = ?", (ids[0],)).fetchone()
    return bool(row and row["superseded_by"] == ids[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from primnox2.settings import tunables
    if tunables.get("memory.embeddings") or tunables.get("memory.embeddings_supersede"):
        from primnox2.memory import embeddings as enc
        if not enc.prepare(300):
            raise SystemExit("sentence encoder failed to load")
    report = {}
    for f in FILES:
        pairs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        hits, by_source = [], defaultdict(lambda: [0, 0])
        for p in pairs:
            h = replaced({"text": p["old_text"], "date": p["old_date"]}, {"text": p["new_text"], "date": p["new_date"]})
            hits.append(int(h))
            by_source[p["source"]][0] += int(h)
            by_source[p["source"]][1] += 1
        y = [p["label"] for p in pairs]
        tp = sum(a and b for a, b in zip(y, hits))
        prec, rec = (tp / sum(hits) if sum(hits) else 0.0), (tp / sum(y) if sum(y) else 0.0)
        report[f.stem] = {"pairs": len(pairs), "positives": sum(y), "precision": prec, "recall": rec,
                          "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                          "called_replace_by_source": {k: f"{a}/{b}" for k, (a, b) in sorted(by_source.items())}}
        r = report[f.stem]
        print(f'{f.stem:<14} P {r["precision"]:.2f}  R {r["recall"]:.2f}  F1 {r["f1"]:.2f}  {r["called_replace_by_source"]}',
              flush=True)
    (HERE / "results" / "external_rules.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
