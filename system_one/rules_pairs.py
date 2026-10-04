"""Which (old, new) statement pairs the current rule-based memory retires.

Loads each scenario through the real `remember()` path, with the memory clock
set to each statement's date (as `scripts/bench_memory_blind.py --load chat`
does), and writes the pairs it marked superseded. `screen.py` scores them
against the labels next to the model candidates.

Runs in the Primnox backend's environment:

    PYTHONIOENCODING=utf-8 <primnox>/backend/venv/Scripts/python system_one/rules_pairs.py \
        --backend <primnox>/backend --out system_one/results/rules_dev.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV_FILES = ["scripts/blind_memory/dev.json", "scripts/blind_memory/change.json"]


def ms(day: str) -> int:
    # midday UTC, as the blind runner stamps statements
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000) + 12 * 3_600_000


def run(sc: dict) -> dict:
    from primnox2.storage import db
    db.configure(Path(tempfile.mkdtemp(prefix="s1-rules-")) / "primnox.db")
    db.init()
    from primnox2.memory import service as mem

    sid: dict[str, str] = {}
    refused, real_now = [], mem.now_ms
    try:
        for s in sorted(sc["statements"], key=lambda s: (s["date"], s["id"])):
            stamp = ms(s["date"])
            mem.now_ms = lambda stamp=stamp: stamp
            try:
                out = mem.remember(s["text"])
            except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                refused.append(s["id"])
                continue
            if out.get("duplicate_of") is None:
                sid[out["id"]] = s["id"]
    finally:
        mem.now_ms = real_now
    rows = db.connect().execute("SELECT id, superseded_by FROM memories WHERE superseded_by IS NOT NULL")
    pairs = sorted({(sid.get(r["id"]), sid.get(r["superseded_by"])) for r in rows})
    return {"pairs": [list(p) for p in pairs], "refused": refused}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="Primnox backend directory (holds primnox2/)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from primnox2.settings import tunables
    if tunables.get("memory.embeddings") or tunables.get("memory.embeddings_supersede"):
        # as the runner does: never score against the word-rules fallback while the encoder warms up
        from primnox2.memory import embeddings as enc
        if not enc.prepare(300):
            raise SystemExit("sentence encoder failed to load")
    result = {}
    for f in DEV_FILES:
        for sc in json.loads((ROOT / f).read_text(encoding="utf-8")):
            result[sc["scenario"]] = run(sc)
            print(sc["scenario"], len(result[sc["scenario"]]["pairs"]), "retired", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
