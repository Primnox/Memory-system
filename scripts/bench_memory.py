"""How good is permanent memory at remembering, updating and recalling?

Retrieval only, no language model in the loop, against a scratch database that
is never the app's. It answers three questions separately, because one number
hides which of them broke:

  RECALL         does the right memory come back for the question?
  SUPERSESSION   when a fact changes, is the old one retired — and ONLY it?
  TIME           can "what was it in month N" be answered?

Two corpora:

  SDL pack    a generated 24-month life (`backend/sdl`), scored against its own
              ground truth. Large, reproducible, but template-shaped: tuning
              to it alone would measure the generator.
  HELD-OUT    hand-written, real-sounding statements below. Small, but never
              used to tune anything. A gain that shows on SDL and not here is
              a gain on templates.

Report all of top-1, stale-on-top and false supersessions together. A recall
gain bought with more wrongly retired memories is a failure, not a win.

Usage:
    python scripts/bench_memory.py                         # memory-100, import mode
    python scripts/bench_memory.py --mode remember --query question
    python scripts/bench_memory.py --pack office-500 --seed 7
    python scripts/bench_memory.py --check                 # exit 1 below baseline
    python scripts/bench_memory.py --write-baseline        # record this run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BASELINE = ROOT / "scripts" / "bench_memory.baseline.json"

# ── held-out corpus ──────────────────────────────────────────────────────────
# Statements as a person would say them, in the order they were learned.
# `stale` are the ones a LATER statement in the list genuinely replaces; every
# other fact must survive. Queries are what a model might pass to
# `recall_memory`: a keyword, a paraphrase, never the stored words verbatim.
HELD_OUT = {
    "facts": [
        "My name is Priya Raman.",
        "I code in Python every day.",
        "My editor is VS Code.",
        "I drink black coffee in the morning.",
        "I use Postgres for the main database.",
        "I use Redis for the job queue.",
        "I live in Lisbon.",
        "I take the metro to the office.",
        "My manager is Tomas Berg.",
        "I prefer dark mode in every app.",
        "I take notes in Notion.",
        "I am allergic to peanuts.",
        # ── later statements: the first group changes, the rest do not ───────
        "I switched to Rust for most of my work.",
        "My editor is Helix now.",
        "I quit coffee and drink green tea instead.",
        "I moved to Berlin.",
        "I cycle to the office now.",
        "I moved my notes to Obsidian.",
        "I use Postgres 16 for the main database.",
        "I am training for a half marathon in March.",
    ],
    # (old, the later statement that genuinely replaces it). A refinement
    # ("Postgres" -> "Postgres 16") does NOT replace: both stay true.
    "replaced": [
        ("I code in Python every day.", "I switched to Rust for most of my work."),
        ("My editor is VS Code.", "My editor is Helix now."),
        ("I drink black coffee in the morning.", "I quit coffee and drink green tea instead."),
        ("I live in Lisbon.", "I moved to Berlin."),
        ("I take the metro to the office.", "I cycle to the office now."),
        ("I take notes in Notion.", "I moved my notes to Obsidian."),
    ],
    # (query, text that must come back, kind). `must_live` texts are the
    # independent facts a careless conflict engine retires by mistake.
    "queries": [
        ("what language do I code in", "I switched to Rust for most of my work."),
        ("editor", "My editor is Helix now."),
        ("coffee", "I quit coffee and drink green tea instead."),
        ("where do I live", "I moved to Berlin."),
        ("commute", "I cycle to the office now."),
        ("notes app", "I moved my notes to Obsidian."),
        ("database", "I use Postgres 16 for the main database."),
        ("job queue", "I use Redis for the job queue."),
        ("allergy", "I am allergic to peanuts."),
        ("theme", "I prefer dark mode in every app."),
        ("who is my manager", "My manager is Tomas Berg."),
        ("my name", "My name is Priya Raman."),
    ],
}

TOP = (1, 3, 10)


def pct(n: int, d: int) -> float | None:
    return None if not d else round(100.0 * n / d, 1)


def generate(pack: str, seed: int) -> Path:
    out = Path(tempfile.gettempdir()) / "sdl-bench" / f"{pack}-{seed}"
    if not (out / pack / "ground_truth.json").exists():
        subprocess.run([sys.executable, str(ROOT / "backend" / "sdl" / "generate.py"),
                        "--pack", pack, "--out", str(out), "--seed", str(seed)],
                       check=True, capture_output=True)
    return out / pack


def scratch_db():
    from primnox2.storage import db
    path = Path(tempfile.mkdtemp(prefix="bench-memory-")) / "primnox.db"
    db.configure(path)
    db.init()
    return db, path


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    return {"p50_ms": round(statistics.median(values), 2),
            "p95_ms": round(values[min(len(values) - 1, int(len(values) * 0.95))], 2)}


def search(mem, query: str, limit: int = 10, as_of=None) -> list[dict]:
    # `as_of` arrives with Phase 3; until then the keyword is simply not passed.
    if as_of is not None:
        return mem.search(query, limit=limit, as_of=as_of)
    return mem.search(query, limit=limit)


# ── SDL ──────────────────────────────────────────────────────────────────────
def run_sdl(pack: str, seed: int, mode: str, query_mode: str, use_as_of: bool = True) -> dict:
    d = generate(pack, seed)
    db, _ = scratch_db()
    from primnox2.memory import service as mem
    from sdl import inject

    rows = [json.loads(line) for line in
            (d / "memory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = sorted(rows, key=lambda m: (m["month"], m["id"]))
    by_id = {r["id"]: r for r in rows}

    from sdl import world as world_mod, packs
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    world = world_mod.build(seed=manifest["seed"], months=packs.get(pack).months,
                            people_count=packs.get(pack).people,
                            project_count=packs.get(pack).projects)

    def end_of_month(month: int) -> int:
        # inject.py spreads a month's memories over its first SPREAD_DAYS days.
        from datetime import datetime, timezone
        start = datetime.combine(world.month_date(month), datetime.min.time(), tzinfo=timezone.utc)
        return int(start.timestamp() * 1000) + inject.SPREAD_DAYS * 86_400_000

    write_ms: list[float] = []
    if mode == "import":
        t = time.perf_counter()
        # import_many directly, with the same timestamps inject.py would give.
        stamps = inject._timestamps(ordered, world.month_date)
        mem.import_many([{"text": r["text"], "category": r.get("category"),
                          "created_at": s} for r, s in zip(ordered, stamps)])
        write_ms.append((time.perf_counter() - t) * 1e3 / max(1, len(ordered)))
    else:
        for r in ordered:
            t = time.perf_counter()
            mem.remember(r["text"], category=r.get("category") or "personal",
                         provenance="imported")
            write_ms.append((time.perf_counter() - t) * 1e3)

    conn = db.connect()
    stored = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM memories")}
    retired = {r["text"] for r in stored.values() if r["superseded_by"]}
    true_pairs = {(by_id[s]["text"], r["text"]) for r in rows
                  for s in ([r["supersedes"]] if isinstance(r.get("supersedes"), str)
                            else (r.get("supersedes") or [])) if s in by_id}
    supersession = judge_supersession(list(stored.values()), true_pairs,
                                      {r["text"] for r in rows})

    queries = {q["id"]: q for q in json.loads((d / "queries.json").read_text(encoding="utf-8"))}
    truth = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    results, skipped = [], 0
    for qid, q in queries.items():
        if q["subsystem"] != "Memory Service":
            continue
        ans = truth[qid]["answer"]
        if not (isinstance(ans, dict) and "text" in ans):
            skipped += 1                       # a timeline "which month" query
            continue
        kw = re.search(r"regarding (\w+)", q["question"])
        text_q = kw.group(1) if (kw and query_mode == "kw") else q["question"]
        # Only an import keeps real timestamps; `remember()` stamps "now", so
        # in that mode a time question is asked without a time.
        as_of = (end_of_month(q["as_of_month"])
                 if (q["level"] == 4 and mode == "import" and use_as_of) else None)
        t = time.perf_counter()
        hits = search(mem, text_q, as_of=as_of)
        ms = (time.perf_counter() - t) * 1e3
        texts = [h["text"] for h in hits]
        rank = texts.index(ans["text"]) + 1 if ans["text"] in texts else None
        results.append({"id": qid, "level": q["level"], "q": text_q, "want": ans["text"],
                        "rank": rank, "top": texts[0] if texts else None,
                        # For a time question the right answer may be a retired
                        # memory, so "stale on top" only means something for now.
                        "stale_on_top": as_of is None and bool(texts) and texts[0] in retired,
                        "ms": ms})
    return {"corpus": f"sdl:{pack}:{seed}", "mode": mode, "query": query_mode,
            "memories": len(rows), "results": results, "skipped": skipped,
            "supersession": supersession, "write": percentiles(write_ms)}


# ── held-out ─────────────────────────────────────────────────────────────────
def run_held_out() -> dict:
    db, _ = scratch_db()
    from primnox2.memory import service as mem
    write_ms: list[float] = []
    for text in HELD_OUT["facts"]:
        t = time.perf_counter()
        mem.remember(text, provenance="explicit")
        write_ms.append((time.perf_counter() - t) * 1e3)
    stored = [dict(r) for r in db.connect().execute("SELECT * FROM memories")]
    retired = {r["text"] for r in stored if r["superseded_by"]}
    results = []
    for q, want in HELD_OUT["queries"]:
        t = time.perf_counter()
        hits = search(mem, q)
        ms = (time.perf_counter() - t) * 1e3
        texts = [h["text"] for h in hits]
        results.append({"id": f"ho:{q}", "level": 1, "q": q, "want": want,
                        "rank": texts.index(want) + 1 if want in texts else None,
                        "top": texts[0] if texts else None,
                        "stale_on_top": bool(texts) and texts[0] in retired, "ms": ms})
    return {"corpus": "held-out", "mode": "remember", "query": "natural",
            "memories": len(HELD_OUT["facts"]), "results": results, "skipped": 0,
            "supersession": judge_supersession(stored, set(HELD_OUT["replaced"]),
                                               set(HELD_OUT["facts"])),
            "write": percentiles(write_ms)}


def judge_supersession(stored: list[dict], true_pairs: set, all_texts: set) -> dict:
    """Scored by PAIR, not by "was it retired". A stale memory retired by the
    wrong successor is a wrong retirement: it was how one "switched to
    cortados" retiring eight unrelated memories passed as eight catches."""
    by_id = {r["id"]: r for r in stored}
    got = {(r["text"], by_id[r["superseded_by"]]["text"])
           for r in stored if r.get("superseded_by") in by_id}
    stale = {old for old, _ in true_pairs}
    correct = got & true_pairs
    wrong_successor = {p for p in got if p[0] in stale and p not in true_pairs}
    not_stale = {p for p in got if p[0] not in stale}
    return {"truly_stale": len(stale), "caught": len(correct),
            "wrong_successor": len(wrong_successor),
            "missed": len(stale - {old for old, _ in got}),
            "wrongly_retired": len(not_stale),
            "false_rate_pct": pct(len(not_stale), len(all_texts - stale)),
            "wrong_examples": sorted(f"{o} -> {n}" for o, n in not_stale | wrong_successor)[:6]}


# ── reporting ────────────────────────────────────────────────────────────────
def summarise(run: dict) -> dict:
    out = {}
    groups = {"all": run["results"]}
    for lvl in sorted({r["level"] for r in run["results"]}):
        groups[f"L{lvl}"] = [r for r in run["results"] if r["level"] == lvl]
    for name, rs in groups.items():
        n = len(rs)
        out[name] = {"n": n,
                     **{f"top{k}": pct(sum(1 for r in rs if r["rank"] and r["rank"] <= k), n)
                        for k in TOP},
                     "stale_on_top": pct(sum(r["stale_on_top"] for r in rs), n),
                     **percentiles([r["ms"] for r in rs])}
    return out


def render(run: dict, summary: dict) -> str:
    s = run["supersession"]
    lines = [f"\n== {run['corpus']}  mode={run['mode']}  query={run['query']}  "
             f"({run['memories']} memories) =="]
    lines.append(f"{'':<8}{'n':>4}{'top1':>8}{'top3':>8}{'top10':>8}{'stale-top':>11}{'p50 ms':>9}{'p95 ms':>9}")
    for name, m in summary.items():
        lines.append(f"{name:<8}{m['n']:>4}{m['top1']!s:>8}{m['top3']!s:>8}{m['top10']!s:>8}"
                     f"{m['stale_on_top']!s:>11}{m.get('p50_ms', '-')!s:>9}{m.get('p95_ms', '-')!s:>9}")
    lines.append(f"supersession: stale {s['truly_stale']}, caught by the right successor {s['caught']}, "
                 f"wrong successor {s['wrong_successor']}, missed {s['missed']}, "
                 f"wrongly retired {s['wrongly_retired']} ({s['false_rate_pct']}% of live)")
    if s.get("wrong_examples"):
        lines.append("  wrong: " + "; ".join(s["wrong_examples"]))
    lines.append(f"write: {run['write']}   skipped timeline queries: {run['skipped']}")
    return "\n".join(lines)


def check(report: dict) -> list[str]:
    """Regressions against the recorded baseline. Higher-is-better metrics may
    not fall by more than 2 points; false supersession may not rise by more."""
    if not BASELINE.exists():
        return ["no baseline recorded (run --write-baseline)"]
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    bad = []
    for key, now in report["runs"].items():
        was = base["runs"].get(key)
        if not was:
            continue
        for grp, m in now["summary"].items():
            for metric in ("top1", "top3", "top10"):
                a, b = m.get(metric), was["summary"].get(grp, {}).get(metric)
                if a is not None and b is not None and a < b - 2:
                    bad.append(f"{key} {grp} {metric}: {b} -> {a}")
        a = now["supersession"]["false_rate_pct"]; b = was["supersession"]["false_rate_pct"]
        if a is not None and b is not None and a > b + 2:
            bad.append(f"{key} false supersession: {b}% -> {a}%")
        if now["supersession"].get("wrong_successor", 0) > was["supersession"].get("wrong_successor", 0):
            bad.append(f"{key} wrong successor: {was['supersession'].get('wrong_successor', 0)} -> "
                       f"{now['supersession']['wrong_successor']}")
        if now["supersession"]["missed"] > was["supersession"]["missed"]:
            bad.append(f"{key} stale missed: {was['supersession']['missed']} -> {now['supersession']['missed']}")
    return bad


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", default="memory-100")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--mode", choices=("import", "remember"), default="import",
                    help="import keeps real timestamps, so time questions can be asked; "
                         "remember is the one-at-a-time chat path, stamped now")
    ap.add_argument("--query", choices=("kw", "question"), default="kw")
    ap.add_argument("--no-as-of", action="store_true",
                    help="ask time questions without a time (the pre-Phase-3 behaviour)")
    ap.add_argument("--no-sdl", action="store_true")
    ap.add_argument("--no-held-out", action="store_true")
    ap.add_argument("--embeddings", choices=("search", "supersede", "both"),
                    help="turn on the sentence-encoder arms (memory.embeddings, "
                         "memory.embeddings_supersede); without it they are off here, as the "
                         "baseline was recorded")
    ap.add_argument("--actor-guard", action="store_true", help="memory.actor_guard")
    ap.add_argument("--misses", action="store_true", help="list every non-top-1 query")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--check", action="store_true", help="exit 1 on regression")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()
    if args.write_baseline and (args.embeddings or args.actor_guard):
        ap.error("the baseline is recorded with the embeddings arms off")
    if not args.embeddings:
        # The arms ship on; this corpus' baseline was recorded with them off.
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "0"
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE"] = "0"
    if args.embeddings or args.actor_guard:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cached encoder, no network probe
        if args.embeddings:
            os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "1" if args.embeddings in ("search", "both") else "0"
            os.environ["PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE"] = (
                "1" if args.embeddings in ("supersede", "both") else "0")
            from primnox2.memory import embeddings
            if not embeddings.prepare(300):
                ap.error("the sentence encoder could not be loaded; refusing to score the fallback")
        if args.actor_guard:
            os.environ["PRIMNOX2_MEMORY_ACTOR_GUARD"] = "1"

    runs = {}
    if not args.no_sdl:
        r = run_sdl(args.pack, args.seed, args.mode, args.query, not args.no_as_of)
        runs[f"sdl:{args.pack}:{args.mode}:{args.query}"] = r
    if not args.no_held_out:
        runs["held-out"] = run_held_out()

    report = {"runs": {}}
    for key, run in runs.items():
        summary = summarise(run)
        print(render(run, summary))
        if args.misses:
            for r in run["results"]:
                if r["rank"] != 1:
                    print(f"  miss rank={r['rank']} | {r['q']}\n     want: {r['want']}\n     got : {r['top']}")
        report["runs"][key] = {"summary": summary, "supersession": run["supersession"],
                               "write": run["write"], "memories": run["memories"]}

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.write_baseline:
        BASELINE.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nbaseline written: {BASELINE}")
    if args.check:
        bad = check(report)
        if bad:
            print("\nREGRESSION:\n  " + "\n  ".join(bad))
            return 1
        print("\nno regression against baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
