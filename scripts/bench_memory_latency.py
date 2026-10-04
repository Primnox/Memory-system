"""What the embeddings search arm costs: model load, and search latency at 100
and 1,000 memories, with the vector cache cold and warm.

Memories are written straight into a scratch database (the conflict engine is
quadratic in store size and is not what is being timed here), as generated
two-clause sentences so the vectors differ. Queries cycle through paraphrases.

Rows, per store size:
  lexical          memory.embeddings off. What search costs today.
  first use        flag on, encoder never loaded: the first search must answer
                   by words while the encoder loads in the background, and
                   `load` is how long until it is ready (torch import + weights).
  hybrid, cold     encoder loaded, vector cache empty: one search embeds every
                   memory and writes the cache. Paid once per memory, ever.
  hybrid, warm     encoder loaded, vectors cached. Steady state.

Usage:
    python scripts/bench_memory_latency.py
    python scripts/bench_memory_latency.py --sizes 100 1000 5000 --runs 80
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cached encoder, no network probe

SUBJECTS = ["my dog", "my sister", "the project", "my manager", "my laptop", "the flat", "my bike",
            "my doctor", "our cat", "my phone", "the garden", "my car", "my partner", "the office"]
PREDICATES = ["is called {x}", "moved to {x}", "was bought in {x}", "needs {x}", "costs about {x}",
              "belongs to {x}", "was fixed by {x}", "is scheduled for {x}", "was renamed {x}"]
FILLERS = ["Tuesday", "Lisbon", "a new battery", "Priya", "the spring", "forty pounds", "Oakfield",
           "Biscuit", "next month", "a replacement", "the north side", "Monzo", "Helix", "the hospital"]
QUERIES = ["what pets do I have", "which handset do I carry", "where do I live", "who is my supervisor",
           "what vehicle do I own", "when is my appointment", "how do I get to work", "what do I owe",
           "who looks after the garden", "what did I buy recently", "what is my occupation",
           "do I have any siblings", "which computer do I use", "what is scheduled next month",
           "who fixed it", "what is it called", "how much did it cost", "what needs replacing",
           "where was it bought", "what was renamed"]

DAY = 86_400_000


def build(n: int) -> None:
    from primnox2.storage import db
    db.configure(Path(tempfile.mkdtemp(prefix="memlat-")) / "primnox.db")
    db.init()
    from primnox2.cognition import topics
    rng = random.Random(7)
    base = 1_735_689_600_000
    rows = []
    for i in range(n):
        text = (f"{rng.choice(SUBJECTS).capitalize()} {rng.choice(PREDICATES).format(x=rng.choice(FILLERS))}, "
                f"and {rng.choice(SUBJECTS)} {rng.choice(PREDICATES).format(x=rng.choice(FILLERS))} ({i}).")
        topic, kind = topics.classify(text)
        rows.append((f"mem_{uuid.uuid4().hex[:12]}", text, "personal", "imported", topic or "", kind,
                     base + i * DAY // 4, base + i * DAY // 4))
    with db.tx() as c:
        c.executemany("INSERT INTO memories (id,text,category,provenance,topic,kind,created_at,updated_at)"
                      " VALUES (?,?,?,?,?,?,?,?)", rows)


def timed(search, runs: int) -> tuple[float, float]:
    samples = []
    for i in range(runs):
        t = time.perf_counter()
        search(QUERIES[i % len(QUERIES)], limit=10)
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.95 * len(samples)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    from primnox2.memory import embeddings, service as mem
    from primnox2.storage import db
    from primnox2.tools import retrieval

    print(f"encoder: {retrieval.ENCODER}")
    loaded_at = None
    for n in args.sizes:
        build(n)
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "0"
        mem.search("warm the lexical path", limit=10)          # topic backfill, imports
        p50, p95 = timed(mem.search, args.runs)
        print(f"\n{n} memories")
        print(f"  lexical            p50 {p50:7.2f} ms   p95 {p95:7.2f} ms")

        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "1"
        if loaded_at is None:
            start = time.perf_counter()
            t = time.perf_counter()
            first = mem.search(QUERIES[0], limit=10)
            first_ms = (time.perf_counter() - t) * 1000
            ready = embeddings.prepare(300)
            loaded_at = (time.perf_counter() - start) if ready else None
            print(f"  first use          {first_ms:7.2f} ms (answered by words, {len(first)} hits); "
                  f"encoder ready after {loaded_at:.1f} s" if ready else "  encoder failed to load")
            if not ready:
                return 1

        with db.tx() as c:
            c.execute("UPDATE memories SET embedding=NULL")
        t = time.perf_counter()
        mem.search(QUERIES[0], limit=10)
        cold = (time.perf_counter() - t) * 1000
        print(f"  hybrid, cold cache {cold:7.1f} ms   (one search embeds all {n}; once per memory ever)")

        p50, p95 = timed(mem.search, args.runs)
        print(f"  hybrid, warm       p50 {p50:7.2f} ms   p95 {p95:7.2f} ms")
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "0"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
