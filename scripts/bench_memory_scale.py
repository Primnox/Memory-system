"""How does permanent memory behave as the store grows?

`bench_memory.py` asks whether memory is RIGHT on a store of tens to hundreds of
facts. This asks what happens at 100 / 1k / 5k / 10k, in four parts, against
scratch databases that are never the app's:

  LATENCY   p50/p95 of `remember`, `search` and `render_for_prompt`.
  RANKING   known facts and hand-written questions, then 1k/5k/10k distractors:
            does the right fact stay on top as the store fills with look-alikes?
  PROMPT    which of N > 200 live facts reach the chat prompt, what is dropped,
            whether a retired fact can leak in, whether every current safety
            fact is always there, and what the downstream token clip keeps.
  WRITE     how much work the write/read paths do when there is nothing to do
            (the topic backfill, the full-store scan in `remember`).

The corpus is a seeded template mix (`corpus()`), not SDL: SDL's 12k-memory pack
is a generator-shaped life and its decisions are all one sentence form, which
flatters a lexical ranker. Targets and questions are written below by hand and
the questions never reuse a statement's wording.

Usage:
    python scripts/bench_memory_scale.py                   # everything
    python scripts/bench_memory_scale.py --sizes 100,1000  # quick
    python scripts/bench_memory_scale.py --only latency,prompt
    python scripts/bench_memory_scale.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
# This measures the word rules. The sentence-encoder arms are on by default in the
# app; bench_memory_latency.py is where they are timed.
os.environ.setdefault("PRIMNOX2_MEMORY_EMBEDDINGS", "0")
os.environ.setdefault("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE", "0")

DAY_MS = 86_400_000
SEED = 20261001

# ── vocabulary ───────────────────────────────────────────────────────────────
FIRST = ("Aiko Amara Anders Beatriz Bram Carmen Chidi Dalia Dmitri Elena Emeka Farah Felix Greta Hana "
         "Hugo Idris Ingrid Jamal Jonas Kavya Kofi Lena Leo Marta Mateo Mei Nadia Nikos Noor Omar Petra "
         "Quinn Rafael Rhea Samir Sanna Soren Talia Tomas Uma Viktor Wren Xavier Yara Zainab Zoran "
         "Ayesha Bruno Cleo Dario Esme Fiona Gideon Helga Ivo Jada Kenji Lucia Milos Neha Orla Pavel "
         "Ravi Selma Tariq Una Vera Waleed Ximena Yusuf Zelda Arjun Bianca Corin Dhruv Elif Faisal "
         "Gwen Haruto Imani Joon Karim Livia Mikael Nuria Osei Pilar Rangi Sunita Teodor Ulrik Vikram").split()
LAST = ("Okonkwo Raman Berg Castillo Haddad Lindqvist Moreau Nakamura Petrov Quispe Rossi Silva Tanaka "
        "Ueda Varga Weiss Yilmaz Zhao Abara Brandt Chaudhry Dumont Eriksen Fontaine Garza Hoang Ivanova "
        "Jansen Kowalski Laurent Mbeki Novak Ortega Pham Reyes Sandhu Thorne Underhill Vasquez Wójcik "
        "Xu Yamada Zielinski Adeyemi Bianchi Cardoso Dlamini Echeverria Farouk Grigoryan Holm Ito").split()
ORG = ("Fathom Ridge Northwind Lumen Tidewater Corvid Halcyon Ironbark Juniper Kestrel Marlow Nimbus "
       "Orchard Pinecone Quarry Redwood Saltmarsh Thistle Umbra Vellum Wintergreen Yarrow Zephyr "
       "Brightwater Cobalt Driftwood Emberly Foxglove Granite Harbor Isinglass").split()
ROLE = ("Staff Engineer|Engineering Manager|Product Designer|Data Analyst|Support Lead|CTO|Head of Sales|"
        "Finance Controller|Security Architect|Site Reliability Engineer|Research Scientist|Recruiter|"
        "Operations Manager|Technical Writer|QA Lead|Account Executive|Solutions Architect|Legal Counsel").split("|")
PROJECT_A = "Atlas Beacon Cinder Delta Ember Falcon Granite Harvest Indigo Jetty Keystone Lantern Meridian Nova Onyx Paragon Quasar Relay Summit Tundra Vector Willow Zenith".split()
PROJECT_B = ("Migration Rollout Revamp Pilot Audit Launch Refresh Rebuild Integration Cutover Review "
             "Rewrite Expansion Overhaul Initiative").split()
CITY = ("Lisbon Berlin Pune Oslo Nairobi Osaka Toronto Lyon Seville Gdansk Auckland Tbilisi Montreal Hanoi "
        "Cusco Reykjavik Split Bergen Porto Kyoto Antwerp Valencia Tallinn Cork Leeds Graz Malmo Utrecht").split()
PLACE = ("the annex|the roof terrace|the Mill Street cafe|the north canteen|a quiet room on floor 3|"
         "the airport lounge|the old library|the harbour office|a video call|the loading dock|"
         "the east wing|the greenhouse|the print room").split("|")
FOOD = ("lentil soup|mango lassi|sourdough|dumplings|shakshuka|ramen|gnocchi|banh mi|falafel|pierogi|"
        "paella|kimchi stew|pad thai|focaccia|chilli oil noodles|jollof rice|miso salmon|aloo gobi").split("|")
HOBBY = ("bouldering|pottery|chess|film photography|trail running|birdwatching|woodworking|sailing|"
         "calligraphy|salsa dancing|urban sketching|fermenting|beekeeping|astronomy|kite flying").split("|")
BOOK_A = "The Salt Road|A Quiet Engine|Winter Ledger|Paper Harbours|The Long Audit|Small Mercies|Iron Orchard|Night Ferry|The Cartographer's Debt|Low Tide Arithmetic".split("|")
RELATION = "sister brother cousin neighbour mentor landlord roommate friend uncle aunt".split()
SUBJECT = ("the deployment target|the retry policy|the on-call rota|the release cadence|the review threshold|"
           "the data model|the error budget|the naming convention|the test strategy|the rollout order|"
           "the backup schedule|the alert routing|the database migration|the job queue limit|"
           "the logging format|the cache lifetime|the vendor shortlist|the pricing tier").split("|")
CHOICE = ("Postgres|a queue|two weeks|one reviewer|canary first|a feature flag|weekly|nightly|opt-in|"
          "a hard cap|Friday|a shared doc|the staging cluster|a three-day freeze|SQLite|Redis").split("|")
VERB_OBJ = ("a standing desk|a second monitor|noise-cancelling headphones|a mechanical keyboard|a road bike|"
            "a film camera|a espresso grinder|a rice cooker|hiking boots|a standing lamp|a label printer").split("|")
TRIP = ("Lisbon in May|Hokkaido over New Year|a week on the Amalfi coast|the Atacama in October|"
        "a long weekend in Tallinn|Iceland in March|Marrakech in April|a rail trip across Poland").split("|")
SPORT = "squash badminton rowing climbing cycling swimming rugby tennis".split()


def _name(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _project(r: random.Random) -> str:
    return f"{r.choice(PROJECT_A)} {r.choice(PROJECT_B)}" if r.random() < 0.5 else r.choice(PROJECT_A)


def _week(r: random.Random) -> int:
    return r.randint(1, 100)


# (weight, builder). Each is a shape a person (or a model summarising a chat)
# would plausibly file. Some deliberately mention words the questions use
# ("database", "queue", "coffee", "allergic") so ranking has real look-alikes.
def _templates():
    return [
        (14, lambda r: f"{_name(r)} is a {r.choice(ROLE)} at {r.choice(ORG)}."),
        (8,  lambda r: f"Works on project {_project(r)}."),
        (16, lambda r: f"Met {_name(r)} about {_project(r)} at {r.choice(PLACE)} (week {_week(r)})."),
        (12, lambda r: f"Decided {r.choice(SUBJECT)} for {_project(r)} is {r.choice(CHOICE)} "
                       f"(revision {r.randint(1, 9)})."),
        (6,  lambda r: f"{_name(r)}'s {r.choice(RELATION)} {r.choice(FIRST)} likes {r.choice(FOOD)}."),
        (6,  lambda r: f"{r.choice(FIRST)} {r.choice(LAST)} took up {r.choice(HOBBY)} last spring."),
        (5,  lambda r: f"Bought {r.choice(VERB_OBJ)} for the home office."),
        (5,  lambda r: f"Is reading {r.choice(BOOK_A)} with {_name(r)}'s book club."),
        (5,  lambda r: f"Plans a trip to {r.choice(TRIP)} with {r.choice(FIRST)}."),
        (4,  lambda r: f"Plays {r.choice(SPORT)} on Thursdays with {_name(r)}."),
        (4,  lambda r: f"{_name(r)} recommended {r.choice(FOOD)} near {r.choice(CITY)} station."),
        (3,  lambda r: f"{_name(r)} asked about the database for {_project(r)}."),
        (3,  lambda r: f"{_name(r)} owns the queue dashboard for {_project(r)}."),
        (2,  lambda r: f"{_name(r)} brought coffee beans from {r.choice(CITY)} to the team."),
        (1,  lambda r: f"{_name(r)} mentioned a shellfish allergy at the {r.choice(ORG)} offsite."),
        (3,  lambda r: f"Wants to learn {r.choice(['Portuguese', 'Japanese', 'Norwegian', 'Swahili', 'Tamil'])} "
                       f"before visiting {r.choice(CITY)}."),
    ]


def corpus(n: int, seed: int = SEED, *, start: int = 0) -> list[dict]:
    """`n` varied statements, chronological over ~two years ending now.

    Deterministic for a given (seed, start); `start` lets a store grow by
    appending the next slice without regenerating the first.
    """
    r = random.Random(seed * 1_000_003 + start)
    templates = _templates()
    weights = [w for w, _ in templates]
    now = int(time.time() * 1000)
    span = 730 * DAY_MS
    out = []
    for i in range(n):
        _, build = r.choices(templates, weights=weights)[0]
        out.append({"text": build(r), "category": r.choice(("personal", "work", "project", "personal")),
                    "created_at": now - span + int(span * (start + i) / max(1, start + n + 1) * 0.98)
                    + r.randint(0, 3_600_000)})
    return out


# ── hand-written targets and questions ───────────────────────────────────────
# (statement, question that must bring it back). Questions paraphrase; none
# repeats the statement's wording. Single-valued slots are written as the user
# would say them, so supersession has something real to do around them.
TARGETS = [
    ("My name is Priya Raman.", "what should you call me"),
    ("I live in Lisbon.", "which city am I based in right now"),
    ("I code in Rust every day.", "what programming language do I mostly write"),
    ("My editor is Helix.", "which text editor do I open for code"),
    ("I drink flat whites in the morning.", "what do I have to drink first thing"),
    ("I use Postgres 16 for the main database.", "which database engine backs my app"),
    ("I cycle to the office.", "how do I get to work"),
    ("I prefer dark mode in every app.", "light or dark appearance"),
    ("I keep my notes in Obsidian.", "where do my notes live"),
    ("My manager is Tomas Berg.", "who do I report to"),
    ("I am allergic to peanuts.", "any food I must avoid because of an allergy to nuts"),
    ("I am allergic to penicillin.", "which antibiotic reaction should a doctor know about"),
    ("I work at Fathom as a staff engineer.", "who is my employer"),
    ("My daughter Mira turns seven in June.", "how old is my daughter getting"),
    ("My partner Jonas is vegetarian.", "does my partner eat meat"),
    ("I am training for the Porto half marathon in March.", "which race am I preparing for"),
    ("My passport expires in November 2027.", "when do I need to renew my travel document"),
    ("Our team standup is at 9:30 on Zoom.", "what time is the daily sync meeting"),
    ("I use Redis for the job queue.", "what backs the background jobs"),
    ("My landlord is Mrs Okafor and rent is due on the first.", "when do I pay for the flat"),
    ("I own a Kestrel road bike with a carbon frame.", "what bicycle do I ride"),
    ("My favourite author is Ursula Le Guin.", "which writer do I like best"),
    ("I speak Portuguese and English at home.", "what languages are used in my household"),
    ("My dentist is Dr Halloran on Mill Street.", "who looks after my teeth"),
    ("I keep the sourdough starter in the pantry.", "where is the bread culture stored"),
    ("My car is a blue 2019 Skoda Octavia.", "what vehicle do I drive"),
    ("My blood type is O negative.", "which blood group am I"),
    ("I volunteer at the food bank on Saturdays.", "what do I do on weekend mornings for charity"),
    ("My sister Anika lives in Pune.", "where does my sibling reside"),
    ("I am saving for a deposit on a flat near the river.", "what am I putting money aside for"),
]
# The same facts asked the way a model calls recall_memory: a keyword or a short
# noun phrase. The paraphrases above share few or no words with their fact (a
# lexical ranker cannot find "who looks after my teeth" -> "dentist"; that is a
# design limit, not a scale effect), so they are reported apart from these.
KEYWORDS = [
    "my name", "where do I live", "programming language", "editor", "coffee", "database",
    "commute to the office", "theme", "notes", "who is my manager", "allergic peanuts", "penicillin",
    "employer", "daughter Mira", "Jonas vegetarian", "half marathon", "passport", "standup",
    "job queue", "landlord rent", "road bike", "favourite author", "languages spoken at home",
    "dentist", "sourdough starter", "car", "blood type", "food bank volunteer", "Anika", "deposit flat",
]
assert len(KEYWORDS) == len(TARGETS)

# Statements whose job is to be current when the prompt block is built.
PROMPT_SAFETY = [
    "I am allergic to penicillin.",
    "My son Leo has a severe nut allergy and carries an epi-pen.",
    "I take insulin twice a day for type 1 diabetes.",
    "I cannot take ibuprofen because of asthma.",
    "My partner has coeliac disease so everything at home is gluten-free.",
]


# ── helpers ──────────────────────────────────────────────────────────────────
def pcts(values_ms: list[float]) -> dict:
    v = sorted(values_ms)
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50": round(statistics.median(v), 2),
            "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))], 2), "max": round(v[-1], 2)}


def rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:
        return None


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return (time.perf_counter() - t) * 1e3, out


# ── stores ───────────────────────────────────────────────────────────────────
# Stores are built once per size and kept as snapshots, so a before/after run
# measures the same rows, and so the (slow, O(n^2)) import is not repaid by every
# section. `remember` and friends are always timed on a COPY.
def open_db(path: Path):
    from primnox2.storage import db
    db.configure(path)
    db.init()
    return db


def snapshot(db, dest: Path) -> None:
    import sqlite3
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    src = db.connect()
    src.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out = sqlite3.connect(str(dest))
    src.backup(out)
    out.close()


def working_copy(snap: Path):
    import shutil
    path = Path(tempfile.mkdtemp(prefix="bench-memory-scale-")) / "primnox.db"
    shutil.copyfile(snap, path)
    return open_db(path)


def counts(db) -> dict:
    row = db.connect().execute(
        "SELECT COUNT(*) total, SUM(deleted_at IS NULL AND superseded_by IS NULL) live, "
        "SUM(deleted_at IS NULL AND superseded_by IS NOT NULL) retired FROM memories").fetchone()
    return {"rows": row["total"], "live": row["live"] or 0, "retired": row["retired"] or 0}


def plant(mem) -> None:
    """The oldest facts in the store: the safety facts the prompt must never
    lose, then the ranking targets, spread over the last year."""
    now = int(time.time() * 1000)
    rows = [{"text": t, "created_at": now - 700 * DAY_MS + i} for i, t in enumerate(PROMPT_SAFETY)]
    rows += [{"text": t, "created_at": now - (len(TARGETS) - i) * 11 * DAY_MS}
             for i, (t, _) in enumerate(TARGETS)]
    mem.import_many(rows)


def build_chain(mem, store_dir: Path, chain: list[int], rebuild: bool, log) -> dict[int, Path]:
    """Snapshot per size. A size's snapshot is the previous one grown, so the
    10k store contains the 1k store's rows."""
    snaps = {n: store_dir / f"store-{n}.db" for n in chain}
    if not rebuild and all(p.exists() for p in snaps.values()):
        return snaps
    work = Path(tempfile.mkdtemp(prefix="bench-memory-scale-build-")) / "primnox.db"
    db = open_db(work)
    spent = 0.0
    cursor = 0
    for n in chain:
        if n == 0:
            plant(mem)
        while counts(db)["rows"] < n:
            need = max(50, int((n - counts(db)["rows"]) * 1.05))
            ms, res = timed(mem.import_many, corpus(need, start=cursor))
            cursor += need
            spent += ms
            log(f"    {counts(db)['rows']:>6} rows  (+{res['stored']}, {res['duplicates']} dup, "
                f"{res['superseded']} retired)  {ms / 1e3:6.1f}s")
        snapshot(db, snaps[n])
        log(f"  snapshot {n}: {counts(db)}  cumulative import {spent / 1e3:.0f}s")
    return snaps


# ── 1 + 4. latency and write-path cost ───────────────────────────────────────
def probe_queries(r: random.Random, n: int) -> list[str]:
    """Questions of the shapes a model sends to recall_memory: a bare keyword,
    a paraphrase, and a name from the corpus. Some match nothing."""
    qs = [q for _, q in TARGETS]
    qs += ["database", "queue", "allergy", "coffee", "editor", "where do I live", "project", "notes",
           "what did we decide about the retry policy", "who is a staff engineer", "trip", "book club",
           "something nobody ever said xyzzy"]
    qs += [f"what do I know about {_name(r)}" for _ in range(8)]
    qs += [f"{r.choice(PROJECT_A)} {r.choice(PROJECT_B)}" for _ in range(8)]
    return (qs * (n // len(qs) + 1))[:n]


def fresh_statements(r: random.Random, n: int, salt: int) -> list[str]:
    return [f"{_name(r)} prefers {r.choice(FOOD)} when visiting {r.choice(CITY)} (note {salt}-{i})."
            for i in range(n)]


def run_latency(mem, reps: int) -> dict:
    r = random.Random(SEED)
    out = {}
    queries = probe_queries(r, reps)
    out["cold"] = {"first_search_ms": round(timed(mem.search, "database")[0], 1),
                   "first_render_ms": round(timed(mem.render_for_prompt)[0], 1)}
    out["search"] = pcts([timed(mem.search, q, limit=10)[0] for q in queries])
    out["search_incl_retired"] = pcts([timed(mem.search, q, limit=10, include_superseded=True)[0]
                                       for q in queries[:max(10, reps // 3)]])
    out["render"] = pcts([timed(mem.render_for_prompt)[0] for _ in range(max(10, reps // 2))])
    facts = [{"text": "I live in Oslo now."}, {"text": "My editor is Zed."}, {"text": "I switched to Go."}]
    out["render+session"] = pcts(
        [timed(mem.render_for_prompt, session_facts=facts)[0] for _ in range(max(10, reps // 4))])
    stmts = fresh_statements(r, max(15, reps // 3), salt=r.randint(0, 10**6))
    out["remember"] = pcts([timed(mem.remember, s, provenance="explicit")[0] for s in stmts])
    return out


def run_write_path(mem, db) -> dict:
    """What a call costs when there is nothing to do, counted in SQL statements
    (via a trace on the connection), so it measures work rather than this
    machine's mood. `rows_pulled` is how many rows the call's SELECTs return."""
    conn = db.connect()
    r = random.Random(SEED + 1)
    seen: list[str] = []
    out = []

    def tally(label, fn, reps=5):
        ms, selects, scans = [], [], []
        for _ in range(reps):
            seen.clear()
            ms.append(timed(fn)[0])
            sel = [s for s in seen if s.lstrip().upper().startswith("SELECT")]
            selects.append(len(sel))
            scans.append(sum(1 for s in sel if "FROM memories" in s and "COUNT" not in s
                             and "WHERE id IN" not in s and "WHERE id=" not in s))
        out.append({"call": label, "ms_p50": round(statistics.median(ms), 2),
                    "selects_per_call": max(selects), "whole_table_selects_per_call": max(scans)})

    conn.set_trace_callback(seen.append)
    try:
        tally("backfill_topics(), nothing to backfill", mem.backfill_topics)
        tally("search('database')", lambda: mem.search("database"))
        tally("remember(new fact)", lambda: mem.remember(fresh_statements(r, 1, r.randint(0, 10**9))[0],
                                                          provenance="explicit"))
        tally("render_for_prompt()", mem.render_for_prompt)
    finally:
        conn.set_trace_callback(None)
    plan = [row["detail"] for row in conn.execute(
        "EXPLAIN QUERY PLAN SELECT id, text FROM memories WHERE topic IS NULL")]
    return {"calls": out, "backfill_query_plan": plan}


# ── 2. ranking under distractors ─────────────────────────────────────────────
def run_ranking(mem, db, found_with_no_distractors: dict | None = None) -> dict:
    """Top-1/top-3 of the planted facts for two question styles. A fact retired
    by a distractor cannot be returned (search hides retired rows), so each
    figure is given over the facts still standing and over all of them; the
    retirement is itself a finding, not a ranking miss.

    `found_with_no_distractors` is {style: set(targets top-1 at stage 0)}; with
    it, `kept` is how many of those are STILL top-1 now — the number that
    isolates what growth did from what the ranker could never do."""
    texts = [t for t, _ in TARGETS]
    rows = {r["text"]: dict(r) for r in db.connect().execute(
        "SELECT text, superseded_by FROM memories WHERE text IN (%s)" % ",".join("?" * len(texts)), texts)}
    live_t = [t for t in texts if rows.get(t) and not rows[t]["superseded_by"]]
    out = {"targets": len(texts), "retired_by_distractors": len(texts) - len(live_t)}
    top1_now = {}
    for style, questions in (("paraphrase", [q for _, q in TARGETS]), ("keyword", KEYWORDS)):
        ranks, tops = {}, {}
        for (t, _), q in zip(TARGETS, questions):
            hits = [h["text"] for h in mem.search(q, limit=10)]
            ranks[t] = hits.index(t) + 1 if t in hits else None
            tops[t] = hits[0] if hits else None

        def top(k, pool):
            return sum(1 for t in pool if ranks[t] and ranks[t] <= k)

        top1_now[style] = {t for t in texts if ranks[t] == 1}
        res = {"top1_of_live": f"{top(1, live_t)}/{len(live_t)}", "top3_of_live": f"{top(3, live_t)}/{len(live_t)}",
               "top1_of_all": f"{top(1, texts)}/{len(texts)}", "top3_of_all": f"{top(3, texts)}/{len(texts)}",
               "misses": [{"q": q, "want": t, "rank": ranks[t], "got": tops[t], "retired": t not in live_t}
                          for (t, _), q in zip(TARGETS, questions) if not ranks[t] or ranks[t] > 3]}
        if found_with_no_distractors is not None:
            base = found_with_no_distractors[style]
            res["kept_top1"] = f"{len(base & top1_now[style])}/{len(base)}"
        out[style] = res
    out["_top1"] = {k: sorted(v) for k, v in top1_now.items()}
    return out


# ── 3. what reaches the prompt ───────────────────────────────────────────────
def run_prompt(mem, db) -> dict:
    from primnox2.context import service as ctx
    from primnox2.settings import tunables
    budget = int(tunables.get("context.memory_tokens"))
    cpt = float(tunables.get("context.chars_per_token"))
    limit = 200
    live = mem.live(limit=1_000_000)
    retired_texts = {m["text"] for m in live if m.get("superseded_by")}
    current = [m for m in live if not m.get("superseded_by")]
    current_texts = {m["text"] for m in current}
    safety = [m for m in current if mem._is_safety_fact(m["text"])]
    planted = [m for m in current if m["text"] in set(PROMPT_SAFETY)]

    block = mem.render_for_prompt(limit=limit)
    clipped = ctx._clip(block, budget)

    def audit(b):
        ls = {ln[2:] for ln in b.split("\n") if ln.startswith("- ")}
        kept = [m for m in current if m["text"] in ls and not mem._is_safety_fact(m["text"])]
        dropped = [m for m in current if m["text"] not in ls and not mem._is_safety_fact(m["text"])]
        return {
            "lines": len(ls), "chars": len(b),
            "retired_in_block": len((ls & retired_texts) - current_texts),
            "safety_in_block": f"{sum(1 for m in safety if m['text'] in ls)}/{len(safety)}",
            "planted_safety_in_block": f"{sum(1 for m in planted if m['text'] in ls)}/{len(planted)}",
            "non_safety_kept_are_newest": (not kept or not dropped
                                           or min(m["created_at"] for m in kept)
                                           >= max(m["created_at"] for m in dropped)),
            "non_safety_dropped": len(dropped),
        }
    return {"live": len(current), "retired_rows": len(live) - len(current), "safety_facts": len(safety),
            "limit": limit, "budget_tokens": budget, "budget_chars": int(budget * cpt),
            "render": audit(block), "after_clip": audit(clipped)}


# ── driver ───────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="100,1000,5000,10000", help="store sizes for latency")
    ap.add_argument("--only", default="latency,write,ranking,prompt")
    ap.add_argument("--reps", type=int, default=60, help="search calls per size")
    ap.add_argument("--store-dir", default=str(Path(tempfile.gettempdir()) / "bench-memory-scale-stores"))
    ap.add_argument("--rebuild", action="store_true", help="rebuild the store snapshots")
    ap.add_argument("--json")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    only = set(args.only.split(","))
    log = lambda s: print(s, flush=True)

    from primnox2.memory import service as mem
    ranking_sizes = [0, 1_000, 5_000, 10_000]
    prompt_sizes = [150, 300, 1_000, 5_000, 10_000]
    chain = sorted({0, *(sizes if only & {"latency", "write"} else []),
                    *(ranking_sizes if "ranking" in only else []),
                    *(prompt_sizes if "prompt" in only else [])})
    log(f"-- stores {chain} in {args.store_dir}")
    snaps = build_chain(mem, Path(args.store_dir), chain, args.rebuild, log)
    report: dict = {"sizes": sizes}

    if "latency" in only:
        report["latency"] = []
        for n in sizes:
            db = working_copy(snaps[n])
            row = {"size": n, **counts(db), **run_latency(mem, args.reps), "rss_mb": rss_mb()}
            report["latency"].append(row)
            log(json.dumps(row))
    if "write" in only:
        db = working_copy(snaps[max(sizes)])
        report["write"] = run_write_path(mem, db)
        log(json.dumps(report["write"]))
    if "ranking" in only:
        report["ranking"] = {}
        stage0 = None
        for n in ranking_sizes:
            db = working_copy(snaps[n])
            res = run_ranking(mem, db, stage0)
            if stage0 is None:
                stage0 = {k: set(v) for k, v in res["_top1"].items()}
            report["ranking"][n] = res
            brief = {k: ({kk: vv for kk, vv in v.items() if kk != "misses"} if isinstance(v, dict) else v)
                     for k, v in res.items() if k != "_top1"}
            log(f"distractors {n:>6}: " + json.dumps(brief))
    if "prompt" in only:
        report["prompt"] = {}
        for n in prompt_sizes:
            db = working_copy(snaps[n])
            res = run_prompt(mem, db)
            report["prompt"][n] = res
            log(f"rows {n:>6}: " + json.dumps(res))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
