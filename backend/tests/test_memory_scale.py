"""Permanent memory at a few hundred rows and beyond.

Two kinds of test live here.

GOLDEN. `tests/golden/memory_ranking.json` was recorded from the implementation
that predates the scale work (a `SELECT *` of the whole table, per-row regexes,
no caches). Search results in every mode, which memories an import retires, what
`remember` stores, and the prompt block on a small store must match it exactly,
ties included, so a speed-up cannot move a ranking. To re-record after an
INTENDED behaviour change:

    PRIMNOX2_UPDATE_GOLDEN=1 pytest tests/test_memory_scale.py

BEHAVIOUR. What changed on purpose: `remember` looked only at the newest 500
rows, and the prompt block cut to the newest 200 rows before dropping retired
ones, so an allergy older than that never reached the model.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from primnox2.cognition import conflict
from primnox2.memory import service as memory
from primnox2.storage import db

GOLDEN = Path(__file__).parent / "golden" / "memory_ranking.json"
EPOCH = 1_735_689_600_000  # 2025-01-01 UTC
DAY = 86_400_000


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


# ── a fixed corpus ───────────────────────────────────────────────────────────
# (day, text). Chains where a later line really replaces an earlier one, lines
# that must NOT replace anything (refinement, other scope, events), a duplicate,
# and safety facts, spread among seeded filler. Days are whole, so many rows
# share a timestamp and tie-breaking is exercised.
HAND = [
    (2, "My name is Priya Raman."),
    (3, "I live in Lisbon."), (170, "I moved to Berlin."),
    (4, "I code in Python every day."), (200, "I switched to Rust for most of my work."),
    (5, "My editor is VS Code."), (210, "My editor is Helix now."),
    (6, "I drink black coffee in the morning."), (150, "I quit coffee and drink green tea instead."),
    (7, "I use Postgres for the main database."), (120, "I use Postgres 16 for the main database."),
    (8, "I use Redis for the job queue."), (9, "I use Redis for the job queue."),
    (10, "I take the metro to the office."), (190, "I cycle to the office now."),
    (11, "My manager is Tomas Berg."),
    (12, "I prefer dark mode in every app."),
    (13, "I take notes in Notion."), (180, "I moved my notes to Obsidian."),
    (14, "I am allergic to peanuts."), (15, "I am allergic to penicillin."),
    (16, "My son Leo has a severe nut allergy and carries an epi-pen."),
    (17, "I take insulin twice a day for type 1 diabetes."),
    (18, "The deadline for Atlas is Friday."), (160, "The deadline for Atlas is Monday."),
    (19, "The deadline for Beacon is Friday."),
    (20, "Met Nadia about Atlas at the annex (week 3)."),
    (21, "Met Nadia about Atlas at the annex (week 9)."),
    (22, "Nadia Okonkwo is a CTO at Fathom."), (23, "Omar Okonkwo is an Engineering Manager at Fathom."),
    (24, "My sister Anika lives in Pune."),
    (25, "Works on project Granite."), (26, "Works on project Cinder."),
    (27, "I also use Vim for quick edits."),
    (28, "I have a dentist appointment on 2025-03-04."),
    (140, "My favourite language is Elixir."), (141, "My favourite language is Haskell."),
]

_FIRST = "Aiko Amara Beatriz Carmen Dalia Elena Farah Greta Hana Idris Jonas Kavya Lena Marta Noor Petra Quinn Rhea".split()
_LAST = "Haddad Moreau Nakamura Petrov Rossi Silva Tanaka Varga Weiss Zhao Brandt Dumont Eriksen Garza Hoang".split()
_PROJ = "Atlas Beacon Cinder Delta Ember Falcon Granite Harvest Indigo Jetty Keystone Lantern".split()
_THING = "retry policy|release cadence|data model|error budget|rollout order|backup schedule|alert routing".split("|")
_CHOICE = "Postgres|a queue|two weeks|one reviewer|canary first|weekly|nightly|opt-in|Friday".split("|")
_FOOD = "dumplings|ramen|gnocchi|falafel|paella|focaccia|shakshuka|miso salmon".split("|")


def _filler(r: random.Random) -> str:
    name = f"{r.choice(_FIRST)} {r.choice(_LAST)}"
    proj = r.choice(_PROJ)
    return r.choice([
        lambda: f"{name} is a {r.choice(['Staff Engineer', 'Designer', 'Analyst', 'Recruiter'])} at {r.choice(_PROJ)}Co.",
        lambda: f"Decided the {r.choice(_THING)} for {proj} is {r.choice(_CHOICE)} (revision {r.randint(1, 4)}).",
        lambda: f"Met {name} about {proj} at the annex (week {r.randint(1, 60)}).",
        lambda: f"{name} recommended {r.choice(_FOOD)} near the station.",
        lambda: f"Works on project {proj} {r.choice(['Migration', 'Rollout', 'Audit'])}.",
        lambda: f"Bought {r.choice(['a standing desk', 'a road bike', 'a film camera'])} for the home office.",
        lambda: f"{name} asked about the database for {proj}.",
        lambda: f"{name} owns the queue dashboard for {proj}.",
    ])()


def fixed_corpus(n: int = 620) -> list[dict]:
    r = random.Random(20261001)
    rows = [{"text": t, "created_at": EPOCH + d * DAY} for d, t in HAND]
    while len(rows) < n:
        rows.append({"text": _filler(r), "created_at": EPOCH + r.randrange(30, 700) * DAY})
    return rows


QUERIES = [
    "what should you call me", "which city am I based in", "what language do I code in", "language",
    "which editor do I use", "editor", "what do I drink in the morning", "coffee", "database",
    "which database engine", "job queue", "queue", "how do I get to work", "commute", "theme",
    "light or dark appearance", "where do my notes live", "notes", "who is my manager", "manager",
    "allergy", "any food I must avoid", "epi-pen", "insulin", "diabetes", "deadline", "deadline Atlas",
    "Atlas", "Beacon", "Granite", "projects I work on", "Nadia", "who is a CTO", "Fathom", "Okonkwo",
    "favourite language", "Elixir", "Vim", "dentist", "Pune", "sister", "Berlin", "Lisbon",
    "Postgres 16", "Postgres", "Decided retry policy", "what did we decide about the rollout order",
    "database for Atlas", "queue dashboard", "standing desk", "film camera", "dumplings", "Staff Engineer",
    "week 3", "the", "what is the", "what do I", "zzzz nothing matches", "c++", "2025-03-04", "Rossi Silva",
]
AS_OF = ["2025-02", "2025-06-15", "2025-12", EPOCH + 300 * DAY]
AS_OF_QUERIES = QUERIES[::3]
LIMIT = 15


def _texts(hits: list[dict]) -> list[str]:
    return [h["text"] for h in hits]


def snapshot_outputs() -> dict:
    """Everything the golden file pins, as plain JSON-able data. Texts are
    interned into one table so the file stays readable and small."""
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    table: list[str] = []
    index: dict[str, int] = {}

    def ref(text: str) -> int:
        if text not in index:
            index[text] = len(table)
            table.append(text)
        return index[text]

    corpus = fixed_corpus()
    summary = memory.import_many(corpus)
    texts_by_id = {r["id"]: r["text"] for r in memory.live(limit=1_000_000)}
    edges = sorted((texts_by_id[r["id"]], texts_by_id[r["superseded_by"]])
                   for r in memory.live(limit=1_000_000) if r["superseded_by"])

    out: dict = {
        "import": {k: summary[k] for k in ("stored", "duplicates", "superseded")},
        "edges": [[ref(a), ref(b)] for a, b in edges],
        "search": {}, "search_all": {}, "as_of": {}, "browse": {},
    }
    for q in QUERIES:
        out["search"][q] = [ref(t) for t in _texts(memory.search(q, limit=LIMIT))]
        out["search_all"][q] = [ref(t) for t in _texts(memory.search(q, limit=LIMIT, include_superseded=True))]
    for q in AS_OF_QUERIES:
        for moment in AS_OF:
            out["as_of"][f"{q}|{moment}"] = [ref(t) for t in _texts(memory.search(q, limit=LIMIT, as_of=moment))]
    for name, kwargs in {"newest": {}, "newest_all": {"include_superseded": True},
                         "as_of_empty_query": {"as_of": "2025-06"}}.items():
        out["browse"][name] = [ref(t) for t in _texts(memory.search("", limit=LIMIT, **kwargs))]

    # The one-at-a-time path, on its own small store, with a clock that cannot tie.
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    tick = iter(range(EPOCH, EPOCH + 10**9, 1000))
    real_now = memory.now_ms
    memory.now_ms = lambda: next(tick)
    try:
        trace = []
        for row in sorted(corpus, key=lambda r: r["created_at"])[:140]:
            res = memory.remember(row["text"])
            by_id = {m["id"]: m["text"] for m in memory.live(limit=1_000_000)}
            trace.append([ref(row["text"]), res["stored"],
                          ref(by_id[res["duplicate_of"]]) if res["duplicate_of"] else None,
                          sorted(ref(by_id[i]) for i in res["superseded"])])
        out["remember"] = trace
        out["render"] = memory.render_for_prompt()
        out["render_session"] = memory.render_for_prompt(
            session_facts=[{"text": "I live in Oslo now."}, {"text": "My editor is Zed."}])
    finally:
        memory.now_ms = real_now
    out["texts"] = table
    return out


def test_ranking_and_supersession_are_unchanged_on_a_fixed_corpus(clean_memory):
    got = snapshot_outputs()
    if os.environ.get("PRIMNOX2_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(got, indent=0, ensure_ascii=False), encoding="utf-8")
        pytest.skip("recorded memory_ranking.json — re-run to verify")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def resolved(blob: dict, section: str) -> dict:
        t = blob["texts"]
        return {k: [t[i] for i in v] for k, v in blob[section].items()}

    for section in ("search", "search_all", "as_of", "browse"):
        a, b = resolved(got, section), resolved(want, section)
        diff = [f"{section}[{k!r}]:\n   was {b[k]}\n   now {a[k]}" for k in b if a.get(k) != b[k]]
        assert not diff, "ranking moved:\n" + "\n".join(diff[:4])
    t, w = got["texts"], want["texts"]
    assert [[t[a], t[b]] for a, b in got["edges"]] == [[w[a], w[b]] for a, b in want["edges"]]
    assert got["import"] == want["import"]
    assert [[t[x[0]], x[1], None if x[2] is None else t[x[2]], [t[i] for i in x[3]]] for x in got["remember"]] \
        == [[w[x[0]], x[1], None if x[2] is None else w[x[2]], [w[i] for i in x[3]]] for x in want["remember"]]
    assert got["render"] == want["render"]
    assert got["render_session"] == want["render_session"]


# ── what changed on purpose ──────────────────────────────────────────────────
def _filler_rows(n: int, *, start_day: int = 30) -> list[dict]:
    # Each line carries four tokens no other line has, so none of them is a
    # near-duplicate of, or an update to, another.
    return [{"text": f"Noted gadget{i} for bench{i} from shelf{i} near dock{i}.",
             "created_at": EPOCH + (start_day + i) * DAY} for i in range(n)]


def test_remember_still_retires_what_it_replaces_past_five_hundred_rows(clean_memory):
    memory.import_many([{"text": "I live in Lisbon.", "created_at": EPOCH}] + _filler_rows(650))
    res = memory.remember("I moved to Berlin.")
    old = next(m for m in memory.live(limit=1_000_000) if m["text"] == "I live in Lisbon.")
    assert res["superseded"] == [old["id"]]
    assert memory.get(old["id"])["superseded_by"] == res["id"]
    assert [h["text"] for h in memory.search("where do I live")][:1] == ["I moved to Berlin."]


def test_remember_still_recognises_a_restatement_past_five_hundred_rows(clean_memory):
    memory.import_many([{"text": "I am allergic to penicillin.", "created_at": EPOCH}] + _filler_rows(650))
    again = memory.remember("I am allergic to penicillin.")
    assert again["stored"] is False and again["duplicate_of"] is not None
    assert sum(1 for m in memory.live(limit=1_000_000) if m["text"] == "I am allergic to penicillin.") == 1


def test_an_old_allergy_still_reaches_the_prompt(clean_memory):
    """It is older than the newest 200 rows, so the old cut never selected it."""
    memory.import_many([{"text": "I am allergic to penicillin.", "created_at": EPOCH}] + _filler_rows(400))
    block = memory.render_for_prompt()
    assert "- I am allergic to penicillin." in block
    assert block.count("\n- ") == 200


def test_every_current_safety_fact_is_in_the_block_even_past_the_limit(clean_memory):
    allergies = [{"text": f"Guest {i} at the lodge is allergic to {food}.", "created_at": EPOCH + i * DAY}
                 for i, food in enumerate(["peanuts", "kiwi", "sesame", "mustard", "celery", "lupin"])]
    memory.import_many(allergies + _filler_rows(100, start_day=100))
    block = memory.render_for_prompt(limit=3)
    for a in allergies:
        assert f"- {a['text']}" in block
    assert block.count("\n- ") == len(allergies)


def test_retired_memories_never_spend_a_slot_or_appear(clean_memory):
    # The retired row is newer than some current ones, so a cut made before
    # retired rows are dropped would take a current fact's slot.
    rows = _filler_rows(30, start_day=10)
    rows += [{"text": "I live in Lisbon.", "created_at": EPOCH + 200 * DAY},
             {"text": "I moved to Berlin.", "created_at": EPOCH + 201 * DAY}]
    memory.import_many(rows)
    retired = {m["text"] for m in memory.live(limit=1_000_000) if m["superseded_by"]}
    assert "I live in Lisbon." in retired
    block = memory.render_for_prompt(limit=len(rows) - len(retired))
    assert "Lisbon" not in block
    lines = block.split("\n- ")[1:]
    assert len(lines) == len(rows) - len(retired)
    assert "I moved to Berlin." in lines


def test_the_safety_facts_lead_and_the_rest_is_newest_first(clean_memory):
    memory.import_many([
        {"text": "I am allergic to peanuts.", "created_at": EPOCH},
        {"text": "I enjoy hiking.", "created_at": EPOCH + 5 * DAY},
        {"text": "I carry an epi-pen everywhere.", "created_at": EPOCH + 2 * DAY},
        {"text": "I collect stamps.", "created_at": EPOCH + 9 * DAY},
    ])
    body = memory.render_for_prompt().split("\n- ")[1:]
    assert body == ["I carry an epi-pen everywhere.", "I am allergic to peanuts.",
                    "I collect stamps.", "I enjoy hiking."]


# ── the pieces the speed-up rests on ─────────────────────────────────────────
def test_search_hands_back_the_same_row_shape_as_live(clean_memory):
    memory.import_many([{"text": "My editor is Helix now.", "created_at": EPOCH}])
    keys = set(memory.live()[0])
    for hit in (memory.search("editor"), memory.search(""), memory.search("", include_superseded=True),
                memory.search("editor", as_of="2026-01")):
        assert hit and set(hit[0]) == keys


def test_nothing_to_backfill_is_answered_from_an_index_not_a_scan(clean_memory):
    memory.import_many(_filler_rows(50))
    plan = " ".join(r["detail"] for r in db.connect().execute(
        "EXPLAIN QUERY PLAN SELECT id, text FROM memories WHERE topic IS NULL"))
    assert "idx_memories_unclassified" in plan
    assert memory.backfill_topics() == 0


def test_backfill_still_classifies_rows_written_before_topics_existed(clean_memory):
    with db.tx() as c:
        c.execute("INSERT INTO memories (id,text,category,provenance,created_at,updated_at,topic)"
                  " VALUES ('mem_old1','I live in Lisbon.','personal','explicit',?,?,NULL)", (EPOCH, EPOCH))
    assert memory.backfill_topics() == 1
    assert memory.get("mem_old1")["topic"] == "residence"
    assert memory.backfill_topics() == 0


def test_the_conflict_fast_reject_changes_no_decision(clean_memory, monkeypatch):
    """resolve() skips candidates that cannot be superseded before running the
    full sequence on them. Run every corpus line against every other, with the
    skip on and off: the answers must be identical."""
    rows = [{"id": str(i), "text": r["text"], "topic": memory.memory_topics.classify(r["text"])[0] or "",
             "kind": memory.memory_topics.classify(r["text"])[1]}
            for i, r in enumerate(fixed_corpus(260))]

    def decisions() -> list:
        out = []
        for row in rows:
            topic, kind = memory.memory_topics.classify(row["text"])
            res = conflict.resolve(row["text"], rows, similarity=memory._similarity,
                                   duplicate_threshold=0.85, new_topic=topic, new_kind=kind)
            out.append(res.superseded_ids)
        return out

    fast = decisions()
    monkeypatch.setattr(conflict, "_PRESCREEN", False)
    assert fast == decisions()
    assert any(fast), "the corpus should retire something or this proves nothing"
