"""Memory topics: what a statement is about decides whether it replaces another.

Before topics, `cognition/conflict.py` decided "update" from sentence shape
alone. On a 100-memory synthetic life that retired 63 memories that were never
stale — "Nadia is a CTO at X" replaced by "Omar is a Manager at X" — and every
retired memory silently left the prompt. These pin the behaviour that replaced
it, using the cases that were measured wrong.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from primnox2.cognition import topics
from primnox2.cognition.conflict import resolve
from primnox2.memory import service as memory
from primnox2.storage import db


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


def _retired() -> set[str]:
    return {r["text"] for r in db.connect().execute(
        "SELECT text FROM memories WHERE superseded_by IS NOT NULL")}


def _sim(a: str, b: str) -> float:
    import re
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# ── what a statement is about ────────────────────────────────────────────────

@pytest.mark.parametrize("text,topic", [
    ("Devan drinks black filter coffee now.", "coffee"),
    ("Devan moved to Neovim.", "editor"),          # "moved to" is NOT a residence
    ("I moved to Berlin.", "residence"),
    ("I live in Lisbon.", "residence"),
    ("My manager is Tomas Berg.", "manager"),
    ("Omar Okonkwo is a Engineering Manager at Halcyon Systems.", "role"),
    ("Works on project Cinder.", "project"),
    ("I use Redis for the job queue.", "queue"),
    ("I use Postgres 16 for the main database.", "database"),
    ("I switched to Rust for most of my work.", "language"),
    ("Devan takes the tram since moving.", "commute"),
    ("I am allergic to peanuts.", "allergy"),
    ("I prefer dark mode in every app.", "theme"),
    ("I use a MacBook.", None),                    # unnameable is None, never a guess
])
def test_topic_is_named_or_left_none(text, topic):
    assert topics.topic_of(text) == topic


@pytest.mark.parametrize("text", [
    "Met Nadia Okonkwo about Atlas Core at the annex (week 1).",
    "Reviewed a proposal with Elena Reyes about Drift (week 2).",
    "Shipped the release on 2026-03-04.",
])
def test_dated_log_lines_are_events_with_no_slot(text):
    assert topics.kind_of(text) == "event"
    assert topics.classify(text) == (None, "event")


def test_a_standing_fact_is_not_an_event():
    assert topics.kind_of("I use Helix.") == "fact"


# ── the conflict decision, on the slot ───────────────────────────────────────

def _decide(new, old, **kw):
    t, k = topics.classify(new)
    ot, ok = topics.classify(old)
    return resolve(new, [{"id": "old", "text": old, "topic": ot, "kind": ok}],
                   similarity=_sim, duplicate_threshold=0.85,
                   new_topic=t, new_kind=k, **kw).action


def test_same_single_valued_slot_with_a_new_value_is_an_update():
    assert _decide("My editor is Helix now.", "My editor is VS Code.") == "supersede"
    assert _decide("I quit coffee and drink green tea instead.",
                   "I drink black coffee in the morning.") == "supersede"


def test_a_different_slot_is_never_an_update_however_alike_the_sentences():
    assert _decide("I use Redis for the job queue.",
                   "I use Postgres for the main database.") == "new"


def test_a_multi_valued_slot_adds_rather_than_replaces():
    assert _decide("Works on project Granite.", "Works on project Cinder.") == "new"
    assert _decide("I am allergic to shellfish.", "I am allergic to peanuts.") == "new"


def test_an_additive_statement_does_not_replace():
    assert _decide("I also use Vim for quick edits.", "My editor is Helix.") == "new"


def test_two_different_people_in_the_same_role_shape_are_two_facts():
    assert _decide("Omar Okonkwo is a Manager at Halcyon Systems.",
                   "Nadia Okonkwo is a CTO at Halcyon Systems.") == "new"


def test_the_same_person_changing_role_is_an_update():
    assert _decide("Nadia Okonkwo is a VP at Halcyon Systems.",
                   "Nadia Okonkwo is a CTO at Halcyon Systems.") == "supersede"


def test_an_event_neither_retires_nor_is_retired():
    log = "Met Nadia Okonkwo about Atlas Core at the annex (week 1)."
    later = "Met Nadia Okonkwo about Atlas Core at the cafe (week 1)."
    assert _decide(later, log) == "new"


def test_without_topics_the_original_rules_still_apply():
    # A caller that passes no topic or kind behaves exactly as before.
    # The shared word has to identify the thing. This pair used to share only
    # "this project", which is on half the store and retired a risk rating in
    # the wild; with topics supplied the language slot decides it instead.
    r = resolve("I switched to tabs for indentation",
                [{"id": "o", "text": "I prefer spaces for indentation"}],
                similarity=_sim, duplicate_threshold=0.85)
    assert r.action == "supersede"
    unrelated = resolve("I switched to Rust for this project",
                        [{"id": "o", "text": "The risk rating is amber for this project"}],
                        similarity=_sim, duplicate_threshold=0.85)
    assert unrelated.action == "new"


# ── through the store ────────────────────────────────────────────────────────

def test_remember_does_not_retire_sibling_facts(clean_memory):
    for text in ["Nadia Okonkwo is a CTO at Halcyon Systems.",
                 "Omar Okonkwo is a Engineering Manager at Halcyon Systems.",
                 "Works on project Cinder.", "Works on project Granite.",
                 "Met Ravi Lindqvist about Atlas Core at the office (week 1).",
                 "Met Ravi Lindqvist about Atlas Core at a call (week 1)."]:
        memory.remember(text)
    assert _retired() == set()


def test_remember_retires_what_a_later_statement_replaces(clean_memory):
    memory.remember("My editor is VS Code.")
    memory.remember("I use Redis for the job queue.")
    memory.remember("My editor is Helix now.")
    assert _retired() == {"My editor is VS Code."}


def test_topic_and_kind_are_stored(clean_memory):
    memory.remember("I live in Lisbon.")
    memory.remember("Met Nadia about Atlas at the annex (week 2).")
    rows = {r["text"]: r for r in memory.live()}
    assert rows["I live in Lisbon."]["topic"] == "residence"
    assert rows["Met Nadia about Atlas at the annex (week 2)."]["kind"] == "event"


# ── bulk import ──────────────────────────────────────────────────────────────

def test_import_builds_supersession_chains_in_time_order(clean_memory):
    out = memory.import_many([
        # handed over newest-first on purpose: order must come from the stamps
        {"text": "My editor is Helix now.", "created_at": 3_000},
        {"text": "My editor is VS Code.", "created_at": 1_000},
        {"text": "I use Redis for the job queue.", "created_at": 2_000},
    ])
    assert out["superseded"] == 1
    assert _retired() == {"My editor is VS Code."}


def test_a_fact_that_returns_after_being_replaced_is_not_dropped_as_a_duplicate(clean_memory):
    out = memory.import_many([
        {"text": "My editor is VS Code.", "created_at": 1_000},
        {"text": "My editor is Helix now.", "created_at": 2_000},
        {"text": "My editor is VS Code.", "created_at": 3_000},
    ])
    assert out["duplicates"] == 0
    assert "My editor is Helix now." in _retired()


def test_an_import_stamps_the_retirement_with_the_successors_own_time(clean_memory):
    memory.import_many([{"text": "My editor is VS Code.", "created_at": 1_000},
                        {"text": "My editor is Helix now.", "created_at": 5_000}])
    row = db.connect().execute(
        "SELECT updated_at FROM memories WHERE text='My editor is VS Code.'").fetchone()
    assert row["updated_at"] == 5_000


# ── existing stores ──────────────────────────────────────────────────────────

def test_backfill_classifies_old_rows_and_is_idempotent(clean_memory):
    ts = 1
    with db.tx() as c:
        c.execute("INSERT INTO memories (id,text,created_at,updated_at) "
                  "VALUES ('m1','I live in Lisbon.',?,?)", (ts, ts))
        c.execute("INSERT INTO memories (id,text,created_at,updated_at) "
                  "VALUES ('m2','I use a MacBook.',?,?)", (ts, ts))
    assert memory.backfill_topics() == 2
    assert memory.backfill_topics() == 0          # '' marks "looked, nothing to name"
    rows = {r["id"]: r for r in memory.live()}
    assert rows["m1"]["topic"] == "residence"
    assert rows["m2"]["topic"] == ""


def test_backfill_never_changes_what_is_already_retired(clean_memory):
    with db.tx() as c:
        c.execute("INSERT INTO memories (id,text,created_at,updated_at) "
                  "VALUES ('new','My editor is Helix.',2,2)")
        c.execute("INSERT INTO memories (id,text,superseded_by,created_at,updated_at) "
                  "VALUES ('old','My editor is VS Code.','new',1,1)")
    memory.backfill_topics()
    assert _retired() == {"My editor is VS Code."}


def test_recheck_is_a_dry_run_until_applied(clean_memory):
    with db.tx() as c:
        c.execute("INSERT INTO memories (id,text,created_at,updated_at) "
                  "VALUES ('a','My editor is VS Code.',1,1)")
        c.execute("INSERT INTO memories (id,text,created_at,updated_at) "
                  "VALUES ('b','My editor is Helix now.',2,2)")
    proposals = memory.recheck_conflicts()
    assert [(p["old_id"], p["new_id"]) for p in proposals] == [("a", "b")]
    assert _retired() == set()                     # nothing changed yet
    memory.recheck_conflicts(apply=True)
    assert _retired() == {"My editor is VS Code."}


# ── the migration ────────────────────────────────────────────────────────────

def test_v11_database_upgrades_keeping_its_memories(tmp_path):
    path = tmp_path / "primnox.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at INTEGER);
        INSERT INTO schema_migrations VALUES (11, 'v11', 1);
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, category TEXT, provenance TEXT,
            conversation_id TEXT, turn_id TEXT, embedding BLOB, superseded_by TEXT,
            supersedes TEXT, disputed INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, deleted_at INTEGER);
        INSERT INTO memories (id,text,created_at,updated_at) VALUES ('m','I live in Lisbon.',1,1);
    """)
    conn.commit()
    conn.close()
    original = getattr(db, "_db_path", None)
    try:
        db.configure(path)
        db.init()
        db.init()                                   # idempotent
        row = db.connect().execute("SELECT text, topic, kind FROM memories").fetchone()
        assert row["text"] == "I live in Lisbon."   # not lost, not re-judged
        assert row["topic"] is None and row["kind"] == "fact"
        assert memory.backfill_topics() == 1
    finally:
        if original is not None:
            db.configure(original)


# ── names, scopes and attributes ─────────────────────────────────────────────

def test_the_persons_own_name_is_not_a_shared_subject():
    # "switched to" + the shared word "Devan" retired eight unrelated memories.
    r = resolve("Devan switched to cortados.",
                [{"id": "o", "text": "Devan prefers dark mode."}],
                similarity=_sim, duplicate_threshold=0.85)
    assert r.action == "new"


def test_statements_scoped_to_different_named_things_are_two_facts():
    assert _decide("Decided the alert routing for Ridge is one reviewer.",
                   "Decided the alert routing for Fathom is a queue.") == "new"


def test_a_different_attribute_of_the_same_thing_is_not_an_update():
    assert _decide("Decided the data model for Lantern is one reviewer.",
                   "Decided the alert routing for Lantern is Postgres.") == "new"


def test_a_new_value_for_the_same_attribute_is_an_update():
    assert _decide("The deadline for Atlas is Friday.",
                   "The deadline for Atlas is Tuesday.") == "supersede"


def test_a_place_is_a_value_not_a_scope():
    # "in" is deliberately not a scoping word: moving cities is an update.
    assert _decide("I live in Berlin.", "I live in Lisbon.") == "supersede"
