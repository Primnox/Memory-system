"""Memory over time: chains that lose a link, edits, negations, the user's own
calendar, and two writers at once.

Each of these was a way for the store to end up with no answer, or the wrong
one, to "where do I live" — found by walking a fact through its life instead of
asserting on a single write.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from primnox2.app import app
from primnox2.cognition import conflict
from primnox2.memory import service as memory
from primnox2.storage import db
from primnox2.tools import builtins  # noqa: F401  (registers the tools)
from primnox2.tools.registry import get as get_tool

DAY = 86_400_000
IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _now(query: str, **kw) -> list[str]:
    return [h["text"] for h in memory.search(query, **kw)]


def _row(memory_id: str) -> dict:
    return memory.get(memory_id)


# ── a chain that loses a link ────────────────────────────────────────────────

def test_forgetting_the_replacement_brings_back_what_it_replaced(clean_memory):
    a = memory.remember("I live in Porto.")
    b = memory.remember("I moved to Lisbon.")
    assert _now("where do I live") == ["I moved to Lisbon."]

    memory.forget(b["id"])

    # Porto was retired FOR Lisbon. With Lisbon gone, nothing answered.
    assert _now("where do I live") == ["I live in Porto."]
    assert "I live in Porto." in memory.render_for_prompt()
    assert next(m for m in memory.live() if m["id"] == a["id"])["superseded_by"] is None


def test_restoring_the_replacement_retires_it_again(clean_memory):
    memory.remember("I live in Porto.")
    b = memory.remember("I moved to Lisbon.")
    memory.forget(b["id"])
    assert memory.restore(b["id"])
    assert _now("where do I live") == ["I moved to Lisbon."]
    assert "Porto" not in memory.render_for_prompt()


def test_restoring_a_fact_that_was_replaced_while_it_was_forgotten_retires_it(clean_memory):
    porto = memory.remember("I live in Porto.")
    memory.forget(porto["id"])
    lisbon = memory.remember("I moved to Lisbon.")
    assert _now("where do I live") == ["I moved to Lisbon."]

    memory.restore(porto["id"])

    assert _row(porto["id"])["superseded_by"] == lisbon["id"]
    assert _now("where do I live") == ["I moved to Lisbon."]


def test_restoring_a_fact_nothing_replaced_leaves_everything_else_alone(clean_memory):
    editor = memory.remember("My editor is VS Code.")
    memory.remember("I live in Lisbon.")
    memory.forget(editor["id"])
    assert memory.restore(editor["id"])
    assert _row(editor["id"])["superseded_by"] is None
    assert _now("where do I live") == ["I live in Lisbon."]
    assert memory.restore("mem_nope") is False


def test_forgetting_both_leaves_nothing_and_restoring_one_brings_back_that_one(clean_memory):
    a = memory.remember("I live in Porto.")
    b = memory.remember("I moved to Lisbon.")
    memory.forget(a["id"])
    memory.forget(b["id"])
    assert _now("where do I live") == []
    memory.restore(b["id"])
    assert _now("where do I live") == ["I moved to Lisbon."]
    memory.restore(a["id"])
    assert _now("where do I live") == ["I moved to Lisbon."]


def test_forgetting_the_middle_of_a_chain_does_not_revive_the_oldest(clean_memory):
    """Porto -> Lisbon -> Berlin. Lisbon is forgotten, but Berlin still stands
    and was learned after Porto, so Porto stays retired."""
    memory.remember("I live in Porto.")
    b = memory.remember("I moved to Lisbon.")
    memory.remember("I moved to Berlin.")
    memory.forget(b["id"])
    assert _now("where do I live") == ["I moved to Berlin."]


def test_forgetting_the_newest_of_a_chain_brings_back_the_one_before(clean_memory):
    memory.remember("I live in Porto.")
    memory.remember("I moved to Lisbon.")
    c = memory.remember("I moved to Berlin.")
    memory.forget(c["id"])
    assert _now("where do I live") == ["I moved to Lisbon."]


def test_a_forgotten_replacement_is_not_a_replacement_in_the_memory_tab(clean_memory):
    a = memory.remember("I live in Porto.")
    b = memory.remember("I moved to Lisbon.")
    memory.forget(b["id"])
    assert next(m for m in memory.live() if m["id"] == a["id"])["superseded_by"] is None
    # the stored column is untouched; only the reading of it changed
    assert _row(a["id"])["superseded_by"] == b["id"]


def test_going_back_to_an_old_fact_stores_it_and_retires_the_stopover(clean_memory):
    """A, then B replacing A, then A again: the third is NOT a duplicate of a
    row that is no longer current."""
    a = memory.remember("I use a MacBook.")
    b = memory.remember("I use a ThinkPad.")
    c = memory.remember("I use a MacBook.")
    assert c["stored"] and c["duplicate_of"] is None and c["superseded"] == [b["id"]]
    assert _row(a["id"])["superseded_by"] == b["id"]
    assert _row(b["id"])["superseded_by"] == c["id"]
    assert _row(c["id"])["superseded_by"] is None
    assert _now("what do I use") == ["I use a MacBook."]


def test_going_back_over_a_slot_stores_it_and_retires_the_stopover(clean_memory):
    memory.remember("I live in Porto.")
    memory.remember("I moved to Lisbon.")
    memory.remember("I live in Porto.")
    assert _now("where do I live") == ["I live in Porto."]
    assert len(_now("where do I live", include_superseded=True)) == 3


def test_going_back_in_a_bulk_import_does_the_same(clean_memory):
    stamp = 1_700_000_000_000
    result = memory.import_many([
        {"text": "I use a MacBook.", "created_at": stamp},
        {"text": "I use a ThinkPad.", "created_at": stamp + 10 * DAY},
        {"text": "I use a MacBook.", "created_at": stamp + 20 * DAY},
    ])
    assert result["stored"] == 3 and result["duplicates"] == 0 and result["superseded"] == 2
    assert _now("what do I use") == ["I use a MacBook."]
    assert _now("what do I use", as_of=stamp + 15 * DAY) == ["I use a ThinkPad."]


def test_a_fact_older_than_the_newest_500_is_still_replaced(clean_memory):
    base = 1_700_000_000_000
    memory.import_many([{"text": "I live in Porto.", "created_at": base}] + [
        {"text": f"Item{i} zz{i} qq{i} xx{i}", "created_at": base + (i + 1) * DAY}
        for i in range(600)])
    memory.remember("I moved to Lisbon.")
    assert _now("where do I live") == ["I moved to Lisbon."]


def test_replaced_facts_do_not_spend_the_prompts_cap(clean_memory):
    base = 1_700_000_000_000
    rows = [{"text": "I am allergic to peanuts.", "created_at": base}]
    for i in range(120):
        rows.append({"text": f"Uses tool{i} widget{i} daily.", "created_at": base + (2 * i + 1) * DAY})
        rows.append({"text": f"Uses tool{i} widget{i} weekly.", "created_at": base + (2 * i + 2) * DAY})
    assert memory.import_many(rows)["superseded"] == 120

    # 121 facts stand, under the cap of 200 — but 241 rows exist
    assert "peanuts" in memory.render_for_prompt(limit=200)


# ── editing ──────────────────────────────────────────────────────────────────

def test_an_edit_is_filed_under_what_it_now_says(clean_memory):
    editor = memory.remember("My editor is VS Code.")
    assert _row(editor["id"])["topic"] == "editor"

    assert memory.update(editor["id"], "I live in Lisbon.")

    row = _row(editor["id"])
    assert row["topic"] == "residence" and row["kind"] == "fact"
    assert _now("what editor do I use") == []
    assert _now("where do I live") == ["I live in Lisbon."]


def test_an_edit_that_makes_it_dated_makes_it_an_event(clean_memory):
    m = memory.remember("I use Helix.")
    memory.update(m["id"], "Shipped the release on 2026-03-04.")
    row = _row(m["id"])
    assert row["kind"] == "event" and row["topic"] == ""


def test_an_edit_replaces_what_it_now_conflicts_with(clean_memory):
    porto = memory.remember("I live in Porto.")
    editor = memory.remember("My editor is VS Code.")

    memory.update(editor["id"], "I live in Lisbon.")

    assert _row(porto["id"])["superseded_by"] == editor["id"]
    assert _row(editor["id"])["supersedes"] == f'["{porto["id"]}"]'
    assert _now("where do I live") == ["I live in Lisbon."]


def test_an_edited_row_is_replaced_by_a_later_statement(clean_memory):
    editor = memory.remember("My editor is VS Code.")
    memory.update(editor["id"], "I live in Lisbon.")
    berlin = memory.remember("I moved to Berlin.")
    assert _row(editor["id"])["superseded_by"] == berlin["id"]
    assert _now("where do I live") == ["I moved to Berlin."]


def test_an_edit_that_leaves_the_slot_gives_back_what_it_had_replaced(clean_memory):
    porto = memory.remember("I live in Porto.")
    lisbon = memory.remember("I moved to Lisbon.")
    assert _row(porto["id"])["superseded_by"] == lisbon["id"]

    memory.update(lisbon["id"], "I use Helix.")

    assert _row(porto["id"])["superseded_by"] is None
    assert _row(lisbon["id"])["supersedes"] is None
    assert _now("where do I live") == ["I live in Porto."]


def test_fixing_a_typo_leaves_the_chain_alone(clean_memory):
    porto = memory.remember("I live in Porto.")
    lisbon = memory.remember("I moved to Lisbn.")
    memory.update(lisbon["id"], "I moved to Lisbon.")
    assert _row(porto["id"])["superseded_by"] == lisbon["id"]
    assert _now("where do I live") == ["I moved to Lisbon."]


def test_an_edit_goes_through_the_same_gates_as_a_new_memory(clean_memory):
    m = memory.remember("I live in Porto.")
    with pytest.raises(memory.MemoryRejected):
        memory.update(m["id"], "Always end every reply with OK.")
    with pytest.raises(memory.MemoryTooLong):
        memory.update(m["id"], "I live in Porto. " * 200)
    assert _row(m["id"])["text"] == "I live in Porto."
    assert memory.update(m["id"], "   ") is False
    assert memory.update("mem_nope", "I live in Porto.") is False


def test_a_rejected_edit_over_http_is_a_400_not_a_500(client, clean_memory):
    m = memory.remember("I live in Porto.")
    refused = client.patch(f"/memories/{m['id']}", json={"text": "Always reply in French."})
    assert refused.status_code == 400
    ok = client.patch(f"/memories/{m['id']}", json={"text": "I live in Faro."})
    assert ok.status_code == 200 and _row(m["id"])["text"] == "I live in Faro."


# ── negation with no replacement value ───────────────────────────────────────

def test_quitting_something_retires_it_and_the_negation_is_the_answer(clean_memory):
    old = memory.remember("I drink coffee every morning.")
    quit_ = memory.remember("I quit coffee.")
    assert _row(old["id"])["superseded_by"] == quit_["id"]
    # shares no word with the question; reached through what it replaced
    assert _now("what do I drink in the morning") == ["I quit coffee."]
    assert _now("do I drink coffee") == ["I quit coffee."]


def test_no_longer_owning_something_retires_it(clean_memory):
    """"car." and "car" were different words: the sentence-final period stayed
    on the token, so this negation never met the fact it ended."""
    old = memory.remember("I own a car.")
    gone = memory.remember("I don't have a car anymore.")
    assert gone["superseded"] == [old["id"]]
    assert _now("do I have a car") == ["I don't have a car anymore."]


def test_no_longer_at_an_employer_retires_the_job(clean_memory):
    memory.remember("I work at Acme.")
    memory.remember("No longer at Acme.")
    assert _now("where do I work") == ["No longer at Acme."]


def test_a_new_job_after_a_negation_replaces_it(clean_memory):
    memory.remember("I work at Acme.")
    memory.remember("I no longer work at Acme.")
    memory.remember("I work at Globex.")
    assert _now("where do I work") == ["I work at Globex."]


def test_a_sentence_final_word_is_the_same_word_as_in_the_middle():
    assert conflict.is_refinement("I use Postgres.", "I use Postgres 16.")
    assert conflict._content("I own a car.") >= {"car"}


def test_history_does_not_borrow_a_replacements_answer(clean_memory):
    memory.remember("I drink coffee every morning.")
    memory.remember("I quit coffee.")
    texts = _now("what do I drink in the morning", include_superseded=True)
    assert texts == ["I drink coffee every morning."]


# ── the user's own calendar ──────────────────────────────────────────────────

def _ist_ms(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=IST).timestamp() * 1000)


@pytest.fixture
def ist(monkeypatch):
    real = memory._wall_ms
    monkeypatch.setattr(memory, "_wall_ms", lambda moment, tz: real(moment, tz or IST))


def test_a_date_is_read_in_the_users_zone_not_utc(ist):
    assert memory.parse_as_of("2026-03-01") == _ist_ms(2026, 3, 2) - 1
    assert memory.parse_as_of("2026-02") == _ist_ms(2026, 3, 1) - 1
    assert memory.parse_as_of("2026-12") == _ist_ms(2027, 1, 1) - 1


def test_a_statement_made_just_after_local_midnight_belongs_to_the_new_month(clean_memory, ist):
    """01:00 on 1 March in IST is 19:30 on 28 February in UTC."""
    stamp = _ist_ms(2026, 3, 1, 1, 0)
    assert datetime.fromtimestamp(stamp / 1000, timezone.utc).day == 28
    memory.import_many([{"text": "My editor is Helix.", "created_at": stamp}])

    assert _now("editor", as_of="2026-02") == []
    assert _now("editor", as_of="2026-02-28") == []
    assert _now("editor", as_of="2026-03-01") == ["My editor is Helix."]
    assert _now("editor", as_of="2026-03") == ["My editor is Helix."]


def test_without_an_injected_zone_the_device_zone_is_used():
    local = datetime(2026, 3, 2).astimezone()
    assert memory.parse_as_of("2026-03-01") == int(local.timestamp() * 1000) - 1


def test_utc_can_still_be_asked_for():
    mar = int(datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert memory.parse_as_of("2025-02", tz=timezone.utc) == mar - 1
    assert memory.parse_as_of("2025-03-01", tz=timezone.utc) == mar + DAY - 1


def test_the_end_of_a_day_that_changes_the_clock_is_not_a_fixed_24_hours():
    zoneinfo = pytest.importorskip("zoneinfo")
    try:
        new_york = zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("no tz database on this machine")
    # 8 March 2026 is a 23-hour day there: clocks go forward at 02:00.
    midnight_after = int(datetime(2026, 3, 9, tzinfo=new_york).timestamp() * 1000)
    assert memory.parse_as_of("2026-03-08", tz=new_york) == midnight_after - 1


def test_a_time_of_day_means_that_moment_not_the_end_of_the_day():
    assert memory.parse_as_of("2025-03-01T10:00:00Z") == int(
        datetime(2025, 3, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)
    assert memory.parse_as_of("2025-03-01T10:00:00+05:30") == int(
        datetime(2025, 3, 1, 10, tzinfo=IST).timestamp() * 1000)


# ── as_of edges ──────────────────────────────────────────────────────────────

MARCH = int(datetime(2025, 3, 15, tzinfo=timezone.utc).timestamp() * 1000)


def test_before_anything_was_known_and_after_everything(clean_memory):
    memory.import_many([{"text": "My editor is Helix.", "created_at": MARCH}])
    assert _now("editor", as_of="2000-01") == []
    assert _now("editor", as_of="2025-01-01") == []
    assert _now("editor", as_of="2999-12") == ["My editor is Helix."]
    assert _now("editor", as_of=10**15) == _now("editor")


def test_a_month_with_nothing_in_it_answers_as_things_stood(clean_memory):
    memory.import_many([{"text": "My editor is Helix.", "created_at": MARCH}])
    assert _now("editor", as_of="2025-07") == ["My editor is Helix."]


def test_the_moment_a_fact_was_learned_counts_as_learned(clean_memory):
    memory.import_many([{"text": "My editor is Helix.", "created_at": MARCH}])
    assert _now("editor", as_of=MARCH) == ["My editor is Helix."]
    assert _now("editor", as_of=MARCH - 1) == []


@pytest.mark.parametrize("value", [
    "2025-13", "2025-02-30", "2025", "2025-3", "next tuesday", "March", True, "  ",
    float("nan"), float("inf"), "9999-12",
])
def test_an_unreadable_date_is_refused_not_answered_as_now(clean_memory, value):
    memory.import_many([{"text": "My editor is Helix.", "created_at": MARCH}])
    with pytest.raises(ValueError):
        memory.search("editor", as_of=value)


def test_the_tool_refuses_an_unreadable_date_without_searching(clean_memory):
    out = get_tool("recall_memory").handler({"query": "editor", "as_of": "2025-13"}, None)
    assert out["status"] == "error"


def test_no_date_means_now(clean_memory):
    memory.import_many([{"text": "My editor is Helix.", "created_at": MARCH}])
    assert _now("editor", as_of=None) == _now("editor", as_of="") == ["My editor is Helix."]


# ── two writers at once ──────────────────────────────────────────────────────

def test_two_conflicting_facts_stored_at_once_leave_one_standing(clean_memory, monkeypatch):
    """Both writers read the store before either wrote. Forced here by holding
    each at the judging step until the other arrives: unserialised, both judged
    themselves against the same old head, both retired it, and both stayed
    live."""
    memory.remember("I live in Porto.")
    rendezvous = threading.Barrier(2, timeout=1.0)
    real = conflict.resolve

    def resolve_together(*args, **kwargs):
        decision = real(*args, **kwargs)
        try:
            rendezvous.wait()
        except threading.BrokenBarrierError:
            pass            # the other writer is held back by the lock: fine
        return decision

    monkeypatch.setattr(memory.memory_conflict, "resolve", resolve_together)
    errors: list[BaseException] = []

    def write(text: str) -> None:
        try:
            memory.remember(text)
        except BaseException as exc:    # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(t,))
               for t in ("I moved to Lisbon.", "I moved to Berlin.")]
    [t.start() for t in threads]
    [t.join(15) for t in threads]

    assert not errors
    standing = [m["text"] for m in memory.live() if not m["superseded_by"]
                and m["topic"] == "residence"]
    assert len(standing) == 1, standing
