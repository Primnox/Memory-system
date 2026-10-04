"""Time phrases ("back in March", "last summer") turned into dates in code, not by
the local 9B, which put "back in October" a year off.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from primnox2.memory import service as memory
from primnox2.memory import when
from primnox2.storage import db
from primnox2.tools import builtins  # noqa: F401  (registers the tools)
from primnox2.tools.registry import get as get_tool

D = date.fromisoformat


def _span(phrase, today):
    span = when.resolve(phrase, D(today))
    return None if span is None else (span.start.isoformat(), span.end.isoformat())


@pytest.mark.parametrize("phrase,today,start,end", [
    # a month by name is the latest one that has ENDED
    ("back in March", "2026-10-03", "2026-03-01", "2026-03-31"),
    ("in March", "2026-02-10", "2025-03-01", "2025-03-31"),
    ("in March", "2026-03-20", "2025-03-01", "2025-03-31"),      # still in March: last March
    ("in October", "2026-10-01", "2025-10-01", "2025-10-31"),
    ("last March", "2026-10-03", "2026-03-01", "2026-03-31"),
    ("in january", "2026-10-03", "2026-01-01", "2026-01-31"),
    ("last December", "2026-01-10", "2025-12-01", "2025-12-31"),   # year boundary
    ("last December", "2026-12-15", "2025-12-01", "2025-12-31"),
    ("this March", "2026-10-03", "2026-03-01", "2026-03-31"),
    # seasons, northern hemisphere
    ("last summer", "2026-10-03", "2026-06-01", "2026-08-31"),
    ("this summer", "2026-10-03", "2026-06-01", "2026-08-31"),      # asked in October
    ("last summer", "2026-07-15", "2025-06-01", "2025-08-31"),
    ("last summer", "2026-01-10", "2025-06-01", "2025-08-31"),
    ("last winter", "2026-10-03", "2025-12-01", "2026-02-28"),
    ("last winter", "2026-01-10", "2024-12-01", "2025-02-28"),      # this one is still running
    ("this winter", "2026-01-10", "2025-12-01", "2026-02-28"),
    ("in spring 2025", "2026-10-03", "2025-03-01", "2025-05-31"),
    ("winter 2025", "2026-10-03", "2024-12-01", "2025-02-28"),
    ("back in the summer", "2026-10-03", "2026-06-01", "2026-08-31"),
    # holidays
    ("around Christmas", "2026-10-03", "2025-12-24", "2025-12-26"),
    ("around Christmas", "2026-01-10", "2025-12-24", "2025-12-26"),
    ("last Christmas", "2026-12-30", "2026-12-24", "2026-12-26"),
    ("at New Year", "2026-01-10", "2025-12-31", "2026-01-02"),
    ("Christmas 2023", "2026-10-03", "2023-12-24", "2023-12-26"),
    # relative to today
    ("last month", "2026-10-03", "2026-09-01", "2026-09-30"),
    ("last month", "2026-01-05", "2025-12-01", "2025-12-31"),
    ("last year", "2026-10-03", "2025-01-01", "2025-12-31"),
    ("last week", "2026-10-05", "2026-09-28", "2026-10-04"),         # asked on a Monday
    ("yesterday", "2026-03-01", "2026-02-28", "2026-02-28"),
    ("two months ago", "2026-10-03", "2026-07-19", "2026-08-18"),
    ("two months ago", "2026-01-20", "2025-11-05", "2025-12-05"),
    ("2 months ago", "2026-04-30", "2026-02-13", "2026-03-15"),
    ("a few weeks ago", "2026-10-03", "2026-09-09", "2026-09-15"),
    ("a couple of weeks ago", "2026-10-03", "2026-09-16", "2026-09-22"),
    ("3 days ago", "2026-10-03", "2026-09-29", "2026-10-01"),
    ("a year ago", "2026-10-03", "2025-07-04", "2026-01-02"),
    ("earlier this year", "2026-10-03", "2026-01-01", "2026-09-30"),
    ("earlier this year", "2026-01-20", "2026-01-01", "2026-01-19"),
    # exact dates and sub-spans
    ("2025-03-14", "2026-10-03", "2025-03-14", "2025-03-14"),
    ("2025-03", "2026-10-03", "2025-03-01", "2025-03-31"),
    ("March 14th", "2026-10-03", "2026-03-14", "2026-03-14"),
    ("14 March 2025", "2026-10-03", "2025-03-14", "2025-03-14"),
    ("29 February", "2026-10-03", "2024-02-29", "2024-02-29"),
    ("in 2024", "2026-10-03", "2024-01-01", "2024-12-31"),
    ("toward the end of July", "2026-10-03", "2026-07-21", "2026-07-31"),
    ("early March", "2026-10-03", "2026-03-01", "2026-03-10"),
    ("mid-March", "2026-10-03", "2026-03-11", "2026-03-20"),
    ("the end of last year", "2026-10-03", "2025-09-01", "2025-12-31"),
])
def test_resolve(phrase, today, start, end):
    assert _span(phrase, today) == (start, end)


@pytest.mark.parametrize("phrase,today", [
    ("back then", "2026-10-03"), ("a while ago", "2026-10-03"), ("", "2026-10-03"),
    (None, "2026-10-03"), (7, "2026-10-03"), ("what do I use", "2026-10-03"),
    ("I may use it", "2026-10-03"), ("in the summer", "2026-10-03"),   # a habit, not a time
    ("this summer", "2026-04-01"), ("this March", "2026-02-01"),        # still ahead
    ("2025-13-40", "2026-10-03"), ("31 February", "2026-10-03"),
    ("99999 years ago", "2026-10-03"), ("0000-01", "2026-10-03"),
])
def test_what_is_not_a_time_resolves_to_nothing(phrase, today):
    assert when.resolve(phrase, D(today)) is None


@pytest.mark.parametrize("phrase", ["these days", "currently", "right now", "this month", "this year",
                                    "this week", "today"])
def test_the_present_resolves_to_now(phrase):
    today = D("2026-10-03")
    assert when.resolve(phrase, today) is not None
    assert when.resolve(phrase, today).point(today) is None


def test_a_phrase_inside_a_question():
    today = D("2026-09-20")
    assert when.as_of_for(query="Where were we living back in October?", today=today) == "2025-10-15"
    assert when.as_of_for(query="What phone was I using around Christmas?", today=today) == "2025-12-25"
    assert when.as_of_for(query="Which day was I swimming toward the end of July?", today=today) == "2026-07-26"
    assert when.as_of_for(query="What do I use these days?", today=today) is None


def test_midpoint_or_end():
    today = D("2026-10-03")
    span = when.resolve("back in March", today)
    assert span.point(today, "mid").isoformat() == "2026-03-15"
    assert span.point(today, "end").isoformat() == "2026-03-31"


def test_a_lone_month_counts_only_in_the_when_argument():
    today = D("2026-10-01")
    assert when.resolve("November", today) is None          # in a question it might be anything
    assert when.resolve("November", today, bare=True).start == D("2025-11-01")
    assert when.resolve("in the summer", today, bare=True) is None   # as likely a habit as a past
    assert when.as_of_for(when="November", today=today) == "2025-11-15"
    assert when.as_of_for(query="What was November like", today=today) is None


def test_the_middle_of_a_month_is_its_fifteenth():
    today = D("2026-10-03")
    for phrase in ("in February", "in March", "last month", "in 2025-04"):
        assert when.resolve(phrase, today).point(today).day == 15


def test_the_words_beat_the_models_own_date():
    today = D("2026-09-20")
    # the 9B wrote this year's October (still ahead) for "back in October"
    assert when.as_of_for(when="back in October", as_of="2026-10", today=today) == "2025-10-15"
    assert when.as_of_for(when="these days", as_of="2026-09", today=today) is None
    assert when.as_of_for(when="a while ago", as_of="2025-03", today=today) == "2025-03"
    assert when.as_of_for(as_of="2025-03", query="editor back in July", today=today) == "2025-03"
    assert when.as_of_for(when="garbage", query="editor back in July", today=today) == "2026-07-15"
    assert when.as_of_for(today=today) is None


# ── the tool path ────────────────────────────────────────────────────────────

JAN, JUN = 1_735_689_600_000, 1_748_736_000_000   # 2025-01-01, 2025-06-01 UTC


@pytest.fixture
def editor_history(monkeypatch):
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    memory.import_many([{"text": "My editor is VS Code.", "created_at": JAN},
                        {"text": "My editor is Helix now.", "created_at": JUN}])
    # a Friday in October 2025, in the device's own zone like the real clock
    monkeypatch.setattr(memory, "now_ms", lambda: int(datetime(2025, 10, 3, 12).timestamp() * 1000))
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


def _recall(**args):
    return get_tool("recall_memory").handler(args, None)


def test_tool_when_answers_as_things_stood_then(editor_history):
    out = _recall(query="editor", when="back in March")
    assert "VS Code" in out["output"] and "Helix" not in out["output"]


def test_tool_prefers_when_over_a_wrong_as_of(editor_history):
    out = _recall(query="editor", when="back in March", as_of="2026-03")
    assert "VS Code" in out["output"] and "Helix" not in out["output"]


def test_tool_present_words_search_now_even_with_a_date(editor_history):
    out = _recall(query="editor", when="these days", as_of="2025-03")
    assert "Helix" in out["output"] and "VS Code" not in out["output"]


def test_tool_reads_a_time_phrase_left_in_the_query(editor_history):
    assert "VS Code" in _recall(query="what editor did I use back in March")["output"]
    assert "Helix" in _recall(query="what editor do I use")["output"]


def test_tool_falls_back_to_as_of_for_words_it_cannot_read(editor_history):
    out = _recall(query="editor", when="a while back", as_of="2025-03")
    assert "VS Code" in out["output"] and "Helix" not in out["output"]


def test_tool_still_refuses_an_unreadable_as_of(editor_history):
    assert _recall(query="editor", as_of="someday")["status"] == "error"


def test_tool_offers_when():
    spec = get_tool("recall_memory")
    assert "when" in spec.parameters and "as_of" in spec.parameters
