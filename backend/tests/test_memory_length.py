"""A memory is one fact, and the store enforces it.

Nothing capped a memory's length. The context service injects the whole store
into every prompt — its own comment says "injected whole, because it is small by
construction" — but nothing constructed it small. A model that passed a whole
paragraph instead of a distilled fact stored the paragraph, and that one memory
then spent the entire `context.memory_tokens` budget, clipping every other fact
out of the prompt. The user's real preferences went missing to make room for one
verbose one, and nothing said so.
"""
from __future__ import annotations

import pytest

from primnox2.memory import service as memory
from primnox2.settings import tunables
from primnox2.storage import db
from primnox2.tools import builtins
from primnox2.tools.registry import ToolContext


@pytest.fixture
def clean_memory(fresh_db):
    """An empty store either side. `remember` scans every live memory to reject
    duplicates, so a row left behind by another test changes the answer here."""
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


@pytest.fixture
def cap() -> int:
    return int(tunables.get("memory.max_chars"))


def test_a_normal_fact_is_stored(clean_memory):
    assert memory.remember("Prefers concise answers.")["stored"] is True


def test_a_pasted_paragraph_is_refused(clean_memory, cap):
    """The actual failure: a whole message handed to `remember` verbatim."""
    paragraph = (
        "So the way I usually work is that I start in the morning with the "
        "highest priority item, and I really do not like being interrupted "
        "before lunch, although after two o'clock I am generally happy to take "
        "meetings, and on Fridays I try to keep the afternoon clear entirely "
        "for deep work on whatever the current project happens to be."
    )
    assert len(paragraph) > cap
    with pytest.raises(memory.MemoryTooLong):
        memory.remember(paragraph)
    assert memory.live() == [], "the paragraph was stored despite being refused"


def test_the_limit_is_a_refusal_not_a_truncation(clean_memory, cap):
    """Truncation can reverse a fact.

    'Does not want the report sent to marketing' cut short becomes 'Does not
    want the report sent', which states something the user never said and means
    close to the opposite. A memory that is wrong is worse than one that is
    missing, so the store refuses instead of trimming.
    """
    text = "Does not want the report sent to marketing " + ("x" * cap)
    with pytest.raises(memory.MemoryTooLong):
        memory.remember(text)
    stored = [m["text"] for m in memory.live()]
    assert not any(s.startswith("Does not want the report sent") for s in stored)


def test_a_fact_exactly_at_the_limit_is_allowed(clean_memory, cap):
    """The boundary is inclusive — an off-by-one here silently rejects valid
    facts near the edge."""
    assert memory.remember("y" * cap)["stored"] is True


def test_the_limit_is_tunable(clean_memory):
    """It is a knob, not a constant, because 'one fact' is a judgement and the
    right value depends on how much context the model has to spend."""
    row = next(t for t in tunables.describe() if t["key"] == "memory.max_chars")
    assert row["type"] == "int"
    assert row["min"] <= row["value"] <= row["max"]


def test_the_tool_reports_it_as_advice_rather_than_failing_the_turn(clean_memory, cap):
    """The model has to be able to recover. A raised exception would fail the
    whole turn over a fact that only needed shortening."""
    # Null attribution rather than invented ids: conversation_id and turn_id are
    # real foreign keys, and a memory with no conversation is a legitimate state
    # (they are ON DELETE SET NULL — a fact outlives the chat it was said in).
    ctx = ToolContext(conversation_id=None, turn_id=None)
    result = builtins._remember({"text": "z" * (cap + 50)}, ctx)
    assert result["status"] == "error"
    assert "distil" in result["output"].lower()
    assert str(cap) in result["output"], "the model is not told what the limit is"


def test_the_tool_still_saves_a_short_fact(clean_memory):
    # Null attribution rather than invented ids: conversation_id and turn_id are
    # real foreign keys, and a memory with no conversation is a legitimate state
    # (they are ON DELETE SET NULL — a fact outlives the chat it was said in).
    ctx = ToolContext(conversation_id=None, turn_id=None)
    result = builtins._remember({"text": "Works mainly in Python.",
                                 "asked_by_user": True}, ctx)
    assert result["status"] == "success"
    assert [m["text"] for m in memory.live()] == ["Works mainly in Python."]
