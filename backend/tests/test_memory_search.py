"""Memory search: the slot a question asks about, what is true now, and what
was true then.

Before this, search ranked by word-overlap ratio alone. On a synthetic life it
found the current answer to "theme" or "language" 40% of the time (the memory
says "switched to light mode", not "theme"), returned retired memories as the
answer about now, and could not answer "what was it in March" at all.
"""
from __future__ import annotations

from datetime import timezone

import pytest

from primnox2.memory import service as memory
from primnox2.storage import db
from primnox2.tools import builtins  # noqa: F401  (registers the tools)
from primnox2.tools.registry import get as get_tool

DAY = 86_400_000
JAN, MAR, JUN = 1_735_689_600_000, 1_740_787_200_000, 1_748_736_000_000  # 2025-01/03/06-01 UTC


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


def _texts(hits):
    return [h["text"] for h in hits]


def test_a_question_finds_its_slot_without_sharing_a_word(clean_memory):
    memory.remember("I switched to light mode.")
    memory.remember("I use Redis for the job queue.")
    assert _texts(memory.search("theme"))[0] == "I switched to light mode."


def test_question_words_do_not_count_as_matches(clean_memory):
    memory.remember("The user is in a good mood.")
    memory.remember("I write Rust daily.")
    assert _texts(memory.search("what language does the user prefer"))[0] == "I write Rust daily."


def test_a_retired_memory_is_not_the_answer_about_now(clean_memory):
    memory.remember("My editor is VS Code.")
    memory.remember("My editor is Helix now.")
    assert _texts(memory.search("editor")) == ["My editor is Helix now."]


def test_history_views_can_still_see_retired_memories(clean_memory):
    memory.remember("My editor is VS Code.")
    memory.remember("My editor is Helix now.")
    assert set(_texts(memory.search("editor", include_superseded=True))) == {
        "My editor is VS Code.", "My editor is Helix now."}


def test_a_newer_more_specific_refinement_ranks_first(clean_memory):
    memory.import_many([{"text": "I use Postgres for the main database.", "created_at": JAN},
                        {"text": "I use Postgres 16 for the main database.", "created_at": MAR}])
    assert _texts(memory.search("database"))[0] == "I use Postgres 16 for the main database."


def test_as_of_answers_with_what_was_true_then(clean_memory):
    memory.import_many([{"text": "My editor is VS Code.", "created_at": JAN},
                        {"text": "My editor is Helix now.", "created_at": JUN}])
    assert _texts(memory.search("editor", as_of="2025-03")) == ["My editor is VS Code."]
    assert _texts(memory.search("editor", as_of="2025-06")) == ["My editor is Helix now."]
    assert _texts(memory.search("editor")) == ["My editor is Helix now."]


def test_as_of_before_anything_was_known_finds_nothing(clean_memory):
    memory.import_many([{"text": "My editor is VS Code.", "created_at": MAR}])
    assert memory.search("editor", as_of="2025-01-15") == []


@pytest.mark.parametrize("value,expected", [
    ("2025-02", MAR - 1),                 # end of February
    ("2025-03-01", MAR + DAY - 1),        # end of that day
    (12345, 12345),
    ("", None), (None, None), ("next tuesday", None),
])
def test_parse_as_of(value, expected):
    assert memory.parse_as_of(value, tz=timezone.utc) == expected


def test_recall_memory_tool_accepts_as_of(clean_memory):
    memory.import_many([{"text": "My editor is VS Code.", "created_at": JAN},
                        {"text": "My editor is Helix now.", "created_at": JUN}])
    tool = get_tool("recall_memory")
    out = tool.handler({"query": "editor", "as_of": "2025-03"}, None)
    assert "VS Code" in out["output"] and "Helix" not in out["output"]


def test_recall_memory_tool_refuses_an_unreadable_date(clean_memory):
    out = get_tool("recall_memory").handler({"query": "editor", "as_of": "someday"}, None)
    assert out["status"] == "error"
