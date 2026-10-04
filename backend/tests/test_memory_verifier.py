"""The background verifier: the local model's second opinion on updates the rules
did not decide.

The model is never called here. `verifier.ask` (or the HTTP post under it) is
replaced, and a fake encoder maps text onto a few concepts so that pairs the word
rules cannot tie together still look similar to the encoder.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error

import pytest

from primnox2.memory import service as memory
from primnox2.memory import verifier
from primnox2.settings import tunables
from primnox2.storage import db
from primnox2.tools import retrieval

JAN, MAR, JUN = 1_735_689_600_000, 1_740_787_200_000, 1_748_736_000_000

CONCEPTS = {
    "music": {"spotify", "apple", "music", "playlist", "tidal"},
    "phone": {"phone", "pixel", "iphone", "galaxy", "samsung"},
    "commute": {"subway", "bike", "cycling", "commute", "streetcar"},
    "work": {"work", "works", "job", "bank", "hospital"},
    "pet": {"dog", "cat", "kitten"},
}


class FakeEncoder:
    def __call__(self, texts):
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([1.0 if words & group else 0.0 for group in CONCEPTS.values()] + [0.3])
        return out


@pytest.fixture(autouse=True)
def verifying(monkeypatch):
    monkeypatch.setattr(retrieval, "_encoder", FakeEncoder())
    monkeypatch.setattr(retrieval, "_failed", False)
    monkeypatch.setenv("PRIMNOX2_MEMORY_LLM_VERIFY", "1")
    monkeypatch.setattr(verifier, "_state", {"down_until": 0.0, "warned": False})
    verifier.reset_stats()
    with db.tx() as c:
        c.execute("DELETE FROM memories")
        c.execute("DELETE FROM memory_verifications")
    yield
    memory.drain_verifications()
    with db.tx() as c:
        c.execute("DELETE FROM memories")
        c.execute("DELETE FROM memory_verifications")


def answers(monkeypatch, relation="replaces", confidence=0.99):
    asked: list[tuple[str, str]] = []

    def fake(new, old):
        asked.append((new, old))
        return verifier.Verdict(relation, confidence, 5)

    monkeypatch.setattr(verifier, "ask", fake)
    return asked


OLD_APP, NEW_APP = "Spotify is basically my entire day.", "Switched to Apple Music, it came free with the Mac."


def remembered(old=OLD_APP, new=NEW_APP):
    first = memory.remember(old)
    second = memory.remember(new)
    memory.drain_verifications()
    return first, second


def retired_by(memory_id):
    return memory.get(memory_id)["superseded_by"]


# ── the flag ─────────────────────────────────────────────────────────────────
def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("PRIMNOX2_MEMORY_LLM_VERIFY", raising=False)
    assert tunables.get("memory.llm_verify") == 0


def test_flag_off_never_asks_the_model(monkeypatch):
    monkeypatch.setenv("PRIMNOX2_MEMORY_LLM_VERIFY", "0")
    asked = answers(monkeypatch)
    first, second = remembered()
    assert asked == []
    assert retired_by(first["id"]) is None


def test_the_rules_alone_leave_this_update_standing():
    first, second = remembered()
    assert second["superseded"] == []


# ── what a confirmed update does ─────────────────────────────────────────────
def test_a_confirmed_update_retires_the_old_memory_afterwards(monkeypatch):
    asked = answers(monkeypatch)
    first, second = remembered()
    assert asked == [(NEW_APP, OLD_APP)]
    assert retired_by(first["id"]) == second["id"]
    assert json.loads(memory.get(second["id"])["supersedes"]) == [first["id"]]


def test_the_save_returns_before_the_model_answers(monkeypatch):
    release, asked = threading.Event(), threading.Event()

    def slow(new, old):
        asked.set()
        release.wait(10)
        return verifier.Verdict("replaces", 0.99, 5)

    monkeypatch.setattr(verifier, "ask", slow)
    first = memory.remember(OLD_APP)
    second = memory.remember(NEW_APP)
    assert second["superseded"] == []
    assert asked.wait(10)
    assert retired_by(first["id"]) is None
    release.set()
    memory.drain_verifications()
    assert retired_by(first["id"]) == second["id"]


def test_the_retirement_is_dated_by_the_new_memory(monkeypatch):
    answers(monkeypatch)
    first = memory.import_many([{"text": OLD_APP, "created_at": JAN}])
    memory.import_many([{"text": NEW_APP, "created_at": JUN}])
    memory.drain_verifications()
    row = memory.get(first["ids"][0])
    assert row["superseded_by"] is not None
    assert row["updated_at"] == JUN


def test_an_import_is_verified_too(monkeypatch):
    asked = answers(monkeypatch)
    out = memory.import_many([{"text": OLD_APP, "created_at": JAN}, {"text": NEW_APP, "created_at": MAR}])
    memory.drain_verifications()
    assert out["superseded"] == 0
    assert asked == [(NEW_APP, OLD_APP)]
    assert retired_by(out["ids"][0]) == out["ids"][1]


@pytest.mark.parametrize("relation", ["adds_detail", "another_value", "other_person", "unrelated"])
def test_only_replaces_retires(monkeypatch, relation):
    answers(monkeypatch, relation)
    first, second = remembered()
    assert retired_by(first["id"]) is None


def test_a_hesitant_replaces_retires_nothing(monkeypatch):
    answers(monkeypatch, "replaces", 0.7)
    first, second = remembered()
    assert retired_by(first["id"]) is None


def test_the_confidence_bar_is_the_tunable(monkeypatch):
    answers(monkeypatch, "replaces", 0.7)
    monkeypatch.setenv("PRIMNOX2_MEMORY_LLM_VERIFY_CONFIDENCE", "0.6")
    first, second = remembered()
    assert retired_by(first["id"]) == second["id"]


# ── the guards still veto ────────────────────────────────────────────────────
@pytest.mark.parametrize("old,new", [
    ("I work at a bank.", "My brother works at a hospital."),                  # another person
    ("I have a dog called Rex.", "Got a kitten named Pip, a cat at last."),    # a second of something
    ("Spotify is basically my entire day.", "Also switched to Apple Music."),  # an addition
    ("Reviewed the Spotify playlist 2025-03-01.", "Switched to Apple Music."),  # a dated log line
    ("Allergic to peanuts, carry an EpiPen.", "Eating at the new music cafe now."),  # a safety fact
])
def test_a_guard_keeps_the_pair_from_being_asked(monkeypatch, old, new):
    asked = answers(monkeypatch)
    first, second = remembered(old, new)
    assert asked == []
    assert retired_by(first["id"]) is None


def test_a_memory_forgotten_meanwhile_is_not_retired(monkeypatch):
    release = threading.Event()

    def after_forgetting(new, old):
        release.wait(10)
        return verifier.Verdict("replaces", 0.99, 5)

    monkeypatch.setattr(verifier, "ask", after_forgetting)
    first = memory.remember(OLD_APP)
    second = memory.remember(NEW_APP)
    memory.forget(first["id"])
    release.set()
    memory.drain_verifications()
    assert retired_by(first["id"]) is None
    assert memory.get(second["id"])["supersedes"] is None
    assert [d["applied"] for d in verifier.decisions()] == [0]


def test_a_memory_the_rules_retired_meanwhile_keeps_its_successor(monkeypatch):
    answers(monkeypatch)
    first = memory.remember(OLD_APP)
    second = memory.remember(NEW_APP)
    memory.drain_verifications()
    later = memory.remember("Switched to Tidal, the playlist sync is better.")
    memory.drain_verifications()
    assert retired_by(first["id"]) == second["id"]
    assert later["superseded"] != [first["id"]]


def test_an_edited_memory_is_not_retired_on_the_old_wording(monkeypatch):
    release = threading.Event()

    def after_edit(new, old):
        release.wait(10)
        return verifier.Verdict("replaces", 0.99, 5)

    monkeypatch.setattr(verifier, "ask", after_edit)
    first = memory.remember(OLD_APP)
    memory.remember(NEW_APP)
    memory.update(first["id"], "I stream everything on Tidal.")
    release.set()
    memory.drain_verifications()
    assert retired_by(first["id"]) is None


# ── how many questions ───────────────────────────────────────────────────────
def test_the_first_confirmed_candidate_ends_the_job(monkeypatch):
    asked = []

    def fake(new, old):
        asked.append(old)
        return verifier.Verdict("replaces" if new.startswith("Switched") else "another_value", 0.99, 5)

    monkeypatch.setattr(verifier, "ask", fake)
    for line in ("Spotify is basically my entire day.", "Apple Music has the better playlist for the gym.",
                 "My music taste is mostly Afrobeats playlist stuff."):
        memory.remember(line)
    memory.drain_verifications()
    asked.clear()
    memory.remember("Switched to Tidal last month.")
    memory.drain_verifications()
    assert len(asked) == 1
    assert len([d for d in verifier.decisions() if d["applied"]]) == 1


def test_a_save_asks_about_at_most_the_candidate_cap(monkeypatch):
    asked = answers(monkeypatch, "another_value")
    monkeypatch.setenv("PRIMNOX2_MEMORY_LLM_VERIFY_CANDIDATES", "2")
    for line in ("Spotify is basically my entire day.", "Apple Music has the better gym playlist.",
                 "My music taste is mostly Afrobeats playlist stuff.", "Radio plays music all morning."):
        memory.remember(line)
    memory.drain_verifications()
    asked.clear()
    memory.remember("Switched to Tidal last month.")
    memory.drain_verifications()
    assert len(asked) == 2


def test_a_pair_below_the_similarity_floor_is_not_asked(monkeypatch):
    asked = answers(monkeypatch)
    first, second = remembered("I work at a bank.", "Switched to Apple Music, it came free with the Mac.")
    assert asked == []


def test_nothing_is_asked_without_an_encoder(monkeypatch):
    asked = answers(monkeypatch)
    monkeypatch.setattr(retrieval, "_encoder", None)
    monkeypatch.setattr(retrieval, "_failed", True)
    first, second = remembered()
    assert asked == []
    assert retired_by(first["id"]) is None


# ── the model is not there ───────────────────────────────────────────────────
def test_an_unreachable_model_leaves_the_rules_result_and_logs_once(monkeypatch, caplog):
    def refused(old, new):
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(verifier, "_post", refused)
    with caplog.at_level(logging.WARNING, logger=verifier.log.name):
        first, second = remembered()
        third = memory.remember("Switched to Tidal, the playlist sync is better.")
        memory.drain_verifications()
    assert second["stored"] and third["stored"]
    assert retired_by(first["id"]) is None
    assert len([r for r in caplog.records if "unavailable" in r.getMessage()]) == 1
    assert verifier.stats["skipped"] >= 1
    assert verifier.decisions() == []


def test_a_timeout_is_retried(monkeypatch):
    monkeypatch.setattr(verifier.time, "sleep", lambda s: None)
    calls = []

    def flaky(old, new):
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError()
        return '{"relation": "replaces", "confidence": 0.9}', []

    monkeypatch.setattr(verifier, "_post", flaky)
    verdict = verifier.ask("new", "old")
    assert verdict is not None and verdict.relation == "replaces"
    assert len(calls) == 3


def test_a_reply_that_is_not_a_relation_is_not_a_decision(monkeypatch):
    monkeypatch.setattr(verifier.time, "sleep", lambda s: None)
    monkeypatch.setattr(verifier, "_post", lambda old, new: ('{"relation": "maybe"}', []))
    assert verifier.ask("new", "old") is None


# ── reading the model's reply ────────────────────────────────────────────────
def _tokens(*pieces):
    return [{"token": t, "logprob": 0.0, "top_logprobs": alts} for t, alts in pieces]


def test_confidence_is_the_probability_the_model_gave_the_relation():
    content = '{"relation": "replaces", "confidence": 0.9}'
    logprobs = _tokens(
        ('{"', []), ("relation", []), ('":', []), (' "', []),
        ("re", [{"token": "re", "logprob": -0.1}, {"token": "adds", "logprob": -2.5},
                {"token": "another", "logprob": -5.0}]),
        ("places", []), ('",', []))
    relation, confidence = verifier._parse(content, logprobs)
    assert relation == "replaces"
    assert 0.88 < confidence < 0.92


def test_the_stated_confidence_is_used_when_there_are_no_token_probabilities():
    assert verifier._parse('{"relation": "replaces", "confidence": 0.75}', []) == ("replaces", 0.75)
    assert verifier._parse('{"relation": "replaces", "confidence": 90}', []) == ("replaces", 0.9)
    assert verifier._parse('{"relation": "replaces"}', []) == ("replaces", 0.0)


# ── the training record ──────────────────────────────────────────────────────
def test_every_answer_is_recorded_with_the_pair_and_whether_it_acted(monkeypatch):
    answers(monkeypatch)
    first, second = remembered()
    (row,) = verifier.decisions()
    assert (row["old_id"], row["new_id"]) == (first["id"], second["id"])
    assert (row["old_text"], row["new_text"]) == (OLD_APP, NEW_APP)
    assert (row["relation"], row["confidence"], row["applied"]) == ("replaces", 0.99, 1)
    assert row["model"] == verifier.model_name() and row["similarity"] > 0


def test_an_answer_that_changed_nothing_is_recorded_too(monkeypatch):
    answers(monkeypatch, "another_value", 0.9)
    remembered()
    (row,) = verifier.decisions()
    assert (row["relation"], row["applied"]) == ("another_value", 0)
