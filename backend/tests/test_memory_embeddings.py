"""Memory embeddings: fusion, fallback, filtering, the vector cache, and the
meaning-based supersession arm.

No real model here. A fake encoder maps text onto a handful of concepts, so
"what pets do I have" lands next to "My dog is called Biscuit" the way the real
one does, and the tests need neither torch nor a download. The real encoder is
measured by scripts/bench_memory_embeddings.py.

Both arms are default OFF, so most tests turn one on and compare to off.
"""
from __future__ import annotations

import re

import pytest

from primnox2.cognition import conflict
from primnox2.memory import embeddings
from primnox2.memory import service as memory
from primnox2.settings import tunables
from primnox2.storage import db
from primnox2.tools import retrieval

JAN, MAR, JUN = 1_735_689_600_000, 1_740_787_200_000, 1_748_736_000_000

CONCEPTS = {
    "pet": {"dog", "cat", "hamster", "pet", "pets", "puppy", "kitten", "biscuit", "miso", "animals"},
    "phone": {"phone", "pixel", "iphone", "handset", "android"},
    "car": {"car", "drive", "civic", "tesla", "vehicle", "driving"},
    "job": {"work", "works", "job", "nurse", "teacher", "occupation", "google", "acme"},
    "city": {"live", "flat", "leeds", "bristol", "city"},
    "drink": {"coffee", "tea", "drink", "beverage"},
}


class FakeEncoder:
    """Concept-bag vectors, and a record of every text it was asked to embed."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen.extend(texts)
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([1.0 if words & group else 0.0 for group in CONCEPTS.values()] + [0.05])
        return out


@pytest.fixture
def encoder(monkeypatch):
    fake = FakeEncoder()
    monkeypatch.setattr(retrieval, "_encoder", fake)
    monkeypatch.setattr(retrieval, "_failed", False)
    return fake


@pytest.fixture
def no_encoder(monkeypatch):
    monkeypatch.setattr(retrieval, "_encoder", None)
    monkeypatch.setattr(retrieval, "_failed", True)


@pytest.fixture
def still_loading(monkeypatch):
    # `_loading` makes retrieval._load() return at once, so no real model starts.
    monkeypatch.setattr(retrieval, "_encoder", None)
    monkeypatch.setattr(retrieval, "_failed", False)
    monkeypatch.setattr(retrieval, "_loading", True)


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS", "1")


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


def _texts(hits):
    return [h["text"] for h in hits]


def _stored_vector(memory_id: str):
    return db.connect().execute(
        "SELECT embedding FROM memories WHERE id=?", (memory_id,)).fetchone()["embedding"]


# ── defaults ─────────────────────────────────────────────────────────────────
def test_every_arm_is_on_by_default(monkeypatch):
    # The suite pins the encoder arms off (conftest); the default is what ships.
    for key in ("EMBEDDINGS", "EMBEDDINGS_SUPERSEDE", "ACTOR_GUARD"):
        monkeypatch.delenv(f"PRIMNOX2_MEMORY_{key}", raising=False)
    assert tunables.get("memory.embeddings") == 1
    assert tunables.get("memory.embeddings_supersede") == 1
    assert tunables.get("memory.actor_guard") == 1


def test_flag_off_never_touches_the_encoder(encoder, clean_memory):
    memory.remember("My phone is a Pixel 9.")
    memory.remember("I drive a 2015 Honda Civic.")
    assert memory.search("which handset do I carry") == []
    assert encoder.seen == []


# ── fusion ───────────────────────────────────────────────────────────────────
def test_fuse_ranks_what_both_orderings_like_above_what_one_does():
    assert embeddings.fuse(["a", "b"], ["c", "b"], trusted={"a", "b"}) == ["b", "a", "c"]


def test_fuse_breaks_ties_with_the_lexical_order():
    assert embeddings.fuse(["a", "b"], ["b", "a"], trusted={"a", "b"}) == ["a", "b"]


def test_fuse_keeps_untrusted_lexical_hits_after_the_fused_ones():
    fused = embeddings.fuse(["noise", "topic"], ["meaning"], trusted={"topic"})
    assert fused == ["topic", "meaning", "noise"]


def test_fuse_untrusted_hits_do_not_outrank_meaning():
    # A bare shared word is in both lists here, and still must not beat the
    # memory the encoder put first.
    assert embeddings.fuse(["noise"], ["right", "noise"], trusted=set())[0] == "right"


def test_fuse_loses_nothing_the_lexical_search_returned():
    lexical = ["a", "b", "c"]
    assert set(embeddings.fuse(lexical, ["d"], trusted={"a"})) == {"a", "b", "c", "d"}


# ── search ───────────────────────────────────────────────────────────────────
def test_hybrid_closes_the_synonym_gap(encoder, hybrid, clean_memory):
    memory.remember("My dog is called Biscuit.")
    memory.remember("We adopted a cat named Miso.")
    memory.remember("I drive a 2015 Honda Civic.")
    assert set(_texts(memory.search("what pets do I have"))) == {
        "My dog is called Biscuit.", "We adopted a cat named Miso."}


def test_hybrid_keeps_a_topic_match_the_lexicon_found(encoder, hybrid, clean_memory):
    memory.remember("My editor is VS Code.")
    memory.remember("I use Redis for the job queue.")
    assert _texts(memory.search("editor"))[0] == "My editor is VS Code."


def test_questions_about_nothing_said_come_back_empty(encoder, hybrid, clean_memory):
    memory.remember("My dog is called Biscuit.")
    assert memory.search("which beverage do I like") == []


def test_encoder_unavailable_falls_back_to_the_lexical_ranking(no_encoder, hybrid, clean_memory):
    memory.remember("My editor is VS Code.")
    memory.remember("I use Redis for the job queue.")
    assert _texts(memory.search("editor")) == ["My editor is VS Code."]
    assert memory.search("what pets do I have") == []


def test_encoder_still_loading_falls_back_without_blocking(still_loading, hybrid, clean_memory):
    memory.remember("My editor is VS Code.")
    assert _texts(memory.search("editor")) == ["My editor is VS Code."]


def test_encoder_that_raises_falls_back_to_the_lexical_ranking(monkeypatch, hybrid, clean_memory):
    def boom(texts):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(retrieval, "_encoder", boom)
    monkeypatch.setattr(retrieval, "_failed", False)
    memory.remember("My editor is VS Code.")
    assert _texts(memory.search("editor")) == ["My editor is VS Code."]


def test_hybrid_search_still_returns_the_lexical_hits_when_meaning_finds_none(encoder, hybrid, clean_memory):
    memory.remember("I use Obsidian for notes.")
    assert _texts(memory.search("notes")) == ["I use Obsidian for notes."]


# ── as_of and supersession filter first, then rank ───────────────────────────
def test_hybrid_does_not_return_a_retired_memory(encoder, hybrid, clean_memory):
    memory.import_many([{"text": "My phone is an iPhone 13.", "created_at": JAN},
                        {"text": "My phone is a Pixel 9 now.", "created_at": JUN}])
    assert _texts(memory.search("which handset do I carry")) == ["My phone is a Pixel 9 now."]


def test_hybrid_history_view_still_sees_retired_memories(encoder, hybrid, clean_memory):
    memory.import_many([{"text": "My phone is an iPhone 13.", "created_at": JAN},
                        {"text": "My phone is a Pixel 9 now.", "created_at": JUN}])
    hits = memory.search("which handset do I carry", include_superseded=True)
    assert set(_texts(hits)) == {"My phone is an iPhone 13.", "My phone is a Pixel 9 now."}


def test_hybrid_as_of_answers_with_what_was_true_then(encoder, hybrid, clean_memory):
    memory.import_many([{"text": "My phone is an iPhone 13.", "created_at": JAN},
                        {"text": "My phone is a Pixel 9 now.", "created_at": JUN}])
    assert _texts(memory.search("which handset do I carry", as_of="2025-03")) == [
        "My phone is an iPhone 13."]


def test_hybrid_as_of_before_anything_was_known_finds_nothing(encoder, hybrid, clean_memory):
    memory.import_many([{"text": "My dog is called Biscuit.", "created_at": MAR}])
    assert memory.search("what pets do I have", as_of="2025-01-15") == []


def test_hybrid_never_returns_a_forgotten_memory(encoder, hybrid, clean_memory):
    kept = memory.remember("My dog is called Biscuit.")
    gone = memory.remember("We adopted a cat named Miso.")
    memory.forget(gone["id"])
    assert [h["id"] for h in memory.search("what pets do I have")] == [kept["id"]]


# ── the vector cache ─────────────────────────────────────────────────────────
def test_rows_never_carry_the_cache_blob(encoder, hybrid, clean_memory):
    made = memory.remember("My dog is called Biscuit.")
    memory.search("what pets do I have")
    assert _stored_vector(made["id"]) is not None
    assert all("embedding" not in row for row in memory.live())
    assert "embedding" not in memory.get(made["id"])
    assert all("embedding" not in row for row in memory.search("what pets do I have"))


def test_vectors_are_computed_once_per_memory(encoder, hybrid, clean_memory):
    memory.remember("My dog is called Biscuit.")
    memory.remember("We adopted a cat named Miso.")
    memory.search("what pets do I have")
    memory.search("which pets do I own")
    memory_texts = [t for t in encoder.seen if t in ("My dog is called Biscuit.",
                                                     "We adopted a cat named Miso.")]
    assert len(memory_texts) == 2


def test_editing_a_memory_drops_its_vector_and_search_follows_the_new_text(encoder, hybrid, clean_memory):
    made = memory.remember("My dog is called Biscuit.")
    memory.search("what pets do I have")
    assert _stored_vector(made["id"]) is not None

    assert memory.update(made["id"], "My phone is a Pixel 9.")
    assert _stored_vector(made["id"]) is None
    assert memory.search("what pets do I have") == []
    assert _texts(memory.search("which handset do I carry")) == ["My phone is a Pixel 9."]


def test_a_text_changed_behind_the_services_back_is_not_matched_on_its_old_vector(encoder, hybrid, clean_memory):
    made = memory.remember("I drive a 2015 Honda Civic.")
    memory.search("which vehicle do I own")
    with db.tx() as c:
        c.execute("UPDATE memories SET text=? WHERE id=?", ("My phone is a Pixel 9.", made["id"]))
    assert memory.search("which vehicle do I own") == []
    assert _texts(memory.search("which handset do I carry")) == ["My phone is a Pixel 9."]


def test_a_vector_from_a_different_encoder_is_not_reused(encoder, hybrid, clean_memory, monkeypatch):
    memory.remember("My dog is called Biscuit.")
    memory.search("what pets do I have")
    before = len([t for t in encoder.seen if t == "My dog is called Biscuit."])
    monkeypatch.setattr(retrieval, "ENCODER", "another/encoder")
    memory.search("what pets do I have")
    assert len([t for t in encoder.seen if t == "My dog is called Biscuit."]) == before + 1


# ── the write path ───────────────────────────────────────────────────────────
def test_remember_needs_no_model_with_every_arm_on(no_encoder, monkeypatch, clean_memory):
    for key in ("EMBEDDINGS", "EMBEDDINGS_SUPERSEDE", "ACTOR_GUARD"):
        monkeypatch.setenv(f"PRIMNOX2_MEMORY_{key}", "1")
    memory.remember("My phone is an iPhone 13.")
    made = memory.remember("Got a Pixel 9 last week.")
    assert made["stored"] is True
    assert made["superseded"] == []
    assert _stored_vector(made["id"]) is None


def test_remember_survives_an_encoder_that_raises(monkeypatch, clean_memory):
    def boom(texts):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(retrieval, "_encoder", boom)
    monkeypatch.setattr(retrieval, "_failed", False)
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE", "1")
    memory.remember("My phone is an iPhone 13.")
    assert memory.remember("Got a Pixel 9 last week.")["stored"] is True


# ── supersession by meaning ──────────────────────────────────────────────────
@pytest.fixture
def by_meaning(encoder, monkeypatch):
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE", "1")


def _remembered(old: str, new: str):
    first = memory.remember(old)
    return first, memory.remember(new)


def test_an_update_that_shares_no_words_is_missed_without_the_arm(encoder, clean_memory):
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 last week.")
    assert second["superseded"] == []


def test_meaning_finds_the_update_the_words_missed(by_meaning, clean_memory):
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 last week.")
    assert second["superseded"] == [first["id"]]
    assert memory.get(first["id"])["superseded_by"] == second["id"]


def test_a_second_pet_does_not_retire_the_first(by_meaning, clean_memory):
    first, second = _remembered("My dog is called Biscuit.", "We adopted a cat named Miso.")
    assert second["superseded"] == []


def test_similar_but_not_worded_as_a_change_retires_nothing(by_meaning, clean_memory):
    first, second = _remembered("I drive a 2015 Honda Civic.", "It's a silver Civic.")
    assert second["superseded"] == []


def test_another_one_is_an_addition_even_when_worded_as_a_change(by_meaning, clean_memory):
    first, second = _remembered("My dog is called Biscuit.", "Got another dog named Rex.")
    assert second["superseded"] == []


def test_a_relatives_statement_never_retires_yours_by_meaning(by_meaning, clean_memory):
    first, second = _remembered("I work at Acme.", "My brother just got a job at Google.")
    assert second["superseded"] == []


def test_a_dated_log_line_never_retires_anything_by_meaning(by_meaning, clean_memory):
    first, second = _remembered("I drive a 2015 Honda Civic.", "Reviewed the Tesla quote 2025-03-01.")
    assert second["superseded"] == []


def test_relative_time_does_not_make_a_change_an_event(by_meaning, clean_memory):
    # "last week" is how someone dates a change; kind_of() files it as an event.
    first, second = _remembered("My phone is an iPhone 13.", "Switched to a Pixel 9 last week.")
    assert second["superseded"] == [first["id"]]


def test_the_threshold_is_the_tunable(by_meaning, monkeypatch, clean_memory):
    # Phone and car concepts against phone alone: similar, but not identical.
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE_SIMILARITY", "0.95")
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 and a Tesla last week.")
    assert second["superseded"] == []
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE_SIMILARITY", "0.3")
    third = memory.remember("Bought a new android handset and a Civic.")
    assert third["superseded"] != []


def test_only_the_most_similar_memory_is_retired(by_meaning, clean_memory):
    near = memory.remember("My phone is an iPhone 13.")
    far = memory.remember("My handset is cracked and my android tablet too.")
    second = memory.remember("Got a Pixel 9 last week.")
    assert len(second["superseded"]) == 1


def test_import_proposes_updates_by_meaning_dated_by_the_successor(by_meaning, clean_memory):
    out = memory.import_many([{"text": "My phone is an iPhone 13.", "created_at": JAN},
                              {"text": "Got a Pixel 9 last week.", "created_at": JUN}])
    assert out["superseded"] == 1
    old = next(r for r in memory.live() if r["text"] == "My phone is an iPhone 13.")
    assert old["superseded_by"] is not None
    assert old["updated_at"] == JUN


def test_import_without_the_arm_leaves_both_standing(encoder, clean_memory):
    out = memory.import_many([{"text": "My phone is an iPhone 13.", "created_at": JAN},
                              {"text": "Got a Pixel 9 last week.", "created_at": JUN}])
    assert out["superseded"] == 0


# ── editing a memory that meaning retired (memory.update re-judges) ──────────
def test_editing_the_newer_memory_does_not_undo_what_meaning_retired(by_meaning, clean_memory):
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 last week.")
    assert second["superseded"] == [first["id"]]
    assert memory.update(second["id"], "Got a Pixel 9 last month.")
    assert memory.get(first["id"])["superseded_by"] == second["id"]
    assert _stored_vector(second["id"]) is None


def test_an_edit_that_stops_being_a_change_gives_back_what_meaning_retired(by_meaning, clean_memory):
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 last week.")
    assert memory.update(second["id"], "It is a lovely Pixel 9.")
    assert memory.get(first["id"])["superseded_by"] is None


def test_editing_without_the_arm_judges_by_words_alone(encoder, clean_memory):
    first, second = _remembered("My phone is an iPhone 13.", "Got a Pixel 9 last week.")
    assert memory.update(second["id"], "Got a Pixel 9 last month.")
    assert memory.get(first["id"])["superseded_by"] is None
    assert encoder.seen == []


# ── the actor guard (not an embeddings setting) ──────────────────────────────
def test_without_the_guard_a_relatives_job_retires_yours(monkeypatch, clean_memory):
    monkeypatch.setenv("PRIMNOX2_MEMORY_ACTOR_GUARD", "0")
    # "brother" is a relation conflict.different_subjects already lists; "nephew" is not.
    first, second = _remembered("I work at Acme.", "My nephew works at Google.")
    assert second["superseded"] == [first["id"]]


@pytest.mark.parametrize("relative", ["brother", "nephew"])
def test_the_guard_keeps_a_siblings_job_from_retiring_yours(relative, monkeypatch, clean_memory):
    monkeypatch.setenv("PRIMNOX2_MEMORY_ACTOR_GUARD", "1")
    first, second = _remembered("I work at Acme.", f"My {relative} works at Google.")
    assert second["superseded"] == []


def test_the_guard_leaves_a_real_update_alone(monkeypatch, clean_memory):
    monkeypatch.setenv("PRIMNOX2_MEMORY_ACTOR_GUARD", "1")
    first, second = _remembered("I work at Acme.", "I work at Google now.")
    assert second["superseded"] == [first["id"]]


@pytest.mark.parametrize("a,b,same", [
    ("I work at Acme.", "I work at Google.", True),
    ("My wife is a vet.", "My wife's sister is a lawyer.", False),
    ("My sister lives in Madrid.", "I live in Leeds.", False),
    ("My wife works at Acme.", "My wife now works at Google.", True),
    ("Priya prefers dark mode.", "I prefer dark mode.", True),
    ("She works at Google.", "I work at Acme.", False),
])
def test_actor(a, b, same):
    assert (conflict.actor(a) == conflict.actor(b)) is same


# ── resolve() directly ───────────────────────────────────────────────────────
def _resolve(new, old, sim, **kw):
    return conflict.resolve(
        new, [{"id": "old", "text": old, "topic": None, "kind": "fact"}],
        similarity=lambda a, b: 0.0, duplicate_threshold=0.85,
        semantic=lambda _id: sim, **kw).superseded_ids


def test_resolve_without_semantic_is_unchanged():
    result = conflict.resolve(
        "Got a Pixel 9 last week.", [{"id": "old", "text": "My phone is an iPhone 13."}],
        similarity=lambda a, b: 0.0, duplicate_threshold=0.85)
    assert result.superseded_ids == []


def test_resolve_semantic_needs_the_threshold():
    assert _resolve("Got a Pixel 9.", "My phone is an iPhone 13.", 0.29) == []
    assert _resolve("Got a Pixel 9.", "My phone is an iPhone 13.", 0.31) == ["old"]


def test_resolve_semantic_defers_to_a_rule_that_already_decided():
    # Two topics that differ: the lexicon has said these are different slots.
    result = conflict.resolve(
        "Got a new editor.", [{"id": "old", "text": "I drink tea.", "topic": "coffee", "kind": "fact"}],
        similarity=lambda a, b: 0.0, duplicate_threshold=0.85, new_topic="editor",
        semantic=lambda _id: 0.9)
    assert result.superseded_ids == []


def test_resolve_semantic_ignores_a_multi_valued_slot():
    result = conflict.resolve(
        "Got a new project.", [{"id": "old", "text": "I work on project Granite.", "topic": None, "kind": "fact"}],
        similarity=lambda a, b: 0.0, duplicate_threshold=0.85, new_topic="project",
        semantic=lambda _id: 0.9)
    assert result.superseded_ids == []


def test_resolve_semantic_ignores_a_refinement():
    assert _resolve("I drive a 2019 Corolla now", "I drive a Corolla", 0.9) == []
    assert _resolve("I drive a Tesla now", "I drive a Corolla", 0.9) == ["old"]
