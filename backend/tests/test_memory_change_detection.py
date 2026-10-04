"""Noticing that a new statement REPLACES an old fact.

Written from the ways a person words a change, not from any one corpus. Three
families: what counts as an event (a one-off that retires nothing) and what is an
acquisition or state change that says what is true now; whose statement it is
(a verb-first sentence is the user's own, a named person's is theirs); and what
the meaning arm may retire once it has the words and the guards.
"""
from __future__ import annotations

import re

import pytest

from primnox2.cognition import conflict, topics
from primnox2.memory import embeddings
from primnox2.memory import service as memory
from primnox2.storage import db
from primnox2.tools import retrieval

# ── events and state changes ─────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "Got a Pixel 9 last week.",
    "Started at Deloitte on Monday.",
    "Swapped the Golf for a used Nissan Leaf last week.",
    "Migrated the main database to Postgres 16 last month.",
    "Gave up the Italian app yesterday and joined a class.",
    "Picked up a Tesla Model 3 last month and sold the Civic.",
])
def test_an_acquisition_or_change_with_a_time_phrase_is_a_fact(text):
    assert topics.kind_of(text) == "fact"


@pytest.mark.parametrize("text", [
    "Had sushi with Mei on Friday.",
    "Went to a Radiohead gig last week.",
    "Visited Whitby last weekend, the fish and chips were great.",
    "Isla won a swimming medal at the school gala.",
    "Saw a Hokusai exhibition at the museum last weekend.",
    "Back from Puglia! Orecchiette every single day.",
    "I took the train to Edinburgh last week.",
    "Met Nadia about Atlas (week 3).",
])
def test_a_one_off_is_an_event(text):
    assert topics.kind_of(text) == "event"


@pytest.mark.parametrize("text", [
    "I read mostly crime novels before bed.",
    "Started swimming at Sartor pool on Tuesday mornings.",
    "Sundays I play pickup football at Christie Pits.",
    "I run five kilometres on Tuesday evenings.",
])
def test_a_habit_is_not_an_event(text):
    assert topics.kind_of(text) == "fact"


def test_a_bare_purchase_is_an_event_to_the_word_rules_but_not_to_meaning():
    # "Bought a film camera" then "Bought a road bike" are two purchases, not one
    # slot with two values; but "Bought a Pixel 9" is also how an upgrade is worded.
    text = "Bought a road bike for the weekends."
    assert topics.kind_of(text) == "event"
    assert not topics.is_episode(text)
    assert topics.is_episode("Met Nadia (week 3).")


# ── whose statement it is ────────────────────────────────────────────────────


@pytest.mark.parametrize("new,old", [
    ("Got a Pixel 9 last week.", "My phone is an iPhone 13."),
    ("Moved to Lisbon in March.", "I live in Leeds."),
    ("Switched to Rust for most of my work.", "I code in Python every day."),
    ("Eating meat again since the spring.", "I'm vegetarian."),
    ("Picked up a refurbished MacBook Air.", "My laptop is a ThinkPad."),
    ("Learning Italian on Duolingo.", "I'm studying Spanish."),
    ("Coffee's out, it's peppermint tea now.", "Two cups of strong filter coffee, no milk."),
    ("Deleted Duolingo.", "Learning Italian on Duolingo, ten minutes at lunch."),
    ("Subway all the way now, Line 1 from Finch.", "Getting to campus means the 506 streetcar."),
])
def test_the_users_own_update_is_not_about_a_stranger(new, old):
    assert not conflict.different_subjects(new, old)


@pytest.mark.parametrize("new,old", [
    ("Yaw just started at a fintech in Accra.", "I start an internship on Monday."),
    ("Santiago lost his first tooth tonight.", "Mercado Pago terminals replaced Clip at the counter."),
    ("Devan writes Rust daily.", "The user's favorite language is Rust."),
    ("Priya works at Acme as a designer.", "Tom works at Acme as a designer."),
    ("My sister lives in Madrid.", "I live in Leeds."),
])
def test_another_persons_statement_is_a_different_subject(new, old):
    assert conflict.different_subjects(new, old)


@pytest.mark.parametrize("new,old", [
    ("Danny just got a new job at Stripe.", "My brother Danny works at Google."),
    ("Tolu moved back to Lagos last month.", "My sister Tolu lives in Toronto."),
    ("Ada left to start her own thing.", "My cofounder Ada handles the finance side."),
    ("Mum moved to Nara last month.", "My mother lives in Kyoto."),
])
def test_a_person_the_old_memory_names_is_the_same_subject(new, old):
    assert not conflict.different_subjects(new, old)


@pytest.mark.parametrize("new,old", [
    # "<Name> is my <role>" names a value of the user's own slot.
    ("Henrik started on the ward this week as the new manager.", "Kjersti is my ward manager."),
    ("Go is my main language now.", "Python is my main language."),
    # Both speak of the user, whatever their first word is.
    ("Bloodwork looks great, so I'm off statins.", "Muscle aches on the atorvastatin, so the GP switched me."),
])
def test_a_statement_that_names_a_value_of_the_users_slot_is_about_the_user(new, old):
    assert not conflict.different_subjects(new, old)


def test_a_relatives_synonyms_are_one_relative():
    assert conflict.actor("Mum moved to Nara.") == conflict.actor("My mother lives in Kyoto.")
    assert conflict.actor("My dad works at Acme.") != conflict.actor("My mum works at Acme.")


# ── what two statements can share ────────────────────────────────────────────


def _sim(a, b):
    return memory._similarity(a, b)


def _resolve(new, old, *, sim=None, old_topic=None, **kw):
    t, k = topics.classify(new)
    cand = {"id": "old", "text": old, "topic": old_topic or topics.topic_of(old), "kind": topics.kind_of(old)}
    extra = {"semantic": lambda _id: sim} if sim is not None else {}
    return conflict.resolve(new, [cand], similarity=_sim, duplicate_threshold=0.85,
                            new_topic=t, new_kind=k, **extra, **kw).superseded_ids


def test_a_contraction_is_not_a_shared_word():
    # "I've" / "We've" both leave a "ve" behind a naive word split.
    assert _resolve("We've moved! Ground-floor flat in Fana. No more stairs.",
                    "Hi! I'm Ingrid, I've been a cardiology nurse in Bergen for twelve years.") == []


def test_a_filler_word_is_not_a_shared_word():
    # Both say "so"; that is the whole overlap.
    assert _resolve("The new rota killed Tuesdays, so swimming has moved to Wednesday evenings.",
                    "Coffee's out. Night shifts made my heart race, so it's peppermint tea now.") == []


def test_the_word_that_says_something_changed_is_not_what_two_statements_share():
    assert _resolve("The new rota killed Tuesdays, so swimming has moved to Wednesdays.",
                    "We've moved! Ground-floor flat in Fana with a little garden.") == []


def test_too_before_an_adjective_is_not_an_addition():
    assert _resolve("Moved my notes over to Obsidian, Notion got too slow.",
                    "I use Notion for all my notes and to-dos.") != []
    assert _resolve("I use Vim too.", "I use Emacs.") == []


def test_a_condition_or_a_without_clause_is_not_a_claim_of_change():
    assert _resolve("I cycle to school when it isn't raining.",
                    "I teach year 4 at a primary school in Lisbon.") == []
    assert _resolve("Ordered dinner in Italian without anyone switching to English.",
                    "Learning Italian on Duolingo.", sim=0.62) == []


def test_a_numbered_thing_given_a_new_number_is_replaced_without_a_word_of_change():
    new, old = "Arne gave me an iPhone 15 for my birthday.", "Phone situation: iPhone 12, and the battery is flat."
    assert _resolve(new, old, sim=0.5) == ["old"]
    assert _resolve("Arne gave me a bunch of books for my birthday.", old, sim=0.5) == []


# ── the meaning arm's guards ─────────────────────────────────────────────────


def test_a_pure_meaning_match_needs_the_caller_floor_and_a_shared_word_needs_less():
    assert _resolve("Got a Pixel 9 last week.", "My phone is an iPhone 13.", sim=0.29) == []
    assert _resolve("Got a Pixel 9 last week.", "My phone is an iPhone 13.", sim=0.31) == ["old"]
    assert _resolve("Swapped the Golf for a Leaf last week.", "I drive a Volkswagen Golf.", sim=0.26) == ["old"]
    assert _resolve("Swapped the Golf for a Leaf last week.", "I drive a Volkswagen Golf.", sim=0.24) == []


def test_a_shared_word_with_no_meaning_behind_it_retires_nothing():
    # "flat" in a flat and in a battery that is flat; "isn't" is a cue.
    assert _resolve("We've moved to a ground-floor flat in Fana, no more stairs.",
                    "Phone situation: iPhone 12, and the battery is flat by mid-afternoon.", sim=0.15) == []


def test_two_nearly_tied_candidates_retire_neither():
    cands = [{"id": "a", "text": "I drive a Honda Civic.", "topic": None, "kind": "fact"},
             {"id": "b", "text": "I commute by car most days.", "topic": None, "kind": "fact"}]
    sims = {"a": 0.46, "b": 0.43}
    result = conflict.resolve("Got a Tesla last month.", cands, similarity=_sim, duplicate_threshold=0.85,
                              semantic=sims.__getitem__)
    assert result.superseded_ids == []
    sims["b"] = 0.30
    result = conflict.resolve("Got a Tesla last month.", cands, similarity=_sim, duplicate_threshold=0.85,
                              semantic=sims.__getitem__)
    assert result.superseded_ids == ["a"]


@pytest.mark.parametrize("new,old", [
    ("Bought a road bike for the weekends.", "Bought a standing desk for the home office."),
    ("Got a cat last month, a grey tabby called Mochi.", "We have a dog called Rex."),
    ("Joined a climbing gym on Thursdays.", "Started pottery on Thursday evenings."),
])
def test_two_acquisitions_are_not_one_slot_with_two_values(new, old):
    assert _resolve(new, old, sim=0.55) == []


@pytest.mark.parametrize("new,old", [
    ("Got a new keyboard for my laptop.", "My laptop is a ThinkPad X1."),
    ("Bought a case for my phone.", "My phone is a Pixel 8."),
    ("Got new tyres for the Civic yesterday.", "I drive a Honda Civic."),
    ("Replaced the tyres on the Focus yesterday.", "I drive a Ford Focus to work every day."),
    ("I bought a new bike lock for my commute.", "I drive a Ford Focus to work every day."),
])
def test_a_part_or_accessory_is_not_a_new_value_of_the_thing(new, old):
    assert _resolve(new, old, sim=0.6) == []


def test_an_accessory_blocks_every_other_candidate_too():
    cands = [{"id": "laptop", "text": "My laptop is a ThinkPad X1.", "topic": None, "kind": "fact"},
             {"id": "phone", "text": "My phone is a Pixel 8.", "topic": None, "kind": "fact"}]
    sims = {"laptop": 0.5, "phone": 0.34}
    assert conflict.resolve("Got a new keyboard for my laptop.", cands, similarity=_sim,
                            duplicate_threshold=0.85, semantic=sims.__getitem__).superseded_ids == []


def test_a_replacement_that_says_what_it_is_for_still_replaces():
    assert _resolve("Switched to Obsidian for my notes.", "I use Notion for all my notes.", sim=0.5) != []


@pytest.mark.parametrize("new,old", [
    ("Isla started violin lessons last month.", "I play violin in a community orchestra."),
    ("My husband got a new laptop for work.", "I use a MacBook Air for marking."),
    ("My brother just got a job at Google.", "I work at Acme."),
])
def test_someone_elses_statement_does_not_retire_yours_by_meaning(new, old):
    assert _resolve(new, old, sim=0.6) == []


def test_a_named_persons_statement_is_retired_only_by_one_that_names_them():
    assert _resolve("Mercado Pago terminals replaced Clip at the counter this week.",
                    "Santiago lost his first tooth tonight and the tooth fairy paid in pesos.", sim=0.5) == []


def test_a_statement_about_a_named_scope_is_not_the_users_own_slot():
    assert _resolve("Decided the alert routing for Indigo is Postgres (revision 2).",
                    "I use Postgres for the main database.") == []


@pytest.mark.parametrize("text", [
    "I like hiking at the weekends.",
    "Isla loves swimming.",
    "We have a dog called Rex.",
])
def test_the_things_a_person_has_several_of_are_never_swapped_by_overlap(text):
    assert topics.is_multi_valued(topics.topic_of(text))
    assert _resolve("I like pottery.", "I like hiking at the weekends.") == []
    assert _resolve("We adopted a cat called Miso.", "We have a dog called Rex.") == []


def test_a_move_names_the_residence_slot_even_when_it_mentions_the_commute():
    new = "Moved to Victoria Island in April, so the commute is ten minutes now."
    old = "I live in Lekki, about forty minutes from the office."
    assert topics.topic_of(new) == topics.topic_of(old) == "residence"
    assert _resolve(new, old) == ["old"]


def test_getting_to_school_is_a_commute():
    assert topics.topic_of("I cycle to school when it isn't raining.") == "commute"
    assert topics.topic_of("Ferry to the south bank is how I get to school now.") == "commute"
    assert _resolve("Ferry to the south bank is how I get to school now.",
                    "I cycle to school when it isn't raining.") == ["old"]


def test_a_relatives_move_retires_the_relatives_old_home_not_the_users():
    assert _resolve("My brother moved to Seattle in the spring.", "My brother lives in Denver.") == ["old"]
    assert _resolve("My brother moved to Seattle in the spring.", "I live in Denver.") == []


# ── the write path: the encoder is optional and never waited for ─────────────

CONCEPTS = {
    "phone": {"phone", "pixel", "iphone", "handset", "android"},
    "car": {"car", "drive", "civic", "tesla", "vehicle", "driving"},
    "city": {"live", "flat", "leeds", "bristol", "city"},
}


class _Encoder:
    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen.extend(texts)
        return [[1.0 if set(re.findall(r"[a-z]+", t.lower())) & g else 0.0 for g in CONCEPTS.values()] + [0.05]
                for t in texts]


@pytest.fixture
def encoder(monkeypatch):
    fake = _Encoder()
    monkeypatch.setattr(retrieval, "_encoder", fake)
    monkeypatch.setattr(retrieval, "_failed", False)
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS", "1")
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE", "1")
    return fake


@pytest.fixture
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")


def test_an_update_dated_with_a_time_phrase_retires_the_old_value(encoder, clean_memory):
    first = memory.remember("My phone is an iPhone 13.")
    second = memory.remember("Got a Pixel 9 last week.")
    assert second["superseded"] == [first["id"]]
    assert memory.get(second["id"])["kind"] == "fact"


def test_a_real_event_beside_it_retires_nothing(encoder, clean_memory):
    first = memory.remember("I drive a 2016 Honda Civic.")
    second = memory.remember("Went to a Tesla showroom last week.")
    assert second["superseded"] == []
    assert memory.get(second["id"])["kind"] == "event"
    assert memory.get(first["id"])["superseded_by"] is None


def test_a_store_never_embedded_is_filled_behind_the_callers_back(encoder, clean_memory, monkeypatch):
    monkeypatch.setattr(embeddings, "SYNC_LIMIT", 3)
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS", "0")
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE", "0")
    for i in range(8):
        memory.remember(f"Note {i}: I drive a car on road {i}.")
    monkeypatch.setenv("PRIMNOX2_MEMORY_EMBEDDINGS", "1")
    rows = [{"id": r["id"], "text": r["text"]} for r in memory.live()]
    encoder.seen.clear()
    assert embeddings.vectors(rows) is None          # too many to encode inline
    assert encoder.seen == []                        # and none were, on this thread
    embeddings.fill()
    assert embeddings.vectors(rows) is not None      # now every one is cached
    assert len(embeddings.vectors(rows)) == len(rows)


# ── medicines and stopping ───────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "My GP put me on atorvastatin after the cholesterol numbers came back.",
    "I take bisoprolol, five milligrams.",
    "Doctor changed my beta-blocker to metoprolol.",
    "Started amoxicillin on Tuesday.",
])
def test_a_drug_is_named_by_its_stem(text):
    assert topics.topic_of(text) == "medication"


def test_a_month_that_ends_like_a_drug_is_not_one():
    assert topics.topic_of("We start in April.") is None


def test_stopping_a_drug_class_retires_the_drug_in_it():
    new = "Bloodwork looks great, so I'm off statins entirely, doctor's orders."
    old = "Muscle aches on the atorvastatin, so the GP switched me to rosuvastatin."
    assert conflict._inside("statin", "rosuvastatin")
    assert _resolve(new, old) == ["old"]
    # …and a second drug is added, never swapped in.
    assert _resolve("I also take aspirin every morning.", old) == []


def test_off_to_somewhere_is_not_stopping_anything():
    assert conflict._REVERSAL.search("I'm off to Paris on Friday") is None
    assert conflict._REVERSAL.search("I'm off work today") is None
    assert conflict._REVERSAL.search("I'm off caffeine")


def test_the_thing_changed_is_not_part_of_the_wording_that_says_it_changed():
    new = "Doctor changed my beta-blocker to metoprolol after the check-up."
    old = "I take beta-blockers for my heart rhythm, bisoprolol, five milligrams."
    assert _resolve(new, old) == ["old"]
