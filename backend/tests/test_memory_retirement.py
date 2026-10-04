"""Memory retirement and merging: what the store must never do to a true fact.

A memory that is retired leaves every future prompt, and one that is dropped as
a duplicate is never stored. For an allergy or a medication that is the
assistant forgetting something that can hurt, so these pin the cases where the
old rules got it wrong -- all found by running real sentence pairs through both
write paths (`remember` and `import_many`), not by reading the code.
"""
from __future__ import annotations

import pytest

from primnox2.cognition import conflict, topics
from primnox2.memory import service as memory
from primnox2.settings import tunables
from primnox2.storage import db

BASE_MS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def clean_memory():
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    yield
    with db.tx() as c:
        c.execute("DELETE FROM memories")
    tunables.set_many({"memory.duplicate_threshold": 0.85})


@pytest.fixture(params=["remember", "import"])
def write(request):
    """Write a sequence of statements the way one of the two paths would, and
    return (live, retired) texts."""
    def go(*statements: str) -> tuple[list[str], list[str]]:
        if request.param == "remember":
            for text in statements:
                memory.remember(text)
        else:
            memory.import_many([{"text": t, "created_at": BASE_MS + i * 1000}
                                for i, t in enumerate(statements)])
        rows = memory.live(limit=1_000_000)
        return ([r["text"] for r in rows if not r.get("superseded_by")],
                [r["text"] for r in rows if r.get("superseded_by")])
    return go


def _both_survive(write, *statements):
    live, retired = write(*statements)
    assert retired == [], f"retired {retired}"
    assert sorted(live) == sorted(statements), f"stored {live}"


# ── siblings are different facts ─────────────────────────────────────────────

def test_a_sister_and_a_brother_in_the_same_city_both_stand(write):
    _both_survive(write, "My sister lives in Pune", "My brother lives in Pune")


def test_two_pets_named_in_the_same_shape_both_stand(write):
    _both_survive(write, "My cat Miso", "My cat Mochi")


def test_one_new_name_in_a_long_sentence_is_not_a_restatement(write):
    """Jaccard on a 13-word sentence with one word changed is 0.86, over the
    0.85 threshold, so the second pet was dropped as a duplicate."""
    _both_survive(
        write,
        "My cat Miso is a grey tabby who loves sitting by the window every morning",
        "My cat Mochi is a grey tabby who loves sitting by the window every morning")


def test_two_allergies_in_long_sentences_are_both_kept(write):
    _both_survive(
        write,
        "I am allergic to peanuts and carry an epipen everywhere I go including work",
        "I am allergic to penicillin and carry an epipen everywhere I go including work")


def test_siblings_survive_a_loosened_duplicate_threshold(write):
    tunables.set_many({"memory.duplicate_threshold": 0.6})
    _both_survive(write, "My sister lives in Pune", "My brother lives in Pune")


def test_a_negation_is_not_a_restatement(write):
    """Dropping "I am not allergic to ..." as a duplicate of the allergy would
    leave the store contradicting what the user just said."""
    live, _ = write(
        "I am allergic to peanuts and tree nuts and carry an epipen at all times everywhere",
        "I am not allergic to peanuts and tree nuts and carry an epipen at all times everywhere")
    assert len(live) == 2


def test_a_real_restatement_is_still_merged(write):
    live, retired = write("They deploy on Tuesday mornings.", "They deploy on Tuesday mornings")
    assert len(live) == 1 and retired == []


def test_the_same_long_fact_with_only_filler_changed_is_still_merged(write):
    live, _ = write(
        "My cat Miso is a grey tabby who loves sitting by the window every morning",
        "My cat Miso is a grey tabby who loves sitting by the window every morning.")
    assert len(live) == 1


# ── a safety fact is not retired by an unrelated statement ───────────────────

def test_a_second_medication_does_not_retire_the_first(write):
    _both_survive(write, "I take metformin every morning.", "I take aspirin every morning.")


def test_a_vitamin_does_not_retire_insulin(write):
    _both_survive(write, "I take insulin before meals.", "I take vitamin C before meals.")


def test_a_second_prohibition_does_not_retire_the_first(write):
    _both_survive(write, "I can't take aspirin.", "I can't take ibuprofen.")


def test_a_diet_statement_does_not_retire_a_gluten_restriction(write):
    _both_survive(write, "My diet is gluten-free.", "My diet is keto.")


@pytest.mark.parametrize("later", [
    "I'm also allergic to peanuts.",
    "I'm allergic to peanuts.",
    "I have a cat allergy.",
    "Allergic to latex.",
])
def test_another_allergy_never_retires_an_allergy(write, later):
    _both_survive(write, "I'm allergic to penicillin.", later)


@pytest.mark.parametrize("later", [
    "I'm trying a keto diet now.",
    "I no longer eat sugar.",
    "My diet is mostly vegetarian.",
    "Doctor switched my blood pressure med to amlodipine.",
])
def test_diabetes_is_not_retired_by_other_health_or_diet_talk(write, later):
    _both_survive(write, "Diabetic, type 2.", later)


def test_a_blood_pressure_change_does_not_touch_an_allergy(write):
    _both_survive(write, "I'm allergic to penicillin.",
                  "Doctor switched my blood pressure med to amlodipine.")


def test_stopping_the_named_medication_does_retire_it(write):
    live, retired = write("I take metformin 500mg twice a day for type 2 diabetes.",
                          "I'm no longer on metformin, switched to insulin.")
    assert retired == ["I take metformin 500mg twice a day for type 2 diabetes."]


def test_stopping_a_different_supplement_does_not(write):
    _both_survive(write, "I take metformin 500mg twice a day for type 2 diabetes.",
                  "I stopped taking vitamin D.")


def test_stopping_the_same_daily_medication_retires_it(write):
    _, retired = write("I take metformin every morning.", "I stopped taking metformin.")
    assert retired == ["I take metformin every morning."]


def test_safety_facts_are_recognised_without_the_service_helper():
    assert topics.is_safety("I can't take aspirin.")
    assert topics.is_safety("My diet is gluten-free.")
    assert not topics.is_safety("I use Helix.")


# ── "favorite X" of a different X is a different fact ────────────────────────

def test_a_favorite_number_does_not_retire_a_favorite_language(write):
    _both_survive(write, "The user's favorite programming language is Rust",
                  "The user's favorite number is 17")


def test_a_favorite_language_does_not_retire_a_favorite_number(write):
    _both_survive(write, "The user's favorite number is 17",
                  "The user's favorite language is Rust")


def test_a_favorite_color_does_not_retire_a_favorite_language(write):
    _both_survive(write, "The user's favorite color is blue",
                  "The user's favorite language is Rust")


def test_the_same_favorite_reworded_is_still_an_update(write):
    _, retired = write("My favorite language is Python",
                       "My favorite programming language is Rust")
    assert retired == ["My favorite language is Python"]


def test_a_floor_exact_score_is_not_the_reason(write):
    """The pair scores exactly CONFLICT_FLOOR; the fix is the slot key, not
    moving the boundary, so a pair just under or over it behaves the same."""
    _both_survive(write, "The user's favorite programming language is Rust",
                  "The user's favorite number is 17")
    assert conflict.CONFLICT_FLOOR == 0.5


# ── a reversal verb plus one generic shared word is not evidence ─────────────

def test_a_reversal_sharing_only_the_word_project_retires_nothing(write):
    _both_survive(write, "The Zircon-Falcon risk rating is amber for this project",
                  "ZPROBE switched to Rust for this project")


def test_a_reversal_sharing_a_generic_word_with_an_unnamed_rating(write):
    _both_survive(write, "The project risk rating is amber",
                  "ZPROBE switched to Rust for this project")


def test_a_reversal_that_names_the_same_thing_still_retires(write):
    _, retired = write("I prefer Rust for this project", "I switched to Rust for this project")
    assert retired == ["I prefer Rust for this project"]


# ── events and other people's facts do not compete with the user's ───────────

@pytest.mark.parametrize("event", [
    "Finished the Rust book in March",
    "Finished reading the Rust book",
])
def test_a_finished_book_is_an_event_and_retires_nothing(write, event):
    assert topics.kind_of(event) == "event"
    _both_survive(write, event, "The user's favorite language is Rust")


def test_a_named_person_is_not_the_user(write):
    _both_survive(write, "Devan writes Rust daily", "The user's favorite language is Rust")


def test_a_named_person_is_not_the_user_in_the_other_order(write):
    _both_survive(write, "The user's favorite language is Rust", "Devan writes Rust daily")


# ── what must keep working ───────────────────────────────────────────────────

def test_one_slot_under_two_words_is_still_an_update(write):
    _, retired = write("My manager is Tomas Berg", "My boss is Priya Nair")
    assert retired == ["My manager is Tomas Berg"]


def test_a_swapped_value_in_a_long_single_valued_slot_is_an_update_not_a_duplicate(write):
    """Scores 0.88, over the duplicate threshold, so the new editor used to be
    dropped as a restatement and the old one stayed current."""
    live, retired = write(
        "I use Helix as my main editor for all of my work and side projects at home",
        "I use Neovim as my main editor for all of my work and side projects at home")
    assert retired == ["I use Helix as my main editor for all of my work and side projects at home"]
    assert len(live) == 1


@pytest.mark.parametrize("old,new", [
    ("My name is Sam", "My name is Alex"),
    ("I live in Lisbon", "I live in Berlin"),
    ("My sister lives in Pune", "My sister lives in Delhi"),
    ("My sister Priya lives in Pune", "Priya lives in Delhi"),
    ("I prefer tabs for indentation", "I switched to spaces for indentation"),
    ("I use Postgres for the main database", "I switched to MySQL for the main database"),
])
def test_a_genuine_update_still_supersedes(write, old, new):
    _, retired = write(old, new)
    assert retired == [old]


def test_two_different_relatives_are_different_subjects_and_one_person_is_not():
    assert conflict.different_subjects("My sister lives in Pune", "My brother lives in Pune")
    assert conflict.different_subjects("My cat Miso", "My cat Mochi")
    assert conflict.different_subjects("Devan writes Rust daily", "The user's favorite language is Rust")
    assert not conflict.different_subjects("My sister Priya lives in Pune", "Priya lives in Delhi")
    assert not conflict.different_subjects("I live in Lisbon", "I'm moving to Berlin")
    assert not conflict.different_subjects("My manager is Tomas", "My boss is Priya")


def test_restatement_needs_no_swapped_word_and_no_negation():
    assert conflict.restates("I use Postgres for the main database",
                             "I use Postgres for the main database.")
    assert not conflict.restates("allergic to peanuts and tree nuts here",
                                 "allergic to penicillin and tree nuts here")
    assert not conflict.restates("I am allergic to nuts", "I am not allergic to nuts")
    assert not conflict.restates("uses Postgres", "uses Postgres 16")


def test_a_capitalised_attribute_word_is_not_a_person(write):
    """"Budget is 200 USD per day" opens with a capital and names no one. Read
    as a name it made every first-person statement a different subject, so the
    changed budget never superseded it."""
    _, retired = write("Budget is 200 USD per day",
                       "The user's budget is 120 USD per day")
    assert retired == ["Budget is 200 USD per day"]
