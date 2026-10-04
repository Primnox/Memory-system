"""The Memory Conflict Engine — decides whether a new memory updates an old
one or is simply new.

`memory/service.py::remember()` already has duplicate suppression (Jaccard
similarity >= 0.85 means "the same restated fact, don't store it twice") but
nothing between that and "unrelated, store both forever". Two memories that
plainly overlap in subject WITHOUT being the same sentence — "prefers
Python" then, weeks later, "switched to Rust" — currently sit side by side
in the store permanently, and `render_for_prompt()` injects both into every
future turn with no indication either one is stale. The model receives a
user who apparently prefers two different languages at once.

`v2/world_model.py` already solves this for observed Facts — slot-based and
similarity-based conflict detection, `is_stronger()` confidence comparison,
`on_conflict` modes. This module is the same PATTERN, not the same storage:
`memories` (explicit, user-told facts, in `primnox.db`) and `facts`
(observed, in a separate `primnox_v2.db`) stay two different tables on
purpose — see that module's own docstring for why "the user told me" and
"I worked this out" are different claims. What's ported here is the
judgement, not the schema: a supersession decision is the same reasoning
either way.

`remember()` is always treated as an explicit correction (`on_conflict` is
not exposed as a parameter, unlike `world_model.remember()`) — a user
actively saying "remember that X" about something similar to what's already
stored is a deliberate update far more often than a weak, uncertain new
belief, which is the case `world_model.py`'s "auto" mode exists to hedge
against for facts the SYSTEM inferred on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Callable

from . import topics as _topics

# A reversal reuses almost none of the original's words — "prefers Python"
# then "switched to Rust" — so Jaccard puts it far below CONFLICT_FLOOR and
# the band check alone never sees it. These markers name the shape of an
# update in plain language; paired with a shared subject word (below) they
# let a reversal supersede without the lexical overlap the band needs.
_REVERSAL = re.compile(
    r"\b(no longer|not any\s?more|used to|no more|"
    r"stopped|quit|gave up|dropped|abandoned|ditched|"
    r"switch(?:ed|ing)|moved (?:[\w'-]+ ){0,3}?(?:from|to|away|off)|"
    r"changed (?:[\w'-]+ ){0,3}?(?:from|to)|migrated (?:[\w'-]+ ){0,3}?(?:from|to|off)|went back to|"
    r"replaced .+ with|instead of|rather than|"
    r"isn't|aren't|is not|are not|doesn't|don't|does not|do not|"
    r"never again|not anymore|(?:i'm|i am|came|come|coming|gone|went|getting) off "
    r"(?!(?:to|work|today|tomorrow|duty|sick|shift)\b))\b",
    re.I,
)

# "I also use Vim" adds to what is already true; it does not replace it.
_ADDITIVE = re.compile(r"\b(?:also|as well|in addition|on top of that|besides)\b|\btoo\b(?!\s+\w)", re.I)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*", re.I)
_CONTRACTION = re.compile(r"(?<=\w)(?:n['’]t|['’](?:s|ve|ll|re|d|m))\b", re.I)
_FILLER = frozenset(
    "the a an and or but not no of to in on at for with from by as is are was "
    "were be been being do does did have has had will would can could should "
    "may might must i you he she it we they me my your his her our their this "
    "that these those now then here there also just very more most any some "
    "prefer prefers preferred like likes use uses using used ca wo sha".split())


# The helpers below are pure functions of one text, called once per candidate
# per write. Measured at 10,000 memories, re-running their regexes on every
# stored sentence for every new one was most of what `resolve()` cost, so each is
# memoised on the text itself. Keyed by text rather than by memory id, a cached
# value can never be stale: an edited memory is simply a different key.
_MEMO = 65_536


@lru_cache(maxsize=_MEMO)
def _content(text: str) -> frozenset:
    # The class lets "." through for "node.js" and "3.5", which also leaves a
    # sentence-final period on the last word: "I own a car." gave "car." and
    # never met "I don't have a car anymore", so that negation retired nothing.
    # "I've" and "we've" otherwise leave a shared "ve", and "don't" a shared
    # "don": a contraction is a function word, not a word two statements share.
    text = _CONTRACTION.sub("", text)
    return frozenset({w.lower().rstrip(".-") for w in _WORD_RE.findall(text)} - _FILLER)


# Words that two unrelated statements share by accident. A reversal ("switched
# to", "no longer") plus one of these is not evidence the two are about the same
# thing: "ZPROBE switched to Rust for this project" retired a risk rating
# because both said "project". The drug and the allergen are what identify a
# medical fact, not that it was "taken" in the "morning".
_GENERIC = frozenset(
    "project projects work thing things stuff time times day days week weeks "
    "month months year years today lately recently app apps code system "
    "systems team company everything anything something way lot new old good "
    "great main mostly daily often usually every everyday morning evening "
    "night take takes taking taken dose doses allergic allergy allergies "
    "medication medicine med meds afternoon weekend weekday monday tuesday "
    "wednesday thursday friday saturday sunday".split())

# What two statements can share and still be about different things, beyond the
# short `_FILLER` that the refinement and duplicate checks depend on: "so" is in
# half of all sentences, and once retired a coffee memory because a swimming
# memory also said it.
_STOP = _FILLER | frozenset(
    "so up out off over into onto about than after before while when where what which "
    "who whom whose how why if because all each every one two three four five six "
    "seven eight nine ten first second last next other another both either neither "
    "only even still yet ever never always already again once twice much many few "
    "little less least lot lots own same such quite really get gets got getting go "
    "goes going went gone make makes made making keep kept let put take took come "
    "came see saw say said tell told ask asked think thought know knew want wants "
    "wanted need needs needed try tried tries someone anyone everyone nobody kind "
    "sort bit pretty well down around through across along against between among "
    "during since until within without per via".split())


def _root(word: str) -> str:
    for _ in range(2):
        if len(word) > 4:
            word = re.sub(r"(?:ing|ed|es|s)$", "", word)
    return word


_GENERIC_ROOTS = frozenset(_root(w) for w in _GENERIC)


def _anchors(words: frozenset) -> frozenset:
    """The words of a statement worth matching another on: not a function word,
    not one of the generic words, with plurals and tenses folded together."""
    return frozenset(_root(w) for w in words if w not in _STOP and _root(w) not in _GENERIC_ROOTS)


# A statement that opens with a person or thing by name ("Nadia Okonkwo is a
# CTO", "Omar Okonkwo is an Engineering Manager") is about THAT subject. Two
# such statements with different subjects are two facts, however much of the
# sentence shape they share: measured on a 100-memory synthetic life, the
# word-overlap band alone superseded 63 memories that were never stale, every
# one a "<Name> is a <role> at <Org>" sibling of another person's line.
_SUBJECT_RE = re.compile(r"^((?:[A-Z][\w'’\-]+)(?:\s+[A-Z][\w'’\-]+)*)\s+\S")
_NOT_SUBJECT = frozenset({"I", "My", "The", "A", "An", "User", "Users", "They",
                          "He", "She", "We", "It", "This", "That", "Our", "Doctor",
                          "I'm", "I've", "I'll", "I'd", "We're", "We've", "They're",
                          "He's", "She's", "It's", "That's", "Today", "Now",
                          "Always", "Never", "Also", "Allergic", "Diabetic",
                          "No", "Not", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                          "Eight", "Nine", "Ten", "Several", "Many", "Few", "Some", "Most",
                          "Half", "Both", "Each", "Every", "All", "Another", "Other"})

# Every sentence starts with a capital, so a leading capitalised word is a name
# only when something says so. The user's own updates are written with the
# subject left out — "Got a Pixel 9", "Moved to Lisbon", "Subway all the way now"
# — and read as a stranger's by the old rule, which then called every one of
# them "a different subject" from the user's earlier statements.
#   * a function word next ("Got a", "Moved to", "Cards at") means the capitalised
#     word is the start of a clause, not somebody;
#   * so does a gerund ("Learning Italian") or a long past participle with no
#     verb after it ("Deleted Duolingo");
#   * a name is followed by a verb, a copula or "'s" ("Henrik started", "Lucía's
#     manager"), and is not a word that also appears in lower case in either text.
_FUNCTION = frozenset(
    "a an the this that these those my your his her its our their some any no all every "
    "each both another other such to at in on for with from by of about into onto over "
    "under after before during since until up down out off back away around through "
    "across against between among and or but so if when while as because than i me we "
    "us you he she they them him it myself ourselves not too also again i'm i've i'd "
    "i'll we're we've".split())
_ADVERB = frozenset("just also then now recently finally still always never really already "
                    "soon currently actually basically mostly usually often".split())
_AUXILIARY = frozenset(
    "is are was were am be been has have had do does did will would can could should may "
    "might must gave left took went got made ran came saw found told sold bought wrote "
    "spoke began knew met kept became brought held felt lost paid sent built won quit "
    "drove flew wore ate said".split())
_FIRST = re.compile(r"\b(?:i|i'm|i've|i'll|i'd|me|my|mine|myself|we|we're|we've|us|our|ours)\b", re.I)

# "My sister lives in Pune" and "My brother lives in Pune" share a sentence and
# nothing else. The relation (and a name after it: "my cat Miso") is the
# subject, exactly as a leading name is.
_RELATIONS = (
    "sister|brother|wife|husband|partner|mother|mom|mum|father|dad|son|daughter|"
    "kid|kids|child|children|parents|parent|friend|roommate|neighbou?r|"
    "grandmother|grandfather|grandma|grandpa|uncle|aunt|cousin|girlfriend|"
    "boyfriend|fianc[eé]e?|teacher|doctor|therapist|landlord|landlady|colleague|"
    "coworker|cat|dog|pet|bird|fish|rabbit|hamster")
_RELATION_RE = re.compile(
    rf"^(?i:my|our)\s+(?i:(?P<rel>{_RELATIONS}))\b"
    r"(?:\s+(?P<name>[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*))?")
# "Mum moved to Nara": a relative named by the word alone is a relation, not a name.
_BARE_KIN = re.compile(r"(?:mum|mom|mama|dad|papa|mother|father|grandma|grandpa|granny|nana)\b(?!['’]s)", re.I)
_COPULA = frozenset({"is", "are", "was", "were", "=", ":"})
_USER_RE = re.compile(r"^(?:i|i'm|i've|i'll|i'd|my|our|the user|user|users?'s?)\b", re.I)


def _verbish(word: str) -> bool:
    return word in _COPULA or word in _AUXILIARY or (
        len(word) > 3 and word.endswith(("s", "ed")) and not word.endswith("ss"))


@lru_cache(maxsize=_MEMO)
def _opening(text: str):
    """What a leading capitalised run is: ("verb",) when it starts a clause with
    its subject left out, ("name", words, copula) when it is followed the way a
    name is, else None."""
    m = _SUBJECT_RE.match(text.strip())
    if not m:
        return None
    raw = m.group(1).split()
    if raw[0] in _NOT_SUBJECT:
        return None
    words = [re.sub(r"['’]s$", "", w) for w in raw]
    rest = [re.sub(r"[^\w'’]+$", "", w).lower() for w in text.strip()[m.end(1):].split()]
    follower = next((w for w in rest if w not in _ADVERB), "")
    head = words[0].lower()
    if (follower in _FUNCTION
            or (len(head) > 5 and head.endswith("ing") and follower not in _COPULA)
            or (len(head) > 5 and head.endswith("ed") and not _verbish(follower))):
        return ("verb",)
    possessive = raw[-1].lower().endswith(("'s", "’s"))
    # After a longer run an "-s" is as likely a plural noun ("Mercado Pago
    # terminals") as a verb ("Devan writes"), so only a surer verb counts there.
    verb = (_verbish(follower) if len(words) == 1
            else follower in _AUXILIARY or follower.endswith("ed"))
    if follower in _COPULA or verb or possessive:
        return "name", frozenset(w.lower() for w in words), follower in _COPULA
    return None


def _about(text: str, other: str = ""):
    """Who a statement is about, when it says: ("relation", rel, name),
    ("name", words, followed_by_is, mentions_the_user) or ("user",).

    A leading word that also appears in lower case in either text ("Coffee's
    out" beside "…filter coffee") is a common word, not a name."""
    t = text.strip()
    rel = _RELATION_RE.match(t)
    if rel:
        return "relation", _kin(rel.group("rel")), (rel.group("name") or "").lower()
    bare = _BARE_KIN.match(t)
    if bare:
        return "relation", _kin(bare.group(0)), ""
    mentions = bool(_FIRST.search(t))
    lead = _opening(t)
    if lead and lead[0] == "verb":
        return ("user",)
    if lead and not any(re.search(rf"(?<![\w'’]){re.escape(w)}(?![\w'’])", f"{t} {other}")
                        for w in lead[1]):
        return "name", lead[1], lead[2], mentions
    if mentions or _USER_RE.match(t):
        return ("user",)
    return None


def _name_words(text: str, other: str) -> frozenset:
    about = _about(text, other)
    return about[1] if about and about[0] == "name" else frozenset()


def same_person(a: str, b: str) -> bool:
    """One of the two is led by a name the other also says: "Tolu moved back to
    Lagos" after "My sister Tolu lives in Toronto", or "Ada left" after "My
    cofounder Ada handles finance"."""
    for x, y in ((a, b), (b, a)):
        names = _name_words(x, y)
        if names and all(re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", y, re.I) for n in names):
            return True
    return False


def different_subjects(a: str, b: str) -> bool:
    """The two are about different people or things. Two leading names where
    neither contains the other; a relative against another relative, against a
    named person or against the user; a named person against the user. A shared
    surname or a shared leading verb ("Met …") is not a shared subject."""
    ka, kb = _about(a, b), _about(b, a)
    if same_person(a, b):
        return False
    if not ka or not kb:
        # "Santiago lost his first tooth" is about Santiago: a statement that
        # does not name him, or the user, cannot be its other value.
        lone = ka or kb
        return bool(lone) and lone[0] == "name" and not (lone[2] or lone[3])
    if ka[0] == "name" and kb[0] == "name":
        # "Kjersti is my ward manager" names a value of the user's own slot, and
        # "Henrik started as the new manager" may name the next one; two that both
        # speak of the user ("Bloodwork looks great, so I'm off statins") are about
        # the user, whatever their first word is.
        if (ka[2] and ka[3]) or (kb[2] and kb[3]) or (ka[3] and kb[3]):
            return False
        return not (ka[1] <= kb[1] or kb[1] <= ka[1])
    if ka[0] == "user" and kb[0] == "user":
        return False
    if ka[0] == "relation" and kb[0] == "relation":
        return ka[1] != kb[1] or bool(ka[2] and kb[2] and ka[2] != kb[2])
    if {ka[0], kb[0]} == {"name", "user"}:
        # "Budget is 200 USD per day" opens with a capital but names no one;
        # "Devan writes Rust daily" does. Only a verb after the word says so.
        # "Kjersti is my ward manager" is about the user's manager, whoever she is.
        named = ka if ka[0] == "name" else kb
        return not (named[2] or named[3])
    pair = {ka[0]: ka, kb[0]: kb}
    if len({ka[0], kb[0]}) == 2 and set(pair) == {"relation", "name"}:
        # "My sister Priya lives in Pune" and "Priya lives in Pune" are one person.
        named = pair["name"][1]
        given = pair["relation"][2].split()
        return not (given and set(given) <= named)
    return True


# What a statement is scoped TO: the named thing after "for", "about" or "of".
# "Decided the alert routing for Fathom is a queue" and "… for Ridge is one
# reviewer" share a whole sentence and are decisions about two projects —
# measured on a 400-memory pack, 61 such records retired each other.
# Locative "in"/"at" are left out on purpose: "I live in Lisbon" then "I live
# in Berlin" is a real update, and a place is the VALUE there, not the scope.
_SCOPE_RE = re.compile(r"\b(?:for|about|of)\s+((?:[A-Z][\w\-]*)(?:\s+(?:[A-Z][\w\-]*|\d+\b))*)")


@lru_cache(maxsize=_MEMO)
def _scopes(text: str) -> frozenset:
    return frozenset(m.group(1).lower() for m in _SCOPE_RE.finditer(text))


# "the deadline for Atlas is Friday", "the owner of Atlas is Sam": an attribute
# of a named thing. The (attribute, thing) pair is the slot — a new value for
# the same pair is an update, a different attribute of the same thing is not.
_ATTRIBUTE_RE = re.compile(
    r"\bthe (?P<key>[a-z][a-z \-]{1,40}?) (?:for|of) "
    r"(?P<scope>[A-Z][\w\-]*(?:\s+(?:[A-Z][\w\-]*|\d+\b))*) (?:is|are|=|:) (?P<val>.+)")


@lru_cache(maxsize=_MEMO)
def attribute_slot(text: str):
    m = _ATTRIBUTE_RE.search(text)
    if not m:
        return None
    return m.group("key").strip().lower(), m.group("scope").strip().lower()


def _outside_scope(text: str, other: str) -> bool:
    slot = attribute_slot(text)
    return bool(slot) and slot[1] not in other.lower()


# Whose statement it is. "My brother works at Google" has the shape of "I work
# at Meta" and says nothing about the user. Only consulted when meaning, not
# words, is proposing that two statements fill one slot: word overlap already
# stops a sibling's job retiring yours, and a similarity score does not.
_KIN = re.compile(
    r"\b(?:brother|sister|sibling|mum|mom|mother|dad|father|parent|wife|husband|spouse|"
    r"partner|girlfriend|boyfriend|son|daughter|child|kid|cousin|aunt|uncle|nephew|niece|"
    r"grandma|grandpa|grandmother|grandfather|friend|colleague|coworker|neighbou?r|"
    r"flatmate|roommate)s?\b", re.I)


_POSSESSIVE = re.compile(r"(?:^|\s)(?:my|our|his|her|their|your)$|['’]s$", re.I)
_KIN_ALIAS = {"mum": "mother", "mom": "mother", "dad": "father", "kids": "kid", "child": "kid",
              "children": "kid", "parents": "parent", "neighbour": "neighbor", "neighbours": "neighbor",
              "kid": "kid", "friends": "friend", "colleagues": "colleague", "coworkers": "coworker"}


def _kin(word: str) -> str:
    word = word.lower()
    return _KIN_ALIAS.get(word, word)


@lru_cache(maxsize=_MEMO)
def actor(text: str) -> tuple:
    """Who a statement is about: the relatives in its first words ("my wife's
    sister" and "my wife" are two people), a bare pronoun, or () for the user.
    A capitalised name is NOT read as another person: memories distilled from a
    chat are often "Priya prefers …", with the user's own name, and treating
    that as someone else blocked the very updates it describes. Two different
    names are `different_subjects`' job."""
    t = text.strip()
    # A relative is the subject when the sentence opens with one or says whose
    # ("my wife's sister"); "hockey with colleagues" is the user's own.
    words = t.split()[:4]
    kin = tuple(_kin(m.group(0)) for m in _KIN.finditer(" ".join(words))
                if m.start() == 0 or _POSSESSIVE.search(" ".join(words)[:m.start()].rstrip()))
    if kin:
        return kin
    if re.match(r"(?:he|she|they)\b", t, re.I):
        return ("other",)
    return ()


# "Another cat" adds to what is true; it does not replace it.
_ADDITION = re.compile(r"\b(?:another|a second|one more|additional|extra|both)\b", re.I)

# Wording that says something CHANGED, beyond the `_REVERSAL` shapes: shared
# with `topics.kind_of`, which must not file such a statement as a one-off event.
_CHANGE = _topics.TRANSITION

# "…ordered dinner in Italian without anyone switching to English" and "I cycle
# to school when it isn't raining" say nothing changed: what follows "without"
# is what did NOT happen, and a "when"/"if" clause is a condition, not a claim.
_NOT_ASSERTED = re.compile(r"\b(?:without|when|whenever|if|unless)\b[^,.;!?]*", re.I)


def _asserted(text: str) -> str:
    return _NOT_ASSERTED.sub(" ", text)


@lru_cache(maxsize=_MEMO)
def _cue_words(text: str) -> frozenset:
    """The words that make a statement read as a change. They say THAT something
    changed, so sharing one with an old memory ("We've moved" / "swimming has
    moved to Wednesday") is no evidence the two are about the same thing."""
    # The verb only: "changed my beta-blocker to" also holds the thing changed.
    verbs = (m.group(0).split()[0] for r in (_REVERSAL, _CHANGE) for m in r.finditer(_asserted(text)))
    return frozenset(_root(w) for v in verbs for w in _content(v))


def different_scopes(a: str, b: str) -> bool:
    """Both are scoped to named things, and to none in common."""
    sa, sb = _scopes(a), _scopes(b)
    return bool(sa and sb and not (sa & sb))


def is_refinement(a: str, b: str) -> bool:
    """`b` restates `a` with one extra qualifier — "uses Postgres" then "uses
    Postgres 16". A refinement is new information, not a duplicate and not a
    contradiction: both rows should survive. Recognised by the whole
    difference between the two being tokens that carry a digit (a version, a
    quantity, a year)."""
    diff = _content(a) ^ _content(b)
    return bool(diff) and all(any(ch.isdigit() for ch in w) for w in diff)


_NEGATION = re.compile(r"\b(?:not|no|never|none|without|cannot|neither|nor)\b|n't\b", re.I)


def _stem(word: str) -> str:
    return re.sub(r"(?:ing|ed|es|s)$", "", word) if len(word) > 4 else word


def is_distinct(a: str, b: str) -> bool:
    """Two sentences that score as near-identical and are still different facts:
    a content word swapped for another ("allergic to peanuts" / "allergic to
    penicillin", "my cat Miso" / "my cat Mochi"), or a negation on one side
    only. Word overlap cannot see either — on a 13-word sentence one swapped
    word scores 0.86 — and treating them as a restatement drops the second
    fact, which for an allergy is the assistant never learning it.

    A word only ONE side has (an extra qualifier, a filler) is not a swap: that
    is the restatement the duplicate check exists to merge."""
    if bool(_NEGATION.search(a)) != bool(_NEGATION.search(b)):
        return True
    sa = {_stem(w) for w in _content(a)}
    sb = {_stem(w) for w in _content(b)}
    return bool(sa - sb) and bool(sb - sa)


def restates(a: str, b: str) -> bool:
    """`a` says what `b` already says: the same sentence, not a refinement of it
    and not a different fact that happens to share most of its words."""
    return not is_refinement(a, b) and not is_distinct(a, b)


# "the user's favorite number is 17", "my main database is Postgres": a
# possessor, the attribute, and a value. Two of these about DIFFERENT
# attributes are different facts however much of the template they share —
# "favorite language is Rust" and "favorite number is 17" score exactly
# CONFLICT_FLOOR on the words "the user's favorite … is", and the old order of
# checks retired one with the other.
_ATTRIBUTE_KEY_RE = re.compile(
    r"^(?:my|our|the user'?s?|users?'?s?|the)\s+"
    r"(?P<key>[a-z][a-z \-]{1,40}?)\s+(?:is|are|=|:)\s+\S", re.I)
_KEY_ALIASES = {"db": "database", "ide": "editor", "zone": "timezone", "system": "os",
                "title": "role", "job": "role", "company": "employer", "city": "location"}


def _attribute_key(text: str):
    m = _ATTRIBUTE_KEY_RE.match(text.strip())
    if not m:
        return None
    noun_phrase = re.split(r"\s+(?:for|of|in|on|at)\s+", m.group("key").lower())[0]
    head = noun_phrase.split()[-1]
    return _KEY_ALIASES.get(head, head)


# "my favourite X is Y", "my main database is Y", "my name is Y" — a small,
# high-confidence set of shapes where two statements about the SAME key with
# DIFFERENT values are a contradiction even when the values share no words
# ("favourite language is Python" vs "favourite language is Rust").
_SLOT_RE = re.compile(
    r"^(?:my |the user's |user'?s? |i'?m? |i am |the )?"
    r"(?P<key>favou?rite [a-z ]{2,25}?|name|"
    r"(?:daily |monthly |total |travel |weekly )?budget|diet|deadline|"
    r"(?:main |primary |default |preferred )?"
    r"(?:language|editor|ide|database|db|browser|os|operating system|shell|"
    r"distro|framework|city|country|location|timezone|time zone|role|title|"
    r"job|company|employer|team|phone|email))"
    r"\s*(?:is|are|=|:)\s+"
    r"(?P<val>.+?)[.!?]*\s*$",
    re.I,
)


@lru_cache(maxsize=_MEMO)
def _slot(text: str):
    m = _SLOT_RE.match(text.strip())
    if not m:
        return None
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return norm(m.group("key")), norm(m.group("val"))


# Two memories in this band are plausibly about the same thing — close
# enough to be a real update, not so close they're the restated-duplicate
# case `memory/service.py`'s own DUPLICATE_THRESHOLD already catches (that
# check runs first; anything reaching this module already scored below it).
# Below this floor, shared words are treated as coincidence: "I use Postgres"
# and "I use a MacBook" share "I use" and nothing else worth acting on.
#
# Measured against real sentence pairs, not guessed: "I use Postgres for the
# main database" vs "I use Redis for the job queue" — two facts about
# different tools that must both survive — scores 0.4 (shared filler: "I",
# "use", "for", "the"). "I prefer Rust for this project" vs "I switched to
# Rust for this project" — the actual update case this module exists for —
# scores 0.625. A floor of 0.35 sat inside the first pair's score and would
# have superseded a real, independent fact off of shared stopwords alone;
# 0.5 sits cleanly between the two.
CONFLICT_FLOOR = 0.5


# Off only in tests, to prove the fast reject in resolve() changes no outcome.
_PRESCREEN = True


# What a retirement that rests on a shared word still needs from meaning. The
# word may be a coincidence ("flat" in a flat and in a phone battery that is
# flat); the meaning is not. Without a shared word, similarity alone has to
# clear the caller's own, higher floor (`semantic_threshold`).
ANCHORED_FLOOR = 0.25

# Two memories this close to equally likely to be the one replaced is a coin
# flip, and the wrong flip deletes a true fact from the prompt. A stale memory
# left standing costs far less, so a near tie retires neither.
MARGIN = 0.08


def _inside(a: str, b: str) -> bool:
    """One word ends the other: "statin" in "rosuvastatin", "phone" in "smartphone"."""
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 5 and short != long_ and long_.endswith(short)


def _shared_anchors(new_text: str, old: str) -> frozenset:
    names = frozenset(_root(w) for w in _name_words(new_text, old) | _name_words(old, new_text))
    kin = frozenset(_root(m.group(0).lower()) for m in _KIN.finditer(f"{new_text} {old}"))
    mine, theirs = _anchors(_content(new_text)), _anchors(_content(old))
    shared = mine & theirs | {a for a in mine for b in theirs if _inside(a, b)}
    return shared - names - kin - _cue_words(new_text)


# The same word given a different number: "iPhone 12" then "iPhone 15". An upgrade
# is often stated with no word of change at all.
_NUMBERED = re.compile(r"\b([a-z][a-z\-]{2,})\s+(\d[\w.]*)", re.I)


def _renumbered(a: str, b: str) -> bool:
    before = {w.lower(): n.lower() for w, n in _NUMBERED.findall(b) if w.lower() not in _STOP}
    return any(before.get(w.lower(), n.lower()) != n.lower()
               for w, n in _NUMBERED.findall(a) if w.lower() in before)


# "Bought a film camera", then "Bought a road bike": two acquisitions, not one
# slot with two values. Only a word that says the first is GONE makes the second
# a replacement.
_ACQUIRED = re.compile(
    r"^(?:(?:i|we)(?:'ve| have)? )?(?:just |also )?"
    r"(?:bought|got|picked up|started|began|joined|signed up|took up|opened)\b", re.I)
# "Got a new keyboard for my laptop" and "Replaced the tyres on the Civic" are about
# the keyboard and the tyres.
_MAINTAINED = re.compile(
    r"^(?:(?:i|we)(?:'ve| have)? )?(?:just )?(?:replaced|changed|swapped|fixed|repaired|upgraded|installed)\b", re.I)
_FOR_MY = re.compile(r"\b(?:for|on|in|of) (?:my|our|the|his|her|their) ((?:[\w-]+ ){0,2}?[\w-]+)", re.I)
_REPLACES = re.compile(r"\b(?:now|instead|upgraded|replac(?:ed|es|ing)|swapped|new|no longer|any ?more)\b", re.I)


def _parallel(new_text: str, old: str) -> bool:
    return bool(_ACQUIRED.match(new_text.strip()) and _ACQUIRED.match(old.strip())
                and not (_REPLACES.search(_asserted(new_text)) or _REVERSAL.search(_asserted(new_text))))


def _accessory(new_text: str, old: str, old_topic: str | None = None) -> bool:
    """A purchase made FOR something the old memory is about: not a new value of it."""
    t = new_text.strip()
    if not (_ACQUIRED.match(t) or _MAINTAINED.match(t)):
        return False
    for m in _FOR_MY.finditer(new_text):
        if ({_root(w) for w in _content(m.group(1))} & _anchors(_content(old))
                or (old_topic and _topics.topic_of(m.group(1)) == old_topic)):
            return True
    return False


def _by_meaning(new_text, new_topic, candidates, semantic, floor) -> str | None:
    """The one candidate `new_text` most plausibly replaces by meaning, or None."""
    asserted = _asserted(new_text)
    if (_ADDITIVE.search(new_text) or _ADDITION.search(new_text)
            or _topics.is_episode(new_text) or _topics.is_multi_valued(new_topic)):
        return None
    worded = bool(_REVERSAL.search(asserted) or _CHANGE.search(asserted))
    if not (worded or _NUMBERED.search(new_text)):
        return None
    # A purchase made FOR something already stored is about the purchase: nothing
    # else it resembles is what it replaces either.
    if any(_accessory(new_text, c["text"], c.get("topic")) for c in candidates):
        return None
    new_actor = actor(new_text)
    scored: list[tuple[float, bool, str]] = []
    for c in candidates:
        old, old_topic = c["text"], c.get("topic")
        if (is_refinement(new_text, old) or different_scopes(new_text, old)
                or different_subjects(new_text, old) or _topics.is_episode(old)
                or _topics.is_multi_valued(old_topic) or _parallel(new_text, old)
                or (actor(old) != new_actor and not same_person(new_text, old))
                or not (worded or _renumbered(new_text, old))):
            continue
        # The slot lexicon, where it knows both sides, has already decided.
        if new_topic and old_topic:
            continue
        scored.append((semantic(c["id"]), bool(_shared_anchors(new_text, old)), c["id"]))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < (ANCHORED_FLOOR if scored[0][1] else floor):
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < MARGIN:
        return None
    return scored[0][2]


def may_replace(new_text: str, c: dict, *, new_topic: str | None = None, guard_actors: bool = False) -> bool:
    """Whether any judge, a person or a model, may still say `new_text` replaces
    candidate `c`. These are the guards that hold however the judge reads the two
    sentences: other people, other named things, an addition, a one-off event,
    a slot the lexicon already told apart, and a fact that can hurt if forgotten
    (unless the new statement names that very item and stops it)."""
    old, old_topic = c["text"], c.get("topic")
    if (_ADDITIVE.search(new_text) or _ADDITION.search(new_text)
            or _topics.is_episode(new_text) or _topics.is_episode(old)
            or _topics.is_multi_valued(new_topic) or _topics.is_multi_valued(old_topic)
            or (new_topic and old_topic)
            or is_refinement(new_text, old) or different_scopes(new_text, old)
            or different_subjects(new_text, old) or _parallel(new_text, old)
            or _outside_scope(new_text, old) or _outside_scope(old, new_text)
            or _accessory(new_text, old, old_topic)
            or (guard_actors and actor(old) != actor(new_text) and not same_person(new_text, old))):
        return False
    new_key, old_key = _attribute_key(new_text), _attribute_key(old)
    if new_key and old_key and new_key != old_key:
        return False
    return not _topics.is_safety(old) or bool(
        _REVERSAL.search(_asserted(new_text)) and _shared_anchors(new_text, old))


@dataclass
class ConflictResult:
    action: str  # "new" | "supersede"
    superseded_ids: list[str] = field(default_factory=list)
    reason: str = ""


def resolve(
    new_text: str,
    candidates: list[dict],
    *,
    similarity: Callable[[str, str], float],
    duplicate_threshold: float,
    conflict_floor: float = CONFLICT_FLOOR,
    new_topic: str | None = None,
    new_kind: str = "fact",
    semantic: Callable[[str], float] | None = None,
    semantic_threshold: float = 0.30,
    guard_actors: bool = False,
) -> ConflictResult:
    """Compare a new memory's text against existing ones and decide.

    `new_topic` / `new_kind` come from `cognition/topics.py`, and so do the
    optional `topic` / `kind` keys on each candidate. When both sides name a
    slot the decision is made on the slot, not the sentence shape: the same
    single-valued slot with a different statement is an update, a different or
    multi-valued slot never is, and an event never retires anything. When
    either side has no nameable slot, the original overlap rules below still
    apply, so a caller that passes neither behaves exactly as before.

    `semantic` maps a candidate's id to its embedding similarity with
    `new_text` (memory/embeddings.py). It is one more way to find "same slot,
    different value" — for updates that share no words with what they replace.
    It is consulted only when no rule retired anything, and it needs the new
    text to be WORDED as a change ("got", "now", "these days", "switched"):
    similarity alone cannot tell a replacement from an addition — "a cat named
    Miso" after "my dog is called Biscuit" scores like "Got a Pixel 9" after
    "My phone is an iPhone 13", 0.31 each — so similarity says which memory,
    the wording says that something changed. At most one memory is retired this
    way: the most similar, if it clears `semantic_threshold` (or the lower
    ANCHORED_FLOOR when it shares a specific word with the new text) and leads
    the runner-up by MARGIN, behind the guards in `_by_meaning`. Meaning also
    vetoes the weakest word rule, a reversal that shares one word: that word
    has to have meaning behind it. None (the default) leaves every decision as
    it was.

    `guard_actors` applies `actor()` to every rule, not only the semantic one:
    without it "My brother works at Google" retires "I work at Acme" through the
    topic rule, because both fill the `employer` slot.

    `candidates` are dicts with at least `id` and `text` — the caller's own
    `live()` rows, unmodified, so this module never touches storage itself
    (same separation `relevance.py` keeps from its callers: judgement here,
    persistence there). Anything scoring at or above `duplicate_threshold`
    is skipped — that is a restatement, `remember()`'s existing duplicate
    check already short-circuits before this runs, and re-flagging it here
    as a "conflict" would supersede a memory with an identical copy of
    itself.
    """
    new_slot = _slot(new_text)
    new_key = _attribute_key(new_text)
    new_reversal = bool(_REVERSAL.search(_asserted(new_text)))
    additive = bool(_ADDITIVE.search(new_text))
    new_attr = attribute_slot(new_text)
    new_actor = actor(new_text)

    def replaces(c: dict, old: str, score: float) -> bool:
        """Would `new_text` retire this candidate on its own evidence?"""
        # The person the memories are ABOUT is not a shared subject: every
        # memory of "Devan …" names Devan. Counting the name let one
        # "Devan switched to cortados." retire eight unrelated memories —
        # dark mode, VS Code, Python, the afternoon cola — measured on a
        # synthetic life, and invisible to any check that only asked whether
        # stale memories got retired rather than by what.
        shared_item = _shared_anchors(new_text, old)

        # Both sides name a slot: decide on the slot.
        old_topic = c.get("topic")
        if new_topic and old_topic:
            if new_topic != old_topic or additive:
                return False
            if _topics.is_multi_valued(new_topic):
                # A second medication is added, not swapped in. Only a
                # statement that stops or replaces THAT drug retires it.
                return (new_topic in _topics.RETIRABLE_BY_REVERSAL
                        and new_reversal and bool(shared_item))
            return not (score >= duplicate_threshold and not is_distinct(new_text, old))

        # The original band: lexically close without being a restatement.
        if conflict_floor <= score < duplicate_threshold:
            return True

        # Same slot, different value — "favourite editor is vim" then
        # "favourite editor is helix". High precision: both sides have to
        # match one of a short list of "key is value" shapes with the same
        # key, so an unrelated sentence cannot land here.
        old_slot = _slot(old)
        if new_slot and old_slot and new_slot[0] == old_slot[0] \
                and new_slot[1] != old_slot[1]:
            return True

        # The new statement fills a named slot and the old one fills none, so
        # the old one is not that slot's previous value; what they share is
        # incidental.
        if new_topic and not old_topic:
            return False

        # A reversal in plain language ("switched to", "no longer", "quit")
        # that still names something specific the old memory named. The shared
        # word is what keeps "I quit my job" from sweeping away "I drink
        # coffee" — and it has to be a word that identifies the thing: "this
        # project" is on half the store.
        return (new_reversal and score > 0 and bool(shared_item)
                and (semantic is None or semantic(c["id"]) >= ANCHORED_FLOOR))

    superseded: list[str] = []
    for c in candidates:
        old = c["text"]

        # Fast reject. Each way a candidate can be superseded below needs one
        # of: the same attribute slot, the same topic, a similarity at or above
        # the floor, a reversal sharing any word, or the same "key is value"
        # key. Every check after this one only ever SKIPS a candidate, so
        # dropping one that meets none of them here changes no outcome — it
        # just avoids running the whole sequence against every stored memory
        # (most of them unrelated) on every write.
        if _PRESCREEN and not ((new_attr and new_attr == attribute_slot(old))
                               or (new_topic and new_topic == c.get("topic"))):
            score = similarity(new_text, old)
            if score < conflict_floor and not (new_reversal and score > 0):
                old_slot = _slot(old)
                if not (new_slot and old_slot and new_slot[0] == old_slot[0]):
                    continue

        # A refinement ("Postgres" -> "Postgres 16") is new information: it is
        # neither the same fact nor a competing one, so it must not supersede.
        if is_refinement(new_text, old):
            continue

        # Different named subjects, or different named scopes, are different
        # facts (see `_about`, `_scopes`).
        if different_subjects(new_text, old) or different_scopes(new_text, old):
            continue
        # "the alert routing for Indigo is Postgres" is about Indigo: only a
        # statement that mentions Indigo can be the other value of its slot.
        if (_outside_scope(new_text, old) or _outside_scope(old, new_text)
                or _accessory(new_text, old, c.get("topic"))):
            continue
        if guard_actors and new_actor != actor(old) and not same_person(new_text, old):
            continue

        # An event was true when it happened and stays true: it neither
        # retires nor is retired by anything.
        if new_kind == "event" or c.get("kind") == "event":
            continue

        score = similarity(new_text, old)

        # An attribute of a named thing on both sides: the pair is the slot.
        new_attr, old_attr = attribute_slot(new_text), attribute_slot(old)
        if new_attr and old_attr:
            if new_attr == old_attr and score < duplicate_threshold and not additive:
                superseded.append(c["id"])
            continue

        # "favorite language is Rust" and "favorite number is 17" are two
        # attributes of the same person. Decided before any overlap rule,
        # because the overlap is the template, not the fact.
        # Skipped when the topics already agree: "my manager is …" and "my
        # boss is …" are one slot under two words, and the topic knows that.
        old_key = _attribute_key(old)
        same_topic = bool(new_topic) and new_topic == c.get("topic")
        if new_key and old_key and new_key != old_key and not same_topic:
            continue

        if not replaces(c, old, score):
            continue

        # A fact that can hurt if forgotten is retired only by a statement that
        # stops or replaces the very thing it names — never on word overlap, a
        # shared slot or a lookalike sentence. Staying on a stale allergy costs
        # a caveat; losing a live one costs the user.
        if _topics.is_safety(old) and not (
                new_reversal and _shared_anchors(new_text, old)):
            continue

        superseded.append(c["id"])

    if semantic is not None and not superseded:
        found = _by_meaning(new_text, new_topic, candidates, semantic, semantic_threshold)
        if found is not None:
            superseded.append(found)

    if superseded:
        return ConflictResult(
            "supersede", superseded,
            f"updates {len(superseded)} existing "
            f"memor{'y' if len(superseded) == 1 else 'ies'} about the same thing "
            f"(reworded, a reversal, or the same fact with a different value)")
    return ConflictResult("new", [], "no existing memory is similar enough to conflict with")


def overrides(
    new_texts: list[str],
    candidates: list[dict],
    *,
    similarity: Callable[[str, str], float],
    duplicate_threshold: float,
    conflict_floor: float = CONFLICT_FLOOR,
) -> set[str]:
    """Which of `candidates` at least one of `new_texts` plausibly updates.

    The many-callers form of `resolve()` — for a caller checking SEVERAL new
    statements against the same pool at once (a session's working facts,
    typically) and wanting the union of everything any of them would
    supersede, rather than resolving them one at a time and stitching the
    results together itself. Each text is still judged independently through
    the same `resolve()`; this only collects ids across more than one call,
    it does not change what counts as a conflict.
    """
    overridden: set[str] = set()
    for text in new_texts:
        result = resolve(text, candidates, similarity=similarity,
                         duplicate_threshold=duplicate_threshold, conflict_floor=conflict_floor)
        overridden.update(result.superseded_ids)
    return overridden
