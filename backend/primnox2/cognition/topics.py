"""What a memory is ABOUT, and whether it is a fact or an event.

Two judgements `cognition/conflict.py` could not make from sentence shape
alone, and the reason it retired memories that were never stale.

TOPIC  The slot a statement fills: `editor`, `coffee`, `residence`. Two
       statements about the same slot with different values are an update;
       two about different slots are not, however many words they share.

KIND   `fact` (a standing truth: "I use Helix") or `event` (something that
       happened: "Met Nadia about Atlas at the annex (week 1)"). An event
       cannot retire another event or a fact — it was true when it happened
       and stays true. Measured on a 100-memory synthetic life, the overlap
       rule alone retired dozens of meeting lines against each other. A
       statement of what is true NOW is a fact however it is dated: "Got a
       Pixel 9 last week" and "Started at Deloitte on Monday" are changes, not
       one-offs, and have to be able to retire what they replace.

CARDINALITY  Some slots hold one value (where you live) and some hold many
       (the projects you work on). A new project does not replace an old one,
       so multi-valued topics never supersede on overlap.

Deterministic on purpose. `memory/service.py` refuses to put a model in the
path of "remember this", and this runs on every write. A topic that cannot be
named is `None`, never a guess — `conflict.py` then falls back to the original
overlap behaviour for that pair rather than inventing a slot.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Ordered: the first topic whose pattern matches wins, so a narrower noun
# ("job queue") must come before a product that could mean either ("redis").
# Patterns are matched on the lowercased text, as whole words or phrases.
_TOPICS: list[tuple[str, re.Pattern]] = [(name, re.compile(pat, re.I)) for name, pat in (
    ("name",      r"\bmy name is\b|\bcall me\b|\bi(?:'m| am) called\b"),
    ("role",      r"\bis an? [\w \-]+ at\b|\bworks as an? \b|\bmy (?:job|role|title) is\b"),
    ("manager",   r"\bmanager\b|\b(?:my|their) (?:boss|lead)\b"),
    ("allergy",   r"\ballerg(?:y|ic|ies)\b|\bintoleran"),
    ("medication", r"\bmedications?\b|\bmedicines?\b|\bmeds?\b|\bprescri\w+|"
                   r"\b\d+\s?(?:mg|mcg|ml|iu)\b|"
                   r"\b\w*(?:statin|sartan|olol|dipine|cillin|mycin|prazole)s?\b|\b(?!april)\w+pril\b|"
                   r"\b(?:tablets?|pills?|capsules?|inhalers?|insulin|injections?|doses?|dosage|"
                   r"vitamins?|supplements?|antibiotics?|antidepressants?|statins?|metformin|"
                   r"aspirin|ibuprofen|paracetamol|acetaminophen|lisinopril|amlodipine|warfarin)\b|"
                   r"\btak(?:e|es|ing)\s+(?!(?:the|a|an|my)\s+(?:train|tram|bus|metro|subway|ferry|"
                   r"taxi|cab|bike|lift|stairs|flight)\b)\w+(?:\s+\w+){0,2}\s+"
                   r"(?:daily|twice(?:\s+a\s+day)?|once\s+a\s+day|every\s+(?:morning|day|night|evening)|"
                   r"before\s+(?:meals|bed)|after\s+(?:meals|food)|at\s+(?:night|bedtime)|with\s+food)\b"),
    ("queue",     r"\b(?:job |task |message )?queue\b"),
    ("database",  r"\bdatabase\b|\bpostgres(?:ql)?\b|\bmysql\b|\bsqlite\b|\bmongo(?:db)?\b|\bmariadb\b"),
    ("editor",    r"\beditors?\b|\bide\b|\bvs ?code\b|\bvscode\b|\bneovim\b|\bvim\b|\bhelix\b|"
                  r"\bemacs\b|\bsublime\b|\bintellij\b|\bpycharm\b|\bzed\b|\bcursor\b"),
    ("theme",     r"\bdark mode\b|\blight mode\b|\bdark theme\b|\blight theme\b|\btheme\b|\bcolou?r scheme\b"),
    ("notes",     r"\bnotes?\b|\bnotion\b|\bobsidian\b|\bevernote\b|\blogseq\b|\bmarkdown\b"),
    ("standup",   r"\bstand-?ups?\b"),
    ("soda",      r"\bsoda\b|\bcola\b|\bfizzy\b|\bsoft drinks?\b"),
    ("coffee",    r"\bcoffee\b|\bespresso\b|\blatte\b|\bflat whites?\b|\bcappuccinos?\b|\bcortados?\b|"
                  r"\bamericanos?\b|\bmochas?\b|\bmacchiatos?\b|\bcold brew\b|\btea\b"),
    ("language",  r"\b(?:python|rust|golang|typescript|javascript|kotlin|swift|ruby|php|java|scala|"
                  r"haskell|elixir|clojure|c\+\+|c#)\b|\bprogramming language\b|\bcode in\b|\bwrites? \w+ daily\b"),
    ("residence", r"\b(?:i|we) (?:live|moved|relocated)\b.{0,12}\b(?:in|to)\b|^(?:moved|relocated) (?:back )?(?:to|into)\b|"
                  r"\b(?:he|she|they|(?:my|our) (?:sister|brother|mother|mum|mom|father|dad|son|daughter|"
                  r"wife|husband|partner|parents|friend|cousin|aunt|uncle)(?: \w+)?) (?:just )?(?:moved|relocated) (?:back )?to"
                  r"\b|"
                  r"\blives in\b|\bbased in\b|\bmy (?:city|home ?town)\b|"
                  r"\b(?:our|my) (?:house|home|flat|apartment)\b(?! office)|"
                  r"\b(?:rent|rented|renting|own|owns|bought|buy|sold)(?: [\w-]+){0,3} (?:house|flat(?! white)|apartment|condo)\b"),
    ("commute",   r"\bcommut(?:e|es|ing)\b|\b(?:metro|tram|subway|train|bus|bike|cycles?|cycling|drives?|walks?)\b"
                  r"(?:.{0,20})\b(?:to|into) (?:the |my )?(?:office|work|school|campus|uni|university|college)\b|"
                  r"\bto the office\b|\bget(?:s|ting)? to (?:the |my )?(?:office|work|school|campus|uni|university|college)\b|"
                  r"\b(?:takes?|taking|rides?|riding) (?:the |a )?(?:metro|tram|subway|train|bus|bike)\b"),
    ("employer",  r"\bworks? (?:at|for)\b|\bmy (?:company|employer) is\b"),
    ("pet",       r"\b(?:dogs?|cats?|puppy|puppies|kitten|pets?|hamster|rabbit|parrot)\b"),
    ("interest",  r"\b(?:loves?|enjoys?)\b|\b(?:i|we|they|he|she) likes?\b|\bhobb(?:y|ies)\b|\binterested in\b"),
)]

# Topics that legitimately hold several values at once. A new one is added,
# never swapped in: working on project Granite does not end working on
# project Cinder, and being allergic to peanuts does not retire the shellfish.
MULTI_VALUED = frozenset({"project", "allergy", "queue", "medication", "pet", "interest"})

# Of those, the ones a statement may still retire — but only one that stops or
# replaces the named thing ("I'm no longer on metformin"), never one that merely
# overlaps ("I take aspirin every morning"). An allergy is not here: no
# sentence the engine can read is good enough evidence to forget one.
RETIRABLE_BY_REVERSAL = frozenset({"medication"})

_PROJECT = re.compile(r"\bworks? on (?:the )?project\b|\bworking on (?:the )?project\b|"
                      r"\b(?:part of|member of|on) (?:the )?project\b", re.I)

# An event is only ever a POSITIVE call, because a fact filed as an event
# escapes conflict handling entirely, while an event filed as a fact merely
# behaves as it always did. Three kinds of evidence, strongest first:
#   an explicit log marker ("(week 3)", an ISO date);
#   an opener that is an episode, never a change of state ("Met Nadia", "Went
#   to a gig");
#   a time phrase ("yesterday", "last week", "on Friday") on a statement that
#   does NOT say something changed. "Got a Pixel 9 last week" carries the same
#   time phrase and is the opposite of an event: it says what is true now.
_LOG_MARK = re.compile(r"\((?:week|month|day)\s*\d+\)|\b\d{4}-\d{2}-\d{2}\b", re.I)
_EPISODE = re.compile(
    r"^(?:Met|Reviewed|Had (?:lunch|coffee|dinner|breakfast|a call|a meeting)|Was unblocked|"
    r"Attended|Presented|Spoke|Discussed|Called|Emailed|Shipped|Merged|Demoed|Paired|"
    r"Finished|Completed|Read|Visited|Watched|Booked|Launched|Published|Deployed|Fixed|"
    r"Submitted|Won|Hosted|Gave(?! up)|Wrote|Went (?:to|on|for)|Saw|Ate|Hiked|Climbed|Cooked|Back from)\b"
    r"|^(?:I|We|He|She|They|[A-Z][a-z]+) (?:met|won(?!['’]t)|went (?:to|on|for)|saw|ate|visited|watched|attended|"
    r"presented|spoke|finished|completed|hosted|hiked|climbed|cooked|reviewed|submitted|launched|"
    r"published|shipped|merged|deployed)\b")
_WHEN = re.compile(
    r"\b(?:yesterday|last (?:week|weekend|month|night)|this morning|the other day|earlier today)\b"
    r"|\b(?:on|last|this) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"(?!\s+(?:morning|afternoon|evening|night))", re.I)

# Wording that says something CHANGED: generic English for a transition, not
# slot words. "adopted" is left out on purpose: a second pet is an addition.
TRANSITION = re.compile(
    r"\b(?:new|now|nowadays|these days|lately|recently|any ?more|no longer|no more|instead|again|"
    r"switch(?:ed|ing)|chang(?:ed|ing)|moved(?! (?:the|my|our|a|an|his|her|their) (?:[\w'-]+ ){0,2}?"
    r"(?:into|onto|across|around|inside))|migrat(?:ed|ing)|replac(?:ed|es|ing)|swapped|"
    r"upgraded|got|bought|picked up|started|began|took up|taken up|took over|taken over|"
    r"takes over|gone over|went over|went back to|ditched|opened|signed up|dropped|quit|"
    r"stopped|gave up|abandoned|used to|sold|deleted|cancell?ed|joined|relocated|promoted|"
    r"resigned|transferred|done with|(?:i'm|i am|came|come|coming|gone|went|getting) off "
    r"(?!(?:to|work|today|tomorrow|duty|sick|shift)\b))\b", re.I)


# "Bought a film camera for the home office", then "Bought a road bike for the
# home office": parallel purchases that the word rules must not take for one slot
# with two values. It is also how an upgrade is worded ("Bought a Pixel 9"), which
# is why `is_episode`, the test meaning-based supersession uses, leaves it out.
_PURCHASE = re.compile(r"^Bought\b")


@lru_cache(maxsize=65_536)
def kind_of(text: str) -> str:
    """`event` when the text carries a positive marker of one, else `fact`."""
    t = text.strip()
    if _LOG_MARK.search(t) or _EPISODE.match(t) or _PURCHASE.match(t):
        return "event"
    if _WHEN.search(t) and not TRANSITION.search(t):
        return "event"
    return "fact"


def is_episode(text: str) -> bool:
    """A one-off that says nothing about what is true now — kind_of's events, less
    a bare purchase."""
    return kind_of(text) == "event" and not _PURCHASE.match(text.strip())


def topic_of(text: str) -> str | None:
    """The slot this statement fills, or None when it cannot be named."""
    t = text.strip()
    if not t:
        return None
    if _PROJECT.search(t):
        return "project"
    for name, pattern in _TOPICS:
        if pattern.search(t):
            return name
    return None


# A QUESTION names its slot differently from a statement: "what language do I
# use" says "language", while the memory says "writes Rust daily". These words
# point at a slot only when asked about it, so they live here and never reach
# `topic_of`, where "live" or "home" would misfile ordinary statements.
_QUERY_ALIASES: dict[str, str] = {
    "language": "language", "languages": "language", "programming": "language",
    "editor": "editor", "ide": "editor",
    "theme": "theme", "mode": "theme", "appearance": "theme",
    "commute": "commute", "commuting": "commute", "travel": "commute",
    "coffee": "coffee", "tea": "coffee",
    "soda": "soda", "cola": "soda",
    "notes": "notes", "note": "notes",
    "live": "residence", "lives": "residence", "city": "residence",
    "home": "residence", "address": "residence", "location": "residence",
    "manager": "manager", "boss": "manager",
    "allergy": "allergy", "allergies": "allergy", "allergic": "allergy",
    "medication": "medication", "medications": "medication", "meds": "medication",
    "medicine": "medication", "pills": "medication",
    "database": "database", "db": "database",
    "queue": "queue", "standup": "standup", "standups": "standup",
    "name": "name", "role": "role", "job": "role", "title": "role",
    "employer": "employer", "company": "employer", "project": "project",
    "projects": "project",
}
_QWORD = re.compile(r"[a-z]+")


def topic_of_query(query: str) -> str | None:
    """The slot a question asks about: a statement-style match first, then a
    word that names the slot outright."""
    found = topic_of(query)
    if found:
        return found
    for word in _QWORD.findall(query.lower()):
        if word in _QUERY_ALIASES:
            return _QUERY_ALIASES[word]
    return None


def is_multi_valued(topic: str | None) -> bool:
    return topic in MULTI_VALUED


def classify(text: str) -> tuple[str | None, str]:
    """`(topic, kind)`. An event has no slot of its own to compete for."""
    kind = kind_of(text)
    return (None if kind == "event" else topic_of(text)), kind


# Facts that carry a safety consequence if the model does not have them: a
# named condition, an allergy, a medication, or a prohibition on one. Kept here
# because three callers need the same answer -- render_for_prompt puts these
# first, the conflict engine refuses to retire one on overlap alone, and the
# directive check lets one through an unfamiliar verb.
_CONDITION_RE = re.compile(
    r"\b(?:al?lerg\w*|anaphyla\w*|epi-?pen|intoleran\w*|coeliac|celiac|"
    r"gluten[- ]free|diabet\w*|insulin|asthma\w*|seizure|epilep\w*|"
    r"medication|medicine|prescri\w+|contraindicat\w*|life-threatening|"
    r"dietary\s+restriction)\b",
    re.I,
)
_PROHIBITION_RE = re.compile(
    r"\b(?:can ?not|cannot|must not|can't|do not|don't|never)\s+"
    r"(?:take|use|eat|drink|have|touch|be\s+given|be\s+prescribed)\b",
    re.I,
)


def names_condition(text: str) -> bool:
    """A health condition, allergy or medication is named outright."""
    return bool(_CONDITION_RE.search(text))


def is_safety(text: str) -> bool:
    return bool(_CONDITION_RE.search(text) or _PROHIBITION_RE.search(text)
                or topic_of(text) == "medication")
