"""Split a chat message into the statements about the user it contains.

No language model: sentences are split on end punctuation, line breaks, semicolons
and "but"/"also" joints, then a sentence is kept when it states something (it
speaks about the user, or is a plain statement of fact) and is neither a question
nor a request to the assistant. A long chat turn ("...can you suggest a stretching
routine? Oh and I finally switched to the night shift last month.") becomes the
one or two statements worth comparing.

Version 2 (recall fixes). Version 1 dropped true facts in five ways (checked by hand on
example messages; the proxy score in results/extract_v1.json / extract_v2.json barely moves):
  * a sentence without I/my/we was dropped ("Moved to Pune last month",
    "Got a dog! His name is Max.");
  * a leading "sure / ok / no / great / thanks" made the whole sentence look like a
    request ("Sure, I'm vegetarian now.");
  * anything under 15 characters was dropped ("I'm vegan.");
  * one joined sentence stayed one blob ("... since 2019 but quit in May");
  * a question sentence took the fact in front of it down with it
    ("I'm moving to Pune, any good schools?").
"""
from __future__ import annotations

import re

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|;\s+|,?\s+but\s+(?=\S)|,?\s+and also\s+|,\s+also\s+|\s+by the way,?\s+", re.I)
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|i'd|i'll|my|me|mine|myself|we|we're|we've|our|us)\b", re.I)
# opening filler is peeled off before the request test: "Sure, I'm vegetarian now."
FILLER = re.compile(r"^(sure|ok|okay|yeah|yes|yep|yup|no|nope|great|thanks|thank you|thx|so|well|oh|anyway|anyways|"
                    r"btw|by the way|also|and|but|hi|hello|hey|right|cool|awesome|nice|got it|"
                    r"that'?s (really |very |so )?(helpful|great|awesome|good|perfect))\b[\s,.!:-]*", re.I)
REQUEST = re.compile(r"^(can|could|would|will|do|does|did|is|are|what|how|which|where|when|why|who|any|"
                     r"please|i was wondering|i wonder|i'd like (you|some)|i'd love (some|help|your)|"
                     r"i need (some )?(help|advice|tips|ideas)|i want (some )?(help|advice|tips|ideas)|"
                     r"let me know|tell me|give me|help me|suggest|recommend)\b", re.I)
# cues that a sentence without I/my/we still states a fact (a person, a thing, a change)
FACT_CUE = re.compile(r"\b(is|are|was|were|has|have|had|got|get|gets|became|become|started|moved|quit|left|joined|"
                      r"bought|sold|named|called|switched|changed|finished|began|born|married|engaged|divorced|"
                      r"works?|lives?|loves?|likes?|hates?|plays?|studies|studying|working|living|"
                      r"\w+ed|\w+en)\b|\d", re.I)
MIN_CHARS = 8
MIN_CHARS_NO_PRONOUN = 10


def _strip_filler(p: str) -> str:
    prev = None
    while prev != p:
        prev = p
        p = FILLER.sub("", p, count=1).strip()
    return p


def _declarative_head(p: str) -> str:
    """For a sentence that ends in "?": the statement in front of the question
    ("I'm moving to Pune, any good schools?" -> "I'm moving to Pune"), else ""."""
    if "," not in p:
        return ""
    head = p.rsplit(",", 1)[0].strip()
    if REQUEST.match(_strip_filler(head)) or head.endswith("?"):
        return ""
    return head


def statements(text: str, keep_all_if_none: bool = True) -> list[str]:
    parts = [p.strip(" -–•*\t") for p in SPLIT.split(text) if p and p.strip()]
    kept = []
    for p in parts:
        p = _strip_filler(p)
        if p.endswith("?"):
            p = _declarative_head(p)
        if len(p) < MIN_CHARS or REQUEST.match(p):
            continue
        if FIRST_PERSON.search(p):
            kept.append(p)
        elif len(p) >= MIN_CHARS_NO_PRONOUN and FACT_CUE.search(p):
            kept.append(p)
    if not kept and keep_all_if_none:
        # nothing survived: fall back to the whole text rather than nothing
        return [text.strip()]
    return kept
