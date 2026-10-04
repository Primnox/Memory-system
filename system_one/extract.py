"""Split a chat message into the statements about the user it contains.

No language model: sentences are split on end punctuation and line breaks,
then a sentence is kept when it speaks about the user (I / my / we / our ...)
and is neither a question nor a request to the assistant. A long chat turn
("...can you suggest a stretching routine? Oh and I finally switched to the
night shift last month.") becomes the one or two statements worth comparing.
"""
from __future__ import annotations

import re

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|ive|i'd|i'll|my|me|mine|myself|we|we're|we've|our|us)\b", re.I)
REQUEST = re.compile(r"^(can|could|would|will|do|does|did|is|are|what|how|which|where|when|why|who|any|"
                     r"please|thanks|thank|that's|that is|great|ok|okay|sure|yes|no|hi|hello|hey|also,? can|"
                     r"also,? could|i was wondering|i wonder|i'd like (you|some)|i'd love (some|help|your)|"
                     r"i need (some )?(help|advice|tips|ideas)|i want (some )?(help|advice|tips|ideas)|"
                     r"let me know|tell me|give me|help me|suggest|recommend)\b", re.I)
MIN_CHARS = 15


def statements(text: str, keep_all_if_none: bool = True) -> list[str]:
    parts = [p.strip(" -–•*\t") for p in SPLIT.split(text) if p and p.strip()]
    kept = []
    for p in parts:
        if len(p) < MIN_CHARS or p.endswith("?") or REQUEST.match(p):
            continue
        if FIRST_PERSON.search(p):
            kept.append(p)
    if not kept and keep_all_if_none:
        # nothing first-person survived: fall back to the whole text rather than nothing
        return [text.strip()]
    return kept
