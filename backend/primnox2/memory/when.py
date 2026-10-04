"""The user's own time words ("back in March", "last summer", "two months ago")
turned into the span of days they point at, in code.

The local 9B was asked to do this itself, writing `as_of` as a date, and did it
badly: "back in October" came out as the October still ahead, a year off. It is
good at copying the user's words and bad at calendar arithmetic, so `recall_memory`
takes the words (`when`) and this module does the arithmetic. On the dev split's
past questions that took the model's top-1 from 8/15 to 11/15, the same as handing
it the true dates.

Rules, all relative to "today" (a date the caller supplies; never the wall clock,
so a benchmark can replay a scenario's own day):

  * A month or season by name ("in March", "last March", "back in the summer") is
    the most recent one that has ended, so one still running is last year's: a
    person in March who means this March says "this month". "this March" is the
    one in the current year, or None if it is still ahead. "last December" asked
    in January is last year's.
  * Seasons are northern-hemisphere (spring Mar-May ... winter Dec-Feb). The
    device's UTC offset cannot tell Sao Paulo from Nuuk, so there is no guess; a
    southern user should say the month. Winter 2025 is Dec 2024-Feb 2025.
  * early / mid / late (or "the end of") narrow a span to its first, middle or
    last third: "toward the end of July" is 21-31 July.
  * "N days/weeks ago" is that day +-1 / +-3 days; "N months ago" is that date
    +-15 days; "N years ago" +-91 days. "a few" is 3, "a couple of" 2.
  * A span that reaches today is NOW: `as_of_for` returns None for it, so "this
    month", "these days" or a model that dated a present-tense question all search
    as things stand.

Anything unrecognised resolves to None and the caller falls back to whatever the
model wrote as `as_of`.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

# Where in the span to ask "as things stood then". The middle: a person who says
# "back in March" means how things were through March, not on the 31st, and it is
# the less biased point when something changed inside the span. Measured on the dev
# split it ties with the end (33/45 top-1 each, no dev past question changes inside
# its span, so dev cannot separate them); the dev set's own dates for "back in X" are
# all the middle of X.
PICK = "mid"

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
                "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?"
_SEASONS = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}
_SEASON = "(?:" + "|".join(_SEASONS) + ")"
_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_NUMBERS.update({"a": 1, "an": 1, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3, "several": 4})
_COUNT = "(?:" + "|".join(sorted(_NUMBERS, key=len, reverse=True)) + r"|\d+)"
_YEAR = r"(?:19|20)\d\d"
# What lets a bare month or year count as a time: "in March", "early March", "the end of March".
_LICENCE = (r"(?:\b(?:in|during|around|as of|throughout)\s+(?:the\s+)?|"
            r"\b(?:early|mid|late)[ -]*|\b(?:the\s+)?(?:end|start|beginning|middle) of\s+(?:the\s+)?)")
_THIRD = re.compile(r"(?:\b(?:toward|towards|near|at|around)\s+)?(?:\bthe\s+)?"
                    r"\b(early|mid|middle of|late|end of|start of|beginning of)\s*-?\s*(?:in\s+)?(?:the\s+)?$")
_THIRDS = {"early": 0, "start of": 0, "beginning of": 0, "mid": 1, "middle of": 1, "late": 2, "end of": 2}


@dataclass(frozen=True)
class Span:
    start: date
    end: date

    def point(self, today: date, pick: str | None = None) -> date | None:
        """The day to search as of, or None when the span reaches today."""
        if self.end >= today:
            return None
        if (pick or PICK) == "end":
            return self.end
        if self.start.day == 1 and self.start.month == self.end.month and (self.end + timedelta(days=1)).day == 1:
            return self.start.replace(day=15)   # the middle of a calendar month, by convention
        return self.start + timedelta(days=(self.end - self.start).days // 2)

    def third(self, which: int) -> "Span":
        n = (self.end - self.start).days + 1
        return Span(self.start + timedelta(days=n * which // 3),
                    self.start + timedelta(days=n * (which + 1) // 3 - 1))


def _day(y: int, m: int, d: int) -> Span | None:
    try:
        return Span(date(y, m, d), date(y, m, d))
    except ValueError:
        return None


def _month(y: int, m: int) -> Span:
    return Span(date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]))


def _months_from(y: int, m: int, n: int) -> Span:
    y2, m2 = divmod(y * 12 + m - 1 + n - 1, 12)
    return Span(date(y, m, 1), _month(y2, m2 + 1).end)


def _year(y: int) -> Span:
    return Span(date(y, 1, 1), date(y, 12, 31))


def _season(y: int, name: str) -> Span:
    return _months_from(y, _SEASONS[name], 3)


def _latest(make, today: date) -> Span | None:
    """The most recent instance that has ended: "in March" asked during March is
    last March, since a person still in the month says "this month"."""
    for y in range(today.year, today.year - 3, -1):
        span = make(y)
        if span and span.end < today:
            return span
    return None


def _this(make, today: date) -> Span | None:
    """The instance in the current year; the one still running if it began last year."""
    for y in (today.year, today.year - 1):
        span = make(y)
        if span and span.start <= today and (span.start.year == today.year or span.end >= today):
            return span
    return None


def _holiday(new_year: bool):
    if new_year:
        return lambda y: Span(date(y, 12, 31), date(y + 1, 1, 2))
    return lambda y: Span(date(y, 12, 24), date(y, 12, 26))


def _ago(n: int, unit: str, today: date) -> Span:
    if unit == "day":
        anchor, tol = today - timedelta(days=n), 1
    elif unit == "week":
        anchor, tol = today - timedelta(weeks=n), 3
    else:
        months = n * (12 if unit == "year" else 1)
        y, m = divmod(today.year * 12 + today.month - 1 - months, 12)
        last = calendar.monthrange(y, m + 1)[1]
        anchor, tol = date(y, m + 1, min(today.day, last)), (91 if unit == "year" else 15)
    return Span(anchor - timedelta(days=tol), anchor + timedelta(days=tol))


def _num(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBERS[token]


def _rel(word: str, make, today: date) -> Span | None:
    return _latest(make, today) if word == "last" else _this(make, today)


def _period(text: str, today: date, bare: bool):
    """(span, cut) for the first kind of time phrase found, else None. `cut` is
    where the phrase proper starts, so an early/mid/late before it can be read.
    `bare` lets a lone month or year count ("November"): safe in the `when`
    argument, where the whole text is a time, not in a question. A lone season
    never does: "in the summer" is as often a habit as a past."""
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return _day(*map(int, m.groups())), m.start()
    m = re.search(r"\b(\d{4})-(\d{2})\b(?!-)", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return _month(int(m.group(1)), int(m.group(2))), m.start()
    for day_first, pattern in ((False, rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\d)(?:,?\s+({_YEAR}))?"),
                               (True, rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})(?:,?\s+({_YEAR}))?")):
        m = re.search(pattern, text)
        if m:
            name, d, y = (m.group(2), m.group(1), m.group(3)) if day_first else m.groups()
            mo, d = _MONTHS[name.rstrip(".")], int(d)
            return (_day(int(y), mo, d) if y else _latest(lambda y: _day(y, mo, d), today)), m.start()
    m = re.search(rf"\b({_MONTH}),?\s+({_YEAR})\b", text)
    if m:
        return _month(int(m.group(2)), _MONTHS[m.group(1).rstrip(".")]), m.start()
    m = re.search(rf"\b({_SEASON})\s+(?:of\s+)?({_YEAR})\b", text)
    if m:
        return _season(int(m.group(2)) - (m.group(1) == "winter"), m.group(1)), m.start()
    m = re.search(rf"\b(?:(last|this)\s+)?(christmas|xmas|new year)(?:'?s)?(?:\s+(?:eve|day))?(?:\s+({_YEAR}))?", text)
    if m:
        new_year = m.group(2) == "new year"
        make = _holiday(new_year)
        if m.group(3):
            return make(int(m.group(3)) - new_year), m.start()
        return (_rel(m.group(1), make, today) if m.group(1) else _latest(make, today)), m.start()
    m = re.search(rf"\b({_COUNT})\s+(day|week|month|year)s?\s+ago\b", text)
    if m:
        return _ago(_num(m.group(1)), m.group(2), today), m.start()
    m = re.search(r"\bearlier this (year|month)\b", text)
    if m:
        first = date(today.year, 1 if m.group(1) == "year" else today.month, 1)
        end = _month(today.year, today.month).start - timedelta(days=1)
        if m.group(1) == "month" or end < first:
            end = today - timedelta(days=1)
        return (Span(first, end) if end >= first else None), m.start()
    m = re.search(r"\b(last|this)\s+(week|month|year)\b", text)
    if m:
        word, unit = m.groups()
        if unit == "year":
            return _year(today.year - (word == "last")), m.start()
        if unit == "month":
            y, mo = divmod(today.year * 12 + today.month - 1 - (word == "last"), 12)
            return _month(y, mo + 1), m.start()
        monday = today - timedelta(days=today.weekday() + (7 if word == "last" else 0))
        return Span(monday, monday + timedelta(days=6)), m.start()
    m = re.search(rf"\b(last|this)\s+({_SEASON})\b", text)
    if m:
        return _rel(m.group(1), lambda y: _season(y, m.group(2)), today), m.start()
    m = re.search(rf"\b(last|this)\s+({_MONTH})", text)
    if m:
        mo = _MONTHS[m.group(2).rstrip(".")]
        return _rel(m.group(1), lambda y: _month(y, mo), today), m.start()
    m = re.search(r"\byesterday\b", text)
    if m:
        return Span(today - timedelta(days=1), today - timedelta(days=1)), m.start()
    m = re.search(rf"\bback in\s+(?:the\s+)?({_SEASON})\b", text)
    if m:
        return _latest(lambda y: _season(y, m.group(1)), today), m.start()
    licence = f"(?:{_LICENCE})?" if bare else _LICENCE
    m = re.search(rf"{licence}({_MONTH})", text)
    if m:
        mo = _MONTHS[m.group(1).rstrip(".")]
        return _latest(lambda y: _month(y, mo), today), m.start(1)
    m = re.search(rf"{licence}({_YEAR})\b", text)
    if m:
        return _year(int(m.group(1))), m.start(1)
    m = re.search(r"\b(?:right now|these days|nowadays|currently|at the moment|at present|today|now)\b", text)
    if m:
        return Span(today, today), m.start()
    return None


def resolve(text: str | None, today: date, *, bare: bool = False) -> Span | None:
    """The span a time phrase inside `text` points at, or None. Works on a phrase
    ("last summer") and on a whole question containing one."""
    text = re.sub(r"\s+", " ", text.lower()).strip() if isinstance(text, str) else ""
    try:
        found = _period(text, today, bare) if text else None
    except (ValueError, OverflowError):   # "99999 years ago", "0000-01"
        return None
    if not found or found[0] is None:
        return None
    span, cut = found
    third = _THIRD.search(text[:cut])
    if third and (span.end - span.start).days >= 6:
        span = span.third(_THIRDS[third.group(1)])
    return span


def as_of_for(*, when: str | None = None, as_of=None, query: str = "", today: date,
              pick: str | None = None):
    """The date `recall_memory` should search as of, from the model's hints.

    The user's words in `when` win over the model's own `as_of`: it supplies the
    date only when `when` is empty or not a time we know. With neither, a time
    phrase left inside the query still counts. None means "as things stand now",
    which is also what a phrase that reaches today resolves to."""
    span = resolve(when, today, bare=True)
    if span is None and not as_of:
        span = resolve(query, today)
    if span is None:
        return as_of or None
    day = span.point(today, pick)
    return day.isoformat() if day else None
