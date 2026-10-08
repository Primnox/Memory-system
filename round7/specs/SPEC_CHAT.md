# Writing chat histories for a personal assistant's long-term memory

You are writing evaluation data for the long-term memory of a personal AI
assistant. The user chats with the assistant over many months, mostly to get
help with everyday things — and along the way mentions facts about their own
life, usually in passing. The memory must pull those facts out of the chats,
notice when a later fact makes an earlier one out of date, and answer questions
about the user later.

You will not see the memory system and must not guess at how it works. Write
what a real user would plausibly say and ask; label it honestly. Hard, realistic
data is the goal — not data that is easy to score.

## What to produce

ONE fictional person per reply:

```
{
  "scenario": "firstname-city-role",   // lowercase slug, e.g. "lucia-valencia-architect"
  "today": "YYYY-MM-DD",                 // the date the questions are asked
  "sessions":  [ ... ],                  // the chats, oldest first
  "facts":     [ ... ],                  // the answer key: every fact the user stated
  "questions": [ ... ]                   // asked on `today`
}
```

### Sessions (10–16 per person)

```
{"id": "c1", "date": "YYYY-MM-DD", "turns": [
   {"role": "user", "text": "..."},
   {"role": "assistant", "text": "..."},
   ...
]}
```

- `id`: c1, c2 … in order; `date` strictly increasing across sessions, spread over
  12–24 months before `today`, the last within ~3 weeks of `today`.
- 6–12 turns per session, alternating user / assistant, starting with the user.
  Real chats are mostly about the TASK: about HALF of all user turns should state
  no new personal fact at all ("ok that works, what about dessert?", "thanks!
  can you make it shorter?", "hmm, cheaper options?").
- Each session has a real PURPOSE the user wants help with — planning a trip,
  a recipe for guests, an email to a landlord, a workout plan, a budget, a gift,
  homework help for a child, fixing a phone setting, a cover letter, choosing a
  laptop. Vary purposes across sessions.
- The user's personal facts appear AS THEY WOULD IN REAL CHATS:
  - in passing, inside a request ("can you suggest a recipe — oh and I'm off
    dairy now, doctor's orders");
  - several in one message; a fact buried mid-paragraph;
  - with pronouns and references that need the conversation to resolve ("we
    finally moved there", "she starts next week", "the new one");
  - with relative time ("last month", "since Easter", "two weeks ago");
  - as a correction later in the same chat ("wait, it's Thursdays, not Tuesdays");
  - some user turns contain NO personal fact at all (pure requests, thanks).
- Assistant turns are short (1–3 sentences) and helpful. The assistant never
  invents facts about the user and never repeats a fact back as new information.
- User turn length varies: 3–150 words, typically 30–70; a few long rambling
  turns (100+ words) where one fact hides in the middle of a request.

### Facts — the answer key (40–60 per person)

Every fact about the user (or about people and pets in their life) that the user
STATES, rewritten as a clean memory entry:

```
{"id": "f1", "session": "c3", "turn": 2, "text": "...", "replaces": []}
```

- `session`, `turn`: where the fact was stated (`turn` = index in that session's
  `turns`, 0-based; always a user turn).
- `text`: ONE self-contained fact in the user's first person, 5–25 words, with
  every pronoun, reference and relative date resolved using the conversation and
  the session date ("We moved to Denver in March 2026", not "we finally moved
  there"; "My daughter Mia started school at Lincoln Elementary"). Exactly what a
  perfect note-taker would save. Questions, requests, hypotheticals and the
  assistant's words are NOT facts.
- `id`: f1, f2 … in the order stated.
- `replaces`: the ids of EARLIER facts this fact makes no longer true right now
  (of the user, or of the person that earlier fact is about). Usually `[]`.

Life changes — the heart of the set, 14–24 updates per person, of every kind:
explicit moves and job changes; implicit ones (a new phone named, the old never
mentioned); chains of 3–4 steps (each replaces only the step before); a reversal
(back to an earlier state: replaces the current one); a correction in a later
chat ("I said Tuesdays — it's Wednesdays"); a changed dose, rent or schedule;
another person's fact changing ("my brother left Patras, he's in Athens now"); a
negative ending (break-up, quitting, a pet that died) with the facts tied to it.
At least two facts replace two or three older ones.

Traps — at least 10 facts that replace NOTHING: a second pet or allergy (an
addition); a detail about a current fact; someone else's fact that looks like the
user's ("Mum got a new iPhone"); a one-off event; a plan or maybe; a restatement;
a temporary state that is over by `today`; quoted advice not acted on; a
childhood memory. Plans, jokes and quoted advice that a note-taker would still
save are facts with `replaces: []`; pure hypotheticals ("if I ever move…") are
not facts at all.

Label rule: `replaces` lists X only if, after this fact, X is no longer true
right now. A fact is replaced at most once.

### Questions (25–32 per person), asked on `today`

```
{"id": "q1", "text": "...", "type": "current|past|multi|absent", "answer_ids": [...], "as_of": null | "YYYY-MM-DD"}
```

`answer_ids` are FACT ids. Written as the user would type them, never copying
the fact's wording.
- `current` (10–13): true now; at least 3 at the end of a chain, a reversal or a
  correction; at least 2 about other people; at least 2 yes/no.
- `past` (7–9): true at an earlier date and different from today; `as_of` = a
  date inside the time meant; vary phrasing (a month, a holiday, an event:
  "back when I was still at Paisabridge"); at least 2 on the middle of a chain.
- `multi` (4–5): several standing facts together; at least one where a trap
  must be left out.
- `absent` (4–5): never said and not inferable, but plausible — including one
  about something the ASSISTANT said but the user never confirmed.

## Before you answer, check your own work

Session ids and dates in order; every fact's `session`/`turn` points at a user
turn that really states it; fact dates are their session's date; every
`replaces` points only to earlier facts; nothing replaced twice; every `current`
answer still true on `today`; every `past` answer true on its `as_of` and
different from today's; counts within the ranges. Write from scratch, not from a
template, and do not generate it with code.

Reply with `{"scenarios": [ <ONE person> ]}` — the JSON only, nothing else. Do
not browse the web, run code, or use any other tool.
