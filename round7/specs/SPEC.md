# Writing a test set for a personal assistant's long-term memory

You are writing evaluation data for the long-term memory of a personal AI
assistant. The assistant remembers things a user tells it in chat over many
months, and later answers questions about that user. People's lives change: they
move, change jobs, swap phones, change diets, break up, get pets. A good memory
knows which facts are true *now*, which were true *back then*, and which never
stopped being true.

You will not see the memory system and must not guess at how it works. Write
what a real user would plausibly say and ask; label it honestly. Hard, realistic
data is the goal — not data that is easy to score.

## What to produce

A set of SCENARIOS. Each scenario is one fictional person.

```
{
  "scenario": "firstname-city-role",          // lowercase slug, e.g. "lucia-valencia-architect"
  "today": "YYYY-MM-DD",                        // the date the questions are asked
  "statements": [ ... ],                        // what the user said, oldest first
  "questions":  [ ... ]                         // asked on `today`
}
```

### Statements (24–28 per scenario)

Each is ONE chat message the user sent the assistant, first person, in the
user's own voice:

```
{"id": "s1", "date": "YYYY-MM-DD", "text": "...", "replaces": []}
```

- `id`: s1, s2, s3 … in order. `date`: strictly increasing, spread over the
  10–12 months before `today`, the last one within ~3 weeks of `today`.
- `text`: natural chat. Casual, sometimes with a little emotion, filler, or an
  aside. Mostly one fact per message, sometimes two. Vary length (6–35 words).
- `replaces`: the ids of EARLIER statements whose fact this message makes **no
  longer true**. Usually `[]`.

Cover a realistic spread of life areas: where they live, job / employer /
manager / role, commute, phone or laptop or car, diet, allergies or health,
exercise, hobbies, pets, partner and family, friends, routines, preferences.

### Updates — the heart of the set (8–11 per scenario)

An update is a later statement that makes an earlier fact out of date. Label it
by putting the old statement's id in the new statement's `replaces`.

Vary how updates are expressed:
- explicit ("Moved out of the shared flat — I've got my own place in Ruzafa now")
- implicit, the old fact is never mentioned ("Finally picked up a Fairphone 5,
  the camera is lovely" — when an older message named a different phone)
- about a relationship or person ("Neha moved to another team; my manager is
  Samir now")
- partial or reversed ("Not strictly vegetarian any more — the doctor wants me
  eating fish")
- a chain (A → B → C: C replaces B; B replaced A)
- one message that replaces two older ones (≥1 per writer, ideally 2–3 overall)
- a negative ending ("Tom and I broke up") that retires facts tied to that
  person ("weekend trips with Tom")

Include TRAPS — later statements that look related but replace NOTHING
(`replaces: []`). At least 4 per scenario:
- an ADDITION: a second pet, a second allergy, a new hobby alongside an old
  one, another sibling.
- a DETAIL about a current fact: "the new flat has a tiny balcony" adds to the
  move; it does not replace it.
- SOMEONE ELSE'S fact: "My brother just got a job at Siemens" does not replace
  the user's own job.
- a ONE-OFF event: "Had dinner at Casa Montaña on Friday" is not where they
  eat every day.
- a PLAN, WISH, or MAYBE: "Thinking about moving to Lisbon next year" changes
  nothing yet.
- a RESTATEMENT of something still true ("Still at Acme, still loving it").

Label rule: `replaces` lists X only if, after this message, X's fact is no longer
true of the user right now. If X is still true, or was only ever about someone
else, or is merely more detailed now, do not list it.

### Questions (15–17 per scenario), asked on `today`

```
{"id": "q1", "text": "...", "type": "current|past|multi|absent", "answer_ids": [...], "as_of": null | "YYYY-MM-DD"}
```

Write each question as the user would type it to their assistant — natural, in
their own words, and **not** copying the wording of the statement that answers
it ("Who do I report to these days?", not "Who is my manager at work?" if the
statement said "Karthik is my manager at work").

Mix per scenario:
- `current` (7–8): about what is true now. Ask especially about facts that
  CHANGED — that is where memory fails. `answer_ids`: every statement that is
  part of the current answer (e.g. the move AND its balcony detail). Never a
  replaced statement. `as_of: null`.
- `past` (4–5): about what was true at an earlier time ("Back in November, where
  was I living?", "Which phone did I have around Christmas?"). `as_of`: the
  calendar date that time refers to (e.g. "2025-11-15", "2025-12-25"). Pick
  times where the answer DIFFERS from today's. `answer_ids`: the statement(s)
  true on that date.
- `multi` (1–2): the answer is several standing statements together ("Which
  pets do I have?", "What foods do I have to avoid?"). `as_of: null`.
- `absent` (2): something the user NEVER said and that cannot be inferred
  ("What's my mum's name?"). Make it plausible — near topics that were
  discussed. `answer_ids: []`. `as_of: null`.

`answer_ids` must contain only ids from that scenario, must be non-empty for
`current`, `past`, `multi`, and must contain exactly the statements needed — not
everything loosely related.

## Variety across scenarios

Different countries and continents, cultures, ages (students to retirees),
occupations, family situations, and writing voices (terse, chatty, emoji-free
formal, slangy). Different life areas should carry the updates in different
scenarios — not always home, job and phone.

Do NOT use any of these (already used elsewhere): people named Anika, Bjorn,
Carmen, Dmitri, Esther, Hiroshi, Nandini, Wanjiru, Rafael, Sven, Tessa, Walter;
cities Bengaluru, Bergen, Guadalajara, Toronto, Leeds, Osaka, Hyderabad,
Nairobi, Curitiba, Gothenburg, Perth, Ohio (any city in Ohio).

## Before you answer, check your own work

For every scenario: ids are sequential; dates strictly increase and every
`as_of` falls inside the statements' date range; every `replaces` points only
to earlier ids; a statement is replaced at most once (in a chain, the newest
replaces only the one before it); every `current` answer is a statement still true
on `today`; every `past` answer was true on its `as_of` date and is not the
same as today's answer; counts are within the ranges above.

Reply with JSON only: `{"scenarios": [ ... ]}`. No commentary, no markdown
fences. Do not use any tools.
