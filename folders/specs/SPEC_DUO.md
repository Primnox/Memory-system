# Writing a two-person memory set (facts in the third person)

You are writing training data for the long-term memory of an AI system that follows a long
chat between TWO friends over many months. A note-taker has already turned what they said into
short third-person facts that name who each fact is about ("Maya moved to Denver", "Jordan's
sister got married"). The memory must notice when a later fact makes an earlier one no longer
true — and, just as important, must NOT retire facts that are still true: hobbies, past events,
experiences and history stay true even when something new is added.

You will not see the memory system and must not guess at how it works. Hard, realistic data is
the goal, not data that is easy to score.

## What to produce

ONE pair of friends per reply:

```
{
  "scenario": "firstname-firstname-city",   // lowercase slug of both first names and a city
  "speakers": ["Firstname1", "Firstname2"],
  "today": "YYYY-MM-DD",
  "statements": [ ... ],                     // 50-70, oldest first
  "questions": [ ... ]                       // 20-26, asked on `today`
}
```

### Statements: the facts as the note-taker wrote them

```
{"id": "s1", "date": "YYYY-MM-DD", "text": "...", "about": "Firstname1", "replaces": []}
```

- `id` s1, s2 … in order; `date` never earlier than the previous one (several facts share a
  chat date), spread over 6-14 months before `today`.
- `text`: ONE third-person fact, 5-25 words, naming the person by first name (or "Maya's
  brother Leo"), past or present tense as the chat put it. Mostly about the two friends, some
  about people in their lives. Include what real chats are full of: hobbies and interests,
  things they did (trips, concerts, a hike), opinions and likes, plans, feelings, work and
  study, family news, pets, health — not only life changes.
- `about`: whose fact it is — one of the two speakers, or the name of the third person.
- `replaces`: ids of EARLIER facts this one makes no longer true RIGHT NOW (about the same
  person). Usually `[]`.

Life changes (12-20 across both friends): moves, jobs, break-ups, a new phone or car, a
changed routine, a pet that died, an injury that healed, a plan that was cancelled; chains of
2-3 steps; one reversal; one correction ("Maya said Tuesday earlier; it's Thursday").

Traps that replace NOTHING (at least 15): a second hobby (adds, does not replace the first); a
new trip or event (events accumulate); the OTHER friend doing the same kind of thing ("Jordan
also moved" does not replace Maya's move); a third person's change that looks like a speaker's;
an opinion; a plan or maybe; a detail of a standing fact; a restatement; a past experience told
later ("Maya went to Japan as a kid"); a feeling that day.

Label rule: `replaces` lists X only if, after this fact, X is no longer true right now. A fact
is replaced at most once. Facts about different people never replace each other.

### Questions (20-26), asked on `today`

```
{"id": "q1", "text": "...", "type": "current|past|history|absent", "answer_ids": [...], "as_of": null | "YYYY-MM-DD"}
```

Asked by someone who read the chat, naming whom they ask about, never copying a fact's wording.
- `current` (6-8): true now. At least 3 where the answer is a long-standing fact that later
  facts did NOT replace ("What are Maya's hobbies?" → all her hobbies); at least 2 at the end of
  a change chain.
- `past` (3-5): true at `as_of` and different now.
- `history` (6-8): about things that happened or were said, whether or not still true ("Where did
  Jordan go on holiday?", "What jobs has Maya had?", "How did Jordan feel about the move?"):
  answer_ids include every fact that answers it, replaced or not; as_of null.
- `absent` (3-4): plausible but never said; [] and null.

## Before you answer, check your own work

Ids and dates in order; every `replaces` points to an earlier fact about the same person;
nothing replaced twice; `current` answers still true on `today`; `past` answers true on their
as_of and different from today; counts in range. Write from scratch, not from a template, and do
not generate it with code.

Reply with `{"scenarios": [ <ONE pair> ]}` — the JSON only, nothing else. Do not browse the web,
run code, or use any other tool.
