# Writing a long test set for a personal assistant's long-term memory

You are writing evaluation data for the long-term memory of a personal AI
assistant. The assistant remembers what a user tells it in chat over months and
years, and later answers questions about that user. People's lives change: they
move, change jobs, swap phones, change medication, break up, get pets, move back.
A good memory knows which facts are true *now*, which were true *back then*, and
which never stopped being true — across a long, busy, messy history.

You will not see the memory system and must not guess at how it works. Write
what a real user would plausibly say and ask; label it honestly. Hard, realistic
data is the goal — not data that is easy to score.

## What to produce

ONE fictional person per reply, followed over 18–30 months:

```
{
  "scenario": "firstname-city-role",          // lowercase slug, e.g. "lucia-valencia-architect"
  "today": "YYYY-MM-DD",                        // the date the questions are asked
  "statements": [ ... ],                        // what the user said, oldest first
  "questions":  [ ... ]                         // asked on `today`
}
```

### Statements (55–75)

Each is ONE chat message the user sent the assistant, first person:

```
{"id": "s1", "date": "YYYY-MM-DD", "text": "...", "replaces": []}
```

- `id`: s1, s2, s3 … in order. `date`: strictly increasing, spread over the
  18–30 months before `today`, the last within ~3 weeks of `today`. Uneven
  rhythm: bursts of messages in a busy week, quiet months.
- `text`: how real people type. Vary it: one-fact messages; two-fact messages
  (one part may update an old fact while the other adds something new); long
  rambling ones where the fact is buried mid-sentence; terse ones ("new job.
  finally."); the odd typo, abbreviation, or word from the person's own
  language. 4–60 words.
- `replaces`: the ids of EARLIER statements whose fact this message makes **no
  longer true**. Usually `[]`.

Cover at least 12 of these life areas, with real detail:
home and neighbourhood · partner, housemates · job, employer, role, manager,
colleagues, hours, pay · commute and vehicles · phone, laptop, subscriptions ·
money (bank, rent, savings goal, debt) · health conditions, medications and
doses, allergies, doctors · diet and drinks · exercise and sport · hobbies and
classes · pets · family (parents, siblings, children — their schools, jobs,
homes) · friends · education and languages · travel and plans · routines and
schedules (days, times) · preferences and dislikes · faith, community,
volunteering · important dates (birthdays, anniversaries).

### Updates — the heart of the set (18–28 per person)

An update is a later statement that makes an earlier fact out of date. Label it
by putting the old statement's id in the new statement's `replaces`. Include all
of these kinds:

- explicit ("Moved out of the shared flat — got my own place in Ruzafa now")
- implicit, the old fact never mentioned ("Picked up a Fairphone 5" — when an
  older message named a different phone), sometimes months after the old one
- CHAINS of 3–4 steps in at least two life areas (A → B → C → D): each step
  replaces only the step before it
- a REVERSAL: going back to an earlier state ("moved back into my old flat",
  "rejoined the old gym") — it replaces the CURRENT state (the newest step),
  never the original statement, which was already replaced
- a CORRECTION: "Sorry, I said Tuesdays earlier — physio is Wednesdays"
- a change of QUANTITY or SCHEDULE: a dose from 50 mg to 100 mg, the gym from
  three days to five, the rent going up
- a change in ANOTHER PERSON's fact ("My brother left Patras, he's in Athens
  now" replaces the earlier statement about where the brother lives). Another
  person's fact is only ever replaced by a later statement about that person
- a NEGATIVE ending: a break-up, quitting, a pet that died or was rehomed, a
  cancelled subscription — and the facts tied to it end too ("weekend trips
  with Tom")
- at least two messages that each replace two or three older ones

### Traps — replace NOTHING (at least 12 per person)

Later statements that look related but leave every earlier fact standing:
- an ADDITION: a second pet, a second allergy, a hobby alongside an old one
- a DETAIL about a current fact: "the new flat has a tiny balcony"
- SOMEONE ELSE's fact that looks like the user's: "Mum got a new iPhone" when
  the user's phone was mentioned; "my sister switched jobs"
- a ONE-OFF event: "had dinner at Casa Montaña on Friday"
- a PLAN, WISH, or MAYBE: "thinking about moving to Lisbon next year"
- a RESTATEMENT of something still true: "still at Acme, still loving it"
- a TEMPORARY state that is over by `today`: "staying at my sister's for two
  weeks while the flat is painted" — home did not change
- a JOKE or SARCASM: "sure, I'll just quit and become a monk"
- QUOTED ADVICE not acted on: "my doctor says I should cut coffee"
- a CHILDHOOD or long-ago memory: "when I was a kid we had a dog called Bruno"
  — not a current pet, and replaces nothing

Label rule: `replaces` lists X only if, after this message, X's fact is no longer
true right now (of the user, or of the person X is about). If X is still true,
is merely more detailed now, or the new message is about someone else, do not
list it. A statement is replaced at most once.

### Questions (28–36 per person), asked on `today`

```
{"id": "q1", "text": "...", "type": "current|past|multi|absent", "answer_ids": [...], "as_of": null | "YYYY-MM-DD"}
```

Write each as the user would type it — natural, their own words, **not** copying
the wording of the statement that answers it.

- `current` (11–14): what is true now. At least 3 about facts at the END of a
  chain, a reversal, or a correction; at least 2 about OTHER people's current
  facts ("Where's my brother living these days?"); at least 2 yes/no ("Am I
  still on the 50 mg?"); at least 1 about a quantity or schedule.
  `answer_ids`: every statement that is part of the current answer (the move
  AND its balcony detail). Never a replaced statement. `as_of: null`.
- `past` (8–10): what was true at an earlier time, where the answer DIFFERS
  from today's. Vary the phrasing: a month ("last March"), a season, a holiday,
  and an EVENT ("back when I was still at Paisabridge", "before I got
  Biscuit"). `as_of`: a calendar date inside the time meant. At least 2 ask
  about the MIDDLE of a chain. `answer_ids`: the statement(s) true on that date.
- `multi` (4–5): several standing statements together ("Which pets do I
  have?", "What can't I eat?"). At least one where a trap must be left out
  (the childhood dog is not a current pet). `as_of: null`.
- `absent` (4–5): never said and not inferable, but plausible — at least one
  sitting right next to a trap ("What's my sister's address?" when only a stay
  at her place was mentioned). `answer_ids: []`, `as_of: null`.

`answer_ids` must contain only ids from this person, must be non-empty for
`current`, `past`, `multi`, and must contain exactly the statements needed — not
everything loosely related.

## Variety

Every person is different: country and continent, city or village, age (18 to
85), occupation (include non-office lives: farmer, driver, carer, nurse on
shifts, unemployed, retired, student, small-shop owner, gig worker), family
shape, health, money situation, and voice (terse, chatty, formal, slangy,
second-language English). Different life areas carry the updates for different
people — not always home, job and phone.

Do NOT use these first names or cities (already used): __USED__.
Italy and Greece are already well covered — avoid them.

## Before you answer, check your own work

Ids sequential; dates strictly increase; every `as_of` falls inside the
statements' date range; every `replaces` points only to earlier ids; nothing
is replaced twice; every `current` answer is still true on `today`; every
`past` answer was true on its `as_of` date and differs from today's; every
trap has `replaces: []`; counts are within the ranges above. Write this person
from scratch: no sentences, question wording or update pattern reused from
anyone earlier, and do not generate it with code or a template.

Reply with `{"scenarios": [ <ONE person> ]}` — the JSON in ONE code block and
nothing else. Do not browse the web, run code, or use any other tool.
