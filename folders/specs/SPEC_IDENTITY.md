# Writing who-is-who timelines for a personal assistant's memory

You are writing evaluation and training data for the long-term memory of a personal AI
assistant. One user talks to the assistant over many months. Each thing the user says is a
short STATEMENT (one or two sentences, the way people really type). The memory keeps a
FOLDER for the user (the speaker) and one for every person, pet, place, organisation, group,
object, activity or event in their life, and files every statement into the folder it is
mainly about.

The hard part is identity: two people called Rohan; "Ro" who might be either; "my boss"
after a job change; a quote that is not the user's own claim; a joke; a story; a mistaken
merge discovered later. Your job is to write ONE realistic user whose statements are full of
such traps, and to label exactly what a careful human note-taker would decide at each
statement, knowing ONLY what had been said up to and including that statement — never
anything said later.

You will not see the memory system and must not guess at how it works. Hard, realistic data
is the goal, not data that is easy to score.

## What to produce

```
{
  "scenario": "firstname-city-role",     // lowercase slug
  "speaker": "Firstname",
  "today": "YYYY-MM-DD",
  "statements": [ ... ],                   // 35-50, oldest first
  "new_traps": [ ... ],                    // traps you invented that are NOT in your case list
  "questions": [ ... ]                     // 10-14, asked on `today`
}
```

### Statements

```
{"id": "s1", "date": "YYYY-MM-DD", "text": "...",
 "kind": "fact",
 "about": "person",
 "folder": "rohan_cousin",
 "new_folders": [{"id": "rohan_cousin", "type": "person", "name": "Rohan", "relation": "my cousin"}],
 "updates": [{"folder": "rohan_cousin", "alias": "Ro"}],
 "mentions": [{"text": "my cousin Rohan", "folder": "rohan_cousin"}],
 "replaces": [],
 "merge": null,
 "split": null,
 "ask_between": null,
 "cases": [1, 20]}
```

- `id` s1, s2 … in order; `date` never earlier than the previous statement's, spread over
  6–18 months before `today`. Several statements may share a day (one conversation).
- `text`: what the user typed, first person, informal, 4–40 words. Use nicknames, pronouns,
  "my boss", "the new one", "she", typos and half-sentences as real people do. Pronouns may
  point back to the previous one or two statements of the same day.
- `kind` — exactly one of:
  - `fact`: the user asserts it as true now or in the past (about anyone), including things
    they learned from others and accept ("my sister moved to Pune").
  - `plan`: an intention, booking or schedule that has not happened yet.
  - `hypothetical`: if / imagine / maybe-one-day; not asserted.
  - `question`: the user asks something (even if it presupposes a fact).
  - `quote`: repeating someone else's words or claim without endorsing it ("Kiran says he's
    the best cook in Pune").
  - `joke`: sarcasm, irony, exaggeration not meant literally ("my cat is basically my landlord").
  - `rumour`: hearsay flagged as unverified ("apparently", "I heard", "people say").
  - `correction`: fixes something the user said EARLIER in this timeline ("Ro was my lab
    partner, not my cousin"; "I said Tuesday, it's Wednesday").
  - `fiction`: about a book, film, game, roleplay, dream or story world.
- `about` — who the statement is mainly about: `speaker` (the user), `person` (one other
  person), `group` (a team, family, class, "we" meaning a group), `thing` (anything that is
  not a person: pet, place, organisation, object, activity, event).
- `folder` — the ONE folder the statement is mainly about, as a careful note-taker would file
  it NOW. Rule of thumb: when the USER is the one doing, owning, feeling or planning ("I got a
  new app", "driving to Savuti camp tomorrow", "my knee hurts") it goes in `"speaker"`, even if
  it names a thing or place (that thing still gets its own folder via `new_folders` and
  `mentions`); it goes in another folder only when the statement is mainly about THAT one —
  what it is, what it did, what happened to it ("Kwaku moved to Accra", "the app crashed
  again", "the boys start school Monday"). Options: `"speaker"`, the id of an existing folder, the id of a folder created in this same
  statement (list it in `new_folders`), or `"ask"`.
  - `"ask"` ONLY when the statement refers to someone or something that could be two or more
    existing folders (or an existing one or a new one) and NOTHING said so far tells which, so
    a careful assistant would ask the user. Put the candidates in `ask_between` (folder ids,
    and/or "new"). Do not overuse it: about 1 statement in 10, never when context resolves it,
    and never when only ONE existing folder fits the reference (one Akosua in memory → the
    statement about Akosua goes in her folder).
    The user may clarify in a later statement; label that clarifying statement `fact` (or
    `correction` if it contradicts an earlier claim), filed in the folder meant.
  - Never merge by name alone: a second "Rohan" with nothing linking him to the first gets a
    NEW folder (or "ask" if it is genuinely unclear which).
- `new_folders` — folders created by this statement: `id` (short snake_case you choose),
  `type` (person, pet, place, organisation, group, thing, activity, event), `name` (the name
  known so far, or null if unnamed, e.g. "my new manager"), `relation` (how it links to the
  user so far, e.g. "my cousin", "my manager at Infosys", "" if unknown). Create a folder for
  every named or clearly identified other entity the first time it matters, not only the main
  one.
- `updates` — what this statement teaches about an existing folder: `{"folder", "alias"}` (a
  new nickname or other name for it), `{"folder", "name"}` (its real name is learned or
  corrected), `{"folder", "relation"}` (its link to the user is learned or changes).
- `mentions` — every reference in the text to a person or thing other than the user
  (names, nicknames, "my boss", "he", "the gym one"…): `{"text": <exact words>, "folder":
  <folder id, or "ask">}`.
- `replaces` — ids of EARLIER statements this one makes no longer true right now (a move, new
  job, a break-up, a replaced phone, a corrected detail). Usually `[]`. Questions,
  hypotheticals, jokes, quotes, rumours and fiction never replace anything. A statement is
  replaced at most once.
- `merge` — `["keep_id", "gone_id"]` when this statement makes it clear that two EXISTING
  folders are one and the same (an alias discovered later, a duplicate). Only when the
  statement really establishes it.
- `split` — when ONE existing folder turns out to hold two different entities:
  `{"folder": <id>, "new_folder": {id, type, name, relation}, "move": [statement ids that
  belong to the new one]}`. (A careful note-taker only ever has such a folder when the earlier
  statements gave every reason to think it was one entity.)
- `cases` — the numbers of the cases from your list that this statement exercises, plus ids of
  your own new traps ("N1"). `[]` for plain statements.

Real changes: 8-14 statements replace an earlier one (a move, a new job or manager, a
replaced phone, a break-up, a changed dose or schedule, someone else's change, a chain of 2-3
steps, a correction that makes the corrected statement untrue). A statement that wrongly
looks like a change but is not one (another person's move, a repair instead of a replacement,
a hypothetical) replaces nothing.

At least a third of the statements are ordinary (plain facts about the user's life with no
trap) so the traps sit in a normal life. The traps must interact: a name collision that is
later resolved by a correction; a merge that is later split; a role ("my manager") whose
holder changes; a quote that a later statement shows was false.

### New traps

Invent at least 3 identity traps that are NOT in your case list and use each in a statement:
`{"id": "N1", "name": "<short name>", "description": "<one sentence>"}`. Real, everyday ones —
the kind that would actually confuse a memory.

### Questions (10-14), asked on `today`

```
{"id": "q1", "text": "...", "type": "current|past|who|absent", "answer_ids": [...], "as_of": null | "YYYY-MM-DD"}
```

`answer_ids` are statement ids whose content answers the question. Written as the user would
ask, never copying the statement's wording.
- `current` (3-5): true now, after any changes and corrections.
- `past` (2-3): true at `as_of` and different from today.
- `who` (3-4): needs the right identity ("what does my cousin do?" when two Rohans exist;
  "where does Ro live?" after Ro was revealed to be the lab partner).
- `absent` (2): plausible but never said; answer_ids [] and as_of null.

## Before you answer, check your own work

Ids and dates in order; every folder id used was created earlier or in that statement (in
`new_folders` or a `split`); merged-away folders are not used after the merge; `ask_between`
present exactly when folder is "ask"; every `replaces` points to an earlier statement, nothing
replaced twice, and only facts, plans and corrections replace; labels use only what was said
up to that statement; `current` answers are still true on `today`; counts in range. Write from
scratch, not from a template, and do not generate it with code.

Reply with `{"scenarios": [ <ONE person> ]}` — the JSON only, nothing else. Do not browse the
web, run code, or use any other tool.
