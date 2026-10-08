# Checking the answer key of a chat history for a personal assistant's memory

Below is ONE fictional person's chats with a personal assistant (`sessions`, oldest
first, each dated), a list of `facts` someone wrote down from the user's messages,
and `questions` the user asks on `today`. The labels have been removed. You are an
independent annotator: judge from the text alone.

## Label 1 — `supported`, for every fact

Does the user turn the fact points at (`session`, `turn`, 0-based index into that
session's turns) actually state it — allowing for references and relative dates
resolved from the conversation ("last month" in a chat dated 2026-03-14 = February
2026)? `true` or `false`. A fact the assistant said but the user never confirmed,
a hypothetical, or one that misreads the message is `false`.

## Label 2 — `replaces`, for every fact

The ids of EARLIER facts this fact makes **no longer true right now** (of the user,
or of the person that earlier fact is about). Usually `[]`.

- Updates can be explicit, implicit (a new phone named when an older fact named a
  different one), partial ("not strictly vegetarian any more"), or a negative ending
  ("we broke up" ends "weekend trips with Tom"). One fact may replace two or three.
- In a chain A → B → C, C replaces only B. A reversal ("moved back into my old
  flat") replaces the current state, not the original.
- A correction ("I said Tuesdays — it's Wednesdays") and a changed quantity or
  schedule (a dose from 50 mg to 100 mg) replace the old fact.
- Another person's fact can be replaced, but only by a later fact about that person.
- These replace NOTHING: an addition (a second pet, a second allergy); a detail about
  a current fact; someone else's fact, against the user's own; a one-off event; a
  plan or wish; a restatement; a temporary state that is over; a joke; quoted advice
  not acted on; a childhood memory.
- A fact is replaced at most once.

## Label 3 — `answers`, for every question

The fact ids that answer it, as of `today` — or as of `as_of` when the question has
one. About now: every fact that is part of the current answer, never one no longer
true. About a past date: the facts true on that date. Asking for several things:
all standing facts that together answer it. Never said and not inferable: `[]`.

## Output

Reply with JSON only, no commentary:

```
{"labels": [
  {"scenario": "<slug>",
   "supported": {"f1": true, "f2": true, ...},       // every fact id
   "replaces":  {"f1": [], ..., "f9": ["f2"], ...},   // every fact id
   "answers":   {"q1": ["f9"], ..., "q8": []}         // every question id
  }
]}
```

Work only from the data below. Do not use any tools: do not read, list or search any
files or folders, run commands or browse.

## The data
