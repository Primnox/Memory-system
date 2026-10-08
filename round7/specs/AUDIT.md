# Labelling a test set for a personal assistant's long-term memory

Below is evaluation data for the long-term memory of a personal AI assistant.
Each SCENARIO is one fictional person: the chat messages they sent the assistant
over many months (`statements`, oldest first, each dated), and questions they
ask on `today`. The labels have been removed. You are an independent annotator:
re-derive them from the text alone. Do not guess what anyone else labelled;
label what the text says.

## Label 1 — `replaces`, on every statement

The ids of EARLIER statements whose fact this message makes **no longer true
right now** (of the user, or of the person that statement is about). Usually `[]`.

- Updates can be explicit ("Moved out of the shared flat, got my own place"),
  implicit (a new phone named when an older message named a different one),
  partial ("Not strictly vegetarian any more"), or a negative ending ("Tom and
  I broke up" ends "weekend trips with Tom"). One message may replace two.
- In a chain A → B → C, C replaces only B. A reversal ("moved back into my old
  flat") replaces the current state (the newest step), not the original.
- A correction ("Sorry, I said Tuesdays — it's Wednesdays") and a change of
  quantity or schedule (a dose from 50 mg to 100 mg) replace the old fact.
- Another person's fact can be replaced, but only by a later statement about
  that same person ("my brother left Patras, he's in Athens now").
- A statement is replaced at most once.
- These replace NOTHING: an addition (a second pet, a second allergy, a new
  hobby alongside an old one); a detail about a current fact ("the new flat
  has a balcony"); someone else's fact, against the user's own ("my brother
  got a job at Siemens" leaves the user's job standing); a
  one-off event ("dinner at Casa Montaña on Friday"); a plan or wish
  ("thinking of moving next year"); a restatement of something still true; a
  temporary state that is over ("staying at my sister's for two weeks"); a joke
  or sarcasm; quoted advice not acted on ("my doctor says I should cut
  coffee"); a childhood or long-ago memory ("as a kid we had a dog").

## Label 2 — `answer_ids`, on every question

The statements that answer the question, judged as of `today` — or, when the
question has an `as_of` date, as of that date.

- About now: every statement that is part of the current answer (e.g. a move
  AND its balcony detail). Never a statement that is no longer true.
- About a past date (`as_of` given): the statement(s) true on that date.
- Asking for several things ("Which pets do I have?"): all standing statements
  that together answer it.
- If the user never said it and it cannot be inferred: `[]`.
- Exactly the statements needed — not everything loosely related.

## Output

Reply with JSON only, no commentary, no markdown fences, no tools:

```
{"labels": [
  {"scenario": "<slug>",
   "replaces": {"s1": [], "s2": [], ..., "s9": ["s2"], ...},   // every statement id
   "answers":  {"q1": ["s9", "s12"], "q2": ["s2"], ..., "q8": []}  // every question id
  }, ...
]}
```

## The data
