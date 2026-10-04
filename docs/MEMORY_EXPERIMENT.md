# Memory experiment: topics, slot-based supersession, time-aware recall

Started 2026-10-01 on branch `exp/memory-v3`; finished 2026-10-02 on
`fix/memory-change-detection`, which merges every fix that passed (see below).
Two more steps followed on 2026-10-03: `fix/memory-time-phrases` and
`exp/memory-change-verify` (the current best, 63%). On 2026-10-04 a small
trained decision model replaced the change rules in a blind replay (95/112
changes noticed, 2 wrong retirements): see [`RESEARCH.md`](RESEARCH.md) §6.

## Why

Primnox's permanent memory (`backend/primnox2/memory/service.py`) was fast, at
about 1 ms per search, but it was weak at the two jobs memory exists for:
recalling the right fact and knowing when a fact has changed.

## Results — blind tests (the numbers to quote)

Two evaluation sets, each written from a spec by agents that never saw this
code, labels independently re-derived from a label-free copy before any
scored run (agreement 93/93 + 166/168 on the first, 111/112 + 195/195 on the
second), frozen by SHA-256:

- `scripts/blind_memory/test.json` — 10 people, 267 statements, 168 questions.
  Used to choose between branches (~10 times), so it is no longer fully blind.
- `scripts/blind_memory/v2/test.json` — 12 people, 309 statements, 195
  questions, written after all tuning, scored once per version. **The
  headline.**

Top-1 = the first search result is a right answer. Retrieval only, with the
true date passed for "back then" questions (`--query oracle`).

| Version | v2 answerable top-1 (95% CI) | "now" questions | updates caught | wrong retirements |
|---|---|---|---|---|
| Before (`89ae45f`) | 19% [14–25] | 14/93 | 16/112 | 55 of 71 |
| `exp/memory-v3` (topics/slots/time) | 34% [27–41] | 22/93 | 12/112 | 27 of 39 |
| `exp/memory-v4` (+ scale, lifecycle, safety, embeddings) | 57% [50–64] | 47/93 | 54/112 | 28 of 82 |
| `fix/memory-change-detection` | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| `fix/memory-time-phrases` | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| **`exp/memory-change-verify`** (verifier on) | **63% [55–69]** | **56/93** | **73/112** | **20 of 93** |

Paired exact McNemar on v2: change detection vs before p<0.0001, vs v3
p<0.0001, vs v4 p=0.55 (change detection's gain is in supersession quality,
not yet in answers); verifier vs time phrases p=0.03 with 0 questions lost.
The first set gives the same picture (12% → 34% → 54% → 56%).

**Time phrases** resolve "back in March", "last summer" etc. in code
(`memory/when.py`, `recall_memory` gains a `when` argument). Identical with the
true date supplied, but with the local 9B writing the query on v2: 48% → 53%,
"back then" questions 24 → 34 of 54 (p=0.002).

**Verifier** (`memory/verifier.py`, tunable `memory.llm_verify`, env
`PRIMNOX2_MEMORY_LLM_VERIFY=1`, off by default): pairs the rules and
embeddings are unsure about are queued and checked in the background by the
local 9B, which must give P(replaces) ≥ 0.95 from its token probabilities.
Every decision is logged to the `memory_verifications` table — future
training data for the small model (see `SYSTEM_ONE_PLAN.md`). The queue is in
memory, so it is lost on restart.

With the app's own local model (`huihui_ai/qwen3.5-abliterated:9b`) writing the
search query and date itself, on the first set: before 21%, v3 29%, final 45%.
Rewording every statement and question (two paraphrase variants) flips ~15% of
individual outcomes with no systematic direction, so per-question results are
wording-sensitive while totals are stable.

What each fix did on the blind set, measured once each against v3: lifecycle
+1, safety ±0 (fewer wrong retirements), scale ±0 (identical answers, ~10x
faster), embeddings +30 of 148 (p<0.0001), robustness −7 (not merged: its
"now/new" update markers retire true facts).

Re-run: `python scripts/bench_memory_blind.py --split test --data
scripts/blind_memory/v2` (`--backend`, `--embeddings`, `--query model`,
`--mcnemar A.json B.json`). `--misses` is refused on test splits.

## Development scores (contaminated — kept for history)

The table below is what this document originally reported. It is a dev score:
the rules were tuned by reading the same benchmark's misses, and the
"held-out" set was written in the same session. Every "100%" here fell to
34% on the first blind test.

SDL `memory-100` (seed 20260815).

| | Before | After |
|---|---|---|
| Current-fact recall, top-1 (SDL L1) | 40% | **100%** |
| "What was it in month N", top-1 (SDL L4) | 0% | **100%** |
| Held-out recall, top-1 | 33% | **100%** |
| Stale memories retired by the *right* successor (SDL) | not measured ¹ | **10 of 10** |
| Stale memories retired (held-out) | 3 of 6 | **6 of 6** |
| Memories wrongly retired, `remember()` path (SDL) | 63 of 100 | **0** |
| Memories wrongly retired, `office-500` (400 memories) | not measured | **2 (0.5%)** |
| Bulk import supersession | none at all | **same rules as chat** |
| Search latency p95 | ~1 ms | ~1.5 ms (5 ms at 400 memories) |

¹ The old metric reported 10/10 here because it only asked *whether* each
stale memory was retired. In import mode, a mid-experiment dump showed one
"Devan switched to cortados." retiring eight unrelated memories through the
person's own name, and each of those counted as a catch. Supersession is now
scored by (old, successor) pair. The pre-change code was never re-run under
that stricter metric, so there's no "before" figure.

Unseen corpora (`memory-100` seed 7, `memory-10`, `office-500`) also score 100%
on recall. That suggests the topic lexicon isn't just fitting one seed, but the
SDL statements are still template-generated. The held-out set is the more
honest signal, and it's small (12 queries).

## What changed

- **`cognition/topics.py` (new).** Deterministic `topic` (the slot a statement
  fills: editor, residence, coffee…) and `kind` (`fact` or `event`). Some
  topics hold several values (projects, allergies) and never supersede each
  other. There is no model in the write path.
- **`cognition/conflict.py`.** Decisions are made on the slot when both sides
  name one. Events never retire anything. Different named subjects ("Nadia…"
  vs "Omar…") and different named scopes ("…for Fathom" vs "…for Ridge") are
  different facts. An attribute of a named thing ("the deadline for Atlas")
  is its own slot. The person's own name no longer counts as a shared word for
  the reversal rule. Callers that pass no topic behave exactly as before.
- **`memory/service.py`.**
  - `remember()` and `import_many()` both classify and resolve. An import
    resolves in timestamp order and dates each retirement by its successor's
    own time.
  - `search()` ranks by topic match, then question coverage (stems, question
    words removed), then recency, then specificity. Retired memories are
    excluded by default. `as_of` answers as of a date.
  - `backfill_topics()` classifies old rows and never re-judges them.
  - `recheck_conflicts()` is a dry run by default.
- **Schema v12.** Adds `memories.topic` and `memories.kind` as plain
  `ADD COLUMN`s.
- **`recall_memory` tool.** Takes `as_of` (`YYYY-MM` or `YYYY-MM-DD`).
  `forget_memory` still reaches retired versions.
- **API.** `GET /memories/recheck` returns proposals, `POST` applies them. The
  Memory tab's search keeps showing history.

## Decisions

- **Embeddings (plan phase 2b): built as two arms on `exp/memory-embeddings`,
  ON by default since `fix/memory-change-detection` (see Change detection).** The dev corpora
  above have no headroom, so the numbers that exist are dev numbers on
  paraphrase questions written for the purpose. Search: `memory.embeddings`
  fuses cosine similarity with the topic matches by reciprocal rank.
  Supersession: `memory.embeddings_supersede` lets similarity pick which memory
  a change-worded statement replaces. Run the blind runner with
  `--embeddings search|supersede|both`. `memory.actor_guard` is separate and
  not an embeddings setting. The write path never needs the model: every arm
  falls back to the word rules when the encoder is not up.
- **Same-revision disagreements.** These are two claims about one attribute,
  such as "deployment target for Atlas Core is Postgres" and "…is a feature
  flag". The newer claim is treated as an update. The SDL calls them disputes.
  The remaining `office-500` false positives are all this case. Real dispute
  handling would set `memories.disputed`, which exists but nothing writes.
- **Existing user stores.** Topics are backfilled, but nothing that is already
  retired gets un-retired automatically. The re-check endpoint shows what the
  new rules would change before anything does.

## Change detection (`fix/memory-change-detection`)

Most missed updates were implicit ("Got a Pixel 9 last week") and two rules
hid them from the engine. All numbers below are DEV numbers (tuned on).

- **A change is not an event.** `kind_of` filed "Got a Pixel 9 last week" and
  "Started at Deloitte on Monday" as one-offs because of the time phrase, and
  events never retire anything. A time phrase now makes an event only when the
  statement does not say something changed (`topics.TRANSITION`); "Had sushi
  with Mei on Friday", "Went to a gig" and "Isla won a medal" stay events. A bare
  "Bought …" stays an event for the word rules (two purchases are not one slot)
  and not for meaning (`is_episode`).
- **A capital is not a name.** `different_subjects` read every sentence-initial
  word ("Got", "Moved", "Coffee's", "Bloodwork") as someone's name, so the user's
  own update looked like a stranger's. A leading word is a name only when a verb,
  a copula or "'s" follows it, it has no lowercase twin in either text, and the
  sentence does not speak of the user; a function word or gerund next makes it the
  start of the user's own clause. A name the other memory also says is the same
  person ("Tolu moved back" after "My sister Tolu lives in Toronto"); "<Name> is my
  <role>" names a value of the user's own slot.
- **What two statements share must be specific.** Filler ("so", "ve" from "I've"),
  the word that says something changed, a relative's name and generic time words
  no longer count as a shared word, which had retired a coffee memory over a
  swimming one. A statement inside "when/if/without …" asserts nothing.
- **Meaning is gated.** Without a shared word it needs 0.30 (the tunable) and a
  clear lead (0.08) over the runner-up; with one, 0.25, and the same floor now
  vetoes the weak "reversal + one shared word" word rule. It never retires: a
  second acquisition ("Bought a bike" after "Bought a desk"), a purchase FOR the
  old thing ("a keyboard for my laptop"), a pet, interest, medicine or project,
  another person's statement, or an event. A swapped number ("iPhone 12" to
  "iPhone 15") is a cue in itself.
- **Lexicon.** Residence also reads "we moved", "Moved to" and "my brother moved
  to"; commute reads "get to school"; drug stems (-statin, -olol …) are
  medication and "off statins" is a stop; pets and interests are multi-valued.
- **Defaults.** `memory.embeddings`, `memory.embeddings_supersede` and
  `memory.actor_guard` are on. With the encoder down, or still loading, every
  call answers by words alone and none waits for it; a store with more than
  `SYNC_LIMIT` memories never embedded is filled on a thread, not inline. The
  test suite pins the arms off. `bench_memory.py` and `bench_memory_scale.py` pin
  them off unless asked; `bench_memory_blind.py` waits for the encoder when they
  are on and takes `--embeddings off`.
- **`recall_memory`.** The 9B put today's date or month in `as_of` on 79% of
  questions about now. Rule first, present-tense wording named, today and this
  month forbidden: 8% (2 of 24), with all 15 questions about the past still dated.

Dev split, `--embeddings both`, chat load, before (`ad1f2f8`) and after:
supersession recall 52% (14/27) to 78% (21/27), precision 45% (14/31) to 91%
(21/23), 12 of 31 retirements wrong to 1 of 23; answerable top-1 with oracle
queries 67% (30/45) to 76% (34/45), and with the 9B writing the query 58% (26/45)
to 64% (29/45). On `change.json` (own cases): recall 61% to 95%, precision 59% to
93%. Left alone: a chain of changes whose statements share no word ("streetcar",
"subway", "bike"), and different attributes of one person or thing that the
encoder scores alike ("my husband got a laptop" then "my husband has a company
car"). Those need a slot lexicon of phones, jobs and cars, or a better encoder.

## Lifecycle

Walking one fact through its life (`tests/test_memory_lifecycle.py`):

- **Forgetting a replacement undoes it.** `superseded_by` pointing at a
  forgotten row now counts as unset, or as whatever replaced the forgotten row.
  Before, "Porto" stayed retired for a "Lisbon" the user had forgotten and
  nothing answered "where do I live". Restoring Lisbon retires Porto again with
  no write: `live()` reports the effective successor, the column is untouched.
  Restoring a fact that something else replaced while it was forgotten
  retires it on the spot.
  *Decision for the user:* this reads "forget" as "undo", also when it was said
  to mean "I don't want to say where I live" (forget both, and nothing is left).
- **Editing re-files.** `update()` re-classifies topic/kind, applies the same
  gates as `remember()` (length, directives) and decides again what the edit
  replaces and what replaces it. A typo fix changes nothing.
- **Negations.** "I quit coffee" and "I don't have a car anymore" retire the old
  fact. The second did not: a sentence-final "." stayed on the token, so "car."
  never met "car". A negation that shares no word with what it ends ("I quit
  caffeine" vs "I drink coffee") still retires nothing. A search that finds only
  a retired fact is answered with what replaced it.
- **`as_of` is read in the device's time zone**, as the calendar is, then stored
  as UTC. An unreadable `as_of` raises instead of answering about now.
- **Writes are serialised.** Two facts stored at once could both retire the
  same old row and both stay live.
- The store was read 500 rows deep on `remember()` and 200 deep for the prompt,
  counting retired rows; both now see every live row.

## Known limits

- In `remember` mode every memory is stamped "now", so time questions are only
  answerable for chats after this change or for imported corpora. Run with
  `--mode remember` and L4 is 0% by construction.
- The topic lexicon is English and hand-written. An unnameable statement falls
  back to the old overlap rules.
- These are retrieval scores. End-to-end quality with a real model writing the
  query (plan phase 5) is not measured yet.
- The Memory tab has no UI for the re-check endpoint yet.
