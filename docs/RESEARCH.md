# Local Long-Term Memory for AI Assistants: Blind Evaluation, a Small Trained Change Detector, and a Graph Index

**Kakanur Aniketh Pani** — Primnox

Technical report, October 2026. Code, data and every score:
[github.com/Primnox/Memory-system](https://github.com/Primnox/Memory-system).

---

## Abstract

Personal AI assistants need a memory that keeps details, notices when a fact
about the user stops being true, and can answer questions about the past. We
study a fully local memory (SQLite, a 22M-parameter sentence encoder, no
large language model on the write path) under a strict blind protocol: test
sets are written by agents that never saw the code, labels are re-derived
independently, sets are frozen by hash, and per-item failures on test sets are
never inspected. Rule and embedding improvements raise top-1 retrieval on a
fresh blind set from 19% to 63% [55–69]. We then fine-tune a 421M-parameter
non-generative decision model (Laya) to decide whether a new statement replaces
an earlier one, using only synthetic people from two independent generator
agents and twelve minutes of one free cloud GPU. Replayed on the same blind
set, it notices 95 of 112 changes (85% [77–90]) with 2 mistaken retirements,
against 73 of 112 (65% [56–73]) with 20 for the best rule-based system with a
9B verifier; three further training runs give 92–94/112 with 1–2. With its
decisions in the memory, top-1 answers rise from 59.1% to 66.7% [59–73]
(paired p = 0.007), against a 70.8% ceiling for perfect change detection. On data not written by
our generator family the advantage shrinks but persists: F1 0.51 vs 0.26 for
the rules on crowdworker-written persona facts, and 0.24 vs 0.09 on long chat
turns from LongMemEval. Adding restatements and long chatty messages to the
training data cut restatement errors 4.6-fold (170 → 37 per 1,000) without
loss on the blind set (94/112), but did not help with changes buried in long
chat turns; splitting messages into statements about the user before comparing
them did (6 → 33 of 72 changes caught, F1 0.15 → 0.54). Escalating the 4%
least certain decisions to a supervisor would remove about 80% of the
detector's errors. Finally, selecting facts for the prompt instead of
sending all of them cuts memory tokens by 45% to 89% (4,075 → 454 tokens at 200
facts) while keeping 97% and 79% of the facts the answers need; a graph index
over people, things and topics does not select better than plain similarity.

---

## 1. Introduction

A memory for an assistant has two jobs: return the right fact, and notice when
a fact stops being true. "I moved to Pune" must retire "I live in Lisbon" but
keep it for "where did I live in March?"; "my sister lives in Lisbon" must
retire nothing. The decisions involved are small and typed: a change, a second
value (another pet), more detail, someone else, a one-off event.

Deployed systems show three recurring problems. Summaries that are rewritten
over time lose detail. Similarity search returns outdated and current facts
side by side. And several systems call a large language model (LLM) on every
message to maintain memory, which is slow, costly and, for a cloud LLM,
private data leaving the device.

This report makes four contributions:

1. **A blind evaluation protocol for assistant memory,** motivated by a
   measured failure: rules scoring 100% on development data scored 34% on the
   first blind set (§4).
2. **A local memory system** improved from 19% to 63% top-1 on a fresh blind
   set without any LLM on the write path (§5).
3. **A small trained change detector** that replaces hand-written rules:
   85% of changes noticed with 2 mistaken retirements on the blind set (rules
   with a 9B verifier: 65% and 20), raising top-1 answers from 59% to 67%,
   with tests on data from other authors that show where it breaks and two
   fixes that address it (§6).
4. **A measured negative result on graph indexing for prompt selection:**
   selection saves 45–89% of memory tokens, but a rule-built graph does not
   select better than similarity (§7).

---

## 2. Related work

**Memory systems.** Table 1 summarises published designs; none was run here.

*Table 1. Memory designs, as published.*

| System | What it keeps | Who decides updates | When a fact changes | LLM calls per write | Local |
|---|---|---|---|---|---|
| Vector RAG | text chunks | nobody | nothing; old and new kept as equals | 0 | yes |
| ChatGPT memory [W25] | short facts + a profile rewritten in the background | the chat model | the profile is rewritten | 1+ | no |
| Claude memory [W25] | raw past chats | nothing extracted; tool search on demand | left to the model reading old chats | 0 | no |
| MemGPT / Letta [P23] | self-edited memory blocks + archive | the main model, via tools | the model edits its note | main-model turns | yes |
| Mem0 [C25] | LLM-extracted facts (+ optional graph) | an extraction LLM, then an update LLM | update or delete, with a change log | ~2 | yes |
| Zep / Graphiti [R25] | episodes + temporal knowledge graph | several LLM calls | fact invalidated with an end date, kept | several | yes |
| MemOS [L25] | memory graph of "MemCubes"; graph + vector stores | LLM extraction and organisation, asynchronous | user feedback and correction; versions | 1+ | cloud, self-hosted or local plugin |
| Cortex (prem-research) [CX1] | short-term buffer + vector long-term store, "smart collections" | keyword tagging, then an LLM in the background | links and merges; no explicit contradiction handling | 1 (background) | needs a vector server + LLM key |
| Cortex (gambletan) [CX2] | SQLite + vector index; working / episodic / semantic / procedural tiers; people graph | rule-based extraction, no LLM | belief confidence; contradicted beliefs decay, kept | 0 | yes (MCP, encrypted sync) |
| HippoRAG [G24] | documents + a knowledge graph as index | LLM extraction, once | not handled (documents) | many, once | yes |
| **This work** | the user's words + date + topic; "replaced by" links | rules + embeddings; a small trained classifier | retired, linked, kept with dates | **0** | **yes** |

Several systems report LoCoMo [M24] and LongMemEval [Wu25] scores, all
self-reported: MemOS 92.3 and 93.4, Mem0 92.5 and 94.4, Cortex about 71–74 on
LoCoMo; third-party re-runs often report lower numbers. We do
not report on these benchmarks yet; the closest designs to ours are Graphiti
(dated invalidation instead of deletion) and gambletan's Cortex (local, no LLM
on the write path, decaying contradicted beliefs).

**Decision models.** TypeSafe AI's Jev [J26] introduced hosted "System One" models
that answer typed questions with calibrated probabilities instead of generated
text; open replicas followed (Open-Jev, Kev, Laya). We fine-tune Laya [LY], a
ModernBERT-large encoder [Wa24] with a decision head, trained by its authors
with reinforcement learning against proper scoring rules.

**Graphs and smaller models.** HippoRAG [G24] uses a knowledge graph as an
index over passages, walked with personalized PageRank. RETRO [B22] and
Atlas [I22] show that retrieval can let smaller models match larger ones on
knowledge tasks; personal memory is a different problem, and we make no claim
about it.

---

## 3. System

![Architecture](architecture.png)

*Figure 1. Architecture. Solid: built. Dashed: planned.*

- **Write path, no LLM.** A statement is filtered (junk, length, instructions
  vs facts), labelled with a topic and as a lasting fact or a one-off event,
  matched against related live facts (same topic, plus similarity under the
  all-MiniLM-L6-v2 sentence encoder [RG19, Wn20]), and a change decision is
  made: change, second value, someone else, detail, one-off, duplicate.
- **Storage.** One SQLite file. A retired fact keeps a link to its successor;
  nothing is deleted or rewritten.
- **Read path.** Every prompt carries a memory block: safety facts (allergies,
  medication) first and uncapped, then current facts up to 200. On demand,
  `recall_memory` runs hybrid word and embedding search; time phrases ("back
  in March", "last summer") are resolved to dates in code, and only facts true
  at that date are returned.
- **Optional verifier.** Pairs the rules are unsure about are queued and
  checked in the background by a local 9B model, which must give
  P(replaces) ≥ 0.95 from its token probabilities.

---

## 4. Evaluation method

**Motivation.** Development benchmarks whose misses had been used to tune the
rules reached 100%; the first blind test of the same code reached 34%. Every
result in this report follows one protocol:

1. **Blind authorship.** Test sets are written from a written spec by Claude
   Sonnet agents that never saw the code: invented people, dated first-person
   statements with `replaces` labels, traps that must not replace (second
   values, refinements, other people, one-off events), and questions (current,
   past with a time phrase, multi-value, absent).
2. **Independent labels.** A second agent re-derives every label from a
   label-free copy before any scored run.
3. **Freeze, score, never look.** Sets are frozen by SHA-256. Per-item failures
   on test sets are never read (the runner refuses to print them), and nothing
   is tuned on a test set.
4. **Intervals and tests.** 95% Wilson intervals [Wi27]. Paired exact McNemar
   tests [Mc47] where per-item outcomes exist for both systems, otherwise
   two-proportion z-tests (unpaired, conservative for paired designs).

*Table 2. Data. "Authors" are the writers of the text.*

| Set | Authors | Size | Used for |
|---|---|---|---|
| `test.json` | 2 Sonnet agents | 10 people, 267 statements, 93 changes, 168 questions | choosing between branches (~10 runs; no longer fully blind) |
| `v2/test.json` | 2 other Sonnet agents | 12 people, 309 statements, 112 changes, 195 questions (171 answerable) | **headline blind set**; scored once per version |
| `dev.json`, `change.json` | Sonnet agents | 10 people, 263 statements, 70 changes, 85 questions | development: failures read, rules tuned |
| Training people A | 1 Sonnet agent (prose) | 12 people, 350 statements | training the change detector only |
| Training people B | 1 Sonnet agent (texting style) | 12 people, 327 messages | training the change detector only |
| Training people C | 1 Sonnet agent (long chatty messages, restatements) | 12 people, 308 messages | training (round 3, §6.8) |
| Dialogue NLI, verified test [We19] | crowdworkers | 3,000 pairs (1,000 per label) | external test |
| LongMemEval, knowledge update [Wu25] | GPT-4o, human-checked | 421 pairs, 72 changes | external test |

Label agreement before scoring: 93/93 changes and 166/168 answers on
`test.json`; 111/112 and 195/195 on `v2`.

**Metrics.** *Top-1:* the first search result is a correct answer, with the
true date supplied for "back then" questions (*oracle*) or with the local 9B
writing the query and date (*model*). *Changes noticed:* (old, successor)
pairs retired correctly, out of all labelled pairs. *Mistaken retirement:* a
retired fact the labels never replace. *Pair precision:* correct retirements
over all retirements.

---

## 5. Experiment 1: rules, embeddings, time, verifier

*Table 3. Blind set `v2`, oracle dates.*

| Version | Top-1 (95% CI) | Changes noticed | Mistaken retirements |
|---|---|---|---|
| Starting point | 19% [14–25] | 16/112 | 55 of 71 |
| Topics, slots, time | 34% [27–41] | 12/112 | 27 of 39 |
| + scale, lifecycle, safety, embeddings | 57% [50–64] | 54/112 | 28 of 82 |
| + change detection | 59% [52–66] | 62/112 | 19 of 81 |
| + time phrases | 59% [52–66] | 62/112 | 19 of 81 |
| + 9B background verifier | **63% [55–69]** | 73/112 | 20 of 93 |

- Paired McNemar on top-1: change detection vs start p < 0.0001, vs topics
  p < 0.0001, vs embeddings p = 0.55 (its gain is in change quality, not yet
  in answers); verifier vs time phrases p = 0.03 with no question lost.
- Time phrases matter when the local model writes the search: top-1 48% → 53%,
  and "back then" questions 24 → 34 of 54 (p = 0.002).
- Rewording every statement and question (two paraphrase variants) flips about
  15% of individual outcomes with no direction: totals are stable, single
  items are wording-sensitive.
- A branch scoring 32/32 on hand-written practice cases scored worse on the
  blind set and was not merged.

---

## 6. Experiment 2: a small trained change detector

### 6.1 Task

For a new statement and each candidate earlier statement, the model answers
one typed choice: `replaces`, `adds`, `detail`, `other` or `event`
(Appendix B gives the exact wording). Input: `Earlier (date) the user said:
"…"` / `Later (date) the user said: "…"`. Output: a probability per option.
P(`replaces`) ≥ 0.5, fixed in advance, retires the old fact.

### 6.2 Zero-shot screen

Development pairs: 3,371 earlier/later pairs (70 changes); shortlist = the 5
most similar earlier statements (1,165 pairs, 69 changes).

*Table 4. Untrained scorers on the development shortlist.*

| Scorer | AP | Best F1 | F1 @0.5 |
|---|---|---|---|
| Rules (tuned on these sets) | — | — | 0.91* |
| Laya untrained, relation choice | 0.57 | 0.60 | 0.40 |
| Laya untrained, yes/no question | 0.29 | 0.43 | 0.10 |
| NLI cross-encoder (deberta-v3-small), P(contradiction) | 0.09 | 0.18 | 0.16 |

\* Inflated: the rules were developed on these sets. On blind `v2` the same
rules reach F1 0.64. "Best F1" picks its threshold on the same data and is
optimistic.

"Replaces" is not "contradicts": the NLI model flags almost everything.
Question framing matters: a choice beats a yes/no question by 0.28 AP.

### 6.3 Candidate generation

A pairwise classifier can only retire facts it is shown. On the training
people, who include many chain reactions ("my dog died" ends the walks, the
vet visits and the dog's age; "got laid off" ends the job, the commute and the
work laptop):

*Table 5. Share of true replacements among the candidates.*

| Candidates | Reachable |
|---|---|
| top-5 similar + shared names | 77% |
| top-10 + shared names | 86% |
| top-15 + shared names | 93% |
| top-20 + shared names | 98% |

On the development sets, with few chain reactions, top-5 already reaches
69/70. Training uses top-10 plus shared names; the blind replay uses top-15
plus up to 12 facts sharing a name (110 of 112 reachable).

### 6.4 Training data

- **Generated people.** Two Sonnet agents, each given only a spec (no code, no
  test sets): writer A in varied prose, writer B in texting style
  (abbreviations, code-switching, two facts per message, reversals, plans not
  yet carried out, temporary stays labelled as events). Each person has 8–12
  changes, at least two chain reactions and at least six labelled traps.
  Files were validated structurally and labels spot-checked.
- **Rows.** For every statement, each candidate pair becomes a training row.
  Labelled replacements and traps get one-hot targets; every other candidate
  gets a soft target (`replaces` 0, the other four 0.25 each: "not a
  replacement, kind unknown"). 280 replacement rows and 145 trap rows; soft
  rows capped at twice the labelled ones (1,275 rows in total).
- **Dialogue NLI** [We19], in one variant: 150 training pairs per label
  (contradiction → `replaces`, entailment → `detail`, neutral → `other`).

### 6.5 Fine-tuning

Laya's published recipe, unchanged: reinforcement learning against proper
scoring rules plus cross-entropy; AdamW (encoder 2.5e-5, head 1e-4, weight
decay 0.01), cosine schedule, effective batch 32, 3 epochs, then temperature
calibration on a held-back 10%. One Kaggle T4 GPU, about 12 minutes per run.

*Table 6. Development shortlist (never trained on), threshold 0.5.*

| Model | AP | Precision | Recall | F1 |
|---|---|---|---|---|
| Laya untrained | 0.57 | 0.65 | 0.29 | 0.40 |
| + writers A, B + Dialogue NLI | 0.92 | 0.84 | 0.90 | 0.87 |
| **+ writers A, B** | **0.96** | **0.91** | 0.84 | **0.87** |

Dialogue NLI did not help; the variant without it was chosen on development
data, before any blind run.

### 6.6 Blind change detection

Blind set `v2`, replayed in date order as the application would run: each new
statement is compared with up to 15 most similar live facts plus up to 12 live
facts sharing a name; candidates with P(`replaces`) ≥ 0.5 are retired and
leave the live set. Totals only.

*Table 7. Change detection on blind `v2`.*

| Change detector | Changes noticed (95% CI) | Retired | Mistaken | Pair precision (95% CI) |
|---|---|---|---|---|
| Rules | 62/112 = 55% [46–64] | 81 | 19 | 77% [66–84] |
| Rules + 9B verifier | 73/112 = 65% [56–73] | 93 | 20 | 78% [69–86] |
| **Fine-tuned Laya, run 1** | **95/112 = 85% [77–90]** | 100 | **2** | **95% [89–98]** |
| Fine-tuned Laya, run 2 | 92/112 = 82% [74–88] | 101 | 2 | 91% [84–95] |
| Fine-tuned Laya, round 3 (§6.8) | 94/112 = 84% [76–90] | 100 | 2 | 94% [88–97] |
| Fine-tuned Laya, round 4 (§6.9) | 94/112 = 84% [76–90] | 99 | 1 | 95% [89–98] |

Run 1 vs rules, paired over the 112 labelled changes: 41 caught only by the
detector, 8 only by the rules (exact McNemar p = 2·10⁻⁶). Run 1 vs rules +
verifier (no per-pair outcomes were kept for the verifier run): changes
noticed z = 3.39, p = 0.0007; mistaken retirements 2/100 vs 20/93,
z = −4.26, p < 0.001 (two-proportion tests). Runs 2–4 repeat training on
fresh GPU sessions.

### 6.7 External tests: text from other authors

Both sets were scored once, at the same fixed threshold.

- **Dialogue NLI, verified test** [We19]: crowdworker-written persona facts;
  the shipped model never saw Dialogue NLI. Contradiction should replace;
  entailment (restatements) and neutral should not.
- **LongMemEval, knowledge-update sessions** [Wu25]: long, chatty user turns
  from GPT-4o-generated chats. Each earlier turn carrying an evidence flag is
  paired with each evidence turn of the latest session (72 changes) and with
  that session's other user turns (same topic, no change).

*Table 8. External tests.*

| Detector | DNLI contradictions caught | DNLI restatements called a change | DNLI F1 | LongMemEval changes caught | LongMemEval F1 |
|---|---|---|---|---|---|
| Rules | 170/1000 = 17% [15–19] | 148/1000 | 0.26 | 4/72 = 6% [2–13] | 0.09 |
| Laya untrained | 254/1000 = 25% [23–28] | 6/1000 | 0.40 | 0/72 = 0% [0–5] | 0.00 |
| Fine-tuned Laya, run 2 | **406/1000 = 41% [38–44]** | 170/1000 | **0.51** | **10/72 = 14% [8–24]** | **0.24** |

The fine-tuned model remains the best of the three, but the 85% on `v2` holds
for short, clean, single-fact statements. Two gaps are visible: restatements
are over-called (the training data has no "same fact, said differently"
examples), and changes buried in long multi-topic turns are mostly missed (the
training data has none).

### 6.8 A targeted fix (round 3)

The two gaps in §6.7 suggested two additions to the training data. A third
Sonnet writer (C), blind to the code and to every test set, wrote 12 more
people in long, chatty, multi-topic messages (mean ~245 characters): changes
buried mid-message, questions with no new fact, and 58 restatements labelled
as not replacing. The model was retrained on writers A, B and C (415
replacement rows, 236 trap rows) with the same recipe.

*Table 9. Before and after the targeted data. The external sets informed this
change (by their aggregate failure types, not by any item), so round-3
external scores are not blind.*

| | Run 2 (A, B) | Round 3 (A, B, C) |
|---|---|---|
| Blind `v2`: changes noticed / mistaken | 92/112 / 2 | 94/112 = 84% [76%–90%] / 2 |
| Development shortlist F1 @0.5 | 0.87 | 0.91 |
| DNLI F1 | 0.51 | 0.58 |
| DNLI restatements called a change | 170/1000 | **37/1000** |
| DNLI contradictions caught | 406/1000 | 427/1000 = 43% [40%–46%] |
| LongMemEval changes caught | 10/72 | 6/72 = 8% [4%–17%] |

Restatement errors fell 4.6-fold with no loss on the blind set. Changes buried
in long chat turns did not improve: examples of the style were not enough.
Long messages hold several facts, and retiring a whole message when one of
them changes is itself wrong; the remedy is to extract individual facts before
change detection.

### 6.9 Fact extraction for long messages (round 4)

Long chat turns hold several facts; comparing whole turns hides the change.
A rule-based splitter (`extract.py`, no language model) breaks a message into
sentences and keeps those about the user (first person) that are neither
questions nor requests to the assistant; a message pair is scored as the
highest score over its statement pairs (3.8 comparisons per pair on average
for LongMemEval). The model was retrained on writers A, B and C (same recipe
as round 3) and scored both ways. The splitter was designed after §6.7 showed
the long-message gap, and one LongMemEval input was read to check that it
splits correctly; these LongMemEval scores are therefore not blind.

*Table 10. Round 4: whole messages vs extracted statements.*

| | Whole messages | Extracted statements |
|---|---|---|
| LongMemEval changes caught | 6/72 = 8% [4–17] | **33/72 = 46% [35–57]** |
| LongMemEval precision / F1 | 0.86 / 0.15 | 0.66 / **0.54** |
| DNLI F1 (single sentences; little to split) | 0.51 | 0.48 |
| Blind `v2` (whole statements, unchanged) | 94/112, 1 mistaken | — |

Extraction multiplies the changes caught in long chat turns by 5.5 at some
cost in precision. Round 3 and round 4 used the same data and recipe; their
DNLI scores (F1 0.58 vs 0.51, restatements called a change 37 vs 60 per
1,000) show the run-to-run variation on that set.

### 6.10 Answer accuracy with the trained detector

Change detection matters only through answers. Each blind `v2` person was
loaded through the application's `remember()` exactly as the blind runner
does; the "replaced by" links were then set to (a) the rules' own decisions,
(b) the decisions of the fine-tuned detector (run 1, replayed as in §6.6), or
(c) the labels (perfect change detection, a ceiling), and every answerable
question was scored top-1 with oracle dates. Condition (a) reproduces the
blind runner's 59.1% exactly, so the procedure matches.

*Table 11. Top-1 on blind `v2` (171 answerable questions) by change decisions.*

| Change decisions | Top-1 (95% CI) | "Now" questions | Past | Multi |
|---|---|---|---|---|
| Rules (current system) | 101 = 59.1% [52–66] | 50/93 | 36/54 | 15/24 |
| Rules + 9B verifier (Table 3) | 107 = 62.6% [55–69] | 56/93 | 36/54 | 15/24 |
| **Fine-tuned detector** | **114 = 66.7% [59–73]** | **62/93** | 36/54 | 16/24 |
| Labels (ceiling) | 121 = 70.8% [64–77] | 69/93 | 36/54 | 16/24 |

Paired exact McNemar tests: detector vs rules on answers, 17 questions right
only with the detector and 4 only with the rules, p = 0.007; on changes, 41
caught only by the detector and 8 only by the rules, p = 2·10⁻⁶. Against the
rules + 9B verifier on answers: 12 vs 5, p = 0.14 (not significant). The
detector closes 13 of the 20 answers between the rules and perfect change
detection; the gain is almost entirely on "now" questions, as expected.

### 6.11 Escalation and calibration

The intended design keeps the small model as the agent and the 9B as a
supervisor for uncertain decisions. On the development shortlist (1,165
decisions, 69 changes; run 1), sending decisions with P(`replaces`) inside a
band to a supervisor that is assumed correct (a best case):

*Table 12. Escalation band vs errors left in the small model's own decisions.*

| Escalate when | Escalated | Errors kept (missed / wrong retirement) |
|---|---|---|
| never | 0% | 17 (11 / 6) |
| 0.3 < P < 0.7 | 2% | 9 (7 / 2) |
| 0.2 < P < 0.8 | 4% | 3 (3 / 0) |
| 0.1 < P < 0.9 | 8% | 1 (1 / 0) |
| 0.05 < P < 0.95 | 17% | 0 |

Expected calibration error is 0.047 (10 bins); the model is slightly
under-confident at the top (mean P 0.85 in its 0.8–0.9 bin, all correct).
Escalating 4% of decisions would remove about 80% of the errors, which
supports the agent / supervisor split. The development data come from the
same writer family as training, so these figures are optimistic.

### 6.12 Speed

*Table 13. Time per comparison.*

| Setting | Time |
|---|---|
| Untrained, two questions, one at a time, desktop CPU | 0.67–1.0 s |
| Fine-tuned, one question, batched, desktop CPU | 0.42 s |
| Fine-tuned, batched, Kaggle T4 GPU | 0.014 s |

The blind replay averaged 9.9 comparisons per statement: about 4 s per message
on a desktop CPU. Laya's own notes put int8 ONNX at about 2× faster with an
accuracy cost.

---

## 7. Experiment 3: a graph as the prompt index

Every chat currently carries all current facts (up to 200, ~3–4k tokens). We
link each fact to the people, pets, places, things, topics and rare content
words it mentions (rule-based, no LLM), seed a personalized PageRank walk with
the message's own names and topic and its 3 most similar facts (as in
HippoRAG [G24]), and send the top k. Baselines: similarity top-k, a
reciprocal-rank fusion of both ("hybrid"), and sending everything. Safety facts
are always included. For each current and multi-value question we check
whether the block contains every answer fact; live facts come from the labels,
so selection is measured alone. Tokens are estimated as characters / 4,
header included. To reach the application's 200-fact cap, each store is padded
with other people's non-safety facts — a harsh test, since the padding
includes strangers' "I live in…" facts. Developed on dev (64 questions);
scored once on `v2` (117 questions).

*Table 14. Prompt selection on blind `v2`.*

| Store | Method | Answer facts in block (95% CI) | Tokens |
|---|---|---|---|
| ~16 facts | Everything | 117/117 = 100% | 384 |
| | Similarity, top 8 | 113/117 = 97% [92–99] | 213 |
| | Graph, top 8 | 109/117 = 93% [87–96] | 214 |
| | Hybrid, top 8 | 112/117 = 96% [90–98] | 214 |
| 200 facts | Everything | 117/117 = 100% | 4,075 |
| | Similarity, top 20 | 92/117 = 79% [70–85] | 454 |
| | Graph, top 20 | 80/117 = 68% [59–76] | 467 |
| | Hybrid, top 20 | 92/117 = 79% [70–85] | 463 |

Selection cuts memory tokens by 45% (small stores) to 89% (200 facts). The
graph does not select better than similarity and is worse at 200 facts (the
intervals barely overlap). Likely reasons: the questions are paraphrased and
rarely name what they ask about; rule-based names and keywords link padding
facts through common words; and multi-hop questions, where a graph should
help, are rare. At 200 facts, selection also loses about one answer fact in
five, so a trimmed block needs the `recall_memory` tool as a fallback.

---

## 8. Discussion

**A small classifier beats hand-written rules on change detection.** It was
trained only on synthetic data, in minutes, and on the blind set it noticed
22 more changes than the best rule-based system while cutting wrong
retirements tenfold. On text from other authors its advantage shrinks a lot,
but it stays ahead of the rules on both external sets.

**Candidates matter as much as the classifier.** A top-5 similarity shortlist
reaches only 77% of chain-reaction changes. Shared names help a little; a
learned way to link facts about the same person or thing is the natural next
step.

**Change detection need not be inline.** It can run after the reply, as the
9B verifier already does, which makes the 4 s per message on a CPU much less
critical.

**For tokens, selection is what pays.** Trimming the memory block to the
top-ranked facts gives most of the saving; the rule-built graph adds nothing
on these questions.

---

## 9. Threats to validity

- **Same generator family.** Training people, development sets and `v2` were
  all written by Claude agents (different agents, separate specs). The model
  may partly learn these writers' phrasing; §6.7 shows the drop on other
  authors. The owner's real chats have not been tested.
- **Simulated integration.** Answer accuracy (§6.10) sets the detector's
  decisions after loading, rather than running it live as a background job;
  timing effects of a background queue are not measured.
- **Small samples.** 112 blind changes and 117 block questions: intervals are
  ±7–9 points.
- **Test reuse.** `v2` has been scored in 19 runs (9 retrieval configurations
  across six versions, five change-detection replays, two block tests, three
  answer runs including a reproduction of the rules). No
  per-item result was read and nothing was tuned on it, but a fresh blind set
  is needed for a final verdict.
- **Partly unpaired comparisons.** Detector vs rules is paired (changes and
  answers); detector vs rules + verifier is paired on answers but unpaired on
  changes, since that run kept no per-pair outcomes.
- **Different loading paths.** Rule runs load statements through the
  application's `remember()`, which can refuse or deduplicate; the model
  replay scores every statement.
- **Runs and model size.** Four training runs agree on blind `v2` (95, 92, 94
  and 94 of 112, 1–2 mistaken retirements); run-to-run variation is larger on
  DNLI (F1 0.51–0.58). Only the 421M English checkpoint was trained.
- **External sets reused.** §6.7 informed the round-3 data and the round-4
  splitter, so their external scores are not blind.
- **Token estimate.** Tokens are characters / 4, not a model tokenizer.
- **Language.** The rules are English-only; the external tests are English.

---

## 10. Conclusion and future work

A local memory without any LLM on the write path reached 63% top-1 on a
strict blind set; a small decision model trained in minutes on synthetic
people replaced its hand-written change rules and lifted top-1 answers to 67%
(ceiling 71%). Its gain shrinks on text from other authors; restatement
examples and rule-based fact extraction recovered much of the loss. Next:

1. Integrate the trained detector as a live background job, with the 9B
   supervising the ~4% least certain decisions, and score it on a fresh blind
   set written by a different model family, with chain reactions.
2. Better fact extraction: the rule-based splitter raised long-message recall
   from 8% to 46% at 0.66 precision; a learned extractor (a "worth keeping?"
   head) is the next step, and it is also what LoCoMo and LongMemEval need,
   since the memory deliberately refuses multi-fact text.
3. A smaller or faster student (~150M parameters, or one pass over all
   candidates), with the 9B model as supervisor for low-confidence and
   safety-critical retirements.
4. Ship similarity-based prompt selection with the recall tool as fallback;
   revisit graph indexing only with learned entity links and multi-hop
   questions.
5. Report on LoCoMo and LongMemEval to compare with other systems, once fact
   extraction exists (both are built from long raw chat turns).

---

## Acknowledgements

Experiments, code and drafting were carried out with the assistance of Claude
(Anthropic) models: Claude Opus for analysis and implementation, Claude Sonnet
agents for writing test and training data under the blind protocol in §4.
Fine-tuning ran on Kaggle's free GPU tier.

---

## References

- [B22] Borgeaud, S. et al. *Improving language models by retrieving from trillions of tokens* (RETRO). ICML 2022. arXiv:2112.04426.
- [C25] Chhikara, P. et al. *Mem0: Building production-ready AI agents with scalable long-term memory.* arXiv:2504.19413, 2025.
- [CX1] prem-research. *Cortex: advanced memory system for AI agents.* [github.com/prem-research/cortex](https://github.com/prem-research/cortex).
- [CX2] gambletan. *Cortex: local-first memory for AI agents.* [github.com/gambletan/cortex](https://github.com/gambletan/cortex).
- [G24] Gutiérrez, B. J. et al. *HippoRAG: Neurobiologically inspired long-term memory for large language models.* NeurIPS 2024. arXiv:2405.14831.
- [I22] Izacard, G. et al. *Atlas: Few-shot learning with retrieval augmented language models.* arXiv:2208.03299, 2022.
- [J26] Willison, S. *Jev introduces a new shape of LLM — System One, aka decision models.* 2026. [simonwillison.net](https://simonwillison.net/2026/Sep/21/jev/).
- [L25] Li, Z. et al. *MemOS: A memory OS for AI system.* arXiv:2507.03724, 2025. [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS).
- [LY] Convai Innovations. *Laya: open decision model.* [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).
- [M24] Maharana, A. et al. *Evaluating very long-term conversational memory of LLM agents* (LoCoMo). ACL 2024. arXiv:2402.17753.
- [Mc47] McNemar, Q. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12, 1947.
- [P23] Packer, C. et al. *MemGPT: Towards LLMs as operating systems.* arXiv:2310.08560, 2023.
- [R25] Rasmussen, P. et al. *Zep: A temporal knowledge graph architecture for agent memory.* arXiv:2501.13956, 2025.
- [RG19] Reimers, N., Gurevych, I. *Sentence-BERT.* EMNLP 2019. arXiv:1908.10084.
- [W25] Willison, S. *Comparing the memory implementations of Claude and ChatGPT.* 2025. [simonwillison.net](https://simonwillison.net/2025/Sep/12/claude-memory/).
- [Wa24] Warner, B. et al. *Smarter, better, faster, longer: a modern bidirectional encoder* (ModernBERT). arXiv:2412.13663, 2024.
- [We19] Welleck, S., Weston, J., Szlam, A., Cho, K. *Dialogue natural language inference.* ACL 2019. arXiv:1811.00671.
- [Wi27] Wilson, E. B. *Probable inference, the law of succession, and statistical inference.* JASA 22, 1927.
- [Wn20] Wang, W. et al. *MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers.* NeurIPS 2020. arXiv:2002.10957.
- [Wu25] Wu, D. et al. *LongMemEval: Benchmarking chat assistants on long-term interactive memory.* ICLR 2025. arXiv:2410.10813.

---

## Appendix A. Reproducing

| Step | Command / file |
|---|---|
| Blind retrieval runs (needs the Primnox backend) | `scripts/bench_memory_blind.py --split test --data scripts/blind_memory/v2` |
| All retrieval scores | `scripts/blind_memory/results/RESULTS.md` |
| Zero-shot screen | `system_one/screen.py --scorers nli laya` |
| Rules' pairs on development data | `system_one/rules_pairs.py --backend <primnox>/backend` |
| Training rows | `system_one/build_train.py`, `system_one/build_dnli.py` |
| Fine-tuning on Kaggle | `system_one/kaggle/train_kaggle.py` (`kaggle kernels push`) |
| Fine-tuning locally | `system_one/train_laya.py --device cpu` |
| Blind change detection | `system_one/blind_pairs.py --model <checkpoint> --tag <name>` |
| External pairs and scoring | `system_one/build_external.py`, `system_one/external_eval.py`, `system_one/rules_external.py` |
| Second run + external tests on Kaggle | `system_one/kaggle/eval_kaggle.py` |
| Fact extraction scoring | `system_one/external_eval.py --model <checkpoint> --tag <name> --sentences` |
| Answer accuracy under given decisions | `system_one/answers_with_decisions.py --backend <primnox>/backend --decisions rules` (or `labels`, or a `--save-pairs` file) |
| Paired tests | `system_one/paired.py --changes A B --answers A B` |
| Escalation and calibration | `system_one/escalation.py --scores <dev scores> --tag <name>` |
| Prompt selection | `graph/eval_block.py --split v2 --pad 200` |

Training data: `system_one/data/train_writer_a.json`, `train_writer_b.json`.
Results: `system_one/results/`, `graph/results/`. Checkpoints (~800 MB each)
are not in git.

## Appendix B. Question wording

Instruction: *"How does the later message relate to the earlier statement?"*

| Option | Description given to the model |
|---|---|
| `replaces` | the earlier statement is no longer true now; the later one changed or ended it |
| `adds` | both stay true: a second or additional value of the same kind (another pet, another language) |
| `detail` | both stay true: the later one adds detail to the same fact |
| `other` | about a different person, thing or topic |
| `event` | a one-off event or plan that does not change the earlier fact |

Yes/no variant (§6.2): *"Does the later message mean the earlier statement is
no longer true now (it was replaced, changed or ended)?"*
