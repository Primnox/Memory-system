# Local long-term memory for AI assistants: blind evaluation, and a small trained model for change detection

Research notes, Primnox Memory System. Work done 2026-10-01 to 2026-10-04.

## Abstract

Personal AI assistants need memory that keeps details, knows when a fact about
the user has changed, and can answer questions about the past. We describe a
fully local memory system (SQLite, a small sentence encoder, no language model
on the write path) and evaluate it on blind test sets written by agents that
never saw the code. Rule and embedding improvements raised the share of
questions whose first search result is correct from 19% to 63% [55–69] on a
fresh blind set. We then trained a small non-generative decision model (Laya,
421M parameters) to decide whether a new statement replaces an old one. On the
same blind set it noticed 95 of 112 changes (85% [77–90]) with 2 mistaken
retirements, against 73 of 112 (65% [56–73]) with 20 mistaken for the best
rule-based system with a local 9B verifier (recall p < 0.001, mistaken
retirements p < 0.001). The model was trained only on synthetic people from
two independent generator agents, in about 12 minutes on one free cloud GPU; a
second training run gave 92/112. On data not written by Claude agents the
advantage shrinks: F1 0.51 on crowdworker-written facts (rules 0.26) and 0.24
on long chatty messages from LongMemEval (rules 0.09). Answer-level accuracy
with the model in the loop, messy input and CPU speed (about 0.4 s per
comparison) remain open.

## 1. Problem

A memory for an assistant has two jobs: return the right fact, and notice when
a fact stops being true. "I moved to Pune" must retire "I live in Lisbon"
while keeping it for "where did I live in March?"; "my sister lives in Lisbon"
must retire nothing. The decisions that matter are typed and small: a change,
a second value (another pet), more detail, someone else, a one-off event.

Two failure modes recur in deployed systems: summaries rewritten over time
lose detail; and similarity search returns outdated and current facts side by
side. A third concern is cost: several systems call a large language model on
every message to maintain memory.

## 2. Related systems

From each project's published design; not measured here.

| System | What it keeps | Who decides updates | When a fact changes | LLM calls per write | Local |
|---|---|---|---|---|---|
| Vector RAG | text chunks | nobody | nothing; old and new both kept as equal | 0 | yes |
| ChatGPT memory | short facts + a compressed profile | the chat model; background rewriting | the profile is rewritten | 1+ | no |
| Claude memory | raw past chats | nothing extracted; tool search on demand | left to the model reading old chats | 0 | no |
| Letta (MemGPT) | self-edited memory blocks + archive | the main model, via tools | the model edits its note | main-model turns | yes |
| Mem0 | LLM-extracted facts (+ graph) | an extraction LLM, then an update LLM | update or delete (change log kept) | ~2 | yes |
| Graphiti / Zep | raw episodes + temporal knowledge graph | several LLM calls | fact invalidated with an end date, kept | several | yes |
| HippoRAG | documents + a graph used as index | LLM extraction, once | not handled (built for documents) | many, once | yes |
| **This work** | the user's words + date + topic; "replaced by" links | rules + embeddings; now a small decision model | retired, linked, kept with dates | **0** | **yes** |

Closest relative: Graphiti (dated facts, invalidation instead of deletion). The
difference studied here is who makes the change decision: a large LLM per
message, or rules / a small calibrated classifier.

The small model follows the "System One" idea of TypeSafe AI's Jev (a hosted,
non-generative decision model) and its open replicas; we use Laya (Convai
Innovations, Apache-2.0), a ModernBERT-large encoder with a decision head that
returns a probability per typed option.

## 3. System

![architecture](architecture.png)

- **Write path (no LLM).** Each statement is filtered (junk, length,
  instructions vs facts), labelled with a topic and as a lasting fact or a
  one-off event, matched against related live facts (same topic plus MiniLM
  sentence-embedding similarity), and a change decision is made: change,
  second value, someone else, detail, one-off, duplicate.
- **Storage.** One SQLite file. A retired fact is kept with a link to its
  successor; nothing is deleted or rewritten.
- **Read path.** Every prompt carries a memory block (safety facts such as
  allergies first and uncapped, then current facts up to 200). On demand,
  `recall_memory` runs hybrid word + embedding search; time phrases ("back in
  March", "last summer") are resolved to dates in code and only facts true at
  that date are returned.
- **Optional verifier.** Pairs the rules are unsure about are queued and
  checked in the background by a local 9B model, which must give
  P(replaces) ≥ 0.95 from its token probabilities.

## 4. Evaluation method

The work started from a result that did not hold: development scores of 100%
on benchmarks whose misses had been used to tune the rules; the first blind
test of the same code gave 34%. Everything since follows one protocol.

- **Blind authorship.** Test sets are written from a spec by Claude Sonnet
  agents that never saw the code: dated first-person statements per invented
  person, with `replaces` labels, traps that must not replace (second values,
  refinements, other people, one-off events), and questions (current, past
  with a time phrase, multi-value, absent).
- **Independent labels.** A second agent re-derives all labels from a
  label-free copy before any scored run (agreement 93/93 and 166/168 on the
  first set; 111/112 and 195/195 on the second).
- **Freeze, then score once.** Sets are frozen by SHA-256. Per-question misses
  on test sets are never read (the runner refuses). Each version is scored
  once per configuration.
- **Statistics.** 95% Wilson intervals; paired exact McNemar where per-item
  outcomes for both systems exist; otherwise a two-proportion z-test
  (unpaired, which is conservative for a paired design).
- **Sets.** `test.json` (10 people, 267 statements, 168 questions; used about
  ten times to choose between branches, so no longer fully blind) and
  `v2/test.json` (12 people, 309 statements, 112 true changes, 195 questions,
  171 answerable; written after all rule tuning). `dev.json` and `change.json`
  are development sets: misses read, rules tuned on them.

Retrieval is scored top-1 (the first result is a right answer), with the true
date supplied for "back then" questions (`oracle`) or with the local 9B model
writing the query and date (`model`). Change detection is scored by
(old, successor) pair; a mistaken retirement is a retired fact the labels never
replace.

## 5. Experiment 1: rules, embeddings, time, verifier

Blind set `v2`, answerable top-1, oracle dates:

| Version | Top-1 (95% CI) | Changes noticed | Mistaken retirements |
|---|---|---|---|
| Starting point | 19% [14–25] | 16/112 | 55 of 71 |
| Topics, slots, time | 34% [27–41] | 12/112 | 27 of 39 |
| + scale, lifecycle, safety, embeddings | 57% [50–64] | 54/112 | 28 of 82 |
| + change detection | 59% [52–66] | 62/112 | 19 of 81 |
| + time phrases | 59% [52–66] | 62/112 | 19 of 81 |
| + 9B background verifier | **63% [55–69]** | 73/112 | 20 of 93 |

- Paired McNemar: change detection vs start p < 0.0001; vs v3 p < 0.0001; vs
  v4 p = 0.55 (its gain is in change quality, not yet in answers); verifier vs
  time phrases p = 0.03, no question lost.
- Time phrases matter when the local model writes the search: 48% → 53%,
  "back then" questions 24 → 34 of 54 (p = 0.002).
- Rewording every statement and question (two paraphrase variants) flips
  about 15% of individual outcomes with no direction: totals are stable,
  single items are wording-sensitive.
- One branch (robustness markers) scored 32/32 on practice cases and worse on
  the blind set; it was not merged.

## 6. Experiment 2: a small decision model for change detection

### 6.1 Task

For a new statement and each candidate earlier statement, one typed choice:
`replaces` / `adds` / `detail` / `other` / `event`. The model sees
`Earlier (date) the user said: "…" / Later (date) the user said: "…"` and
returns a probability per option; P(`replaces`) ≥ 0.5 retires the old fact.

### 6.2 Zero-shot screen (development pairs)

3,371 earlier/later pairs from the development sets (70 true replacements);
shortlist = 5 most similar earlier statements (1,165 pairs, 69 replacements).

| Scorer | AP | Best F1 | F1 @0.5 |
|---|---|---|---|
| Rules (tuned on these sets; inflated) | — | — | 0.91 |
| Laya untrained, relation choice | 0.57 | 0.60 | 0.40 |
| Laya untrained, yes/no question | 0.29 | 0.43 | 0.10 |
| NLI cross-encoder (deberta-v3-small), P(contradiction) | 0.09 | 0.18 | 0.16 |

"Replaces" is not "contradicts": an off-the-shelf NLI model flags almost
everything. Question framing matters (choice beats yes/no by 0.28 AP).

### 6.3 Candidate generation

On generated data heavy in chain reactions ("my dog died" ending the walks,
the vet visits and the dog's age; "got laid off" ending job, commute and
laptop), the share of true replacements reachable from the candidates:

| Candidates | Reachable |
|---|---|
| top-5 similar + shared names | 77% |
| top-10 + shared names | 86% |
| top-15 + shared names | 93% |
| top-20 + shared names | 98% |

Pairwise decisions over a top-5 shortlist cannot catch a quarter of these
changes, whatever the classifier. Training used top-10 plus links; the blind
replay used top-15 plus up to 12 linked facts (110 of 112 reachable).

### 6.4 Training data

- **Generated people.** Two Sonnet agents, each given only a spec (no code, no
  test sets): writer A, 12 people, 350 statements in varied prose; writer B,
  12 people, 327 messages in texting style (abbreviations, code-switching,
  two facts per message, reversals, plans that are not yet changes, temporary
  stays labelled as events). Each person has 8–12 changes, at least two chain
  reactions and six labelled traps. Validated structurally; labels
  spot-checked.
- **Rows.** For every statement, candidates = top-10 similar earlier
  statements + earlier statements sharing a name. Labelled replacements and
  traps get one-hot targets; every other candidate gets a soft target
  (`replaces` 0, the other four 0.25 each: "not a replacement, kind unknown").
  280 replacement rows, 145 trap rows, soft rows capped at 2× the labelled
  ones.
- **Dialogue NLI** (Welleck et al. 2019, MIT): in one variant, 150 pairs per
  label (contradiction → `replaces`, entailment → `detail`, neutral →
  `other`).
- The development sets were never trained on, and no test set was read.

### 6.5 Fine-tuning

Laya's own recipe, unchanged: reinforcement learning against proper scoring
rules plus cross-entropy, AdamW (encoder 2.5e-5, head 1e-4), cosine schedule,
effective batch 32, 3 epochs, then temperature calibration on a held-back 10%.
One Kaggle T4, about 12 minutes per run, one run per variant (no seed
variance measured).

Development shortlist (never seen in training), threshold fixed at 0.5:

| Model | AP | Precision | Recall | F1 |
|---|---|---|---|---|
| Laya untrained | 0.57 | 0.65 | 0.29 | 0.40 |
| + writers A, B + Dialogue NLI | 0.92 | 0.84 | 0.90 | 0.87 |
| **+ writers A, B** | **0.96** | **0.91** | 0.84 | **0.87** |

Dialogue NLI did not help; the variant without it was chosen on development
data before the blind run.

### 6.6 Blind change detection

`v2/test.json`, replayed in date order as the application would: each new
statement compared with up to 15 most similar live facts plus up to 12 live
facts sharing a name; candidates with P(`replaces`) ≥ 0.5 retired and removed
from the live set. Scored once, totals only.

| Change detector | Changes noticed (95% CI) | Retired | Mistaken | Pair precision (95% CI) |
|---|---|---|---|---|
| Rules | 62/112 = 55% [46–64] | 81 | 19 | 77% [66–84] |
| Rules + 9B verifier | 73/112 = 65% [56–73] | 93 | 20 | 78% [69–86] |
| **Fine-tuned Laya** | **95/112 = 85% [77–90]** | 100 | **2** | **95% [89–98]** |

Fine-tuned vs rules + verifier: recall z = 3.39, p = 0.0007; mistaken
retirements 2/100 vs 20/93, z = −4.26, p < 0.001 (two-proportion tests,
unpaired). Against rules alone: recall z = 4.82, p < 0.001.

### 6.7 Second training run

Same recipe, fresh run on a Kaggle T4: blind `v2` replay 92/112 changes
noticed (82%), 101 retired, 2 mistaken (first run: 95/112, 100, 2). The blind
result is not a lucky seed.

### 6.8 External blind tests (data not written by Claude agents)

Two sets, fixed 0.5 threshold, scored once:

- **Dialogue NLI, verified test** (crowdworker-written persona facts; the
  shipped head never saw Dialogue NLI): 1,000 pairs per label; contradiction
  should replace, entailment and neutral should not.
- **LongMemEval, knowledge-update sessions** (GPT-4o-generated chats, long and
  chatty user turns): 421 pairs, 72 true changes buried in multi-topic
  messages; the other pairs are same-session, same-topic turns with no change.

| Detector | DNLI F1 | DNLI restatements called a change | LongMemEval changes caught | LongMemEval F1 |
|---|---|---|---|---|
| Rules | 0.26 (P 0.53, R 0.17) | 148/1000 | 4/72 (P 0.31) | 0.09 |
| Laya untrained | 0.40 (P 0.97, R 0.25) | 6/1000 | 0/72 | 0.00 |
| Laya fine-tuned (second run) | **0.51** (P 0.70, R 0.41) | 170/1000 | **~10/72** (P 0.77) | **0.24** |

The fine-tuned head is the best of the three on outside data too, but all
three are weak on long, messy messages: the 85% on `v2` holds for short,
clean, single-fact statements (which is what the app stores today, because the
chat model writes the memory). Two concrete gaps: no "same fact, said
differently" examples in training (restatements are over-called), and no long
multi-topic messages (saving straight from raw user messages would need fact
extraction first, or messier training data).

### 6.9 Speed

| Setting | Time per comparison |
|---|---|
| Untrained Laya, two questions, one at a time, dev PC CPU | 0.67–1.0 s |
| Fine-tuned, one question, batched, dev PC CPU | 0.42 s |
| Fine-tuned, batched, Kaggle T4 GPU | 0.014 s |

The blind replay averaged 9.9 comparisons per statement: about 4 s per message
on the dev PC's CPU. Laya's own notes put int8 ONNX at about 2× faster with a
real accuracy cost, so a smaller student or a background queue is the path.

## 7. Discussion

A small trained classifier replaced hand-written change rules and beat them
clearly on data neither was tuned on, while cutting wrong retirements tenfold.
Two findings shape what comes next. First, the classifier is not the only
bottleneck: how candidates are chosen decides what can be caught at all, and
chain reactions need more than a similarity shortlist. Second, the decision
does not have to be inline: change detection can run after the reply, as the
9B verifier already does, which makes CPU latency much less critical.

## 8. Limitations

- **Same model family, and messy text.** Training people, development sets
  and `v2` were all written by Claude agents. On outside data (§6.8) the gain
  shrinks sharply: F1 0.51 on crowdworker-written facts and 0.24 on long chat
  messages, though still above the rules (0.26 and 0.09). The owner's real
  chats have not been tested.
- **Change detection only.** Answer accuracy (top-1) with the model in the
  loop has not been measured; it needs integration into the memory service.
- **Small sets.** 112 blind changes; intervals are wide (±7 points).
- **Test reuse.** `v2` has now been scored once per version for seven
  versions; nothing was tuned on it, but a fresh `v3` is planned for the final
  verdict.
- **Unpaired tests.** Per-pair outcomes were not kept for both systems, so the
  comparison uses unpaired tests (conservative).
- **Slightly different loading.** Rule runs load statements through the app's
  `remember()` (which can refuse or deduplicate); the model replay scores
  every statement.
- **One seed, one size.** No variance across training seeds; only the 421M
  English checkpoint was trained.
- **Cost of the model.** About 800 MB on disk; ~0.4 s per comparison on CPU.
- **English only** for the rules; the training people include code-switching
  but the evaluation does not test other languages.

## 9. Next

1. Wire the head into the memory service as a background job; score answer
   top-1 on the blind set.
2. A smaller or faster student (~150M ModernBERT, or one pass over all
   candidates); keep the 9B as supervisor for low-confidence and
   safety-critical retirements.
3. Real-chat check (with consent) and a fresh blind set `v3` with chain
   reactions, written by a different model family.
4. Saving straight from user messages ("worth keeping?" head) and context
   selection, then public benchmarks (LongMemEval, LoCoMo) for comparison
   with other systems.

## 10. Reproducing

| Step | Command / file |
|---|---|
| Blind retrieval runs (needs the Primnox backend) | `scripts/bench_memory_blind.py --split test --data scripts/blind_memory/v2` |
| All retrieval scores | `scripts/blind_memory/results/RESULTS.md` |
| Zero-shot screen | `system_one/screen.py --scorers nli laya` |
| Rules' pairs on development data | `system_one/rules_pairs.py --backend <primnox>/backend` |
| Training rows | `system_one/build_train.py`, `system_one/build_dnli.py` |
| Fine-tuning (Kaggle) | `system_one/kaggle/train_kaggle.py` (pushed with `kaggle kernels push`) |
| Fine-tuning (local) | `system_one/train_laya.py --device cpu` |
| Blind change detection | `system_one/blind_pairs.py --model <checkpoint> --tag <name>` |
| Results | `system_one/results/` (screen, Kaggle logs and scores, blind totals) |

Training data: `system_one/data/train_writer_a.json`, `train_writer_b.json`.
Checkpoints are not in git (about 800 MB each).

## References

- Laya: [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) · [What is Laya](https://laya.studio/learn/what-is-laya)
- Open Jev-style data and models: [Hugging Face collection](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) · [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) · [Kev](https://github.com/jaredpalmer/kev)
- Dialogue NLI: Welleck, Weston, Szlam, Cho (2019), *Dialogue Natural Language Inference* · [data](https://huggingface.co/datasets/xksteven/dialogue_nli)
- Graphiti: [github.com/getzep/graphiti](https://github.com/getzep/graphiti) · Mem0: [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0) · Letta: [github.com/letta-ai/letta](https://github.com/letta-ai/letta) · HippoRAG: [github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- Memory designs of ChatGPT and Claude: [Simon Willison, 2025](https://simonwillison.net/2025/Sep/12/claude-memory/)
- Benchmarks: LongMemEval (Wu et al.), LoCoMo (Maharana et al. 2024)
- Retrieval-augmented smaller models: RETRO (Borgeaud et al. 2022), Atlas (Izacard et al. 2022)
