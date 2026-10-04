# Primnox Memory System

Long-term memory for a local-first AI assistant. It remembers what you tell it,
notices when a fact has changed ("I moved to Pune" retires "I live in Lisbon"),
keeps the history, and answers questions about the past ("where did I live
back in March?"). It runs entirely on your own machine: SQLite, a small
sentence encoder, and optionally a local model as a second opinion. No cloud.

It was built inside Primnox, a desktop assistant. The app is closed source;
this repo publishes the memory work in the open: the code, the blind test
sets, every score, and the plan for what comes next.

## In short

- **Fully local.** It keeps your exact words with dates and knows when
  something changed, without calling a big AI.
- **Blind tests:** the right fact comes up first 63% of the time (from 19%).
  A small model trained in 12 minutes now catches 85% of life changes (from
  65%) and wrongly retires 2 facts instead of 20.
- **Still early:** the small model is not in the app yet, the tests use
  AI-written conversations, and there is no head-to-head comparison with
  other memory systems yet.

**How it differs from other AI memory** (by design; accuracy not yet compared):

| | Keeps | Updates memory with | Local |
|---|---|---|---|
| ChatGPT | a summary of you, rewritten over time | the AI itself | no |
| Claude | old chats, searched when the AI decides to | nothing; it searches raw chats | no |
| Mem0 / Graphiti | facts extracted by an AI | a big AI on every message | yes |
| **Primnox** | **your exact words + dates; old facts kept, marked replaced** | **rules + a small model, no big AI** | **yes** |

![architecture](docs/architecture.png)

## Results

Measured on a blind test set: written by agents that never saw the code,
labels re-checked independently, frozen by SHA-256 before the first scored run,
and each version scored on it once. Top-1 = the first search result is a right
answer. 95% confidence intervals in brackets.

| Version | Answerable top-1 | "Now" questions | Changes noticed | Wrong retirements |
|---|---|---|---|---|
| Starting point | 19% [14–25] | 14/93 | 16/112 | 55 of 71 |
| Topics, slots, time | 34% [27–41] | 22/93 | 12/112 | 27 of 39 |
| + scale, lifecycle, safety, embeddings | 57% [50–64] | 47/93 | 54/112 | 28 of 82 |
| + change detection | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| + time phrases | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| **+ background verifier** | **63% [55–69]** | **56/93** | **73/112** | **20 of 93** |

Time phrases show up when the local model writes the search itself: 48% → 53%,
and "back then" questions 24 → 34 of 54 (p=0.002).

### What these numbers do and do not show

- The gains are real: 19% → 63% on a fresh set, p < 0.0001.
- 63% means the top result is still wrong for about 1 in 3 questions, and that
  is with the true date supplied; with a local 9B model writing the search,
  about 53%.
- About 1 in 5 retirements is a mistake.
- These are retrieval scores, not end-to-end answers, on synthetic,
  agent-written conversations (171 answerable questions).
- No comparison with other memory systems yet. Public benchmarks
  (LongMemEval, LoCoMo) are on the roadmap; until then there is no claim of
  being better than anything else.

### A small trained model for change detection (new)

A 421M-parameter decision model (Laya, fine-tuned on synthetic people in ~12
minutes on one free GPU) replaced the hand-written change rules in a blind
replay of the same test set:

| Change detector | Changes noticed | Wrongly retired facts |
|---|---|---|
| Rules | 62/112 (55%) | 19 |
| Rules + local 9B verifier | 73/112 (65%) | 20 |
| **Small trained model** | **95/112 (85%)** | **2** |

Not yet measured: answer accuracy with the model in the loop, real
(non-synthetic) chats, and CPU speed (~0.4 s per comparison).

Research write-up with method, statistics and limitations:
[`docs/RESEARCH.md`](docs/RESEARCH.md). Full experiment log:
[`docs/MEMORY_EXPERIMENT.md`](docs/MEMORY_EXPERIMENT.md). Every score file:
[`scripts/blind_memory/results/RESULTS.md`](scripts/blind_memory/results/RESULTS.md).

## How it works

- **Write path, no language model needed.** Each new fact is classified (topic, fact or
  one-off event) and compared with what is already stored. Rules plus sentence
  embeddings decide whether it replaces an older fact, is a second value
  (another pet), a refinement, or is about someone else. Replaced facts are
  kept with a link to their successor, not deleted.
- **Optional second opinion.** Pairs the rules are unsure about are queued and
  checked in the background by a local model, which must be at least 95% sure
  from its token probabilities. Every decision is logged.
- **Search.** Hybrid word and embedding ranking, time-aware: "back in March"
  is resolved to a date in code, and only facts true at that date are returned.
- **Prompt block.** Live facts go into the assistant's prompt, safety-critical
  ones (allergies, medications) first, so they never fall off the end.

## What is in this repo

| Path | What |
|---|---|
| `backend/primnox2/memory/` | Storage, search, prompt block, embeddings, time phrases, verifier, directives |
| `backend/primnox2/cognition/conflict.py`, `topics.py` | Change detection: does a new fact replace an old one |
| `patches/core-integration.patch` | How the memory plugs into the rest of the app (database schema, settings, tools, API) |
| `backend/tests/test_memory_*.py` | The memory test suite |
| `scripts/blind_memory/` | The frozen test sets, dev sets, paraphrase variants, and all results |
| `scripts/bench_memory_blind.py` | The blind-test runner (intervals, paired McNemar, query modes) |
| `scripts/e2e_memory_chat.py` | Full-chat harness: the assistant chats, a judge model grades the answers |
| `docs/RESEARCH.md` | Research write-up: problem, related systems, method, all results, limitations |
| `docs/architecture.png` | Architecture diagram |
| `docs/SYSTEM_ONE_PLAN.md` | The small local decision model (≤500M parameters): plan and status |
| `docs/DATASETS_AND_MODELS.md` | Open datasets, teacher models and starting models for it |
| `system_one/` | The small model: screen, training data, Kaggle training job, blind replay, results |

**Can I run it?** Not on its own yet. The memory modules import a few parts of
the Primnox app that are not published (database layer, settings, tool
registry), so the code and tests here are for reading. A standalone package is
the next milestone. The test sets and score files are usable by anyone: the
format is described in [`scripts/blind_memory/README.md`](scripts/blind_memory/README.md).

## Roadmap

1. Save facts straight from the user's messages, so memory does not depend on
   the chat model deciding to call `remember`.
2. A standalone package: `pip install`, `remember / search / forget` on SQLite.
3. Public benchmarks: LongMemEval and LoCoMo, for a comparison with other
   memory systems.
4. The small local decision model (≤500M parameters): change detection
   trained and blind-tested (above); next, wire it into the memory service,
   make it faster, then the "worth keeping?", context, privacy and
   computer-use heads — see [`docs/SYSTEM_ONE_PLAN.md`](docs/SYSTEM_ONE_PLAN.md).

## Licence

Business Source License 1.1 — see [`LICENSE`](LICENSE). Non-commercial use,
including personal, educational and research use, is permitted.
