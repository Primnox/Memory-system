# Primnox Memory System

The permanent memory behind [Primnox](https://github.com/Primnox/main), a
local-first AI assistant. It remembers what you tell it, notices when a fact
has changed ("I moved to Pune" retires "I live in Lisbon"), keeps the history,
and answers questions about the past ("where did I live back in March?").
Everything runs on your own machine.

This repo is the home of the memory work: the code, the blind test sets,
every score, and the plan for the next step — a small local decision model.

## Results

Measured on a blind test set: written by agents that never saw this code,
labels re-checked independently, frozen by SHA-256 before the first scored run,
and each version scored on it once. Top-1 = the first search result is a right
answer. 95% confidence intervals in brackets.

| Version | Answerable top-1 | "Now" questions | Changes noticed | Wrong retirements |
|---|---|---|---|---|
| Before | 19% [14–25] | 14/93 | 16/112 | 55 of 71 |
| Topics, slots, time (`exp/memory-v3`) | 34% [27–41] | 22/93 | 12/112 | 27 of 39 |
| + scale, lifecycle, safety, embeddings (`exp/memory-v4`) | 57% [50–64] | 47/93 | 54/112 | 28 of 82 |
| + change detection | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| + time phrases | 59% [52–66] | 50/93 | 62/112 | 19 of 81 |
| **+ background verifier (`exp/memory-change-verify`)** | **63% [55–69]** | **56/93** | **73/112** | **20 of 93** |

Time phrases show up when the local model writes the search itself: 48% → 53%,
and "back then" questions 24 → 34 of 54 (p=0.002). Full write-up, method, and
what did not work: [`docs/MEMORY_EXPERIMENT.md`](docs/MEMORY_EXPERIMENT.md).
Every score file: [`scripts/blind_memory/results/RESULTS.md`](scripts/blind_memory/results/RESULTS.md).

## What is in this repo

Paths mirror the Primnox repo, so files map one to one.

| Path | What |
|---|---|
| `backend/primnox2/memory/` | Storage, search, prompt block, embeddings, time phrases, verifier, directives |
| `backend/primnox2/cognition/conflict.py`, `topics.py` | Change detection: does a new fact replace an old one |
| `patches/core-integration.patch` | The memory changes to shared Primnox files (DB schema v12, tunables, tools, API) |
| `backend/tests/test_memory_*.py` | The memory test suite |
| `scripts/bench_memory_blind.py` | The blind-test runner (intervals, paired McNemar, query modes) |
| `scripts/blind_memory/` | The frozen test sets (`test.json`, `v2/test.json`), dev set, paraphrase variants, all results |
| `scripts/e2e_memory_chat.py` | Full-chat harness: the real chat stack and local model, a judge model grades answers (parked) |
| `docs/SYSTEM_ONE_PLAN.md` | Next step: a small local decision model (≤500M parameters) |
| `docs/DATASETS_AND_MODELS.md` | Open datasets, teacher models and starting models for it |
| `docs/HANDOFF.md` | How to pick the work up on another machine |

## Running it

The code runs inside the Primnox backend. The runnable version is branch
[`exp/memory-change-verify`](https://github.com/Primnox/main/tree/exp/memory-change-verify)
(commit `c013cad`) of `Primnox/main`:

```bash
git clone -b exp/memory-change-verify https://github.com/Primnox/main.git primnox
cd primnox/backend && python -m venv venv && venv/Scripts/pip install -r requirements.txt
venv/Scripts/python -m pytest tests -q -k memory
cd .. && backend/venv/Scripts/python scripts/bench_memory_blind.py --split dev
```

The verifier needs a local Ollama model (`--llm-verify`). Use the dev split
while working; the test splits are for final scored runs only.

## Status

Not merged into Primnox's `main` yet. Schema v12 is a one-way migration, so
back up `primnox.db` before running this branch against real data.

## Licence

Business Source License 1.1, same as Primnox — see [`LICENSE`](LICENSE).
