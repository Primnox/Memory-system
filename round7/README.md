# Memory round 7 — new data, a messier world, and the first local extractors

Work of 2026-10-07/08. Everything here was trained on or tested with the Kaggle jobs in
`kaggle/`; numbers are copied from the logs in `results/`.

## Results

### The memory head, round 6 → round 7 (proof sets never trained on)

| Test | Round 6 | **Round 7** |
|---|---|---|
| Blind v2 **clean** — changes caught / wrong retirements | 103/112 · 0 | **104/112 · 0** (precision 99%) |
| Blind v2 **noisy** typing (`data/noise_eval/`, same labels) | 81/112 · 7 | **90/112 · 4** |
| Blind v2 **heavy** noise | 58/112 · 6 | **82/112 · 6** |
| LongMemEval KU, Gemini-extracted facts — F1 | 0.69 (P 0.70 · R 0.68) | **0.78** (P 0.77 · R 0.79) |
| LongMemEval KU, rule-split — F1 | 0.51 | 0.55 |
| LongMemEval KU, whole messages — F1 | 0.11 | 0.13 |
| Dialogue NLI test, 16,500 pairs — F1 | 0.49 | **0.61** |
| Dialogue NLI verified, 12,376 pairs — F1 | 0.57 | **0.69** |

Round 7 trained on 127 clean people (existing writers A–D, v3, the chat people's answer-key
facts, the mixed-language people; 1,970 updates) plus 381 noisy copies (`pipeline/noisy.py`,
seeds 101–103; the test copies use 11 and 12). 24,359 training items, 3 epochs, one T4.

### Gemini 3.1 Pro as the change detector, on the same exams (`results/gemini_3.1_pro_as_judge.txt`)

| Test | Round 7 (421M, local) | Gemini 3.1 Pro |
|---|---|---|
| Blind v2 clean / noisy / heavy | 104 / 90 / 82 of 112 | 112 / 112 / 112, 0 wrong |
| LongMemEval KU — F1 | 0.78 (with Gemini's facts) | 0.80 (P 0.71 · R 0.92, reading whole messages) |

Gemini sees each person's whole history at once; the head judges one shortlisted pair at a time.

### Extraction is the biggest lever

LongMemEval KU with the same head, only the pieces differ: whole messages 0.11 → rule split
0.51 → Gemini-extracted facts 0.69 (round 6). On **REALTALK** (10 real 21-day chats, 505
questions all inputs can answer; Primnox's own `remember()` and search, rules deciding updates;
`results/realtalk_e2e_rules_decisions.txt`):

| Memory holds | Right message at #1 | in top 3 |
|---|---|---|
| raw messages | 10% | 16% |
| rule-split | 9% | 15% |
| **Gemini facts** | **34%** | **50%** |

REALTALK itself has no licence, so its messages and anything derived from them are **not** in
this repo; `pipeline/realtalk_e2e.py` rebuilds the test from a local clone.

### Local fact extractors (first attempt)

QLoRA on `data/extractor/` (user messages of 20 audited chat people, 2 turns of context, →
the facts the audit kept; noisy copies in train; 4 people held out): Qwen 2.5 1.5B (36 min)
and 7B (2 h) on Kaggle T4s.

| | Held-out people: fact / no-fact right | LongMemEval: facts written | messages with none |
|---|---|---|---|
| Qwen 2.5 1.5B fine-tuned | 231/240 | 314 | 265 of 467 |
| Qwen 2.5 7B fine-tuned | 232/240 | 298 | 276 of 467 |
| Gemini 3.8 Flash (prompt) | — | 653 | 75 of 467 |

Training loss fell to 0.01–0.05 while held-out loss stayed 0.67–0.74 — likely memorising 20
people. Scored on LongMemEval KU (`kaggle/primnox-extract-score`), F1 at the fixed 0.5 threshold:

| Facts written by | Round 6 head | Round 7 head (P · R) |
|---|---|---|
| Gemini 3.8 Flash, prompt only | 0.69 | **0.78** (0.77 · 0.79) |
| **Qwen 2.5 7B, fine-tuned** | 0.65 | **0.67** (0.78 · 0.58) |
| Rule splitter (extract.py) | 0.51 | 0.55 (0.71 · 0.44) |
| Qwen 2.5 1.5B, fine-tuned | 0.42 | 0.41 (0.63 · 0.31) |
| Qwen 2.5 3B, prompt only | 0.37 | 0.38 (0.78 · 0.25) |

The fine-tuned 7B beats the rules and matches Gemini's precision; its recall is the gap (it
writes 298 facts for the 467 messages against Gemini's 653). The 1.5B is below the rules.
Next for the extractor: more varied training people, a weaker "no fact" prior, fewer epochs.

## Data

| File | What |
|---|---|
| `data/chat_people.json` | 24 people as chats with the assistant (facts in passing, ~half the user turns task-only) + the audited answer key (facts, updates, questions). Written by Gemini 3.1 Pro / 3.8 Flash (agy) from `specs/SPEC_CHAT.md`; 20 drawn at random by `pipeline/sampler.py`. |
| `data/multilang_people.json` | 21 people writing in mixed languages (Hinglish, Spanglish, Taglish, Arabizi, Pidgin, Singlish, …), short statement format. |
| `*_as_written.json` | the writers' original labels |
| `audits/` | every auditor's labels. Chat: Claude Opus + Claude Sonnet. Mixed-language: two separate Sonnet runs. Labels settled 2 of 3 with the writer; split votes go to `skip`. |
| `data/noise_eval/` | blind v2 with messy typing — identical labels and questions; test only |
| `data/extractor/` | the extractor's train / val examples |
| `data/lme_facts/` | each extractor's facts for LongMemEval's 467 messages |

Audit agreement: chat updates 92%, facts judged really stated 2,525/2,538, answer sets 93%;
mixed-language updates 98%, answer sets 98%. Every auditor's file access was checked from its
transcript (`pipeline/check_bounds.py`); every agy run's tool calls were checked
(`pipeline/long.py: strayed`).

## Rules kept

- LongMemEval, Dialogue NLI, blind v2 and REALTALK are proof sets: never trained or tuned on.
- v3 (a separate generated set, kept private) is training data, never scored.
- Noise for training and noise for testing use different seeds.
- Open-model writers (Qwen 2.5 14B, Mistral Nemo 12B) were tried and wrote 0 usable people.
