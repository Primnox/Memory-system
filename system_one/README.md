# system_one

Work on the small local decision model; plan in
[`../docs/SYSTEM_ONE_PLAN.md`](../docs/SYSTEM_ONE_PLAN.md), current step in
[`SPEC_memory_head.md`](SPEC_memory_head.md).

## Zero-shot screen (2026-10-04)

Dev pairs only (`dev.json` + `change.json`: 3,371 earlier/later pairs, 70 true
replacements); the frozen test sets are never read. Shortlist = the 5 most
similar earlier statements (69 of the 70 replacements are inside it).

| Scorer | Shortlist AP | Best F1 | Note |
|---|---|---|---|
| Current rules | — | 0.91 (P 0.93, R 0.90) | inflated: these dev sets were used to build the rules; on the blind `v2` set the same rules reach ~0.64 F1 (~0.71 with the verifier) |
| Laya, untrained, relation choice | 0.57 | 0.60 | ~0.7–1.0 s a pair on the dev PC's CPU |
| Laya, untrained, yes/no | 0.29 | 0.43 | the question's framing matters |
| NLI cross-encoder, P(contradiction) | 0.09 | 0.18 | "replaces" is not "contradicts" |

Best F1 picks its threshold on the same data, so it is optimistic. Full
output: [`results/screen_dev.log`](results/screen_dev.log).

## Fine-tuned (2026-10-04, Kaggle T4)

Laya trained with its own recipe on the generated people (writers A and B,
24 scenarios; ±150 Dialogue NLI pairs per label in one variant), 3 epochs,
~12 minutes per run. Scored on the same dev shortlist, which the model never
saw; threshold fixed at 0.5.

| Model | AP | P @0.5 | R @0.5 | F1 @0.5 |
|---|---|---|---|---|
| Laya, untrained | 0.57 | 0.65 | 0.29 | 0.40 |
| Laya + writers A, B + Dialogue NLI | 0.92 | 0.84 | 0.90 | 0.87 |
| **Laya + writers A, B** | **0.96** | **0.91** | 0.84 | **0.87** |

Dialogue NLI did not help. GPU scoring: 1,165 pairs in 16 s. On the dev PC's
CPU the fine-tuned model runs at ~0.43 s per comparison (batched), still too
slow for ~15 candidates per message. Logs and scores: `results/kaggle_v2/`.

Caveat: the training people and the dev sets were both written by Claude
agents, so part of the gain may be learning how those writers describe life
changes; only a blind set settles it (`blind_pairs.py`).

## Blind set (2026-10-04)

`v2/test.json` (12 people, 112 true changes), never used for training or
tuning; replayed in date order as the app would (up to 15 similar live facts
+ linked ones per new fact, fixed 0.5 threshold), scored by (old, successor)
pair exactly as the blind runner scores the rules. Scored once.

| Change detector | Changes noticed | Retired | Mistaken retirements | Pair precision |
|---|---|---|---|---|
| Rules (`fix/memory-change-detection`) | 62/112 (55%) | 81 | 19 | 77% |
| Rules + 9B verifier (`exp/memory-change-verify`) | 73/112 (65%) | 93 | 20 | 78% |
| **Fine-tuned Laya (writers A, B)** | **95/112 (85%)** | 100 | **2** | **95%** |

What this does not show yet: answer accuracy (the 63% top-1) — that needs the
head wired into the memory service. Speed on the dev PC's CPU: ~0.42 s per
comparison, ~10 comparisons per message (~4 s, fine as a background job, too
slow inline). Training data and test set were both written by Claude agents
(different ones, from separate specs), so real chats still need checking.
Totals: `results/blind_v2_v2-nodnli.json`.

## External blind tests (2026-10-04)

| Detector | Dialogue NLI F1 | Restatements called a change | LongMemEval changes caught | LongMemEval F1 |
|---|---|---|---|---|
| Rules | 0.26 | 148/1000 | 4/72 | 0.09 |
| Laya untrained | 0.40 | 6/1000 | 0/72 | 0.00 |
| Laya fine-tuned (second run) | 0.51 | 170/1000 | ~10/72 | 0.24 |

Second run on blind `v2`: 92/112, 2 mistaken. Files: `results/external_*.json`,
`results/blind_v2_seed2.json`; pairs in `data/blind_external/`
(`build_external.py`); rules via `rules_external.py`; Kaggle job
`kaggle/eval_kaggle.py`.

## Round 3: targeted data (2026-10-04)

Writer C added long chatty messages and 58 restatements. Retrained on A+B+C:
blind `v2` 94/112 with 2 mistaken; dev F1 0.91; Dialogue NLI F1 0.58 with
restatements called a change 37/1000 (was 170); LongMemEval 6/72 (was 10).
The external sets informed this data, so these external scores are not blind.
Files: `results/external_v3.json`, `results/blind_v2_v3.json`,
`results/screen_dev_v3.json`.

## Round 4 and answer accuracy (2026-10-04)

- Fact extraction (`extract.py`, rules): LongMemEval changes caught 6/72 →
  33/72 (F1 0.15 → 0.54). Blind `v2` 94/112 with 1 mistaken.
- Answer top-1 on blind `v2` with the detector's decisions
  (`answers_with_decisions.py`): rules 59.1%, rules + verifier 62.6%,
  detector **66.7%**, perfect change detection 70.8%. Paired vs rules:
  answers 17 vs 4 (p = 0.007), changes 41 vs 8 (p = 2e-6) (`paired.py`).
- Escalation: with a perfect supervisor the 4% least certain dev decisions
  hold 14 of 17 errors (`escalation.py`, ECE 0.047); with the real local 9B
  (`supervisor_check.py`) it is right on 80% of them and over-calls changes;
  overriding only when it is at least 95% sure gives 17 -> 12 errors.
- Candidate reach at 200 facts (`candidates_scale.py`): top-15 80%, + names
  and topic 84%, top-30 89%, top-50 93%.
- Event-time dating (`answers_with_decisions.py --event-time`): no effect on
  `v2`, whose labels date facts by mention.

## Files

| File | What |
|---|---|
| `screen.py` | zero-shot screen; also scores a fine-tuned checkpoint (`--scorers laya_ft --laya-ft DIR`) |
| `rules_pairs.py` | which pairs the current rules retire (runs in the Primnox backend environment) |
| `build_train.py` | generated scenarios → Laya training rows (top-10 similar + people-and-things links, soft labels) |
| `build_dnli.py` | Dialogue NLI (MIT) → capped extra rows |
| `train_laya.py` | fine-tunes Laya with its own recipe on our rows |
| `kaggle/train_kaggle.py` | the Kaggle GPU job: clone, train two variants, score on dev |
| `blind_pairs.py` | change detection on a blind set, replayed in order as the app would; totals only |
| `laya_finetune_upstream.py` | Laya's fine-tuning script, unchanged, from [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) |
| `data/train_writer_*.json` | generated training people (Sonnet writers that never saw the code or the test sets) |

Setup: `py -3.11 -m venv system_one/.venv` then
`system_one/.venv/Scripts/pip install laya scikit-learn datasets`.
