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
