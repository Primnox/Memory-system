# Datasets and models for System One

Researched 2026-10-04 for [`SYSTEM_ONE_PLAN.md`](SYSTEM_ONE_PLAN.md).
**Used so far:** Laya (fine-tuned, the memory head) and Dialogue NLI (tried as
extra training rows; it did not help and the shipped variant drops it). The
main training data is our own generated people (`system_one/data/`: writers
A, B and C, 36 people; C adds long chatty messages and restatements). Results:
[`RESEARCH.md`](RESEARCH.md). Sizes and
licences are as published on the linked pages at that date; "check" means the
licence was not confirmed yet. Confirm every licence before training weights
that ship.

## Starting models (shippable: ≤500M parameters)

| Model | Size | Base | Licence | Notes |
|---|---|---|---|---|
| [Laya](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`) | 421M | ModernBERT-large + decision head | Apache-2.0 | Best training kit: Kaggle 2×T4 notebook (~4–5 h, RLCD + calibration). CPU, ~1 GB RAM. Typed-decisions accuracy 36% untrained → 77% fine-tuned. Weak on prompt-injection detection in one benchmark. |
| Laya multilingual (`convaiinnovations/laya-multilingual`) | 322M | mmBERT-base | Apache-2.0 | 100+ languages. |
| [Open-Jev DeBERTa-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) | ~0.4B | DeBERTa-v3-large | Apache-2.0 | Well calibrated (ECE 0.022 in-domain), OOD accuracy 0.69. Trained on Banking77, SST-5, BoolQ only. Slow on CPU (~1.8 s for 4 questions, M1 Max, fp32). Code: `kotoba-lang/typed-decisions`. |
| [rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) ("OpenJev Verdict") | 151M | ModernBERT-base (GLiClass) | Apache-2.0 | ONNX export included, ~35 ms p50. No fine-tuning guide; one benchmark found near-constant probabilities on unfamiliar tasks. |

A directory of ~73 local replicas, including more 151M–310M ModernBERT
variants not yet reviewed: [Jevland](https://jevland.mnimiy.com/collections/local-replica-models/).

## Teachers and supervisors (not shipped)

| Model | Size | Licence | Use |
|---|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev) 0.8B / 4B / 9B / 27B | 0.8–27B | Apache-2.0 (27B: check) | Kev-9B is the main teacher: within a few points of Jev on its out-of-domain suite. Training code public. Runs on Kaggle's free GPUs. |
| [Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | 9B | check | Candidate supervisor with calibrated yes/no probabilities. |
| [openjev](https://huggingface.co/openjev) 27B | 27B | CC BY-NC 4.0 | **Do not use** as a teacher for a commercial model. |
| Jev (TypeSafe AI) | — | cloud API | Not used: cloud-only, terms likely forbid distillation. |

## Datasets by head

**Memory manager**

| Dataset | Size | Use | Licence |
|---|---|---|---|
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | 412k | contradiction / neutral / entailment, human labels | check |
| WANLI (in [Open-Jev v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1)) | 101k | contradiction, human labels | check |
| [Dialogue NLI](https://huggingface.co/datasets/xksteven/dialogue_nli) (Welleck et al., 2019) | 310k | personal-fact pairs labelled contradict / neutral / entail; tried, no gain | MIT |
| PersonaChat, Multi-Session Chat | — | personal facts in chats: "worth keeping?" and unlabelled pairs for the teacher | check |
| Open-Jev `citation-control-v1` | 4k | supported / contradicted / insufficient | CC0 |

**Context splitter**

| Dataset | Size | Use | Licence |
|---|---|---|---|
| Open-Jev `ir-control-v1` | 11.6k | passage relevance, four levels | CC0 |
| Open-Jev `context-retention-control-v1` | 9.8k | which context to keep across tool calls | CC0 |
| [jev-rag-benchmark](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) | — | evaluation only | check |

**CUA controls**

| Dataset | Size | Use | Licence |
|---|---|---|---|
| Open-Jev `browser-drone-expansion-v1` | 160.8k | simulated browser state → action | CC0 |
| Open-Jev `silent-failure-control-v1` | 9.6k | did the step actually work | CC0 |
| [JevForge-Mind2Web](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) | 7k | real-site element selection, evaluation | check |

**Privacy mirror:** nothing suitable in the Jev collections; needs a separate
personal-information tagging dataset.

**General decision warm-up**

| Dataset | Size | Note |
|---|---|---|
| [procedural-typed-decisions](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) | 572k | labels computed from rules in the input, no noise |
| [tasksource-jev-typed-decisions](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) | 3.56M | classification / multiple choice recast as decisions |
| [jev-bench](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data) | 166k | 22 public sets recast, human labels |

## Evaluation only (never train on these)

Used 2026-10-04 as external blind tests (`system_one/data/blind_external/`):
Dialogue NLI verified test (3,000 sampled pairs, MIT) and LongMemEval
knowledge-update sessions (421 pairs, [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval), MIT).

- Our blind sets: `scripts/blind_memory/test.json`, `v2/test.json`, and a fresh
  `v3` to be written for the final verdict.
- LongMemEval (has a "knowledge update" category) and LoCoMo — the benchmarks
  Mem0, Zep and others publish scores on, so they double as the competitor
  comparison. Both test the full answer pipeline, not just retrieval.

## Cautions

- Open-Jev's test and OOD labels are public: training on those configs means
  we cannot compare against Open-Jev's own numbers on them.
- Template-generated sets (e.g. large distillation corpora) teach templates;
  our blind sets stay the only judge.
- Kaggle runs are a few hours each: train on slices, not all ~4M rows.

## Sources

- [Jev-style decision models: open data (Hugging Face collection)](https://huggingface.co/collections/davanstrien/jev-style-decision-models-open-data)
- [ZefanCai/Open-Jev dataset](https://huggingface.co/datasets/ZefanCai/Open-Jev) · [project page](https://zefan-cai.github.io/open-jev/) · [GitHub](https://github.com/Zefan-Cai/Open-Jev)
- [Laya on GitHub](https://github.com/NandhaKishorM/laya) · [What is Laya](https://laya.studio/learn/what-is-laya) · [The Menon Lab write-up](https://themenonlab.blog/blog/laya-local-system-1-decision-model)
- [Kev on GitHub](https://github.com/jaredpalmer/kev) · [kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b)
- [LangWatch: Jev vs tiny open models](https://langwatch.ai/compare/jev-benchmark)
- [Open-weights Jev alternatives](https://rohitraj.tech/notes/jev-alternatives-open-weights-decision-models-2026)
- [Jevland: local & replica models](https://jevland.mnimiy.com/collections/local-replica-models/)
