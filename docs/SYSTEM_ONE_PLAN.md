# System One: a small local decision model

Status, 2026-10-04: **memory head (change detection) trained and
blind-tested.** Fine-tuned Laya (421M) noticed 95/112 changes with 2 mistaken
retirements on the blind set, against 73/112 and 20 for rules + the 9B
verifier (four training runs: 92–95/112, 1–2 mistaken). With its decisions in
the memory, top-1 answers rise from 59.1% to 66.7% (ceiling 70.8%; paired
p = 0.007 vs rules). On text from other authors the edge shrinks but stays
ahead of the rules; restatement examples cut "same fact said differently"
errors from 17% to 4%, and rule-based fact extraction raised changes caught in
long chat turns from 6 to 33 of 72. Not yet wired into the memory service;
~0.4 s per comparison on CPU. Details: [`RESEARCH.md`](RESEARCH.md) §6 and §9,
and [`../system_one/`](../system_one/).

## Idea

Primnox makes many small decisions that today go to rules or to the 9B chat
model: does this new fact replace an old one, is this message worth
remembering, which memories does this message need, is this text private,
which button should the agent click. These are typed decisions with a fixed
set of answers, not text generation. A small non-generative model that returns
a probability per answer can make them in milliseconds on a CPU.

Inspired by TypeSafe AI's Jev ("System One" decision model, cloud API) and the
open replicas that followed. Jev itself is cloud-only, which conflicts with a
local-first, privacy-first assistant, so we train our own.

## Constraints

- **Shipped model: 500M parameters or fewer.** Runs on CPU in about 1 GB of RAM
  (an 8 GB laptop is the target). Bigger open models can be used as teachers
  and are never shipped.
- **Small model is the agent, big model is the supervisor.** The small model
  decides; when its calibrated confidence is low, or the decision is
  safety-critical (retiring an allergy or medication fact, a destructive or
  risky computer action), it escalates to the local 9B. A random sample is
  audited too. Every disagreement is logged and becomes training data.
  *Measured 2026-10-04:* the 9B is a weak supervisor — right on 80% of the
  uncertain decisions and biased towards "replaces"; letting it override can
  make things worse. It should override only when its own probability is at
  least 0.95 (17 → 12 errors on dev). A stronger supervisor is worth more
  than a wider escalation band.
- **Saving must not depend on the chat model.** Today a fact reaches memory only
  if the chat model calls `remember`, in its own words. Every user message
  should pass through the small model ("worth keeping?") and the user's own
  words get stored.
- **Training runs off this machine.** Kaggle's free GPUs (about 30 h a week,
  runs continue with the laptop closed). The dev PC's AMD GPU on Windows is not
  used for training.

## Heads

One shared encoder, one small add-on per head (a few MB each), so training one
head cannot damage another.

| Head | Decisions | Order |
|---|---|---|
| Memory manager | worth keeping? · does A replace B / second value / other person / refinement / one-off event | 1st |
| Context splitter | which memories and chunks this message needs (embeddings shortlist ~20, the head scores them) | 2nd |
| Privacy mirror | which words are private (word tagging, a different add-on); the current rules stay as a backstop | 3rd |
| CUA controls | which accessibility-tree element, which action, did the step work, is it risky | 4th |

Not its job: writing replies, planning multi-step tasks, reading pixels, date
arithmetic (stays in `memory/when.py`). Those stay with the 9B or with code.

## Plan for the memory head

Done 2026-10-04: steps 1 (Laya screened: AP 0.57 untrained), 3 (two generated
writers + soft labels; Dialogue NLI tried and dropped), 4 (Laya only; trained
on Kaggle) and a blind change-detection replay ahead of step 6. Step 2 (Kev
teacher) skipped: training on generated labels was enough to clear the bar.
Also done: answer top-1 with the detector's decisions (66.7%), a real-9B
supervisor check, writer C (long messages, restatements) and rule-based fact
extraction. Open: step 5 (wiring, live), the speed work, a learned extractor
("worth keeping?" head), and step 6 on a fresh `v3` written by another model
family.

1. **Screen starting models, untrained**, on the dev set (`scripts/blind_memory/dev.json`):
   Laya (421M), Laya multilingual (322M), Open-Jev DeBERTa-v3-large (~400M),
   a ModernBERT around 150M. Cheap; tells us where each starts.
2. **Test the teacher on dev.** Kev-9B (Apache-2.0) on Kaggle. A student rarely
   beats its teacher, so if the teacher cannot beat the current system, stop
   here.
3. **Build training data** (never from the test sets):
   - contradiction data with human labels: MultiNLI, WANLI, Dialogue NLI;
   - unlabelled personal-fact pairs from PersonaChat / Multi-Session Chat and
     our generators, labelled by the teacher with probabilities;
   - a few hundred hand-checked examples of the traps public data lacks
     (second pet, a brother's job, "it's a 2019 Corolla", one-off events),
     written by several different writers so the model learns the task, not
     one writer's style.
4. **Train two students** on Kaggle and compare: Laya (best training kit) and a
   ~150M ModernBERT (smaller download, ONNX). Calibrate probabilities
   (temperature fit) after training.
5. **Wire it in** where rules and embeddings decide today; the 9B verifier
   becomes the supervisor for low-confidence and safety-critical calls. Tune the
   escalation threshold on dev from an accuracy-vs-escalation-rate curve.
6. **Score once** on a fresh blind set (write `v3` the same way `v2` was
   written). Report top-1 with 95% intervals, paired McNemar against the
   current best, and the escalation rate.

**Ship only if** it beats the current best (63% top-1 on `v2`, 73/112 changes
noticed) with no rise in wrong retirements, at an escalation rate low enough to
matter. A tie that needs the 9B far less often still saves time and compute,
and counts as a result worth reporting.

## Rules

- Never train or tune on `scripts/blind_memory/test.json`, `test_para*.json`
  or `v2/test.json`. Never read their per-question misses.
- The `memory_verifications` rows produced so far came from test-set runs, so
  they are off-limits for training. Only rows from real use (later) count.
- Check every dataset's licence before training weights that ship. Some are
  non-commercial: fine for evaluation, not for the product.
- Download size matters (the installer is being shrunk). Prefer int8 or ONNX
  exports; record the size of every candidate.

## Risks

- Training data quality decides everything; synthetic, template-made data
  scores well on templates and badly on real text (this happened with the old
  synthetic benchmark).
- Small models know less about rare things ("got a Pixel 9" replaces "my phone
  is an iPhone 13" needs knowing both are phones) — those go to the supervisor.
- Untrained open replicas trail Jev by 12–68 points on unfamiliar tasks; only
  fine-tuning on our own decisions makes them useful.

Models, datasets and sources: [`DATASETS_AND_MODELS.md`](DATASETS_AND_MODELS.md).
