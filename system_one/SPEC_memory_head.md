# Spec: the memory head (System One, step 1)

Status: 2026-10-04. Plan in [`../docs/SYSTEM_ONE_PLAN.md`](../docs/SYSTEM_ONE_PLAN.md).

## The decision

When a new fact arrives, for each candidate old fact: how does the new one
relate to it? One choice question, the framing that screened best zero-shot
(average precision 0.57 vs 0.29 for a yes/no question):

| Option | Meaning | Effect |
|---|---|---|
| `replaces` | the old fact is no longer true now; the new one changed or ended it | old fact retired, linked to the new one |
| `adds` | both stay true: a second value of the same kind (another pet) | none |
| `detail` | both stay true: the new one adds detail to the same fact | none |
| `other` | about a different person, thing or topic | none |
| `event` | a one-off event or plan that does not change the old fact | none |

The model returns a probability per option. P(`replaces`) drives the action:
above a high threshold, retire; in a middle band, escalate to the local 9B
supervisor; below, keep. Thresholds are tuned on dev from an
accuracy-vs-escalation curve. Safety facts (allergies, medications) are always
escalated before retirement.

State given to the model (JSON):
`{"earlier": {"date": "...", "text": "..."}, "later": {"date": "...", "text": "..."}}`

## Candidates: which old facts get asked about

1. The 5 most similar live facts by MiniLM (the screen found 69 of 70 true
   replacements inside this shortlist on dev).
2. Plus every live fact that shares a named person, pet, place or thing with
   the new one (people-and-things links). This is what catches chain
   reactions: "we broke up" reaches "Tom and I live together" and "weekend
   trips with Tom" even when they are not similar sentences.

**Measured 2026-10-04 (generated training data, 166 true replacements,
heavy in chain reactions):** share of replacements reachable from the
candidates — top-5 + links 77%, top-10 + links 86%, top-15 + links 93%,
top-20 + links 98%. On dev (few chain reactions) top-5 already reaches 69/70.
So in use the head must see ~15 candidates per new fact: at the ~0.7 s per
pair Laya measured on CPU that is ~10 s per message, too slow. Speed (ONNX /
int8 export, a smaller student, or one pass over all candidates) is a
requirement, not an optimisation. Training rows use top-10 + links.

Version 1 links entities with simple capitalised-name and possessive matching
("Tom", "my sister", "the Corolla"); a learned linker can replace it later.
Decisions stay pairwise; a set-level question ("which of these facts about
Tom are now false?") is version 2, only if pairwise misses chain reactions on
dev.

## Training data

Never from `test.json`, `test_para*.json`, `v2/test.json`. `dev.json` and
`change.json` are validation only (not trained on), so dev numbers stay
comparable to the zero-shot screen.

1. **Generated scenarios** (main source). New people, written by Sonnet agents
   that never saw the code or the test sets, in the blind-set schema plus a
   relation label for every trap. Several writers, varied styles. Each
   scenario: 20–30 dated first-person statements, 8–12 updates (explicit and
   implicit, some A→B→C chains), at least 2 chain reactions (one statement
   ending several facts), 6+ traps covering every non-replace option.
2. **Soft labels for unlabelled pairs.** A shortlisted pair that is neither a
   labelled update nor a labelled trap gets `replaces: 0` and the remaining
   probability spread evenly over the other four options: "not a replacement,
   kind unknown". Laya's training format takes probability targets directly.
3. **Teacher labels (later).** Kev-9B on Kaggle relabels the unlabelled pairs
   with real probabilities, replacing the even spread.
4. **Public contradiction data (if the licence allows).** Dialogue NLI persona
   pairs: contradiction → `replaces`, entailment → `detail`, neutral → `other`.

Format (one JSON line per state, Laya's own training format):
`{"state": "<json>", "questions": "<json>", "gold": "<json>"}` with
`gold = {"relation": {"probabilities": {"replaces": 1.0, ...}}}`.

## Training

Laya (`convaiinnovations/laya`, 421M, Apache-2.0) with its own recipe
(RLCD + cross-entropy, temperature calibration): the Kaggle 2×T4 notebook, or
the CPU/MPS script for a small pilot. A ~150M ModernBERT student is the second
candidate if Laya is too slow (screened at ~0.7 s a pair on the dev PC's CPU).

## Gates

1. **Dev (validation pairs):** average precision and best F1 on the shortlist
   must clearly beat the zero-shot 0.57 / 0.60. Report speed per pair.
2. **Blind set, once:** plugged into the memory service as the change
   detector, scored on a fresh blind set (`v3`, to be written the same way as
   `v2`, with chain reactions included). Ships only if it beats the current
   best (63% top-1 on `v2`, 73/112 changes noticed, 20 wrong retirements)
   with no rise in wrong retirements.
