# HANDOFF — Primnox memory experiment (HISTORICAL)

> Superseded 2026-10-02. The work below was finished, measured on two blind sets and merged on
> `fix/memory-change-detection`; see docs/MEMORY_EXPERIMENT.md for the results. Kept for history.

Written 2026-10-01 by Claude (Opus 5.5) for another Claude session continuing this
work on a different machine. There is no shared memory: everything needed is here.

Branch: `exp/memory-v3` (created from `fix/alpha-2.0.0-bughunt`).
**State: ALL CHANGES UNCOMMITTED** at time of writing. If you are reading this on
another machine, the user must have committed + pushed the branch first. If the
files listed below are missing, ask the user to push `exp/memory-v3`.

---

## 1. What the user wants

Make Primnox's permanent memory measurably better and *prove* it honestly. The user
knows ML and caught that my reported "100%" scores were dev-set scores (I tuned the
rules by looking at the same benchmark's misses). The current task is:

> Build a **blind** test set (written by someone/something that never saw the code,
> frozen before the first run), never look at its misses to tune, use realistic
> queries (a model writes the search query and infers the date), and report 95%
> confidence intervals, not point scores. Then run it.

User tone: casual, blunt, wants honesty over hype. Don't claim results you can't back.

---

## 2. Architecture (only what matters here)

- Current backend: `backend/primnox2/`. Permanent memory = `memory/service.py`
  (`remember`, `import_many`, `search`, `render_for_prompt`), table `memories` in
  `primnox.db`. `render_for_prompt` injects ≤200 live, non-superseded memories into
  every prompt; `recall_memory` tool (`tools/builtins.py`) calls `search()`.
- Conflict engine: `cognition/conflict.py::resolve()` decides if a new memory
  supersedes old ones (`superseded_by` column). Superseded rows leave the prompt.
- Older substrate `backend/v2/` (world_model, episodes, task_state, result_store)
  exists; not touched by this work.
- Synthetic benchmark data generator: `backend/sdl/` (packs memory-10, memory-100,
  office-500; `generate.py`, `queries.json`, `ground_truth.json`, `memory.jsonl`).
- Project rule: after code changes run `graphify update .` (done once already).
- Tests: `cd backend && python -m pytest tests -q` → **2180 passed** after these
  changes (full suite, ~5.5 min). Note: `tests/test_account_backup.py` has 8 failures
  that are PRE-EXISTING and order-dependent (fail only when run after some other
  files in one process); verified identical with my changes stashed. Not ours.
- Tests use a temp `PRIMNOX2_HOME` (conftest), so `DELETE FROM memories` in tests is
  safe. **Never** point benchmarks at the user's real DB
  (`~/Documents/Primnox2/primnox.db`); all scripts use scratch DBs.
- Windows: set `PYTHONIOENCODING=utf-8` before running scripts. When editing via
  Python heredocs, `\b` in regex can turn into a literal backspace (0x08) — happened
  once; use the Edit tool for regex text.

---

## 3. Problems found (verified)

1. `search()` was Jaccard word-overlap only → "theme" can't find "switched to light mode".
2. No time dimension → couldn't answer "what was it in March".
3. Conflict engine superseded by sentence shape → retired many non-stale memories
   ("Nadia is a CTO at X" retired by "Omar is a Manager at X"; meeting logs retiring
   each other; "Devan switched to cortados." retired 8 unrelated memories because the
   person's own name counted as a shared word).
4. `import_many` skipped conflict handling entirely, and dropped refinements
   ("Postgres 16" after "Postgres") as duplicates.
5. `search()` returned superseded rows as answers about "now".

## 4. What was changed (uncommitted, on `exp/memory-v3`)

| File | Change |
|---|---|
| `backend/primnox2/cognition/topics.py` (NEW) | Deterministic `topic_of`, `kind_of` (`fact`/`event`), `classify`, `topic_of_query` (query-word aliases), `MULTI_VALUED` topics. No model in write path (deliberate design rule). |
| `backend/primnox2/cognition/conflict.py` | `resolve(..., new_topic=, new_kind=)` (backward compatible). New guards: `different_subjects` (different named subject), `different_scopes` ("for/about/of <Name>"), `attribute_slot` ("the <attr> for <Thing> is …"), events never supersede, additive words ("also", "too") never supersede, person's name excluded from reversal-rule shared words. Slot decision when both sides have a topic. |
| `backend/primnox2/memory/service.py` | `remember`/`import_many` classify + resolve (import resolves chronologically, retirement dated by successor's `created_at`, refinement exception in import dedupe, returned-after-replaced facts not treated as duplicates). New `search(query, limit, *, as_of=None, include_superseded=False)` ranking: topic match → stemmed question coverage (stop words removed) → recency → specificity. `parse_as_of` (epoch ms / `YYYY-MM` / `YYYY-MM-DD`). `backfill_topics()` (NULL=unclassified, ''=no topic; never un-supersedes). `recheck_conflicts(apply=False)` dry-run. |
| `backend/primnox2/storage/schema.sql`, `storage/db.py` | Schema v12: `memories.topic TEXT`, `memories.kind TEXT NOT NULL DEFAULT 'fact'`, migration `_m012_memory_topic_and_kind` (plain ADD COLUMN). |
| `backend/primnox2/tools/builtins.py` | `recall_memory` takes `as_of` (+ description tells model when to use it; errors on unreadable date). `forget_memory` uses `include_superseded=True`. |
| `backend/primnox2/app.py` | `/memories?q=` uses `include_superseded=True` (Memory tab shows history); new `GET/POST /memories/recheck`. |
| `backend/tests/test_memory_topics.py`, `test_memory_search.py` (NEW) | ~56 tests (topics, conflict rules, import chains, backfill, recheck, migration v11→v12, search ranking, as_of, tool arg). |
| `scripts/bench_memory.py` (NEW) + `scripts/bench_memory.baseline.json` | Dev benchmark on SDL packs + small hand-written "held-out" set. `--check` regression gate. Scores supersession by (old, successor) pair. |
| `scripts/bench_memory_blind.py` (NEW, never run) | Blind-test runner — see §6. |
| `docs/MEMORY_EXPERIMENT.md` (NEW) | Write-up. **Its numbers are DEV scores and the doc currently overstates them — must be corrected (§7).** |

## 5. Numbers — and why they are NOT test scores

Dev benchmark (`python scripts/bench_memory.py`), before → after:
- SDL L1 current-fact top-1: 40% → 100% (n=5). L4 "month N" top-1: 0% → 100% (n=6, import mode, `as_of` from ground truth = oracle time).
- "Held-out" top-1: 33% → 100% (n=12).
- Wrongly retired memories (memory-100, remember path): 63 → 0; office-500: 0.5%.

**These are contaminated** (the user called this out, correctly):
- I looked at every SDL miss and patched rules/lexicon for it (cortado, "takes the tram", role-before-manager, name rule, etc.).
- The "held-out" set was written by me in the same session and I added aliases ("live", "allergy", "notes", "theme") after seeing its misses.
- "Unseen" seeds/packs come from the same template generator — same distribution.
- `kw` query mode uses the exact topic word; my alias table maps exactly those.
- L4 used oracle `as_of`. n is tiny (12/12 → 95% CI ≈ 74–100%).

**Probably real (structural, not string-tuned):** retired memories excluded from "now"
answers; `as_of` filtering; events never retire; name-not-a-shared-subject fix;
import conflict handling; refinement-not-duplicate fix; pair-based supersession metric.
**Overfit:** anything depending on the topic lexicon / query aliases.

## 6. The task in progress: blind test

Status: a sub-agent was writing the dataset but was **stopped by the user before
finishing**; the scratchpad it wrote to is machine-local temp — assume no dataset
exists. Regenerate it.

### 6a. Generate the dataset (must be blind)
Give a fresh agent (or a human) ONLY this spec; it must not read the repo or this
handoff beyond the spec. Output two JSON files: `test.json` (10 scenarios) and
`dev.json` (3 different scenarios). Recommended location (inside repo so it travels,
but never used for tuning): `scripts/blind_memory/test.json`, `dev.json`.

Schema (list of scenarios):
```
{"scenario": "id", "today": "YYYY-MM-DD",
 "statements": [{"id": "s1", "date": "YYYY-MM-DD", "text": "...", "replaces": []}],   // 18-28, chronological; replaces = ids of EARLIER statements made no longer current
 "questions": [{"id": "q1", "text": "...", "type": "current|past|multi|absent",
                "answer_ids": ["s.."], "as_of": "YYYY-MM-DD or null"}]}             // 12-18
```
Content rules: first-person, natural, varied phrasing (no shared templates); wide
topics (diet, pets, family names, city, job/manager, commute, devices, apps, languages,
hobbies, health, music, coffee, car, habits, work projects, travel, goals — mostly
non-coding). 8–12 updates per scenario mixing explicit ("switched to") and IMPLICIT
("Got a Pixel 9 last week" replacing "My phone is an iPhone 13"), some A→B→C chains.
6+ traps per scenario that must NOT replace: second value of multi-valued things
(another pet), refinements ("It's a 2019 Corolla"), same-shape facts about another
person ("My brother works at Google"), same attribute of a different thing, one-off
events. Questions paraphrased (never copy statement words); ~45% current, ~30% past
with natural time phrases relative to `today` ("back in March", "last summer") and
resolved `as_of`, ~15% multi, ~10% absent. Double-check all labels.

**Freeze it** (commit it) before the first scored run.

### 6b. Run (`scripts/bench_memory_blind.py`, written, never executed — expect small bugs)
- `--split dev|test` (test refuses `--misses`), `--data <dir with test.json/dev.json>`
  (default points at the old machine's temp scratchpad — pass `--data scripts/blind_memory`).
- `--load chat` (real `remember()` with clock monkeypatched to each statement's date —
  `mem.now_ms = lambda: stamp`) or `--load import`.
- `--query question` (raw question, no date = floor), `oracle` (raw question + true
  `as_of` = retrieval upper bound), `model` (local Ollama `qwen2.5:7b-instruct` — the
  same 7B the app uses — gets today's date + `recall_memory` description and writes
  `{query, as_of}`; realistic).
- Reports top-1/top-3 per type with 95% Wilson intervals, multi all-values@k+2,
  stale-on-top for "now" questions, absent-returned-nothing, model date behaviour,
  supersession precision/recall by (old, successor) pair.
- Ollama on the original machine: `http://localhost:11434`, models
  `qwen2.5:7b-instruct`, `qwen2.5:0.5b`. Check what exists on the new machine.

Suggested run order: fix runner bugs on **dev** only → then run **test** once per
configuration (chat×{question,oracle,model}, import×oracle) → record aggregates.
Any rule change after seeing test results = the test set is spent; write a new one.

## 7. Remaining TODO
1. Generate + freeze blind dataset (§6a). Run §6b. Report with intervals.
2. Fix `docs/MEMORY_EXPERIMENT.md`: relabel its table as dev scores, add the
   contamination section (§5), add blind-test results when available. (It currently
   overstates.)
3. Ask the user before committing; commit messages end with
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Don't push/PR unless asked.
4. Not done / decided later: embeddings (plan phase 2b — skipped because dev scores had
   no headroom; the blind test may justify it; encoder exists in
   `backend/primnox2/tools/retrieval.py`), dispute detection (`memories.disputed`
   column unused), Memory-tab UI for `/memories/recheck`, event routing to episodic
   store (plan phase 4), end-to-end with the real chat gateway (plan phase 5).
5. The full plan is in `~/.claude/plans/silly-beaming-firefly.md` on the old machine
   (not portable); §4–§7 here supersede it.
