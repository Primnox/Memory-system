# Handoff — continuing the memory work on another machine

Written 2026-10-04 for a Claude Code session (or a person) picking this up
without the previous session's notes. Read this, then
[`MEMORY_EXPERIMENT.md`](MEMORY_EXPERIMENT.md) and
[`SYSTEM_ONE_PLAN.md`](SYSTEM_ONE_PLAN.md).

## Where things stand

- Best version: branch `exp/memory-change-verify` of the Primnox app repo
  (`Primnox/main`, commit `c013cad`; the app repo is going private), 63% top-1
  on the fresh blind set `v2`. All memory branches are pushed there; this
  public repo mirrors the memory files.
- **Not merged into Primnox `main`.** The owner decides when. Before any merge
  or any run against real data, back up `~/Documents/Primnox2/primnox.db`:
  schema v12 is one-way and older code refuses a v12 database.
- The verifier is off by default (`memory.llm_verify`). Undecided whether to
  turn it on: its queue is in memory and lost on restart, and edited memories
  are re-judged by rules only.
- The full-chat harness (`scripts/e2e_memory_chat.py`) is parked. Plan: one run
  at the very end, on the final version. No more work on it before then.

## Next steps, in order

1. **Wire the trained memory head into the memory service** as a live
   background job (like the verifier), with the 9B overriding only uncertain
   decisions and only when at least 95% sure (`RESEARCH.md` §6.11). Simulated
   so far: blind answers 59.1% → 66.7% (`system_one/answers_with_decisions.py`).
   The round-4 checkpoint (writers A+B+C) is in
   `system_one/kaggle_out4/models/laya-memory-r4` (not in git, ~800 MB);
   `system_one/kaggle/eval_kaggle.py` retrains and re-scores on Kaggle in
   ~25 min.
2. **A learned fact extractor** ("worth keeping?" head): the rule-based
   splitter (`system_one/extract.py`) raised long-message recall from 8% to
   46% at 0.66 precision. The same step unlocks LoCoMo / LongMemEval, since
   the memory refuses multi-fact text.
3. **Make it faster** (~0.4 s per comparison on CPU; ~15–50 comparisons per
   message at scale, `system_one/candidates_scale.py`): a ~150M student, one
   pass over all candidates, or a background queue.
4. **Fresh blind set `v3`** written by a different model family, with chain
   reactions and "it happened earlier" questions (needed to measure
   event-time dating), plus a small human-labelled set. The owner's real chats
   need an explicit OK first (a local-only check was proposed and blocked
   pending that OK).
5. **Open design limitations** (`RESEARCH.md` §9): validity windows per fact,
   a re-check pass for wrong retirements, pinning standing preferences in
   trimmed prompts, a safety head instead of the English word list.
6. Send only relevant memories to the prompt: similarity top-20 plus the
   recall tool as fallback (blind: 89% fewer tokens at 200 facts, 79% of
   answer facts kept; the graph index did not beat similarity, `graph/`).
7. Graph memory only with learned entity links and multi-hop tests.
8. Head-to-head against Mem0, Graphiti/Zep, Letta, LangMem on LongMemEval /
   LoCoMo.

## Working rules

- **Blind testing is the method.** Never tune on, train on, or read the misses
  of `scripts/blind_memory/test*.json` or `v2/test.json` (`--misses` is refused
  on test splits). Work on the dev split. Each version is scored on a test set
  once. A rule changed after seeing test results means the set is spent —
  write a new one with agents that never saw the code, re-label it
  independently, freeze it with SHA-256, then score.
- Report top-1 with 95% Wilson intervals and paired exact McNemar, not point
  scores. Development scores are labelled as such.
- Benchmarks always use scratch databases. Never point anything at the real
  `primnox.db`, and never set up a vault from a scratch backend.
- This repo is public (the Primnox app repo is going private): scan for
  secrets and personal data before every push, and never copy app code beyond
  the memory files into it without the owner's OK. Never commit raw full-chat outputs (they hold test
  transcripts). Do not push `fix/memory-robustness` (its fixtures hold the
  owner's first name; it also scored worse blind and was parked).
- No merges to `main` and no pull requests without the owner's OK.
- The owner is cost-sensitive: sub-agents run on Sonnet, not Opus; run one
  heavy job at a time; prefer local or free compute (Kaggle) over paid APIs.
  OpenRouter "free" routes were unavailable or paid on 2026-10-02 — check
  before relying on them.
- Honest numbers over impressive ones.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  (Sonnet agents use their own model line).

## Machines

- **Dev PC** (Windows, AMD RX 9060 XT, 32 GB RAM): Ollama with the app's model
  `huihui_ai/qwen3.5-abliterated:9b` (thinking off) and the judge
  `gemma4:26b-a4b-it-q4_K_M` (run CPU-only, `num_gpu: 0`). Needed for
  `--llm-verify`, `--query model`, and the full-chat harness.
- **8 GB laptop**: fine for the memory code, the tests, blind runs without the
  9B (`--query oracle`, no `--llm-verify`), writing data, and running small
  models (~1 GB). Not for the 9B or for training — training goes to Kaggle.

## Setup on a new machine

Needs access to the private Primnox app repo.

```bash
git clone -b exp/memory-change-verify https://github.com/Primnox/main.git primnox
```

```bash
cd primnox/backend && python -m venv venv && venv/Scripts/pip install -r requirements.txt
```

```bash
cd primnox/backend && PYTHONIOENCODING=utf-8 venv/Scripts/python -m pytest tests -q -k memory
```

```bash
cd primnox && PYTHONIOENCODING=utf-8 backend/venv/Scripts/python scripts/bench_memory_blind.py --split dev
```

The first run downloads the MiniLM encoder (`sentence-transformers/all-MiniLM-L6-v2`).
On Windows set `PYTHONIOENCODING=utf-8` for every script.

## Kaggle (training on a free GPU)

`system_one/.venv` has the Kaggle CLI. Sign in once with
`system_one/.venv/Scripts/kaggle.exe auth login --no-launch-browser` in an
interactive shell (paste the code back); the account must be phone-verified or
jobs get no internet. Create `system_one/kaggle/kernel-metadata.json` (kept out
of git: `id` = `<username>/primnox-memory-head`, `kernel_type` script,
`enable_gpu` and `enable_internet` true, `code_file` `train_kaggle.py`), then
`kaggle kernels push -p system_one/kaggle`, poll `kaggle kernels status`, fetch
with `kaggle kernels output`.

## Runner reference

`scripts/bench_memory_blind.py`:
`--split dev|test|<file stem>` · `--data DIR` (e.g. `scripts/blind_memory/v2`) ·
`--load chat|import` · `--query question|oracle|model` · `--model NAME`
(Ollama name, or `omniroute:<combo>`) · `--backend DIR` (score another
checkout) · `--embeddings search|supersede|both|off` · `--actor-guard` ·
`--llm-verify` · `--verify-log FILE` · `--json FILE` ·
`--mcnemar A.json B.json` · `--misses` (dev only).
Score files go in `scripts/blind_memory/results/`; regenerate the table with
`results/summarize.py`.
