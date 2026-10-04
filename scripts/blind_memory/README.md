# Blind memory evaluation set

Written 2026-10-01 from a spec by three Claude Sonnet 5 agents that never saw
the memory code (test: two writers, 5 scenarios each, people A–H and N–Z;
dev: one writer, people I–M). `parts/` holds each writer's original file.

- `test.json` — 10 scenarios, 267 statements, 93 updates, 168 questions
  (78 current, 50 past, 20 multi, 20 absent). FROZEN: see `FROZEN.sha256`.
- `dev.json` — 3 scenarios, 82 statements, 27 updates, 51 questions. For
  runner debugging and reading misses.

Label audit: an independent Sonnet 5 annotator relabelled the test split from a
copy with every label removed. Agreement: 93/93 supersession pairs, 166/168
answer sets. The two disagreements were adjudicated in the auditor's favour
before any scored run (a refinement listed as a separate vehicle in a multi
answer; a "new place has a yard" statement that does not say where).

Rules: never read test misses (`bench_memory_blind.py` refuses `--misses` on
any split starting "test"). A rule change made after seeing test results spends
the set — write a new one.

Wording variants: `test_para1.json`, `test_para2.json` — every statement and
question of the test split reworded by a Sonnet 5 agent (meaning, names,
numbers and explicitness of change preserved; not machine-verified), labels
copied by id from `test.json`. Use them to measure sensitivity to wording.

`change.json` is not blind: 7 scenarios (181 statements, 43 updates) written by
the author of the change-detection work, with the system in front of them and
tuned against. Three are plain life updates, one is phrased differently, two are
traps (additions, accessories, other people's facts, one-off events) where almost
nothing may be retired. Run it for supersession numbers (`--split change
--query oracle`); its few questions are not a retrieval benchmark.
