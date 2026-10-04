# Blind memory evaluation set, v2

A second, larger held-out set. `../test.json` (v1) was spent: the memory code
was changed after its results were seen. This one is frozen before the first
scored run.

Written 2026-10-02 from a spec by two Claude Sonnet 5 agents that never saw the
memory code (6 scenarios each; `parts/` holds each writer's original file).

- `test.json` - 12 scenarios, 309 statements, 112 updates (supersession
  pairs), 195 questions. FROZEN: see `FROZEN.sha256`.

Label audit: an independent third Sonnet 5 agent relabelled the set from a copy
with every label removed. Agreement: 111/112 supersession pairs, 195/195 answer
sets.

Adjudication: the one disagreement (scenario `tessa-perth-teacher`, statement
s16 superseded by s21) was flagged by the auditor as ambiguous. It is kept as
the writer labelled it; it was not resolved in either direction after any
scored run.

Rules: never read misses on this split. `bench_memory_blind.py` refuses
`--misses` on any split starting "test", and reports aggregates only. A rule or
code change made after seeing these results spends the set; write a new one.
