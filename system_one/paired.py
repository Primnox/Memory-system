"""Paired exact McNemar tests on blind v2 (ids only; no statement text is read).

Change detection: for every labelled (old, new) change, was it caught by A and
by B? Answers: for every answerable question, was top-1 right under A and
under B? Exact two-sided binomial test on the discordant items.

    system_one/.venv/Scripts/python system_one/paired.py \\
        --changes results/rules_v2_pairs.json results/blind_v2_run1-pairs.json \\
        --answers results/answers_v2_rules.json results/answers_v2_run1.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pairs_of(path: Path) -> dict[str, set[tuple[str, str]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "per_scenario" in d:   # blind_pairs.py --save-pairs
        return {sc: {tuple(p) for p in row["retired_pairs"]} for sc, row in d["per_scenario"].items()}
    return {sc: {tuple(p) for p in row["pairs"]} for sc, row in d.items()}   # rules_pairs.py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "scripts/blind_memory/v2/test.json"))
    ap.add_argument("--changes", nargs=2, metavar=("A", "B"))
    ap.add_argument("--answers", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    report = {}
    if args.changes:
        truth = {sc["scenario"]: {(o, s["id"]) for s in sc["statements"] for o in s.get("replaces", [])}
                 for sc in json.loads(Path(args.data).read_text(encoding="utf-8"))}
        a, b = (pairs_of(Path(p)) for p in args.changes)
        only_a = only_b = both = 0
        for sc, true in truth.items():
            for t in true:
                ha, hb = t in a.get(sc, set()), t in b.get(sc, set())
                both += ha and hb
                only_a += ha and not hb
                only_b += hb and not ha
        report["changes"] = {"A": Path(args.changes[0]).name, "B": Path(args.changes[1]).name,
                             "caught_by_both": both, "only_A": only_a, "only_B": only_b,
                             "p": mcnemar(only_a, only_b)}
        print(f"changes: both {both}, only A {only_a}, only B {only_b}, exact McNemar p = {report['changes']['p']:.2g}")
    if args.answers:
        a, b = (json.loads(Path(p).read_text())["hits"] for p in args.answers)
        only_a = sum(1 for k in a if a[k] and not b[k])
        only_b = sum(1 for k in a if b[k] and not a[k])
        report["answers"] = {"A": Path(args.answers[0]).name, "B": Path(args.answers[1]).name,
                             "only_A": only_a, "only_B": only_b, "p": mcnemar(only_a, only_b)}
        print(f"answers: only A right {only_a}, only B right {only_b}, exact McNemar p = {report['answers']['p']:.2g}")
    (HERE / "results" / "paired_v2.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
