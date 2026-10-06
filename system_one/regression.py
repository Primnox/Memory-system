"""The memory head's regression cases: behaviours every round must keep.

`data/regression_pairs.json` holds hand-written pairs, one per behaviour the
memory has to get right (explicit and implicit updates, chain reactions, a
misdiagnosed allergy; restatements, second values, other people, trips, plans,
another allergy). They are not a benchmark: written by the same hand that knows
the failure modes, they check that a new round has not lost something an older
one had. A case counts as passed when P(replaces) >= 0.5 for "replaces" and
< 0.5 for "not".

    python regression.py --model kaggle_out4/models/laya-memory-r4 --tag r4
    python regression.py --model models/laya-memory-r5 --tag r5 --against r4

`--against` compares with an earlier run's results and exits 1 when a case the
earlier model passed now fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_train import QUESTIONS
from screen import state_text

HERE = Path(__file__).resolve().parent
CASES = HERE / "data" / "regression_pairs.json"


def run(agent, cases: list[dict]) -> dict[str, float]:
    states = [state_text({"old_text": c["old"], "new_text": c["new"],
                          "old_date": "2025-03-01", "new_date": "2025-06-01"}) for c in cases]
    results = agent.predict_batch(states, QUESTIONS)
    return {c["id"]: float(r["answers"]["relation"]["probabilities"]["replaces"]) for c, r in zip(cases, results)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", help="the model's ONNX export, run on a DirectX 12 GPU (onnx_dml.py)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--against", help="an earlier --tag to compare with")
    args = ap.parse_args()
    if args.onnx:
        from onnx_dml import load
        agent = load(args.model, args.onnx)
    else:
        from laya.agent import Agent
        agent = Agent(args.model)
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    probs = run(agent, cases)
    passed = {c["id"]: (probs[c["id"]] >= 0.5) == (c["kind"] == "replaces") for c in cases}
    for c in cases:
        mark = "ok  " if passed[c["id"]] else "FAIL"
        print(f"{mark} {c['kind']:8s} P={probs[c['id']]:.2f}  {c['id']}")
    n_ok = sum(passed.values())
    print(f"\n{n_ok}/{len(cases)} passed "
          f"(replaces {sum(passed[c['id']] for c in cases if c['kind'] == 'replaces')}"
          f"/{sum(c['kind'] == 'replaces' for c in cases)}, "
          f"not {sum(passed[c['id']] for c in cases if c['kind'] == 'not')}/{sum(c['kind'] == 'not' for c in cases)})")
    out = HERE / "results" / f"regression_{args.tag}.json"
    out.write_text(json.dumps({"model": Path(args.model).name, "probs": probs, "passed": passed}, indent=1))
    if args.against:
        before = json.loads((HERE / "results" / f"regression_{args.against}.json").read_text())["passed"]
        lost = [k for k, ok in before.items() if ok and not passed.get(k, False)]
        gained = [k for k, ok in passed.items() if ok and not before.get(k, False)]
        print(f"against {args.against}: gained {gained or 'none'}; lost {lost or 'none'}")
        if lost:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
