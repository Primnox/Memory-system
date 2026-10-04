"""Change detection by the fine-tuned memory head on a blind set, run as the app would.

Statements arrive in date order. Each new one is compared with up to --k most
similar LIVE facts plus live facts sharing a named person or thing (names
learned only from statements seen so far); every candidate with
P(replaces) >= --threshold is retired and leaves the live set. The threshold
is fixed in advance, never tuned on the blind set.

Scored exactly as scripts/bench_memory_blind.py scores supersession: by
(old, successor) pair; "mistaken" = a retired fact that the labels never
replace. Only totals are printed and saved: no statement text, no per-pair
outcome, so nothing about individual test items is read.

    system_one/.venv/Scripts/python system_one/blind_pairs.py \\
        --model system_one/kaggle_out/models/laya-memory-v2-nodnli --tag v2-nodnli
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from build_train import QUESTIONS, embed, entities, names_in
from screen import state_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "scripts/blind_memory/v2/test.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-linked", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-pairs", action="store_true",
                    help="also save the retired (old, new) statement ids per scenario (ids only, no text), "
                         "for paired tests and for answer accuracy with these decisions")
    args = ap.parse_args()

    from laya.agent import Agent
    agent = Agent(args.model)
    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    vec = embed(sorted({s["text"] for sc in scenarios for s in sc["statements"]}))

    tot = {"true": 0, "predicted": 0, "correct": 0, "false_retire": 0, "candidates_scored": 0,
           "true_reachable": 0}
    per = {}
    t0 = time.perf_counter()
    for sc in scenarios:
        sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
        truth = {(o, s["id"]) for s in sts for o in s.get("replaces", [])}
        replaced_ever = {o for o, _ in truth}
        live, got, reachable = [], set(), 0
        for i, new in enumerate(sts):
            names = names_in(sts[:i + 1])
            ent_new = entities(new["text"], names)
            ranked = sorted(live, key=lambda o: -float(vec[o["text"]] @ vec[new["text"]]))
            cands = ranked[:args.k]
            ids = {o["id"] for o in cands}
            cands += [o for o in ranked[args.k:] if entities(o["text"], names) & ent_new][:args.max_linked]
            reachable += sum(1 for o in cands if (o["id"], new["id"]) in truth)
            if cands:
                states = [state_text({"old_text": o["text"], "new_text": new["text"],
                                      "old_date": o["date"], "new_date": new["date"]}) for o in cands]
                results = agent.predict_batch(states, QUESTIONS)
                retire = {o["id"] for o, r in zip(cands, results)
                          if r["answers"]["relation"]["probabilities"]["replaces"] >= args.threshold}
                got |= {(o, new["id"]) for o in retire}
                live = [o for o in live if o["id"] not in retire]
                tot["candidates_scored"] += len(cands)
            live.append(new)
        row = {"true": len(truth), "predicted": len(got), "correct": len(got & truth),
               "false_retire": sum(1 for o, _ in got if o not in replaced_ever), "true_reachable": reachable}
        if args.save_pairs:
            row["retired_pairs"] = sorted([o, n] for o, n in got)
        per[sc["scenario"]] = row
        for k, v in row.items():
            if k in tot:
                tot[k] += v
        print(f"{len(per)}/{len(scenarios)} scenarios done", flush=True)

    secs = time.perf_counter() - t0
    out = {"data": Path(args.data).name, "model": Path(args.model).name, "k": args.k,
           "max_linked": args.max_linked, "threshold": args.threshold, "seconds": round(secs),
           "ms_per_candidate": round(secs * 1000 / max(1, tot["candidates_scored"]), 1),
           "total": tot, "per_scenario": per}
    (HERE / "results" / f"blind_{Path(args.data).parent.name}_{args.tag}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    p = tot["correct"] / tot["predicted"] if tot["predicted"] else 0.0
    print(f'\nchanges noticed {tot["correct"]}/{tot["true"]} ({tot["correct"] / tot["true"]:.0%}); '
          f'retired {tot["predicted"]}, mistaken {tot["false_retire"]}; pair precision {p:.0%}; '
          f'reachable from candidates {tot["true_reachable"]}/{tot["true"]}; '
          f'{out["ms_per_candidate"]} ms per candidate on this machine')


if __name__ == "__main__":
    main()
