"""Score a Laya checkpoint on the external blind pairs (build_external.py).

Same question and wording as everywhere else (relation choice); a pair counts
as "replaces" when P(replaces) >= 0.5, fixed in advance. Reports AP and
precision / recall / F1 per file, plus how often each source label is called a
replacement (e.g. how many Dialogue NLI neutral pairs were wrongly retired).

    system_one/.venv/Scripts/python system_one/external_eval.py \\
        --model convaiinnovations/laya --tag zeroshot
    system_one/.venv/Scripts/python system_one/external_eval.py \\
        --model system_one/kaggle_out/models/laya-memory-v2-nodnli --tag v2-nodnli
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from build_train import QUESTIONS
from screen import metrics, state_text

HERE = Path(__file__).resolve().parent
FILES = [HERE / "data" / "blind_external" / "pairs_dnli.jsonl", HERE / "data" / "blind_external" / "pairs_lme_ku.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    from laya.agent import Agent
    agent = Agent(args.model)
    report = {"model": args.model, "files": {}}
    for f in FILES:
        pairs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        t0, probs = time.perf_counter(), []
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i:i + args.batch]
            res = agent.predict_batch([state_text(p) for p in batch], QUESTIONS)
            probs += [float(r["answers"]["relation"]["probabilities"]["replaces"]) for r in res]
        y = [p["label"] for p in pairs]
        by_source = defaultdict(lambda: [0, 0])
        for p, s in zip(pairs, probs):
            by_source[p["source"]][0] += int(s >= 0.5)
            by_source[p["source"]][1] += 1
        report["files"][f.stem] = {
            "pairs": len(pairs), "positives": sum(y), **metrics(y, probs),
            "called_replace_by_source": {k: f"{a}/{b}" for k, (a, b) in sorted(by_source.items())},
            "ms_per_pair": round((time.perf_counter() - t0) * 1000 / len(pairs), 1)}
        r = report["files"][f.stem]
        print(f'{f.stem:<14} AP {r["ap"]:.2f}  P {r["precision_at_0.5"]:.2f}  R {r["recall_at_0.5"]:.2f}  '
              f'F1 {r["f1_at_0.5"]:.2f}  {r["called_replace_by_source"]}  {r["ms_per_pair"]} ms/pair', flush=True)
    out = HERE / "results" / f"external_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
