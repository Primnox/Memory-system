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
from extract import statements
from screen import metrics, state_text

HERE = Path(__file__).resolve().parent
FILES = [HERE / "data" / "blind_external" / "pairs_dnli.jsonl", HERE / "data" / "blind_external" / "pairs_lme_ku.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sentences", action="store_true",
                    help="split both messages into statements about the user (extract.py) and score every "
                         "statement pair; the message pair's score is the highest")
    args = ap.parse_args()
    from laya.agent import Agent
    agent = Agent(args.model)
    report = {"model": args.model, "files": {}}
    for f in FILES:
        pairs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        t0 = time.perf_counter()
        # one unit per comparison; a message pair owns one or more units
        units, owner = [], []
        for j, p in enumerate(pairs):
            olds = statements(p["old_text"]) if args.sentences else [p["old_text"]]
            news = statements(p["new_text"]) if args.sentences else [p["new_text"]]
            for o in olds:
                for n in news:
                    units.append({**p, "old_text": o, "new_text": n})
                    owner.append(j)
        unit_probs = []
        for i in range(0, len(units), args.batch):
            batch = units[i:i + args.batch]
            res = agent.predict_batch([state_text(u) for u in batch], QUESTIONS)
            unit_probs += [float(r["answers"]["relation"]["probabilities"]["replaces"]) for r in res]
        probs = [0.0] * len(pairs)
        for j, s in zip(owner, unit_probs):
            probs[j] = max(probs[j], s)
        y = [p["label"] for p in pairs]
        by_source = defaultdict(lambda: [0, 0])
        for p, s in zip(pairs, probs):
            by_source[p["source"]][0] += int(s >= 0.5)
            by_source[p["source"]][1] += 1
        report["files"][f.stem] = {
            "pairs": len(pairs), "positives": sum(y), **metrics(y, probs),
            "called_replace_by_source": {k: f"{a}/{b}" for k, (a, b) in sorted(by_source.items())},
            "comparisons": len(units),
            "ms_per_pair": round((time.perf_counter() - t0) * 1000 / len(pairs), 1)}
        r = report["files"][f.stem]
        print(f'{f.stem:<14} AP {r["ap"]:.2f}  P {r["precision_at_0.5"]:.2f}  R {r["recall_at_0.5"]:.2f}  '
              f'F1 {r["f1_at_0.5"]:.2f}  {r["called_replace_by_source"]}  {r["ms_per_pair"]} ms/pair', flush=True)
    report["sentences"] = args.sentences
    out = HERE / "results" / f"external_{args.tag}{'_sentences' if args.sentences else ''}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
