"""Escalation curve and calibration of a fine-tuned head, from saved dev scores.

The small model decides; decisions with lo < P(replaces) < hi go to the
supervisor (the local 9B). Assuming the supervisor is right (a best case),
how many decisions are escalated and how many errors stay in the rest?
Also expected calibration error over 10 equal-width bins.

    system_one/.venv/Scripts/python system_one/escalation.py \\
        --scores system_one/results/kaggle_v2/scores_laya_ft_v2-nodnli.json --tag run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from screen import key, load_pairs

HERE = Path(__file__).resolve().parent
BANDS = [(0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.05, 0.95)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    labels = {key(p): p["label"] for p in load_pairs()}
    scores = json.loads(Path(args.scores).read_text())
    items = [(v["choice_replaces"], labels[k]) for k, v in scores.items() if k in labels]
    n = len(items)
    curve = []
    for lo, hi in BANDS:
        kept = [(p, y) for p, y in items if not (lo < p < hi)]
        missed = sum(1 for p, y in kept if y and p < 0.5)
        wrong = sum(1 for p, y in kept if not y and p >= 0.5)
        curve.append({"band": [lo, hi], "escalated": n - len(kept), "escalated_share": (n - len(kept)) / n,
                      "missed_kept": missed, "wrong_retire_kept": wrong,
                      "accuracy_kept": 1 - (missed + wrong) / len(kept)})
    bins = [[] for _ in range(10)]
    for p, y in items:
        bins[min(9, int(p * 10))].append((p, y))
    ece = sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in bins if b)
    out = {"scores": Path(args.scores).name, "decisions": n, "positives": sum(y for _, y in items),
           "curve": curve, "ece_10_bins": ece}
    (HERE / "results" / f"escalation_{args.tag}.json").write_text(json.dumps(out, indent=1))
    for c in curve:
        print(f'band {c["band"]}: escalate {c["escalated"]} ({c["escalated_share"]:.0%}), '
              f'errors kept {c["missed_kept"]} missed / {c["wrong_retire_kept"]} wrong')
    print(f"ECE {ece:.3f}")


if __name__ == "__main__":
    main()
