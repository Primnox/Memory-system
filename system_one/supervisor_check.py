"""Escalation with a real supervisor instead of an assumed-perfect one.

The small model's uncertain development decisions (lo < P(replaces) < hi) are
sent to the application's own 9B verifier (`memory/verifier.py`, local Ollama,
relation read from token probabilities); its answer replaces the small model's
for those pairs. Reports errors with no escalation, with a perfect supervisor
(escalation.py's assumption) and with the real 9B.

Runs in the Primnox backend environment, with Ollama serving the 9B:
    PYTHONIOENCODING=utf-8 <primnox>/backend/venv/Scripts/python system_one/supervisor_check.py \\
        --backend <primnox>/backend --scores system_one/results/kaggle_v2/scores_laya_ft_v2-nodnli.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen import key, load_pairs   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--bands", nargs="+", default=["0.3-0.7", "0.2-0.8", "0.1-0.9"])
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve()))
    from primnox2.memory import verifier

    pairs = {key(p): p for p in load_pairs()}
    scores = json.loads(Path(args.scores).read_text())
    items = [(k, v["choice_replaces"], pairs[k]["label"]) for k, v in scores.items() if k in pairs]
    lo_all = min(float(b.split("-")[0]) for b in args.bands)
    hi_all = max(float(b.split("-")[1]) for b in args.bands)
    asked, t0 = {}, time.perf_counter()
    for k, p, _ in items:
        if lo_all < p < hi_all:
            v = verifier.ask(pairs[k]["new_text"], pairs[k]["old_text"])
            asked[k] = None if v is None else {"relation": v.relation, "confidence": v.confidence}
    secs = time.perf_counter() - t0
    out = {"scores": Path(args.scores).name, "supervisor": verifier.model_name(),
           "asked": len(asked), "unanswered": sum(1 for v in asked.values() if v is None),
           "seconds": round(secs), "bands": []}

    def errors(decide) -> tuple[int, int]:
        missed = sum(1 for k, p, y in items if y and not decide(k, p))
        wrong = sum(1 for k, p, y in items if not y and decide(k, p))
        return missed, wrong

    base = errors(lambda k, p: p >= 0.5)
    out["no_escalation"] = {"missed": base[0], "wrong_retire": base[1]}
    for band in args.bands:
        lo, hi = (float(x) for x in band.split("-"))
        esc = [k for k, p, _ in items if lo < p < hi]
        label = {k: y for k, _, y in items}
        perfect = errors(lambda k, p: label[k] if lo < p < hi else p >= 0.5)

        def real(k, p):
            if not lo < p < hi:
                return p >= 0.5
            v = asked.get(k)
            return p >= 0.5 if v is None else v["relation"] == "replaces"   # no answer: keep the small model's call

        def veto(k, p):
            # the supervisor may stop a retirement, never force one
            if not lo < p < hi or p < 0.5:
                return p >= 0.5
            v = asked.get(k)
            return True if v is None else v["relation"] == "replaces"

        def sure(k, p):
            # the supervisor overrides only when its own probability is >= 0.95 (the app's verifier rule)
            if not lo < p < hi:
                return p >= 0.5
            v = asked.get(k)
            if v is None or v["confidence"] < 0.95:
                return p >= 0.5
            return v["relation"] == "replaces"

        r, vt, sr = errors(real), errors(veto), errors(sure)
        sup_right = sum(1 for k in esc if asked.get(k) and (asked[k]["relation"] == "replaces") == bool(label[k]))
        out["bands"].append({"band": band, "escalated": len(esc), "supervisor_right": sup_right,
                             "perfect_supervisor": {"missed": perfect[0], "wrong_retire": perfect[1]},
                             "real_supervisor_overrides": {"missed": r[0], "wrong_retire": r[1]},
                             "real_supervisor_veto_only": {"missed": vt[0], "wrong_retire": vt[1]},
                             "real_supervisor_only_when_sure": {"missed": sr[0], "wrong_retire": sr[1]}})
        print(f"band {band}: escalated {len(esc)}, 9B right on {sup_right}; errors: none {sum(base)}, "
              f"perfect {sum(perfect)}, 9B overrides {sum(r)} ({r[0]}/{r[1]}), veto-only {sum(vt)} ({vt[0]}/{vt[1]}), "
              f"only-when-sure {sum(sr)} ({sr[0]}/{sr[1]})")
    print(f"9B: {len(asked)} questions in {secs:.0f}s, {out['unanswered']} unanswered")
    out["answers"] = asked   # development pairs only
    (HERE / "results" / "supervisor_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
