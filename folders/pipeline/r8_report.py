"""Round 8 at a glance, against round 7.   python r8_report.py [../kaggle/r8_out]

Reads the job's result files (blind_*.json from blind_pairs.py, external_r8i-*.json from
external_eval.py, clerk.log) and prints one table. Round 7's numbers are its published ones
(round7/results): blind v2 clean 104/112 caught, 0 wrong; LongMemEval KU with Gemini facts
F1 0.78 (P 0.77 R 0.79); Dialogue NLI test / verified F1 0.61 / 0.69.
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "kaggle" / "r8_out")
R7 = {"blind": "104/112 caught, 0 wrong", "lme": "F1 0.78 (P 0.77, R 0.79)", "dnli": "0.61 / 0.69"}


def blind(tag: str) -> str:
    f = next(iter(D.glob(f"blind_*_{tag}.json")), None)
    if not f:
        return "(missing)"
    t = json.loads(f.read_text(encoding="utf-8"))["total"]
    return f"{t['correct']}/{t['true']} caught, {t['false_retire']} wrong (retired {t['predicted']})"


def external(tag: str) -> dict:
    f = next(iter(D.glob(f"external_{tag}*.json")), None)
    return json.loads(f.read_text(encoding="utf-8"))["files"] if f else {}


print(f"Round 8 results from {D}\n")
print(f"{'test':52s} {'round 7':28s} round 8 (reads the index)")
print(f"{'blind v2 clean, no tags':52s} {R7['blind']:28s} {blind('r8i-clean-notags')}")
print(f"{'blind v2 clean, tags from cloud clerk (one at a time)':52s} {'(r7: ' + blind('r7-clean-b1tags') + ')':28s} {blind('r8i-clean-b1tags')}")
print(f"{'blind v2 clean, tags from the LOCAL 3B clerk':52s} {'':28s} {blind('r8i-clean-clerktags')}")
lme = external("r8i-lme-llm").get("pairs_lme_ku", {})
if lme:
    print(f"{'LongMemEval KU, Gemini facts, F1@0.5':52s} {R7['lme']:28s} F1 {lme['f1_at_0.5']:.2f} "
          f"(P {lme['precision_at_0.5']:.2f}, R {lme['recall_at_0.5']:.2f}; best F1 {lme['best_f1']:.2f}, AP {lme['ap']:.2f})")
dn = external("r8i-dnli")
if dn:
    vals = [v["f1_at_0.5"] for k, v in sorted(dn.items())]
    print(f"{'Dialogue NLI test / verified, F1':52s} {R7['dnli']:28s} " + " / ".join(f"{x:.2f}" for x in vals))
log = D / "clerk.log"
if log.exists():
    print("\nlocal clerk (Qwen 2.5 3B):")
    for line in log.read_text(encoding="utf-8").splitlines():
        print("  " + line)
