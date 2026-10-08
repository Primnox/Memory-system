"""Gemini as the memory's change detector, on the head's own exams.   python gemini_judge.py v2 | lme | report

v2   blind v2 (clean, noisy, heavy copies): Gemini labels each person's label-free statements
     with AUDIT.md's rules; scored exactly as blind_pairs.py scores the head — true changes
     noticed (of 112) and "mistaken" = retired facts the labels never replace.
lme  LongMemEval's 421 knowledge-update pairs: for each (older message, newer message), does the
     newer one make the older one's fact no longer true? Precision / recall / F1, as external_eval.

Neither set was ever trained on. Totals only; no item text is printed.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drive
from drive import AUDIT, log, parse, strip_labels

HERE = Path(__file__).resolve().parent
OUT = HERE / "gemini_judge"
MODEL = "gemini-3.1-pro-high"
V2 = {"clean": HERE.parent / "ms" / "scripts" / "blind_memory" / "v2" / "test.json",
      "noisy": HERE.parent / "kaggle" / "noise_ds" / "v2_noisy.json",
      "heavy": HERE.parent / "kaggle" / "noise_ds" / "v2_noisy_heavy.json"}
LME = HERE.parent / "ms" / "system_one" / "data" / "blind_external" / "pairs_lme_ku.jsonl"
NO_TOOLS = "\nWork only from the data below. Do not use any tools: do not read, list or search any files or folders, run commands or browse.\n"

LME_PROMPT = """Each item below is a pair of messages a user sent a personal assistant, the OLDER one first,
with their dates. Decide for each pair: does the NEWER message make a fact stated in the OLDER message
no longer true of the user right now (an update, a change, a correction, a replacement)? Additions,
details, restatements and unrelated messages are NOT updates.

Reply with JSON only: {"<pair id>": 1 or 0, ...} with every pair id.
""" + NO_TOOLS + "\nPairs:\n"


def v2_one(job: tuple[str, int, dict]) -> None:
    variant, i, sc = job
    cwd = OUT / f"v2_{variant}" / f"p{i + 1:02d}"
    if (cwd / "labels.json").exists():
        return
    body = json.dumps(strip_labels([sc]), indent=1, ensure_ascii=False)
    for attempt in range(3):
        try:
            text, _ = drive.agy(AUDIT + NO_TOOLS + body, MODEL, cwd)
            lab = parse(text)["labels"][0]
            (cwd / "labels.json").write_text(json.dumps(lab, ensure_ascii=False), encoding="utf-8")
            return
        except Exception as e:  # noqa: BLE001
            log(f"gemini_judge v2_{variant} p{i + 1:02d}: attempt {attempt} failed: {str(e)[:120]}")


def v2() -> None:
    jobs = [(v, i, sc) for v, f in V2.items() for i, sc in enumerate(json.loads(f.read_text(encoding="utf-8")))]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(v2_one, jobs))


def lme_one(job: tuple[int, list[dict]]) -> None:
    k, rows = job
    cwd = OUT / "lme" / f"b{k:02d}"
    if (cwd / "verdicts.json").exists():
        return
    body = "\n".join(json.dumps({"id": r["id"], "older": {"date": r["old_date"], "text": r["old_text"]},
                                 "newer": {"date": r["new_date"], "text": r["new_text"]}}, ensure_ascii=False)
                     for r in rows)
    for attempt in range(3):
        try:
            text, _ = drive.agy(LME_PROMPT + body + "\n", MODEL, cwd)
            got = parse(text)
            if sum(1 for r in rows if r["id"] in got) < len(rows) * 0.95:
                raise ValueError("too many pair ids missing")
            (cwd / "verdicts.json").write_text(json.dumps(got), encoding="utf-8")
            return
        except Exception as e:  # noqa: BLE001
            log(f"gemini_judge lme b{k:02d}: attempt {attempt} failed: {str(e)[:120]}")


def lme() -> None:
    rows = [json.loads(line) for line in LME.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = [(k, rows[i:i + 40]) for k, i in enumerate(range(0, len(rows), 40))]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lme_one, jobs))


def report() -> None:
    from long import strayed
    stray = [str(d) for d in OUT.glob("*/*") if d.is_dir() and strayed(d)]
    if stray:
        print(f"WARNING: runs strayed outside their folders, results not trustworthy: {stray[:3]}")
    for variant, f in V2.items():
        truth_people = json.loads(f.read_text(encoding="utf-8"))
        noticed = total = retired = mistaken = done = 0
        for i, sc in enumerate(truth_people):
            lf = OUT / f"v2_{variant}" / f"p{i + 1:02d}" / "labels.json"
            truth = {(o, s["id"]) for s in sc["statements"] for o in s["replaces"]}
            truly_gone = {o for o, _ in truth}
            total += len(truth)
            if not lf.exists():
                continue
            done += 1
            lab = json.loads(lf.read_text(encoding="utf-8"))
            pred = {(o, sid) for sid, os_ in (lab.get("replaces") or {}).items() for o in os_ or []}
            noticed += len(pred & truth)
            gone = {o for o, _ in pred}
            retired += len(gone)
            mistaken += len(gone - truly_gone)
        print(f"v2 {variant:5s} ({done}/12 people): changes noticed {noticed}/{total} ({100 * noticed / max(total, 1):.0f}%); "
              f"retired {retired}, mistaken {mistaken}")
    rows = [json.loads(line) for line in LME.read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts = {}
    for f in (OUT / "lme").glob("b*/verdicts.json"):
        verdicts.update(json.loads(f.read_text(encoding="utf-8")))
    scored = [(int(bool(verdicts[r["id"]])), r["label"]) for r in rows if r["id"] in verdicts]
    tp = sum(1 for p, y in scored if p and y); fp = sum(1 for p, y in scored if p and not y)
    fn = sum(1 for p, y in scored if not p and y)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"LongMemEval KU ({len(scored)}/{len(rows)} pairs judged): P {prec:.2f}  R {rec:.2f}  F1 {f1:.2f}  "
          f"(updates caught {tp}/{tp + fn}, false alarms {fp})")


if __name__ == "__main__":
    {"v2": v2, "lme": lme, "report": report}[sys.argv[1]]()
