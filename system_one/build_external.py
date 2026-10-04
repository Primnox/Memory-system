"""External blind test pairs: data not written by Claude agents.

1. Dialogue NLI, verified test split (Welleck et al. 2019, MIT; written by
   crowdworkers): a persona fact, then a later utterance. contradiction ->
   should replace; entailment and neutral -> should not. The shipped memory
   head never saw any Dialogue NLI. Sampled to --per-label pairs per label.
2. LongMemEval, knowledge-update questions (Wu et al., MIT; long, chatty user
   messages from GPT-4o-generated sessions). User turns flagged `has_answer`
   are the evidence. Each evidence turn in an earlier session paired with each
   evidence turn in the latest evidence session -> should replace. The earlier
   evidence turn paired with the latest session's other user turns (same
   topic, no change) -> should not.

Writes data/blind_external/pairs_dnli.jsonl and pairs_lme_ku.jsonl:
{"id", "source", "old_text", "new_text", "old_date", "new_date", "label"}.
"""
from __future__ import annotations

import argparse
import json
import random
import zipfile
from datetime import datetime
from pathlib import Path

from build_dnli import tidy

HERE = Path(__file__).resolve().parent
EXT = HERE / "data" / "external"
MAX_CHARS = 600   # long chat turns are cut, keeping the start


def iso(lme_date: str) -> str:
    return datetime.strptime(lme_date.split(" (")[0], "%Y/%m/%d").date().isoformat()


def dnli(per_label: int, seed: int) -> list[dict]:
    z = zipfile.ZipFile(EXT / "dnli.zip")
    rows = json.loads(z.read("dnli/dialogue_nli/dialogue_nli_verified_test.jsonl"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out, taken = [], {"negative": 0, "positive": 0, "neutral": 0}
    for i, ex in enumerate(rows):
        lab = ex["label"]
        if taken.get(lab, per_label) >= per_label:
            continue
        old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
        if len(old) < 8 or len(new) < 8:
            continue
        taken[lab] += 1
        out.append({"id": f"dnli-{i}", "source": f"dnli:{lab}", "old_text": old, "new_text": new,
                    "old_date": "2025-03-01", "new_date": "2025-06-01", "label": int(lab == "negative")})
    print("dnli verified test:", len(rows), "rows; sampled", taken)
    return out


def lme() -> list[dict]:
    data = json.loads((EXT / "longmemeval_oracle.json").read_text(encoding="utf-8"))
    out = []
    for item in data:
        if item["question_type"] != "knowledge-update":
            continue
        sessions = list(zip(item["haystack_dates"], item["haystack_sessions"]))
        sessions.sort(key=lambda s: s[0])
        ev = [(si, ti, d, t) for si, (d, s) in enumerate(sessions) for ti, t in enumerate(s)
              if t["role"] == "user" and t.get("has_answer")]
        if not ev:
            continue
        last = max(si for si, *_ in ev)
        newest = [e for e in ev if e[0] == last]
        older = [e for e in ev if e[0] < last]
        others = [(last, ti, sessions[last][0], t) for ti, t in enumerate(sessions[last][1])
                  if t["role"] == "user" and not t.get("has_answer")]
        for o in older:
            for label, group in ((1, newest), (0, others)):
                for n in group:
                    out.append({"id": f'{item["question_id"]}-{o[0]}.{o[1]}-{n[0]}.{n[1]}', "source": "lme_ku",
                                "old_text": o[3]["content"][:MAX_CHARS], "new_text": n[3]["content"][:MAX_CHARS],
                                "old_date": iso(o[2]), "new_date": iso(n[2]), "label": label})
    print("longmemeval knowledge-update pairs:", len(out), "positives", sum(p["label"] for p in out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    for name, rows in (("pairs_dnli", dnli(args.per_label, args.seed)), ("pairs_lme_ku", lme())):
        with open(HERE / "data" / "blind_external" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
