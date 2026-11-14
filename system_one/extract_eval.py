"""Score extract.py against the reference facts written for LongMemEval's 467
knowledge-update messages (folders/data/lme_facts/). No model, runs anywhere.

The reference facts are an LLM's output, not human labels, so these are a proxy:
  recall     share of reference facts covered by some extracted statement
             (a statement covers a fact when it holds >= 70% of the fact's content words)
  precision  share of extracted statements that cover some reference fact
  empty      messages where the rules keep nothing (before the whole-text fallback)
  stmts/msg  statements per message (fewer, tighter statements compare better)

    python system_one/extract_eval.py --module extract --tag v2
    python system_one/extract_eval.py --module path/to/extract_v1.py --tag v1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTS = HERE.parent / "folders" / "data" / "lme_facts" / "facts_lme_ku_v2-q7e1.json"
STOP = set("a an the and or but of to in on at for with is are was were be been i my me we our it its that this "
           "as by from so just also have has had do did does not no".split())


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP}


def covers(stmt: str, fact: str, thr: float = 0.7) -> bool:
    f = words(fact)
    return bool(f) and len(f & words(stmt)) / len(f) >= thr


def load(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location("ex", p if p.suffix == ".py" else HERE / f"{path}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.statements


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="extract")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    statements = load(args.module)
    data = json.loads(FACTS.read_text(encoding="utf-8"))
    n_fact = hit_fact = n_stmt = hit_stmt = empty = n_msg = 0
    for msg, facts in data.items():
        facts = [f for f in facts if f.strip()]
        if not facts:
            continue
        n_msg += 1
        st = statements(msg, keep_all_if_none=False)
        empty += not st
        n_stmt += len(st)
        n_fact += len(facts)
        hit_fact += sum(any(covers(s, f) for s in st) for f in facts)
        hit_stmt += sum(any(covers(s, f) for f in facts) for s in st)
    rec, prec = hit_fact / n_fact, hit_stmt / max(n_stmt, 1)
    out = {"tag": args.tag, "messages": n_msg, "reference_facts": n_fact, "statements": n_stmt,
           "recall": round(rec, 3), "precision": round(prec, 3),
           "f1": round(2 * rec * prec / max(rec + prec, 1e-9), 3),
           "empty_messages": f"{empty}/{n_msg}", "statements_per_message": round(n_stmt / n_msg, 2)}
    print(json.dumps(out))
    (HERE / "results" / f"extract_{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
