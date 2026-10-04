"""Zero-shot screen: can an untrained decision model tell when a new fact
replaces an old one?

Pairs come from the DEV data only (`dev.json`, `change.json`): for every
statement, every earlier statement of the same person is a candidate; the
label is whether the later one lists it in `replaces`. The frozen test sets
are never read.

Scorers (each caches its scores in results/scores_<name>.json):
  laya       convaiinnovations/laya, a yes/no question and a relation choice
  nli        cross-encoder/nli-deberta-v3-small, P(contradiction)
The current rule-based system is scored from `rules_pairs.py` output.

Reported: average precision (threshold-free), best F1 over thresholds (picked
on the same data, so optimistic), F1 at 0.5, on all pairs and on a
production-like shortlist (the 5 most similar earlier statements by MiniLM).

    system_one/.venv/Scripts/python system_one/screen.py --scorers nli laya
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
DEV_FILES = ["scripts/blind_memory/dev.json", "scripts/blind_memory/change.json"]
SHORTLIST = 5

NOUL = ("Does the later message mean the earlier statement is no longer true now "
        "(it was replaced, changed or ended)?")
RELATION = {
    "replaces": "the earlier statement is no longer true now; the later one changed or ended it",
    "adds": "both stay true: a second or additional value of the same kind (another pet, another language)",
    "detail": "both stay true: the later one adds detail to the same fact",
    "other": "about a different person, thing or topic",
    "event": "a one-off event or plan that does not change the earlier fact",
}


def load_pairs() -> list[dict]:
    pairs = []
    for f in DEV_FILES:
        for sc in json.loads((ROOT / f).read_text(encoding="utf-8")):
            sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
            for i, new in enumerate(sts):
                gone = set(new.get("replaces") or [])
                for old in sts[:i]:
                    pairs.append({"scenario": sc["scenario"], "old": old["id"], "new": new["id"],
                                  "old_text": old["text"], "new_text": new["text"],
                                  "old_date": old["date"], "new_date": new["date"],
                                  "label": int(old["id"] in gone)})
    return pairs


def key(p: dict) -> str:
    return f'{p["scenario"]}|{p["old"]}|{p["new"]}'


def state_text(p: dict) -> str:
    return (f'Earlier ({p["old_date"]}) the user said: "{p["old_text"]}"\n'
            f'Later ({p["new_date"]}) the user said: "{p["new_text"]}"')


def score_laya(pairs: list[dict], save=None) -> dict:
    from laya import Router
    router = Router()
    questions = {
        "replaces": {"type": "noul", "instructions": NOUL},
        "relation": {"type": "choice", "instructions": "How does the later message relate to the earlier statement?",
                     "criteria": RELATION},
    }
    out = {}
    t0 = time.perf_counter()
    for n, p in enumerate(pairs, 1):
        ans = router.predict(state_text(p), questions)["answers"]
        out[key(p)] = {"noul": float(ans["replaces"]["noul"]),
                       "choice_replaces": float(ans["relation"]["probabilities"]["replaces"]),
                       "choice": ans["relation"]["choice"]}
        if n % 100 == 0:
            print(f"laya {n}/{len(pairs)}  {(time.perf_counter() - t0) / n * 1000:.0f} ms/pair", flush=True)
            if save:
                save(out)   # ~0.7 s a pair on CPU: never lose a long run to a restart
    return out


LAYA_FT = None   # set from --laya-ft: a fine-tuned checkpoint directory


def score_laya_ft(pairs: list[dict], save=None) -> dict:
    """A fine-tuned Laya, asked only the relation question it was trained on, in batches."""
    from laya.agent import Agent
    agent = Agent(str(LAYA_FT))
    questions = {"relation": {"type": "choice",
                              "instructions": "How does the later message relate to the earlier statement?",
                              "criteria": RELATION}}
    out, t0 = {}, time.perf_counter()
    for i in range(0, len(pairs), 16):
        batch = pairs[i:i + 16]
        results = agent.predict_batch([state_text(p) for p in batch], questions)
        for p, r in zip(batch, results):
            rel = r["answers"]["relation"]
            out[key(p)] = {"choice_replaces": float(rel["probabilities"]["replaces"]), "choice": rel["choice"]}
        if (i // 16) % 10 == 9:
            print(f"laya_ft {i + len(batch)}/{len(pairs)}  {(time.perf_counter() - t0) / (i + len(batch)) * 1000:.0f} ms/pair",
                  flush=True)
            if save:
                save(out)
    return out


def score_nli(pairs: list[dict], save=None) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "cross-encoder/nli-deberta-v3-small"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    contra = next(i for i, l in model.config.id2label.items() if l.lower().startswith("contra"))
    out = {}
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            batch = pairs[i:i + 32]
            enc = tok([p["old_text"] for p in batch], [p["new_text"] for p in batch],
                      padding=True, truncation=True, return_tensors="pt")
            probs = model(**enc).logits.softmax(-1)[:, contra].tolist()
            for p, s in zip(batch, probs):
                out[key(p)] = {"contradiction": s}
    return out


def shortlist(pairs: list[dict]) -> set[str]:
    """Keys of the SHORTLIST most similar earlier statements for each new one (MiniLM cosine)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    texts = sorted({p["old_text"] for p in pairs} | {p["new_text"] for p in pairs})
    vec = {}
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc = tok(chunk, padding=True, truncation=True, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            vec.update(zip(chunk, e))
    by_new: dict[tuple, list] = {}
    for p in pairs:
        by_new.setdefault((p["scenario"], p["new"]), []).append(
            (float(vec[p["old_text"]] @ vec[p["new_text"]]), key(p)))
    keep = set()
    for cands in by_new.values():
        keep.update(k for _, k in sorted(cands, reverse=True)[:SHORTLIST])
    return keep


def metrics(y: list[int], s: list[float]) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    pr, rc, _ = precision_recall_curve(y, s)
    f1 = max((2 * a * b / (a + b) for a, b in zip(pr, rc) if a + b), default=0.0)
    tp = sum(1 for a, b in zip(y, s) if a and b >= 0.5)
    pp, pos = sum(1 for b in s if b >= 0.5), sum(y)
    f05 = 2 * tp / (pp + pos) if pp + pos else 0.0
    return {"ap": average_precision_score(y, s), "best_f1": f1, "f1_at_0.5": f05,
            "precision_at_0.5": tp / pp if pp else 0.0, "recall_at_0.5": tp / pos if pos else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", nargs="*", default=["nli", "laya"])
    ap.add_argument("--limit", type=int, help="score only the first N pairs (smoke test)")
    ap.add_argument("--scope", choices=("all", "shortlist"), default="shortlist",
                    help="which pairs slow scorers (laya) score; nli always scores all")
    ap.add_argument("--laya-ft", help="fine-tuned Laya checkpoint directory (scorer laya_ft)")
    ap.add_argument("--tag", default="", help="suffix for the laya_ft score cache, one per checkpoint")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    pairs = load_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} pairs, {sum(p['label'] for p in pairs)} true replacements", flush=True)

    global LAYA_FT
    LAYA_FT = args.laya_ft
    short = shortlist(pairs)
    scores = {}
    for name in args.scorers:
        cache = OUT / f"scores_{name}{'_' + args.tag if name == 'laya_ft' and args.tag else ''}.json"
        have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        wanted = pairs if (name == "nli" or args.scope == "all") else [p for p in pairs if key(p) in short]
        todo = [p for p in wanted if key(p) not in have]
        if todo:
            t0 = time.perf_counter()
            save = lambda part, have=have, cache=cache: cache.write_text(json.dumps({**have, **part}), encoding="utf-8")
            have.update({"laya": score_laya, "nli": score_nli, "laya_ft": score_laya_ft}[name](todo, save))
            print(f"{name}: {len(todo)} pairs in {time.perf_counter() - t0:.0f}s", flush=True)
            cache.write_text(json.dumps(have), encoding="utf-8")
        scores[name] = have

    report = {"pairs": len(pairs), "positives": sum(p["label"] for p in pairs),
              "shortlist_pairs": len(short),
              "shortlist_positives": sum(p["label"] for p in pairs if key(p) in short), "rows": []}

    rules_file = OUT / "rules_dev.json"
    if rules_file.exists():
        rules = json.loads(rules_file.read_text(encoding="utf-8"))
        got = {f"{sc}|{o}|{n}" for sc, r in rules.items() for o, n in r["pairs"]}
        for scope, sel in (("all", pairs), ("shortlist", [p for p in pairs if key(p) in short])):
            y = [p["label"] for p in sel]
            hit = [int(key(p) in got) for p in sel]
            tp = sum(a and b for a, b in zip(y, hit))
            prec, rec = (tp / sum(hit) if sum(hit) else 0.0), (tp / sum(y) if sum(y) else 0.0)
            report["rows"].append({"scorer": "rules (current system)", "scope": scope,
                                   "precision": prec, "recall": rec,
                                   "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})

    fields = {"laya": ["noul", "choice_replaces"], "nli": ["contradiction"], "laya_ft": ["choice_replaces"]}
    for name, sc in scores.items():
        for field in fields[name]:
            for scope, sel in (("all", pairs), ("shortlist", [p for p in pairs if key(p) in short])):
                if any(key(p) not in sc for p in sel):
                    continue   # this scorer did not score every pair in this scope
                y = [p["label"] for p in sel]
                s = [sc[key(p)][field] for p in sel]
                report["rows"].append({"scorer": f"{name}:{field}", "scope": scope, **metrics(y, s)})

    (OUT / "screen_dev.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f'\npairs {report["pairs"]} (positives {report["positives"]}); shortlist '
          f'{report["shortlist_pairs"]} (positives {report["shortlist_positives"]})')
    for r in report["rows"]:
        vals = "  ".join(f"{k}={v:.2f}" for k, v in r.items() if isinstance(v, float))
        print(f'{r["scorer"]:<26} {r["scope"]:<9} {vals}')


if __name__ == "__main__":
    main()
