"""Head-to-head: Hindsight (vectorize-io/hindsight, open source) on blind v2, scored like Primnox.

    hs_venv/Scripts/python hs_bench.py

Fixed before the run:
  - Hindsight 0.10.3, embedded mode (its own Postgres "pg0", its own embedder and reranker),
    with a LOCAL model for its fact extraction: Qwen 2.5 7B Instruct through Ollama on this
    PC's CPU (Hindsight's published scores use far larger cloud / 20B+ models).
  - One memory bank per person. Every statement retained once, in date order, with its own
    date as timestamp and its id as document_id; default retain settings.
  - Each question recalled with query_timestamp = the person's `today`, budget "mid", default
    settings. Results are ranked by Hindsight; each maps back to its source statement through
    document_id; repeats are collapsed in order.
  - Scored like Primnox on the same 171 questions (absent ones left out): right statement
    first / in the first three / share of answer statements in the first ten.
Writes hs_bench_results.json (ids and scores only) and prints the totals.
"""
import json
import time
from pathlib import Path

from hindsight import HindsightServer
from hindsight_client import Hindsight

BLIND = Path(__file__).resolve().parents[2] / "scripts" / "blind_memory" / "v2" / "test.json"
OUT = Path(__file__).resolve().parent / "hs_bench_results.json"


def main() -> None:
    people = json.loads(BLIND.read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    server = HindsightServer(db_url="pg0", llm_provider="ollama", llm_api_key="none", llm_model="qwen2.5:7b-instruct",
                             llm_base_url="http://localhost:11434", log_level="warning")
    server.start(timeout=900)
    try:
        c = Hindsight(base_url=server.url)
        for sc in people:
            if sc["scenario"] in done:
                continue
            bank = "blind-" + sc["scenario"]
            c.create_bank(bank_id=bank)
            t0 = time.time()
            sts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
            for s in sts:                                   # one at a time, in date order
                c.retain(bank_id=bank, content=s["text"], timestamp=f"{s['date']}T12:00:00Z", document_id=s["id"])
            t_retain = time.time() - t0
            rows = []
            for q in sc["questions"]:
                if q["type"] == "absent" or not q.get("answer_ids"):
                    continue
                t1 = time.time()
                res = c.recall(bank_id=bank, query=q["text"], query_timestamp=f"{sc['today']}T12:00:00")
                d = res.to_dict() if hasattr(res, "to_dict") else res
                order = []
                for r in d.get("results") or []:
                    doc = r.get("document_id")
                    if doc and doc not in order:
                        order.append(doc)
                rows.append({"q": q["id"], "type": q["type"], "want": q["answer_ids"], "order": order[:10],
                             "seconds": round(time.time() - t1, 2)})
            done[sc["scenario"]] = {"retain_seconds": round(t_retain), "rows": rows}
            OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
            hit1 = sum(r["order"][:1] and r["order"][0] in r["want"] for r in rows)
            print(f"{sc['scenario']}: retained {len(sts)} in {t_retain:.0f}s; first {hit1}/{len(rows)}", flush=True)
    finally:
        server.stop()
    rows = [r for p in done.values() for r in p["rows"]]
    for t in ("all", "current", "past", "multi"):
        rs = rows if t == "all" else [r for r in rows if r["type"] == t]
        if not rs:
            continue
        t1 = sum(bool(r["order"]) and r["order"][0] in r["want"] for r in rs)
        t3 = sum(bool(set(r["order"][:3]) & set(r["want"])) for r in rs)
        r10 = sum(len(set(r["order"][:10]) & set(r["want"])) / len(r["want"]) for r in rs)
        print(f"{t + f' ({len(rs)})':16s} {100 * t1 / len(rs):5.1f} / {100 * t3 / len(rs):5.1f} / {100 * r10 / len(rs):5.1f}")


if __name__ == "__main__":
    main()
