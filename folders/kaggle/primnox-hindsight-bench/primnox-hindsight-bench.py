"""Kaggle GPU job: Hindsight (vectorize-io/hindsight, open source, MIT) on blind v2, scored like Primnox.

Same rules as hs_bench.py, fixed before the run: Hindsight 0.10.x embedded (own Postgres, embedder,
reranker), its fact extraction run by a LOCAL model — Qwen 2.5 7B Instruct through Ollama on a T4
(Hindsight's published scores use far larger models); one bank per person; every statement
retained once in date order with its date and id; each question recalled with
query_timestamp = the person's today, budget "mid"; results mapped back to statements by
document_id; scored first / top 3 / share in top 10 on the 171 answerable questions.
Blind v2 comes from the public Memory-system repo.
"""
import json
import subprocess
import time
from pathlib import Path

OUT_DIR = Path("/kaggle/working")


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a /kaggle/working/run.log | tail -n 5"], check=True)


sh("apt-get -qq update && apt-get -qq install -y zstd")
sh("curl -fsSL https://ollama.com/install.sh | sh")
subprocess.Popen(["bash", "-c", "OLLAMA_NUM_PARALLEL=4 ollama serve > /kaggle/working/ollama.log 2>&1"])
time.sleep(10)
sh("ollama pull qwen2.5:7b-instruct")
# warm the model into GPU memory first: the first call (model load) outlasted the client timeout last run
sh("ollama run qwen2.5:7b-instruct 'Say OK.'")
import os
os.environ["HINDSIGHT_API_LLM_TIMEOUT"] = "900"
sh("pip install -q hindsight-all")
sh("rm -rf /tmp/ms && git clone -q --depth 1 https://github.com/Primnox/Memory-system /tmp/ms")

from hindsight import HindsightServer
from hindsight_client import Hindsight

BLIND = Path("/tmp/ms/scripts/blind_memory/v2/test.json")
OUT = OUT_DIR / "hs_bench_results.json"


def main() -> None:
    people = json.loads(BLIND.read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    server = HindsightServer(db_url="pg0", llm_provider="ollama", llm_api_key="none", llm_model="qwen2.5:7b-instruct",
                             llm_base_url="http://localhost:11434", log_level="warning")
    server.start(timeout=900)
    try:
        c = Hindsight(base_url=server.url, timeout=1800)
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
            open("/kaggle/working/progress.log", "a").write(sc["scenario"] + " done" + chr(10))
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



main()
