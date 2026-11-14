"""Stale-value test on LongMemEval-S knowledge-update questions.
Evidence sessions = answer_session_ids; the LATEST by haystack date holds the current value, earlier ones are stale.
Measured per question: top1 is the current-value session / top1 is a stale session / current in top 3.
A raw (MemPalace protocol)  B statements, session = best statement  C = B + newer-fact-wins (thresholds fixed in advance)"""
import json, sys, time, re
from datetime import datetime
import numpy as np
sys.path.insert(0, "/home/user/Memory-system/system_one")
from extract import statements
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
emb = DefaultEmbeddingFunction(); cache = {}
def vecs(texts):
    need = [t for t in dict.fromkeys(texts) if t not in cache]
    for i in range(0, len(need), 64):
        b = need[i:i+64]
        for t, v in zip(b, emb(b)): cache[t] = np.asarray(v, dtype=np.float32)
    return np.stack([cache[t] for t in texts])
SIM_SAME, MARGIN = 0.75, 0.10
def day(s): return datetime.strptime(s.split(" (")[0], "%Y/%m/%d")
data = [e for e in json.load(open("/tmp/lme_dl/longmemeval_s_cleaned.json")) if e["question_type"] == "knowledge-update"]
import os
out = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {"A": [], "B": [], "C": []}
done = {r["qid"] for r in out["A"]}; t0 = time.time()
for qi, e in enumerate(data):
    sids, dates, sessions = e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]
    if e["question_id"] in done: continue
    ans = [s for s in e["answer_session_ids"] if s in sids]
    if len(ans) < 2: continue
    cur = max(ans, key=lambda s: day(dates[sids.index(s)])); stale = set(ans) - {cur}
    q = vecs([e["question"]])[0]
    raw = vecs(["\n".join(t["content"] for t in s if t["role"] == "user") for s in sessions]) @ q
    best = np.full(len(sessions), -9.0, np.float32); bestv = [None] * len(sessions); allst = []
    for j, s in enumerate(sessions):
        st = [x for t in s if t["role"] == "user" for x in statements(t["content"])]
        if not st: continue
        V = vecs(st); sc = V @ q; k = int(sc.argmax())
        best[j], bestv[j] = sc[k], V[k]; allst.append((j, V, sc))
    order = {"A": np.argsort(-raw), "B": np.argsort(-best)}
    # C: demote a session when a later session holds a near-identical statement scoring about as well
    c = best.copy(); jdate = [day(d) for d in dates]
    for j in range(len(sessions)):
        if bestv[j] is None: continue
        for j2, V2, sc2 in allst:
            if jdate[j2] > jdate[j] and (V2 @ bestv[j]).max() >= SIM_SAME and sc2.max() >= best[j] - MARGIN:
                c[j] = -5.0; break
    order["C"] = np.argsort(-c)
    for m, o in order.items():
        top = [sids[j] for j in o[:3]]
        out[m].append({"qid": e["question_id"], "top1_current": top[0] == cur, "top1_stale": top[0] in stale,
                       "cur_top3": cur in top, "n_evidence": len(ans)})
    if qi % 5 == 0:
        n = len(out["A"]); print(qi, len(data), n, {m: round(np.mean([r["top1_current"] for r in out[m]]), 3) for m in out}, round(time.time() - t0), "s", flush=True)
        json.dump(out, open(sys.argv[1], "w"))
json.dump(out, open(sys.argv[1], "w"))
print("DONE", {m: (round(np.mean([r["top1_current"] for r in v]), 3), round(np.mean([r["top1_stale"] for r in v]), 3), round(np.mean([r["cur_top3"] for r in v]), 3)) for m, v in out.items()}, len(out["A"]), flush=True)
