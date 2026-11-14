"""LongMemEval-S retrieval, MemPalace 'raw' protocol: per question a fresh index of its haystack sessions,
one document per session = user turns joined by newline, all-MiniLM-L6-v2 (chroma default), rank by similarity,
recall_any@5 against answer_session_ids, all 500 questions (abstention included).
mode raw    : the document is the session (their protocol)
mode stmts  : index each statement from system_one/extract.py (user turns), session score = best statement"""
import json, sys, time, hashlib
import numpy as np
mode, limit, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, "/home/user/Memory-system/system_one")
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
emb = DefaultEmbeddingFunction()
data = json.load(open("/tmp/lme_dl/longmemeval_s_cleaned.json"))[:limit or None]
cache = {}
def vecs(texts):
    need = [t for t in dict.fromkeys(texts) if t not in cache]
    for i in range(0, len(need), 64):
        b = need[i:i+64]
        for t, v in zip(b, emb(b)): cache[t] = np.asarray(v, dtype=np.float32)
    return np.stack([cache[t] for t in texts])
if mode == "stmts":
    from extract import statements
rows = []; t0 = time.time()
for qi, e in enumerate(data):
    sids = e["haystack_session_ids"]; sessions = e["haystack_sessions"]
    q = vecs([e["question"]])[0]
    if mode == "raw":
        docs = ["\n".join(t["content"] for t in s if t["role"] == "user") for s in sessions]
        score = vecs(docs) @ q
    else:
        score = np.full(len(sessions), -9.0, dtype=np.float32)
        for j, s in enumerate(sessions):
            st = [x for t in s if t["role"] == "user" for x in statements(t["content"])]
            if st: score[j] = (vecs(st) @ q).max()
    top = [sids[j] for j in np.argsort(-score)[:5]]
    ans = set(e["answer_session_ids"])
    rows.append({"qid": e["question_id"], "type": e["question_type"], "any5": any(a in top for a in ans), "all5": all(a in top for a in ans)})
    if qi % 25 == 0:
        print(qi, len(data), round(np.mean([r["any5"] for r in rows]), 3), round(time.time() - t0), "s", flush=True)
        json.dump(rows, open(out, "w"))
json.dump(rows, open(out, "w"))
print("DONE", round(np.mean([r["any5"] for r in rows]), 4), flush=True)
