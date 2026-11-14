#!/usr/bin/env python3
"""Name-collision test: which folder does a statement belong to when two folders share a name?

Three conditions on folders/data/collision/collision_set.json:
  code   - among folders whose name/alias appears in the statement, pick the most recently used
  names  - an LLM sees only each folder's label, aliases and last fact
  whole  - an LLM sees the WHOLE of each matching folder (every fact)
The LLM may also answer "ask" (ask the user).

The API key is not read or sent by this script: the sandbox proxy injects it for api.apinex.bond.
Only the synthetic test text in collision_set.json is sent.

    python3 folders/pipeline/collision_test.py            # all conditions (resumes from cache)
    python3 folders/pipeline/collision_test.py --no-llm   # code baseline only
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "collision", "collision_set.json")
OUT = os.path.join(HERE, "..", "results", "collision")
URL = "https://api.apinex.bond/v1/chat/completions"
MODEL = "free/gemini-3.8-flash"
MIN_GAP = 5.0          # seconds between calls (limit: 15 requests / minute)
CONFIDENT = 0.8        # a wrong non-"ask" answer at or above this is "confidently wrong"

SYSTEM = (
    "You are the filing clerk of a personal memory system. Memory is a set of folders, one per person/pet/thing. "
    "A new statement from the user mentions a name that matches more than one folder. Decide which folder the "
    "statement belongs in. If the information given does not let you tell, answer \"ask\" (the system will then ask "
    "the user); asking is better than guessing wrong. Reply with ONLY a JSON object: "
    "{\"folder\": \"<folder id or ask>\", \"confidence\": <number 0 to 1>}"
)


def words_match(text, name):
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) is not None


def matching(sc, stmt):
    return [f for f in sc["folders"] if any(words_match(stmt["text"], n) for n in f["aliases"])]


def last_used(f):
    return max(x["date"] for x in f["facts"])


def code_pick(sc, stmt):
    m = matching(sc, stmt)
    return max(m, key=last_used)["id"] if m else "ask"


def render_folder(f, whole):
    facts = sorted(f["facts"], key=lambda x: x["date"])
    shown = facts if whole else facts[-1:]
    head = f"Folder id: {f['id']}\nLabel: {f['label']}\nAlso called: {', '.join(f['aliases'])}\n"
    body = "".join(f"  [{x['date']}] {x['text']}\n" for x in shown)
    return head + ("Facts:\n" if whole else "Last fact:\n") + body


def build_prompt(sc, stmt, whole):
    folders = matching(sc, stmt)
    return ("Matching folders:\n\n" + "\n".join(render_folder(f, whole) for f in folders)
            + f"\nNew statement (dated {stmt['date']}): \"{stmt['text']}\"\n"
            + "Valid answers: " + ", ".join(f['id'] for f in folders) + ", ask")


def call(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 1024,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}).encode()
    for attempt in range(8):
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 15 * (attempt + 1)
                print(f"  HTTP {e.code}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10)
    raise RuntimeError("giving up after retries")


def parse(raw, valid):
    m = re.search(r"\{.*?\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0))
        folder = str(d.get("folder", "")).strip()
        conf = float(d.get("confidence", 1.0))
    except Exception:
        return "invalid", 0.0
    if folder.lower() == "ask":
        return "ask", conf
    return (folder, conf) if folder in valid else ("invalid", conf)


def run_condition(data, cond, whole):
    path = os.path.join(OUT, f"raw_{cond}.jsonl")
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            cache[r["id"]] = r
    last = 0.0
    with open(path, "a") as fh:
        for sc in data["scenarios"]:
            for st in sc["statements"]:
                if st["id"] in cache:
                    continue
                gap = MIN_GAP - (time.time() - last)
                if gap > 0:
                    time.sleep(gap)
                prompt = build_prompt(sc, st, whole)
                raw = call(prompt)
                last = time.time()
                valid = {f["id"] for f in matching(sc, st)}
                pick, conf = parse(raw, valid)
                rec = {"id": st["id"], "pick": pick, "confidence": conf, "raw": raw}
                cache[st["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{cond} {st['id']}: {pick} ({conf}) truth={st['answer']}")
    return cache


def score(data, picks):
    """picks: {statement id: (pick, confidence)}"""
    n = right = ask = ask_right = conf_wrong = over_ask = 0
    lv = {"easy": [0, 0], "hard": [0, 0]}
    n_ask_truth = sum(1 for sc in data["scenarios"] for s in sc["statements"] if s["answer"] == "ask")
    wrong_items = []
    for sc in data["scenarios"]:
        for s in sc["statements"]:
            pick, conf = picks[s["id"]]
            n += 1
            ok = pick == s["answer"]
            right += ok
            lv[s["level"]][0] += ok
            lv[s["level"]][1] += 1
            if pick == "ask":
                ask += 1
                ask_right += s["answer"] == "ask"
                over_ask += s["answer"] != "ask"
            elif not ok and conf >= CONFIDENT:
                conf_wrong += 1
            if not ok:
                wrong_items.append({"id": s["id"], "text": s["text"], "truth": s["answer"], "pick": pick, "confidence": conf})
    n_resolv = n - n_ask_truth
    resolv_right = sum(1 for sc in data["scenarios"] for s in sc["statements"]
                       if s["answer"] != "ask" and picks[s["id"]][0] == s["answer"])
    return {
        "statements": n, "should_ask": n_ask_truth, "correct": right, "accuracy": round(right / n, 3),
        "accuracy_resolvable_only": round(resolv_right / n_resolv, 3), "resolvable": n_resolv,
        "easy": f"{lv['easy'][0]}/{lv['easy'][1]}", "hard": f"{lv['hard'][0]}/{lv['hard'][1]}",
        "confident_wrong": conf_wrong, "asked": ask, "asked_rightly": ask_right, "asked_needlessly": over_ask,
        "wrong_items": wrong_items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    data = json.load(open(DATA))
    os.makedirs(OUT, exist_ok=True)
    results = {"model": MODEL, "confident_threshold": CONFIDENT}
    code = {s["id"]: (code_pick(sc, s), 1.0) for sc in data["scenarios"] for s in sc["statements"]}
    results["code_last_used"] = score(data, code)
    if not a.no_llm:
        for cond, whole in (("names_last_fact", False), ("whole_folders", True)):
            cache = run_condition(data, cond, whole)
            results[cond] = score(data, {k: (v["pick"], v["confidence"]) for k, v in cache.items()})
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in results.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(k, {x: v[x] for x in v if x != "wrong_items"})


if __name__ == "__main__":
    main()
