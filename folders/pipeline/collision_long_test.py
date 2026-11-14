#!/usr/bin/env python3
"""Long-folder collision test: whole folder file (c) vs a cloud-written index card + last N facts (d).

Set: folders/data/collision/collision_long_set.json (5 of the collision scenarios, each folder padded to
28-33 facts; the statements and the key facts are the same as in collision_set.json).

  (a) code      most recently used matching folder
  (b) names     LLM sees label, aliases and the last fact of each candidate folder
  (c) whole     LLM sees every fact of each candidate folder, in date order
  (d) card+N    LLM sees, per candidate folder, an index card plus only its most recent N facts.
                The card is written by the cloud model (free/gemini-3.8-flash) reading ONLY that folder,
                in date order, CHUNK facts per call, rewriting the card after each chunk.
No 500M model and no pre-built slot/card index: the card is only what the cloud model wrote.
Calls are spaced >= 5 s (15 requests/minute limit) and retried on 429. The key is injected by the
sandbox proxy; this script never reads or sends it. Only the synthetic test text is sent.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_test as ct

LONG = os.path.join(os.path.dirname(ct.DATA), "collision_long_set.json")
N_RECENT = 8
CHUNK = 5

CARD_SYSTEM = (
    "You maintain the index card of ONE person's folder in a personal memory system. The card lets a reader "
    "understand the folder without reading every fact, and it must survive the folder growing long. "
    "You are given the current card and the next facts, in date order. Rewrite the whole card so it covers "
    "everything seen so far, with these parts: WHO (who or what this is, relationship to the user); "
    "STABLE FACTS (things that do not change); CURRENT VALUES (latest known state: place, job, plans, dates); "
    "CHANGES (what changed and when, with dates); DISTINGUISHING DETAILS (specifics that tell this one apart "
    "from someone else with the same name). Keep every date and concrete detail that could matter later. "
    "At most 220 words. Output only the card."
)

_last = [0.0]


def throttled(prompt, system):
    gap = ct.MIN_GAP - (time.time() - _last[0])
    if gap > 0:
        time.sleep(gap)
    text, usage = ct.call(prompt, system=system, with_usage=True)
    _last[0] = time.time()
    return text, usage


def tokens(usage, prompt, system):
    return usage.get("prompt_tokens") or (len(prompt) + len(system)) // 4


def build_cards(data):
    path = os.path.join(ct.OUT, "index_cards.json")
    cards = json.load(open(path)) if os.path.exists(path) else {}
    for sc in data["scenarios"]:
        for f in sc["folders"]:
            if f["id"] in cards and cards[f["id"]].get("done"):
                continue
            facts = sorted(f["facts"], key=lambda x: x["date"])
            rec = cards.setdefault(f["id"], {"card": "", "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "done_facts": 0})
            for i in range(rec["done_facts"], len(facts), CHUNK):
                chunk = facts[i:i + CHUNK]
                prompt = (f"Folder: {f['label']} (also called: {', '.join(f['aliases'])})\n\n"
                          f"Current card:\n{rec['card'] or '(empty)'}\n\nNew facts, in date order:\n"
                          + "".join(f"[{x['date']}] {x['text']}\n" for x in chunk) + "\nWrite the updated card.")
                text, usage = throttled(prompt, CARD_SYSTEM)
                rec["card"] = text.strip()
                rec["calls"] += 1
                rec["prompt_tokens"] += tokens(usage, prompt, CARD_SYSTEM)
                rec["completion_tokens"] += usage.get("completion_tokens") or len(text) // 4
                rec["done_facts"] = i + len(chunk)
                json.dump(cards, open(path, "w"), indent=1, ensure_ascii=False)
                print(f"card {f['id']}: {rec['done_facts']}/{len(facts)} facts")
            rec["done"] = True
            json.dump(cards, open(path, "w"), indent=1, ensure_ascii=False)
    return cards


def prompt_card(sc, stmt, cards):
    folders = ct.matching(sc, stmt)
    parts = []
    for f in folders:
        recent = sorted(f["facts"], key=lambda x: x["date"])[-N_RECENT:]
        parts.append(f"Folder id: {f['id']}\nLabel: {f['label']}\nAlso called: {', '.join(f['aliases'])}\n"
                     f"Index card (written by a model from the whole folder):\n{cards[f['id']]['card']}\n"
                     f"Most recent {len(recent)} facts:\n" + "".join(f"  [{x['date']}] {x['text']}\n" for x in recent))
    return ("Matching folders:\n\n" + "\n".join(parts)
            + f"\nNew statement (dated {stmt['date']}): \"{stmt['text']}\"\n"
            + "Valid answers: " + ", ".join(f['id'] for f in folders) + ", ask")


def run(data, cond, make_prompt):
    path = os.path.join(ct.OUT, f"raw_long_{cond}.jsonl")
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            cache[r["id"]] = r
    with open(path, "a") as fh:
        for sc in data["scenarios"]:
            for st in sc["statements"]:
                if st["id"] in cache:
                    continue
                prompt = make_prompt(sc, st)
                raw, usage = throttled(prompt, ct.SYSTEM)
                valid = {f["id"] for f in ct.matching(sc, st)}
                pick, conf = ct.parse(raw, valid)
                rec = {"id": st["id"], "pick": pick, "confidence": conf, "raw": raw,
                       "prompt_tokens": tokens(usage, prompt, ct.SYSTEM)}
                cache[st["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{cond} {st['id']}: {pick} ({conf}) truth={st['answer']} tokens={rec['prompt_tokens']}")
    return cache


def main():
    data = json.load(open(LONG))
    os.makedirs(ct.OUT, exist_ok=True)
    cards = build_cards(data)
    res = {"model": ct.MODEL, "recent_facts_n": N_RECENT, "facts_per_card_call": CHUNK}
    code = {s["id"]: (ct.code_pick(sc, s), 1.0) for sc in data["scenarios"] for s in sc["statements"]}
    res["a_code_last_used"] = ct.score(data, code)
    conds = (("b_names_last_fact", lambda sc, st: ct.build_prompt(sc, st, False)),
             ("c_whole_folders", lambda sc, st: ct.build_prompt(sc, st, True)),
             ("d_card_plus_recent", lambda sc, st: prompt_card(sc, st, cards)))
    for cond, mk in conds:
        cache = run(data, cond, mk)
        r = ct.score(data, {k: (v["pick"], v["confidence"]) for k, v in cache.items()})
        r["mean_prompt_tokens_per_call"] = round(sum(v["prompt_tokens"] for v in cache.values()) / len(cache))
        res[cond] = r
    res["d_index_build"] = {"calls": sum(c["calls"] for c in cards.values()), "folders": len(cards),
                            "prompt_tokens": sum(c["prompt_tokens"] for c in cards.values()),
                            "completion_tokens": sum(c["completion_tokens"] for c in cards.values())}
    json.dump(res, open(os.path.join(ct.OUT, "summary_long.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {x: v[x] for x in v if x != "wrong_items"})


if __name__ == "__main__":
    main()
