"""REALTALK end to end: real people's messages through Primnox's memory.   (proof only, never trained on)

    python realtalk_e2e.py extract    # Gemini (agy) writes down each message's facts -> realtalk_extract/
    python realtalk_e2e.py build      # three inputs from the same messages -> ../realtalk/e2e_{raw,rules,llm}.json

REALTALK (Lee et al., 2025): 10 chats between real people over ~21 days, 8,944 messages,
728 memory-probing questions, each tagged with the message(s) that answer it. One
scenario per chat, in lme_e2e.py's schema, so `lme_e2e.py score` runs it unchanged:
statements {id, date, text} (id = "<dia_id>#<k>"), questions {id, type, text,
answer_ids} where answer_ids are every statement taken from an evidence message.

  raw    each message as one statement, "Speaker: text"
  rules  extract.py's splitter on each message, prefixed with the speaker
  llm    the facts Gemini wrote down, third person, named and dated
Statements over the memory's 240-character limit are dropped and counted.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import drive
from drive import log, parse

HERE = Path(__file__).resolve().parent
RT = HERE.parent / "realtalk"
EXT = HERE / "realtalk_extract"
MODEL = "gemini-3.8-flash-medium"
BATCH = 50
MAX_CHARS = 240
CAT = {1: "multi-hop", 2: "temporal", 3: "other"}

PROMPT = """You are the note-taker for a long-term memory. Below are messages from a real chat
between two people, each with its id, speaker and date. For each message, write down every
fact it states about the speaker's life (or the people, pets, places and things in it) that
would be worth remembering later.

Rules for each fact:
- One self-contained fact, third person, starting with the speaker's name: "Emi adopted a
  kitten named Luna in January 2024". 5-25 words.
- Resolve references and relative dates using the message and its date ("yesterday" in a
  message dated 2024-01-14 -> "on 13 January 2024"); never write "it", "there", "this".
- Keep names, numbers, places and brands exactly as written.
- Greetings, reactions, jokes, questions and small talk are NOT facts. A message with no
  fact gets [].

Work only from the messages below. Do not use any tools: do not read, list or search any
files or folders, run commands or browse.

Reply with JSON only: {"<message id>": ["fact", ...], ...} with every message id.

Messages:
"""


def chats() -> list[dict]:
    out = []
    for f in sorted(RT.glob("data/Chat_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        msgs = []
        for k in sorted((k for k in d if k.startswith("session_") and isinstance(d[k], list)),
                        key=lambda k: int(k.split("_")[1])):
            for t in d[k]:
                text = " ".join(str(t.get("clean_text") or "").split())
                if text:
                    msgs.append({"dia": t["dia_id"], "speaker": t["speaker"], "date": day(t["date_time"]),
                                 "text": text})
        out.append({"name": f.stem, "messages": msgs, "qa": d.get("qa", [])})
    return out


def day(stamp: str) -> str:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y, %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(stamp.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return stamp.strip()[:10]


def extract_one(job: tuple[str, int, list[dict]]) -> None:
    name, k, batch = job
    cwd = EXT / name / f"b{k:03d}"
    if (cwd / "facts.json").exists():
        return
    body = "\n".join(json.dumps({"id": m["dia"], "speaker": m["speaker"], "date": m["date"], "text": m["text"]},
                                ensure_ascii=False) for m in batch)
    for attempt in range(3):
        try:
            text, _ = drive.agy(PROMPT + body + "\n", MODEL, cwd)
            got = parse(text)
            miss = [m["dia"] for m in batch if not isinstance(got.get(m["dia"]), list)]
            if len(miss) > len(batch) // 10:
                raise ValueError(f"{len(miss)} ids missing")
            out = {m["dia"]: [str(f).strip() for f in (got.get(m["dia"]) or []) if str(f).strip()] for m in batch}
            (cwd / "facts.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return
        except Exception as e:  # noqa: BLE001
            log(f"realtalk {name} b{k:03d}: attempt {attempt} failed: {str(e)[:150]}")


def extract() -> None:
    jobs = [(c["name"], k, c["messages"][i:i + BATCH])
            for c in chats() for k, i in enumerate(range(0, len(c["messages"]), BATCH))]
    log(f"realtalk: {sum(len(j[2]) for j in jobs)} messages in {len(jobs)} batches with {MODEL}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(extract_one, jobs))
    done = sum(1 for n, k, _ in jobs if (EXT / n / f"b{k:03d}" / "facts.json").exists())
    print(f"realtalk extraction: {done}/{len(jobs)} batches")


def build() -> None:
    sys.path.insert(0, str(HERE.parent / "ms" / "system_one"))
    from extract import statements
    from long import strayed
    stray = [str(d) for d in EXT.glob("*/b*") if strayed(d)]
    if stray:
        raise SystemExit(f"extraction runs strayed outside their folders: {stray[:3]} — redo those batches")
    llm: dict[tuple[str, str], list[str]] = {}
    for f in EXT.glob("*/b*/facts.json"):
        for dia, facts in json.loads(f.read_text(encoding="utf-8")).items():
            llm[(f.parent.parent.name, dia)] = facts
    for variant in ("raw", "rules", "llm"):
        scenarios, counts = [], {"statements": 0, "too_long": 0, "questions": 0, "answer_lost": 0,
                                 "messages_without_llm_facts": 0}
        for c in chats():
            sts, by_dia = [], {}
            for m in c["messages"]:
                if variant == "raw":
                    pieces = [f"{m['speaker']}: {m['text']}"]
                elif variant == "rules":
                    pieces = [f"{m['speaker']}: {p}" for p in statements(m["text"])]
                else:
                    pieces = llm.get((c["name"], m["dia"]))
                    if pieces is None:
                        counts["messages_without_llm_facts"] += 1
                        pieces = []
                for k, text in enumerate(pieces):
                    if len(text) > MAX_CHARS:
                        counts["too_long"] += 1
                        continue
                    sid = f"{m['dia']}#{k}"
                    sts.append({"id": sid, "date": m["date"], "text": text})
                    by_dia.setdefault(m["dia"], []).append(sid)
            qs = []
            for i, q in enumerate(c["qa"]):
                ans = [sid for dia in q.get("evidence") or [] for sid in by_dia.get(dia, [])]
                if not ans:
                    counts["answer_lost"] += 1
                    continue
                qs.append({"id": f"q{i + 1}", "type": CAT.get(q.get("category"), str(q.get("category"))),
                           "text": q["question"], "answer_ids": ans, "stale_ids": []})
            counts["statements"] += len(sts)
            counts["questions"] += len(qs)
            scenarios.append({"scenario": c["name"], "today": max(m["date"] for m in c["messages"]),
                              "statements": sts, "questions": qs})
        out = RT / f"e2e_{variant}.json"
        out.write_text(json.dumps(scenarios, ensure_ascii=False), encoding="utf-8")
        print(f"{variant}: {counts}")


if __name__ == "__main__":
    {"extract": extract, "build": build}[sys.argv[1]]()
