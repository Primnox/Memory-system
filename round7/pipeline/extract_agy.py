"""LLM fact extraction through agy, for the LongMemEval knowledge-update pairs.

    python extract_agy.py run      # -> extract/facts_lme_ku.json  {message text: [facts]}
    python extract_agy.py report   # counts only

Every distinct message (old and new side of each pair) goes to a Gemini model in
batches with its date; the model returns the facts a careful note-taker would
save: self-contained, first person, references and relative dates resolved,
requests and questions dropped. The prompt is generic (nothing about this
benchmark), so it doubles as a candidate prompt for the app.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drive
from drive import log, parse

HERE = Path(__file__).resolve().parent
OUT = HERE / "extract"
PAIRS = HERE.parent / "ms" / "system_one" / "data" / "blind_external" / "pairs_lme_ku.jsonl"
MODEL = "gemini-3.8-flash-medium"
BATCH = 24

PROMPT = """You are the note-taker for a personal assistant's long-term memory. For each message
the user sent, write down every fact it states about the user's own life (or about the
people, pets and things in it) that would be worth remembering later.

Rules for each fact:
- One self-contained fact, first person, 5-25 words ("I set a 5K personal best of 27:12").
- Resolve every reference and relative date using the message and its date ("last month"
  in a message dated 2023-05-25 -> "in April 2023"); never write "it", "there", "this".
- Keep numbers, names, places, brands and dates exactly as given.
- Requests, questions, thanks, hypotheticals and things the user only asks about are NOT
  facts. Plans and intentions the user states about themselves ARE facts ("I plan to...").
- No facts at all is a valid answer: [].

Work only from the messages below. Do not use any tools: do not read, list or search any
files or folders, run commands or browse. Everything you need is in this prompt.

Reply with JSON only: {"m1": ["fact", ...], "m2": [...], ...} with every message id.

Messages:
"""


def batches() -> list[list[tuple[str, str, str]]]:
    rows = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[str, str] = {}
    for r in rows:
        for side in ("old", "new"):
            seen.setdefault(r[f"{side}_text"], r[f"{side}_date"])
    items = [(f"m{i + 1}", t, dt) for i, (t, dt) in enumerate(seen.items())]
    return [items[i:i + BATCH] for i in range(0, len(items), BATCH)]


def one(k: int, batch: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    cwd = OUT / f"b{k:03d}"
    done = cwd / "facts.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    body = "\n".join(json.dumps({"id": mid, "date": dt, "text": t}, ensure_ascii=False) for mid, t, dt in batch)
    for attempt in range(3):
        try:
            text, _ = drive.agy(PROMPT + body + "\n", MODEL, cwd)
            got = parse(text)
            missing = [mid for mid, _, _ in batch if not isinstance(got.get(mid), list)]
            if missing:
                raise ValueError(f"missing ids {missing[:5]}")
            out = {t: [str(f).strip() for f in got[mid] if str(f).strip()] for mid, t, _ in batch}
            done.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
            log(f"extract b{k:03d}: {sum(map(len, out.values()))} facts from {len(batch)} messages")
            return out
        except Exception as e:  # noqa: BLE001
            log(f"extract b{k:03d}: attempt {attempt} failed: {str(e)[:160]}")
    return {}


def run() -> None:
    OUT.mkdir(exist_ok=True)
    bs = batches()
    log(f"extract: {sum(map(len, bs))} messages in {len(bs)} batches with {MODEL}")
    facts: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for part in pool.map(lambda kb: one(*kb), enumerate(bs)):
            facts.update(part)
    (OUT / "facts_lme_ku.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    report()


def report() -> None:
    bs = batches()
    total = sum(map(len, bs))
    f = OUT / "facts_lme_ku.json"
    facts = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    n = [len(v) for v in facts.values()]
    print(f"{len(facts)}/{total} messages extracted; {sum(n)} facts; "
          f"{sum(1 for x in n if x == 0)} messages with none; max {max(n) if n else 0} in one message")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
