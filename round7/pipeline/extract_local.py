"""The same fact extraction as extract_agy.py, done by a LOCAL model through Ollama.

    python extract_local.py qwen2.5:3b-instruct-q4_K_M      # -> extract_local/<model>/facts_lme_ku.json
    python extract_local.py qwen2.5:7b-instruct

Same note-taker prompt, same 467 LongMemEval messages, smaller batches (small models
lose track of long id lists). Resumable: finished batches are kept. Prints counts only.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from extract_agy import PROMPT, batches

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:11434/api/generate"
BATCH = 6


def ask(model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["response"]


def main(model: str) -> None:
    out = HERE / "extract_local" / re.sub(r"[^\w.-]", "_", model)
    out.mkdir(parents=True, exist_ok=True)
    items = [it for b in batches() for it in b]
    small = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    facts: dict[str, list[str]] = {}
    t0 = time.time()
    for k, batch in enumerate(small):
        done = out / f"b{k:03d}.json"
        if done.exists():
            facts.update(json.loads(done.read_text(encoding="utf-8")))
            continue
        body = "\n".join(json.dumps({"id": mid, "date": dt, "text": t}, ensure_ascii=False) for mid, t, dt in batch)
        got = {}
        for attempt in range(2):
            try:
                got = json.loads(ask(model, PROMPT + body + "\n"))
                break
            except Exception as e:  # noqa: BLE001
                print(f"b{k:03d} attempt {attempt}: {str(e)[:120]}", flush=True)
        part = {t: [str(f).strip() for f in (got.get(mid) or []) if isinstance(f, str) and f.strip()]
                for mid, t, _ in batch}
        missing = sum(1 for mid, _, _ in batch if not isinstance(got.get(mid), list))
        done.write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
        facts.update(part)
        rate = (time.time() - t0) / (k + 1)
        print(f"b{k:03d}/{len(small)}: {sum(map(len, part.values()))} facts, {missing} ids missing; "
              f"~{rate * (len(small) - k - 1) / 60:.0f} min left", flush=True)
    (out / "facts_lme_ku.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
    n = [len(v) for v in facts.values()]
    print(f"{model}: {len(facts)}/{len(items)} messages, {sum(n)} facts, {sum(1 for x in n if x == 0)} with none")


if __name__ == "__main__":
    main(sys.argv[1])
