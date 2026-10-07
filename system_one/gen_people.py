"""Write generated people with models from other families, and check their labels
with a model from yet another one.

Every person so far was written by Claude agents, and on LongMemEval (GPT-4o's
chats) the head's gain shrank: some of what it learned is how Claude describes a
life changing. This writes new people through OmniRoute with any OpenRouter
model, in the blind-set schema, then has a CHECKER of a different family answer,
for every labelled pair, "is the earlier statement no longer true now because of
the later one?". Disputed labels are dropped from training data (`skip`) and
adjudicated by a third model for test data; disputed test questions are dropped.

    python gen_people.py --kind train --writer openrouter/deepseek/deepseek-v4-flash \\
        --checker openrouter/z-ai/glm-5.3-flash -n 8 --out data/train_writer_deepseek.json
    python gen_people.py --kind test --writer openrouter/google/gemini-3.7-flash \\
        --checker openrouter/moonshotai/kimi-k3 --judge openrouter/mistralai/mistral-medium-3-5 \\
        -n 6 --out ../scripts/blind_memory/v3/parts/gemini.json

Writers never see the code, the tests, or any other person.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

GATEWAY = "http://127.0.0.1:20128/v1/chat/completions"
OLLAMA = "http://127.0.0.1:11434/api/chat"
RELATIONS = ("adds", "detail", "other", "event")
TYPES = ("current", "past", "multi", "absent")
TEST_MAX_CHARS = 220
FIRST = ["Amara", "Bao", "Chiara", "Dawit", "Elif", "Farid", "Greta", "Hamza", "Ines", "Jun", "Kalani",
         "Lucia", "Mateus", "Nadia", "Oskar", "Priya", "Quentin", "Rania", "Sanjay", "Tove", "Ugo", "Valeria",
         "Wei", "Ximena", "Yusuf", "Zofia", "Adaeze", "Bjarne", "Camila", "Deniz", "Emeka", "Fatima",
         "Giorgos", "Hyejin", "Imani", "Joaquin", "Kirra", "Laszlo", "Mei", "Nikolai", "Olga", "Pita",
         "Rosalind", "Siddharth", "Thandiwe", "Uri", "Vikram", "Wanjiku", "Yara", "Zeynep"]
CITIES = ["Accra", "Adelaide", "Almaty", "Bogotá", "Bordeaux", "Busan", "Cape Town", "Chennai", "Cluj-Napoca",
          "Córdoba", "Dhaka", "Dublin", "Edinburgh", "Gdańsk", "Guadalajara", "Hanoi", "Izmir", "Kampala",
          "Kraków", "Kuala Lumpur", "Lagos", "Leipzig", "Lima", "Lyon", "Manila", "Medellín", "Montréal",
          "Nagoya", "Nairobi", "Oaxaca", "Ottawa", "Porto Alegre", "Pune", "Reykjavík", "Rotterdam",
          "Santiago", "Seville", "Tallinn", "Tbilisi", "Tunis", "Valencia", "Vilnius", "Wellington", "Winnipeg"]
JOBS = ["nurse", "bus driver", "data analyst", "pastry chef", "high-school teacher", "electrician",
        "graduate student", "UX designer", "pharmacist", "warehouse supervisor", "freelance translator",
        "civil engineer", "physiotherapist", "barista", "junior lawyer", "retired accountant", "game developer",
        "veterinary assistant", "real-estate agent", "lab technician", "farmer", "flight attendant",
        "social worker", "carpenter", "marketing manager", "museum guide", "dental hygienist"]
STYLES = {
    "terse": "Mostly short, casual messages (one or two sentences), texting style, lowercase allowed, "
             "abbreviations and the odd typo.",
    "chatty": "Mostly long, rambling messages of three to six sentences that mix a question or request to "
              "the assistant with personal details; changes are often buried mid-message or mentioned in passing.",
    "mixed": "A natural mix: some one-liners, some long messages that wander between topics, some "
             "messages that are mostly a question with a personal detail slipped in.",
    "formal": "Fairly formal, complete sentences, a little reserved; changes are stated plainly but often "
              "without words like 'now' or 'moved' (e.g. 'The keys to the Kadıköy flat were handed over on Friday').",
}

SPEC = """You are writing realistic test data for an AI assistant's long-term memory: the messages ONE person
sends an assistant over about {months} months, in the first person, as themselves.

The person: {first}, a {job} living in {city}. Invent everything else (family, home, commute, devices,
pets, diet, health, hobbies, routines, money, travel). Do not use real celebrities. Do not ask the
assistant to remember anything; just talk.

Style of this person's messages: {style}

Write {lo} to {hi} messages, dated, oldest first, between {start} and {end}. Several may share a date.
{length_rule}

What the messages must contain, spread naturally:
1. 9 to 13 UPDATES: a later message makes an earlier fact no longer true (moved, changed job or role,
   new phone/laptop/car, quit or started a habit, changed diet, a relationship began or ended, a pet died,
   switched gym or app or bank, a medication changed). At least half must be IMPLICIT: the change is
   only implied ("finally got the keys to the Kadıköy place", "my first week on nights", "the new
   ThinkPad is so much lighter"). Include at least two chains (A, later B replaces A, later C replaces B).
2. At least 2 CHAIN REACTIONS: one message ends several earlier facts at once (a breakup ends "we live
   together" and "weekend trips with him"; a new job ends the old employer, the old commute and the
   old lunch routine). Such a message replaces ALL of them.
3. At least 8 TRAPS: later messages that look related to an earlier one but do NOT make it untrue:
   - adds: a second value of the same kind (another pet, another language, a second job on weekends)
   - detail: more detail on the same fact, or the same fact restated in other words
   - other: the same kind of fact about SOMEONE ELSE ("my sister just moved to Leeds")
   - event: a one-off event, a trip, a temporary situation, or a plan not yet carried out
     ("in Rome for work this week", "thinking about switching to Android")
   Use every one of the four kinds at least twice.

Labels (be exact; they are the answer key):
- "replaces": ids of EARLIER messages whose fact is no longer true BECAUSE of this message. Empty if none.
  A message can replace several. Never label a trap as replaces.
- "relations": for each trap, {{"to": "<earlier id>", "relation": "adds|detail|other|event"}}.
{questions_rule}
Return ONLY one JSON object, no prose, no code fences:
{{"scenario": "{slug}", {today_field}"statements": [{{"id": "s1", "date": "YYYY-MM-DD", "text": "...",
"replaces": [], "relations": []}}, ...]{questions_field}}}
Ids are s1, s2, ... in order."""

TEST_LENGTH = (f"Each message is what would be saved to memory: one or two sentences in the person's own "
               f"voice, at most {TEST_MAX_CHARS} characters, about the person (no requests to the assistant).")
TRAIN_LENGTH = "Messages may be long (up to about 600 characters) when the style calls for it."
QUESTIONS_RULE = """
Then write 14 to 18 QUESTIONS the person might ask later, on "today" ({today}, after the last message):
- 6 to 8 "current": about what is true now ("Where do I live these days?"); answer_ids = the message(s)
  holding the current value, including the latest detail messages for it.
- 4 to 5 "past": about a moment in the past, with "as_of" = a date (YYYY-MM-DD) inside that period
  ("Back in March, which phone did I have?"); answer_ids = what was true on that date.
- 2 "multi": asking for all values ("What pets do I have?"); answer_ids = every message holding a current value.
- 2 "absent": about something never mentioned; answer_ids = [].
Each question: {{"id": "q1", "text": "...", "type": "current|past|multi|absent", "answer_ids": [...], "as_of": null or date}}.
"""

REPAIR = """Your JSON below has these problems:
{problems}

Return the corrected JSON object only: ids s1, s2, ... in order with no gaps or repeats, every
date YYYY-MM-DD and never earlier than the message before, labels pointing only at earlier ids.
Keep the content; fix only what is listed.

{reply}"""

CHECK = """Below are dated messages one person sent an assistant, oldest first, then numbered PAIRS
(earlier id, later id). For each pair decide: is the EARLIER message's fact no longer true now
BECAUSE of the later message (it was changed, replaced or ended)? Answer "yes" or "no".
A second value of the same kind, more detail on the same fact, a fact about someone else, a trip,
a temporary situation or an unexecuted plan are "no".

MESSAGES
{messages}

PAIRS
{pairs}

Return ONLY JSON: {{"answers": ["yes"|"no", ...]}} in pair order."""

ANSWER = """Below are dated messages one person sent an assistant, oldest first. Today is {today}.
QUESTION ({kind}): {question}{as_of}
Which message ids hold the answer{when}? For "what is true now" questions give the message(s) with the
current value and any later details about it; for "all values" questions give every message holding a
current value; if nothing answers it, give an empty list.

MESSAGES
{messages}

Return ONLY JSON: {{"answer_ids": ["s.."]}}"""


# ── talking to models ────────────────────────────────────────────────────────
_spend = {"calls": 0, "in": 0, "out": 0}
_spend_lock = threading.Lock()


def _ollama_target(model: str) -> tuple[str, str]:
    """`ollama:<model>` on the default server, or `ollama@<port>:<model>` on another
    (one Ollama server per GPU)."""
    if model.startswith("ollama@"):
        port, name = model[len("ollama@"):].split(":", 1)
        return name, f"http://127.0.0.1:{port}/api/chat"
    return model[len("ollama:"):], OLLAMA


def _ask_ollama(model: str, prompt: str, max_tokens: int, temperature: float, url: str = OLLAMA) -> str:
    # Ollama's native route: its OpenAI-compatible /v1 ignores num_ctx and loads every
    # model at 4,096 tokens, shorter than one written person.
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        reply = json.loads(r.read())
    with _spend_lock:
        _spend["calls"] += 1
        _spend["in"] += reply.get("prompt_eval_count") or 0
        _spend["out"] += reply.get("eval_count") or 0
    return (reply.get("message") or {}).get("content") or ""


def _opencode_exe() -> str:
    # the native binary: the npm .cmd shim garbles long multi-line arguments
    import os
    import shutil
    root = Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "opencode-ai"
    for exe in (root / "bin" / "opencode.exe", root / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"):
        if exe.is_file():
            return str(exe)
    return shutil.which("opencode") or "opencode"


def _ask_opencode(model: str, prompt: str) -> str:
    """OpenCode Zen's free models answer only from inside OpenCode, so this runs
    `opencode run` with the read-only plan agent in an empty folder: it can read
    nothing of ours and change nothing."""
    import subprocess
    import tempfile
    sandbox = Path(tempfile.gettempdir()) / "oc-sandbox"
    sandbox.mkdir(exist_ok=True)
    # OpenCode's free tier refuses any request made with a project config (even one
    # that only denies tools), so the folder holds none; the read-only plan agent
    # can neither edit files nor run commands
    (sandbox / "opencode.json").unlink(missing_ok=True)
    # a command-line argument loses everything after its first line, so the prompt
    # goes in as an attached file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=sandbox, delete=False, encoding="utf-8") as f:
        f.write(prompt)
    try:
        done = subprocess.run([_opencode_exe(), "run", "--pure", "--agent", "plan", "--dir", str(sandbox),
                               "-m", model, "--format", "json", "-f", f.name, "--",
                               ("This is not a coding or planning task. Produce the output the attached file asks for, "
                                "now, and reply with only that output: no plan, no table, no explanation.")],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=900)
    finally:
        Path(f.name).unlink(missing_ok=True)
    texts, tokens = [], {}
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        part = event.get("part") or {}
        if event.get("type") == "text" and part.get("text"):
            texts.append(part["text"])
        if event.get("type") == "step_finish":
            tokens = part.get("tokens") or {}
    with _spend_lock:
        _spend["calls"] += 1
        _spend["in"] += tokens.get("input") or 0
        _spend["out"] += tokens.get("output") or 0
    return "".join(texts)


def _ask_vllm(spec: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """`vllm:<port>:<model>`, a vLLM OpenAI-compatible server on this machine."""
    port, model = spec.split(":", 1)
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        reply = json.loads(r.read())
    usage = reply.get("usage") or {}
    with _spend_lock:
        _spend["calls"] += 1
        _spend["in"] += usage.get("prompt_tokens") or 0
        _spend["out"] += usage.get("completion_tokens") or 0
    return (reply.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def ask(model: str, prompt: str, max_tokens: int = 8000, attempts: int = 4) -> str:
    temperature = 0.9 if "write" in prompt[:40] else 0
    if model.startswith("vllm:"):
        for attempt in range(attempts):
            try:
                content = _ask_vllm(model[len("vllm:"):], prompt, max_tokens, temperature)
                if content.strip():
                    return content
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, OSError) as e:
                print(f"    {model}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{model}: no usable reply after {attempts} attempts")
    if model.startswith("opencode:"):
        for attempt in range(attempts):
            try:
                content = _ask_opencode(model[len("opencode:"):], prompt)
                if content.strip():
                    return content
            except (OSError, ValueError) as e:
                print(f"    {model}: {e}", flush=True)
            except Exception as e:                       # subprocess.TimeoutExpired
                print(f"    {model}: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{model}: no usable reply after {attempts} attempts")
    if model.startswith(("ollama:", "ollama@")):
        for attempt in range(attempts):
            try:
                name, url = _ollama_target(model)
                content = _ask_ollama(name, prompt, max_tokens, temperature, url)
                if content.strip():
                    return content
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, OSError):
                pass
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{model}: no usable reply after {attempts} attempts")
    request = {"model": model, "stream": False, "temperature": temperature, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}],
               # thinking models (DeepSeek V4, Qwen 3.8) otherwise spend the whole
               # budget reasoning and return no content at all
               "reasoning": {"enabled": False}}
    for attempt in range(attempts):
        body = json.dumps(request).encode()
        req = urllib.request.Request(GATEWAY, data=body, headers={"Content-Type": "application/json",
                                                                   "x-omniroute-no-cache": "true"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                reply = json.loads(r.read())
            content = (reply.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            usage = reply.get("usage") or {}
            with _spend_lock:
                _spend["calls"] += 1
                _spend["in"] += usage.get("prompt_tokens") or 0
                _spend["out"] += usage.get("completion_tokens") or 0
            if content.strip():
                return content
        except urllib.error.HTTPError as e:
            if e.code == 400 and "reasoning" in request:
                request.pop("reasoning")                 # a model that cannot switch thinking off
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, OSError):
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{model}: no usable reply after {attempts} attempts")


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


# ── validation ───────────────────────────────────────────────────────────────
def problems(person: dict, kind: str) -> list[str]:
    """Everything structurally wrong with a written person; empty when usable."""
    out = []
    sts = person.get("statements")
    if not isinstance(sts, list) or len(sts) < 18:
        return ["fewer than 18 statements"]
    seen, last = set(), ""
    updates, chain_reactions, traps = 0, 0, {r: 0 for r in RELATIONS}
    for i, s in enumerate(sts, 1):
        sid = s.get("id")
        if sid != f"s{i}":
            out.append(f"id {sid!r} out of order (expected s{i})")
        d = str(s.get("date") or "")
        try:
            date.fromisoformat(d)
        except ValueError:
            out.append(f"{sid}: bad date {d!r}")
        if d < last:
            out.append(f"{sid}: date goes backwards")
        last = max(last, d)
        text = str(s.get("text") or "").strip()
        if len(text) < 8:
            out.append(f"{sid}: empty text")
        if kind == "test" and len(text) > TEST_MAX_CHARS:
            out.append(f"{sid}: {len(text)} characters, over {TEST_MAX_CHARS}")
        reps = s.get("replaces") or []
        for r in reps:
            if r not in seen:
                out.append(f"{sid}: replaces {r!r}, which is not an earlier message")
        rels = s.get("relations") or []
        for rel in rels:
            if rel.get("to") not in seen:
                out.append(f"{sid}: relation to {rel.get('to')!r}, which is not an earlier message")
            if rel.get("relation") not in RELATIONS:
                out.append(f"{sid}: unknown relation {rel.get('relation')!r}")
            elif rel.get("to") in reps:
                out.append(f"{sid}: {rel.get('to')} is both replaced and a trap")
            else:
                traps[rel["relation"]] += 1
        updates += len(reps)
        chain_reactions += len(reps) >= 2
        seen.add(sid)
    if updates < 7:
        out.append(f"only {updates} updates")
    if chain_reactions < 1:
        out.append("no chain reaction")
    if sum(traps.values()) < 5 or sum(1 for v in traps.values() if v) < 3:
        out.append(f"too few traps {traps}")
    if kind == "test":
        out += question_problems(person, seen)
    return out


def question_problems(person: dict, ids: set[str]) -> list[str]:
    out = []
    qs = person.get("questions")
    if not isinstance(qs, list) or len(qs) < 10:
        return ["fewer than 10 questions"]
    try:
        date.fromisoformat(str(person.get("today")))
    except ValueError:
        out.append("no valid 'today'")
    for q in qs:
        if q.get("type") not in TYPES:
            out.append(f"{q.get('id')}: unknown type {q.get('type')!r}")
        missing = [a for a in q.get("answer_ids") or [] if a not in ids]
        if missing:
            out.append(f"{q.get('id')}: answer ids {missing} do not exist")
        if q.get("type") == "past" and not q.get("as_of"):
            out.append(f"{q.get('id')}: a past question without as_of")
        if q.get("type") == "absent" and q.get("answer_ids"):
            out.append(f"{q.get('id')}: an absent question with answers")
        if q.get("type") != "absent" and not q.get("answer_ids"):
            out.append(f"{q.get('id')}: no answers")
    return out


def labelled_pairs(person: dict) -> list[tuple[str, str, bool]]:
    """(earlier id, later id, replaces?) for every label the writer gave."""
    pairs = []
    for s in person["statements"]:
        pairs += [(r, s["id"], True) for r in s.get("replaces") or []]
        pairs += [(rel["to"], s["id"], False) for rel in s.get("relations") or []]
    return pairs


def _messages(person: dict) -> str:
    return "\n".join(f'{s["id"]} ({s["date"]}): {s["text"]}' for s in person["statements"])


# ── one person ───────────────────────────────────────────────────────────────
def write_one(args, rng: random.Random, n: int) -> dict | None:
    first, city, job = rng.choice(FIRST), rng.choice(CITIES), rng.choice(JOBS)
    style = rng.choice(list(STYLES)) if args.style == "any" else args.style
    start = date(2025, rng.randint(1, 6), rng.randint(1, 28))
    months = rng.randint(9, 15)
    end = date(start.year + (start.month + months - 1) // 12, (start.month + months - 1) % 12 + 1, 28)
    today = date(end.year + (end.month // 12), end.month % 12 + 1, 10)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{first}-{city}-{job}".lower()).strip("-")
    test = args.kind == "test"
    prompt = SPEC.format(
        months=months, first=first, job=job, city=city, style=STYLES[style], lo=22, hi=30,
        start=start.isoformat(), end=end.isoformat(), length_rule=TEST_LENGTH if test else TRAIN_LENGTH,
        questions_rule=QUESTIONS_RULE.format(today=today.isoformat()) if test else "",
        slug=slug, today_field=f'"today": "{today.isoformat()}", ' if test else "",
        questions_field=', "questions": [...]' if test else "")
    reply = None
    for attempt in range(3):
        try:
            # after a flawed reply, ask for that reply repaired rather than a new person:
            # small models slip on one id or date in a long JSON far more often than they
            # get the whole person wrong
            reply = ask(args.writer, prompt if reply is None else REPAIR.format(
                problems="\n".join(f"- {x}" for x in issues[:12]), reply=reply))
            person = parse_json(reply)
        except (RuntimeError, ValueError) as e:
            print(f"  #{n} {slug}: {e}", flush=True)
            reply = None
            continue
        issues = problems(person, args.kind)
        if not issues:
            person["scenario"], person["writer"], person["style"] = slug, args.writer, style
            return person
        print(f"  #{n} {slug} attempt {attempt + 1}: {issues[:4]}", flush=True)
    return None


def check(args, person: dict) -> dict:
    """Ask the checker about every labelled pair; settle disputes (test) or set them
    aside (train). Returns audit counts."""
    pairs = labelled_pairs(person)
    listing = "\n".join(f"{i}. ({a}, {b})" for i, (a, b, _) in enumerate(pairs, 1))
    answers = parse_json(ask(args.checker, CHECK.format(messages=_messages(person), pairs=listing),
                             max_tokens=4000))["answers"]
    if len(answers) != len(pairs):
        raise ValueError(f"checker gave {len(answers)} answers for {len(pairs)} pairs")
    disputed = [(a, b, label) for (a, b, label), ans in zip(pairs, answers)
                if (str(ans).strip().lower() == "yes") != label]
    audit = {"pairs": len(pairs), "disputed": len(disputed), "overturned": 0}
    if not disputed:
        return audit
    if args.kind == "train":
        person["skip"] = [[a, b] for a, b, _ in disputed]
        return audit
    listing = "\n".join(f"{i}. ({a}, {b})" for i, (a, b, _) in enumerate(disputed, 1))
    verdicts = parse_json(ask(args.judge, CHECK.format(messages=_messages(person), pairs=listing),
                              max_tokens=2000))["answers"]
    by_id = {s["id"]: s for s in person["statements"]}
    for (a, b, label), verdict in zip(disputed, verdicts):
        if (str(verdict).strip().lower() == "yes") == label:
            continue                                   # the judge sides with the writer
        audit["overturned"] += 1
        s = by_id[b]
        if label:                                      # not a replacement after all: a trap of unknown kind
            s["replaces"] = [r for r in s["replaces"] if r != a]
        else:                                          # a trap that is really a replacement
            s["relations"] = [r for r in s["relations"] if r["to"] != a]
            s["replaces"] = sorted(set(s.get("replaces") or []) | {a}, key=lambda x: int(x[1:]))
    return audit


def check_questions(args, person: dict) -> dict:
    kept, dropped = [], 0
    for q in person["questions"]:
        if q["type"] == "absent":
            kept.append(q)
            continue
        prompt = ANSWER.format(today=person["today"], kind=q["type"], question=q["text"],
                               as_of=f" (as of {q['as_of']})" if q.get("as_of") else "",
                               when=f" as of {q['as_of']}" if q.get("as_of") else " today",
                               messages=_messages(person))
        try:
            theirs = set(parse_json(ask(args.checker, prompt, max_tokens=1000))["answer_ids"])
        except (RuntimeError, ValueError, KeyError):
            dropped += 1
            continue
        if theirs & set(q["answer_ids"]) if q["type"] == "multi" else theirs == set(q["answer_ids"]):
            kept.append(q)
        elif q["type"] == "current" and theirs & set(q["answer_ids"]):
            # agreed on the value, not on which details belong with it: keep the overlap
            q["answer_ids"] = sorted(theirs & set(q["answer_ids"]), key=lambda x: int(x[1:]))
            kept.append(q)
        else:
            dropped += 1
    person["questions"] = kept
    return {"questions": len(kept) + dropped, "dropped": dropped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("train", "test"), required=True)
    ap.add_argument("--writer", required=True)
    ap.add_argument("--checker", required=True)
    ap.add_argument("--judge", help="third model for disputed test labels")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--style", default="any", choices=("any", *STYLES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-check", action="store_true",
                    help="write people without checking their labels (check them later with --recheck)")
    ap.add_argument("--recheck", metavar="FILE",
                    help="check the labels of people already written to FILE instead of writing new ones")
    args = ap.parse_args()
    if args.kind == "test" and not args.judge:
        ap.error("--judge is needed for test data")
    for m in (args.writer, args.checker, args.judge):
        if m and not m.startswith(("openrouter/", "ollama:", "ollama@", "opencode:", "vllm:")):
            ap.error(f"{m}: give an OmniRoute id (openrouter/...), ollama:<model>, opencode:<provider/model> "
                     "or vllm:<port>:<model>")
    written = json.loads(Path(args.recheck).read_text(encoding="utf-8")) if args.recheck else None
    if written is not None:
        args.n = len(written)

    def one(n: int):
        if written is not None:
            person = written[n]
        else:
            rng = random.Random(args.seed * 1000 + n)
            person = write_one(args, rng, n)
        if person is None:
            return None, None
        if args.no_check:
            print(f"  #{n} {person['scenario']} ({person.get('style', '?')}): {len(person['statements'])} messages, "
                  f"not checked yet", flush=True)
            return person, {}
        try:
            audit = check(args, person)
            if args.kind == "test":
                audit |= check_questions(args, person)
        except (RuntimeError, ValueError, KeyError) as e:
            print(f"  #{n} {person['scenario']}: check failed ({e}); person dropped", flush=True)
            return None, None
        print(f"  #{n} {person['scenario']} ({person.get('style', '?')}): {len(person['statements'])} messages, "
              f"audit {audit}", flush=True)
        return person, audit

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, range(args.n)))
    people = [p for p, _ in results if p]
    audits = [a for _, a in results if a]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(people, ensure_ascii=False, indent=1), encoding="utf-8")
    total = {k: sum(a.get(k, 0) for a in audits) for k in ("pairs", "disputed", "overturned", "questions", "dropped")}
    print(f"{len(people)}/{args.n} people -> {args.out}; labels {total}; "
          f"tokens {_spend['in']} in / {_spend['out']} out over {_spend['calls']} calls")
    return 0 if people else 1


if __name__ == "__main__":
    sys.exit(main())
