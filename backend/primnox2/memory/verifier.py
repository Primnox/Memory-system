"""The local model's second opinion on updates the rules were unsure of.

`cognition/conflict.py` retires a memory only on evidence it can name: a shared
slot, a reversal that shares a word, a similarity plus wording that says "changed".
Most updates it misses are implicit ("Subway all the way now" after "I take the
streetcar") or reworded past what a word list or an encoder sees. This asks a
small local model one typed question about the pairs that survive every guard
and that the rules did not decide:

    does NEW replace OLD, add detail to it, sit beside it as another value, concern
    another person or thing, or have nothing to do with it?

Nothing here is on the write path. A save stores what the rules decided and
queues the pairs; a background thread asks the model and, when it says "replaces"
with enough confidence, retires OLD a few seconds later, dated by NEW's own
`created_at`. With Ollama unreachable the queue is skipped and the rule-based
result stands. Every answer is kept in `memory_verifications` (primnox.db, so the
vault covers it) as training data for a smaller local decision model; it goes
nowhere else.

The guards in `conflict.may_replace` apply however confident the model is. The
queue lives in memory: a restart drops what was waiting, and `service.update()`
re-judges an edited memory by the rules alone.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from ..cognition import conflict
from ..storage import db

log = logging.getLogger(__name__)

RELATIONS = ("replaces", "adds_detail", "another_value", "other_person", "unrelated")
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"

# A shared GPU can sit on a request for a minute; a reply that late is still
# worth having, since nobody is waiting for it.
TIMEOUT_S = 180
ATTEMPTS = 3
# After the server refuses a connection (or the model is missing) every queued
# question would fail the same way, so none is asked until this has passed.
COOLDOWN_S = 60

_SYSTEM = """You check whether a person's newer statement about their own life makes an older one out of date.

OLD was said earlier. NEW was said later. First name, in a few words, the one thing each statement tells you about (its attribute: phone, employer, home city, commute, music app, diet...). Then choose exactly one relation:
- replaces: NEW gives a different current value for the same attribute OLD describes, so OLD is no longer true now. The change may be implied rather than stated (got a new one, left, moved, switched, finished, stopped).
- adds_detail: NEW says more about the same fact. Both are still true.
- another_value: NEW is a different value that can be true alongside OLD (a second pet, another hobby, one more language, an extra purchase).
- other_person: NEW is about a different person or thing than OLD.
- unrelated: NEW is about a different attribute or something else.

Answer with JSON only: {"old_about": "<few words>", "new_about": "<few words>", "relation": "<one of the five>", "confidence": <0 to 1>}

Examples:
OLD: I commute by bus, about forty minutes.
NEW: Cycling to the office these days, twenty minutes door to door.
{"old_about": "commute", "new_about": "commute", "relation": "replaces", "confidence": 0.9}

OLD: My laptop is a 2019 MacBook Air.
NEW: The new ThinkPad arrived and I've set it up.
{"old_about": "laptop", "new_about": "laptop", "relation": "replaces", "confidence": 0.85}

OLD: Studying for the bar exam, evenings only.
NEW: Passed the bar, so no more evening study.
{"old_about": "evening study", "new_about": "evening study", "relation": "replaces", "confidence": 0.9}

OLD: I live in Porto.
NEW: I live in Porto, in the Bonfim neighbourhood.
{"old_about": "home city", "new_about": "home city", "relation": "adds_detail", "confidence": 0.9}

OLD: I play the piano.
NEW: Started learning guitar last month.
{"old_about": "instrument played", "new_about": "instrument played", "relation": "another_value", "confidence": 0.85}

OLD: I have a dog called Rex.
NEW: Adopted a kitten named Pip.
{"old_about": "pets", "new_about": "pets", "relation": "another_value", "confidence": 0.9}

OLD: I work at a bank.
NEW: My sister works at a hospital.
{"old_about": "user's employer", "new_about": "sister's employer", "relation": "other_person", "confidence": 0.95}

OLD: I drink two coffees a day.
NEW: The flat has a small balcony.
{"old_about": "coffee", "new_about": "flat", "relation": "unrelated", "confidence": 0.95}

OLD: I take the tram to the office.
NEW: I go swimming on Thursday evenings.
{"old_about": "commute", "new_about": "exercise", "relation": "unrelated", "confidence": 0.95}"""


@dataclass
class Verdict:
    relation: str
    confidence: float
    ms: int


@dataclass
class Job:
    """One new memory and the standing ones worth asking about, most similar first.
    `build` stands in for `candidates` when choosing them is itself left to the
    background thread (`plan`), so a save pays nothing for it."""
    new_id: str
    text: str
    candidates: list[dict] = field(default_factory=list)   # {"id", "text", "sim"}
    build: Callable[[], "Job | None"] | None = None


# What this process has done, for the benchmark and for tests; never persisted.
stats = {"jobs": 0, "asked": 0, "answered": 0, "failed": 0, "skipped": 0, "retired": 0, "ms": 0}


def reset_stats() -> None:
    for key in stats:
        stats[key] = 0


def _tunable(key: str):
    from ..settings import tunables
    return tunables.get(key)


def enabled() -> bool:
    return bool(_tunable("memory.llm_verify"))


# ── Choosing what to ask ─────────────────────────────────────────────────────
def plan(new_id: str, text: str, topic: str | None, standing: list[dict], similarity,
         *, guard_actors: bool = False) -> Job | None:
    """The questions worth asking about a save the rules left alone, or None.

    `similarity` maps a standing memory's id to its encoder similarity with
    `text`. A pair is asked about when it clears the floor and every guard in
    `conflict.may_replace`; the most similar few go first."""
    floor = _tunable("memory.llm_verify_similarity")
    scored = []
    for c in standing:
        sim = similarity(c["id"])
        if sim >= floor and conflict.may_replace(text, c, new_topic=topic, guard_actors=guard_actors):
            scored.append((sim, c))
    if not scored:
        return None
    scored.sort(key=lambda p: -p[0])
    top = scored[:int(_tunable("memory.llm_verify_candidates"))]
    return Job(new_id, text, [{"id": c["id"], "text": c["text"], "sim": round(sim, 4)} for sim, c in top])


# ── Asking ───────────────────────────────────────────────────────────────────
_state = {"down_until": 0.0, "warned": False}


def model_name() -> str:
    return os.getenv("PRIMNOX2_MEMORY_VERIFY_MODEL") or DEFAULT_MODEL


def _post(old: str, new: str) -> tuple[str, list]:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    body = json.dumps({
        "model": model_name(), "stream": False, "format": "json", "think": False,
        "logprobs": True, "top_logprobs": 5, "options": {"temperature": 0},
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": f"OLD: {old}\nNEW: {new}"}],
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        reply = json.loads(r.read())
    return (reply.get("message") or {}).get("content") or "", reply.get("logprobs") or []


_VALUE = re.compile(r'"relation"\s*:\s*"')


def _distribution(content: str, logprobs: list) -> dict[str, float] | None:
    """How likely the model found each relation, read from the tokens it
    weighed where it wrote the relation's name: a number it can be held to,
    unlike the confidence it states, which comes out as 0.85 or 0.9 whatever it
    is shown. None when the server returned no token probabilities (an older
    Ollama) or the reply is not shaped as expected."""
    found = _VALUE.search(content)
    if not found or not logprobs:
        return None
    at, pos = found.end(), 0
    for entry in logprobs:
        token = entry.get("token") or ""
        if pos <= at < pos + len(token):
            weight: dict[str, float] = {}
            for alt in entry.get("top_logprobs") or []:
                word = (alt.get("token") or "")[at - pos:].strip(' "').lower()
                named = [r for r in RELATIONS if word and r.startswith(word)]
                if len(named) == 1:
                    weight[named[0]] = weight.get(named[0], 0.0) + math.exp(alt.get("logprob", -math.inf))
            total = sum(weight.values())
            return {r: w / total for r, w in weight.items()} if total else None
        pos += len(token)
    return None


def _parse(content: str, logprobs: list) -> tuple[str, float] | None:
    start, end = content.find("{"), content.rfind("}")
    try:
        out = json.loads(content[start:end + 1])
        relation = str(out["relation"]).strip().lower()
        if relation not in RELATIONS:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    seen = (_distribution(content, logprobs) or {}).get(relation)
    if seen is not None:
        return relation, seen
    try:
        stated = float(out.get("confidence"))
    except (TypeError, ValueError):
        return relation, 0.0
    return relation, max(0.0, min(1.0, stated / 100 if 1 < stated <= 100 else stated))


def _unavailable(why: str) -> None:
    _state["down_until"] = time.monotonic() + COOLDOWN_S
    if not _state["warned"]:
        _state["warned"] = True
        log.warning("memory verifier: local model unavailable (%s); updates are judged by the rules alone", why)


def ask(new: str, old: str) -> Verdict | None:
    """The model's reading of how `new` relates to `old`, or None when there is no
    usable answer (the server is down, the model is missing, or it kept replying
    with something that is not one of the five relations)."""
    if time.monotonic() < _state["down_until"]:
        stats["skipped"] += 1
        return None
    stats["asked"] += 1
    began = time.perf_counter()
    for attempt in range(ATTEMPTS):
        try:
            parsed = _parse(*_post(old, new))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                _unavailable(f"HTTP {e.code}")
                break
            parsed = None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                parsed = None
            else:
                _unavailable(str(e.reason))
                break
        except (TimeoutError, ConnectionError, OSError, ValueError):
            parsed = None
        if parsed:
            _state["warned"] = False
            ms = int((time.perf_counter() - began) * 1000)
            stats["answered"] += 1
            stats["ms"] += ms
            return Verdict(parsed[0], parsed[1], ms)
        if attempt + 1 < ATTEMPTS:
            time.sleep(2 * (attempt + 1))
    stats["failed"] += 1
    return None


# ── Recording ────────────────────────────────────────────────────────────────
def _record(job: Job, cand: dict, verdict: Verdict, applied: bool) -> None:
    try:
        with db.tx() as c:
            c.execute(
                "INSERT INTO memory_verifications (ts,model,new_id,old_id,new_text,old_text,"
                "similarity,relation,confidence,applied,ms) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (int(time.time() * 1000), model_name(), job.new_id, cand["id"], job.text,
                 cand["text"], cand["sim"], verdict.relation, verdict.confidence,
                 int(applied), verdict.ms))
    except Exception:
        log.warning("memory verifier: could not record a decision", exc_info=True)


def decisions() -> list[dict]:
    """Everything the model has been asked, oldest first."""
    return [dict(r) for r in db.connect().execute("SELECT * FROM memory_verifications ORDER BY id")]


# ── Running in the background ────────────────────────────────────────────────
_jobs: queue.Queue[Job] = queue.Queue()
_lock = threading.Lock()
_thread: threading.Thread | None = None
_running = False


def submit(jobs: list[Job]) -> None:
    jobs = [j for j in jobs if j and (j.candidates or j.build)]
    if not jobs:
        return
    for job in jobs:
        _jobs.put(job)
    _ensure_worker()


def _ensure_worker() -> None:
    global _thread, _running
    with _lock:
        if _running:
            return
        _running = True
        _thread = threading.Thread(target=_work, name="memory-verifier", daemon=True)
        _thread.start()


def _work() -> None:
    # A thread per burst rather than one for the life of the process: it ends with
    # its database connection, which `db.configure()` could otherwise leave
    # pointing at a file nothing else uses.
    global _running
    try:
        while True:
            with _lock:
                try:
                    job = _jobs.get_nowait()
                except queue.Empty:
                    _running = False
                    return
            try:
                _process(job)
            except Exception:
                log.warning("memory verifier: a job failed", exc_info=True)
            finally:
                _jobs.task_done()
    finally:
        db.close_connection()


def _process(job: Job) -> None:
    from . import service
    if not enabled():
        return
    if job.build:
        job = job.build()
        if job is None:
            return
    stats["jobs"] += 1
    bar = _tunable("memory.llm_verify_confidence")
    for cand in job.candidates:
        verdict = ask(job.text, cand["text"])
        if verdict is None:
            return
        applied = (verdict.relation == "replaces" and verdict.confidence >= bar
                   and service.apply_verified(job.new_id, job.text, cand["id"], cand["text"]))
        _record(job, cand, verdict, applied)
        if applied:
            stats["retired"] += 1
            return


def drain() -> None:
    """Block until every queued question has been answered or skipped."""
    _jobs.join()
    thread = _thread
    if thread is not None:
        thread.join()
