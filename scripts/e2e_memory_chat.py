"""Memory END TO END: the real chat stack, the real local model, a judge model.

`bench_memory_blind.py` measures retrieval with a model that only writes the
search query. This measures what a user would see. Per scenario, in a scratch
data dir and a fresh process:

  1. Every dated statement is sent as a user message in its OWN NEW
     conversation, through the real scheduler -> context -> gateway -> tool
     loop, with the memory clock and the prompt's "Today is ..." patched to the
     statement's date. The model decides whether to call `remember`; we record
     whether it did and what it saved.
  2. Every question is asked in a new conversation with "today" set to the
     scenario's `today`. We record the final answer and any `recall_memory`
     call (query, as_of). The memories table is restored after each question so
     one turn's side effects cannot leak into the next.
  3. A judge (local `gemma4:26b-a4b-it-q4_K_M` through Ollama, temperature 0, a
     different model from the one under test) grades each answer against the
     labelled statements: correct | stale | wrong | hallucinated | abstained.
     It runs after generation, so the two models never compete for the GPU.

The model: the app's `huihui_ai/qwen3.5-abliterated:9b` through the gateway's
native Ollama route (`/api/chat` with `num_ctx`, so the prompt is not cut at
4,096 tokens). The gateway leaves `think` unset, which on this model measured a
32,736-token thinking run with an empty reply on a trivial prompt, so the
harness sends `think: false` (the app's thinking-off setting) by wrapping
`urllib.request.Request`. Nothing in `memory/` or `cognition/` is touched.

Usage:
    python scripts/e2e_memory_chat.py --split dev --arm v3 \\
        --backend ../memory-v3/backend
    python scripts/e2e_memory_chat.py --split test --data .../blind_memory/v2 --arm verify \\
        --backend ../mem-verify/backend --env PRIMNOX2_MEMORY_LLM_VERIFY=1
    python scripts/e2e_memory_chat.py --split test --data .../blind_memory/v2 \\
        --stage report --arms v3 verify

Stages (default: run,judge,report): `run` spawns one worker process per
scenario and writes raw JSON; `judge` grades it; `report` prints aggregates
(and a paired comparison when two arms are given). The TEST split prints
aggregates only.
"""
from __future__ import annotations

import argparse
import difflib
import faulthandler
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "scripts" / "blind_memory"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "claude" / "e2e"
MODEL = "huihui_ai/qwen3.5-abliterated:9b"
OLLAMA = "http://127.0.0.1:11434"
JUDGE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
TURN_TIMEOUT_S = 1500
TURN_ATTEMPTS = 3


# ── small helpers ────────────────────────────────────────────────────────────
def wilson(k: int, n: int) -> tuple[float, float, float] | None:
    if n == 0:
        return None
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * p, 1), round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)


def fmt(k: int, n: int) -> str:
    w = wilson(k, n)
    return "n/a" if w is None else f"{w[0]:5.1f}% [{w[1]:5.1f}-{w[2]:5.1f}] ({k}/{n})"


def ms(date: str, end_of_day: bool = False) -> int:
    d = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000) + (86_400_000 - 1 if end_of_day else 12 * 3_600_000)


def load_split(data_dir: str, split: str) -> list[dict]:
    return json.loads((Path(data_dir) / f"{split}.json").read_text(encoding="utf-8"))


FIRST_PERSON = re.compile(r"\b(i|i'm|i’m|i've|i’ve|i'd|i'll|my|me|mine|myself|we|we're|we've|our|ours|us)\b", re.I)


def voice(text: str, name: str) -> str:
    """Who the saved sentence speaks as. `subjectless` is the 'Prefers X.' form
    the `remember` tool's own description teaches."""
    if FIRST_PERSON.search(text):
        return "first"
    if re.match(rf"\s*(the user|user|they|he|she|{re.escape(name)})\b", text, re.I):
        return "third"
    return "subjectless"


def similarity(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


# ═════════════════════════════════════════════════════════════════════════════
# WORKER: one scenario, one process, one scratch home
# ═════════════════════════════════════════════════════════════════════════════
def prepare_env(extra_env: list[str], cache_dir: Path) -> Path:
    home = Path(tempfile.mkdtemp(prefix="e2e-home-"))
    os.environ.update({
        "PRIMNOX2_HOME": str(home),
        "PRIMNOX2_AUTO_APPROVE": "all",
        "PRIMNOX2_MEMORY_PROBE": "0",
        # The background verifier (an arm option) reads OLLAMA_HOST itself, so it
        # stays real. A turn still cannot leave the model named below: the
        # head provider comes from PRIMNOX_* and failover is off. OmniRoute is
        # pointed at a closed port so nothing starts or reaches it.
        "OLLAMA_HOST": OLLAMA,
        "OMNIROUTE_HOST": "http://0.0.0.0:9",
        "PRIMNOX_BASE_URL": f"{OLLAMA}/v1",
        "PRIMNOX_MODEL": MODEL,
        "PRIMNOX_API_TYPE": "openai",
        "PRIMNOX_PROVIDER_KIND": "ollama",
        "PRIMNOX_API_KEY": "",
        "PRIMNOX2_MODELS_FAILOVER_ATTEMPTS": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    os.environ.pop("PRIMNOX_PROVIDER", None)
    for pair in extra_env:
        key, _, value = pair.partition("=")
        os.environ[key] = value
    # The sandbox library probe costs a full sandbox launch and is cached under
    # the home dir; reuse one result across scenarios instead of paying it each.
    cached = cache_dir / ".capabilities.json"
    if cached.is_file():
        (home / "sandbox").mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, home / "sandbox" / ".capabilities.json")
    return home


def think_off() -> None:
    """The app's thinking-off setting, sent where the gateway leaves it unset."""
    original = urllib.request.Request

    class Request(original):
        def __init__(self, url, data=None, *a, **k):
            if isinstance(url, str) and url.endswith("/api/chat") and data:
                try:
                    body = json.loads(data)
                    body["think"] = False
                    data = json.dumps(body).encode()
                except ValueError:
                    pass
            super().__init__(url, data, *a, **k)

    urllib.request.Request = Request


def worker(args) -> int:
    sc = next(s for s in load_split(args.data, args.split) if s["scenario"] == args.scenario)
    cache_dir = Path(args.out) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    home = prepare_env(args.env, cache_dir)
    sys.path.insert(0, str(Path(args.backend).resolve()))
    think_off()

    from primnox2 import paths
    from primnox2.chat import turns
    from primnox2.kernel import scheduler as sched
    from primnox2.memory import service as mem
    from primnox2.storage import db
    from primnox2.tools import runtime
    from v2 import store as v2_store

    paths.configure(home)
    db.configure(home / "primnox.db")
    db.init()
    v2_store.configure(home / "primnox_v2.db")
    sched.scheduler.start()
    from primnox2.settings import tunables

    def tunables_get(key):
        try:
            return tunables.get(key)
        except KeyError:
            return "n/a"

    embeddings_on = False
    try:
        from primnox2.memory import embeddings
        embeddings_on = bool(tunables.get("memory.embeddings"))
    except (ImportError, KeyError):
        embeddings = None
    if embeddings_on:
        up = embeddings.prepare(wait_s=180)
        print(f"embedding encoder up: {up}", flush=True)
        if not up:
            raise SystemExit("this arm has embeddings on and its encoder did not load")

    class Clock(datetime):
        day = _date(2000, 1, 1)

        @classmethod
        def now(cls, tz=None):
            naive = datetime(cls.day.year, cls.day.month, cls.day.day, 12, 0)
            return naive if tz is None else naive.replace(tzinfo=timezone.utc).astimezone(tz)

    runtime.datetime = Clock          # the one place the prompt gets its date

    from primnox2.models import gateway as _gw
    provider, model_in_use = _gw.active_provider()
    assert model_in_use == MODEL and getattr(provider, "is_ollama", False), (model_in_use, provider)
    print(f"model {model_in_use} via {provider.base_url} (native route: {provider.is_ollama}); "
          f"embeddings tunable: {tunables_get('memory.embeddings')}; "
          f"llm verify: {tunables_get('memory.llm_verify')}", flush=True)

    # What every model call was sent and what Ollama says it read. The memory
    # block is the system message `render_for_prompt` produces.
    from primnox2.models import gateway
    calls: list[dict] = []
    first_prompt: list[list[dict]] = []
    real_stream = gateway.stream_completion

    def counted_stream(messages, usage=None, **kw):
        usage = usage if usage is not None else {}
        text = [m["content"] for m in messages if isinstance(m.get("content"), str)]
        rec = {"chars": sum(map(len, text)),
               "memory_chars": sum(len(m["content"]) for m in messages
                                   if m["role"] == "system" and isinstance(m.get("content"), str)
                                   and m["content"].startswith("Background facts the user has shared"))}
        if not first_prompt:
            first_prompt.append([{"role": m["role"], "content": m["content"]} for m in messages
                                 if isinstance(m.get("content"), str)])
        calls.append(rec)
        try:
            yield from real_stream(messages, usage=usage, **kw)
        finally:
            rec["prompt_tokens"] = usage.get("input_tokens")
            rec["output_tokens"] = usage.get("output_tokens")

    gateway.stream_completion = counted_stream

    # `ask_user` parks a turn for ten minutes until a person answers, and then
    # carries on with "nobody answered, use your best judgement". Nobody is here,
    # so that outcome is reached after a second instead. How often it happens
    # is recorded (`asked_user`) rather than hidden.
    from primnox2.tools import permissions
    permissions.DEFAULT_TIMEOUT_S = 1
    # The same for a tool that must always ask whatever the auto-approve setting
    # (`always_ask`): its wait was bound as a keyword default at import, so the
    # line above does not reach it, and each such request parked a turn for ten
    # minutes before being denied. Denied after two seconds is the same outcome.
    type(permissions.broker).request.__kwdefaults__["timeout_s"] = 2

    def set_clock(day: str, stamp: int) -> None:
        Clock.day = _date.fromisoformat(day)
        mem.now_ms = lambda: stamp

    stacks = open(Path(args.out) / f"stacks-{args.scenario}.txt", "w")

    def run_turn(conversation_id: str, text: str) -> dict:
        t0 = time.time()
        first_call = len(calls)
        last: dict = {}
        tid = None
        for attempt in range(1, TURN_ATTEMPTS + 1):
            turn = (turns.create_turn(conversation_id, text) if tid is None
                    else turns.retry_turn(tid))
            tid = turn["turn_id"]
            sched.enqueue(tid, "chat.reply", {"conversation_id": conversation_id,
                                              "text": text, "model_profile": None})
            deadline = time.time() + TURN_TIMEOUT_S
            # A turn that outlives two minutes writes every thread's stack once,
            # so a stall can be located instead of guessed at.
            faulthandler.dump_traceback_later(120, file=stacks)
            while turns.status_of(tid) not in turns.TERMINAL and time.time() < deadline:
                time.sleep(0.4)
            faulthandler.cancel_dump_traceback_later()
            status = turns.status_of(tid)
            if status not in turns.TERMINAL:
                turns.cancel(tid)
                time.sleep(2)
                status = "timeout"
            hist = next((h for h in turns.get_history(conversation_id) if h["turn_id"] == tid), {})
            msg = hist.get("assistant_message") or {}
            blocks = msg.get("blocks") or []
            last = {
                "turn_id": tid, "status": status, "attempts": attempt,
                "text": msg.get("text") or "", "error": hist.get("error"),
                "tools": [{"name": b.get("name"), "arguments": b.get("arguments"),
                           "status": b.get("status"), "summary": b.get("summary")}
                          for b in blocks if b.get("type") == "tool_call"],
                "thinking_blocks": sum(1 for b in blocks if b.get("type") == "thinking"),
                "notices": [b.get("code") for b in blocks if b.get("type") == "notice"],
                "permissions": [{"action": b.get("action"), "resolved": b.get("resolved")}
                                for b in blocks if b.get("type") == "permission"],
            }
            if status == "completed":
                break
        last["seconds"] = round(time.time() - t0, 1)
        last["calls"] = [dict(c) for c in calls[first_call:]]
        return last

    def new_conversation() -> str:
        return turns.create_conversation("e2e")["id"]

    name = sc["scenario"].split("-")[0]
    statements = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
    if args.smoke:                      # plumbing check only; never a result
        statements, sc = statements[:3], {**sc, "questions": sc["questions"][:3]}
    t_start = time.time()

    # ── 1. statements ────────────────────────────────────────────────────────
    st_out, mem_to_stmt = [], {}
    for i, s in enumerate(statements, 1):
        set_clock(s["date"], ms(s["date"]))
        if args.save_mode == "fill":
            # The statement is filed exactly as the retrieval benchmark's chat
            # path files it, with no model turn: memory is complete, so only
            # retrieval and answering are measured, not the decision to save.
            try:
                filed = mem.remember(s["text"])
            except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                filed = None
            if filed and filed.get("duplicate_of") is None:
                mem_to_stmt[filed["id"]] = s["id"]
            st_out.append({"id": s["id"], "date": s["date"], "text": s["text"],
                           "replaces": s.get("replaces", []), "status": "filled" if filed else "refused",
                           "duplicate": bool(filed and filed.get("duplicate_of")),
                           "seconds": 0, "attempts": 0, "answer": "", "tools": [], "calls": [],
                           "thinking_blocks": 0, "called_remember": None, "asked_user": False,
                           "stored": [], "saved_voice": [], "saved_similarity": []})
            continue
        r = run_turn(new_conversation(), s["text"])
        rows = [dict(x) for x in db.connect().execute(
            "SELECT id, text, topic, kind, superseded_by FROM memories WHERE turn_id = ?",
            (r["turn_id"],))]
        for row in rows:
            mem_to_stmt[row["id"]] = s["id"]
        saved = [t for t in r["tools"] if t["name"] == "remember"]
        texts = [str((t["arguments"] or {}).get("text") or "") for t in saved]
        st_out.append({
            "id": s["id"], "date": s["date"], "text": s["text"], "replaces": s.get("replaces", []),
            "status": r["status"], "seconds": r["seconds"], "attempts": r["attempts"],
            "answer": r["text"], "tools": r["tools"], "thinking_blocks": r["thinking_blocks"],
            "calls": r["calls"], "permissions": r["permissions"], "called_remember": bool(saved),
            "asked_user": any(t["name"] == "ask_user" for t in r["tools"]),
            "stored": [row["text"] for row in rows],
            "saved_voice": [voice(t, name) for t in texts],
            "saved_similarity": [round(similarity(t, s["text"]), 3) for t in texts],
        })
        print(f"[{sc['scenario']}] statement {i}/{len(statements)} {r['status']} "
              f"remember={'Y' if saved else 'n'} {r['seconds']}s", flush=True)

    # A background verifier (when the arm has one on) must finish before anything
    # is asked, so the store is the same one the questions would meet.
    if hasattr(mem, "drain_verifications"):
        mem.drain_verifications()

    # Which statement each stored memory came from, and what the store believes
    # was replaced: the e2e analogue of the retrieval benchmark's supersession.
    store = [dict(x) for x in db.connect().execute(
        "SELECT id, text, topic, kind, superseded_by, created_at FROM memories")]
    truth = {(old, s["id"]) for s in statements for old in s.get("replaces", [])}
    got = {(mem_to_stmt.get(r["id"]), mem_to_stmt.get(r["superseded_by"]))
           for r in store if r["superseded_by"]}
    replaced_ever = {old for old, _ in truth}
    supersession = {"true": len(truth), "predicted": len(got), "correct": len(got & truth),
                    "false_retire": sum(1 for old, _ in got if old not in replaced_ever)}

    # ── 2. questions ─────────────────────────────────────────────────────────
    def snapshot() -> list[dict]:
        return [dict(x) for x in db.connect().execute("SELECT * FROM memories ORDER BY id")]

    base_rows = snapshot()

    base_by_id = {r["id"]: r for r in base_rows}

    def restore_if_changed() -> bool:
        """Put the memories table back as it was before the question. Rows are
        updated in place and new ones deleted last: `superseded_by` points at
        another memory, so rewriting the whole table in one go violates it."""
        now = snapshot()
        if now == base_rows:
            return False
        cols = list((base_rows or now)[0].keys())
        now_by_id = {r["id"]: r for r in now}
        with db.tx() as c:
            for rid, row in base_by_id.items():
                if rid not in now_by_id:
                    c.execute(f"INSERT INTO memories ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              tuple(row[k] for k in cols))
                elif now_by_id[rid] != row:
                    sets = [k for k in cols if k != "id"]
                    c.execute(f"UPDATE memories SET {','.join(k + '=?' for k in sets)} WHERE id=?",
                              (*[row[k] for k in sets], rid))
            for rid in now_by_id.keys() - base_by_id.keys():
                c.execute("DELETE FROM memories WHERE id=?", (rid,))
        return True

    q_out = []
    for i, q in enumerate(sc["questions"], 1):
        set_clock(sc["today"], ms(sc["today"]))
        r = run_turn(new_conversation(), q["text"])
        recalls = [t for t in r["tools"] if t["name"] == "recall_memory"]
        wrote = [t["name"] for t in r["tools"] if t["name"] in ("remember", "forget_memory")]
        side_effect = restore_if_changed()
        q_out.append({
            "id": q["id"], "type": q["type"], "text": q["text"], "answer_ids": q.get("answer_ids") or [],
            "as_of": q.get("as_of"), "status": r["status"], "seconds": r["seconds"],
            "attempts": r["attempts"], "answer": r["text"], "tools": r["tools"],
            "thinking_blocks": r["thinking_blocks"], "calls": r["calls"],
            "permissions": r["permissions"],
            "asked_user": any(t["name"] == "ask_user" for t in r["tools"]),
            "recall_calls": [{"query": (t["arguments"] or {}).get("query"),
                              "as_of": (t["arguments"] or {}).get("as_of"),
                              "when": (t["arguments"] or {}).get("when")} for t in recalls],
            "wrote_memory": wrote, "store_restored": side_effect,
        })
        print(f"[{sc['scenario']}] question {i}/{len(sc['questions'])} {q['type']} {r['status']} "
              f"recall={'Y' if recalls else 'n'} {r['seconds']}s", flush=True)

    # Ollama reports only the tokens it had to read after reusing a cached
    # prefix, so a per-turn count understates the prompt. One cold measurement
    # (a nonce in front defeats the cache) gives tokens per character for this
    # prompt style, which turns recorded character counts into prompt tokens.
    tokens_per_char = None
    if first_prompt:
        import uuid
        msgs = [dict(m) for m in first_prompt[0]]
        msgs[0]["content"] = f"[{uuid.uuid4().hex}] " + msgs[0]["content"]
        try:
            req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps({
                "model": MODEL, "messages": msgs, "stream": False,
                "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 1}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                cold = json.loads(resp.read())["prompt_eval_count"]
            tokens_per_char = cold / sum(len(m["content"]) for m in msgs)
        except (OSError, ValueError, KeyError) as exc:
            print(f"cold token calibration failed: {exc}", flush=True)

    out = Path(args.out) / f"{args.split}-{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sc['scenario']}.raw.json").write_text(json.dumps({
        "scenario": sc["scenario"], "today": sc["today"], "arm": args.arm,
        "backend": str(Path(args.backend).resolve()), "env": args.env,
        "model": MODEL, "save_mode": args.save_mode, "tokens_per_char": tokens_per_char, "statements": st_out, "questions": q_out,
        "store_after_statements": store, "supersession": supersession,
        "seconds": round(time.time() - t_start, 1),
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    probe = home / "sandbox" / ".capabilities.json"
    if probe.is_file() and not (cache_dir / ".capabilities.json").is_file():
        shutil.copy(probe, cache_dir / ".capabilities.json")
    sched.scheduler.stop()
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# RUN: one worker process per scenario
# ═════════════════════════════════════════════════════════════════════════════
def stage_run(args) -> None:
    data = load_split(args.data, args.split)
    out = Path(args.out) / f"{args.split}-{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    for sc in data:
        if args.only and sc["scenario"] not in args.only:
            continue
        if (out / f"{sc['scenario']}.raw.json").is_file():
            print(f"skip {sc['scenario']} (already run)")
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--scenario", sc["scenario"],
               "--split", args.split, "--arm", args.arm, "--data", args.data, "--out", args.out,
               "--backend", args.backend]
        for pair in args.env:
            cmd += ["--env", pair]
        cmd += ["--save-mode", args.save_mode]
        if args.smoke:
            cmd.append("--smoke")
        t = time.time()
        rc = subprocess.call(cmd)
        print(f"{sc['scenario']}: worker exit {rc} after {time.time() - t:.0f}s")
        if rc:
            raise SystemExit(f"worker failed for {sc['scenario']}")


# ═════════════════════════════════════════════════════════════════════════════
# JUDGE: a local model that is not the one under test
# ═════════════════════════════════════════════════════════════════════════════
VERDICTS = ("correct", "stale", "wrong", "hallucinated", "abstained")

JUDGE_PROMPT = """Grade an AI assistant's answer to a question the user asked about their own life, using ONLY the ground truth below. Question asked on {today}.

Question: {question}
{truth}

ASSISTANT'S ANSWER: {answer}

Judge only the value the question asks for. Extra, rambling or invented side details that do not contradict the truth do not matter; an answer that guesses or says it is assuming does not count as knowing.
{verdicts}
Reply with JSON only: {{"verdict": "{options}"}}"""

_V = {
    "correct": "correct = states the true value (mentioning an old value as history is fine).",
    "correct_multi": "correct = states ALL of the true values listed.",
    "stale": "stale = states an OUTDATED value as if it were current.",
    "wrong": "wrong = states a different value, or only some of several, or the value from the wrong time.",
    "abstained": "abstained = says it does not know / has no record, and gives no value.",
    "hallucinated": "hallucinated = gives a specific answer although the user never said anything about this.",
}
VERDICT_SETS = {
    "current": ("correct", "stale", "wrong", "abstained"),
    "past": ("correct", "wrong", "abstained"),
    "multi": ("correct_multi", "wrong", "abstained"),
    "absent": ("abstained", "hallucinated"),
}


def verdict_block(qtype: str) -> tuple[str, str]:
    keys = VERDICT_SETS[qtype]
    return chr(10).join(_V[k] for k in keys), "|".join(k.replace("_multi", "") for k in keys)


def build_truth(sc: dict, q: dict) -> str:
    by_id = {s["id"]: s for s in sc["statements"]}
    replaces = {s["id"]: set(s.get("replaces") or []) for s in sc["statements"]}

    def ancestors(i: str) -> set[str]:
        seen, stack = set(), list(replaces.get(i, ()))
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(replaces.get(x, ()))
        return seen

    answers = list(q.get("answer_ids") or [])
    line = lambda sid: f'- "{by_id[sid]["text"]}" (said {by_id[sid]["date"]})'
    if q["type"] == "absent":
        return "TRUTH: the user never said anything that answers this. Any specific answer is invented."
    others: set[str] = set()
    for a in answers:
        others |= ancestors(a)
        if q["type"] == "past":
            others |= {s for s in replaces if a in ancestors(s)}
    others -= set(answers)
    head = {"current": "TRUTH (current; all together are the answer):" if len(answers) > 1 else "TRUTH (current):",
            "past": f"TRUTH (the value that held on {q['as_of']}):",
            "multi": "TRUTH (the user has several values; all are correct and all must be given):"}[q["type"]]
    parts = [head, *[line(a) for a in answers]]
    if others:
        parts += ["NOT the answer (other values the user gave at other times):",
                  *[line(o) for o in sorted(others, key=lambda x: by_id[x]["date"])]]
    return "\n".join(parts)


def judge_one(prompt: str, model: str, host: str = OLLAMA) -> dict:
    """Ask the judge for a verdict. Retried: the first call after a model swap
    can be slow, and a small reasoning budget can come back empty."""
    tries = 0
    for attempt, predict in enumerate((600, 600, 2000, 4000)):
        tries += 1
        try:
            req = urllib.request.Request(f"{host}/api/chat", data=json.dumps({
                "model": model, "stream": False, "format": "json", "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": predict,
                            # CPU only: the judge never takes the chat model's GPU
                            "num_gpu": 0}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                text = json.loads(resp.read())["message"].get("content") or ""
            obj = json.loads(text)
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                return {"verdict": verdict, "judge": model, "tries": tries}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
            pass
        time.sleep(2 * 2 ** attempt)
    return {"verdict": None, "judge": model, "tries": tries}


def stage_judge(args) -> None:
    data = {s["scenario"]: s for s in load_split(args.data, args.split)}
    out = Path(args.out) / f"{args.split}-{args.arm}"
    n = failed = 0
    for path in sorted(out.glob("*.raw.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sc = data[raw["scenario"]]
        qs = {q["id"]: q for q in sc["questions"]}
        judged_path = out / f"{raw['scenario']}.judged.json"
        done = json.loads(judged_path.read_text(encoding="utf-8")) if judged_path.is_file() else {}
        for rq in raw["questions"]:
            if done.get(rq["id"], {}).get("verdict"):
                continue
            answer = rq["answer"].strip()[:1200] or "(no answer: the turn produced no text)"
            verdicts, options = verdict_block(rq["type"])
            prompt = JUDGE_PROMPT.format(today=sc["today"], question=rq["text"],
                                         truth=build_truth(sc, qs[rq["id"]]), answer=answer,
                                         verdicts=verdicts, options=options)
            done[rq["id"]] = judge_one(prompt, args.judge_model)
            judged_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
            n += 1
            failed += not done[rq["id"]]["verdict"]
            if n % 10 == 0:
                print(f"  judged {n}", flush=True)
    print(f"judge done: {n} answers graded by {args.judge_model}; {failed} could not be judged")


# ═════════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════════
def load_arm(args, arm: str) -> tuple[list[dict], list[dict]]:
    out = Path(args.out) / f"{args.split}-{arm}"
    stmts, qs = [], []
    for path in sorted(out.glob("*.raw.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        jp = out / f"{raw['scenario']}.judged.json"
        judged = json.loads(jp.read_text(encoding="utf-8")) if jp.is_file() else {}
        for s in raw["statements"]:
            stmts.append({"scenario": raw["scenario"], **s})
        for q in raw["questions"]:
            v = judged.get(q["id"], {})
            verdict = v.get("verdict")
            right = verdict == ("abstained" if q["type"] == "absent" else "correct")
            qs.append({"scenario": raw["scenario"], **q, "verdict": verdict, "correct": right,
                       "judge": v.get("judge"), "tokens_per_char": raw.get("tokens_per_char")})
    return stmts, qs


def report_arm(arm: str, stmts: list[dict], qs: list[dict], supers: dict) -> None:
    judged = [q for q in qs if q["verdict"]]
    print(f"\n=== arm: {arm}   {len(stmts)} statements, {len(qs)} questions "
          f"({len(qs) - len(judged)} unjudged) ===")
    print("answer accuracy (absent = abstained)")
    for t in ("current", "past", "multi", "absent"):
        sub = [q for q in judged if q["type"] == t]
        print(f"  {t:<8} {fmt(sum(q['correct'] for q in sub), len(sub))}")
    ans = [q for q in judged if q["type"] != "absent"]
    print(f"  answerable overall  {fmt(sum(q['correct'] for q in ans), len(ans))}")
    cur = [q for q in judged if q["type"] == "current"]
    print(f"stale answer on current questions  {fmt(sum(q['verdict'] == 'stale' for q in cur), len(cur))}")
    ab = [q for q in judged if q["type"] == "absent"]
    print(f"hallucinated on absent             {fmt(sum(q['verdict'] == 'hallucinated' for q in ab), len(ab))}")
    gave_up = [q for q in ans if q["verdict"] == "abstained"]
    print(f"abstained on an answerable question {fmt(len(gave_up), len(ans))}")
    print("verdicts, answerable: " + ", ".join(
        f"{v}={sum(q['verdict'] == v for q in ans)}" for v in VERDICTS))

    asked = [s for s in stmts if s["called_remember"] is not None]
    ok = [s for s in asked if s["status"] == "completed"]
    rem = [s for s in asked if s["called_remember"]]
    if asked:
        print(f"remember called on statements      {fmt(len(rem), len(asked))}"
              f"   (turns not completed: {len(asked) - len(ok)})")
        upd = [s for s in asked if s["replaces"]]
        print(f"  ...on update statements          {fmt(sum(s['called_remember'] for s in upd), len(upd))}")
    else:
        print("statements were filed directly (save-mode fill): no remember-call numbers")
    voices = [v for s in rem for v in s["saved_voice"]]
    sims = [x for s in rem for x in s["saved_similarity"]]
    print(f"saved texts: {len(voices)}  first-person {fmt(voices.count('first'), len(voices))}"
          f"  third {voices.count('third')}  subjectless {voices.count('subjectless')}")
    print(f"  paraphrased (similarity < 0.85 to the statement)  "
          f"{fmt(sum(1 for x in sims if x < 0.85), len(sims))}")

    recalled = [q for q in qs if q["recall_calls"]]
    print(f"recall_memory called on questions  {fmt(len(recalled), len(qs))}")
    for t in ("current", "past", "multi", "absent"):
        sub = [q for q in qs if q["type"] == t]
        print(f"  {t:<8} {fmt(sum(bool(q['recall_calls']) for q in sub), len(sub))}")
    past = [q for q in qs if q["type"] == "past"]
    dated = lambda q: any(c.get("as_of") or c.get("when") for c in q["recall_calls"])
    print(f"past: model supplied as_of or when {fmt(sum(dated(q) for q in past), len(past))}"
          f"   (as_of {sum(any(c.get('as_of') for c in q['recall_calls']) for q in past)},"
          f" when {sum(any(c.get('when') for c in q['recall_calls']) for q in past)};"
          f" {sum(bool(q['recall_calls']) for q in past)} called recall at all)")
    now_qs = [q for q in qs if q["type"] in ("current", "multi")]
    print(f"current/multi: model added a date  {fmt(sum(dated(q) for q in now_qs), len(now_qs))}")

    # Prompt size of the question turns. Ollama's own count is only what it had
    # to read after reusing a cached prefix, so the full size is estimated from
    # characters sent and one cold tokens-per-character measurement per scenario.
    def per_turn(q):
        ratio = q.get("tokens_per_char")
        calls = q.get("calls") or []
        if not ratio or not calls:
            return None
        return {"calls": len(calls),
                "first_prompt": calls[0]["chars"] * ratio, "first_memory": calls[0]["memory_chars"] * ratio,
                "total_prompt": sum(c["chars"] for c in calls) * ratio,
                "total_memory": sum(c["memory_chars"] for c in calls) * ratio,
                "evaluated": sum(c.get("prompt_tokens") or 0 for c in calls)}
    rows = [r for r in map(per_turn, qs) if r]
    if rows:
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        print(f"prompt tokens per question turn ({len(rows)} turns, mean): "
              f"first call {mean('first_prompt'):.0f} of which memory block {mean('first_memory'):.0f}; "
              f"all calls {mean('total_prompt'):.0f} of which memory {mean('total_memory'):.0f}; "
              f"model calls per turn {mean('calls'):.2f}; "
              f"tokens Ollama actually evaluated {mean('evaluated'):.0f}")
    bad = [q for q in qs if q["status"] != "completed"]
    think = sum(s["thinking_blocks"] for s in stmts) + sum(q["thinking_blocks"] for q in qs)
    wrote = sum(bool(q["wrote_memory"]) for q in qs)
    print(f"questions whose turn did not complete: {len(bad)}; "
          f"questions that wrote/forgot memory: {wrote}; thinking blocks seen: {think}")
    print(f"turns where the model asked the user a question (auto-unanswered): "
          f"statements {sum(s.get('asked_user', False) for s in stmts)}/{len(stmts)}, "
          f"questions {sum(q.get('asked_user', False) for q in qs)}/{len(qs)}")
    print(f"turns that raised a permission prompt (denied unattended): "
          f"statements {sum(bool(s.get('permissions')) for s in stmts)}, "
          f"questions {sum(bool(q.get('permissions')) for q in qs)}")
    judges = {}
    for q in judged:
        judges[q["judge"]] = judges.get(q["judge"], 0) + 1
    print(f"judge models: {judges}")
    print(f"store after statements: supersession true={supers['true']} predicted={supers['predicted']} "
          f"correct={supers['correct']} false-retire={supers['false_retire']}")


def mcnemar_line(label: str, a: dict, b: dict, keys: list) -> str:
    only_a = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    only_b = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    n = only_a + only_b
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return (f"  {label:<11} n={len(keys):<4} A {sum(a[k]['correct'] for k in keys):>3}  "
            f"B {sum(b[k]['correct'] for k in keys):>3}   only-A {only_a:>3}  only-B {only_b:>3}   "
            f"p={min(1.0, 2 * tail):.4f}")


def stage_report(args) -> None:
    loaded = {}
    for arm in args.arms:
        stmts, qs = load_arm(args, arm)
        sup = {k: 0 for k in ("true", "predicted", "correct", "false_retire")}
        for path in (Path(args.out) / f"{args.split}-{arm}").glob("*.raw.json"):
            for k, v in json.loads(path.read_text(encoding="utf-8"))["supersession"].items():
                sup[k] += v
        loaded[arm] = {(q["scenario"], q["id"]): q for q in qs}
        report_arm(arm, stmts, qs, sup)
        if args.split.startswith("test") is False and args.verbose:
            for q in qs:
                print(f"  [{q['type']}] {q['text']}\n     -> {q['verdict']}: {q['answer'][:200]!r}")
    if len(args.arms) == 2:
        a, b = (loaded[x] for x in args.arms)
        keys = sorted(k for k in a if k in b and a[k]["verdict"] and b[k]["verdict"])
        print(f"\n=== paired: A={args.arms[0]}  B={args.arms[1]}  ({len(keys)} questions judged in both) ===")
        for label, types in (("answerable", ("current", "past", "multi")), ("current", ("current",)),
                             ("past", ("past",)), ("multi", ("multi",)), ("absent", ("absent",)),
                             ("all", ("current", "past", "multi", "absent"))):
            ks = [k for k in keys if a[k]["type"] in types]
            if ks:
                print(mcnemar_line(label, a, b, ks))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, help="dev, test, or another file stem under --data")
    ap.add_argument("--arm", default="base", help="a label for this configuration")
    ap.add_argument("--backend", default=str(ROOT / "backend"),
                    help="a directory containing primnox2 to import")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="set before importing primnox2 (e.g. PRIMNOX2_MEMORY_LLM_VERIFY=1); repeatable")
    ap.add_argument("--stage", default="run,judge,report")
    ap.add_argument("--arms", nargs="+", help="arm labels for the report stage (two = paired)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="*", help="scenario names to run")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--verbose", action="store_true", help="dev only: print each answer and verdict")
    ap.add_argument("--smoke", action="store_true", help="3 statements and 3 questions per scenario")
    ap.add_argument("--save-mode", choices=("model", "fill"), default="model",
                    help="model: statements are chat turns and the model decides whether to save "
                         "(default, realistic). fill: statements are filed straight into memory "
                         "(isolates retrieval and answering from the decision to save)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scenario", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    if args.verbose and args.split.startswith("test"):
        print("refused: --verbose on a test split.")
        return 2
    stages = args.stage.split(",")
    if "run" in stages:
        stage_run(args)
    if "judge" in stages:
        stage_judge(args)
    if "report" in stages:
        args.arms = args.arms or [args.arm]
        stage_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
