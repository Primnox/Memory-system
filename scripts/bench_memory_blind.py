"""Memory against a BLIND dataset: written from a spec by someone who never saw
this code, frozen before the first run.

`bench_memory.py` was tuned against its own corpora — every miss there was
looked at and fixed — so its scores are development scores. This is the test.

Rules this script enforces, because the point is that they cannot be broken by
accident:

  * The TEST split prints aggregates only. `--misses` is refused for it: a
    miss read is a miss tuned against, and the set is then spent.
  * Every rate carries a 95% Wilson interval. 12/12 is "somewhere between 74%
    and 100%", and should be read that way.
  * Ground truth is the dataset's. Nothing here reinterprets it.

Query modes (how the question becomes a search):

  question  the user's question verbatim, no date. The floor.
  oracle    the question verbatim plus the dataset's own `as_of`. Retrieval
            alone, with time handed over — an upper bound for time questions.
  model     a local model (Ollama; by default the 9B the app runs) is given the
            question, today's date and the recall_memory tool's own description
            and arguments, and writes them itself ({query, when, as_of}). The
            date then goes through the tool's own resolver (memory/when.py),
            with the scenario's `today`. The realistic one.

Load paths:

  chat      `remember()` one statement at a time, with the clock set to that
            statement's date — the path a conversation takes.
  import    `import_many()` with each statement's date.

Arms:

  --backend DIR   import primnox2 from DIR instead (a worktree of older code),
                  for a before/after on the same questions. Code without
                  `as_of` searches without it.
  --no-topics     the hand-written topic lexicon off, everything else on — how
                  much of a result is the overfit-prone part.
  --embeddings search|supersede|both|off
                  the sentence-encoder arms. Without the flag they are as the app
                  ships them (both on); `off` scores the word rules alone. `search`
                  fuses meaning with the lexical order; `supersede` lets meaning
                  propose that a new memory updates an old one. Whatever is on, the
                  encoder is loaded and waited for first, and the run aborts rather
                  than score the fallback if it cannot be. Compare arms on identical
                  questions with --json and --mcnemar.
  --actor-guard   memory.actor_guard (on by default now): "My brother works at
                  Google" cannot retire "I work at Acme". Not an embeddings setting.
  --llm-verify    memory.llm_verify: the local model checks updates the rules were
                  unsure of, in the background; the queue is drained after each
                  scenario's load. Prints calls per save and latency per question.
                  --verify-log FILE appends every question and answer, labelled
                  with the dataset's truth, as JSONL.
  --mcnemar A B   paired exact McNemar test between two `--json` outputs.

Usage:
    python scripts/bench_memory_blind.py --split dev --misses
    python scripts/bench_memory_blind.py --split test --query oracle --json out.json
    python scripts/bench_memory_blind.py --split test --query model --load chat
    python scripts/bench_memory_blind.py --split test --query oracle --embeddings both --json hybrid.json
    python scripts/bench_memory_blind.py --mcnemar before.json after.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

DEFAULT_DATA = ROOT / "scripts" / "blind_memory"
DEFAULT_MODEL = "huihui_ai/qwen3.5-abliterated:9b"


def wilson(k: int, n: int) -> tuple[float, float, float] | None:
    """Point estimate and 95% Wilson interval, in percent."""
    if n == 0:
        return None
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * p, 1), round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)


def fmt(k: int, n: int) -> str:
    w = wilson(k, n)
    return "   n/a" if w is None else f"{w[0]:5.1f}% [{w[1]:5.1f}–{w[2]:5.1f}]  ({k}/{n})"


def ms(date: str, end_of_day: bool = False) -> int:
    d = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000) + (86_400_000 - 1 if end_of_day else 12 * 3_600_000)


# ── the model as query writer ────────────────────────────────────────────────
def model_query(question: str, today: str, model: str, tool_description: str,
                params: dict | None = None) -> dict:
    # The arguments are the tool's own, as the app hands them to the model: their
    # descriptions are part of what is measured.
    args = "; ".join(
        f"{name} ({spec.get('type', 'string')}{'' if spec.get('required') else ', optional'}"
        f"{' — ' + spec['description'] if spec.get('description') else ''})"
        for name, spec in (params or {}).items() if name != "limit") or (
        "query (string, what to look for); as_of (optional, YYYY-MM or YYYY-MM-DD — only "
        "when the user asks how things were at an earlier time)")
    shapes = {"query": '"..."', "as_of": 'null or "YYYY-MM-DD"', "when": "null or \"the user's words\""}
    names = [n for n in (params or {}) if n != "limit"] or ["query", "as_of"]
    reply = ", ".join(f'"{n}": ' + shapes.get(n, 'null or "..."') for n in names)
    prompt = (
        f"Today is {today}. You are deciding how to call a memory tool.\n\n"
        f"Tool recall_memory: {tool_description}\n"
        f"Arguments: {args}.\n\n"
        f"User: {question}\n\n"
        f"Reply with JSON only: {{{reply}}}")
    if model.startswith("omniroute:"):
        # OmniRoute's OpenAI-compatible gateway (free OpenRouter combos such as
        # auto/best-free). Its router can land on an unhealthy model and hand
        # back an empty body, so a bad reply is retried like a timeout.
        body = json.dumps({"model": model.split(":", 1)[1], "temperature": 0, "stream": False,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        url = "http://127.0.0.1:20128/v1/chat/completions"
        pick = lambda d: d["choices"][0]["message"]["content"]
    else:
        # think=False: the app runs this model with thinking off, and qwen3.5
        # thinks by default — minutes per question for a one-line JSON reply.
        body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                           "think": False, "options": {"temperature": 0}}).encode()
        url = "http://localhost:11434/api/generate"
        pick = lambda d: d["response"]
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    out = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = pick(json.loads(r.read())) or ""
            start, end = raw.find("{"), raw.rfind("}")
            out = json.loads(raw[start:end + 1])
            break
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            if attempt == 3:
                return {"query": question, "as_of": None, "when": None, "malformed": True}
            time.sleep(2 * (attempt + 1))
    if not isinstance(out, dict):
        return {"query": question, "as_of": None, "when": None, "malformed": True}
    given = {k: (None if out.get(k) in ("", "null", "None") else out.get(k)) for k in ("as_of", "when")}
    return {"query": str(out.get("query") or question), **given}


# ── one scenario ─────────────────────────────────────────────────────────────
def disable_topics() -> None:
    """The --no-topics arm. Patched on the module, which every caller reads
    through, so it holds for the rest of the process."""
    from primnox2.cognition import topics
    topics.classify = lambda text: (None, topics.kind_of(text))
    topics.topic_of = lambda text: None
    topics.topic_of_query = lambda query: None


def enable_arms(embeddings: str | None, actor_guard: bool, llm_verify: bool = False) -> None:
    """The --embeddings / --actor-guard arms, set through the tunables' own
    environment variables so the code under test reads them exactly as the app
    would. The encoder is loaded here, once, so no scenario is scored against
    the lexical fallback while it warms up."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cached encoder, no network probe
    if embeddings:
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS"] = "1" if embeddings in ("search", "both") else "0"
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE"] = "1" if embeddings in ("supersede", "both") else "0"
    if actor_guard:
        os.environ["PRIMNOX2_MEMORY_ACTOR_GUARD"] = "1"
    if llm_verify:
        os.environ["PRIMNOX2_MEMORY_LLM_VERIFY"] = "1"
    # Whatever is on, as the app ships it (the arms are on by default), is waited for:
    # nothing is scored against the word-rules fallback while the encoder warms up.
    from primnox2.settings import tunables
    if (tunables.get("memory.embeddings") or tunables.get("memory.embeddings_supersede")
            or tunables.get("memory.llm_verify")):
        from primnox2.memory import embeddings as enc
        if not enc.prepare(300):
            raise SystemExit("the sentence encoder could not be loaded; refusing to score the "
                             "embeddings arm against the lexical fallback (--embeddings off "
                             "scores the word rules alone)")


def run_scenario(sc: dict, load: str, query_mode: str, model: str, verify_log: str | None = None) -> dict:
    from primnox2.storage import db
    path = Path(tempfile.mkdtemp(prefix="blind-")) / "primnox.db"
    db.configure(path)
    db.init()
    from primnox2.memory import service as mem
    from primnox2.tools import builtins  # noqa: F401
    from primnox2.tools.registry import get as get_tool
    timed = "as_of" in inspect.signature(mem.search).parameters
    parse_as_of = getattr(mem, "parse_as_of", None)
    try:
        from primnox2.memory import when as time_words
    except ImportError:   # an older --backend tree has no resolver
        time_words = None

    statements = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
    loaded = {"statements": len(statements), "refused": 0, "duplicates": 0}
    # memory row id -> statement id. Matched by the returned id (chat) or by the
    # SANITISED text (import): remember() stores sanitised text, so matching on
    # the raw sentence silently lost any statement the sanitiser touched.
    sid: dict[str, str | None] = {}
    if load == "import":
        out = mem.import_many([{"text": s["text"], "created_at": ms(s["date"])} for s in statements])
        loaded["duplicates"] = out.get("duplicates", 0)
        by_text = {mem._sanitize(s["text"]): s["id"] for s in statements}
        for r in db.connect().execute("SELECT id, text FROM memories"):
            sid[r["id"]] = by_text.get(r["text"])
    else:
        real_now = mem.now_ms
        try:
            for s in statements:
                stamp = ms(s["date"])
                mem.now_ms = lambda stamp=stamp: stamp
                try:
                    out = mem.remember(s["text"])
                except (mem.MemoryRejected, mem.MemoryTooLong, ValueError):
                    loaded["refused"] += 1   # the system's answer; scored as a miss downstream
                    continue
                if out.get("duplicate_of") is None:
                    sid[out["id"]] = s["id"]
                else:
                    loaded["duplicates"] += 1
        finally:
            mem.now_ms = real_now

    verified = None
    if getattr(mem, "verifier", None) is not None and mem.verifier.enabled():
        waited = time.perf_counter()
        mem.drain_verifications()
        verified = {**mem.verifier.stats, "drain_s": round(time.perf_counter() - waited, 1)}
        mem.verifier.reset_stats()

    rows = [dict(r) for r in db.connect().execute("SELECT * FROM memories")]

    # supersession, by (old, successor) pair
    truth = {(old, s["id"]) for s in statements for old in s.get("replaces", [])}
    replaced_ever = {old for old, _ in truth}
    got = {(sid.get(r["id"]), sid.get(r["superseded_by"])) for r in rows if r.get("superseded_by")}
    sup = {"true": len(truth), "predicted": len(got), "correct": len(got & truth),
           "false_retire": sum(1 for old, _ in got if old not in replaced_ever)}

    if verified is not None and verify_log:
        with open(verify_log, "a", encoding="utf-8") as f:
            for d in mem.verifier.decisions():
                pair = (sid.get(d["old_id"]), sid.get(d["new_id"]))
                f.write(json.dumps({"scenario": sc["scenario"], "old_statement": pair[0], "new_statement": pair[1],
                                    "truth_replaces": pair in truth, "old_ever_replaced": pair[0] in replaced_ever,
                                    **{k: d[k] for k in ("old_text", "new_text", "similarity", "relation",
                                                         "confidence", "applied", "ms", "model")}},
                                   ensure_ascii=False) + "\n")

    tool = get_tool("recall_memory")
    tool_desc, tool_params = tool.description, tool.parameters
    today_ms = ms(sc["today"], end_of_day=True)
    current_stale = {old for old, new in truth}  # replaced by something dated <= today (all are)
    qs = []
    for q in sc["questions"]:
        if query_mode == "question":
            query, as_of, malformed = q["text"], None, False
        elif query_mode == "oracle":
            query, as_of, malformed = q["text"], (q.get("as_of") or None), False
        else:
            mq = model_query(q["text"], sc["today"], model, tool_desc, tool_params)
            query, as_of, malformed = mq["query"], mq["as_of"], mq.get("malformed", False)
            if time_words is not None:
                # the tool's own resolution, on the scenario's day rather than the wall clock
                as_of = time_words.as_of_for(when=mq.get("when"), as_of=as_of, query=query,
                                             today=date.fromisoformat(sc["today"]))
        if as_of and parse_as_of is not None and parse_as_of(as_of) is None:
            as_of = None
        hits = mem.search(query, limit=10, as_of=as_of) if timed else mem.search(query, limit=10)
        if as_of is None or not timed:
            # "now" for the scenario is its own `today`, not the wall clock
            hits = [h for h in hits if h["created_at"] <= today_ms]
        ids = [sid.get(h["id"]) for h in hits]
        answers = set(q.get("answer_ids") or [])
        k = max(1, len(answers))
        qs.append({
            "type": q["type"], "malformed": malformed,
            "top1": bool(ids) and ids[0] in answers,
            "top3": bool(answers & set(ids[:3])),
            "all_k2": bool(answers) and answers <= set(ids[:k + 2]),
            "recall_k2": (len(answers & set(ids[:k + 2])) / len(answers)) if answers else None,
            "empty": not ids,
            "stale_top": q["type"] == "current" and bool(ids) and ids[0] in current_stale,
            "model_as_of": as_of if query_mode == "model" else None,
            "model_when": mq.get("when") if query_mode == "model" else None,
            "true_as_of": q.get("as_of"),
            "id": q["id"], "text": q["text"], "query": query, "want": sorted(answers),
            "got": ids[:3],
        })
    return {"scenario": sc["scenario"], "sup": sup, "loaded": loaded, "qs": qs, "verified": verified}


# ── reporting ────────────────────────────────────────────────────────────────
def report(results: list[dict], show_misses: bool) -> None:
    qs = [q for r in results for q in r["qs"]]
    print(f"\n{len(results)} scenarios, {len(qs)} questions")
    print(f"{'':<10}{'top-1':>36}{'top-3':>36}")
    for t in ("current", "past", "multi"):
        sub = [q for q in qs if q["type"] == t]
        if not sub:
            continue
        print(f"{t:<10}{fmt(sum(q['top1'] for q in sub), len(sub)):>36}"
              f"{fmt(sum(q['top3'] for q in sub), len(sub)):>36}")
    multi = [q for q in qs if q["type"] == "multi"]
    if multi:
        print(f"multi: every value in top k+2  {fmt(sum(q['all_k2'] for q in multi), len(multi))}")
    answerable = [q for q in qs if q["type"] != "absent"]
    print(f"{'answerable':<10}{fmt(sum(q['top1'] for q in answerable), len(answerable)):>36}"
          f"{fmt(sum(q['top3'] for q in answerable), len(answerable)):>36}")
    cur = [q for q in qs if q["type"] == "current"]
    print(f"stale memory on top for a 'now' question  {fmt(sum(q['stale_top'] for q in cur), len(cur))}")
    absent = [q for q in qs if q["type"] == "absent"]
    if absent:
        print(f"absent: returned nothing (retrieval can rarely do this)  "
              f"{fmt(sum(q['empty'] for q in absent), len(absent))}")
    model_past = [q for q in qs if q["type"] == "past"]
    if any(q["model_as_of"] for q in qs):
        print(f"past: model supplied a date at all  {fmt(sum(bool(q['model_as_of']) for q in model_past), len(model_past))}")
        wrongly_dated = [q for q in qs if q["type"] == "current" and q["model_as_of"]]
        print(f"current: model added a date it shouldn't  {fmt(len(wrongly_dated), len(cur))}")
    mal = sum(q["malformed"] for q in qs)
    if mal:
        print(f"malformed model replies: {mal}")

    ld = {k: sum(r["loaded"][k] for r in results) for k in ("statements", "refused", "duplicates")}
    print(f"\nload: {ld['statements']} statements, {ld['refused']} refused, "
          f"{ld['duplicates']} dropped as duplicates")

    tot = {k: sum(r["sup"][k] for r in results) for k in ("true", "predicted", "correct", "false_retire")}
    print("\nsupersession (by old→successor pair)")
    print(f"  recall     {fmt(tot['correct'], tot['true'])}")
    print(f"  precision  {fmt(tot['correct'], tot['predicted'])}")
    print(f"  retired something never replaced: {tot['false_retire']} of {tot['predicted']} retirements")

    ver = [r["verified"] for r in results if r.get("verified")]
    if ver:
        tv = {k: sum(v[k] for v in ver) for k in ("jobs", "asked", "answered", "failed", "skipped", "retired", "ms")}
        saves = sum(r["loaded"]["statements"] for r in results)
        print(f"\nverifier: {tv['jobs']} saves with candidates of {saves}, {tv['asked']} questions "
              f"({tv['asked'] / max(1, saves):.2f} per save), {tv['answered']} answered, {tv['failed']} failed, "
              f"{tv['skipped']} skipped, {tv['retired']} retired")
        print(f"  latency per question {tv['ms'] / max(1, tv['answered']):.0f} ms; "
              f"queue drained after load in {sum(v['drain_s'] for v in ver):.0f} s over {len(ver)} scenarios")

    if show_misses:
        print("\nmisses (DEV ONLY)")
        for q in qs:
            if q["type"] != "absent" and not q["top1"]:
                print(f"  [{q['type']}] {q['text']}\n     query={q['query']!r} when={q['model_when']!r} as_of={q['model_as_of'] or q['true_as_of']}"
                      f"\n     want={q['want']} got={q['got']}")
        for r in results:
            print(f"  {r['scenario']} supersession {r['sup']}")


def mcnemar(a_path: str, b_path: str) -> int:
    """Exact two-sided McNemar on top-1, paired by (scenario, question id).

    Two systems answering the SAME questions are not independent samples;
    comparing their two intervals throws the pairing away and needs far more
    data to see a real difference. Only the discordant pairs carry signal.
    """
    def load(p: str) -> tuple[dict, dict]:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        return d["config"], {(q["scenario"], q["id"]): q for q in d["qs"]}
    (ca, a), (cb, b) = load(a_path), load(b_path)
    print(f"A = {ca}\nB = {cb}")
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        print(f"warning: {len(a)} vs {len(b)} questions, {len(keys)} paired")
    for label, types in (("answerable", ("current", "past", "multi")),
                         ("current", ("current",)), ("past", ("past",)), ("multi", ("multi",))):
        ks = [k for k in keys if a[k]["type"] in types]
        if not ks:
            continue
        only_a = sum(1 for k in ks if a[k]["top1"] and not b[k]["top1"])
        only_b = sum(1 for k in ks if b[k]["top1"] and not a[k]["top1"])
        n = only_a + only_b
        tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
        p = min(1.0, 2 * tail)
        print(f"{label:<11} n={len(ks):<4} A {sum(a[k]['top1'] for k in ks):>3}  "
              f"B {sum(b[k]['top1'] for k in ks):>3}   only-A {only_a:>3}  only-B {only_b:>3}   p={p:.4f}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", help="dataset file stem under --data: dev, test, or a "
                    "variant such as test_para1. Anything starting 'test' is a test split.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--load", choices=("chat", "import"), default="chat")
    ap.add_argument("--query", choices=("question", "oracle", "model"), default="oracle")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", help="primnox2's parent directory to import instead")
    ap.add_argument("--no-topics", action="store_true")
    ap.add_argument("--embeddings", choices=("search", "supersede", "both", "off"),
                    help="search: hybrid ranking (memory.embeddings); supersede: meaning proposes "
                         "updates at write time (memory.embeddings_supersede); both. Needs the "
                         "cached encoder, which is waited for before anything is scored.")
    ap.add_argument("--actor-guard", action="store_true",
                    help="memory.actor_guard: a statement about someone else never retires one "
                         "about the user. Not an embeddings arm; separate so it can be measured alone.")
    ap.add_argument("--llm-verify", action="store_true",
                    help="memory.llm_verify: the local model checks uncertain updates in the background "
                         "(needs Ollama; the queue is drained before each scenario is scored).")
    ap.add_argument("--verify-log", help="append each verifier question and answer to this JSONL file")
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--mcnemar", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    if args.mcnemar:
        return mcnemar(*args.mcnemar)
    if not args.split:
        ap.error("--split is required")
    if args.misses and args.split.startswith("test"):
        print("refused: --misses on a test split. Reading test misses spends the set.")
        return 2
    if args.backend:
        sys.path.insert(0, str(Path(args.backend).resolve()))
    if args.no_topics:
        disable_topics()
    if args.backend:
        if args.embeddings or args.actor_guard or args.llm_verify:
            ap.error("--embeddings / --actor-guard / --llm-verify need this tree's primnox2, not --backend")
    else:
        enable_arms(args.embeddings, args.actor_guard, args.llm_verify)

    data = json.loads((Path(args.data) / f"{args.split}.json").read_text(encoding="utf-8"))
    config = (f"split={args.split} load={args.load} query={args.query}"
              f"{' model=' + args.model if args.query == 'model' else ''}"
              f"{' backend=' + args.backend if args.backend else ''}"
              f"{' no-topics' if args.no_topics else ''}"
              f"{' embeddings=' + args.embeddings if args.embeddings else ''}"
              f"{' actor-guard' if args.actor_guard else ''}"
              f"{' llm-verify' if args.llm_verify else ''}")
    t = time.perf_counter()
    results = [run_scenario(sc, args.load, args.query, args.model, args.verify_log) for sc in data]
    import primnox2
    print(f"{config}  ({time.perf_counter() - t:.1f}s)  code={Path(primnox2.__file__).parent}")
    report(results, args.misses)
    if args.json:
        slim = {"config": config,
                "sup": {r["scenario"]: r["sup"] for r in results},
                "loaded": {r["scenario"]: r["loaded"] for r in results},
                "qs": [{"scenario": r["scenario"],
                        **{k: q[k] for k in ("id", "type", "top1", "top3", "all_k2", "empty",
                                             "stale_top", "model_as_of")}}
                       for r in results for q in r["qs"]]}
        Path(args.json).write_text(json.dumps(slim, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
