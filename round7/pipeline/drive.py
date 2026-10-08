"""Drive agy (Antigravity CLI) to write a blind memory evaluation set, then audit it.

  python drive.py write          # writers in parallel -> <job>/final.json
  python drive.py audit          # cross-model label audit -> audit_<job>/labels.json
  python drive.py compare        # agreement report + merged sets in out/

Each agy run happens in its own empty folder, so it sees only the spec it is given.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
import shutil
AGY = shutil.which("agy") or "agy"  # the Antigravity CLI
SPEC = (HERE / "SPEC.md").read_text(encoding="utf-8")
AUDIT = (HERE / "AUDIT.md").read_text(encoding="utf-8")

BANNED_NAMES = {"anika", "bjorn", "carmen", "dmitri", "esther", "hiroshi", "nandini",
                "wanjiru", "rafael", "sven", "tessa", "walter"}
BANNED_CITIES = {"bengaluru", "bergen", "guadalajara", "toronto", "leeds", "osaka", "hyderabad",
                 "nairobi", "curitiba", "gothenburg", "perth", "ohio"}

COMMON = ("Give every scenario a different `today` between 2026-03-01 and 2026-10-05. "
          "Across your scenarios include at least one person under 22 and one over 65.")
JOBS = {
    "test_a": dict(model="gemini-3.1-pro-high", n=6, letters="ABCDEFGH", auditor="gemini-3.8-flash-high",
                   brief="Write 6 scenarios. Every person's first name starts with a letter A–H. One person "
                         "each from: Latin America; West or East Africa; Southeast Asia; Eastern Europe; "
                         "the Middle East or North Africa; Oceania or the Pacific islands. " + COMMON),
    "test_b": dict(model="gemini-3.8-flash-high", n=6, letters="NOPQRSTUVWXYZ", auditor="gemini-3.1-pro-high",
                   brief="Write 6 scenarios. Every person's first name starts with a letter N–Z. One person "
                         "each from: the USA or Canada; South Asia; East Asia; Western Europe; Southern "
                         "Africa; Central Asia or the Caribbean. " + COMMON),
    "dev": dict(model="gemini-3.8-flash-high", n=3, letters="IJKLM", auditor=None,
                brief="Write 3 scenarios. Every person's first name starts with a letter I–M. One person "
                      "each from: a Nordic country; the British Isles; South America. " + COMMON),
}

LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOCK:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(HERE / "drive.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def agy(prompt: str, model: str, cwd: Path, conversation: str | None = None,
        schema: Path | None = None) -> tuple[str, str]:
    """One print-mode turn. Returns (response text, conversation id)."""
    # The prompt goes in on stdin as stream-json: an argv prompt is capped at 32K on Windows.
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [AGY, "--input-format", "stream-json", "--output-format", "stream-json", "--model", model,
           "--print-timeout", "2400s"]
    if conversation:
        cmd += ["--conversation", conversation]
    if schema:
        cmd += ["--json-schema", str(schema)]
    cmd += ["-p="]
    msg = json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}) + "\n"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, input=msg, capture_output=True, text=True, encoding="utf-8", timeout=2700)
    stamp = time.strftime("%H%M%S")
    (cwd / f"raw_{stamp}.out").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    steps = {e["step_update"].get("step_type") for e in events if e.get("event") == "step_update"}
    result = next((e["result"] for e in reversed(events) if e.get("event") == "result"), None)
    if not result:
        raise RuntimeError(f"agy gave no result (exit {proc.returncode}): {proc.stderr[-500:]}")
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"agy status {result.get('status')}: {result.get('error')}")
    u = result.get("usage", {})
    log(f"  agy {model} in {cwd.name}: {time.time() - t0:.0f}s, out {u.get('output_tokens')} tok, "
        f"turns {result.get('num_turns')}, step types {sorted(s for s in steps if s)}")
    return result["response"], result["conversation_id"]


def parse(text: str) -> dict:
    """The last complete JSON object in the reply. With --json-schema agy may append the
    `finish` tool's structured copy after a text copy; the structured one comes last."""
    dec = json.JSONDecoder()
    found, i = [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        found.append(obj)
        i = end
    if not found:
        raise ValueError("no JSON object in the reply")
    return found[-1]


def resume_from_raw(cwd: Path) -> tuple[str, str] | None:
    raws = sorted(cwd.glob("raw_*.out"))
    if not raws:
        return None
    for line in reversed(raws[-1].read_text(encoding="utf-8").splitlines()):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "result" and e["result"].get("status") == "SUCCESS":
            return e["result"]["response"], e["result"]["conversation_id"]
    return None


def d(s: str) -> date:
    return date.fromisoformat(s)


def validate(data: dict, n: int, letters: str) -> list[str]:
    errs: list[str] = []
    scs = data.get("scenarios") or []
    if len(scs) != n:
        errs.append(f"expected {n} scenarios, got {len(scs)}")
    slugs = set()
    for sc in scs:
        name = sc.get("scenario", "?")
        p = f"[{name}]"
        if not re.fullmatch(r"[a-z]+(-[a-z]+){2,}", name):
            errs.append(f"{p} slug must be lowercase firstname-city-role")
        first = name.split("-")[0]
        if first in BANNED_NAMES or any(c in name.split("-") for c in BANNED_CITIES):
            errs.append(f"{p} uses a banned name or city")
        if first[:1].upper() not in letters:
            errs.append(f"{p} first name must start with one of {letters[0]}–{letters[-1]}")
        if name in slugs:
            errs.append(f"{p} duplicate slug")
        slugs.add(name)
        try:
            today = d(sc["today"])
        except Exception:  # noqa: BLE001
            errs.append(f"{p} bad today"); continue
        st = sc["statements"]
        long = len(st) >= 45  # the long tier (SPEC_LONG.md); shorter people are held to SPEC.md
        lim = (dict(st=(50, 85), up=(16, 30), q=(26, 38), mins=(("current", 10), ("past", 7), ("multi", 3), ("absent", 3)))
               if long else
               dict(st=(22, 30), up=(7, 12), q=(14, 18), mins=(("current", 6), ("past", 3), ("multi", 1), ("absent", 1))))
        ids = [s["id"] for s in st]
        if ids != [f"s{i}" for i in range(1, len(st) + 1)]:
            errs.append(f"{p} statement ids must be s1..s{len(st)} in order")
        if not lim["st"][0] <= len(st) <= lim["st"][1]:
            errs.append(f"{p} {len(st)} statements (want {'55–75' if long else '24–28'})")
        when = {}
        prev = None
        for s in st:
            try:
                when[s["id"]] = d(s["date"])
            except Exception:  # noqa: BLE001
                errs.append(f"{p} {s['id']} bad date"); continue
            if prev and when[s["id"]] <= prev:
                errs.append(f"{p} {s['id']} date not after the previous statement's")
            prev = when[s["id"]]
        if prev and prev > today:
            errs.append(f"{p} a statement is dated after today")
        replaced_by: dict[str, str] = {}
        for i, s in enumerate(st):
            for old in s["replaces"]:
                if old not in ids[:i]:
                    errs.append(f"{p} {s['id']} replaces {old}, which is not an earlier statement")
                elif old in replaced_by:
                    errs.append(f"{p} {old} is replaced twice ({replaced_by[old]} and {s['id']})")
                else:
                    replaced_by[old] = s["id"]
        if not lim["up"][0] <= len(replaced_by) <= lim["up"][1]:
            errs.append(f"{p} {len(replaced_by)} updates (want {'18–28' if long else '8–11'})")
        qs = sc["questions"]
        if [q["id"] for q in qs] != [f"q{i}" for i in range(1, len(qs) + 1)]:
            errs.append(f"{p} question ids must be q1..q{len(qs)} in order")
        if not lim["q"][0] <= len(qs) <= lim["q"][1]:
            errs.append(f"{p} {len(qs)} questions (want {'28–36' if long else '15–17'})")
        kinds = [q["type"] for q in qs]
        for k, lo in lim["mins"]:
            if kinds.count(k) < lo:
                errs.append(f"{p} only {kinds.count(k)} {k} questions")
        for q in qs:
            qp = f"{p} {q['id']} ({q['type']})"
            ans = q["answer_ids"]
            bad = [a for a in ans if a not in when]
            if bad:
                errs.append(f"{qp} answer ids {bad} do not exist"); continue
            if q["type"] == "absent":
                if ans or q["as_of"]:
                    errs.append(f"{qp} must have answer_ids [] and as_of null")
                continue
            if not ans:
                errs.append(f"{qp} has no answer_ids")
            if q["type"] == "past":
                try:
                    at = d(q["as_of"])
                except Exception:  # noqa: BLE001
                    errs.append(f"{qp} needs an as_of date"); continue
                if not (min(when.values()) <= at <= today):
                    errs.append(f"{qp} as_of {at} is outside the statements' range")
                for a in ans:
                    if when[a] > at:
                        errs.append(f"{qp} answer {a} was said after as_of {at}")
                    elif a in replaced_by and when[replaced_by[a]] <= at:
                        errs.append(f"{qp} answer {a} was already replaced by {replaced_by[a]} on as_of {at}")
            else:
                if q["as_of"]:
                    errs.append(f"{qp} as_of must be null")
                for a in ans:
                    if a in replaced_by:
                        errs.append(f"{qp} answer {a} is no longer true (replaced by {replaced_by[a]})")
    return errs


def reconcile(sc: dict) -> int:
    """After a vote, drop answers the voted updates make untrue on the date asked about:
    a current or multi answer that has since been replaced, a past answer said after its
    date or replaced on or before it. Returns how many answer ids were dropped."""
    when = {s["id"]: s["date"] for s in sc["statements"]}
    gone = {old: s["date"] for s in sc["statements"] for old in s["replaces"]}
    dropped = 0
    for q in sc["questions"]:
        ans = q["answer_ids"]
        if q["type"] in ("current", "multi"):
            keep = [a for a in ans if a not in gone]
        elif q["type"] == "past" and q["as_of"]:
            keep = [a for a in ans if when.get(a, "9") <= q["as_of"] and not (a in gone and gone[a] <= q["as_of"])]
        else:
            keep = ans
        dropped += len(ans) - len(keep)
        q["answer_ids"] = keep
    return dropped


def write_job(name: str) -> None:
    job = JOBS[name]
    cwd = HERE / name
    prompt = SPEC + "\n\n## Your assignment\n\n" + job["brief"] + "\n"
    try:
        resumed = resume_from_raw(cwd) if "--resume" in sys.argv else None
        if resumed:
            log(f"{name}: resuming from the saved reply")
            text, conv = resumed
        else:
            log(f"{name}: writing with {job['model']}")
            text, conv = agy(prompt, job["model"], cwd, schema=HERE / "schema_write.json")
        for attempt in range(3):
            data = parse(text)
            (cwd / f"draft{attempt}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            errs = validate(data, job["n"], job["letters"])
            log(f"{name}: draft {attempt}: {len(errs)} problems")
            if not errs:
                (cwd / "final.json").write_text(json.dumps(data["scenarios"], indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
                (cwd / "conversation.txt").write_text(conv, encoding="utf-8")
                log(f"{name}: DONE")
                return
            fix = ("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                   "\n\nFix every one (relabel, or edit the text, dates or questions as needed; keep everything "
                   "else as it is) and reply with the complete corrected JSON {\"scenarios\": [...]} only.")
            text, conv = agy(fix, job["model"], cwd, conversation=conv, schema=HERE / "schema_write.json")
        data = parse(text)
        errs = validate(data, job["n"], job["letters"])
        (cwd / "final.json").write_text(json.dumps(data["scenarios"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"{name}: FINISHED WITH {len(errs)} PROBLEMS LEFT (see problems.txt)")
    except Exception as e:  # noqa: BLE001
        log(f"{name}: FAILED: {e}")


def strip_labels(scs: list[dict]) -> list[dict]:
    return [{"scenario": sc["scenario"], "today": sc["today"],
             "statements": [{"id": s["id"], "date": s["date"], "text": s["text"]} for s in sc["statements"]],
             "questions": [{"id": q["id"], "text": q["text"], "as_of": q["as_of"]} for q in sc["questions"]]}
            for sc in scs]


def audit_job(name: str) -> None:
    job = JOBS[name]
    scs = json.loads((HERE / name / "final.json").read_text(encoding="utf-8"))
    cwd = HERE / f"audit_{name}"
    bare = strip_labels(scs)
    (cwd).mkdir(exist_ok=True)
    (cwd / "unlabelled.json").write_text(json.dumps(bare, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"audit_{name}: labelling with {job['auditor']}")
    try:
        text, conv = agy(AUDIT + "\n" + json.dumps(bare, indent=1, ensure_ascii=False), job["auditor"], cwd)
        labels = parse(text)
        (cwd / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"audit_{name}: DONE")
    except Exception as e:  # noqa: BLE001
        log(f"audit_{name}: FAILED: {e}")


def audit_split(name: str, model: str | None = None, tag: str | None = None) -> None:
    """The audit one scenario per call, in parallel: a whole part at once can think
    past the model's output limit before it writes a label."""
    model = model or JOBS[name]["auditor"]
    scs = json.loads((HERE / name / "final.json").read_text(encoding="utf-8"))
    root = HERE / (f"audit_{name}_{tag}" if tag else f"audit_{name}")
    got: dict[int, dict] = {}
    job = {"auditor": model}

    def one(i: int, sc: dict) -> None:
        cwd = root / f"sc{i + 1}"
        bare = strip_labels([sc])
        for attempt in range(2):
            try:
                text, _ = agy(AUDIT + "\n" + json.dumps(bare, indent=1, ensure_ascii=False), job["auditor"], cwd)
                lab = parse(text)["labels"]
                got[i] = next(x for x in lab if x["scenario"] == sc["scenario"])
                log(f"audit_{name} sc{i + 1}: done")
                return
            except Exception as e:  # noqa: BLE001
                log(f"audit_{name} sc{i + 1}: attempt {attempt} failed: {str(e)[:200]}")

    log(f"audit_{name}: labelling one scenario per call with {job['auditor']}")
    ts = [threading.Thread(target=one, args=(i, sc)) for i, sc in enumerate(scs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    labels = {"labels": [got[i] for i in sorted(got)]}
    (root / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"audit_{name}: DONE ({len(got)}/{len(scs)} scenarios labelled)")


def compare() -> None:
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    report = []
    test = []
    for name in ("test_a", "test_b"):
        scs = json.loads((HERE / name / "final.json").read_text(encoding="utf-8"))
        test += scs
        lab = {x["scenario"]: x for x in json.loads((HERE / f"audit_{name}" / "labels.json").read_text(encoding="utf-8"))["labels"]}
        pa = pb = pboth = qa = qn = 0
        dis = []
        for sc in scs:
            a = lab.get(sc["scenario"])
            if not a:
                dis.append(f"{sc['scenario']}: auditor skipped the scenario"); continue
            w = {(o, s["id"]) for s in sc["statements"] for o in s["replaces"]}
            r = {(o, sid) for sid, olds in a["replaces"].items() for o in olds}
            pa += len(w); pb += len(r); pboth += len(w & r)
            for x in sorted(w - r):
                dis.append(f"{sc['scenario']}: writer only  {x[1]} replaces {x[0]}")
            for x in sorted(r - w):
                dis.append(f"{sc['scenario']}: auditor only {x[1]} replaces {x[0]}")
            for q in sc["questions"]:
                qn += 1
                if set(q["answer_ids"]) == set(a["answers"].get(q["id"], ["<missing>"])):
                    qa += 1
                else:
                    dis.append(f"{sc['scenario']}: {q['id']} ({q['type']}) answers differ")
        report.append(f"{name}: supersession pairs writer {pa}, auditor {pb}, both {pboth}; "
                      f"answer sets agree {qa}/{qn}")
        (out / f"disagreements_{name}.txt").write_text("\n".join(dis) + "\n", encoding="utf-8")
        (out / f"part_{name}.json").write_text(json.dumps(scs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "test.json").write_text(json.dumps(test, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    dev = json.loads((HERE / "dev" / "final.json").read_text(encoding="utf-8"))
    (out / "dev.json").write_text(json.dumps(dev, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    from collections import Counter
    for label, scs in (("test", test), ("dev", dev)):
        qs = [q for sc in scs for q in sc["questions"]]
        report.append(f"{label}: {len(scs)} scenarios, {sum(len(sc['statements']) for sc in scs)} statements, "
                      f"{sum(len(s['replaces']) for sc in scs for s in sc['statements'])} supersession pairs, "
                      f"{len(qs)} questions {dict(Counter(q['type'] for q in qs))}")
    (out / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    phase = sys.argv[1]
    names = [a for a in sys.argv[2:] if not a.startswith("--")] or None
    if phase == "write":
        ts = [threading.Thread(target=write_job, args=(n,)) for n in (names or JOBS)]
    elif phase == "audit":
        ts = [threading.Thread(target=audit_job, args=(n,)) for n in (names or ["test_a", "test_b"])]
    elif phase == "audit-split":
        opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
        ts = [threading.Thread(target=audit_split, args=(n, opts.get("model"), opts.get("tag")))
              for n in (names or ["test_a", "test_b"])]
    elif phase == "compare":
        compare(); sys.exit()
    else:
        sys.exit(f"unknown phase {phase}")
    for t in ts:
        t.start()
    for t in ts:
        t.join()
