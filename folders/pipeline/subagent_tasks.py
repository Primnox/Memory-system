"""Who-is-who work done by Claude Code subagents (Sonnet) instead of agy, when agy has no Claude/GPT-OSS credits.

    python subagent_tasks.py make              list what is missing -> subagent/tasks.json + one prompt file per task
    python subagent_tasks.py check <id> <file> (for a subagent writing a person) the validator's problems, or "OK"
    python subagent_tasks.py take <id>         file a finished task's output where identity.py / duo.py expect it

Same rules as identity.py: an auditor never audits its own family's writing, one audit per family, a writer only
ever sees its spec. Audits are filed as audit_claude-sonnet-5-5.json, writes as final.json + job.json (writer
claude-sonnet-5-5). Blind and open people are written by Sonnet (never a training writer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
import identity as I   # noqa: E402
import duo as D        # noqa: E402
from drive import parse  # noqa: E402

SUB = "claude-sonnet-5-5"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "subagent"
TASKS = OUT / "tasks.json"


def audits_on_disk(home: Path) -> list[str]:
    return [f.stem[len("audit_"):] for f in home.glob("audit_*.json")]


def needs_claude_audit(home: Path) -> bool:
    if not (home / "final.json").exists():
        return False
    writer = json.loads((home / "job.json").read_text(encoding="utf-8"))["writer"]
    done = audits_on_disk(home)
    return not writer.startswith("claude") and not any(m.startswith("claude") for m in done) and len(done) < 2


def make() -> None:
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "out").mkdir(parents=True, exist_ok=True)
    tasks = []
    for name in ("train", "blind"):
        for home in sorted((I.ROOT / name).glob("p*")):
            if needs_claude_audit(home):
                sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
                tasks.append({"id": f"audit-{name}-{home.name}", "kind": "audit", "set": name, "home": str(home),
                              "prompt": I.audit_prompt(sc)})
    for home in sorted(D.ROOT.glob("p*")):
        if needs_claude_audit(home):
            sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
            tasks.append({"id": f"audit-duo-{home.name}", "kind": "audit-duo", "set": "duo", "home": str(home),
                          "prompt": D.audit_prompt(sc)})
    for name in ("blind", "open"):
        jobs = I.people(name)
        for i, job in enumerate(jobs):
            home = I.ROOT / name / f"p{i + 1:02d}"
            if (home / "final.json").exists():
                continue
            job = dict(job, writer=SUB)
            tasks.append({"id": f"write-{name}-p{i + 1:02d}", "kind": "write", "set": name, "i": i, "home": str(home),
                          "job": job, "prompt": I.prompt_for(name, job)})
    for t in tasks:
        p = OUT / "prompts" / f"{t['id']}.md"
        p.write_text(t.pop("prompt"), encoding="utf-8")
        t["prompt_file"], t["out_file"] = str(p), str(OUT / "out" / f"{t['id']}.json")
    TASKS.write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding="utf-8")
    kinds = {}
    for t in tasks:
        kinds[t["kind"] + ":" + t["set"]] = kinds.get(t["kind"] + ":" + t["set"], 0) + 1
    print(len(tasks), "tasks", kinds)


def task(tid: str) -> dict:
    return next(t for t in json.loads(TASKS.read_text(encoding="utf-8")) if t["id"] == tid)


def person(text: str) -> dict:
    got = parse(text)
    return got["scenarios"][0] if "scenarios" in got else got


def check(tid: str, file: str) -> None:
    t = task(tid)
    try:
        errs = I.validate(person(Path(file).read_text(encoding="utf-8")), t["job"])
    except Exception as e:  # noqa: BLE001
        errs = [f"the file is not the JSON asked for ({str(e)[:200]})"]
    print("OK" if not errs else f"{len(errs)} problems:\n" + "\n".join(f"- {e}" for e in errs[:60]))


def take(tid: str) -> None:
    t = task(tid)
    home, text = Path(t["home"]), Path(t["out_file"]).read_text(encoding="utf-8")
    if t["kind"] == "write":
        sc = person(text)
        errs = I.validate(sc, t["job"])
        if errs:
            raise SystemExit(f"{tid}: {len(errs)} problems, not filed: {errs[:3]}")
        home.mkdir(parents=True, exist_ok=True)
        (home / "final.json").write_text(json.dumps([sc], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (home / "job.json").write_text(json.dumps(t["job"], indent=1), encoding="utf-8")
    else:
        got = parse(text)
        sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
        if t["kind"] == "audit":
            lab = got.get("labels") or {}
            miss = [s["id"] for s in sc["statements"] if s["id"] not in lab]
            if len(miss) > 2:
                raise SystemExit(f"{tid}: {len(miss)} statements unlabelled, not filed")
        elif not isinstance(got.get("answers"), dict):
            raise SystemExit(f"{tid}: no answers, not filed")
        (home / f"audit_{SUB}.json").write_text(json.dumps(got, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{tid}: filed")


if __name__ == "__main__":
    {"make": lambda: make(), "check": lambda: check(ARGS[1], ARGS[2]), "take": lambda: take(ARGS[1])}[ARGS[0]]()
