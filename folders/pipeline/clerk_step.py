"""One filing step at a time, for a clerk that is not driven through agy (a Claude subagent).

    python clerk_step.py spec                          the filing rules
    python clerk_step.py next   <job> <set> <person>   folders and slots so far + the NEXT batch only
    python clerk_step.py submit <job> <set> <person> <reply.json>
                                                       check the reply (same checks as folders.py),
                                                       file it, and show the next batch
<job> is "batch" (25 facts per step, folders/) or "single" (one fact per step, folders_b1/).
The clerk only ever sees what a filing call through agy would see: the facts so far are never
shown again and later facts are never shown. Each accepted reply is kept as
<person>/bNNN/clerk_reply.json (teacher data, like agy's raw_*.out).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import folders

HERE = Path(__file__).resolve().parent
JOBS = {"batch": (HERE / "folders", 25), "single": (HERE / "folders_b1", 1)}


def where(job: str, set_name: str, person: str) -> tuple[Path, dict, int]:
    out, step = JOBS[job]
    sc = next(s for s in folders.load(set_name) if s["scenario"] == person)
    return out / set_name / re.sub(r"[^\w\-]", "_", person), sc, step


def load_state(home: Path, set_name: str, sc: dict) -> dict:
    f = home / "state.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else folders.first_state(set_name, sc)


def next_batch(job: str, set_name: str, person: str) -> str:
    home, sc, step = where(job, set_name, person)
    state = load_state(home, set_name, sc)
    facts = sorted(sc["statements"], key=lambda f: f["date"])
    for k in range(0, len(facts), step):
        batch = facts[k:k + step]
        if any(f["id"] not in state["facts"] for f in batch):
            # a batch is filed whole; a half-filed one (from an interrupted run) is redone whole
            view = {"folders_so_far": [{"id": i, **v} for i, v in state["folders"].items()],
                    "slots_so_far": state["slots"],
                    "new_facts": [{"id": f["id"], "date": f["date"], "text": f["text"]} for f in batch],
                    "step": k // step, "steps": (len(facts) + step - 1) // step}
            return json.dumps(view, ensure_ascii=False, indent=1)
    return "DONE"


def submit(job: str, set_name: str, person: str, reply: str) -> str:
    home, sc, step = where(job, set_name, person)
    state = load_state(home, set_name, sc)
    facts = sorted(sc["statements"], key=lambda f: f["date"])
    for k in range(0, len(facts), step):
        batch = facts[k:k + step]
        if any(f["id"] not in state["facts"] for f in batch):
            break
    else:
        return "DONE"
    got = json.loads(Path(reply).read_text(encoding="utf-8"))
    errs = folders.check(got, batch, state)
    if errs:
        return "REJECTED, fix and submit again: " + "; ".join(errs[:8])
    folders.merge(state, got, batch)
    cwd = home / f"b{k // step:03d}"
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "clerk_reply.json").write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
    home.mkdir(parents=True, exist_ok=True)
    state.setdefault("clerk", "claude-sonnet (subagent)")
    (home / "state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    return "ACCEPTED\n" + next_batch(job, set_name, person)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["spec"]:
        print(folders.SPEC.split("## What you return")[0] + "## What you return" +
              folders.SPEC.split("## What you return")[1].split("Work only from")[0])
    elif a[:1] == ["next"] and len(a) == 4:
        print(next_batch(*a[1:]))
    elif a[:1] == ["submit"] and len(a) == 5:
        print(submit(*a[1:]))
    else:
        raise SystemExit(__doc__)
