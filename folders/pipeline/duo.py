"""Two-person, third-person people (SPEC_DUO.md) for round 10: the real-chat failure (REALTALK
stays eval-only; nothing from it is shown to writers).

  python duo.py write [lo-hi] | audit [lo-hi] | vote | report
Writers Gemini 3.1 Pro / 3.8 Flash; auditors the other two of {Pro, Flash, Sonnet}; `replaces`
pairs and question answers voted 2 of 3. Output duo/pNN/{final,voted}.json and
../kaggle_identity_ds/train_duo.json (change-detection scenarios for build_train.py).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import drive
from drive import log, parse
import sampler
from gpt_long import used_names
from identity import PRO, FLASH, auditors

HERE = Path(__file__).resolve().parent
ROOT = HERE / "duo"
SPEC = (HERE / "SPEC_DUO.md").read_text(encoding="utf-8")
N, SEED = 16, 505
HOW = ["school friends", "former flatmates", "cousins", "colleagues at their first job", "teammates in a sports club",
       "neighbours", "friends from university", "online friends who met in a game", "siblings-in-law", "old exes, now friends"]


def jobs() -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for i in range(N):
        a, b = sampler.person(rng, 2 * i), sampler.person(rng, 2 * i + 1)
        out.append({"writer": (PRO, FLASH)[i % 2], "who": f"{a['who']}; and {b['who']}", "how": rng.choice(HOW),
                    "today": a["today"], "letters": rng.sample("ABCDEFGHIJKLMNOPRSTVWY", 2)})
    return out


def validate(sc: dict, job: dict) -> list[str]:
    errs, p = [], f"[{sc.get('scenario', '?')}]"
    sp = sc.get("speakers") or []
    if len(sp) != 2:
        return [f"{p} speakers must be the two first names"]
    if sorted(n[:1].upper() for n in sp) != sorted(job["letters"]):
        errs.append(f"{p} the two first names must start with {' and '.join(job['letters'])}")
    try:
        today = date.fromisoformat(sc["today"])
    except Exception:  # noqa: BLE001
        return errs + [f"{p} bad today"]
    st = sc.get("statements") or []
    ids = [s.get("id") for s in st]
    if ids != [f"s{i}" for i in range(1, len(st) + 1)]:
        errs.append(f"{p} statement ids must be s1..s{len(st)} in order")
    if not 48 <= len(st) <= 75:
        errs.append(f"{p} {len(st)} statements (want 50-70)")
    when, about, gone, prev = {}, {}, {}, None
    for i, s in enumerate(st):
        try:
            when[s["id"]] = dt = date.fromisoformat(s["date"])
        except Exception:  # noqa: BLE001
            errs.append(f"{p} {s.get('id')} bad date"); continue
        if prev and dt < prev or dt > today:
            errs.append(f"{p} {s['id']} date out of order or after today")
        prev = dt
        about[s["id"]] = s.get("about")
        for old in s.get("replaces") or []:
            if old not in ids[:i]:
                errs.append(f"{p} {s['id']} replaces {old}, not an earlier fact")
            elif old in gone:
                errs.append(f"{p} {old} replaced twice")
            elif about.get(old) != s.get("about"):
                errs.append(f"{p} {s['id']} (about {s.get('about')}) replaces {old} (about {about.get(old)}): facts about "
                            "different people never replace each other")
            else:
                gone[old] = s["id"]
    if not 10 <= len(gone) <= 24:
        errs.append(f"{p} {len(gone)} updates (want 12-20)")
    qs = sc.get("questions") or []
    if [q.get("id") for q in qs] != [f"q{i}" for i in range(1, len(qs) + 1)]:
        errs.append(f"{p} question ids must be q1..q{len(qs)}")
    if not 18 <= len(qs) <= 28:
        errs.append(f"{p} {len(qs)} questions (want 20-26)")
    k = Counter(q.get("type") for q in qs)
    for t, lo in (("current", 6), ("past", 3), ("history", 6), ("absent", 3)):
        if k[t] < lo:
            errs.append(f"{p} only {k[t]} {t} questions")
    for q in qs:
        qp, ans = f"{p} {q.get('id')} ({q.get('type')})", q.get("answer_ids") or []
        if any(a not in when for a in ans):
            errs.append(f"{qp} answer ids must be fact ids"); continue
        if q.get("type") == "absent":
            if ans or q.get("as_of"):
                errs.append(f"{qp} must have [] and null")
            continue
        if not ans:
            errs.append(f"{qp} has no answers")
        if q.get("type") == "past":
            try:
                at = date.fromisoformat(q["as_of"])
            except Exception:  # noqa: BLE001
                errs.append(f"{qp} needs as_of"); continue
            errs += [f"{qp} answer {a} not true on {at}" for a in ans
                     if when[a] > at or (a in gone and when[gone[a]] <= at)]
        elif q.get("type") == "current":
            errs += [f"{qp} answer {a} no longer true (replaced by {gone[a]})" for a in ans if a in gone]
    return errs


def write(i: int) -> None:
    job = jobs()[i]
    cwd = ROOT / f"p{i + 1:02d}"
    if (cwd / "final.json").exists():
        return
    names, cities = used_names()
    prompt = (SPEC + "\n## Do not reuse\n\nFirst names: " + ", ".join(names) + ". Cities: " + ", ".join(cities) +
              f".\n\n## Your assignment\n\nTwo friends ({job['how']}): {job['who']}. Their first names start with "
              f"{job['letters'][0]} and {job['letters'][1]}. `today` = {job['today']}. Write this pair from scratch. "
              "ONE pair, JSON only.\n")
    log(f"duo p{i + 1:02d}: writing with {job['writer']}")
    try:
        text, conv = drive.agy(prompt, job["writer"], cwd)
        for attempt in range(4):
            try:
                sc = parse(text)["scenarios"][0]
                errs = validate(sc, job)
            except Exception as e:  # noqa: BLE001
                sc, errs = None, [f"the reply is not the JSON asked for ({str(e)[:120]})"]
            log(f"duo p{i + 1:02d}: draft {attempt}: {len(errs)} problems")
            if not errs:
                (cwd / "final.json").write_text(json.dumps([sc], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                (cwd / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
                return
            if attempt == 3:
                break
            text, conv = drive.agy("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                                   "\n\nFix every one and reply with the complete corrected JSON {\"scenarios\": [ <the one "
                                   "pair> ]} only.", job["writer"], cwd, conversation=conv)
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"duo p{i + 1:02d}: GAVE UP with {len(errs)} problems")
    except Exception as e:  # noqa: BLE001
        log(f"duo p{i + 1:02d}: FAILED: {str(e)[:200]}")


def audit(i: int) -> None:
    home = ROOT / f"p{i + 1:02d}"
    if not (home / "final.json").exists():
        return
    sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
    job = json.loads((home / "job.json").read_text(encoding="utf-8"))
    rules = SPEC.split("### Statements", 1)[1].split("## Before you answer", 1)[0]
    bare = {"speakers": sc["speakers"], "today": sc["today"],
            "statements": [{"id": s["id"], "date": s["date"], "text": s["text"], "about": s.get("about")} for s in sc["statements"]],
            "questions": [{k: q[k] for k in ("id", "text", "type", "as_of")} for q in sc["questions"]]}
    prompt = ("# Labelling a two-person memory set\n\nBelow are the facts and questions WITHOUT labels. Using the rules "
              "the writer followed, give for every fact the ids of earlier facts it makes no longer true right now, "
              "and for every question the fact ids that answer it.\n\n## Rules\n" + rules +
              "\n## Reply\n\n{\"replaces\": {\"s7\": [\"s2\"], ...only facts that replace something...}, "
              "\"answers\": {\"q1\": [\"s4\"], ...every question...}}\n\nJSON only, no tools, no code.\n\n" +
              json.dumps(bare, indent=1, ensure_ascii=False) + "\n")
    for model in auditors(job["writer"], home):
        out = home / f"audit_{model}.json"
        if out.exists():
            continue
        for attempt in range(3):
            try:
                text, _ = drive.agy(prompt, model, home / f"audit_{model}")
                got = parse(text)
                if not isinstance(got.get("answers"), dict):
                    raise ValueError("no answers")
                out.write_text(json.dumps(got, indent=1, ensure_ascii=False), encoding="utf-8")
                log(f"duo p{i + 1:02d}: audit by {model} done")
                break
            except Exception as e:  # noqa: BLE001
                log(f"duo p{i + 1:02d}: audit by {model} attempt {attempt} failed: {str(e)[:200]}")


def vote() -> None:
    scen, agree = [], Counter()
    for home in sorted(ROOT.glob("p*")):
        if not (home / "final.json").exists():
            continue
        sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
        job = json.loads((home / "job.json").read_text(encoding="utf-8"))
        auds = [json.loads((home / f"audit_{m}.json").read_text(encoding="utf-8"))
                for m in auditors(job["writer"], home) if (home / f"audit_{m}.json").exists()]
        if not auds:          # one audit: a label is kept only when writer and auditor agree (still 2 agreeing)
            continue
        w = {(o, s["id"]) for s in sc["statements"] for o in s.get("replaces") or []}
        a = [{(o, n) for n, olds in (x.get("replaces") or {}).items() for o in olds or []} for x in auds]
        cnt = Counter(pr for s_ in [w] + a for pr in s_)
        keep = {pr for pr, c in cnt.items() if c >= 2}
        agree["writer pairs"] += len(w); agree["kept"] += len(keep); agree["writer kept"] += len(w & keep)
        agree["any"] += len(cnt)
        stmts = [{"id": s["id"], "date": s["date"], "text": s["text"],
                  "replaces": sorted(o for (o, n) in keep if n == s["id"])} for s in sc["statements"]]
        skip = sorted([o, n] for (o, n), c in cnt.items() if c == 1)
        qs = []
        for q in sc["questions"]:
            sets = [set(q["answer_ids"])] + [set((x.get("answers") or {}).get(q["id"]) or []) for x in auds]
            c = Counter(v for s_ in sets for v in s_)
            qs.append({**q, "answer_ids": sorted(v for v, k in c.items() if k >= 2)})
            agree["q same"] += set(qs[-1]["answer_ids"]) == sets[0]; agree["q"] += 1
        v = {"scenario": sc["scenario"], "speakers": sc["speakers"], "today": sc["today"], "statements": stmts,
             "questions": qs, "skip": skip}
        (home / "voted.json").write_text(json.dumps([v], indent=1, ensure_ascii=False), encoding="utf-8")
        scen.append(v)
    out = HERE.parent / "kaggle_identity_ds"
    out.mkdir(exist_ok=True)
    (out / "train_duo.json").write_text(json.dumps(scen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"duo: {len(scen)} pairs voted; replaces pairs: writer {agree['writer pairs']}, kept 2-of-3 {agree['kept']} "
          f"(writer's kept {agree['writer kept']}), named by anyone {agree['any']}; question answers same as writer "
          f"{agree['q same']}/{agree['q']}")


if __name__ == "__main__":
    phase = sys.argv[1]
    lo, hi = map(int, (sys.argv[2] if len(sys.argv) > 2 else f"1-{N}").split("-"))
    ROOT.mkdir(exist_ok=True)
    if phase in ("write", "audit"):
        fn = write if phase == "write" else audit
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fn, range(lo - 1, hi)))
    if phase == "vote":
        vote()
    for home in sorted(ROOT.glob("p*")):
        if (home / "final.json").exists():
            sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
            print(home.name, sc["scenario"], len(sc["statements"]), "facts,",
                  sum(len(s.get("replaces") or []) for s in sc["statements"]), "updates,", len(sc["questions"]), "questions")
