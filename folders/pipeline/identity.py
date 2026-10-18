"""Who-is-who people (SPEC_IDENTITY.md), written and audited through agy.

  python identity.py write <set> [lo-hi]    writers -> identity/<set>/pNN/final.json
  python identity.py audit <set> [lo-hi]    two auditors label the bare statements -> pNN/audit_<model>.json
  python identity.py vote <set>             2-of-3 vote (writer + 2 auditors) -> pNN/voted.json, report
  python identity.py report <set>

Sets (fixed here before any data exists):
  pilot   4 people, Gemini 3.1 Pro / 3.8 Flash writers, cases from TRAINING sections only
  train   training people, same writers and rules
  blind   Claude Sonnet 4.6 writer (a family never used as a training writer), cases from ALL
          model-layer sections incl. the held-out ones, more interacting traps per timeline
  open    GPT-OSS 120B writer with NO case list (the "unknown unknowns" set)
Auditors: the two models of {Pro, Flash, Sonnet} that did not write the person.
Same discipline as chat.py: an empty folder per call, fixes sent back to the same writer.
"""
from __future__ import annotations

import json
import os
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
from long import strayed

HERE = Path(__file__).resolve().parent
ROOT = HERE / "identity"
SPEC = (HERE / "SPEC_IDENTITY.md").read_text(encoding="utf-8")
MAP = json.loads((HERE / "identity_map.json").read_text(encoding="utf-8"))
PRO, FLASH, SONNET, OSS = "gemini-3.1-pro-high", "gemini-3.8-flash-high", "claude-sonnet-4-6", "gpt-oss-120b-medium"
# 2026-10-11: Claude and GPT-OSS through agy ran out of credits; only Gemini works. The blind and open sets
# are written by Gemini models that never write training data (same family: a caveat), audited by Pro + Flash.
FLASH37, FLASH36 = "gemini-3.7-flash-high", "gemini-3.6-flash-high"
# 2026-10-16: agy 1.3.6 dropped Gemini 3.1 Pro, 3.7 Flash and 3.6 Flash; Claude and GPT-OSS have credits again.
# People not yet written go to the nearest writer that still exists: training -> 3.8 Flash (the only training
# writer left), blind -> Sonnet and open -> GPT-OSS (the original design). Written people keep their writer.
GONE = {PRO: FLASH, FLASH37: SONNET, FLASH36: OSS}
KINDS, ABOUT, TYPES = MAP["kinds"], MAP["about"], MAP["types"]
REPLACERS = {"fact", "plan", "correction"}
TAIL = ("Write this person from scratch, not from a template, and do not generate it with code. "
        "ONE person, JSON only, nothing else.")
COMBOS = {
    "A": "The two Rohans: a cousin Rohan in Pune; a Rohan from the lab placed at Microsoft; \"Ro called\"; \"my cousin "
         "is visiting\"; later \"Ro was my lab partner, not my cousin\".",
    "B": "The changing boss: Priya is my manager -> she moves department -> Kiran is my manager -> \"my old manager lives "
         "in Chennai\" -> \"she joined us again as a colleague\".",
    "C": "The phone problem: broken screen -> replaced the screen -> replaced the phone -> old one in the drawer -> given "
         "to my sister, who sold it.",
    "D": "The misleading quote: Rohan told Kiran \"I moved to Delhi\"; Kiran: \"he is my roommate\"; \"my brother lives in "
         "Hyderabad\"; \"that's my cousin, not my brother\"; \"he never moved, it was hypothetical\".",
    "E": "Merge, then split: gym Rohan; neighbour Rohan; good reason to think they are one man; \"they are different "
         "people\"; \"the gym one is actually Rohit\".",
    "F": "The group that changes: a team of four; two join; one leaves; the team submits; \"we should restart\".",
    "G": "Two same-named people; a nickname that belongs to the other one; a quote that pins a job change on the wrong "
         "person; an apparent identity that a later message contradicts; a question about one person's former manager.",
}
SETS = {
    "pilot": dict(n=4, seed=101, writers=[PRO, FLASH], pool="train", n_cases=14, combos=1),
    "train": dict(n=40, seed=202, writers=[PRO, FLASH], pool="train", n_cases=14, combos=1),
    "blind": dict(n=18, seed=303, writers=[FLASH37], pool="all", n_cases=18, combos=2),
    "open": dict(n=6, seed=404, writers=[FLASH36], pool=None, n_cases=0, combos=0),
}


def model_cases(pool: str) -> list[int]:
    ok = ("train",) if pool == "train" else ("train", "held_out")
    return sorted(int(k) for k, c in MAP["cases"].items() if c["layer"] == "model" and c["split"] in ok)


def people(name: str) -> list[dict]:
    cfg = SETS[name]
    rng = random.Random(cfg["seed"])
    pool = model_cases(cfg["pool"]) if cfg["pool"] else []
    out = []
    for i in range(cfg["n"]):
        p = sampler.person(rng, i)
        cases = sorted(rng.sample(pool, cfg["n_cases"])) if pool else []
        combos = rng.sample(sorted(COMBOS), cfg["combos"]) if cfg["combos"] else []
        if name == "blind" and "G" not in combos:
            combos[-1] = "G"
        w = cfg["writers"][i % len(cfg["writers"])]
        w = GONE.get(w, w)
        # IDENTITY_WRITE_AS="gemini-3.8-flash-high=gemini-3.8-flash-medium": retry people whose write keeps running
        # out of output tokens at high effort (same model family, so the auditor rules are unchanged)
        w = dict(kv.split("=", 1) for kv in os.environ.get("IDENTITY_WRITE_AS", "").split(",") if "=" in kv).get(w, w)
        out.append({"writer": w, "letter": rng.choice("ABCDEFGHIJKLMNOPRSTVWY"),
                    "who": p["who"], "style": p["style"], "today": p["today"], "cases": cases, "combos": combos})
    return out


def d(s: str) -> date:
    return date.fromisoformat(s)


def validate(sc: dict, job: dict) -> list[str]:
    errs: list[str] = []
    name = sc.get("scenario", "?")
    p = f"[{name}]"
    if not re.fullmatch(r"[a-z]+(-[a-z]+){2,}", name):
        errs.append(f"{p} slug must be lowercase firstname-city-role")
    if job["letter"] and not name[:1].upper() == job["letter"]:
        errs.append(f"{p} the first name must start with {job['letter']}")
    try:
        today = d(sc["today"])
    except Exception:  # noqa: BLE001
        return errs + [f"{p} bad today"]
    st = sc.get("statements") or []
    if [s.get("id") for s in st] != [f"s{i}" for i in range(1, len(st) + 1)]:
        errs.append(f"{p} statement ids must be s1..s{len(st)} in order")
    if not 33 <= len(st) <= 55:
        errs.append(f"{p} {len(st)} statements (want 35-50)")
    traps = {t.get("id") for t in sc.get("new_traps") or []}
    if job["cases"] and len(traps) < 3:
        errs.append(f"{p} only {len(traps)} new_traps (want at least 3)")
    live, gone_folders, filed = {}, set(), {}
    prev, replaced_by, used_cases, n_ask, when = None, {}, set(), 0, {}
    for i, s in enumerate(st):
        sp = f"{p} {s.get('id')}"
        try:
            when[s["id"]] = dt = d(s["date"])
        except Exception:  # noqa: BLE001
            errs.append(f"{sp} bad date"); continue
        if prev and dt < prev:
            errs.append(f"{sp} date earlier than the previous statement's")
        prev = dt
        if dt > today:
            errs.append(f"{sp} dated after today")
        if not 2 <= len(str(s.get("text", "")).split()) <= 60:
            errs.append(f"{sp} text should be 4-40 words")
        if s.get("kind") not in KINDS:
            errs.append(f"{sp} kind {s.get('kind')!r} is not one of {KINDS}")
        if s.get("about") not in ABOUT:
            errs.append(f"{sp} about {s.get('about')!r} is not one of {ABOUT}")
        here = {}
        for nf in s.get("new_folders") or []:
            fid = nf.get("id") if isinstance(nf, dict) else None
            if not fid or fid in live or fid in gone_folders or fid in ("speaker", "ask", "new"):
                errs.append(f"{sp} new folder id {fid!r} missing, reserved or already used"); continue
            if nf.get("type") not in TYPES:
                errs.append(f"{sp} new folder {fid} type {nf.get('type')!r} is not one of {TYPES}")
            here[fid] = nf
        sp_split = s.get("split")
        if sp_split:
            nf = sp_split.get("new_folder") or {}
            if sp_split.get("folder") not in live:
                errs.append(f"{sp} split of {sp_split.get('folder')!r}, which is not an existing folder")
            if not nf.get("id") or nf.get("id") in live or nf.get("id") in here:
                errs.append(f"{sp} split needs a new_folder with a fresh id")
            else:
                here[nf["id"]] = nf
            for m in sp_split.get("move") or []:
                if m not in when or m == s["id"]:
                    errs.append(f"{sp} split moves {m}, which is not an earlier statement")
        known = set(live) | set(here)
        f = s.get("folder")
        if f == "ask":
            n_ask += 1
            ab = s.get("ask_between") or []
            if len(ab) < 2 or any(x not in known and x != "new" for x in ab):
                errs.append(f"{sp} ask_between must list 2+ existing folder ids (or \"new\")")
        else:
            if s.get("ask_between"):
                errs.append(f"{sp} ask_between only goes with folder \"ask\"")
            if f != "speaker" and f not in known:
                errs.append(f"{sp} folder {f!r} does not exist at this point (create it in new_folders)")
        for u in s.get("updates") or []:
            if u.get("folder") not in known:
                errs.append(f"{sp} update of unknown folder {u.get('folder')!r}")
        for m in s.get("mentions") or []:
            if m.get("folder") not in known and m.get("folder") not in ("ask", "speaker"):
                errs.append(f"{sp} mention {m.get('text')!r} points at unknown folder {m.get('folder')!r}")
        for old in s.get("replaces") or []:
            if old not in [x["id"] for x in st[:i]]:
                errs.append(f"{sp} replaces {old}, which is not an earlier statement")
            elif old in replaced_by:
                errs.append(f"{sp} {old} is replaced twice ({replaced_by[old]} and {s['id']})")
            else:
                replaced_by[old] = s["id"]
        if s.get("replaces") and s.get("kind") not in REPLACERS:
            errs.append(f"{sp} a {s.get('kind')} never replaces anything (only fact, plan, correction)")
        mg = s.get("merge")
        if mg:
            if not (isinstance(mg, list) and len(mg) == 2 and mg[0] != mg[1] and all(x in live for x in mg)):
                errs.append(f"{sp} merge must be [keep_id, gone_id] of two existing folders")
            else:
                gone_folders.add(mg[1]); live.pop(mg[1], None)
        for c in s.get("cases") or []:
            if isinstance(c, int):
                if job["cases"] and c not in job["cases"]:
                    errs.append(f"{sp} case {c} is not in your case list")
                used_cases.add(c)
            elif not (isinstance(c, str) and c in traps):
                errs.append(f"{sp} case {c!r} is neither a listed case number nor one of your new_traps ids")
            else:
                used_cases.add(c)
        live.update(here)
        filed[s["id"]] = f
    n_merge = sum(bool(s.get("merge")) for s in st)
    n_split = sum(bool(s.get("split")) for s in st)
    if job["cases"] and (n_merge < 1 or n_split < 1):      # the blind v1 writer skipped both; every listed set needs them
        errs.append(f"{p} {n_merge} merges and {n_split} splits: include at least one of each (a duplicate folder "
                    "discovered later; a folder that turns out to hold two different people or things)")
    if not 7 <= len(replaced_by) <= 16:
        errs.append(f"{p} {len(replaced_by)} statements are replaced (want 8-14 real changes)")
    missing = [c for c in job["cases"] if c not in used_cases]
    if missing:
        errs.append(f"{p} cases never used: {missing}")
    unused = [t for t in traps if t not in used_cases]
    if unused:
        errs.append(f"{p} new traps never used in a statement: {unused}")
    if st and n_ask > 0.2 * len(st):
        errs.append(f"{p} {n_ask} statements are \"ask\"; keep it to about 1 in 10")
    qs = sc.get("questions") or []
    if [q.get("id") for q in qs] != [f"q{i}" for i in range(1, len(qs) + 1)]:
        errs.append(f"{p} question ids must be q1..q{len(qs)} in order")
    if not 9 <= len(qs) <= 15:
        errs.append(f"{p} {len(qs)} questions (want 10-14)")
    kinds = Counter(q.get("type") for q in qs)
    for k, lo in (("current", 3), ("past", 2), ("who", 3), ("absent", 2)):
        if kinds[k] < lo:
            errs.append(f"{p} only {kinds[k]} {k} questions")
    for q in qs:
        qp = f"{p} {q.get('id')} ({q.get('type')})"
        ans = q.get("answer_ids") or []
        if any(a not in when for a in ans):
            errs.append(f"{qp} answer ids must be statement ids"); continue
        if q.get("type") == "absent":
            if ans or q.get("as_of"):
                errs.append(f"{qp} must have answer_ids [] and as_of null")
            continue
        if not ans:
            errs.append(f"{qp} has no answer_ids")
        if q.get("type") == "past":
            try:
                at = d(q["as_of"])
            except Exception:  # noqa: BLE001
                errs.append(f"{qp} needs an as_of date"); continue
            for a in ans:
                if when[a] > at:
                    errs.append(f"{qp} answer {a} was said after as_of {at}")
                elif a in replaced_by and when[replaced_by[a]] <= at:
                    errs.append(f"{qp} answer {a} was already replaced by {replaced_by[a]} on {at}")
        elif q.get("type") in ("current", "who"):
            if q.get("as_of"):
                errs.append(f"{qp} as_of must be null")
            errs += [f"{qp} answer {a} is no longer true (replaced by {replaced_by[a]})" for a in ans if a in replaced_by]
    return errs


def prompt_for(name: str, job: dict) -> str:
    names, cities = used_names()
    names = sorted(set(names) | {json.loads(f.read_text(encoding="utf-8"))[0]["speaker"]
                                 for f in ROOT.glob("*/p*/final.json")})
    a = (f"Person: {job['who']}. Their first name starts with {job['letter']}. `today` = {job['today']}. "
         f"Writing style: {', '.join(job['style'])}.")
    if job["cases"]:
        lines = "\n".join(f"- {c} {MAP['cases'][str(c)]['name']}" for c in job["cases"])
        a += ("\n\nYour case list (from a catalogue of identity failures; use EVERY one at least once and give its "
              f"number in `cases`; each is a NAME only, so decide yourself what a real instance looks like):\n{lines}")
    if job["combos"]:
        a += "\n\nAlso build the timeline around " + ("these combinations" if len(job["combos"]) > 1 else "this combination") + \
             " (change every name, place and detail to fit this person's life):\n" + \
             "\n".join(f"- {COMBOS[c]}" for c in job["combos"])
    if name == "blind":
        a += "\n\nMake the traps interact more than usual: at least 4 places where one trap depends on another."
    if not job["cases"]:
        a += ("\n\nThere is no case list for you: make up the identity traps yourself, as many different kinds as a "
              "real life would produce; list each kind in `new_traps` (at least 6).")
    return (SPEC + "\n## Do not reuse\n\nFirst names: " + ", ".join(names) + ". Cities: " + ", ".join(cities)
            + ".\n\n## Your assignment\n\n" + a + "\n\n" + TAIL + "\n")


def write(name: str, i: int) -> bool:
    job = people(name)[i]
    cwd = ROOT / name / f"p{i + 1:02d}"
    if (cwd / "final.json").exists():
        return True
    model = job["writer"]
    log(f"identity {name} p{i + 1:02d}: writing with {model}")
    try:
        text, conv = drive.agy(prompt_for(name, job), model, cwd)
        for attempt in range(4):
            try:
                sc = parse(text)["scenarios"][0]
                errs = validate(sc, job)
            except Exception as e:  # noqa: BLE001
                sc, errs = None, [f"the reply is not the JSON asked for ({str(e)[:120]})"]
            log(f"identity {name} p{i + 1:02d}: draft {attempt}: {len(errs)} problems")
            (cwd / f"draft{attempt}.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False), encoding="utf-8")
            if not errs:
                (cwd / "final.json").write_text(json.dumps([sc], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                (cwd / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
                return True
            if attempt == 3:
                break
            fix = ("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                   "\n\nFix every one (relabel, or edit statements or questions as needed; keep everything else) and "
                   "reply with the complete corrected JSON {\"scenarios\": [ <the one person> ]} only.")
            text, conv = drive.agy(fix, model, cwd, conversation=conv)
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"identity {name} p{i + 1:02d}: GAVE UP with {len(errs)} problems")
    except Exception as e:  # noqa: BLE001
        log(f"identity {name} p{i + 1:02d}: FAILED: {str(e)[:200]}")
    return False


# ---------------------------------------------------------------- audit (2 of 3)

AUDIT_FIELDS = ("kind, about, folder, new_folders (id, type, name, relation), ask_between, replaces, merge, split")


def audit_prompt(sc: dict) -> str:
    rules = SPEC.split("### Statements", 1)[1].split("### New traps", 1)[0]
    qrules = SPEC.split("### Questions", 1)[1].split("## Before you answer", 1)[0]
    bare = {"speaker": sc["speaker"], "today": sc["today"],
            "statements": [{"id": s["id"], "date": s["date"], "text": s["text"]} for s in sc["statements"]],
            "questions": [{"id": q["id"], "text": q["text"], "type": q["type"], "as_of": q["as_of"]} for q in sc["questions"]]}
    return ("# Labelling a who-is-who timeline\n\nBelow is one user's timeline of statements to a personal assistant, "
            "oldest first, WITHOUT labels. Label every statement exactly as a careful human note-taker would at that "
            "moment, using only what was said up to and including that statement — never anything said later.\n\n"
            "## The label rules (the same ones the writer followed)\n" + rules +
            f"\nLabel ONLY these fields for each statement: {AUDIT_FIELDS}. Skip mentions, updates and cases.\n\n"
            "## Questions\n" + qrules + "\nFor each question give the statement ids that answer it on `today` "
            "(or on its as_of date); [] for absent ones.\n\n"
            "## Reply\n\n{\"labels\": {\"s1\": {\"kind\": ..., \"about\": ..., \"folder\": ..., \"new_folders\": [...], "
            "\"ask_between\": null, \"replaces\": [], \"merge\": null, \"split\": null}, ...}, "
            "\"answers\": {\"q1\": [\"s4\"], ...}}\n\nEvery statement and question must be labelled. JSON only, no "
            "tools, no code.\n\n## The timeline\n\n" + json.dumps(bare, indent=1, ensure_ascii=False) + "\n")


def auditors(writer: str, home: Path | None = None) -> list[str]:
    """Two auditors from models other than the writer. Audits already on disk count first (the pilot's and some
    duo pairs' were done by Gemini 3.7 Flash / 3.1 Pro, which agy no longer offers); the rest come from Sonnet,
    3.8 Flash and GPT-OSS, so most people are now audited by two other families."""
    done = sorted(f.stem[len("audit_"):] for f in home.glob("audit_*.json")) if home else []
    # IDENTITY_POOL (comma list, in order) narrows or widens the fresh auditors, e.g. when agy's Claude credits
    # are out: "gemini-3.8-flash-high,omni:openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
    pool = [m for m in os.environ.get("IDENTITY_POOL", "").split(",") if m] or [SONNET, FLASH, OSS]
    # one audit per model family: 3.8 Flash at medium effort (a retry when high runs out of output tokens)
    # is the same auditor as 3.8 Flash at high, and never audits what 3.8 Flash wrote
    fam = lambda m: re.sub(r"-(high|medium|low)$", "", m)
    taken = {fam(writer)} | {fam(m) for m in done}
    fresh = []
    for m in pool:
        if fam(m) not in taken:
            fresh.append(m)
            taken.add(fam(m))
    return (done + fresh)[:2]


def audit(name: str, i: int) -> None:
    home = ROOT / name / f"p{i + 1:02d}"
    if not (home / "final.json").exists():
        return
    sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
    job = json.loads((home / "job.json").read_text(encoding="utf-8"))
    for model in auditors(job["writer"], home):
        out = home / f"audit_{model}.json"
        if out.exists():
            continue
        cwd = home / f"audit_{model}"
        for attempt in range(3):
            try:
                text, _ = drive.agy(audit_prompt(sc), model, cwd)
                got = parse(text)
                lab = got.get("labels") or {}
                miss = [s["id"] for s in sc["statements"] if s["id"] not in lab]
                if len(miss) > 2:
                    raise ValueError(f"{len(miss)} statements unlabelled")
                out.write_text(json.dumps(got, indent=1, ensure_ascii=False), encoding="utf-8")
                log(f"identity {name} p{i + 1:02d}: audit by {model} done")
                break
            except Exception as e:  # noqa: BLE001
                log(f"identity {name} p{i + 1:02d}: audit by {model} attempt {attempt} failed: {str(e)[:200]}")


# ---------------------------------------------------------------- vote

def canon_folders(labels: list[tuple[str, dict]]) -> dict[str, str]:
    """Each annotator's folder ids -> the id of the statement that created the folder ('@s5'),
    following merges to the kept folder, so annotators with different ids can be compared."""
    born, alias = {}, {}
    for sid, lab in labels:
        for nf in lab.get("new_folders") or []:
            if isinstance(nf, dict) and nf.get("id"):
                born.setdefault(nf["id"], f"@{sid}")
        sp = lab.get("split") or {}
        nf = sp.get("new_folder") if isinstance(sp, dict) else None
        if isinstance(nf, dict) and nf.get("id"):
            born.setdefault(nf["id"], f"@{sid}")
        mg = lab.get("merge")
        if isinstance(mg, list) and len(mg) == 2:
            alias[mg[1]] = mg[0]
    def root(f):
        seen = set()
        while f in alias and f not in seen:
            seen.add(f); f = alias[f]
        return f
    return {f: born.get(root(f), born[f]) for f in born}


def filed_sets(labels: list[tuple[str, dict]], canon: dict) -> dict[str, set]:
    """canonical folder -> the statements filed in it (including the one that created it)."""
    out: dict[str, set] = {}
    for sid, lab in labels:
        f = lab.get("folder")
        if f in canon:
            out.setdefault(canon[f], set()).add(sid)
    return out


def norm(sid: str, lab: dict, canon: dict) -> dict:
    f = lab.get("folder")
    new_here = {nf.get("id") for nf in lab.get("new_folders") or [] if isinstance(nf, dict)}
    if f in ("speaker", "ask"):
        folder = f
    elif f in new_here:
        nf = next(x for x in lab["new_folders"] if isinstance(x, dict) and x.get("id") == f)
        folder = f"new:{nf.get('type')}"
    else:
        folder = canon.get(f, f"?{f}")
    mg = lab.get("merge")
    merge = tuple(sorted(canon.get(x, x) for x in mg)) if isinstance(mg, list) and len(mg) == 2 else None
    sp = lab.get("split")
    split = canon.get(sp.get("folder"), "?") if isinstance(sp, dict) and sp.get("folder") else None
    return {"kind": lab.get("kind"), "about": lab.get("about"), "folder": folder,
            "replaces": set(lab.get("replaces") or []), "merge": merge, "split": split}


def vote(name: str) -> None:
    rows, agree = [], Counter()
    for home in sorted((ROOT / name).glob("p*")):
        if not (home / "final.json").exists():
            continue
        sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
        job = json.loads((home / "job.json").read_text(encoding="utf-8"))
        auds = [json.loads((home / f"audit_{m}.json").read_text(encoding="utf-8"))
                for m in auditors(job["writer"], home) if (home / f"audit_{m}.json").exists()]
        provisional = "--provisional" in sys.argv and not auds
        if not auds and not provisional:
            log(f"identity vote {name}/{home.name}: no audits, skipped"); continue
        sids = [s["id"] for s in sc["statements"]]
        anns = [[(s["id"], s) for s in sc["statements"]]] + [[(sid, a["labels"].get(sid) or {}) for sid in sids] for a in auds]
        if provisional:       # evaluation only, until audited: the writer's labels alone, marked as such
            anns = anns * 2
        while len(anns) < 3:  # one audit: a decision is kept only when writer and auditor agree
            anns.append([(sid, {}) for sid in sids])
        normed = []
        w_canon = canon_folders(anns[0])
        w_sets = filed_sets(anns[0], w_canon)
        for k, ann in enumerate(anns):
            canon = canon_folders(ann)
            if k:   # an auditor's folder = the writer's folder holding most of the same statements (Jaccard >= 1/3)
                for f, c in list(canon.items()):
                    mine = filed_sets(ann, canon).get(c, set())
                    best = max(w_sets.items(), key=lambda kv: len(kv[1] & mine) / max(len(kv[1] | mine), 1), default=None)
                    if best and mine and len(best[1] & mine) / len(best[1] | mine) >= 1 / 3:
                        canon[f] = best[0]
            normed.append({sid: norm(sid, lab, canon) for sid, lab in ann})
        voted = {}
        for sid in sids:
            v = {}
            for k in ("kind", "about", "folder", "merge", "split"):
                vals = [n[sid][k] for n in normed]
                top, c = Counter(map(json.dumps, vals)).most_common(1)[0]
                v[k] = {"value": json.loads(top), "agree": c} if c >= 2 else None
                agree[(k, "writer=a1")] += vals[0] == vals[1]
                agree[(k, "writer=a2")] += vals[0] == vals[2]
                agree[(k, "all3")] += vals[0] == vals[1] == vals[2]
                agree[(k, "2of3")] += c >= 2
            pairs = Counter(o for n in normed for o in n[sid]["replaces"])
            v["replaces"] = sorted(o for o, c in pairs.items() if c >= 2)
            for o in pairs:
                agree[("replaces pair", "2of3")] += pairs[o] >= 2
                agree[("replaces pair", "any")] += 1
            agree[("n", "statements")] += 1
            voted[sid] = v
        answers = {}
        for q in sc["questions"]:
            sets = [set(q["answer_ids"])] + [set((a.get("answers") or {}).get(q["id"]) or []) for a in auds]
            cnt = Counter(x for s_ in sets for x in s_)
            answers[q["id"]] = sorted(x for x, c in cnt.items() if c >= 2)
            agree[("answers", "2of3 same as writer")] += set(answers[q["id"]]) == sets[0]
            agree[("n", "questions")] += 1
        out = {"scenario": sc["scenario"], "writer": job["writer"], "audits": len(auds), "provisional": provisional,
               "votes": voted, "answers": answers}
        (home / "voted.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        rows.append(home.name)
    n, nq = agree[("n", "statements")], agree[("n", "questions")]
    lines = [f"{name}: {len(rows)} people voted, {n} statements, {nq} questions"]
    for k in ("kind", "about", "folder", "merge", "split"):
        lines.append(f"  {k:8s} writer=auditor1 {100 * agree[(k, 'writer=a1')] / max(n, 1):5.1f}%  "
                     f"writer=auditor2 {100 * agree[(k, 'writer=a2')] / max(n, 1):5.1f}%  all three "
                     f"{100 * agree[(k, 'all3')] / max(n, 1):5.1f}%  2-of-3 {100 * agree[(k, '2of3')] / max(n, 1):5.1f}%")
    lines.append(f"  replaces pairs kept by 2 of 3: {agree[('replaces pair', '2of3')]}/{agree[('replaces pair', 'any')]}")
    lines.append(f"  question answers: voted set = writer's {agree[('answers', '2of3 same as writer')]}/{nq}")
    (ROOT / name / "vote_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def report(name: str) -> None:
    for home in sorted((ROOT / name).glob("p*")):
        stray = strayed(home)
        if not (home / "final.json").exists():
            pr = (home / "problems.txt").read_text(encoding="utf-8")[:300] if (home / "problems.txt").exists() else ""
            print(f"{home.name}: not written {pr}"); continue
        sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
        st = sc["statements"]
        print(f"{home.name} {sc['scenario']}: {len(st)} statements, kinds {dict(Counter(s['kind'] for s in st))}, "
              f"ask {sum(s['folder'] == 'ask' for s in st)}, folders {sum(len(s.get('new_folders') or []) for s in st)}, "
              f"merges {sum(bool(s.get('merge')) for s in st)}, splits {sum(bool(s.get('split')) for s in st)}, "
              f"replaces {sum(len(s.get('replaces') or []) for s in st)}, new traps {len(sc.get('new_traps') or [])}"
              + (f"; STRAYED {stray[:2]}" if stray else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    phase, name = args[0], args[1]
    n = SETS[name]["n"]
    lo, hi = map(int, (args[2] if len(args) > 2 else f"1-{n}").split("-"))
    (ROOT / name).mkdir(parents=True, exist_ok=True)
    if phase == "write":
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: write(name, i), range(lo - 1, hi)))
        report(name)
    elif phase == "audit":
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: audit(name, i), range(lo - 1, hi)))
    elif phase == "vote":
        vote(name)
    else:
        report(name)
