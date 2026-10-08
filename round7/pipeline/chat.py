"""Chat-format people (SPEC_CHAT.md), written by agy.   python chat.py run [1-4] | report

Each person: chats with the assistant where facts come up in passing, plus the
answer key (clean facts with supersession labels) and questions. Same discipline
as the long tier: an empty folder per run, every tool call checked for straying
outside it, fixes sent back to the same writer, a template check across people.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import drive
from drive import log, parse
import sampler
from gpt_long import used_names
from long import strayed

HERE = Path(__file__).resolve().parent
ROOT = HERE / "chat"
PRO, FLASH = "gemini-3.1-pro-high", "gemini-3.8-flash-high"
SCHEMA = None  # the chat shape is checked by validate(); a schema'd finish call doubled long replies
LOCK = threading.Lock()
TAIL = ("Write this person from scratch, not from a template, and do not generate it with code. "
        "ONE person, JSON only, nothing else.")
PILOT = [  # (writer, first letter, who, today): hand-picked, the first four only
    (PRO, "B", "a paramedic in Bristol, 34, on rotating shifts, recently separated", "2026-08-14"),
    (FLASH, "R", "a retired seamstress in Medellín, 74, who uses the assistant for her grandchildren's "
                 "homework and her blood-pressure medicine", "2026-09-02"),
    (PRO, "T", "a PhD student in Toulouse, 27, from Vietnam, who switches labs and supervisors", "2026-07-21"),
    (FLASH, "C", "a bike-shop owner and long-distance cyclist in Christchurch, 51, whose wife starts "
                 "chemotherapy", "2026-09-25"),
]
# Everyone after the pilot is drawn at random (sampler.py): nobody hand-picks the lives or the changes.
SEED = 2026
PEOPLE = PILOT + [((PRO, FLASH)[k % 2], None, sampler.assignment(p), p["today"])
                  for k, p in enumerate(sampler.sample(60, SEED))]


def d(s: str) -> date:
    return date.fromisoformat(s)


def validate(sc: dict, letter: str) -> list[str]:
    errs: list[str] = []
    name = sc.get("scenario", "?")
    p = f"[{name}]"
    if not re.fullmatch(r"[a-z]+(-[a-z]+){2,}", name):
        errs.append(f"{p} slug must be lowercase firstname-city-role")
    if letter and not name[:1].upper() == letter:
        errs.append(f"{p} the first name must start with {letter}")
    try:
        today = d(sc["today"])
    except Exception:  # noqa: BLE001
        return errs + [f"{p} bad today"]
    sess = sc.get("sessions") or []
    if [s["id"] for s in sess] != [f"c{i}" for i in range(1, len(sess) + 1)]:
        errs.append(f"{p} session ids must be c1..c{len(sess)} in order")
    if not 8 <= len(sess) <= 18:
        errs.append(f"{p} {len(sess)} sessions (want 10-16)")
    sdate, prev = {}, None
    for s in sess:
        try:
            sdate[s["id"]] = d(s["date"])
        except Exception:  # noqa: BLE001
            errs.append(f"{p} {s.get('id')} bad date"); continue
        if prev and sdate[s["id"]] <= prev:
            errs.append(f"{p} {s['id']} date not after the previous session's")
        prev = sdate[s["id"]]
        turns = s.get("turns") or []
        if not 4 <= len(turns) <= 14:
            errs.append(f"{p} {s['id']} has {len(turns)} turns (want 6-12)")
        roles = [t.get("role") for t in turns]
        if roles != [("user", "assistant")[i % 2] for i in range(len(roles))]:
            errs.append(f"{p} {s['id']} turns must alternate user/assistant, starting with the user")
    if prev and prev > today:
        errs.append(f"{p} a session is dated after today")
    facts = sc.get("facts") or []
    if [f["id"] for f in facts] != [f"f{i}" for i in range(1, len(facts) + 1)]:
        errs.append(f"{p} fact ids must be f1..f{len(facts)} in order")
    if not 35 <= len(facts) <= 70:
        errs.append(f"{p} {len(facts)} facts (want 40-60)")
    order = {s["id"]: i for i, s in enumerate(sess)}
    fdate, gone, pos_prev = {}, {}, (-1, -1)
    ids = [f["id"] for f in facts]
    for i, f in enumerate(facts):
        s = next((x for x in sess if x["id"] == f.get("session")), None)
        if not s:
            errs.append(f"{p} {f['id']} points at session {f.get('session')}, which does not exist"); continue
        t = f.get("turn")
        if not isinstance(t, int) or not 0 <= t < len(s["turns"]) or s["turns"][t].get("role") != "user":
            errs.append(f"{p} {f['id']} must point at a user turn of {s['id']} (turn {t})"); continue
        pos = (order[s["id"]], t)
        if pos < pos_prev:
            errs.append(f"{p} {f['id']} is out of order (facts are numbered in the order stated)")
        pos_prev = pos
        fdate[f["id"]] = sdate.get(s["id"])
        if not 3 <= len(f.get("text", "").split()) <= 35:
            errs.append(f"{p} {f['id']} text should be one clean fact of 5-25 words")
        for old in f.get("replaces") or []:
            if old not in ids[:i]:
                errs.append(f"{p} {f['id']} replaces {old}, which is not an earlier fact")
            elif old in gone:
                errs.append(f"{p} {old} is replaced twice ({gone[old]} and {f['id']})")
            else:
                gone[old] = f["id"]
    users = [(s_["id"], i) for s_ in sess for i, t in enumerate(s_["turns"]) if t.get("role") == "user"]
    carrying = {(f.get("session"), f.get("turn")) for f in facts}
    if users:
        share = sum(1 for u in users if u in carrying) / len(users)
        if share > 0.7:
            errs.append(f"{p} {share:.0%} of user turns state a fact; about half should be task-only (no new fact)")
        words = sorted(len(t["text"].split()) for s_ in sess for t in s_["turns"] if t.get("role") == "user")
        if words[len(words) // 2] < 25:
            errs.append(f"{p} user turns are too short (median {words[len(words) // 2]} words; want 30-70)")
        if len(users) < 3 * len(sess):
            errs.append(f"{p} chats are too short ({len(users)} user turns in {len(sess)} chats; want 3-6 per chat)")
    if not 12 <= len(gone) <= 28:
        errs.append(f"{p} {len(gone)} updates (want 14-24)")
    qs = sc.get("questions") or []
    if [q["id"] for q in qs] != [f"q{i}" for i in range(1, len(qs) + 1)]:
        errs.append(f"{p} question ids must be q1..q{len(qs)} in order")
    if not 22 <= len(qs) <= 36:
        errs.append(f"{p} {len(qs)} questions (want 25-32)")
    kinds = [q.get("type") for q in qs]
    for k, lo in (("current", 8), ("past", 6), ("multi", 3), ("absent", 3)):
        if kinds.count(k) < lo:
            errs.append(f"{p} only {kinds.count(k)} {k} questions")
    lo_date = min(fdate.values()) if fdate else today
    for q in qs:
        qp = f"{p} {q['id']} ({q.get('type')})"
        ans = q.get("answer_ids") or []
        bad = [a for a in ans if a not in fdate]
        if bad:
            errs.append(f"{qp} answer ids {bad} are not facts"); continue
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
            if not lo_date <= at <= today:
                errs.append(f"{qp} as_of {at} is outside the facts' range")
            for a in ans:
                if fdate[a] > at:
                    errs.append(f"{qp} answer {a} was stated after as_of {at}")
                elif a in gone and fdate[gone[a]] <= at:
                    errs.append(f"{qp} answer {a} was already replaced by {gone[a]} on {at}")
        else:
            if q.get("as_of"):
                errs.append(f"{qp} as_of must be null")
            errs += [f"{qp} answer {a} is no longer true (replaced by {gone[a]})" for a in ans if a in gone]
    return errs


def masked(t: str) -> str:
    return re.sub(r"\b[A-Z][\w'-]*|\d[\d,.:/-]*", "X", t).strip().lower()


def template_problems(sc: dict, others: list[dict]) -> list[str]:
    mine = {masked(t["text"]) for s in sc["sessions"] for t in s["turns"] if t["role"] == "user"}
    errs = []
    for o in others:
        theirs = {masked(t["text"]) for s in o["sessions"] for t in s["turns"] if t["role"] == "user"}
        if len(mine & theirs) > 3:
            errs.append(f"[{sc['scenario']}] reuses {len(mine & theirs)} user messages from {o['scenario']}")
    return errs


def write(i: int) -> bool:
    model, letter, who, today = PEOPLE[i]
    cwd = ROOT / f"p{i + 1:02d}"
    if (cwd / "final.json").exists():
        return True
    names, cities = used_names()
    names = sorted(set(names) | {json.loads(f.read_text(encoding="utf-8"))[0]["scenario"].split("-")[0].capitalize()
                                 for f in ROOT.glob("p*/final.json")})
    spec = (HERE / "SPEC_CHAT.md").read_text(encoding="utf-8")
    prompt = (spec + "\n## Do not reuse\n\nFirst names: " + ", ".join(names) + ". Cities: " + ", ".join(cities)
              + ".\n\n## Your assignment\n\n"
              + (f"Person: {who}. Their first name starts with {letter}. `today` = {today}." if letter else who)
              + " " + TAIL + "\n")
    log(f"chat p{i + 1:02d}: writing with {model}")
    try:
        text, conv = drive.agy(prompt, model, cwd)
        for attempt in range(4):
            try:
                sc = parse(text)["scenarios"][0]
                errs = validate(sc, letter)
                if not errs:
                    with LOCK:
                        others = [json.loads(f.read_text(encoding="utf-8"))[0] for f in ROOT.glob("p*/final.json")]
                    errs = template_problems(sc, others)
            except Exception as e:  # noqa: BLE001
                sc, errs = None, [f"the reply is not the JSON asked for ({str(e)[:120]})"]
            log(f"chat p{i + 1:02d}: draft {attempt}: {len(errs)} problems")
            if not errs:
                (cwd / "final.json").write_text(json.dumps([sc], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                (cwd / "writer.txt").write_text(model, encoding="utf-8")
                return True
            if attempt == 3:
                break
            fix = ("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                   "\n\nFix every one (relabel, or edit the chats, facts or questions as needed; keep everything "
                   "else) and reply with the complete corrected JSON {\"scenarios\": [ <the one person> ]} only.")
            text, conv = drive.agy(fix, model, cwd, conversation=conv)
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"chat p{i + 1:02d}: GAVE UP with {len(errs)} problems")
    except Exception as e:  # noqa: BLE001
        log(f"chat p{i + 1:02d}: FAILED: {str(e)[:200]}")
    return False


def report() -> None:
    for d_ in sorted(ROOT.glob("p*")):
        stray = strayed(d_)
        if not (d_ / "final.json").exists():
            print(f"{d_.name}: not written" + (f"; problems: {(d_ / 'problems.txt').read_text(encoding='utf-8')[:300]}"
                                               if (d_ / "problems.txt").exists() else ""))
            continue
        sc = json.loads((d_ / "final.json").read_text(encoding="utf-8"))[0]
        turns = [t for s in sc["sessions"] for t in s["turns"] if t["role"] == "user"]
        words = sorted(len(t["text"].split()) for t in turns)
        with_fact = len({(f["session"], f["turn"]) for f in sc["facts"]})
        print(f"{d_.name} {sc['scenario']} ({(d_ / 'writer.txt').read_text(encoding='utf-8')}): "
              f"{len(sc['sessions'])} chats, {len(turns)} user turns (median {words[len(words) // 2]} words, "
              f"max {words[-1]}), {with_fact} of them carry facts; {len(sc['facts'])} facts, "
              f"{sum(len(f['replaces']) for f in sc['facts'])} updates, {len(sc['questions'])} questions"
              + (f"; STRAYED: {stray[:2]}" if stray else ""))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        lo, hi = map(int, (sys.argv[2] if len(sys.argv) > 2 else f"1-{len(PEOPLE)}").split("-"))
        ROOT.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(lo - 1, hi)))
        report()
    else:
        report()
