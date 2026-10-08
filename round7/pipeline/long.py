"""The long tier, written and audited by agy.   python long.py run | report

Each person: one agy writer (Gemini 3.1 Pro or Gemini 3.8 Flash, alternating) in an
empty folder with SPEC_LONG.md and an assignment; checked (long limits, an assigned
first letter, no reuse of anyone else's sentences or update pattern) with fixes sent
back to the same conversation; then audited from a label-free copy by the two
models that did not write it; then settled by majority of three.

Every agy run's tool calls are logged. A run that touched any path outside its own
folder (other than agy's own transcript) disqualifies that person: the set must stay
blind to the code.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import drive
from drive import AUDIT, log, parse, strip_labels, validate
from ingest import template_problems

HERE = Path(__file__).resolve().parent
ROOT = HERE / "long"
# Opus is too heavy on quota; Sonnet keeps a second model family in every vote.
PRO, FLASH_W, FLASH, SONNET = ("gemini-3.1-pro-high", "gemini-3.8-flash-high", "gemini-3.8-flash-medium",
                               "claude-sonnet-4-6")
# The Claude slot is audited by Claude subagents (claude_audit/), not through agy.
AUDITORS = {PRO: [FLASH], FLASH_W: [PRO]}
FALLBACK = {FLASH_W: FLASH, FLASH: "gemini-3.8-flash-low"}  # when a reply runs past the output limit
TAIL = ("Write this person from scratch: no sentences, question wording or update pattern reused from anyone "
        "earlier, and do not generate it with code. ONE person, JSON in one code block, nothing else.")

PEOPLE = [  # (first letter, who, today)
    ("W", "a dairy farmer in rural western Kenya, 58, widowed, three adult children", "2026-03-04"),
    ("D", "a long-haul truck driver based in Winnipeg, 44, divorced, shared custody of a 10-year-old", "2026-03-12"),
    ("J", "a night-shift ICU nurse in Manila, 29, sending money home to her parents in Iloilo", "2026-03-21"),
    ("L", "a retired schoolteacher in a village in Kerala, 78, diabetic, living with her son's family", "2026-03-30"),
    ("T", "a delivery rider in São Paulo, 23, studying for university entrance exams at night", "2026-04-07"),
    ("U", "a freelance software contractor in Tallinn, 36, moving between short contracts", "2026-04-15"),
    ("B", "a stay-at-home father in Melbourne, 41, with twins; his partner works fly-in fly-out mining rotations", "2026-04-24"),
    ("Y", "a man resettled in Hamburg from Syria, 31, learning German and retraining as an electrician", "2026-05-02"),
    ("A", "a sheep farmer in the Scottish Highlands, 67, recovering from a hip replacement", "2026-05-11"),
    ("C", "a university student in Lagos switching from medicine to computer science, 20", "2026-05-19"),
    ("H", "a grocery-shop owner in Hanoi, 52, whose daughter gradually takes over the shop", "2026-05-28"),
    ("R", "a laid-off factory worker in Detroit, 49, rebuilding a career", "2026-06-05"),
    ("S", "a former monk turned tour guide in Luang Prabang, 34", "2026-06-14"),
    ("F", "a part-time carer for her mother with dementia in Porto, 55", "2026-06-22"),
    ("O", "a footballer in the Moroccan second division, 26, who has a career-ending injury partway through", "2026-07-01"),
    ("M", "a graduate student in Seoul with a long-distance partner in Busan, 27", "2026-07-09"),
    ("G", "a fisherman in Newfoundland, 61, who sells his boat and takes up carpentry", "2026-07-18"),
    ("P", "a pharmacist in Isfahan, 33, expecting her first child", "2026-07-26"),
    ("E", "a pastor in Ho, in Ghana's Volta region, 47, with a large extended family", "2026-08-04"),
    ("Z", "a 19-year-old competitive gamer in Kraków who quits to work in retail", "2026-08-12"),
    ("V", "a single mother of two in Santiago de Chile, 38, doing a night-school accounting degree", "2026-08-21"),
    ("I", "a hunter and wilderness guide in Iqaluit, 52", "2026-08-29"),
    ("N", "a chef in Dublin, 45, in recovery from alcohol, opening a new restaurant", "2026-09-07"),
    ("K", "a widower in Sapporo, 82, moving into assisted living", "2026-09-15"),
]
USED = ("Anika, Bjorn, Carmen, Dmitri, Esther, Hiroshi, Nandini, Wanjiru, Rafael, Sven, Tessa, Walter, Alejandro, "
        "Fatima, Budi, Daria, Hassan, Chloe, Linnea, Ivor, Mariana, Patrick, Rohan, Ren, Nicolas, Zola, Rustam, Elena, "
        "Nikos; cities Bengaluru, Bergen, Guadalajara, Toronto, Leeds, Osaka, Hyderabad, Nairobi, Curitiba, "
        "Gothenburg, Perth, any city in Ohio, Buenos Aires, Accra, Jakarta, Warsaw, Cairo, Auckland, Trondheim, "
        "Cardiff, Montevideo, Montreal, Pune, Kyoto, Lyon, Cape Town, Tashkent, Bologna, Thessaloniki")
SPEC_LONG = (HERE / "SPEC_LONG.md").read_text(encoding="utf-8").replace("__USED__", USED)
LOCK = threading.Lock()
SCHEMA = HERE / "schema_write.json"


def writer_of(i: int) -> str:
    return PRO if i % 2 == 0 else FLASH_W


def salvage(cwd: Path) -> str | None:
    """A writer that ran past the output limit may already have saved the person to a file."""
    for f in sorted(cwd.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        if f.name.startswith(("final", "labels", "draft")):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if '"scenarios"' in text:
            return text
    return None


def strayed(cwd: Path) -> list[str]:
    """Tool calls in this folder's agy runs that reached outside it."""
    own = str(cwd).lower().replace("/", "\\")
    bad = []
    for raw in cwd.glob("raw_*.out"):
        for line in raw.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            su = e.get("step_update") or {}
            if su.get("step_type") != "tool" or su.get("state") != "DONE" or su.get("tool_name") == "finish":
                continue
            params = json.dumps((su.get("tool_info") or {}).get("parameters", su.get("tool_info")))
            for path in re.findall(r"[A-Za-z]:(?:\\\\|/)[^\"' ]+", params):
                p = path.replace("\\\\", "\\").replace("/", "\\").lower()
                if not p.startswith(own) and "\\.gemini\\antigravity-cli\\brain\\" not in p:
                    bad.append(f"{su.get('tool_name')}: {path[:120]}")
            # Indirect routes: the project named anywhere but inside the scratchpad path (which
            # itself contains "Projects-Primnox"), home-relative paths, or climbing out with "..".
            rest = re.sub(re.escape(str(HERE.parent)).replace(r"\\", r"(?:\\\\|\\|/)"), "", params, flags=re.I)
            if re.search(r"primnox|%userprofile%|\$home|\$env:|~[\\/]|\.\.[\\/]", rest, re.I):
                bad.append(f"{su.get('tool_name')}: {params[:160]}")
    return bad


def everyone_else(me: Path) -> list[dict]:
    out = []
    for f in list(HERE.glob("*/final.json")) + list(ROOT.glob("p*/final.json")):
        if f.parent != me:
            out += json.loads(f.read_text(encoding="utf-8"))
    return out


def write(i: int) -> bool:
    letter, who, today = PEOPLE[i]
    model = writer_of(i)
    cwd = ROOT / f"p{i + 1:02d}"
    if (cwd / "final.json").exists():
        return True
    prompt = (SPEC_LONG + "\n## Your assignment\n\nPerson: " + who + ". Their first name starts with the letter "
              + letter + ". `today` = " + today + ". " + TAIL + "\n")
    log(f"long p{i + 1:02d}: writing with {model}")
    try:
        try:
            text, conv = drive.agy(prompt, model, cwd, schema=SCHEMA)
        except RuntimeError as e:
            saved = salvage(cwd)
            if saved:
                log(f"long p{i + 1:02d}: {model} failed ({str(e)[:80]}); using the file it saved")
                text, conv = saved, None
            elif "output token limit" in str(e) and model in FALLBACK:
                model = FALLBACK[model]
                log(f"long p{i + 1:02d}: past the output limit; rewriting with {model}")
                text, conv = drive.agy(prompt, model, cwd, schema=SCHEMA)
            else:
                raise
        for attempt in range(4):
            try:
                data = parse(text)
                errs = validate(data, 1, letter)
                if not errs:
                    with LOCK:
                        errs = template_problems(data["scenarios"][0], everyone_else(cwd))
            except Exception as e:  # noqa: BLE001
                data, errs = None, [f"the reply is not the JSON asked for ({str(e)[:120]})"]
            log(f"long p{i + 1:02d}: draft {attempt}: {len(errs)} problems")
            if not errs:
                (cwd / "final.json").write_text(json.dumps(data["scenarios"], indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
                (cwd / "writer.txt").write_text(model, encoding="utf-8")
                return True
            if attempt == 3 or conv is None:
                break
            fix = ("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                   "\n\nFix every one (relabel, or edit the text, dates or questions as needed; keep everything "
                   "else) and reply with the complete corrected JSON {\"scenarios\": [ <the one person> ]} only.")
            text, conv = drive.agy(fix, model, cwd, conversation=conv, schema=SCHEMA)
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"long p{i + 1:02d}: GAVE UP with {len(errs)} problems")
    except Exception as e:  # noqa: BLE001
        log(f"long p{i + 1:02d}: FAILED: {str(e)[:200]}")
    return False


def audit(i: int, model: str) -> None:
    cwd = ROOT / f"p{i + 1:02d}" / f"audit_{model}"
    out = cwd.parent / f"labels_{model}.json"
    if out.exists():
        return
    sc = json.loads((cwd.parent / "final.json").read_text(encoding="utf-8"))
    bare = json.dumps(strip_labels(sc), indent=1, ensure_ascii=False)
    for m in (model, FALLBACK[model]) if model in FALLBACK else (model,):
        for attempt in range(2):
            try:
                text, _ = drive.agy(AUDIT + "\n" + bare, m, cwd)
                lab = parse(text)["labels"][0]
                out.write_text(json.dumps({"labels": [lab]}, indent=1, ensure_ascii=False), encoding="utf-8")
                log(f"long p{i + 1:02d}: audit by {m} done")
                return
            except Exception as e:  # noqa: BLE001
                log(f"long p{i + 1:02d}: audit by {m} attempt {attempt} failed: {str(e)[:160]}")


AUDIT_FUTURES: list = []


def person(i: int, pool: ThreadPoolExecutor) -> None:
    if write(i):
        for m in AUDITORS[writer_of(i)]:
            with LOCK:
                AUDIT_FUTURES.append(pool.submit(audit, i, m))


def run() -> None:
    """`run` does everyone; `run 1-8` only those people (1-based), e.g. to retry a failed wave
    while another run is still going."""
    ROOT.mkdir(exist_ok=True)
    who = range(len(PEOPLE))
    if len(sys.argv) > 2:
        lo, hi = map(int, sys.argv[2].split("-"))
        who = range(lo - 1, hi)
    with ThreadPoolExecutor(max_workers=8) as pool:
        writers = [pool.submit(person, i, pool) for i in who]
        for f in writers:
            f.result()  # every audit is submitted before its writer's future completes
        for f in list(AUDIT_FUTURES):
            f.result()
    report()


def vote(sc: dict, labs: list[dict]) -> tuple[dict, Counter]:
    sc = json.loads(json.dumps(sc))
    ch = Counter()
    for s in sc["statements"]:
        v = Counter(s["replaces"])
        for lab in labs:
            v.update((lab.get("replaces") or {}).get(s["id"]) or [])
        new = sorted((o for o, n in v.items() if n >= 2), key=lambda x: int(x[1:]))
        ch["pairs dropped"] += len(set(s["replaces"]) - set(new))
        ch["pairs added"] += len(set(new) - set(s["replaces"]))
        s["replaces"] = new
    for q in sc["questions"]:
        v = Counter(q["answer_ids"])
        for lab in labs:
            v.update((lab.get("answers") or {}).get(q["id"]) or [])
        new = sorted((x for x, n in v.items() if n >= 2), key=lambda x: int(x[1:]))
        if set(new) != set(q["answer_ids"]):
            ch[f"{q['type']} answer sets changed"] += 1
        q["answer_ids"] = new
    ch["answer ids dropped to match the voted updates"] += drive.reconcile(sc)
    # No answer two of three agree on: no ground truth, so the question goes (ids are kept as
    # they are, so the gap shows and the audits still line up).
    kept = [q for q in sc["questions"] if q["type"] == "absent" or q["answer_ids"]]
    ch["questions dropped, no majority answer"] += len(sc["questions"]) - len(kept)
    sc["questions"] = kept
    return sc, ch


def report() -> None:
    written, settled, lines = [], [], []
    agree = Counter()
    changes = Counter()
    for i in range(len(PEOPLE)):
        d = ROOT / f"p{i + 1:02d}"
        if not (d / "final.json").exists():
            lines.append(f"p{i + 1:02d}: not written")
            continue
        stray = strayed(d) + [x for a in d.glob("audit_*") for x in strayed(a)]
        if stray:
            lines.append(f"p{i + 1:02d}: DISQUALIFIED, a run reached outside its folder: {stray[:3]}")
            continue
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        labs = [json.loads(f.read_text(encoding="utf-8"))["labels"][0] for f in sorted(d.glob("labels_*.json"))]
        model = (d / "writer.txt").read_text(encoding="utf-8")
        w = {(o, s["id"]) for s in sc["statements"] for o in s["replaces"]}
        for lab in labs:
            r = {(o, sid) for sid, os_ in (lab.get("replaces") or {}).items() for o in (os_ or [])}
            agree["pairs writer"] += len(w); agree["pairs both"] += len(w & r); agree["pairs auditor extra"] += len(r - w)
            for q in sc["questions"]:
                agree["answers"] += 1
                agree["answers agree"] += set((lab.get("answers") or {}).get(q["id"], ["?"])) == set(q["answer_ids"])
        written.append(sc)
        if len(labs) == 2:
            v, ch = vote(sc, labs)
            errs = [e for e in validate({"scenarios": [v]}, 1, sc["scenario"][0].upper())
                    if "updates (want" not in e and "question ids must be" not in e]
            changes.update(ch)
            settled.append(v)
            lines.append(f"p{i + 1:02d} {sc['scenario']} ({model}): {len(sc['statements'])} statements, "
                         f"{len(w)} updates, {len(sc['questions'])} questions; vote changed {sum(ch.values())} labels"
                         + (f"; AFTER-VOTE PROBLEMS: {errs}" if errs else ""))
        else:
            lines.append(f"p{i + 1:02d} {sc['scenario']} ({model}): written, {len(labs)}/2 audits")
    (ROOT / "test_long.json").write_text(json.dumps(written, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "adjudicated.json").write_text(json.dumps(settled, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    qs = [q for sc in settled for q in sc["questions"]]
    summary = [f"written {len(written)}/{len(PEOPLE)}, settled by vote {len(settled)}",
               f"statements {sum(len(sc['statements']) for sc in settled)}, updates "
               f"{sum(len(s['replaces']) for sc in settled for s in sc['statements'])}, questions {len(qs)} "
               f"{dict(Counter(q['type'] for q in qs))}",
               f"writer vs each auditor: {agree['pairs both']}/{agree['pairs writer']} update pairs found "
               f"(+{agree['pairs auditor extra']} extra); answer sets {agree['answers agree']}/{agree['answers']}",
               f"vote changes: {dict(changes)}"]
    text = "\n".join(summary + [""] + lines) + "\n"
    (ROOT / "report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
