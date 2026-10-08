"""Test case 3: people who mix languages, written by agy.   python multilang.py run [lo-hi] | report

Short-tier statement format (SPEC.md: ~25 statements, ~16 questions, labelled
updates and traps), so the memory head reads it directly. Lives and changes come
from sampler.py (random, not hand-picked); each person also gets a language mix
and a degree of mixing. Same discipline as every agy set: an empty folder per run,
tool calls checked for straying, fixes back to the same writer, a template check.
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import drive
import sampler
from drive import log, parse, validate
from ingest import template_problems
from long import strayed

HERE = Path(__file__).resolve().parent
ROOT = HERE / "cases" / "multilang"
PRO, FLASH = "gemini-3.1-pro-high", "gemini-3.8-flash-high"
SEED = 3003
MIXES = [
    ("Hinglish", "Hindi and English, Hindi written in Latin letters as people text it (\"kal office nahi gaya\")"),
    ("Spanglish", "Spanish and English, switching mid-sentence"),
    ("Taglish", "Tagalog and English"),
    ("Arabizi", "Arabic in Latin letters with numbers for sounds (3, 7, 2) mixed with English"),
    ("Nigerian Pidgin", "Nigerian Pidgin and English"),
    ("Singlish", "Singapore English with its particles (lah, lor, leh) and Malay or Hokkien words"),
    ("Franglais", "French and English"),
    ("Sheng", "Swahili, Sheng slang and English"),
    ("Manglish", "Malaysian English with Malay words"),
    ("Konglish", "Korean (in Hangul or romanised) and English"),
    ("Denglisch", "German and English"),
    ("Portuñol", "Portuguese, Spanish and English mixed"),
]
DEGREES = [
    "light: mostly English with a few words from the other language",
    "medium: switches language mid-sentence often",
    "heavy: many messages mostly in the other language, with English words and names mixed in",
]
SPEC = (HERE / "SPEC.md").read_text(encoding="utf-8")


def plan() -> list[tuple[str, str]]:
    r = random.Random(SEED)
    people = []
    for k, p in enumerate(sampler.sample(24, SEED)):
        name, how = MIXES[k % len(MIXES)]
        degree = r.choice(DEGREES)
        assignment = (
            "Write 1 scenario (not 6): this ONE person.\n" + sampler.assignment(p) +
            f"\nLANGUAGE: this person mixes languages the way real speakers type: {name} — {how}. Degree: "
            f"{degree}. Write every statement AND every question in their real way of typing, including "
            "spelling as people actually text it. The facts must still be clearly stated in the text; labels "
            "follow the same rules as always. Names of people, places and brands stay as they are.")
        people.append(((PRO, FLASH)[k % 2], assignment))
    return people


def write(i: int) -> bool:
    model, assignment = plan()[i]
    cwd = ROOT / f"p{i + 1:02d}"
    if (cwd / "final.json").exists():
        return True
    prompt = SPEC + "\n\n## Your assignment\n\n" + assignment + "\n"
    log(f"multilang p{i + 1:02d}: writing with {model}")
    try:
        text, conv = drive.agy(prompt, model, cwd)
        for attempt in range(4):
            try:
                data = parse(text)
                errs = validate(data, 1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                if not errs:
                    others = [json.loads(f.read_text(encoding="utf-8"))[0] for f in ROOT.glob("p*/final.json")]
                    errs = template_problems(data["scenarios"][0], others)
            except Exception as e:  # noqa: BLE001
                data, errs = None, [f"the reply is not the JSON asked for ({str(e)[:120]})"]
            log(f"multilang p{i + 1:02d}: draft {attempt}: {len(errs)} problems")
            if not errs:
                (cwd / "final.json").write_text(json.dumps(data["scenarios"], indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
                (cwd / "writer.txt").write_text(model, encoding="utf-8")
                return True
            if attempt == 3 or conv is None:
                break
            fix = ("Your JSON breaks these rules of the spec:\n" + "\n".join(f"- {e}" for e in errs[:60]) +
                   "\n\nFix every one and reply with the complete corrected JSON {\"scenarios\": [ <the one "
                   "person> ]} only. Keep the person's way of writing.")
            text, conv = drive.agy(fix, model, cwd, conversation=conv)
        (cwd / "problems.txt").write_text("\n".join(errs), encoding="utf-8")
        log(f"multilang p{i + 1:02d}: GAVE UP with {len(errs)} problems")
    except Exception as e:  # noqa: BLE001
        log(f"multilang p{i + 1:02d}: FAILED: {str(e)[:200]}")
    return False


def report() -> None:
    done = 0
    for i, (model, assignment) in enumerate(plan()):
        d = ROOT / f"p{i + 1:02d}"
        mix = assignment.split("LANGUAGE: ")[1].split(":")[1].split(" —")[0].strip() if "LANGUAGE: " in assignment else "?"
        if not (d / "final.json").exists():
            print(f"p{i + 1:02d} ({mix}): not written")
            continue
        sc = json.loads((d / "final.json").read_text(encoding="utf-8"))[0]
        stray = strayed(d)
        done += 1
        print(f"p{i + 1:02d} {sc['scenario']} ({mix}, {model.split('-')[1]}): {len(sc['statements'])} statements, "
              f"{sum(len(s['replaces']) for s in sc['statements'])} updates, {len(sc['questions'])} questions"
              + (f"; STRAYED {stray[:1]}" if stray else ""))
    print(f"{done}/24 written")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        lo, hi = map(int, (sys.argv[2] if len(sys.argv) > 2 else "1-24").split("-"))
        ROOT.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(write, range(lo - 1, hi)))
        report()
    else:
        report()
