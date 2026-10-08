"""ChatGPT (web) as a third writer family for the long tier.   python gpt_long.py prompt N | take N | check

prompt N  put person N's message on the clipboard (N=1 carries the whole spec)
take N    save the clipboard (ChatGPT's copy button on the reply's code block) to gpt_long/rNN.txt
check     run every saved reply through the writers' checks and the template check;
          prints names, counts and problems only, never the text
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from drive import validate
from ingest import objects, template_problems

HERE = Path(__file__).resolve().parent
OUT = HERE / "gpt_long"
OUT.mkdir(exist_ok=True)
TAIL = ("Write this person from scratch: no sentences, question wording or update pattern reused from anyone "
        "earlier, and do not generate it with code or run any code. ONE person, JSON in one code block, nothing else.")
PEOPLE = [
    ("a retired copper miner in the Peruvian Andes near Huancayo, 71, whose grandson moves in", "2026-03-09"),
    ("a garment-factory supervisor in Dhaka, 34, saving to open her own tailoring shop", "2026-03-27"),
    ("a nomadic herder's daughter in Mongolia who moves to Ulaanbaatar for nursing school, 21", "2026-04-18"),
    ("a primary-school teacher in Kharkiv, 45, displaced and later returning", "2026-05-06"),
    ("a fishing-boat mechanic in Reykjavík, 38, with a new partner and a teenage stepson", "2026-05-25"),
    ("a taxi driver in Havana, 56, caring for his elderly father", "2026-06-11"),
    ("a nurse in a rural clinic in Papua New Guinea's highlands, 30", "2026-06-29"),
    ("a software tester in Cebu, 26, who switches to remote work for a foreign company", "2026-07-15"),
    ("a sheep shearer in Patagonia, 48, whose wife starts a guesthouse", "2026-08-02"),
    ("a market trader in Dakar, 41, with four children and a second business", "2026-08-19"),
    ("a mountain guide in Nepal's Khumbu region, 33, recovering from frostbite", "2026-09-04"),
    ("a retired bus driver in rural Alaska, 69, newly diagnosed with heart failure", "2026-09-22"),
]


def used_names() -> tuple[list[str], list[str]]:
    names, cities = [], []
    files = list(HERE.glob("*/final.json")) + list((HERE / "long").glob("p*/final.json"))
    v2 = Path(__file__).resolve().parents[2] / "scripts" / "blind_memory" / "v2" / "test.json"
    for f in files + [v2]:
        for sc in json.loads(f.read_text(encoding="utf-8")):
            parts = sc["scenario"].split("-")
            names.append(parts[0].capitalize())
            cities.append(" ".join(parts[1:-1]).title() if len(parts) > 2 else "")
    return sorted(set(names)), sorted({c for c in cities if c})


def clip_set(text: str) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"], input=text,
                   text=True, encoding="utf-8", check=True)


def clip_get() -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command",
                           "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"],
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout


def prompt(n: int) -> str:
    who, today = PEOPLE[n - 1]
    line = f"Person {n}: {who}, `today` = {today}. {TAIL}"
    if n > 1:
        return line
    names, cities = used_names()
    used = ", ".join(names + ["Elena", "Nikos"]) + "; cities " + ", ".join(cities + ["Bologna", "Thessaloniki"])
    spec = (HERE / "SPEC_LONG.md").read_text(encoding="utf-8").replace("__USED__", used)
    return (spec + "\n## Your assignment\n\nWe will write several people, ONE PER REPLY. Every person in this "
            "chat has a new first name and city, never one from the lists above or earlier in this chat.\n\n"
            + line + "\n")


def check() -> None:
    others = [sc for f in list(HERE.glob("*/final.json")) + list((HERE / "long").glob("p*/final.json"))
              for sc in json.loads(f.read_text(encoding="utf-8"))]
    names = {sc["scenario"].split("-")[0] for sc in others}
    good = []
    for f in sorted(OUT.glob("r*.txt")):
        objs = objects(f.read_text(encoding="utf-8"))
        if not objs:
            print(f"{f.name}: no JSON found ({f.stat().st_size} bytes)")
            continue
        sc = objs[-1]["scenarios"][0]
        errs = validate({"scenarios": [sc]}, 1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if sc["scenario"].split("-")[0] in names:
            errs.append(f"first name of {sc['scenario']} is already used")
        if not errs:
            errs = template_problems(sc, others + good)
        n_up = sum(len(s["replaces"]) for s in sc["statements"])
        print(f"{f.name}: {sc['scenario']}: {len(sc['statements'])} statements, {n_up} updates, "
              f"{len(sc['questions'])} questions -> {'ok' if not errs else f'{len(errs)} problems'}")
        for e in errs:
            print("   - " + e)
        if not errs:
            good.append(sc)
            names.add(sc["scenario"].split("-")[0])
    (OUT / "final.json").write_text(json.dumps(good, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(good)} GPT people pass")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompt":
        text = prompt(int(sys.argv[2]))
        clip_set(text)
        print(f"person {sys.argv[2]} prompt on the clipboard ({len(text):,} chars)")
    elif cmd == "take":
        text = clip_get()
        (OUT / f"r{int(sys.argv[2]):02d}.txt").write_text(text, encoding="utf-8")
        has = "has" if '"scenarios"' in text else "NO"
        print(f"saved r{int(sys.argv[2]):02d}.txt ({len(text):,} chars, {has} scenarios key)")
    elif cmd == "check":
        check()
