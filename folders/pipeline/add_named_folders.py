"""File every fact also under each named thing it mentions (code only, no model).   python add_named_folders.py SRC DST

The ~500M model picks a fact's main folder and slot; this adds the fact to a folder for every
name ("Paisabridge", "Mochi") and "my <relative or thing>" it mentions — creating the folder
the first time, as the cloud clerk does with its secondary folders. Names follow the app's own
rules (memory/head.py: capitalised words used as names away from a sentence start; "my
<relative>"), learned only from facts said so far. Each new folder also gets a link written by
code from the model's own decision — the slot it chose ("Anika's employer") or the relative named
("Anika's sister"). Reads SRC/blind/*/state.json, writes DST.
"""
import json
import re
import sys
from pathlib import Path

import folders

KIN = re.compile(r"\bmy (?:little |big |older |younger |new |best |ex-?)?(wife|husband|partner|boyfriend|girlfriend|"
                 r"fianc[eé]e?|son|daughter|kids?|mum|mom|mother|dad|father|brother|sister|grandma|grandmother|grandpa|"
                 r"grandfather|aunt|uncle|cousin|niece|nephew|dog|cat|puppy|kitten|boss|manager|landlord|flatmate|"
                 r"roommate|friend|colleague|doctor|therapist|car|bike|flat|apartment|house|phone|laptop)\b", re.I)
WORD = re.compile(r"[A-Z][a-zA-Z'\-]+")
SKIP = {"I", "I'm", "I've", "I'd", "I'll", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
        "December", "Christmas", "Diwali", "Easter"}


def names_so_far(texts: list[str]) -> set[str]:
    seen = set()
    for t in texts:
        for m in WORD.finditer(t):
            before = t[:m.start()].rstrip()
            if before and before[-1] not in ".!?\"“(":
                w = m.group().split("'")[0]
                if w not in SKIP:
                    seen.add(w)
    return seen


LINKS = True


def main(src: Path, dst: Path) -> None:
    people = {re.sub(r"[^\w\-]", "_", sc["scenario"]): sc for sc in folders.load("blind")}
    for f in sorted((src / "blind").glob("*/state.json")):
        st = json.loads(f.read_text(encoding="utf-8"))
        sc = people[f.parent.name]
        facts = sorted(sc["statements"], key=lambda s: (s["date"], s["id"]))
        speaker = st["folders"]["F1"]["name"]
        index = {}                                                # lower-case name or alias -> folder id
        for k, v in st["folders"].items():
            for n in [v.get("name", "")] + list(v.get("aliases") or []):
                if isinstance(n, str) and len(n) >= 3:
                    index.setdefault(n.lower(), k)
        made = 0
        for i, s in enumerate(facts):
            names = names_so_far([x["text"] for x in facts[:i + 1]])
            mentions = [w.split("'")[0] for w in WORD.findall(s["text"]) if w.split("'")[0] in names and w != speaker]
            mentions += [m.group(0).lower() for m in KIN.finditer(s["text"])]
            e = st["facts"][s["id"]]
            # a link from the model's own decision: a thing named in a fact the model filed under the
            # speaker's slot X is "<speaker>'s X" ("Anika's employer"); "my sister" is "<speaker>'s sister"
            subject, slot = e["slot"].split(".", 1)
            kin = {m.group(0).lower(): m.group(1).lower() for m in KIN.finditer(s["text"])}
            for m in dict.fromkeys(mentions):
                k = index.get(m.lower())
                if k is None:
                    k = f"F{max(int(x[1:]) for x in st['folders']) + 1}"
                    st["folders"][k] = {"name": m, "type": "", "aliases": [], "link": ""}
                    index[m.lower()] = k
                    made += 1
                if k not in e["folders"]:
                    e["folders"].append(k)
                if LINKS and not st["folders"][k].get("link") and k != "F1":
                    if m.lower() in kin:
                        st["folders"][k]["link"] = f"{speaker}'s {kin[m.lower()]}"
                    elif subject == "F1" and slot not in ("other", "events", "details"):
                        st["folders"][k]["link"] = f"{speaker}'s {slot.replace('_', ' ')}"
            # the main folder, when the model put the fact under someone else, links through a relative named here
            if LINKS and subject != "F1" and not st["folders"][subject].get("link") and kin:
                st["folders"][subject]["link"] = f"{speaker}'s {next(iter(kin.values()))}"
        out = dst / "blind" / f.parent.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "state.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{f.parent.name}: +{made} folders")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
