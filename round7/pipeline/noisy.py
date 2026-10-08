"""Make memory data messy the way real people type, keeping every label.

    python noisy.py demo                       # show a few statements, clean -> messy
    python noisy.py file IN.json OUT.json [--copies N] [--seed S] [--level L]

Only the typing changes, never the facts: ids, dates, `replaces`, questions and
answers stay exactly as they were, so the noisy copy is labelled for free. Each
statement gets a random mix of layers at a random strength (--level caps it):

  typos        keyboard-neighbour hits, dropped / doubled / swapped letters
  phone        no capitals, little punctuation, SMS shorthand (u, tmrw, bc, pls)
  fragment     the "I" / "my" dropped, filler words cut ("moved to denver last wk")
  filler       lol, idk, umm, haha, ok so
  asr          voice-to-text: homophones, no punctuation, run-on
  stretch      soooo, nooo
  emoji        a trailing emoji or two
  codeswitch   a common word or tag from another language (yaar, pues, lah, wallah, na)

Question texts are left clean: the user's messages are messy, the exam is not.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

NEIGHBOURS = {c: n for c, n in zip(
    "qwertyuiopasdfghjklzxcvbnm",
    ["wa", "qes", "wrd", "etf", "ryg", "tuh", "yij", "uok", "ipl", "o", "qsz", "awdx", "sefc", "drgv", "fthb",
     "gyjn", "hukm", "jil", "kop", "asx", "zsdc", "xdfv", "cfgb", "vghn", "bhjm", "njk"])}
SMS = [(r"\byou\b", "u"), (r"\byour\b", "ur"), (r"\byou're\b", "ur"), (r"\bare\b", "r"), (r"\bbecause\b", "bc"),
       (r"\bplease\b", "pls"), (r"\btomorrow\b", "tmrw"), (r"\btonight\b", "2nite"), (r"\bto\b", "2"),
       (r"\bfor\b", "4"), (r"\bgoing to\b", "gonna"), (r"\bwant to\b", "wanna"), (r"\bgot to\b", "gotta"),
       (r"\bthanks\b", "thx"), (r"\bpeople\b", "ppl"), (r"\bwith\b", "w"), (r"\bwithout\b", "w/o"),
       (r"\bthough\b", "tho"), (r"\bbefore\b", "b4"), (r"\bweek\b", "wk"), (r"\bmonth\b", "mo"),
       (r"\bapartment\b", "apt"), (r"\bappointment\b", "appt"), (r"\bdoctor\b", "dr"), (r"\bprobably\b", "prob"),
       (r"\bsomething\b", "smth"), (r"\bnothing\b", "nth"), (r"\bI don't know\b", "idk"), (r"\band\b", "n")]
HOMOPHONES = [(r"\btheir\b", "there"), (r"\bthere\b", "their"), (r"\bit's\b", "its"), (r"\byour\b", "you're"),
              (r"\bthen\b", "than"), (r"\bweek\b", "weak"), (r"\bknew\b", "new"), (r"\bright\b", "write"),
              (r"\bfour\b", "for"), (r"\bdays\b", "daze"), (r"\bmeet\b", "meat"), (r"\bhear\b", "here")]
FILLERS = ["lol", "idk", "umm", "haha", "ok so", "tbh", "like", "anyway", "ugh", "btw", "lmao", "hmm"]
EMOJI = ["😅", "🙃", "😭", "🙏", "👍", "😩", "🎉", "❤️", "😂", "🤦"]
TAGS = ["yaar", "na", "pues", "lah", "wallah", "ne", "bhai", "o", "jo", "abeg", "sha", "hein", "ja", "bro"]
DROP_LEAD = re.compile(r"^(I am|I'm|I've|I have|I|My|We|We're|We've)\s+", re.I)


def typos(t: str, r: random.Random, k: float) -> str:
    out = []
    for c in t:
        lc = c.lower()
        x = r.random()
        if lc in NEIGHBOURS and x < 0.025 * k:
            out.append(r.choice(NEIGHBOURS[lc]))
        elif c.isalpha() and x < 0.045 * k:
            continue                              # dropped letter
        elif c.isalpha() and x < 0.06 * k:
            out.append(c + c)                     # doubled
        else:
            out.append(c)
    s = "".join(out)
    if k > 1 and len(s) > 6 and r.random() < 0.5:   # one swap of neighbours
        i = r.randrange(len(s) - 1)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


def phone(t: str, r: random.Random, k: float) -> str:
    s = t
    for pat, rep in SMS:
        if r.random() < 0.35 * k:
            s = re.sub(pat, rep, s, flags=re.I)
    if r.random() < 0.8:
        s = s.lower()
    if r.random() < 0.7:
        s = re.sub(r"[.,;:!?'](?=\s|$)", "", s)
    return s


def fragment(t: str, r: random.Random, k: float) -> str:
    s = DROP_LEAD.sub("", t, count=1)
    if s and r.random() < 0.6:
        s = s[0].lower() + s[1:]
    s = re.sub(r"\b(the|a|an|really|just|actually|finally)\s+", lambda m: "" if r.random() < 0.4 * k else m.group(),
               s, flags=re.I)
    return s.rstrip(".")


def filler(t: str, r: random.Random, k: float) -> str:
    f = r.choice(FILLERS)
    return f"{f} {t}" if r.random() < 0.5 else f"{t} {f}"


def asr(t: str, r: random.Random, k: float) -> str:
    s = t
    for pat, rep in HOMOPHONES:
        if r.random() < 0.5 * k:
            s = re.sub(pat, rep, s, count=1, flags=re.I)
    return re.sub(r"[.,;:!?]", "", s)


def stretch(t: str, r: random.Random, k: float) -> str:
    words = t.split()
    cands = [i for i, w in enumerate(words) if re.search(r"[aeiouy]$", w.lower()) and len(w) <= 6]
    if cands:
        i = r.choice(cands)
        words[i] = words[i] + words[i][-1] * r.randint(2, 4)
    return " ".join(words)


def emoji(t: str, r: random.Random, k: float) -> str:
    return t + " " + "".join(r.choice(EMOJI) for _ in range(r.randint(1, 2)))


def codeswitch(t: str, r: random.Random, k: float) -> str:
    tag = r.choice(TAGS)
    return f"{t} {tag}" if r.random() < 0.6 else f"{tag} {t}"


LAYERS = [typos, phone, fragment, filler, asr, stretch, emoji, codeswitch]
WEIGHTS = [0.75, 0.6, 0.4, 0.3, 0.2, 0.15, 0.2, 0.2]


def mess(text: str, r: random.Random, level: float = 1.0) -> str:
    k = r.uniform(0.5, 1.6) * level                      # strength for this statement
    chosen = [f for f, w in zip(LAYERS, WEIGHTS) if r.random() < w * min(level, 1.5)]
    if not chosen:
        chosen = [typos]
    # meaning-preserving order: wording first, then spelling, then decorations
    order = [fragment, asr, phone, typos, stretch, filler, codeswitch, emoji]
    s = text
    for f in order:
        if f in chosen:
            s = f(s, r, k)
    return re.sub(r"\s{2,}", " ", s).strip() or text


def mess_people(scs: list[dict], seed: int, copies: int = 1, level: float = 1.0) -> list[dict]:
    out = []
    for c in range(copies):
        r = random.Random(seed * 1000 + c)
        for sc in scs:
            n = json.loads(json.dumps(sc))
            if copies > 1:
                n["scenario"] = f"{sc['scenario']}-noisy{c + 1}"
            for s in n.get("statements", []):
                s["text"] = mess(s["text"], r, level)
            for sess in n.get("sessions", []):           # chat-format people: only the user's turns
                for t in sess["turns"]:
                    if t.get("role") == "user":
                        t["text"] = mess(t["text"], r, level)
            out.append(n)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "demo":
        src = Path(__file__).resolve().parent / "dev" / "final.json"
        sts = [s["text"] for sc in json.loads(src.read_text(encoding="utf-8")) for s in sc["statements"]][:10]
        r = random.Random(7)
        for t in sts:
            print(f"  {t}\n  -> {mess(t, r)}\n")
    elif sys.argv[1] == "file":
        src, dst = Path(sys.argv[2]), Path(sys.argv[3])
        arg = lambda name, d: type(d)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d  # noqa: E731
        data = json.loads(src.read_text(encoding="utf-8"))
        scs = data if isinstance(data, list) else data["scenarios"]
        out = mess_people(scs, arg("--seed", 1), arg("--copies", 1), arg("--level", 1.0))
        dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(scs)} people -> {len(out)} noisy people in {dst}")
