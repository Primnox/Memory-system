"""Memory embeddings arms on DEV data: paraphrase retrieval, then supersession.

Everything in this file was written by the person building the embeddings arm,
with the system in front of them, and thresholds were chosen by looking at its
numbers. These are DEVELOPMENT scores. The blind test
(`bench_memory_blind.py --embeddings ...`) is the one that counts.

  retrieve    ~40 questions that share no words (or only noise words) with the
              memory that answers them — synonyms, hypernyms ("pets",
              "vehicle", "beverage") — over a 40-memory store. Lexical vs
              hybrid on identical questions. `absent` questions must come back
              empty, and show what the similarity floor costs.
  supersede   implicit updates ("Got a Pixel 9" after "My phone is an iPhone
              13") and look-alike traps that must NOT retire anything (a second
              pet, a sibling's job, a refinement, a one-off event), written
              through the real `remember()`. Scored by (old, successor) pair.

Usage:
    python scripts/bench_memory_embeddings.py retrieve
    python scripts/bench_memory_embeddings.py supersede
    python scripts/bench_memory_embeddings.py supersede --threshold 0.5   # without writing a setting
    python scripts/bench_memory_embeddings.py similarities                 # the cosine of every pair
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cached encoder, no network probe

DAY = 86_400_000
T0 = 1_735_689_600_000

# ── retrieve ─────────────────────────────────────────────────────────────────
STORE = {
    "c1": "My dog is called Biscuit.",
    "c2": "We adopted a cat named Miso.",
    "c3": "I drive a 2015 Honda Civic.",
    "c4": "I'm vegetarian.",
    "c5": "I start my mornings with a flat white.",
    "c6": "My phone is a Pixel 8.",
    "c7": "I work as a nurse at St Mary's hospital.",
    "c8": "My sister Anna lives in Madrid.",
    "c9": "I'm allergic to penicillin.",
    "c10": "I play the cello in a community orchestra.",
    "c11": "I go running along the river every Tuesday.",
    "c12": "I live in a flat in Leeds.",
    "c13": "My ward manager is Priya.",
    "c14": "I'm learning Japanese on Duolingo.",
    "c15": "My favourite film is Spirited Away.",
    "c16": "I take the 8:15 train into the city.",
    "c17": "I have a gym membership at PureGym.",
    "c18": "My partner's name is Jordan.",
    "c19": "I'm saving up for a trip to Iceland.",
    "c20": "I read mostly crime novels before bed.",
    "c21": "I take 10mg of lisinopril every day.",
    "c22": "I'm planning to run the Manchester marathon next April.",
    "c23": "My laptop is a ThinkPad X1.",
    "c24": "I bank with Monzo.",
    "c25": "I prefer window seats on flights.",
    "c26": "My birthday is on the 14th of March.",
    "c27": "I grow tomatoes and basil on the balcony.",
    "c28": "I can't stand loud restaurants.",
    "c29": "I usually go to bed around eleven.",
    "c30": "My brother Sam works at Google.",
    "c31": "I use a standing desk at home.",
    "c32": "My go-to takeaway is Thai green curry.",
    "c33": "I volunteer at the local food bank on Saturdays.",
    "c34": "I wear glasses for reading.",
    "c35": "My favourite band is Radiohead.",
    "c36": "I speak Spanish and a little French.",
    "c37": "I use Obsidian for notes.",
    "c38": "I drink green tea in the afternoon.",
    "c39": "I'm afraid of heights.",
    "c40": "My car insurance renews in October.",
}

# (question, answer ids). Several have more than one right answer.
QUESTIONS = [
    ("what pets do I have", ["c1", "c2"]),
    ("do I have any animals at home", ["c1", "c2"]),
    ("what vehicle do I own", ["c3"]),
    ("what sort of diet do I follow", ["c4"]),
    ("what do I drink first thing in the morning", ["c5"]),
    ("what beverage do I like in the afternoon", ["c38"]),
    ("which handset do I carry", ["c6"]),
    ("what is my occupation", ["c7"]),
    ("where does my sibling live", ["c8"]),
    ("what drugs can't I take", ["c9"]),
    ("which instrument do I play", ["c10"]),
    ("how do I keep fit", ["c11", "c17"]),
    ("who is my supervisor at work", ["c13"]),
    ("what foreign language am I studying", ["c14"]),
    ("what movie do I love most", ["c15"]),
    ("how do I get to the office", ["c16"]),
    ("who is my significant other", ["c18"]),
    ("what holiday am I putting money aside for", ["c19"]),
    ("what genre of books do I enjoy", ["c20"]),
    ("what prescription medicine do I take daily", ["c21"]),
    ("what race am I preparing for", ["c22"]),
    ("which computer do I use", ["c23"]),
    ("which financial institution holds my account", ["c24"]),
    ("where do I like to sit on a plane", ["c25"]),
    ("when is my date of birth", ["c26"]),
    ("do I do any gardening", ["c27"]),
    ("what kind of places do I avoid when eating out", ["c28"]),
    ("what time do I go to sleep", ["c29"]),
    ("what does my brother do for a living", ["c30"]),
    ("what do I like to order when I get food delivered", ["c32"]),
    ("do I do any charity work", ["c33"]),
    ("do I need eyewear", ["c34"]),
    ("which musicians do I enjoy", ["c35"]),
    ("how many languages do I speak", ["c36"]),
    ("what software do I take notes with", ["c37"]),
    ("what am I scared of", ["c39"]),
    ("when do I need to renew my cover for the automobile", ["c40"]),
    ("what is my home city", ["c12"]),
]
ABSENT = [
    "what is my favourite colour",
    "do I have any children",
    "what is my salary",
    "which airline do I fly with",
    "what is my mother's name",
]

# ── supersede ────────────────────────────────────────────────────────────────
# (older statement, newer statement). The newer REPLACES the older.
UPDATES = [
    ("My phone is an iPhone 13.", "Got a Pixel 9 last week."),
    ("I drive a 2015 Honda Civic.", "Picked up a Tesla Model 3 in March."),
    ("I work as a nurse at St Mary's hospital.", "Started a new role as a teacher at Oakfield School."),
    ("I live in Leeds.", "My new flat is in Bristol."),
    ("I take the bus to work.", "Cycling to the office these days."),
    ("My doctor is Dr Patel.", "I see Dr Okafor now for check-ups."),
    ("I'm vegetarian.", "Eating meat again since the spring."),
    ("My go-to workout is yoga.", "I've taken up running as my main exercise."),
    ("I wake up at 6am.", "These days I'm up at 8."),
    ("My favourite band is Radiohead.", "Mostly listening to The National lately."),
    ("My dentist is on Park Road.", "Switched dentists, the new one is on Elm Street."),
    ("My internet provider is Comcast.", "Moved the broadband over to Verizon."),
    ("I'm learning Spanish.", "Dropped Spanish and started on Japanese."),
    ("My laptop is a ThinkPad X1.", "Bought a MacBook Air to replace the old laptop."),
    ("I do my banking with Barclays.", "Opened an account with Monzo and use that for everything."),
    ("I drink black coffee every morning.", "I've gone over to green tea in the mornings."),
]
# Written beside an older statement and must leave it standing.
TRAPS = [
    ("My dog is called Biscuit.", "We adopted a cat named Miso."),
    ("I work at Acme.", "My brother works at Google."),
    ("I drive a 2019 Corolla.", "It's a silver Corolla."),
    ("My phone is an iPhone 13.", "My laptop is a ThinkPad."),
    ("I live in Leeds.", "Visited Lisbon last weekend."),
    ("I'm learning Spanish.", "I also speak some French."),
    ("I like hiking.", "I like pottery."),
    ("My sister lives in Madrid.", "I live in Leeds."),
    ("My manager is Sam.", "My cousin is a manager at Tesco."),
    ("I have a peanut allergy.", "I'm allergic to shellfish."),
    ("I run on Tuesday evenings.", "I swim on Sunday mornings."),
    ("I take the bus to work.", "I took the train to Edinburgh last week."),
    ("I use Notion for notes.", "I use Slack for team chat."),
    ("My wife is a vet.", "My wife's sister is a lawyer."),
    ("I bank with Monzo.", "My mum banks with Barclays."),
    ("I'm training for a half marathon.", "I run five kilometres every Saturday."),
]


# Written after the first set scored too well to be informative: updates with
# NO wording of change, and traps that have some.
HARD_UPDATES = [
    ("My doctor is Dr Patel.", "My GP is Dr Okafor."),
    ("I work as a nurse at St Mary's hospital.", "I'm a teacher at Oakfield School."),
    ("I live in Leeds.", "I'm in Bristol."),
    ("I'm vegetarian.", "I eat meat."),
    ("I wake up at 6am.", "I get up at eight."),
    ("My phone is an iPhone 13.", "My phone is a Pixel 9."),
]
HARD_TRAPS = [
    ("My dog is called Biscuit.", "Recently adopted a cat named Miso."),
    ("My sister lives in Madrid.", "Bought a gift for my sister."),
    ("I live in Leeds.", "New neighbours moved in next door."),
    ("I take the 8:15 train into the city.", "I now also take the train on Fridays."),
    ("I take the 8:15 train into the city.", "Left my umbrella on the train."),
    ("I like hiking.", "Started a podcast about cooking."),
    ("I read mostly crime novels before bed.", "I got a library card."),
    ("I work as a nurse at St Mary's hospital.", "Got a haircut yesterday."),
]


def fresh_store():
    from primnox2.storage import db
    db.configure(Path(tempfile.mkdtemp(prefix="memembed-")) / "primnox.db")
    db.init()
    from primnox2.memory import service as mem
    return mem


def set_flags(search: bool | None = None, supersede: bool | None = None,
              min_similarity: float | None = None, supersede_similarity: float | None = None) -> None:
    for key, val in (("MEMORY_EMBEDDINGS", search), ("MEMORY_EMBEDDINGS_SUPERSEDE", supersede)):
        if val is not None:
            os.environ[f"PRIMNOX2_{key}"] = "1" if val else "0"
    if min_similarity is not None:
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS_MIN_SIMILARITY"] = str(min_similarity)
    if supersede_similarity is not None:
        os.environ["PRIMNOX2_MEMORY_EMBEDDINGS_SUPERSEDE_SIMILARITY"] = str(supersede_similarity)


def retrieve(args) -> None:
    mem = fresh_store()
    ids = {}
    for key, text in STORE.items():
        ids[mem.remember(text)["id"]] = key
    rows = mem.live()
    assert len(rows) == len(STORE), (len(rows), len(STORE))
    from primnox2.memory import embeddings
    embeddings.prepare(180)
    results = {}
    for arm, on in (("lexical", False), ("hybrid", True)):
        set_flags(search=on, min_similarity=args.floor)
        top1 = top3 = multi_all = multi_n = 0
        empty_absent = 0
        misses = []
        for question, answers in QUESTIONS:
            got = [ids[h["id"]] for h in mem.search(question, limit=10)]
            hit1 = bool(got) and got[0] in answers
            top1 += hit1
            top3 += bool(set(answers) & set(got[:3]))
            if len(answers) > 1:
                multi_n += 1
                multi_all += set(answers) <= set(got[:len(answers) + 1])
            if not hit1:
                misses.append((question, answers, got[:3]))
        absent_returned = []
        for question in ABSENT:
            got = [ids[h["id"]] for h in mem.search(question, limit=10)]
            empty_absent += not got
            absent_returned.append((question, got[:3]))
        n = len(QUESTIONS)
        results[arm] = (top1, top3, multi_all, multi_n, empty_absent, misses, absent_returned)
        print(f"{arm:<8} top-1 {top1}/{n} ({100 * top1 / n:.0f}%)  top-3 {top3}/{n}  "
              f"multi all-in-k+1 {multi_all}/{multi_n}  absent returned nothing {empty_absent}/{len(ABSENT)}")
    if args.misses:
        for arm in ("lexical", "hybrid"):
            print(f"\n{arm} misses (top-1):")
            for question, answers, got in results[arm][5]:
                print(f"  {question!r}  want {answers}  got {got}")
            print(f"{arm} absent questions that returned something:")
            for question, got in results[arm][6]:
                if got:
                    print(f"  {question!r} -> {got}")


DISTRACTORS = ["c9", "c10", "c15", "c19", "c26", "c27", "c31", "c33", "c34", "c39"]


def supersede(args) -> None:
    from primnox2.memory import embeddings
    embeddings.prepare(180)
    sets = {
        "easy": [(a, b, True) for a, b in UPDATES] + [(a, b, False) for a, b in TRAPS],
        "hard": [(a, b, True) for a, b in HARD_UPDATES] + [(a, b, False) for a, b in HARD_TRAPS],
    }
    arms = (("rules only", False, False), ("rules + actor guard", False, True),
            ("rules + embeddings", True, False), ("rules + embeddings + actor guard", True, True))
    for arm, emb, actor in arms:
        os.environ["PRIMNOX2_MEMORY_ACTOR_GUARD"] = "1" if actor else "0"
        set_flags(supersede=emb, supersede_similarity=args.threshold)
        line = []
        wrong = []
        total = [0, 0, 0, 0]
        for name, pairs in sets.items():
            tp = fp = fn = tn = 0
            for old, new, should in pairs:
                mem = fresh_store()
                for key in DISTRACTORS:
                    mem.remember(STORE[key])
                a = mem.remember(old)
                b = mem.remember(new)
                retired = a["id"] in b.get("superseded", [])
                if should and retired:
                    tp += 1
                elif should:
                    fn += 1
                    wrong.append((name, "MISSED update", old, new))
                elif retired:
                    fp += 1
                    wrong.append((name, "WRONGLY retired", old, new))
                else:
                    tn += 1
            line.append(f"{name}: caught {tp}/{tp + fn}, wrongly retired {fp}/{fp + tn}")
            total = [total[0] + tp, total[1] + fp, total[2] + fn, total[3] + tn]
        tp, fp, fn, tn = total
        print(f"{arm:<34} {' | '.join(line)} | all: recall {100 * tp / max(1, tp + fn):.0f}% "
              f"precision {100 * tp / max(1, tp + fp):.0f}%")
        if args.misses:
            for name, kind, old, new in wrong:
                print(f"    [{name}] {kind}: {old!r} -> {new!r}")


def similarities(args) -> None:
    from primnox2.memory import embeddings
    embeddings.prepare(180)
    for label, group in (("UPDATES (should retire)", UPDATES + HARD_UPDATES),
                         ("TRAPS (should not)", TRAPS + HARD_TRAPS)):
        print(label)
        vecs = embeddings.encode([t for pair in group for t in pair])
        for i, (old, new) in enumerate(group):
            print(f"  {embeddings.cosine(vecs[2 * i], vecs[2 * i + 1]):.2f}  {old!r} / {new!r}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("retrieve", "supersede", "similarities"))
    ap.add_argument("--floor", type=float, help="memory.embeddings_min_similarity for this run")
    ap.add_argument("--threshold", type=float, help="memory.embeddings_supersede_similarity for this run")
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()
    {"retrieve": retrieve, "supersede": supersede, "similarities": similarities}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
