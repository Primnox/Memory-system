"""Filing as decisions, so the ~500M memory model itself builds the index.   python build_filing_rows.py

Every fact a teacher clerk filed (Gemini through agy, or Claude subagents through clerk_step.py;
the 127 training people only, never blind v2, LongMemEval or REALTALK) becomes up to three
CHOICE decisions, in Laya's row format (the same as the change-detection rows), each made with
only what existed when the fact arrived (the teacher's own filing calls are replayed in order):

  folder  Which folder is this fact mainly about?   options: the speaker, up to 7 other existing
          folders (named in the fact first, then the most similar), and "new"
  type    (only when the answer is "new") What kind of thing is it?   person / pet / place /
          organisation / thing / activity
  slot    Which attribute of that folder does it set?   a FIXED list of 34 slots (SLOTS); the
          teachers' 329 free-form names map onto it (KEY_MAP), the rest to "details" (another
          folder's) or "other" (the speaker's)

Names of new folders are not a decision: code takes them from the fact (memory/head.py's name
and "my <relative>" rules), as the app would. Held out: the clerk's 8 validation people.
Output: ../kaggle_filing_ds/{train,val}_filing_rows.jsonl, slots.json
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import folders
from build_clerk_data import person as replay_batches   # noqa: F401  (same replay as the clerk data)
from build_clerk_data import teacher_reply
from graph_search import Encoder

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle_filing_ds"
SETS = ("chat", "writers", "multilang", "v3short", "v3dev", "v3long")
K_FOLDERS = 8

# slot -> holds ONE value at a time, or MANY
SLOTS = {"job": "one", "employer": "one", "manager": "one", "work schedule": "one", "lives in": "one",
         "rent": "one", "commute": "one", "studies": "one", "partner": "one", "family": "many",
         "friends": "many", "pet": "many", "age": "one", "health": "many", "medication": "many",
         "doctor": "one", "diet": "one", "routine": "many", "sport": "many", "hobby": "many",
         "likes": "many", "phone": "one", "car": "one", "devices": "many", "finances": "many",
         "plans": "many", "trip": "many", "events": "many", "volunteering": "many",
         "languages": "many", "legal": "many", "religion": "one", "details": "many", "other": "many"}
KEY_MAP = {"job": "job", "role": "job", "position": "job", "work": "job", "occupation": "job", "career": "job",
           "employer": "employer", "company": "employer", "workplace": "employer",
           "manager": "manager", "boss": "manager", "supervisor": "manager",
           "work_schedule": "work schedule", "schedule": "work schedule", "shift": "work schedule", "shifts": "work schedule",
           "lives_in": "lives in", "home": "lives in", "address": "lives in", "housing": "lives in", "residence": "lives in",
           "location": "lives in", "rent": "rent", "commute": "commute", "transport": "commute",
           "studies": "studies", "education": "studies", "school": "studies", "course": "studies", "degree": "studies",
           "partner": "partner", "relationship": "partner", "spouse": "partner", "dating": "partner", "marriage": "partner",
           "family": "family", "children": "family", "kids": "family", "siblings": "family", "parents": "family",
           "friends": "friends", "friendship": "friends",
           "pet": "pet", "pets": "pet", "livestock": "pet",
           "age": "age", "birthday": "age", "health": "health", "condition": "health", "allergies": "health",
           "allergy": "health", "fitness_health": "health", "injury": "health", "mental_health": "health",
           "medication": "medication", "medications": "medication", "meds": "medication",
           "doctor": "doctor", "therapist": "doctor", "diet": "diet", "food": "diet",
           "routine": "routine", "habits": "routine", "habit": "routine", "gym": "routine", "exercise": "routine",
           "smoking": "routine", "sleep": "routine",
           "sport": "sport", "fitness": "sport", "running": "sport", "hobby": "hobby", "hobbies": "hobby",
           "interests": "hobby", "likes": "likes", "dislikes": "likes", "preferences": "likes", "favourites": "likes",
           "phone": "phone", "car": "car", "vehicle": "car", "truck": "car", "bike": "car", "motorbike": "car",
           "laptop": "devices", "computer": "devices", "appliances": "devices", "gear": "devices", "device": "devices",
           "finances": "finances", "bank": "finances", "subscriptions": "finances", "subscription": "finances",
           "utilities": "finances", "savings": "finances", "debt": "finances", "income": "finances", "salary": "finances",
           "plans": "plans", "goal": "plans", "goals": "plans", "trip": "trip", "travel": "trip", "trips": "trip",
           "events": "events", "event": "events", "volunteering": "volunteering", "volunteer_work": "volunteering",
           "languages": "languages", "language": "languages", "legal": "legal", "visa": "legal",
           "religion": "religion", "faith": "religion",
           "features": "details", "appearance": "details", "description": "details", "breed": "details",
           "owner": "details", "legal_name": "details", "staff": "details", "status": "details"}
TYPES = {"person": "a person", "pet": "a pet or animal", "place": "a place", "organisation": "a company, school or group",
         "thing": "an object: a car, phone, house…", "activity": "an activity, project or hobby"}
Q_FOLDER = "Which folder in this person's memory is the new fact mainly about?"
Q_TYPE = "The fact is about someone or something new. What kind of thing is it?"
Q_SLOT = "Which attribute of that folder does the fact set or add to?"


def slot_of(key: str, subject_is_speaker: bool) -> str:
    k = key.lower()
    if k in KEY_MAP:
        return KEY_MAP[k]
    for part in k.split("_"):
        if part in KEY_MAP:
            return KEY_MAP[part]
    return "other" if subject_is_speaker else "details"


def card(f: dict) -> str:
    al = [a for a in f.get("aliases") or [] if a.lower() not in ("i", "me", "my", "the user")][:2]
    kind = "the user" if f.get("type") == "speaker" else f.get("type", "")
    return kind + (f"; {', '.join(al)}" if al else "")


def option_key(fid: str, f: dict, taken: set[str]) -> str:
    key = re.sub(r"\s+", " ", str(f.get("name") or fid)).strip()[:40] or fid
    return key if key not in taken else f"{key} ({fid})"


def candidates(fact: str, state: dict, fv, enc) -> list[str]:
    """The speaker, then folders named in the fact, then the most similar folder cards."""
    low = " " + re.sub(r"[^\w']+", " ", fact.lower()) + " "
    speaker = [k for k, v in state["folders"].items() if v.get("type") == "speaker"]
    named, rest = [], []
    for k, v in state["folders"].items():
        if k in speaker:
            continue
        names = [v.get("name", "")] + list(v.get("aliases") or [])
        if any(len(n) >= 3 and f" {n.lower()} " in low for n in names if isinstance(n, str)):
            named.append(k)
        else:
            rest.append(k)
    if rest:
        texts = [f"{state['folders'][k].get('name', '')} ({card(state['folders'][k])})" for k in rest]
        todo = [t for t in dict.fromkeys(texts) if t not in CARD_VEC]      # each card encoded once
        if todo:
            CARD_VEC.update(zip(todo, enc(todo)))
        rest = [rest[i] for i in sorted(range(len(rest)), key=lambda i: -float(CARD_VEC[texts[i]] @ fv))]
    return (speaker + named + rest)[:K_FOLDERS]


CARD_VEC: dict = {}


def rows_for_person(name: str, sc: dict, enc) -> list[dict] | None:
    home = folders.OUT / name / re.sub(r"[^\w\-]", "_", sc["scenario"])
    if not (home / "state.json").exists():
        return None
    final = json.loads((home / "state.json").read_text(encoding="utf-8"))
    state = folders.first_state(name, sc)
    facts = sorted(sc["statements"], key=lambda f: f["date"])
    rows = []
    for k in range(0, len(facts), folders.BATCH):
        batch = facts[k:k + folders.BATCH]
        got = teacher_reply(home / f"b{k // folders.BATCH:03d}", batch, state)
        if got is None:
            return None
        new_folders = {f["id"]: f for f in got.get("new_folders", []) or [] if isinstance(f, dict) and f.get("id")}
        fvs = enc([f["text"] for f in batch])
        for i, f in enumerate(batch):
            e = got["facts"][f["id"]]
            subject = e["slot"].split(".")[0]
            text = f'New fact ({f["date"]}): "{f["text"]}"'
            speaker = next((v.get("name") for v in state["folders"].values() if v.get("type") == "speaker"), None)
            if speaker:
                text += f"\nThe speaker is {speaker}."
            cands = candidates(f["text"], state, fvs[i], enc)
            is_new = subject not in state["folders"]
            if not is_new and subject not in cands:
                cands = cands[:K_FOLDERS - 1] + [subject]
            taken, crit, key_of = set(), {}, {}
            for fid in cands:
                kk = option_key(fid, state["folders"][fid], taken)
                taken.add(kk); key_of[fid] = kk
                crit[kk] = card(state["folders"][fid])
            crit["new"] = "someone or something not in memory yet"
            gold = "new" if is_new else key_of[subject]
            meta = {"scenario": sc["scenario"], "fact": f["id"], "kind": "filing"}
            rows.append({"state": json.dumps(text), "questions": json.dumps({"folder": {"type": "choice", "instructions": Q_FOLDER, "criteria": crit}}),
                         "gold": json.dumps({"folder": {"probabilities": {gold: 1.0}}}), "meta": {**meta, "q": "folder"}})
            sf = new_folders.get(subject) if is_new else state["folders"].get(subject)
            if is_new and sf and sf.get("type") in TYPES:
                rows.append({"state": json.dumps(text), "questions": json.dumps({"type": {"type": "choice", "instructions": Q_TYPE, "criteria": TYPES}}),
                             "gold": json.dumps({"type": {"probabilities": {sf["type"]: 1.0}}}), "meta": {**meta, "q": "type"}})
            subj_speaker = bool(sf) and (sf.get("type") == "speaker")
            about = "the speaker" if subj_speaker else (f'{sf.get("name", "")} ({sf.get("type", "")})' if sf else "a new folder")
            slot = slot_of(e["slot"].split(".", 1)[-1], subj_speaker)
            rows.append({"state": json.dumps(text + f"\nIt is filed under: {about}."),
                         "questions": json.dumps({"slot": {"type": "choice", "instructions": Q_SLOT, "criteria": {s: "" for s in SLOTS}}}),
                         "gold": json.dumps({"slot": {"probabilities": {slot: 1.0}}}), "meta": {**meta, "q": "slot", "teacher_key": e["slot"].split(".", 1)[-1]}})
            # the teacher's state moves on exactly as filed (new folders from this batch become visible)
            for fid in [subject] + list(e.get("folders") or []):
                if fid in new_folders and fid not in state["folders"]:
                    nf = new_folders[fid]
                    state["folders"][fid] = {"name": str(nf.get("name", "")), "type": str(nf.get("type", "")),
                                             "aliases": [a for a in (nf.get("aliases") or []) if isinstance(a, str)],
                                             "link": str(nf.get("link") or "")}
        folders.merge(state, got, batch)
    if state["facts"] != final["facts"]:
        return None
    return rows


def main() -> None:
    enc = Encoder()
    held = set(json.loads((HERE.parent / "kaggle_clerk_ds" / "held_out.json").read_text(encoding="utf-8")))
    train, val, missing = [], [], []
    for name in SETS:
        for sc in folders.load(name):
            rows = rows_for_person(name, sc, enc)
            if rows is None:
                missing.append(f"{name}/{sc['scenario']}")
                continue
            (val if sc["scenario"] in held else train).extend(rows)
    random.Random(2029).shuffle(train)
    OUT.mkdir(exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        (OUT / f"{split}_filing_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "slots.json").write_text(json.dumps({"slots": SLOTS, "key_map": KEY_MAP, "types": TYPES,
                                                 "questions": {"folder": Q_FOLDER, "type": Q_TYPE, "slot": Q_SLOT},
                                                 "k_folders": K_FOLDERS}, indent=1), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "primnox-filing-rows", "id": "anikethpani/primnox-filing-rows", "licenses": [{"name": "other"}]}), encoding="utf-8")
    by_q = Counter(r["meta"]["q"] for r in train)
    gold_new = sum(1 for r in train if r["meta"]["q"] == "folder" and '"new"' in r["gold"])
    slots = Counter(json.loads(r["gold"])["slot"]["probabilities"].popitem()[0] for r in train if r["meta"]["q"] == "slot")
    print(f"train {len(train)} rows {dict(by_q)} (folder answer 'new' {gold_new}); val {len(val)}; people left out {len(missing)} {missing[:5]}")
    print("slot answers:", slots.most_common(12), "| other/details:", slots["other"], slots["details"])


if __name__ == "__main__":
    main()
