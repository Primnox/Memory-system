"""Who-is-who decisions as CHOICE rows for the ~500M memory model.   python build_identity_rows.py <set> [<set> ...]

For every voted person (identity.py vote), statements are replayed in order with the WRITER's
folders as the state (names, relations and aliases only as known at that statement: no
timebleed). A row is made only for a decision that 2 of 3 annotators agreed on; the gold answer
is the voted one:

  folder  Which folder is this statement mainly about?  the speaker, up to 7 existing folders
          (named in the text, then most similar), "new", "ask" (could be several: ask the user)
  type    (gold "new") what kind of thing the new folder is
  kind    fact / plan / hypothetical / question / quote / joke / rumour / correction / fiction
  about   speaker / person / group / thing
  merge   (statements with a voted merge, plus statements naming two folders or correcting one)
          is the folder it is filed under the same as one of the others? "none" or "same as X"
  split   (statements with a voted split, plus corrections) does that folder hold two entities?

Every row's state shows the two statements said just before (pronouns, ellipsis). Row meta keeps
the statement's case numbers and their sections, so scores can be split by seen / held-out section.
Also writes the people as change-detection scenarios (statements + voted `replaces`, pairs the vote
rejected as `skip`) for build_train.py.
Output: ../kaggle_identity_ds/<set>_identity_rows.jsonl, train_identity_<set>.json, identity_rows.json (config)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from graph_search import Encoder

HERE = Path(__file__).resolve().parent
ROOT = HERE / "identity"
OUT = HERE.parent / "kaggle_identity_ds"
MAP = json.loads((HERE / "identity_map.json").read_text(encoding="utf-8"))
K = 8
KINDS = {"fact": "asserted as true", "plan": "an intention or schedule, not happened yet",
         "hypothetical": "if / imagine / maybe one day; not asserted", "question": "the user asks something",
         "quote": "someone else's words or claim, not endorsed", "joke": "sarcasm, irony or exaggeration",
         "rumour": "hearsay flagged as unverified", "correction": "fixes something said earlier",
         "fiction": "a book, film, game, roleplay, dream or story"}
ABOUT = {"speaker": "the user", "person": "one other person", "group": "a group: team, family, class, 'we'",
         "thing": "not a person: pet, place, organisation, object, activity, event"}
TYPES = {"person": "a person", "pet": "a pet or animal", "place": "a place", "organisation": "a company, school or institution",
         "group": "a group of people: team, class, family", "thing": "an object: a car, phone, house…",
         "activity": "an activity, project or hobby", "event": "an event: a wedding, trip, incident"}
Q = {"folder": "Which folder in this person's memory is the new statement mainly about?",
     "type": "The statement is about someone or something new. What kind of thing is it?",
     "kind": "What kind of statement is this?",
     "about": "Who or what is the statement mainly about?",
     "merge": "Does this statement show that the folder it is filed under is the same one as another folder?",
     "split": "Does this statement show that this folder actually holds two different people or things?"}
PRONOUNS = {"he", "she", "they", "him", "her", "them", "it", "his", "hers", "their", "this", "that", "the other one"}


CARD_VEC: dict = {}


def card(f: dict) -> str:
    bits = [f.get("type") or ""]
    if f.get("relation"):
        bits.append(f["relation"])
    al = [a for a in f.get("aliases") or [] if a.lower() != str(f.get("name") or "").lower()][:3]
    if al:
        bits.append("also: " + ", ".join(al))
    return "; ".join(b for b in bits if b)


def label(f: dict) -> str:
    return re.sub(r"\s+", " ", str(f.get("name") or f.get("relation") or f.get("type") or "?")).strip()[:40]


def named_in(text: str, f: dict) -> bool:
    low = " " + re.sub(r"[^\w']+", " ", text.lower()) + " "
    names = [f.get("name") or ""] + list(f.get("aliases") or [])
    return any(isinstance(n, str) and len(n) >= 2 and f" {n.lower()} " in low for n in names)


def sections_of(cases: list) -> list[int]:
    return sorted({MAP["cases"][str(c)]["section"] for c in cases if isinstance(c, int) and str(c) in MAP["cases"]})


def row(state: str, qname: str, criteria: dict, gold: str, meta: dict) -> dict:
    return {"state": json.dumps(state), "questions": json.dumps({qname: {"type": "choice", "instructions": Q[qname],
                                                                          "criteria": criteria}}),
            "gold": json.dumps({qname: {"probabilities": {gold: 1.0}}}), "meta": {**meta, "q": qname}}


def person_rows(sc: dict, voted: dict, enc, setname: str) -> tuple[list[dict], dict]:
    speaker = sc["speaker"]
    folders: dict[str, dict] = {}                 # writer's folders, as known so far
    born: dict[str, str] = {}                     # folder id -> "@sK" (the statement that created it)
    redirect: dict[str, str] = {}                 # merged-away id -> kept id
    rows, sts = [], sc["statements"]
    held = set(MAP["held_out_sections"])

    def root(f):
        seen = set()
        while f in redirect and f not in seen:      # a later merge back the other way must not loop
            seen.add(f); f = redirect[f]
        return f

    def by_canon(c):
        for fid, b in born.items():
            if b == c and fid not in redirect:
                return fid
        return None

    for i, s in enumerate(sts):
        v = voted["votes"][s["id"]]
        prev = [x for x in sts[max(0, i - 2):i]]
        ctx = "".join(f'Said before ({x["date"]}): "{x["text"]}"\n' for x in prev)
        base = ctx + f'New statement ({s["date"]}): "{s["text"]}"\nThe speaker is {speaker}.'
        secs = sections_of(s.get("cases") or [])
        meta = {"set": setname, "scenario": sc["scenario"], "sid": s["id"], "cases": s.get("cases") or [],
                "sections": secs, "held_out": bool(set(secs) & held), "kind": "identity"}
        # ---- candidates: speaker, named folders, then the most similar cards
        live = [fid for fid in folders if fid not in redirect]
        named = [fid for fid in live if named_in(s["text"], folders[fid])]
        rest = [fid for fid in live if fid not in named]
        if rest:
            sv = enc([s["text"]])[0]
            texts = [f"{label(folders[f])} ({card(folders[f])})" for f in rest]
            todo = [t for t in dict.fromkeys(texts) if t not in CARD_VEC]     # each card text encoded once
            if todo:
                CARD_VEC.update(zip(todo, enc(todo)))
            rest = [rest[j] for j in sorted(range(len(rest)), key=lambda j: -float(CARD_VEC[texts[j]] @ sv))]
        cands = (named + rest)[:K - 1]
        fv = v["folder"]["value"] if v["folder"] else None
        must = []
        if fv and fv.startswith("@"):
            must = [by_canon(fv)]
        elif fv == "ask":
            must = [root(x) for x in s.get("ask_between") or [] if x != "new" and root(x) in folders]
        for m in must:
            if m and m not in cands:
                cands = cands[:K - 1 - len(must)] + [m] if len(cands) >= K - 1 else cands + [m]
        crit, key_of, taken = {speaker: "the user (speaker)"}, {}, {speaker}
        for fid in cands:
            kk = label(folders[fid])
            kk = kk if kk not in taken else f"{kk} ({born[fid][1:]})"
            taken.add(kk); key_of[fid] = kk; crit[kk] = card(folders[fid])
        crit["new"] = "someone or something not in memory yet"
        crit["ask"] = "could be more than one of these: ask the user"
        # ---- folder / type
        if fv:
            gold = (speaker if fv == "speaker" else "ask" if fv == "ask" else "new" if fv.startswith("new:")
                    else key_of.get(by_canon(fv)))
            if gold:
                rows.append(row(base, "folder", crit, gold, meta))
            if fv.startswith("new:") and fv[4:] in TYPES:
                rows.append(row(base, "type", TYPES, fv[4:], meta))
        if v["kind"] and v["kind"]["value"] in KINDS:
            rows.append(row(base, "kind", KINDS, v["kind"]["value"], meta))
        if v["about"] and v["about"]["value"] in ABOUT:
            rows.append(row(base, "about", ABOUT, v["about"]["value"], meta))
        # ---- merge / split, asked about the folder the writer filed it under
        filed = root(s["folder"]) if s["folder"] in folders else None
        if filed and v["merge"] is not None:
            others = [f for f in cands if f != filed]
            mv = v["merge"]["value"]
            want = None
            if mv:
                pair = [by_canon(c) for c in mv]
                if filed in pair:
                    want = next((p for p in pair if p != filed), None)
            asked = mv or s["kind"] == "correction" or len([f for f in named if f != filed]) >= 1 or \
                any(label(folders[f]).lower() == label(folders[filed]).lower() for f in others)
            if asked and others and (want is None or want in others):
                mcrit = {"none": "no: it shows nothing of the kind"}
                for f in others[:5]:
                    mcrit[f"same as {key_of[f]}"] = card(folders[f])
                if want is None or f"same as {key_of[want]}" in mcrit:
                    rows.append(row(base + f"\nIt is filed under: {label(folders[filed])} ({card(folders[filed])}).",
                                    "merge", mcrit, "none" if want is None else f"same as {key_of[want]}", meta))
        if filed and v["split"] is not None and (v["split"]["value"] or s["kind"] == "correction"):
            yes = v["split"]["value"] is not None and by_canon(v["split"]["value"]) == filed
            rows.append(row(base + f"\nIt is filed under: {label(folders[filed])} ({card(folders[filed])}).", "split",
                            {"no": "it is one person or thing", "yes": "it is really two different people or things"},
                            "yes" if yes else "no", meta))
        # ---- the writer's state moves on
        for nf in s.get("new_folders") or []:
            folders[nf["id"]] = {"name": nf.get("name"), "type": nf.get("type"), "relation": nf.get("relation") or "",
                                 "aliases": []}
            born[nf["id"]] = f"@{s['id']}"
        sp = s.get("split")
        if sp and sp.get("new_folder", {}).get("id"):
            nf = sp["new_folder"]
            folders[nf["id"]] = {"name": nf.get("name"), "type": nf.get("type"), "relation": nf.get("relation") or "",
                                 "aliases": []}
            born[nf["id"]] = f"@{s['id']}"
        for u in s.get("updates") or []:
            f = folders.get(root(u.get("folder", "")))
            if not f:
                continue
            if u.get("name"):
                if f.get("name"):
                    f["aliases"].append(f["name"])
                f["name"] = u["name"]
            if u.get("alias") and u["alias"].lower() not in PRONOUNS:
                f["aliases"].append(u["alias"])
            if u.get("relation"):
                f["relation"] = u["relation"]
        for m in s.get("mentions") or []:
            f = folders.get(root(m.get("folder", "")))
            t = str(m.get("text") or "").strip()
            if f and t and t.lower() not in PRONOUNS and len(t.split()) <= 4 and t not in f["aliases"] and t != f.get("name"):
                f["aliases"].append(t)
        mg = s.get("merge")
        if mg and len(mg) == 2 and mg[0] in folders and mg[1] in folders:
            keep, gone = folders[root(mg[0])], folders[mg[1]]
            keep["aliases"] += [a for a in [gone.get("name")] + gone["aliases"] if a and a not in keep["aliases"]]
            if root(mg[0]) != mg[1]:
                redirect[mg[1]] = root(mg[0])
                redirect.pop(root(mg[0]), None)
    # change-detection scenario: voted replaces; pairs any annotator gave but the vote rejected are skipped
    stmts = [{"id": s["id"], "date": s["date"], "text": s["text"], "replaces": voted["votes"][s["id"]]["replaces"]}
             for s in sts]
    skip = [[o, s["id"]] for s in sts for o in (s.get("replaces") or []) if o not in voted["votes"][s["id"]]["replaces"]]
    return rows, {"scenario": sc["scenario"], "statements": stmts, "skip": skip}


def main() -> None:
    enc = Encoder()
    OUT.mkdir(exist_ok=True)
    for setname in sys.argv[1:]:
        rows, scen = [], []
        for home in sorted((ROOT / setname).glob("p*")):
            if not (home / "voted.json").exists():
                continue
            sc = json.loads((home / "final.json").read_text(encoding="utf-8"))[0]
            voted = json.loads((home / "voted.json").read_text(encoding="utf-8"))
            r, s = person_rows(sc, voted, enc, setname)
            rows += r; scen.append(s)
        (OUT / f"{setname}_identity_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                            encoding="utf-8")
        prefix = "train" if setname in ("pilot", "train") else "eval"     # blind / open are never training data
        (OUT / f"{prefix}_identity_{setname}.json").write_text(json.dumps(scen, indent=1, ensure_ascii=False), encoding="utf-8")
        q = Counter(r["meta"]["q"] for r in rows)
        golds = Counter(json.loads(r["gold"])["folder"]["probabilities"].popitem()[0] in ("new", "ask") and
                        json.loads(r["gold"])["folder"]["probabilities"].popitem()[0] for r in rows if r["meta"]["q"] == "folder")
        print(f"{setname}: {len(scen)} people, {len(rows)} rows {dict(q)}; folder gold new/ask {dict(golds)}; "
              f"held-out-section rows {sum(r['meta']['held_out'] for r in rows)}")
    (OUT / "identity_rows.json").write_text(json.dumps({"questions": Q, "kinds": KINDS, "about": ABOUT, "types": TYPES,
                                                         "k": K}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
