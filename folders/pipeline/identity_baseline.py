"""Code-only baselines on who-is-who rows (no model).   python identity_baseline.py <rows.jsonl> [...]

The rules a memory without a learned model would use, fixed before scoring:
  folder  options whose name, alias or relation ("my cousin") appears in the statement: exactly one ->
          that folder; two or more -> ask; none -> "new" if the statement names someone (a capitalised
          word not at the start, or "my <relation>"), else the speaker
  kind    keyword rules (?, if/imagine, apparently/I heard, said/told, actually/I meant, will/going to,
          lol/basically, game/book/film/dream), else fact
  about   speaker if the folder rule picked the speaker; group for we/team/family/the boys; else by the
          picked option's type (person -> person, anything else -> thing); new -> person
  type    always person (the most common new folder)
  merge   none (code never merges by name alone);  also scored: "same name -> same" (the naive merge)
  split   no
Scored per question, split by statements from seen vs held-out sections.
"""
import json
import re
import sys
from collections import Counter

KIN = r"(?:wife|husband|partner|boyfriend|girlfriend|son|daughter|kids?|mum|mom|mother|dad|father|brother|sister|" \
      r"grandma|grandpa|aunt|uncle|cousin|niece|nephew|boss|manager|landlord|flatmate|roommate|friend|colleague|" \
      r"neighbou?r|doctor|teacher|dog|cat)"


def words(t: str) -> str:
    return " " + re.sub(r"[^\w']+", " ", t.lower()) + " "


def parse(row: dict) -> tuple[str, str, dict]:
    st = json.loads(row["state"])
    new = re.search(r'New statement \([^)]*\): "(.*)"\n', st)
    speaker = re.search(r"The speaker is (.+?)\.", st)
    return (new.group(1) if new else st), (speaker.group(1) if speaker else ""), json.loads(row["questions"])


def pick_folder(text: str, speaker: str, crit: dict) -> str:
    w = words(text)
    hits = []
    for k, card in crit.items():
        if k in ("new", "ask", speaker):
            continue
        names = [re.sub(r" \(s\d+\)$", "", k)]
        m = re.search(r"also: (.*)$", card)
        if m:
            names += [a.strip() for a in m.group(1).split(",")]
        rel = [p.strip() for p in card.split(";")[1:2] if p.strip() and not p.strip().startswith("also")]
        if any(len(n) >= 2 and words(n).strip() and words(n) in w for n in names + rel):
            hits.append(k)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "ask"
    caps = [m.group() for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+", text) if m.group() not in ("I", speaker)]
    if caps or re.search(r"\bmy (?:new |old |best |little |big )?" + KIN + r"\b", text, re.I):
        return "new"
    return speaker


def pick_kind(text: str) -> str:
    t = text.lower()
    rules = [("question", r"\?\s*$|^(?:do|does|did|is|are|was|can|should|what|who|where|when|why|how)\b"),
             ("hypothetical", r"\b(?:if i|if we|imagine|what if|maybe one day|i wonder if|suppose)\b"),
             ("rumour", r"\b(?:apparently|i heard|rumou?r|people say|supposedly|word is)\b"),
             ("fiction", r"\b(?:in the (?:game|book|novel|film|movie|show|series|play)|my character|dreamt|dreamed|episode)\b"),
             ("correction", r"\b(?:actually|i meant|correction|not my|my bad|i was wrong|i said .* (?:it's|it is))\b"),
             ("quote", r"\b(?:says|said that|told me that|told .* that|according to|claims)\b"),
             ("joke", r"\b(?:lol|lmao|haha|basically my|jk|literally the)\b"),
             ("plan", r"\b(?:going to|gonna|will|planning|plan to|next week|tomorrow|booked|thinking of)\b")]
    for k, rx in rules:
        if re.search(rx, t):
            return k
    return "fact"


def main() -> None:
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        c = Counter()
        folder_pick = {}
        for r in rows:
            text, speaker, qs = parse(r)
            q = r["meta"]["q"]
            crit = qs[q]["criteria"]
            gold = max(json.loads(r["gold"])[q]["probabilities"].items(), key=lambda kv: kv[1])[0]
            key = (r["meta"]["scenario"], r["meta"]["sid"])
            if q == "folder":
                got = pick_folder(text, speaker, crit)
                folder_pick[key] = (got, crit.get(got, ""))
                if gold == "new" and got not in ("new", "ask"):
                    c["false link (gold new)"] += got != speaker
                if gold not in ("new", "ask", speaker) and got not in ("new", "ask", gold):
                    c["wrong existing / speaker"] += 1
                c["ask predicted"] += got == "ask"; c["ask gold"] += gold == "ask"; c["ask both"] += got == gold == "ask"
            elif q == "kind":
                got = pick_kind(text)
            elif q == "about":
                fp, card = folder_pick.get(key, ("", ""))
                low = text.lower()
                got = ("speaker" if fp == speaker else "group" if re.search(r"\b(?:we|our team|the team|family|the boys|the girls|the kids)\b", low)
                       else "person" if fp == "new" or card.startswith("person") else "thing")
            elif q == "type":
                got = "person"
            elif q == "merge":
                got = "none"
                filed = re.search(r"It is filed under: (.+?) \(", json.loads(r["state"]))
                naive = next((k for k in crit if filed and k.startswith("same as ") and
                              re.sub(r" \(s\d+\)$", "", k[8:]).lower() == filed.group(1).lower()), "none")
                c[("merge-by-name", "n")] += 1; c[("merge-by-name", "ok")] += naive == gold
            elif q == "split":
                got = "no"
            split = "held_out" if r["meta"].get("held_out") else "seen"
            for k in (q, f"{q}|{split}"):
                c[(k, "n")] += 1; c[(k, "ok")] += got == gold
        print(f"code-only baseline on {path.split('/')[-1].split(chr(92))[-1]} ({len(rows)} rows)")
        for k in sorted({k for k, _ in [x for x in c if isinstance(x, tuple)]}):
            print(f"  {k:18s} {c[(k, 'ok')]}/{c[(k, 'n')]} ({100 * c[(k, 'ok')] / c[(k, 'n')]:.1f}%)")
        for k in ("false link (gold new)", "wrong existing / speaker", "ask predicted", "ask gold", "ask both"):
            print(f"  {k}: {c[k]}")
        gold_dist = Counter(max(json.loads(r["gold"])[r["meta"]["q"]]["probabilities"].items(), key=lambda kv: kv[1])[0]
                            for r in rows if r["meta"]["q"] == "kind")
        print("  kind gold distribution:", dict(gold_dist.most_common()))


if __name__ == "__main__":
    main()
