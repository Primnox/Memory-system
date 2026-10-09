"""File every fact into folders (people, pets, places, things) and slots, the way the
organiser memory would, using Gemini through agy.   python folders.py run <set> [<set> ...]

Facts go in date order, BATCH at a time; each call sees the folders and slots filed so far
and the next batch only, never later facts (the app would file as facts arrive). Nothing
about replacements or questions is shown. Sets:
  chat      24 chat people (training data; dev for the search test)
  v3long    24 v3 long-tier people (head training data; never seen by this method)
  writers / multilang / v3short / v3dev   the other head-training people (teacher data for
            training a local clerk and an index-aware head)
  blind     blind v2 (proof)
  realtalk  REALTALK with Gemini facts (proof; no licence, never committed)
Output: folders/<set>/<scenario>/state.json  {"folders": {...}, "slots": {...}, "facts": {...}}
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round7" / "pipeline"))   # drive.py, long.py
import drive
from drive import log, parse
from long import strayed

HERE = Path(__file__).resolve().parent
# FOLDERS_OUT points the scorers at another clerk's filing (e.g. folders_qwen/)
OUT = Path(os.environ.get("FOLDERS_OUT") or HERE / "folders")
SPEC = (HERE / "SPEC_FOLDERS.md").read_text(encoding="utf-8")
MODEL = os.environ.get("FOLDERS_MODEL") or "gemini-3.8-flash-medium"   # claude-sonnet-4-6 took over when Gemini credits ran out
BATCH = int(os.environ.get("FOLDERS_BATCH") or 25)   # 1 = one fact per call, as the app files them
WORKERS = int(os.environ.get("FOLDERS_WORKERS") or 16)
BLIND = Path(__file__).resolve().parents[2] / "scripts" / "blind_memory" / "v2" / "test.json"


def load(name: str) -> list[dict]:
    if name == "chat":
        return [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "chat").glob("p*/voted.json"))]
    if name == "writers":       # training people: writers A-D and the DeepSeek pilot
        return [sc for f in sorted((HERE.parent / "ms" / "system_one" / "data").glob("train_writer_*.json"))
                for sc in json.loads(f.read_text(encoding="utf-8"))]
    if name == "multilang":
        return [json.loads(f.read_text(encoding="utf-8"))[0] for f in sorted((HERE / "cases" / "multilang").glob("p*/voted.json"))]
    if name in ("v3short", "v3dev"):
        return json.loads((HERE.parent / "kaggle_ds" / f"train_v3_{name[2:]}.json").read_text(encoding="utf-8"))
    if name == "v3long":
        return json.loads((HERE.parent / "kaggle_ds" / "train_v3_long.json").read_text(encoding="utf-8"))
    if name == "blind":
        return json.loads(BLIND.read_text(encoding="utf-8"))
    if name == "realtalk":
        return json.loads((HERE.parent / "realtalk" / "e2e_llm_common.json").read_text(encoding="utf-8"))
    raise SystemExit(f"unknown set {name}")


def first_state(name: str, sc: dict) -> dict:
    if name == "realtalk":       # third-person facts naming both speakers: the clerk makes their folders
        return {"folders": {}, "slots": {}, "facts": {}}
    who = sc["scenario"].split("-")[0].capitalize()
    return {"folders": {"F1": {"name": who, "type": "speaker", "aliases": ["I", "me", "my", "the user"],
                               "link": ""}},
            "slots": {}, "facts": {}}


def check(got: dict, batch: list[dict], state: dict) -> list[str]:
    errs = []
    folders = set(state["folders"]) | {f.get("id") for f in got.get("new_folders", []) if isinstance(f, dict)}
    slots = set(state["slots"]) | {s.get("key") for s in got.get("new_slots", []) if isinstance(s, dict)}
    facts = got.get("facts")
    if not isinstance(facts, dict):
        return ["no facts object"]
    for f in batch:
        e = facts.get(f["id"])
        if not isinstance(e, dict):
            errs.append(f"{f['id']} not filed")
            continue
        fs, slot = e.get("folders"), e.get("slot")
        if not isinstance(fs, list) or not fs or any(x not in folders for x in fs):
            errs.append(f"{f['id']} folders {fs}")
        # a slot used without being declared is accepted when its folder exists; merge()
        # records it as many-valued, which never ranks another fact down
        if not isinstance(slot, str) or slot.split(".")[0] not in folders:
            errs.append(f"{f['id']} slot {slot}")
    return errs


def merge(state: dict, got: dict, batch: list[dict]) -> None:
    for f in got.get("new_folders", []):
        state["folders"][f["id"]] = {k: f.get(k, "" if k != "aliases" else []) for k in ("name", "type", "aliases", "link")}
    for u in got.get("folder_updates", []):
        cur = state["folders"].get(u.get("id"))
        if not cur:
            continue
        if u.get("name"):
            cur["aliases"] = sorted(set(cur["aliases"]) | {cur["name"]})
            cur["name"] = u["name"]
        cur["aliases"] = sorted(set(cur["aliases"]) | set(u.get("add_aliases") or []))
        if u.get("link"):
            cur["link"] = u["link"]
    for s in got.get("new_slots", []):
        state["slots"].setdefault(s["key"], s.get("holds", "many"))
    for f in batch:
        state["facts"][f["id"]] = got["facts"][f["id"]]
        state["slots"].setdefault(got["facts"][f["id"]]["slot"], "many")


def prompt(state: dict, batch: list[dict]) -> str:
    folders = "\n".join(json.dumps({"id": k, **v}, ensure_ascii=False) for k, v in state["folders"].items()) or "(none yet)"
    slots = "\n".join(f"{k} ({v})" for k, v in state["slots"].items()) or "(none yet)"
    facts = "\n".join(json.dumps({"id": f["id"], "date": f["date"], "text": f["text"]}, ensure_ascii=False) for f in batch)
    return f"{SPEC}\n## Folders so far\n{folders}\n\n## Slots so far\n{slots}\n\n## New facts\n{facts}\n"


def person(name: str, sc: dict) -> str:
    home = OUT / name / re.sub(r"[^\w\-]", "_", sc["scenario"])
    done = home / "state.json"
    state = json.loads(done.read_text(encoding="utf-8")) if done.exists() else first_state(name, sc)
    facts = sorted(sc["statements"], key=lambda f: f["date"])          # stable: same-day order kept
    for k in range(0, len(facts), BATCH):
        batch = [f for f in facts[k:k + BATCH] if f["id"] not in state["facts"]]
        if not batch:
            continue
        cwd = home / f"b{k // BATCH:03d}"
        for attempt in range(3):
            try:
                text, _ = drive.agy(prompt(state, batch), MODEL, cwd)
                got = parse(text)
                errs = check(got, batch, state)
                if errs:
                    raise ValueError(f"{len(errs)} problems: {errs[:4]}")
                merge(state, got, batch)
                break
            except Exception as e:  # noqa: BLE001
                log(f"folders {name}/{home.name} b{k // BATCH:03d}: attempt {attempt} failed: {str(e)[:200]}")
        else:
            log(f"folders {name}/{home.name}: gave up at b{k // BATCH:03d}")
            return f"{home.name}: FAILED"
        bad = strayed(cwd)
        if bad:
            log(f"folders {name}/{home.name} b{k // BATCH:03d}: STRAYED {bad[:3]}")
            state.setdefault("strayed", []).extend(bad)
        done.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"{home.name}: {len(state['folders'])} folders, {len(state['slots'])} slots, {len(state['facts'])} facts"


def run(names: list[str]) -> None:
    jobs = [(n, sc) for n in names for sc in load(n)]
    with ThreadPoolExecutor(WORKERS) as ex:
        for msg in ex.map(lambda j: person(*j), jobs):
            log(f"folders: {msg}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        raise SystemExit(__doc__)
    run(sys.argv[2:])
