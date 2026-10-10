"""Kaggle GPU job, round 9: ONE ~500M memory model that both decides changes and files the index.

Training: round 7's recipe unchanged (its people, noisy copies, settings: build_train.py +
train_laya.py, 3 epochs) plus the filing decisions (primnox-filing-rows: which folder / what
kind of new thing / which of 34 slots, from the teacher clerks' filing of the training people).
The soft "not a replacement" cap still counts change-detection rows only, so the change task
keeps round 7's balance.
Tests: blind v2 change detection (round 7: 104/112, 0 wrong), LongMemEval knowledge updates with
Gemini facts (0.78), Dialogue NLI (0.61 / 0.69); filing accuracy on 8 held-out people; and the
model filing blind v2 by itself, one fact at a time (new folders' names by code, as the app
would), saved for scoring search locally.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

OUT, SRC = "/kaggle/working", "/tmp/ms"
LME = "data/blind_external/pairs_lme_ku.jsonl"
LLM_RUNNER = '''
import json, sys
import external_eval
facts = json.load(open(sys.argv[1], encoding="utf-8"))
external_eval.statements = lambda text, keep_all_if_none=True: facts.get(text) or [text]
sys.argv = ["external_eval.py"] + sys.argv[2:]
external_eval.main()
'''

FILER = r'''
import glob, json, os, re, sys, time
import torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "/tmp/ms/system_one")
from build_train import names_in
from laya.agent import Agent

OUT = "/kaggle/working"
model_dir, tag = sys.argv[1], sys.argv[2]
agent = Agent(model_dir)
src = os.path.dirname(glob.glob("/kaggle/input/**/slots.json", recursive=True)[0])
cfg = json.load(open(f"{src}/slots.json", encoding="utf-8"))
SLOTS, TYPES, Q, K = cfg["slots"], cfg["types"], cfg["questions"], cfg["k_folders"]
log = open(f"{OUT}/filing_{tag}.log", "a", buffering=1)
def say(*a):
    print(*a, flush=True); print(*a, file=log)

def ask(state_text, name, question):
    r = agent.predict_batch([state_text], {name: question})[0]["answers"][name]
    probs = r["probabilities"]
    return max(probs, key=probs.get)

# 1) held-out people: does it make the teacher's choice?
val = [json.loads(l) for l in open(f"{src}/val_filing_rows.jsonl", encoding="utf-8")]
right, total = {}, {}
for i in range(0, len(val), 32):
    chunk = val[i:i + 32]
    for row in chunk:
        qs = json.loads(row["questions"]); name = next(iter(qs))
        gold = json.loads(row["gold"])[name]["probabilities"]
        want = max(gold, key=gold.get)
        got = ask(json.loads(row["state"]), name, qs[name])
        right[name] = right.get(name, 0) + (got == want); total[name] = total.get(name, 0) + 1
say("filing on held-out people: " + ", ".join(f"{k} {right[k]}/{total[k]} ({100 * right[k] / total[k]:.0f}%)" for k in total))

# 2) blind v2 filed by the model itself, one fact at a time
tok = AutoTokenizer.from_pretrained("thenlper/gte-small")
enc = AutoModel.from_pretrained("thenlper/gte-small").eval()
def embed(texts):
    with torch.no_grad():
        e = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = enc(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

KIN = re.compile(r"\bmy (?:little |big |older |younger |new |best |ex-?)?(wife|husband|partner|boyfriend|girlfriend|fianc[eé]e?|son|daughter|"
                 r"kids?|mum|mom|mother|dad|father|brother|sister|grandma|grandmother|grandpa|grandfather|aunt|uncle|"
                 r"cousin|niece|nephew|dog|cat|puppy|kitten|boss|manager|landlord|flatmate|roommate|friend|colleague|"
                 r"doctor|therapist|car|bike|flat|apartment|house|phone|laptop)\b(?:,? (?:called |named )?([A-Z][a-zA-Z'\-]+))?")

def card(f):
    al = [a for a in f.get("aliases") or [] if a.lower() not in ("i", "me", "my", "the user")][:2]
    kind = "the user" if f.get("type") == "speaker" else f.get("type", "")
    return kind + (f"; {', '.join(al)}" if al else "")

def named_in(text, f):
    low = " " + re.sub(r"[^\w']+", " ", text.lower()) + " "
    return any(isinstance(n, str) and len(n) >= 3 and f" {n.lower()} " in low for n in [f.get("name", "")] + list(f.get("aliases") or []))

def candidates(text, state):
    speaker = [k for k, v in state["folders"].items() if v.get("type") == "speaker"]
    named = [k for k, v in state["folders"].items() if k not in speaker and named_in(text, v)]
    rest = [k for k in state["folders"] if k not in speaker and k not in named]
    if rest:
        fv = embed([text])[0]
        cv = embed([f"{state['folders'][k].get('name', '')} ({card(state['folders'][k])})" for k in rest])
        rest = [rest[i] for i in sorted(range(len(rest)), key=lambda i: -float(cv[i] @ fv))]
    return (speaker + named + rest)[:K]

def new_name(text, earlier, state, speaker):
    taken = {n.lower() for f in state["folders"].values() for n in [f.get("name", "")] + list(f.get("aliases") or []) if isinstance(n, str)}
    m = KIN.search(text)
    if m and m.group(2) and m.group(2).lower() not in taken:
        return m.group(2), [f"my {m.group(1)}"]
    names = names_in(earlier)
    for w in re.findall(r"[A-Z][a-zA-Z'\-]+", text):
        if w in names and w.lower() not in taken and w not in ("I", speaker):
            return w, ([f"my {m.group(1)}"] if m else [])
    if m:
        return f"{speaker}'s {m.group(1)}", [f"my {m.group(1)}"]
    return None, []

blind = json.load(open(glob.glob("/tmp/ms/scripts/blind_memory/v2/test.json")[0], encoding="utf-8"))
t0, made = time.time(), 0
for sc in blind:
    who = sc["scenario"].split("-")[0].capitalize()
    state = {"folders": {"F1": {"name": who, "type": "speaker", "aliases": ["I", "me", "my", "the user"], "link": ""}},
             "slots": {}, "facts": {}, "filed_by": tag}
    facts = sorted(sc["statements"], key=lambda f: (f["date"], f["id"]))
    for i, f in enumerate(facts):
        text = f'New fact ({f["date"]}): "{f["text"]}"\nThe speaker is {who}.'
        cands = candidates(f["text"], state)
        crit, key_of, taken = {}, {}, set()
        for fid in cands:
            kk = re.sub(r"\s+", " ", state["folders"][fid].get("name") or fid).strip()[:40] or fid
            kk = kk if kk not in taken else f"{kk} ({fid})"
            taken.add(kk); key_of[kk] = fid; crit[kk] = card(state["folders"][fid])
        crit["new"] = "someone or something not in memory yet"
        pick = ask(text, "folder", {"type": "choice", "instructions": Q["folder"], "criteria": crit})
        subject = key_of.get(pick)
        if subject is None:
            name, aliases = new_name(f["text"], facts[:i + 1], state, who)
            if name is None:                       # nothing to name it by: it stays with the speaker
                subject = "F1"
            else:
                kind = ask(text, "type", {"type": "choice", "instructions": Q["type"], "criteria": TYPES})
                subject = f"F{max(int(k[1:]) for k in state['folders']) + 1}"
                state["folders"][subject] = {"name": name, "type": kind, "aliases": aliases, "link": ""}
                made += 1
        about = "the speaker" if state["folders"][subject].get("type") == "speaker" else f'{state["folders"][subject]["name"]} ({state["folders"][subject]["type"]})'
        slot = ask(text + f"\nIt is filed under: {about}.", "slot",
                   {"type": "choice", "instructions": Q["slot"], "criteria": {s: "" for s in SLOTS}})
        key = f"{subject}.{slot.replace(' ', '_')}"
        state["slots"].setdefault(key, SLOTS.get(slot, "many"))
        linked = [k for k, v in state["folders"].items() if k != subject and v.get("type") != "speaker" and named_in(f["text"], v)]
        state["facts"][f["id"]] = {"folders": [subject] + linked, "slot": key}
    name = re.sub(r"[^\w\-]", "_", sc["scenario"])
    os.makedirs(f"{OUT}/filed_{tag}/blind/{name}", exist_ok=True)
    json.dump(state, open(f"{OUT}/filed_{tag}/blind/{name}/state.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say(f"blind v2 filed by {tag} one fact at a time in {time.time() - t0:.0f}s; {made} folders made")
'''


def sh(cmd: str, check: bool = True) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(["bash", "-c", f"set -o pipefail; ({cmd}) 2>&1 | tee -a {OUT}/run.log"], check=check)


def one(pattern: str) -> str:
    found = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert found, f"{pattern} not in the attached inputs"
    return found[0]


def patch(path: str, old: str, new: str) -> None:
    s = open(path, encoding="utf-8").read()
    assert old in s, f"{path}: {old[:60]}"
    open(path, "w", encoding="utf-8").write(s.replace(old, new))


def main() -> None:
    t0 = time.time()
    os.environ["PYTHONUNBUFFERED"] = "1"
    sh("pip install -q laya scikit-learn")
    sh(f"rm -rf {SRC} && git clone -q --depth 1 https://github.com/Primnox/Memory-system {SRC}")
    os.chdir(f"{SRC}/system_one")
    sys.path.insert(0, os.getcwd())
    # the soft-row cap follows the change-detection rows only, as in round 7
    patch("train_laya.py", 'ours = sum(1 for r in hard if r["meta"].get("source") != "dnli")',
          'ours = sum(1 for r in hard if r["meta"].get("source") != "dnli" and r["meta"]["kind"] != "filing")')
    for f in glob.glob("data/train_*.json"):
        os.remove(f)
    for f in sorted(glob.glob("/kaggle/input/**/train_r7_*.json", recursive=True)):
        shutil.copy(f, "data/")
    sh("python build_train.py")
    filing = one("train_filing_rows.jsonl")
    r9 = "models/laya-memory-r9"
    t1 = time.time()
    sh(f"python train_laya.py --device cuda --rows data/train_rows.jsonl {filing} --soft-ratio 2 --epochs 3 "
       f"--micro-batch 8 --grad-accum 4 --output-dir {r9}")
    print(f"r9 trained in {time.time() - t1:.0f}s", flush=True)
    for leftover in ("train_items.pt", "checkpoint_latest"):
        p = os.path.join(r9, leftover)
        shutil.rmtree(p) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    shutil.copytree(r9, f"{OUT}/{r9}")

    sh(f"python blind_pairs.py --model {r9} --tag r9-clean --save-pairs", check=False)
    open("run_llm_pieces.py", "w").write(LLM_RUNNER)
    sh(f"python run_llm_pieces.py {one('facts_lme_ku.json')} --model {r9} --tag r9-lme-llm --sentences --files {LME}",
       check=False)
    open("/tmp/filer.py", "w").write(FILER)
    sh(f"python /tmp/filer.py {r9} r9", check=False)
    from huggingface_hub import hf_hub_download
    from build_dnli import tidy
    for split in ("test", "verified_test"):
        path = hf_hub_download("xksteven/dialogue_nli", f"dialogue_nli/dialogue_nli/dialogue_nli_{split}.jsonl",
                               repo_type="dataset")
        text = open(path, encoding="utf-8").read()
        try:
            rows = json.loads(text)
        except ValueError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        with open(f"data/blind_external/pairs_dnli_{split}_full.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(rows):
                old, new = tidy(ex["sentence2"]), tidy(ex["sentence1"])
                if len(old) < 8 or len(new) < 8:
                    continue
                f.write(json.dumps({"id": ex.get("id", f"{split}-{i}"), "source": f"dnli-{split}:{ex['label']}",
                                    "old_text": old, "new_text": new, "old_date": "2025-03-01",
                                    "new_date": "2025-06-01", "label": int(ex["label"] == "negative")},
                                   ensure_ascii=False) + "\n")
    sh(f"python external_eval.py --model {r9} --tag r9-dnli --batch 64 --files "
       f"data/blind_external/pairs_dnli_test_full.jsonl data/blind_external/pairs_dnli_verified_test_full.jsonl",
       check=False)
    for f in glob.glob("results/blind_*r9*") + glob.glob("results/external_r9*"):
        if os.path.isfile(f):
            shutil.copy(f, OUT)
    sh(f"cd {OUT} && tar czf filed_r9.tgz filed_r9 || true")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
